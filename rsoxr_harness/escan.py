"""
escan.py
--------
Fixed-angle energy scans (R vs E at constant θ, reduced with rsoxr_reduce `escan`)
fitted with Kramers–Kronig-linked optical constants, alone or co-refined with the
project's angle scans.

A spec (EscanSpec, project.json "escan": {name: spec}) names:
  model      the harness model giving layer order, materials, geometry bounds and dq;
             its best per-energy fits (seed_model, default = model) give start values
  scans      reduced energy-scan files (θ read from the file header)
  kk         layers whose optical constants become KK-linked paths:
             [{"layer", "formula", "density"}]  (formula/density only set the tails
             outside the KK window and the pinned end values)
  window     KK merge window [lo, hi] eV (default: data range − 20 / + 30 eV); the
             merge points are grid nodes, so kkcalc's splice rescale is the identity

Optical constants
  KK layer   isld = piecewise-linear curve through free node values (a node at every
             energy-scan point, OUTER_STEP eV nodes outside the data, end nodes pinned
             to the tabulated isld); sld = KK(isld) + one constant offset.  The KK map is
             linear for fixed merge points → response matrices built once (kkcalc via
             stoich.kk_sld, ≤ 50 % CPUs) and cached in escan/<name>/.
  isld basis (spec.isld_basis) — how the free isld on the grid is parameterised; the KK
             map, grid and pinned end nodes are the same for all three:
             "nodes"   free value at every interior node (above).
             "bspline" clamped cubic B-spline, knots every knot_step eV inside the energy
                       scans (finer over knot_dense [[lo, hi, step], ...]), every
                       knot_step_outside (default outer_step) eV outside; end coefficients
                       pinned to the tabulated isld.  Curvature prior = h0³/σ² ∫ (isld″)² dE
                       (Gram matrix of the basis second derivatives), same σ / σ_outside /
                       curv_regions as the node prior.
             "nexafs"  isld = a stored NEXAFS peak model (baseline + slope + edge steps +
                       Gaussians, nexafs.evaluate maths) on the interior nodes; its varied
                       parameters (with their bounds) are the fit parameters.  spec.nexafs
                       [{"layer", "fit" | "model", "shift"}]: start from a `nexafs fit`
                       result (its energy_shift is honoured) or a stored model; "shift" (eV)
                       is added on top (model evaluated at E + shift).  No curvature prior.
  other layer per the model recipe: an sld window (sld_offset / absolute bounds) → free
             at each angle-scan energy, interpolated across the energy scans inside the
             angle-scan range, material table outside; no window → material table;
             material None → constant.
Priors (per eV, σ quoted per 0.5 eV step so script values carry over)
  curvature  Σ (isld″·h0²·√(h̄/h0) / σ_curv)² over the grid, every KK layer; outside the
             energy scans σ_curv_outside (default σ_curv/10), so the unconstrained nodes
             cannot be used as a KK knob on the in-window sld; curv_regions
             [{"lo", "hi", "sigma"}] override σ_curv over an energy range (outside the
             scans the stiffer of the two applies) — e.g. sharp edge at σ 0.5 but a
             smooth post-edge (Brewer2 carbon two-layer fit oscillated above 300 eV)
  flat       Σ (isld′·h0·√(h/h0) / σ)² below E for the listed layers
Data / weights
  energy scans: ln R residual / sigma_data (per-scan weight, scale 1)
  angle scans (co-refinement): ln R residual / sigma_data; θ < min_theta left out;
             above high_theta σ = observed scatter of the seed model's own fits there
             (high_sigma="auto") or a number; one shared scale, bkg per energy
  angle=False → energy scans only: geometry, scale and non-KK layers fixed at the
             seed model's values.
Geometry: thickness/roughness free where the recipe varies them (recipe bounds);
  thickness_group {"layers": [...], "bounds": [lo, hi]} fits the summed thickness of
  those layers instead (last layer gets the remainder).

Outputs (escan/<name>/): result.npz, summary.json, paths_<layer>.csv, free_<layer>.csv;
the previous result is kept as result_prev.npz.
"""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field

import numpy as np

HC = 12398.42
JAC_CHUNK = 256              # forward-mode Jacobian columns per pass (GPU memory ∝ chunk)
H0 = 0.5                     # eV; σ of the priors is quoted per H0 step
NODE_TOL = 0.06              # eV; energy-scan points closer than this share a node
ISLD_BASES = ("nodes", "bspline", "nexafs")
GRAM_STEP = 0.02             # eV; quadrature step for the B-spline curvature Gram matrix


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

@dataclass
class EscanSpec:
    name: str
    model: str
    scans: list
    kk: list
    seed_model: str | None = None
    criteria: str = "best"
    window: list | None = None
    outer_step: float = 2.0
    sigma_curv: float = 0.05
    sigma_curv_outside: float | None = None             # default sigma_curv / 10
    curv_regions: list = field(default_factory=list)  # [{"lo", "hi", "sigma"}]
    flat: list = field(default_factory=list)          # [{"layer", "below", "sigma"}]
    anchor: list = field(default_factory=list)        # [{"layer", "lo", "hi", "formula", "density", "sigma"}]
    angle: bool = True
    min_theta: float = 0.0
    high_theta: float | None = None
    high_sigma: str | float = "auto"
    sigma_data: float = 0.03
    scan_weights: list | None = None
    scan_scales: list | None = None                     # per scan: True → own fitted scale (default 1, fixed)
    escan_scale_bounds: tuple = (0.5, 1.5)
    angle_scale_bounds: tuple = (0.5, 1.5)
    thickness_group: dict | None = None
    max_nfev: int = 2000
    description: str = ""
    isld_basis: str = "nodes"                           # nodes | bspline | nexafs
    knot_step: float = 0.5                              # bspline: knot spacing inside the scans (eV)
    knot_step_outside: float | None = None              # bspline: outside the scans (default outer_step)
    knot_dense: list = field(default_factory=list)    # bspline: [[lo, hi, step], ...] finer knots
    nexafs: list = field(default_factory=list)        # nexafs: [{"layer", "fit" | "model", "shift"}]

    def __post_init__(self):
        if not self.scans:
            raise ValueError("escan spec needs at least one energy-scan file")
        self.kk = [dict(k) for k in self.kk]
        for k in self.kk:
            if not {"layer", "formula", "density"} <= set(k):
                raise ValueError(f"kk entry {k}: need layer, formula, density")
            k["density"] = float(k["density"])
        if not self.kk:
            raise ValueError("escan spec needs at least one KK layer")
        self.flat = [dict(f) for f in self.flat]
        self.curv_regions = [dict(c) for c in self.curv_regions]
        for c in self.curv_regions:
            if not {"lo", "hi", "sigma"} <= set(c):
                raise ValueError(f"curv_regions entry {c}: need lo, hi, sigma")
        self.anchor = [dict(a) for a in self.anchor]
        for a in self.anchor:
            if not {"layer", "lo", "hi", "formula", "density", "sigma"} <= set(a):
                raise ValueError(f"anchor entry {a}: need layer, lo, hi, formula, density, sigma")
        self.angle_scale_bounds = tuple(float(x) for x in self.angle_scale_bounds)
        if self.scan_weights is not None and len(self.scan_weights) != len(self.scans):
            raise ValueError("scan_weights must match scans")
        if self.scan_scales is not None:
            if len(self.scan_scales) != len(self.scans):
                raise ValueError("scan_scales must match scans")
            self.scan_scales = [bool(x) for x in self.scan_scales]
        self.escan_scale_bounds = tuple(float(x) for x in self.escan_scale_bounds)
        if self.window is not None:
            self.window = [float(self.window[0]), float(self.window[1])]
        if self.isld_basis not in ISLD_BASES:
            raise ValueError(f"isld_basis '{self.isld_basis}': one of {ISLD_BASES}")
        if self.knot_step <= 0 or (self.knot_step_outside is not None and self.knot_step_outside <= 0):
            raise ValueError("knot steps must be > 0")
        self.knot_dense = [[float(v) for v in d] for d in self.knot_dense]
        for d in self.knot_dense:
            if len(d) != 3 or d[0] >= d[1] or d[2] <= 0:
                raise ValueError(f"knot_dense entry {d}: need [lo, hi, step] with lo < hi, step > 0")
        self.nexafs = [dict(n) for n in self.nexafs]
        for n in self.nexafs:
            if "layer" not in n or not (("fit" in n) ^ ("model" in n)):
                raise ValueError(f"nexafs entry {n}: need layer and one of fit / model")
            n["shift"] = float(n.get("shift", 0.0))
        if self.isld_basis == "nexafs":
            have = {n["layer"] for n in self.nexafs}
            miss = [k["layer"] for k in self.kk if k["layer"] not in have]
            if miss:
                raise ValueError(f"isld_basis 'nexafs' needs a nexafs entry for KK layer(s) {miss}")

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**d)

    def describe(self):
        kk = ", ".join(f"{k['layer']} ({k['formula']} {k['density']:g})" for k in self.kk)
        lines = [f"escan '{self.name}': model {self.model}"
                 + (f" (seeded from {self.seed_model})" if self.seed_model else ""),
                 f"  scans: " + "; ".join(os.path.basename(s) for s in self.scans),
                 f"  KK layers: {kk}; window {self.window or 'auto'}; outer step {self.outer_step} eV",
                 f"  isld basis: {self.isld_basis}"
                 + (f" (knots {self.knot_step:g} eV in the scans, "
                    f"{self.knot_step_outside or self.outer_step:g} eV outside"
                    + "".join(f"; {d[2]:g} eV over {d[0]:g}–{d[1]:g}" for d in self.knot_dense) + ")"
                    if self.isld_basis == "bspline" else "")
                 + ("".join(f"; {n['layer']} ← {'fit' if 'fit' in n else 'model'} "
                            f"{n.get('fit') or n.get('model')}" + (f" shift {n['shift']:+g} eV" if n["shift"] else "")
                            for n in self.nexafs) if self.isld_basis == "nexafs" else ""),
                 f"  priors: curvature σ {self.sigma_curv} per {H0} eV "
                 f"(outside the scans {self.sigma_curv_outside or self.sigma_curv / 10:g})"
                 + "".join(f"; σ {c['sigma']} over {c['lo']:g}–{c['hi']:g} eV" for c in self.curv_regions)
                 + "".join(f"; {f['layer']} flat below {f['below']} eV (σ {f['sigma']})" for f in self.flat)
                 + "".join(f"; {a['layer']} isld → bare-atom {a['formula']} {a['density']:g} over "
                           f"{a['lo']:g}–{a['hi']:g} eV (σ {a['sigma']})" for a in self.anchor),
                 f"  mode: {'co-refinement with angle scans' if self.angle else 'energy scans only (geometry fixed)'}"]
        if self.scan_scales and any(self.scan_scales):
            lines.append("  own fitted scale for: "
                         + ", ".join(os.path.basename(s) for s, f in zip(self.scans, self.scan_scales) if f)
                         + f" (bounds {self.escan_scale_bounds}); other energy scans at scale 1")
        if self.angle:
            lines.append(f"  angle scans: θ < {self.min_theta}° left out"
                         + (f"; θ > {self.high_theta}° σ = {self.high_sigma}" if self.high_theta else "")
                         + f"; scale bounds {self.angle_scale_bounds}")
        if self.thickness_group:
            lines.append(f"  thickness group: {self.thickness_group}")
        if self.description:
            lines.append(f"  {self.description}")
        return "\n".join(lines)


RE_A = 2.8179403e-5          # classical electron radius, Å
NA = 6.02214076e23


def bare_isld(formula, density, E):
    """Bare-atom (tabulated scattering factor) imaginary SLD, 1e-6 Å⁻², of a formula at density g/cm³."""
    from kkcalc import data as kd
    st = kd.ParseChemicalFormula(formula)
    Ea, A = kd.calculate_asf(st)
    E = np.atleast_1d(np.asarray(E, float))
    f2 = np.array([kd.coeffs_to_ASF(e, A[(Ea > e).argmax() - 1]) for e in E])
    n_f = density / kd.calculate_FormulaMass(st) * NA * 1e-24
    return RE_A * n_f * f2 * 1e6


def curv_sigma(spec, E, e_lo, e_hi):
    """σ_curv (per H0 step) at energies E: sigma_curv inside the energy scans,
    sigma_curv_outside outside; curv_regions override (outside, the stiffer wins)."""
    E = np.atleast_1d(np.asarray(E, float))
    s_out = spec.sigma_curv_outside or spec.sigma_curv / 10.0
    inside = (E >= e_lo) & (E <= e_hi)
    sig = np.where(inside, spec.sigma_curv, s_out)
    for c in spec.curv_regions:
        r = (E >= float(c["lo"])) & (E <= float(c["hi"]))
        sig = np.where(r, np.where(inside, float(c["sigma"]), np.minimum(float(c["sigma"]), s_out)), sig)
    return sig


def bspline_knots(lo, hi, e_lo, e_hi, step_in, step_out, dense=()):
    """Breakpoints (incl. lo, hi) for the clamped cubic isld spline."""
    def seg(a, b, st):
        n = max(1, int(np.ceil((b - a) / st - 1e-9)))
        return np.linspace(a, b, n + 1)
    br = np.concatenate([seg(lo, e_lo, step_out) if e_lo - lo > 1e-6 else [lo],
                         seg(e_lo, e_hi, step_in),
                         seg(e_hi, hi, step_out) if hi - e_hi > 1e-6 else [hi]])
    for a, b, st in dense:
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            continue
        br = np.concatenate([br[(br < a) | (br > b)], seg(a, b, st)])
    br = np.unique(np.round(br, 6))
    keep = [br[0]]
    for v in br[1:]:
        if v - keep[-1] > 1e-3:
            keep.append(v)
    keep[-1] = hi
    return np.array(keep)


def bspline_basis(E, breaks, sigma_at):
    """Design matrix B (n_E × nb) of the clamped cubic spline on breakpoints, and the
    curvature-prior root L (nb × nb, ‖L c‖² = Σ h0³/σ(E)² ∫ (Σ c_k B_k″)² dE)."""
    from scipy.interpolate import BSpline
    t = np.concatenate([[breaks[0]] * 3, breaks, [breaks[-1]] * 3])
    nb = len(t) - 4
    B = BSpline.design_matrix(np.clip(E, breaks[0], breaks[-1]), t, 3).toarray()
    Ef = np.linspace(breaks[0], breaks[-1], int(np.ceil((breaks[-1] - breaks[0]) / GRAM_STEP)) + 1)
    B2 = BSpline(t, np.eye(nb), 3).derivative(2)(Ef)                  # (nf, nb)
    wq = np.full(Ef.size, Ef[1] - Ef[0])
    wq[[0, -1]] *= 0.5
    wq *= H0 ** 3 / sigma_at(Ef) ** 2
    G = B2.T @ (wq[:, None] * B2)
    lam, V = np.linalg.eigh(0.5 * (G + G.T))
    L = np.sqrt(np.clip(lam, 0.0, None))[:, None] * V.T
    return B, L, t


def nexafs_struct(m, shift=0.0):
    """Index structure of a NEXAFS peak model for the JAX evaluation (nexafs._flat order)."""
    from . import nexafs as NX
    flat = NX._flat(m)
    keys = [k for k, _ in flat]
    ix = {k: i for i, k in enumerate(keys)}
    vals = np.array([p["value"] for _, p in flat], float)
    free = np.array([i for i, (_, p) in enumerate(flat) if p["vary"]], int)
    lo = np.array([flat[i][1]["lo"] for i in free], float)
    hi = np.array([flat[i][1]["hi"] for i in free], float)
    return {"keys": keys, "vals": vals, "free": free, "lo": lo, "hi": hi, "shift": float(shift),
            "slope_ref": float(m["slope_ref"]), "ib": ix["baseline"], "isl": ix["slope"],
            "peaks": [(ix[f"{p['name']}.energy"], ix[f"{p['name']}.width"], ix[f"{p['name']}.height"])
                      for p in m["peaks"]],
            "edges": [tuple(ix[f"{e['name']}.{k}"] for k in ("location", "height", "width", "decay"))
                      for e in m["edges"]],
            "model": m}


def nexafs_eval(st, v, E):
    """NEXAFS model (baseline + slope + NEXAFS.edge_bl_func edges + FWHM Gaussians) at
    E + shift, differentiable in the full parameter vector v (JAX)."""
    import jax.numpy as jnp
    from jax.scipy.special import erf
    E = jnp.asarray(E) + st["shift"]
    y = v[st["ib"]] + v[st["isl"]] * (E - st["slope_ref"])
    for il, ih, iw, idc in st["edges"]:
        loc, h, w, dc = v[il], v[ih], v[iw], v[idc]
        step = jnp.where(E >= loc + w, 1.0, 0.0)
        y = y + 0.5 * h * (1 + erf((E - loc) * 2 * np.log(2) / w)) * (1 + step * (jnp.exp(-dc * (E - loc - w)) - 1))
    for ie, iw, ih in st["peaks"]:
        sg = v[iw] / (2 * np.sqrt(2 * np.log(2)))
        y = y + v[ih] * jnp.exp(-0.5 * ((E - v[ie]) / sg) ** 2)
    return y


def escan_dir(project, name):
    return os.path.join(project.path, "escan", name)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def read_scan(path):
    """Reduced energy scan → (E, R, theta).  θ from the '# theta = X deg' header."""
    theta = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            if "theta =" in line:
                theta = float(line.split("theta =")[1].split()[0])
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                rows.append([float(v) for v in s.split()[:2]])
            except ValueError:
                continue                                  # column header line
    d = np.array(rows)
    d = d[np.argsort(d[:, 0])]
    d = d[d[:, 1] > 0]
    if theta is None:
        raise ValueError(f"{path}: no 'theta =' in the header")
    return d[:, 0], d[:, 1], theta


def build_grid(scans, window, outer_step):
    """Nodes: one per energy-scan point (points within NODE_TOL merged), plus
    outer_step-spaced nodes out to the window edges (which are nodes)."""
    allE = np.sort(np.concatenate([s[0] for s in scans]))
    nodes = [allE[0]]
    for e in allE[1:]:
        if e - nodes[-1] > NODE_TOL:
            nodes.append(e)
    nodes = np.array(nodes)
    lo, hi = window
    if lo > nodes[0] or hi < nodes[-1]:
        raise ValueError(f"KK window {window} must contain the energy scans "
                         f"({nodes[0]:.1f}–{nodes[-1]:.1f} eV)")
    # outer nodes: window edges exactly, then outer_step spacing towards the data
    below = np.arange(lo, nodes[0] - 0.5 * outer_step, outer_step) if nodes[0] - lo > 1e-6 else np.array([])
    above = np.arange(nodes[-1] + outer_step, hi - 0.5 * outer_step, outer_step)
    if hi - nodes[-1] > 1e-6:
        above = np.append(above, hi)
    E = np.concatenate([below, nodes, above])
    data_idx = np.arange(below.size, below.size + nodes.size)
    assert np.all(np.diff(E) > 0)
    maps = []
    for Es, _, _ in scans:
        k = np.searchsorted(E, Es)
        k = np.clip(k, 1, E.size - 1)
        k = np.where(np.abs(E[k - 1] - Es) < np.abs(E[k] - Es), k - 1, k)
        if np.abs(E[k] - Es).max() > NODE_TOL:
            raise ValueError("energy-scan point not on a grid node")
        maps.append(k)
    return E, data_idx, maps


# ---------------------------------------------------------------------------
# seed model fits
# ---------------------------------------------------------------------------

def seed_fits(project, model, criteria="best"):
    """Best per-energy fits of `model` → {E: {"slabs": (nlayer, 4), "params": {name: value}}}."""
    import h5py
    out = {}
    if not os.path.exists(project.h5_path):
        return out
    with h5py.File(project.h5_path, "r") as f:
        s = f.get(project.sample_name)
        if s is None:
            return out
        for key in s:
            g = s[key].get(model) if hasattr(s[key], "get") else None
            if g is None:
                continue
            runs = [g[k] for k in g if k.startswith("run")]
            if not runs:
                continue
            if criteria == "last":
                r = runs[-1]
            else:
                r = min(runs, key=lambda r: r.attrs["chi_sq_final"])
            names = [v.decode() if isinstance(v, bytes) else v for v in r["parameters/names"][()]]
            out[float(key)] = {"slabs": r["structure_slabs_final"][()][:, :4],
                               "params": dict(zip(names, r["parameters/final_values"][()]))}
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# KK response
# ---------------------------------------------------------------------------

def _kk_call(args):
    E, x, formula, density, window = args
    from .stoich import kk_sld
    o = kk_sld(np.column_stack([E, np.zeros_like(E), x]), formula, density, list(window))
    return np.interp(E, o[:, 0], o[:, 1]), np.interp(E, o[:, 0], o[:, 2])


def _kk_col(args):
    E, x0, j, formula, density, window = args
    x = x0.copy()
    x[j] += 1.0
    return j, _kk_call((E, x, formula, density, window))


def kk_response(cache_dir, E, x_seed, formula, density, window, workers=8, log=print):
    """Linear KK map for fixed merge points: sld = s0 + M (x − x0), isld = i0 + N (x − x0).
    The end nodes of x0 are pinned to the tabulated isld (kkcalc rescale ≈ identity)."""
    key = hashlib.sha1(np.round(E, 4).tobytes() + f"{formula}|{density:.5f}|{window}".encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"kk_{formula}_{density:g}_{key}.npz")
    if os.path.exists(path):
        r = np.load(path)
        return {k: r[k] for k in ("x0", "s0", "i0", "M", "N")}, path
    _, it = _kk_call((E, x_seed, formula, density, window))
    x0 = x_seed.copy()
    x0[0], x0[-1] = it[0], it[-1]
    s0, i0 = _kk_call((E, x0, formula, density, window))
    n = E.size
    M, N = np.zeros((n, n)), np.zeros((n, n))
    from concurrent.futures import ProcessPoolExecutor
    w = max(1, int(workers))                 # reserved by the job runner (≤ 50 % CPUs machine-wide)
    with ProcessPoolExecutor(w) as ex:
        for j, (s, i) in ex.map(_kk_col, [(E, x0, j, formula, density, window) for j in range(1, n - 1)],
                                chunksize=4):
            M[:, j], N[:, j] = s - s0, i - i0
    dev = float(np.abs(N[1:-1, 1:-1] - np.eye(n - 2)).max())
    log(f"[escan] KK response {formula} {density:g} g/cm³ on {n} nodes ({w} workers); "
        f"tabulated isld ends {x0[0]:.3f}/{x0[-1]:.3f}; max |N − I| = {dev:.2e}")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, E=E, x0=x0, s0=s0, i0=i0, M=M, N=N, formula=formula, density=density)
    return {"x0": x0, "s0": s0, "i0": i0, "M": M, "N": N}, path


# ---------------------------------------------------------------------------
# problem assembly
# ---------------------------------------------------------------------------

class Problem:
    """Everything the residual needs, assembled from project + spec."""

    def __init__(self, project, spec, log=print, workers=8, start=None):
        from .materials import resolve
        self.p, self.spec, self.log = project, spec, log
        recipe = project.get_model(spec.model)
        self.recipe = recipe
        layers = recipe.layers
        self.lnames = [l.name for l in layers]
        kk_by_layer = {k["layer"]: k for k in spec.kk}
        for nm in kk_by_layer:
            if nm not in self.lnames:
                raise ValueError(f"KK layer '{nm}' not in {spec.model} ({self.lnames})")
        fits = seed_fits(project, spec.seed_model or spec.model, spec.criteria)
        if not fits:
            raise ValueError(f"no fits of {spec.seed_model or spec.model} to seed from "
                             f"(set seed_model to a fitted model)")
        seed_layers = project.get_model(spec.seed_model or spec.model).layer_order

        # ---- energy scans
        self.scans = [read_scan(s if os.path.isabs(s) else os.path.join(project.path, s)) for s in spec.scans]
        lo_d = min(s[0][0] for s in self.scans)
        hi_d = max(s[0][-1] for s in self.scans)
        window = spec.window or [np.floor(lo_d) - 20.0, np.ceil(hi_d) + 30.0]
        self.window = window
        E, data_idx, maps = build_grid(self.scans, window, spec.outer_step)
        self.E, self.data_idx, self.maps = E, data_idx, maps
        self.theta = [s[2] for s in self.scans]
        self.w_scan = np.array(spec.scan_weights or [1.0] * len(self.scans), float)

        # ---- angle scans (energies with data and seed fits)
        self.angle = spec.angle
        if spec.angle:
            data, elist = project.load_data()
            self.Ea = np.array([e for e in sorted(elist) if any(abs(e - f) < 1e-3 for f in fits)])
            self.ang = []
            for e in self.Ea:
                ds = data[e]
                q, R = np.asarray(ds.x, float), np.asarray(ds.y, float)
                ok = R > 0
                self.ang.append(np.column_stack([q[ok], R[ok]]))
        else:
            self.Ea = np.array(sorted(fits))
            self.ang = []
        fitE = np.array(sorted(fits))
        nearest = lambda e: fits[float(fitE[np.argmin(abs(fitE - e))])]

        # ---- per-layer roles and tables
        self.roles, self.tables = [], {}
        for i, l in enumerate(layers):
            if l.name in kk_by_layer:
                self.roles.append("kk")
            elif l.is_constant:
                self.roles.append("const")
            elif l.sld_offset or l.sld_real_bounds or l.sld_imag_bounds:
                self.roles.append("free")
            else:
                self.roles.append("table")
            if not l.is_constant:
                need = np.unique(np.concatenate([E, self.Ea]))
                self.tables[l.name] = resolve(project.material(l.material), need, project.path)

        def table_at(name, Eq):
            t = self.tables[name]
            return np.interp(Eq, t[:, 0], t[:, 1]), np.interp(Eq, t[:, 0], t[:, 2])
        self.table_at = table_at

        def seed_layer_value(name, e, col):     # col 1 sld, 2 isld, 0 thick, 3 rough
            if name in seed_layers:
                return float(nearest(e)["slabs"][seed_layers.index(name), col])
            return None
        self._seed_layer_value = seed_layer_value

        # ---- geometry parameters
        geo = []                                  # (kind, layer index, lo, hi, start)
        group = spec.thickness_group or {}
        gl = [self.lnames.index(n) for n in group.get("layers", [])]
        med = lambda name, col, default: (np.median([seed_layer_value(name, e, col) for e in fitE])
                                          if name in seed_layers else default)
        for i, l in enumerate(layers):
            if i == 0:
                continue                           # fronting
            last = i == len(layers) - 1
            tb, rb = l.thickness_bounds, l.roughness_bounds
            if not last and spec.angle and tb and tb[2]:
                geo.append(("thick", i, tb[0], tb[1], med(l.name, 0, l.thickness)))
            if spec.angle and rb and rb[2]:
                geo.append(("rough", i, rb[0], rb[1], med(l.name, 3, l.roughness)))
        if gl and spec.angle:
            geo = [g for g in geo if not (g[0] == "thick" and g[1] in gl)]
            tot0 = sum(med(self.lnames[i], 0, layers[i].thickness) for i in gl)
            for i in gl[:-1]:
                l = layers[i]
                tb = l.thickness_bounds or (0.0, tot0, True)
                geo.append(("thick", i, tb[0], tb[1], med(l.name, 0, l.thickness)))
            lo_, hi_ = group["bounds"]
            geo.append(("total", gl[-1], float(lo_), float(hi_), float(np.clip(tot0, lo_, hi_))))
        self.geo = geo
        self.group = gl
        # fixed geometry values (seed medians / recipe)
        self.thick0 = np.array([0.0] + [med(l.name, 0, l.thickness) for l in layers[1:]])
        self.rough0 = np.array([0.0] + [med(l.name, 3, l.roughness) for l in layers[1:]])
        self.thick0[-1] = 0.0
        dqs = [v["params"].get("dq - resolution") for v in fits.values()]
        self.dq = float(np.median([d for d in dqs if d is not None])) if any(d is not None for d in dqs) \
            else recipe.instrument.dq

        # ---- KK paths: seeds and responses
        self.kk_layers = [l for l in self.lnames if l in kk_by_layer]
        cache = escan_dir(project, spec.name)
        self.kk = {}
        for name in self.kk_layers:
            k = kk_by_layer[name]
            seed_E = fitE
            if name in seed_layers:
                s_isld = np.array([seed_layer_value(name, e, 2) for e in seed_E])
                s_sld = np.array([seed_layer_value(name, e, 1) for e in seed_E])
            else:
                s_sld, s_isld = table_at(name, seed_E)
            x_seed = np.interp(E, seed_E, s_isld)
            resp, rpath = kk_response(cache, E, x_seed, k["formula"], k["density"], window, workers, log)
            self.kk[name] = dict(resp, path=rpath, seed_E=seed_E, seed_sld=s_sld, seed_isld=s_isld)

        # ---- isld basis per KK layer
        self.basis = {}
        e_lo, e_hi = E[data_idx[0]], E[data_idx[-1]]
        for name in self.kk_layers:
            if spec.isld_basis == "bspline":
                br = bspline_knots(E[0], E[-1], e_lo, e_hi, spec.knot_step,
                                   spec.knot_step_outside or spec.outer_step, spec.knot_dense)
                B, L, t = bspline_basis(E, br, lambda x: curv_sigma(spec, x, e_lo, e_hi))
                self.basis[name] = {"kind": "bspline", "B": B, "L": L, "t": t, "breaks": br,
                                    "npar": B.shape[1] - 2}
            elif spec.isld_basis == "nexafs":
                from . import nexafs as NX
                ent = {n["layer"]: n for n in spec.nexafs}[name]
                if "fit" in ent:
                    fr = NX.load_fit(project, ent["fit"])
                    if fr.get("signal") not in (None, "isld"):
                        raise ValueError(f"nexafs fit '{ent['fit']}' was fitted to {fr['signal']}, "
                                         f"not isld; refit it with --signal isld")
                    m, shift = fr["fitted_model"], float(fr.get("energy_shift", 0.0))
                else:
                    m, shift = NX.get_model(project, ent["model"]), 0.0
                st = nexafs_struct(m, shift + ent["shift"])
                st.update(kind="nexafs", npar=int(st["free"].size), source=ent)
                self.basis[name] = st
                log(f"[escan] {name}: NEXAFS basis {m['name']} ({len(st['peaks'])} peaks, "
                    f"{len(st['edges'])} edges, {st['npar']} free), evaluated at E {st['shift']:+g} eV")
            else:
                self.basis[name] = {"kind": "nodes", "npar": E.size - 2}

        # ---- free layers (per angle-scan energy)
        self.free_layers = [l.name for l, r in zip(layers, self.roles) if r == "free"]
        self.free_bounds = {}
        for name in self.free_layers:
            l = recipe.layer(name)
            ts, ti = table_at(name, self.Ea)
            if l.sld_real_bounds:
                rlo, rhi = np.full(ts.size, l.sld_real_bounds[0]), np.full(ts.size, l.sld_real_bounds[1])
            else:
                o = l.sld_offset["real"]
                rlo, rhi = ts + o[0], ts + o[1]
            if l.sld_imag_bounds:
                ilo, ihi = np.full(ti.size, l.sld_imag_bounds[0]), np.full(ti.size, l.sld_imag_bounds[1])
            else:
                o = l.sld_offset["imag"]
                ilo, ihi = np.maximum(ti + o[0], 0.0), ti + o[1]
            s0 = np.array([seed_layer_value(name, e, 1) if name in seed_layers else np.nan for e in self.Ea])
            i0 = np.array([seed_layer_value(name, e, 2) if name in seed_layers else np.nan for e in self.Ea])
            s0 = np.where(np.isnan(s0), ts, s0)
            i0 = np.where(np.isnan(i0), ti, i0)
            self.free_bounds[name] = (rlo, rhi, ilo, ihi, s0, i0)

        # ---- angle-scan instrument seeds
        self.scale0 = float(np.median([v["params"].get("scale", 1.0) for v in fits.values()])) if spec.angle else 1.0
        self.bkg0 = np.array([nearest(e)["params"].get("bkg", 0.0) for e in self.Ea]) if spec.angle else np.zeros(0)
        self.start = start
        self._layout()
        self._arrays()

    # ------------------------------------------------------------------
    def _layout(self):
        n = self.E.size
        self.nf = n - 2
        idx, k = {}, 0
        for name in self.kk_layers:
            m = self.basis[name]["npar"]
            idx[("x", name)] = slice(k, k + m); k += m
            idx[("c", name)] = k; k += 1
        for j, f in enumerate(self.spec.scan_scales or []):
            if f:
                idx[("es", j)] = k; k += 1
        if self.angle:
            for g in self.geo:
                idx[(g[0], g[1])] = k; k += 1
            idx["scale"] = k; k += 1
            nA = self.Ea.size
            idx["bkg"] = slice(k, k + nA); k += nA
            for name in self.free_layers:
                idx[("fs", name)] = slice(k, k + nA); k += nA
                idx[("fi", name)] = slice(k, k + nA); k += nA
        self.idx, self.NP = idx, k

    def _arrays(self):
        import jax.numpy as jnp
        from gpu_reflect import compute_smear_params
        E, spec = self.E, self.spec
        # energy-scan smearing grids (data points only)
        self.QQe, self.lnRe, self.Ee = [], [], []
        combo = None
        for (Es, Rs, th), m in zip(self.scans, self.maps):
            q = 4 * np.pi * np.sin(np.radians(th)) / (HC / Es)
            qq, combo = compute_smear_params(q, q * self.dq / 100)
            self.QQe.append(jnp.asarray(qq[:, None, :]))
            self.lnRe.append(jnp.asarray(np.log(Rs)))
            self.Ee.append(Es)
        self.COMBO = jnp.asarray(combo if combo is not None else compute_smear_params(np.array([0.01]), np.array([1e-4]))[1])
        # angle scans, padded
        if self.angle:
            nA, nq = self.Ea.size, max(len(d) for d in self.ang)
            QQa = np.zeros((nA, nq, np.asarray(self.COMBO).size))
            lnRa, mask, THa = np.zeros((nA, nq)), np.zeros((nA, nq)), np.zeros((nA, nq))
            for k, d in enumerate(self.ang):
                qq, _ = compute_smear_params(d[:, 0], d[:, 0] * self.dq / 100)
                QQa[k, :len(d)] = qq
                QQa[k, len(d):] = qq[-1]
                lnRa[k, :len(d)] = np.log(d[:, 1])
                mask[k, :len(d)] = 1
                THa[k, :len(d)] = np.degrees(np.arcsin(d[:, 0] * HC / self.Ea[k] / (4 * np.pi)))
            sig = np.full((nA, nq), spec.sigma_data)
            if spec.high_theta:
                for k in range(nA):
                    hi = (THa[k] > spec.high_theta) & (mask[k] > 0)
                    if not hi.any():
                        continue
                    if spec.high_sigma == "auto":
                        sig[k, hi] = max(spec.sigma_data, self._seed_scatter(k, hi))
                    else:
                        sig[k, hi] = float(spec.high_sigma)
            fit = mask * (THa >= spec.min_theta)
            self.QQa, self.lnRa, self.mask = jnp.asarray(QQa), jnp.asarray(lnRa), mask
            self.THa, self.SIGa, self.FITa = THa, sig, fit
            self.Wa = jnp.asarray(fit / sig)
            self.Rmin = np.array([d[:, 1].min() for d in self.ang])
        # fixed tables at escan points and angle energies
        self.tab_E = {n: self.table_at(n, E) for n in self.tables}
        self.tab_a = {n: self.table_at(n, self.Ea) for n in self.tables}
        # priors (per eV, σ per H0 step)
        rows = []
        h = np.diff(E)
        D2 = np.zeros((E.size - 2, E.size))
        e_lo, e_hi = E[self.data_idx[0]], E[self.data_idx[-1]]
        sig_n = curv_sigma(spec, E, e_lo, e_hi)
        for j in range(1, E.size - 1):
            hm, hp = h[j - 1], h[j]
            hb = 0.5 * (hm + hp)
            w = H0 ** 2 * np.sqrt(hb / H0) / sig_n[j]
            D2[j - 1, j - 1:j + 2] = w * 2 * np.array([1 / (hm * (hm + hp)), -1 / (hm * hp), 1 / (hp * (hm + hp))])
        # curvature prior per layer acts on the basis vector (paths(...)[3]): node values
        # (D2), spline coefficients (Gram root L) or nothing (NEXAFS); flat prior on x
        self.Pc, self.Pf = {}, {}
        for name in self.kk_layers:
            kind = self.basis[name]["kind"]
            self.Pc[name] = (jnp.asarray(D2) if kind == "nodes" else
                             jnp.asarray(self.basis[name]["L"]) if kind == "bspline" else None)
            P = []
            for f in spec.flat:
                if f["layer"] != name:
                    continue
                sel = np.where(E[1:] <= float(f["below"]))[0]
                D1 = np.zeros((sel.size, E.size))
                for r, j in enumerate(sel):
                    w = H0 * np.sqrt(h[j] / H0) / float(f["sigma"]) / h[j]
                    D1[r, j], D1[r, j + 1] = -w, w
                P.append(D1)
            self.Pf[name] = jnp.asarray(np.vstack(P)) if P else None
        self.Bj = {n: jnp.asarray(b["B"]) for n, b in self.basis.items() if b["kind"] == "bspline"}
        # anchors: isld at grid nodes in [lo, hi] pulled to a formula's bare-atom isld
        self.anchors = []
        for a in spec.anchor:
            sel = np.where((E >= float(a["lo"])) & (E <= float(a["hi"])))[0]
            tgt = bare_isld(a["formula"], float(a["density"]), E[sel])
            self.anchors.append((a["layer"], jnp.asarray(sel), jnp.asarray(tgt), float(a["sigma"])))
        self.Mj = {n: jnp.asarray(self.kk[n]["M"]) for n in self.kk_layers}
        self.Nj = {n: jnp.asarray(self.kk[n]["N"]) for n in self.kk_layers}

    def _seed_scatter(self, k, sel):
        """rms ln residual of the seed model's own fit at energy k over the selected points."""
        from refnx.reflect import reflectivity
        fits = seed_fits(self.p, self.spec.seed_model or self.spec.model, self.spec.criteria)
        fe = np.array(sorted(fits))
        f = fits[float(fe[np.argmin(abs(fe - self.Ea[k]))])]
        d = self.ang[k]
        par = f["params"]
        R = reflectivity(d[:, 0], f["slabs"], scale=par.get("scale", 1.0), bkg=par.get("bkg", 0.0),
                         dq=par.get("dq - resolution", self.dq), threads=1)
        r = np.log(np.maximum(R, 1e-12)) - np.log(d[:, 1])
        return float(np.sqrt(np.mean(r[sel[:len(d)]] ** 2)))

    # ------------------------------------------------------------------
    def paths(self, p):
        """KK layers → {name: (sld on grid, isld on grid, x, basis vector)}; the basis
        vector is what the curvature prior acts on (x for nodes, the full coefficient
        vector for bspline, the free model parameters for nexafs)."""
        import jax.numpy as jnp
        out = {}
        for name in self.kk_layers:
            k, b = self.kk[name], self.basis[name]
            v = p[self.idx[("x", name)]]
            x0 = jnp.asarray(k["x0"])
            if b["kind"] == "bspline":
                cv = jnp.concatenate([x0[:1], v, x0[-1:]])
                x = self.Bj[name] @ cv
            elif b["kind"] == "nexafs":
                full = jnp.asarray(b["vals"]).at[b["free"]].set(v)
                x = x0.at[1:-1].set(jnp.maximum(nexafs_eval(b, full, self.E[1:-1]), 0.0))
                cv = v
            else:
                x = x0.at[1:-1].set(v)
                cv = x
            s = jnp.asarray(k["s0"]) + self.Mj[name] @ (x - x0) + p[self.idx[("c", name)]]
            i = jnp.asarray(k["i0"]) + self.Nj[name] @ (x - x0)
            out[name] = (s, i, x, cv)
        return out

    def _geometry(self, p):
        import jax.numpy as jnp
        thick = jnp.asarray(self.thick0)
        rough = jnp.asarray(self.rough0)
        if self.angle:
            for g in self.geo:
                v = p[self.idx[(g[0], g[1])]]
                if g[0] == "thick":
                    thick = thick.at[g[1]].set(v)
                elif g[0] == "rough":
                    rough = rough.at[g[1]].set(v)
            if self.group:
                tot = p[self.idx[("total", self.group[-1])]]
                rest = sum(thick[i] for i in self.group[:-1])
                thick = thick.at[self.group[-1]].set(tot - rest)
        return thick, rough

    def _layers(self, Eq, at_angle, P, p):
        """(nE, nlayer, 4) slab stack at energies Eq (angle-scan energies if at_angle)."""
        import jax.numpy as jnp
        thick, rough = self._geometry(p)
        m = Eq.size
        rows = []
        for i, (name, role) in enumerate(zip(self.lnames, self.roles)):
            if role == "const":
                l = self.recipe.layer(name)
                re_, im_ = l.constant_sld["real"], l.constant_sld["imag"]
                s, ii = jnp.full(m, re_), jnp.full(m, im_)
            elif role == "kk":
                sg, ig = P[name][:2]
                s, ii = jnp.interp(jnp.asarray(Eq), jnp.asarray(self.E), sg), jnp.interp(jnp.asarray(Eq), jnp.asarray(self.E), ig)
            elif role == "free":
                fs, fi = self._free_values(name, p)
                if at_angle:
                    s, ii = fs, fi
                else:
                    ts, ti = self.tab_E[name]
                    inside = (self.E >= self.Ea[0]) & (self.E <= self.Ea[-1])
                    s = jnp.where(inside, jnp.interp(jnp.asarray(self.E), jnp.asarray(self.Ea), fs), ts)
                    ii = jnp.where(inside, jnp.interp(jnp.asarray(self.E), jnp.asarray(self.Ea), fi), ti)
                    s = jnp.interp(jnp.asarray(Eq), jnp.asarray(self.E), s)
                    ii = jnp.interp(jnp.asarray(Eq), jnp.asarray(self.E), ii)
            else:
                ts, ti = self.table_at(name, np.asarray(Eq))
                s, ii = jnp.asarray(ts), jnp.asarray(ti)
            rows.append(jnp.column_stack([jnp.full(m, thick[i] if i else 0.0), s, ii, jnp.full(m, rough[i])]))
        return jnp.stack(rows, axis=1)

    def _free_values(self, name, p):
        import jax.numpy as jnp
        if self.angle:
            return p[self.idx[("fs", name)]], jnp.maximum(p[self.idx[("fi", name)]], 0.0)
        b = self.free_bounds[name]
        return jnp.asarray(b[4]), jnp.asarray(b[5])

    def forward(self, p):
        import jax
        import jax.numpy as jnp
        from gpu_reflect import _jabeles_smeared_pos
        smear = jax.vmap(_jabeles_smeared_pos, in_axes=(0, None, 0, 0, 0))
        P = self.paths(p)
        Re = []
        for j, ((Es, _, _), qq) in enumerate(zip(self.scans, self.QQe)):
            lay = self._layers(Es, False, P, p)
            sc = p[self.idx[("es", j)]] if ("es", j) in self.idx else 1.0
            Re.append(sc * smear(qq, self.COMBO, lay, jnp.ones(Es.size), jnp.zeros(Es.size))[:, 0])
        Ra = None
        if self.angle:
            lay = self._layers(self.Ea, True, P, p)
            nA = self.Ea.size
            Ra = smear(self.QQa, self.COMBO, lay, jnp.full(nA, p[self.idx["scale"]]), p[self.idx["bkg"]])
        return Ra, Re, P

    def residual(self, p):
        import jax.numpy as jnp
        Ra, Re, P = self.forward(p)
        parts = []
        if self.angle:
            parts.append(((jnp.log(jnp.maximum(Ra, 1e-12)) - self.lnRa) * self.Wa).ravel())
        for w, R, lnR in zip(self.w_scan, Re, self.lnRe):
            parts.append(w * (jnp.log(R) - lnR) / self.spec.sigma_data)
        for name in self.kk_layers:
            if self.Pc[name] is not None:
                parts.append(self.Pc[name] @ P[name][3])
            if self.Pf[name] is not None:
                parts.append(self.Pf[name] @ P[name][2])
        for layer, sel, tgt, sig in self.anchors:
            parts.append((P[layer][1][sel] - tgt) / sig)
        return jnp.concatenate(parts)

    # ------------------------------------------------------------------
    def x_from_isld(self, name, isld_grid):
        k = self.kk[name]
        free = np.arange(1, self.E.size - 1)
        sol, *_ = np.linalg.lstsq(k["N"][:, free], isld_grid - k["i0"] + k["N"][:, free] @ k["x0"][free], rcond=None)
        return np.clip(sol, 0.0, 10.0)

    def seed_vector(self):
        """Start values + bounds (from the seed model, or a previous result)."""
        E = self.E
        p, lb, ub = np.zeros(self.NP), np.zeros(self.NP), np.zeros(self.NP)
        prev = self.start
        for name in self.kk_layers:
            k = self.kk[name]
            if prev is not None and f"sld_{name}" in prev:
                Eo, so, io = prev["E_grid"], prev[f"sld_{name}"], prev[f"isld_{name}"]
            else:
                Eo, so, io = k["seed_E"], k["seed_sld"], k["seed_isld"]
            tgt = np.interp(E, Eo, io)
            lo, hi = E < Eo[0], E > Eo[-1]
            tgt[lo] = np.interp(E[lo], [E[0], Eo[0]], [k["x0"][0], io[0]])
            tgt[hi] = np.interp(E[hi], [Eo[-1], E[-1]], [io[-1], k["x0"][-1]])
            xf = self.x_from_isld(name, tgt)
            full = np.concatenate([[k["x0"][0]], xf, [k["x0"][-1]]])
            b = self.basis[name]
            sx, sc = self.idx[("x", name)], self.idx[("c", name)]
            if b["kind"] == "bspline":
                # least-squares coefficients for the target x (end coefficients pinned)
                B = b["B"]
                rhs = full - B[:, 0] * k["x0"][0] - B[:, -1] * k["x0"][-1]
                cf, *_ = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)
                if prev is not None and str(prev.get(f"basis_{name}", "")) == "bspline" \
                        and prev[f"bparams_{name}"].size == cf.size:
                    cf = np.asarray(prev[f"bparams_{name}"], float)
                full = B @ np.concatenate([[k["x0"][0]], cf, [k["x0"][-1]]])
                p[sx], lb[sx], ub[sx] = cf, -2.0, 15.0
            elif b["kind"] == "nexafs":
                v0 = b["vals"][b["free"]].copy()
                if prev is not None and str(prev.get(f"basis_{name}", "")) == "nexafs" \
                        and prev[f"bparams_{name}"].size == v0.size:
                    v0 = np.asarray(prev[f"bparams_{name}"], float)
                lo_, hi_ = np.nan_to_num(b["lo"], neginf=-1e3), np.nan_to_num(b["hi"], posinf=1e3)
                v0 = np.clip(v0, lo_, hi_)
                fv = b["vals"].copy()
                fv[b["free"]] = v0
                xi = np.maximum(np.asarray(nexafs_eval(b, fv, E[1:-1])), 0.0)
                full = np.concatenate([[k["x0"][0]], xi, [k["x0"][-1]]])
                p[sx], lb[sx], ub[sx] = v0, lo_, hi_
            else:
                p[sx], lb[sx], ub[sx] = xf, 0.0, 10.0
            sl = k["s0"] + k["M"] @ (full - k["x0"])
            inside = (E >= max(Eo[0], E[self.data_idx[0]])) & (E <= min(Eo[-1], E[self.data_idx[-1]]))
            c = float(np.mean(np.interp(E[inside], Eo, so) - sl[inside])) if inside.any() else 0.0
            p[sc], lb[sc], ub[sc] = c, -10.0, 10.0
        prevp = prev.get("params") if prev is not None else None
        for j, f in enumerate(self.spec.scan_scales or []):
            if f:
                i = self.idx[("es", j)]
                p[i], (lb[i], ub[i]) = 1.0, self.spec.escan_scale_bounds
                if prevp is not None and f"escale:{j}" in prevp:
                    p[i] = prevp[f"escale:{j}"]
        if self.angle:
            for g in self.geo:
                i = self.idx[(g[0], g[1])]
                p[i], lb[i], ub[i] = g[4], g[2], g[3]
                if prevp is not None and f"{g[0]}:{self.lnames[g[1]]}" in prevp:
                    p[i] = prevp[f"{g[0]}:{self.lnames[g[1]]}"]
            i = self.idx["scale"]
            p[i], (lb[i], ub[i]) = self.scale0, self.spec.angle_scale_bounds
            if prevp is not None and "scale" in prevp:
                p[i] = prevp["scale"]
            sb = self.idx["bkg"]
            p[sb], lb[sb], ub[sb] = np.clip(self.bkg0, 0, self.Rmin), 0.0, self.Rmin
            for name in self.free_layers:
                rlo, rhi, ilo, ihi, s0, i0 = self.free_bounds[name]
                if prev is not None and f"free_sld_{name}" in prev:
                    s0, i0 = prev[f"free_sld_{name}"], prev[f"free_isld_{name}"]
                fs, fi = self.idx[("fs", name)], self.idx[("fi", name)]
                p[fs], lb[fs], ub[fs] = s0, rlo, rhi
                p[fi], lb[fi], ub[fi] = i0, ilo, ihi
        eps = 1e-9 * np.maximum(1.0, np.abs(ub - lb))
        return np.clip(p, lb + eps, ub - eps), lb, ub

    def param_dict(self, p):
        d = {}
        if self.angle:
            for g in self.geo:
                d[f"{g[0]}:{self.lnames[g[1]]}"] = float(p[self.idx[(g[0], g[1])]])
            d["scale"] = float(p[self.idx["scale"]])
        for name in self.kk_layers:
            d[f"offset:{name}"] = float(p[self.idx[("c", name)]])
        for j, f in enumerate(self.spec.scan_scales or []):
            if f:
                d[f"escale:{j}"] = float(p[self.idx[("es", j)]])
        return d


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _chunked_jacfwd(f, n, chunk=None):
    """Forward-mode Jacobian of f (ℝⁿ → ℝᵐ) in passes of ``chunk`` columns.

    jax.jacfwd pushes all n tangents through f at once (memory ∝ n); the carbon
    two-layer KK fit (1096 parameters) ran out of 48 GB that way.  Here f is
    linearized once and the tangents go through in blocks (lax.map over vmap).
    n ≤ chunk → plain jacfwd."""
    import jax
    import jax.numpy as jnp
    chunk = chunk or JAC_CHUNK
    if n <= chunk:
        return jax.jacfwd(f)
    nb = -(-n // chunk)

    def jac(x):
        _, f_jvp = jax.linearize(f, x)
        basis = jnp.eye(nb * chunk, n, dtype=x.dtype).reshape(nb, chunk, n)
        cols = jax.lax.map(jax.vmap(f_jvp), basis)          # (nb, chunk, m)
        return cols.reshape(nb * chunk, -1)[:n].T
    return jac


def run(project, name, start="last", max_nfev=None, workers=8, log=print):
    """Fit one escan spec; writes escan/<name>/ and returns a summary dict.
    start: 'last' (previous result if any, else seed model) | 'model' | path to result.npz"""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import least_squares
    jax.config.update("jax_enable_x64", True)
    spec = project.get_escan(name)
    d = escan_dir(project, name)
    os.makedirs(d, exist_ok=True)
    prev = None
    if start == "last" and os.path.exists(os.path.join(d, "result.npz")):
        prev = _load_result(os.path.join(d, "result.npz"))
    elif start not in ("last", "model"):
        prev = _load_result(start)
    pr = Problem(project, spec, log=log, workers=workers, start=prev)
    p0, lb, ub = pr.seed_vector()
    res_j = jax.jit(pr.residual)
    jac_j = jax.jit(_chunked_jacfwd(pr.residual, pr.NP))
    r0 = np.asarray(res_j(jnp.asarray(p0)))
    n_data = sum(s[0].size for s in pr.scans) + (int(pr.FITa.sum()) if pr.angle else 0)
    log(f"[escan] {name}: {pr.NP} parameters, {n_data} data points, KK grid {pr.E.size} nodes "
        f"({pr.E[0]:g}–{pr.E[-1]:g} eV); start cost {0.5 * np.sum(r0 ** 2):.1f} "
        f"(start: {'previous result' if prev is not None else 'seed model'})")
    r = least_squares(lambda x: np.asarray(res_j(jnp.asarray(x))), p0,
                      jac=lambda x: np.asarray(jac_j(jnp.asarray(x))), bounds=(lb, ub),
                      method="trf", x_scale="jac", max_nfev=max_nfev or spec.max_nfev)
    p = r.x
    Ra, Re, P = pr.forward(jnp.asarray(p))
    Re = [np.asarray(x) for x in Re]
    lre = [np.log(R) - np.asarray(l) for R, l in zip(Re, pr.lnRe)]
    cost_e = [float(0.5 * np.sum((w * l / spec.sigma_data) ** 2)) for w, l in zip(pr.w_scan, lre)]
    summ = {"name": name, "model": spec.model, "cost": float(r.cost), "nfev": int(r.nfev),
            "status": int(r.status), "converged": bool(r.status > 0), "cost_escan": cost_e,
            "escan_pts_gt5pct": [int((np.abs(l) > np.log(1.05)).sum()) for l in lre],
            "escan_theta": pr.theta, "params": pr.param_dict(p), "dq": pr.dq,
            "kk_grid": [float(pr.E[0]), float(pr.E[-1]), int(pr.E.size)],
            "kk_response": {n: pr.kk[n]["path"] for n in pr.kk_layers},
            "isld_basis": spec.isld_basis,
            "basis_params": {n: int(pr.basis[n]["npar"]) for n in pr.kk_layers}}
    save = dict(E_grid=pr.E, data_idx=pr.data_idx, p=p)
    for n in pr.kk_layers:
        s, i, x = (np.asarray(a) for a in P[n][:3])
        save[f"sld_{n}"], save[f"isld_{n}"] = s, i
        b = pr.basis[n]
        save[f"basis_{n}"] = b["kind"]
        save[f"bparams_{n}"] = np.asarray(p[pr.idx[("x", n)]])
        if b["kind"] == "bspline":
            save[f"knots_{n}"] = b["breaks"]
        elif b["kind"] == "nexafs":
            from . import nexafs as NX
            vals = dict(zip(b["keys"], b["vals"]))
            vals.update({b["keys"][j]: float(v) for j, v in zip(b["free"], save[f"bparams_{n}"])})
            fm = NX._set_values(b["model"], vals)
            fm["description"] = (f"refined by escan '{name}' ({fm.get('description', '')})").strip()
            with open(os.path.join(d, f"nexafs_{n}.json"), "w") as fh:
                json.dump({"model": fm, "shift": b["shift"], "source": b["source"]}, fh, indent=2, default=float)
            free = {b["keys"][j] for j in b["free"]}
            summ.setdefault("nexafs", {})[n] = {
                "values": {k: float(v) for k, v in vals.items() if k in free},
                "at_bound": [b["keys"][j] for j, v, lo_, hi_ in
                             zip(b["free"], save[f"bparams_{n}"], b["lo"], b["hi"])
                             if np.isfinite(lo_) and np.isfinite(hi_) and min(v - lo_, hi_ - v) < 1e-3 * (hi_ - lo_)],
                "model_json": os.path.join(d, f"nexafs_{n}.json")}
    for k, (Es, Rs, th) in enumerate(pr.scans):
        save[f"escan{k}_E"], save[f"escan{k}_R"], save[f"escan{k}_model"] = Es, Rs, Re[k]
        save[f"escan{k}_theta"] = th
    if pr.angle:
        Ra = np.asarray(Ra)
        lra = (np.log(np.maximum(Ra, 1e-12)) - np.asarray(pr.lnRa)) * pr.mask
        summ["cost_angle"] = float(0.5 * np.sum((lra * np.asarray(pr.Wa)) ** 2))
        summ["bkg"] = [float(v) for v in p[pr.idx["bkg"]]]
        rng = {}
        for nm, sel in (("below_min_theta", pr.THa < spec.min_theta),
                        ("fitted", pr.FITa > 0)):
            v = lra[(pr.mask > 0) & sel]
            if v.size:
                rng[nm] = {"mean": float(v.mean()), "rms": float(np.sqrt(np.mean(v ** 2))), "n": int(v.size)}
        summ["angle_resid"] = rng
        save.update(Ea=pr.Ea, Ra=Ra, lra=lra, mask=pr.mask, fit_mask=pr.FITa, theta_angle=pr.THa,
                    angle_q=_pad([a[:, 0] for a in pr.ang]), angle_R=_pad([a[:, 1] for a in pr.ang]))
        for n in pr.free_layers:
            fs, fi = (np.asarray(a) for a in pr._free_values(n, jnp.asarray(p)))
            save[f"free_sld_{n}"], save[f"free_isld_{n}"] = fs, fi
            np.savetxt(os.path.join(d, f"free_{n}.csv"), np.column_stack([pr.Ea, fs, fi]), delimiter=",",
                       header="E_eV,sld,isld", comments="", fmt="%.5f")
        summ["at_bound"] = [k for k, (lo_, hi_, v) in
                            {f"{g[0]}:{pr.lnames[g[1]]}": (g[2], g[3], p[pr.idx[(g[0], g[1])]]) for g in pr.geo}.items()
                            if min(v - lo_, hi_ - v) < 1e-3 * (hi_ - lo_)]
    for n in pr.kk_layers:
        di = pr.data_idx
        np.savetxt(os.path.join(d, f"paths_{n}.csv"),
                   np.column_stack([pr.E, save[f"sld_{n}"], save[f"isld_{n}"],
                                    np.isin(np.arange(pr.E.size), di).astype(int)]),
                   delimiter=",", header="E_eV,sld,isld,has_escan_data", comments="", fmt="%.5f")
    save["summary"] = json.dumps(summ)
    if os.path.exists(os.path.join(d, "result.npz")):
        shutil.copy(os.path.join(d, "result.npz"), os.path.join(d, "result_prev.npz"))
    np.savez(os.path.join(d, "result.npz"), **save)
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    log(f"[escan] {name}: cost {summ['cost']:.1f} after {summ['nfev']} evaluations "
        f"({'converged' if summ['converged'] else 'max_nfev reached'}); "
        f"energy scans {', '.join(f'{c:.1f}' for c in cost_e)}"
        + (f"; angle scans {summ['cost_angle']:.1f}; scale {summ['params']['scale']:.3f}" if pr.angle else ""))
    return summ


def _pad(arrs):
    n = max(len(a) for a in arrs)
    out = np.full((len(arrs), n), np.nan)
    for k, a in enumerate(arrs):
        out[k, :len(a)] = a
    return out


def _load_result(path):
    r = np.load(path, allow_pickle=False)
    out = {k: r[k] for k in r.files}
    if "summary" in out:
        out["params"] = json.loads(str(out["summary"])).get("params", {})
    return out


def load_result(project, name, which="result"):
    path = os.path.join(escan_dir(project, name), f"{which}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"escan '{name}' has no {which}.npz (run it first)")
    return _load_result(path)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _save_fig(fig, path, kind, name, dpi=140):
    """PNG + .json sidecar (harness figure convention)."""
    import matplotlib.pyplot as plt
    from datetime import datetime
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.splitext(path)[0] + ".json", "w") as fh:
        json.dump({"kind": kind, "created": datetime.now().isoformat(timespec="seconds"),
                   "args": {"escan": name}}, fh, indent=2)
    return path


def plot(project, name, out_dir=None):
    """optics, energy scans, angle scans (co-refinement) and full-window paths → list of PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = project.get_escan(name)
    r = load_result(project, name)
    summ = json.loads(str(r["summary"]))
    out_dir = out_dir or project.dir("figures")
    os.makedirs(out_dir, exist_ok=True)
    INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
    cols = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7", "#eda100", "#e87ba4"]
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    E = r["E_grid"]
    di = r["data_idx"]
    kk_layers = [k["layer"] for k in spec.kk]
    fits = seed_fits(project, spec.seed_model or spec.model, spec.criteria)
    seed_layers = project.get_model(spec.seed_model or spec.model).layer_order
    files = []

    # optics
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True, gridspec_kw={"hspace": 0.1})
    for ax, key, lab in zip(axs, ("sld", "isld"), ("sld", "isld")):
        for c, n in zip(cols, kk_layers):
            ax.plot(E[di], r[f"{key}_{n}"][di], color=c, lw=2, label=f"{n} (KK, energy scans)")
            if n in seed_layers and fits:
                col = 1 if key == "sld" else 2
                fe = np.array(sorted(fits))
                ax.plot(fe, [fits[e]["slabs"][seed_layers.index(n), col] for e in fe], "o", ms=6,
                        mfc=c, mec="white", mew=1.2, alpha=0.8, label=f"{n}, {spec.seed_model or spec.model} per-energy fits")
        ax.set_ylabel(f"{lab} (10⁻⁶ Å⁻²)")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].set_xlabel("Energy (eV)")
    p = summ["params"]
    fig.suptitle(f"{project.sample_name} escan '{name}': KK-linked optical constants "
                 f"(isld basis: {summ.get('isld_basis', 'nodes')})\n"
                 f"cost {summ['cost']:.0f}; " + ", ".join(f"{k} {v:.3g}" for k, v in p.items()),
                 x=0.06, ha="left", fontsize=9.5, color=INK)
    files.append(_save_fig(fig, os.path.join(out_dir, f"escan_{name}_optics.png"), "escan-optics", name, dpi=140))

    # full window
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True, gridspec_kw={"hspace": 0.1})
    for ax, key in zip(axs, ("sld", "isld")):
        ax.axvspan(E[0], E[di[0]], color=GRID, lw=0)
        ax.axvspan(E[di[-1]], E[-1], color=GRID, lw=0)
        for c, n in zip(cols, kk_layers):
            ax.plot(E, r[f"{key}_{n}"], color=c, lw=1.8, label=n)
        ax.set_ylabel(f"{key} (10⁻⁶ Å⁻²)")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].set_xlabel("Energy (eV)  (grey: outside the energy scans — smoothness and KK only)")
    axs[1].set_xlim(E[0], E[-1])
    fig.suptitle(f"escan '{name}': paths over the KK window", x=0.06, ha="left")
    files.append(_save_fig(fig, os.path.join(out_dir, f"escan_{name}_window.png"), "escan-window", name, dpi=130))

    # energy scans
    nsc = sum(1 for k in r if k.endswith("_theta") and k.startswith("escan"))
    fig, axs = plt.subplots(nsc + 1, 1, figsize=(9.5, 3.0 * nsc + 2.2), sharex=True,
                            gridspec_kw={"height_ratios": [2.4] * nsc + [1.6], "hspace": 0.12})
    axs = np.atleast_1d(axs)
    for k in range(nsc):
        Es, Rs, Rm = r[f"escan{k}_E"], r[f"escan{k}_R"], r[f"escan{k}_model"]
        th = float(r[f"escan{k}_theta"])
        axs[k].semilogy(Es, Rs, color=MUTED, lw=3, alpha=0.45, label="measured")
        axs[k].semilogy(Es, Rm, color=cols[0], lw=1.4, label="model")
        axs[k].set_ylabel(f"R at θ = {th:g}°")
        axs[-1].plot(Es, Rm / Rs, color=cols[k % len(cols)], lw=1.2, label=f"θ = {th:g}°")
    axs[0].legend(frameon=False, fontsize=8)
    axs[-1].axhspan(1 / 1.05, 1.05, color=GRID, lw=0)
    axs[-1].axhline(1, color=MUTED, lw=0.8)
    axs[-1].set_ylabel("model / measured")
    axs[-1].set_xlabel("Energy (eV)")
    axs[-1].legend(frameon=False, fontsize=8, ncol=4)
    es = {k: v for k, v in summ["params"].items() if k.startswith("escale:")}
    fig.suptitle(f"escan '{name}': energy scans ("
                 + (", ".join(f"θ = {float(r[f'escan{k.split(':')[1]}_theta']):g}° scale {v:.3f}"
                              for k, v in es.items()) + ", others " if es else "")
                 + "scale 1; grey band ±5 %)", x=0.06, ha="left")
    files.append(_save_fig(fig, os.path.join(out_dir, f"escan_{name}_escans.png"), "escan-escans", name, dpi=140))

    # angle scans
    if "Ea" in r:
        Ea = r["Ea"]
        nA = Ea.size
        nc = 4
        nr = int(np.ceil(nA / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(15, 3.3 * nr), sharex=True)
        for k, ax in enumerate(np.atleast_1d(axs).flat):
            if k >= nA:
                ax.axis("off")
                continue
            q, R = r["angle_q"][k], r["angle_R"][k]
            ok = ~np.isnan(q)
            q, R = q[ok], R[ok]
            used = r["fit_mask"][k, :q.size] > 0
            ax.semilogy(q[used], R[used], "o", ms=2.5, color=MUTED, alpha=0.7)
            if (~used).any():
                ax.semilogy(q[~used], R[~used], "o", ms=3, mfc="none", mec=MUTED, mew=0.8)
            ax.semilogy(q, r["Ra"][k, :q.size], color=cols[0], lw=1.3)
            ax.set_title(f"{Ea[k]:g} eV", loc="left", fontsize=9)
        fig.suptitle(f"escan '{name}': angle scans (open circles: θ < {spec.min_theta:g}°, not fitted); "
                     f"scale {summ['params'].get('scale', 1):.3f}", x=0.06, ha="left")
        files.append(_save_fig(fig, os.path.join(out_dir, f"escan_{name}_angles.png"), "escan-angles", name, dpi=120))
    return files
