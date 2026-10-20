"""
multiedge.py
------------
Multi-edge (e.g. C + O) stoichiometry / density searches of one film's fitted
optical constants measured at several absorption edges — generalised from the
Brewer script Nov2025/Oxygen_Harness_Comparison/twoedge_stoich.py.

A MultiEdgeSpec (project.json "stoich_multiedge": {name: spec}) names:
  edges        [{"label", "source", "sigma", "erange"?}, ...]
               source  "<Model>:<Layer>" (this project), "<project dir>::<Model>:<Layer>"
                       (another harness project, e.g. the carbon project), or a
                       table file [E, Re, Im(, σRe, σIm)]
               sigma   "columns" (σ columns of a table), "spread:<AltModel>" (|model − alt|
                       per energy for Model:Layer sources) or a number; always
                       ⊕ sigma_rel·|value| ⊕ sigma_abs (defaults 3 %, 0.05)
               erange  optional [lo, hi] to restrict the edge's energies
  atoms, counts  candidate grid (int | [lo, hi] | [lo, hi, step]) as in StoichSpec
  ties         {"H": ["C", 2.25]} → N_H = round(2.25·N_C) (elements hard to see, e.g. H)
  step_points  off-resonance energies for the step-edge Im terms (nearest edge point
               within 0.06 eV is used; missing energies are skipped)
  scoring      "twoedge" (absolute KK + step Im), "stepedge" (step Im only) or "both"

Scorings (density profiled per formula on a 401-point χ²(ρ) grid):
  twoedge   the measured Im of every edge, converted to f2 per formula unit AT THE
            TRIAL DENSITY, is spliced into the formula's Henke f2 (plain Henke in the
            gaps and outside; Henke anchors at the merge ends, so kkcalc's merge
            scale is 1) → one KK → Re at that density.
            χ² = Σ_edges ((Re_KK − Re)/σRe)² + Σ_step ((Im_bare − Im)/σIm)²
  stepedge  χ² = Σ_step ((ρ·Im_bare(1 g/cc) − Im)/σIm)², ρ analytic (Im ∝ ρ).
            Weak for elements whose edge is not in the data (H) — and an O step from
            a single post-edge point is easily biased by near-edge structure.

Results: <project>/stoich/<name>[_stepedge]/ in the `stoich chi2` format
(chi2.csv, chi2.json, chi2_grid.npz) plus spec.json and edge_<label>.csv, so
stoich.chi2_intervals / load_chi2 and `plot stoich-chi2 --name` work unchanged.
σ from model spread or CMA-ES fits are not posteriors → use Birge-scaled intervals.
"""

import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

RE_A = 2.8179403e-5          # classical electron radius, Å
NA = 6.02214076e23
SIGMA_FLOOR = 1e-3
STEP_TOL = 0.06              # eV
SCORINGS = ("twoedge", "stepedge", "both")


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

@dataclass
class MultiEdgeSpec:
    name: str
    edges: list
    atoms: list
    counts: list
    ties: dict = field(default_factory=dict)
    step_points: list = field(default_factory=list)
    density_range: tuple = (0.8, 3.0)
    scoring: str = "twoedge"
    sigma_rel: float = 0.03
    sigma_abs: float = 0.05
    criteria: str = "best"
    description: str = ""

    def __post_init__(self):
        from .stoich import _norm_counts
        if len(self.edges) < 1:
            raise ValueError("multi-edge search needs at least one edge")
        self.edges = [dict(e) for e in self.edges]
        labels = []
        for e in self.edges:
            if "source" not in e:
                raise ValueError(f"edge {e}: needs a source")
            e.setdefault("label", f"edge{len(labels) + 1}")
            e.setdefault("sigma", "columns")
            if "erange" in e and e["erange"] is not None:
                lo, hi = (float(x) for x in e["erange"])
                if hi <= lo:
                    raise ValueError(f"edge {e['label']}: erange needs lo < hi")
                e["erange"] = [lo, hi]
            s = e["sigma"]
            if not (s == "columns" or (isinstance(s, str) and s.startswith("spread:"))
                    or isinstance(s, (int, float))):
                raise ValueError(f"edge {e['label']}: sigma must be 'columns', 'spread:<Model>' or a number")
            labels.append(e["label"])
        if len(set(labels)) != len(labels):
            raise ValueError(f"edge labels must be unique: {labels}")
        if not self.atoms:
            raise ValueError("atoms=[...] required")
        self.counts = _norm_counts(self.counts)
        if len(self.counts) != len(self.atoms):
            raise ValueError(f"{len(self.atoms)} atoms but {len(self.counts)} count entries")
        self.ties = {str(k): [str(v[0]), float(v[1])] for k, v in (self.ties or {}).items()}
        for k, (ref, _) in self.ties.items():
            if k not in self.atoms or ref not in self.atoms or k == ref:
                raise ValueError(f"tie {k}←{ref}: both must be in atoms and differ")
        self.step_points = [float(x) for x in self.step_points]
        lo, hi = (float(x) for x in self.density_range)
        if not 0 < lo < hi:
            raise ValueError(f"density_range {self.density_range}: need 0 < lo < hi")
        self.density_range = (lo, hi)
        if self.scoring not in SCORINGS:
            raise ValueError(f"scoring must be one of {SCORINGS}")

    def candidates(self):
        from .stoich import _grid
        free = [(a, _grid(c)) for a, c in zip(self.atoms, self.counts) if a not in self.ties]
        out = []
        for combo in itertools.product(*[g for _, g in free]):
            n = dict(zip([a for a, _ in free], combo))
            for k, (ref, f) in self.ties.items():
                n[k] = int(round(f * n[ref]))
            formula = "".join(f"{a}{n[a]}" for a in self.atoms if n[a] > 0)
            if formula:
                out.append(formula)
        return list(dict.fromkeys(out))

    def n_candidates(self):
        return len(self.candidates())

    def to_json(self):
        d = asdict(self)
        d["density_range"] = list(self.density_range)
        return d

    @classmethod
    def from_json(cls, d):
        return cls(**d)

    def describe(self):
        cnt = ", ".join(f"{a}={c}" if isinstance(c, int) else
                        f"{a}∈[{c[0]}..{c[1]}{'' if len(c) == 2 else f' step {c[2]}'}]"
                        for a, c in zip(self.atoms, self.counts) if a not in self.ties)
        ties = "".join(f"; {k}=round({f:g}·{r})" for k, (r, f) in self.ties.items())
        edges = "; ".join(f"{e['label']}: {e['source']}"
                          + (f" [{e['erange'][0]:g}–{e['erange'][1]:g} eV]" if e.get("erange") else "")
                          + f" (σ {e['sigma']})" for e in self.edges)
        return (f"{self.name}: multi-edge {self.scoring} search  |  {cnt}{ties}  |  "
                f"ρ {self.density_range[0]:g}–{self.density_range[1]:g} g/cm³  |  "
                f"{self.n_candidates()} candidates\n  edges: {edges}\n  step points: "
                f"{', '.join(f'{x:g}' for x in self.step_points) or '-'} eV  |  σ ⊕ "
                f"{self.sigma_rel:g}·|v| ⊕ {self.sigma_abs:g}"
                + (f"\n  {self.description}" if self.description else ""))


# ---------------------------------------------------------------------------
# edge tables
# ---------------------------------------------------------------------------

def _split_source(source):
    """'<dir>::<Model>:<Layer>' | '<Model>:<Layer>' | file → (project dir|None, model|None, layer|None, file|None)."""
    if "::" in source:
        d, ml = source.split("::", 1)
        m, l = ml.split(":", 1)
        return d, m, l, None
    if os.path.exists(source):
        return None, None, None, source
    if ":" in source:
        m, l = source.split(":", 1)
        return None, m, l, None
    raise ValueError(f"edge source '{source}': not a file, Model:Layer or dir::Model:Layer")


def _layer_table(project, model, layer, criteria):
    """[E, Re, Im] of a layer's best per-energy fits of `model`."""
    from .analysis import layer_sld_vs_energy
    arr = np.asarray(layer_sld_vs_energy(project.h5_path, project.sample_name, model, layer, criteria),
                     dtype=float)[:, :3]
    return arr[np.argsort(arr[:, 0])]


def edge_table(project, edge, sigma_rel=0.03, sigma_abs=0.05, criteria="best"):
    """[E, Re, Im, σRe, σIm] for one edge entry + provenance string."""
    from .project import FitProject
    from .materials import load_table
    d, model, layer, path = _split_source(edge["source"])
    sig = edge["sigma"]
    if path:
        t = np.asarray(load_table(path), dtype=float)
        t = t[np.argsort(t[:, 0])]
        if sig == "columns":
            if t.shape[1] < 5:
                raise ValueError(f"{path}: sigma='columns' needs σRe, σIm columns 4–5")
            s = t[:, 3:5]
        elif isinstance(sig, (int, float)):
            s = np.full((len(t), 2), float(sig))
        else:
            raise ValueError("sigma 'spread:<Model>' needs a Model:Layer source")
        base, prov = t[:, :3], f"file {path}"
    else:
        p = FitProject.open(d) if d else project
        base = _layer_table(p, model, layer, criteria)
        prov = f"{p.path}: {model} {layer}"
        if isinstance(sig, str) and sig.startswith("spread:"):
            alt = _layer_table(p, sig.split(":", 1)[1], layer, criteria)
            s = np.column_stack([np.abs(base[:, 1] - np.interp(base[:, 0], alt[:, 0], alt[:, 1])),
                                 np.abs(base[:, 2] - np.interp(base[:, 0], alt[:, 0], alt[:, 2]))])
            prov += f" (σ spread vs {sig.split(':', 1)[1]})"
        elif isinstance(sig, (int, float)):
            s = np.full((len(base), 2), float(sig))
        else:
            raise ValueError(f"edge {edge['label']}: sigma='columns' needs a table file source")
    s = np.sqrt(s ** 2 + (sigma_rel * np.abs(base[:, 1:3])) ** 2 + sigma_abs ** 2)
    t = np.column_stack([base, np.maximum(s, SIGMA_FLOOR)])
    if edge.get("erange"):
        lo, hi = edge["erange"]
        t = t[(t[:, 0] >= lo - 1e-9) & (t[:, 0] <= hi + 1e-9)]
    if len(t) < 3:
        raise ValueError(f"edge {edge['label']}: fewer than 3 energies")
    return t, prov


def step_points(edges, energies):
    """[E, Im, σIm] at the requested off-resonance energies (nearest edge point)."""
    rows = []
    for e in energies:
        best = None
        for t in edges:
            k = int(np.argmin(np.abs(t[:, 0] - e)))
            if abs(t[k, 0] - e) <= STEP_TOL and (best is None or abs(t[k, 0] - e) < abs(best[0] - e)):
                best = t[k]
        if best is not None:
            rows.append([best[0], best[2], best[4]])
    return np.array(rows).reshape(-1, 3)


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------

def _f2(formula, E):
    from kkcalc import data as kd
    Ea, A = kd.calculate_asf(kd.ParseChemicalFormula(formula))
    E = np.atleast_1d(np.asarray(E, float))
    return np.array([kd.coeffs_to_ASF(e, A[(Ea > e).argmax() - 1]) for e in E])


def _mass(formula):
    from kkcalc import data as kd
    return kd.calculate_FormulaMass(kd.ParseChemicalFormula(formula))


def bare_imag(formula, rho, E):
    """Bare-atom imaginary SLD (1e-6 Å⁻²) of a formula at density rho (g/cc)."""
    n_f = rho / _mass(formula) * NA * 1e-24
    return RE_A * n_f * _f2(formula, E) * 1e6


def abs_kk(formula, rho, edges):
    """Absolute multi-edge KK at density rho: (E, Re, Im) SLD on kkcalc's merged grid.
    Measured Im of each edge (→ f2 at rho) spliced into the formula's Henke f2."""
    from kkcalc import data as kd, kk
    st = kd.ParseChemicalFormula(formula)
    M = kd.calculate_FormulaMass(st)
    rel = kk.calc_relativistic_correction(st)
    FE, IM = kd.calculate_asf(st)
    nf = rho / M * NA * 1e-24
    henke = lambda E: np.array([kd.coeffs_to_ASF(e, IM[(FE > e).argmax() - 1]) for e in np.atleast_1d(E)])
    edges = sorted(edges, key=lambda t: t[0, 0])
    lo, hi = float(edges[0][0, 0]) - 2.0, float(edges[-1][-1, 0]) + 1.0
    Es, f2 = [np.array([lo])], [henke(lo)]
    for k, t in enumerate(edges):
        Es.append(t[:, 0])
        f2.append(t[:, 2] * 1e-6 / (RE_A * nf))
        if k + 1 < len(edges):
            mid = np.arange(np.floor(t[-1, 0]) + 2.0, edges[k + 1][0, 0] - 1.0, 2.0)
            if mid.size:
                Es.append(mid)
                f2.append(henke(mid))
    Es.append(np.array([hi]))
    f2.append(henke(hi))
    E = np.concatenate(Es)
    FE2, IM2 = kd.merge_spectra(np.column_stack((E, np.concatenate(f2))), FE, IM, merge_points=[lo, hi])
    f1 = kk.KK_PP(FE2, FE2, IM2, rel)
    f2v = kd.coeffs_to_ASF(FE2, np.vstack((IM2, IM2[-1])))
    return FE2, RE_A * nf * f1 * 1e6, RE_A * nf * f2v * 1e6


def _grid_with_best(rho_grid, coarse, cg, rho_best, chi_best):
    x, y = np.r_[coarse, rho_best], np.r_[cg, chi_best]
    o = np.argsort(x)
    return np.interp(rho_grid, x[o], y[o])


def _twoedge_worker(args):
    from scipy.optimize import minimize_scalar
    formula, edges, pts, drange, rho_grid = args
    rec = {"formula": formula, "density_w": np.nan, "chi2": np.inf,
           "grid": np.full(len(rho_grid), np.inf)}
    d = np.vstack(edges)
    s_re = np.maximum(d[:, 3], SIGMA_FLOOR)
    try:
        unit = bare_imag(formula, 1.0, pts[:, 0]) if len(pts) else np.zeros(0)
    except Exception:
        return rec

    def chi2(rho):
        try:
            E, re_kk, _ = abs_kk(formula, rho, edges)
        except Exception:
            return 1e300
        r = np.interp(d[:, 0], E, re_kk)
        return float(np.sum(((r - d[:, 1]) / s_re) ** 2)
                     + np.sum(((rho * unit - pts[:, 1]) / np.maximum(pts[:, 2], SIGMA_FLOOR)) ** 2))

    res = minimize_scalar(chi2, bounds=drange, method="bounded", options={"xatol": 1e-4})
    coarse = np.linspace(drange[0], drange[1], 41)
    cg = np.array([chi2(r) for r in coarse])
    rec.update(density_w=float(res.x), chi2=float(res.fun),
               grid=_grid_with_best(rho_grid, coarse, cg, float(res.x), float(res.fun)))
    return rec


def _step_worker(args):
    formula, pts, drange, rho_grid = args
    E, im_, s_im = pts[:, 0], pts[:, 1], np.maximum(pts[:, 2], SIGMA_FLOOR)
    try:
        unit = bare_imag(formula, 1.0, E)
    except Exception:
        return {"formula": formula, "density_w": np.nan, "chi2": np.inf,
                "grid": np.full(len(rho_grid), np.inf)}
    w = 1.0 / s_im ** 2
    rho = float(np.clip(np.sum(w * unit * im_) / np.sum(w * unit ** 2), *drange))
    chi2 = lambda r: float(np.sum(((r * unit - im_) / s_im) ** 2))
    return {"formula": formula, "density_w": rho, "chi2": chi2(rho),
            "grid": np.array([chi2(r) for r in rho_grid])}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _write(d, recs, rho_grid, n_pts, meta):
    import pandas as pd
    from .stoich import _composition
    os.makedirs(d, exist_ok=True)
    grids = {r["formula"]: r.pop("grid") for r in recs}
    df = pd.DataFrame(recs)
    df = df[np.isfinite(df.chi2)].sort_values("chi2").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no candidate formula could be scored")
    _, els = _composition(df.head(1).copy())
    nu = max(1, n_pts - len(els))
    best = float(df.chi2.iloc[0])
    df["dchi2"] = df.chi2 - best
    info = {"chi2_best": best, "n_points": int(n_pts), "nu": int(nu), "birge_s2": best / nu,
            "best_formula": df.formula.iloc[0], "best_density": float(df.density_w.iloc[0]), **meta}
    df.to_csv(os.path.join(d, "chi2.csv"), index=False)
    np.savez_compressed(os.path.join(d, "chi2_grid.npz"), rho=rho_grid,
                        formulas=np.array(list(df.formula), dtype=object),
                        chi2=np.array([grids[f] for f in df.formula]))
    with open(os.path.join(d, "chi2.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=float)
    return df, info


def inputs(project, spec):
    """(edge tables, provenance list, step points)."""
    tabs, provs = [], []
    for e in spec.edges:
        t, prov = edge_table(project, e, spec.sigma_rel, spec.sigma_abs, spec.criteria)
        tabs.append(t)
        provs.append(prov)
    return tabs, provs, step_points(tabs, spec.step_points)


def run(project, name, workers=None, verbose=True, log=print):
    """Run a registered multi-edge search → {scoring: (df, info, out_dir)}."""
    from concurrent.futures import ProcessPoolExecutor
    import kk_stoichiometry_fit as K
    from .stoich import reserve_workers, release_workers
    spec = project.get_multiedge(name)
    tabs, provs, pts = inputs(project, spec)
    cands = spec.candidates()
    drange = spec.density_range
    rho_grid = np.linspace(*drange, 401)
    base_dir = project.stoich_dir(name)
    os.makedirs(base_dir, exist_ok=True)
    hdr = "Energy_eV,Real_SLD,Imag_SLD,Real_std,Imag_std"
    for e, t in zip(spec.edges, tabs):
        np.savetxt(os.path.join(base_dir, f"edge_{e['label']}.csv"), t, delimiter=",", header=hdr, comments="")
    meta0 = {"kind": "multiedge", "spec": spec.to_json(), "edge_sources": provs,
             "step_points_used": pts[:, 0].tolist(), "density_range": list(drange),
             "energies": {e["label"]: t[:, 0].tolist() for e, t in zip(spec.edges, tabs)}}
    with open(os.path.join(base_dir, "spec.json"), "w") as fh:
        json.dump(meta0, fh, indent=2, default=float)
    out = {}
    if spec.scoring in ("stepedge", "both"):
        if len(pts) < 2:
            raise ValueError("step-edge scoring needs ≥ 2 step points present in the edges")
        t0 = time.time()
        recs = [_step_worker((f, pts, drange, rho_grid)) for f in cands]
        d = base_dir if spec.scoring == "stepedge" else project.stoich_dir(name + "_stepedge")
        df, info = _write(d, recs, rho_grid, len(pts),
                          dict(meta0, method="step edge (bare-atom Im)", seconds=round(time.time() - t0, 1)))
        out["stepedge"] = (df, info, d)
        if verbose:
            log(f"[multiedge] {name} step edge: best {info['best_formula']} ρ {info['best_density']:.3f} "
                f"χ²/ν {info['chi2_best'] / info['nu']:.2f} ({len(cands)} formulas)")
    if spec.scoring in ("twoedge", "both"):
        args = [(f, tabs, pts, drange, rho_grid) for f in cands]
        n = reserve_workers(workers)
        t0 = time.time()
        try:
            if verbose:
                log(f"[multiedge] {name}: {len(cands)} formulas, "
                    f"{' + '.join(str(len(t)) for t in tabs)} energies + {len(pts)} step points, {n} workers")
            with ProcessPoolExecutor(max_workers=n, mp_context=K._SPAWN_CTX,
                                     initializer=K._spawn_worker_init, initargs=(sys.path,)) as pool:
                recs = list(pool.map(_twoedge_worker, args, chunksize=2))
        finally:
            release_workers()
        n_pts = sum(len(t) for t in tabs) + len(pts)
        df, info = _write(base_dir, recs, rho_grid, n_pts,
                          dict(meta0, method="multi-edge absolute KK + step-edge Im", workers=n,
                               seconds=round(time.time() - t0, 1)))
        out["twoedge"] = (df, info, base_dir)
        if verbose:
            log(f"[multiedge] {name} two-edge: best {info['best_formula']} ρ {info['best_density']:.3f} "
                f"χ²/ν {info['chi2_best'] / info['nu']:.2f} ({time.time() - t0:.0f} s)")
    return out


def summary(project, name, level=1.0, scaled=True):
    """Best formula + profile intervals (Birge-scaled by default) of a finished run;
    flags quantities whose accepted range touches the candidate grid edge."""
    from .stoich import load_chi2, chi2_intervals
    df, info, grid = load_chi2(project.stoich_dir(name))
    iv, dfc, els = chi2_intervals(df, info, levels=(level,), scaled=scaled, grid=grid)
    spec = MultiEdgeSpec.from_json(info["spec"]) if "spec" in info else None
    edge = []
    if spec is not None:
        from .stoich import _grid
        for a, c in zip(spec.atoms, spec.counts):
            if a in spec.ties or isinstance(c, int):
                continue
            g = _grid(c)
            col = f"N_{a}"
            sub = dfc[dfc.dchi2 / (info["birge_s2"] if scaled else 1.0) <= level]
            if col in sub and (sub[col].max() >= max(g) or sub[col].min() <= min(g)):
                edge.append(a)
    return {"best": dfc.formula.iloc[0], "chi2_nu": info["chi2_best"] / info["nu"], "nu": info["nu"],
            "intervals": iv[level], "elements": els, "at_grid_edge": edge}
