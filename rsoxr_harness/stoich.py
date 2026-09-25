"""
stoich.py
---------
Kramers-Kronig stoichiometry / density fits of an SLD(E) table, wrapping
kk_stoichiometry_fit (per-atom and fragment searches, density optimisation,
cost surfaces, KK overlays).

A StoichSpec is a named, JSON-serialisable search.  Its SLD source is
    "<material>"              a registered project material
    "<Model>:<Layer>"         a layer's fitted SLD from the project h5
    "/path/to/table.csv"      any [Energy, Real, Imag] table
Results live in <project>/stoich/<name>/:
    spec.json  sld_input.csv  results.csv  best.json  best_kk_sld.csv

Cost (forward KK):  imag_fit → β → KK(kkcalc) → δ → real_kk ;
    RMSE = sqrt(0.5·MSE(real_kk, real_fit) + 0.5·MSE(imag_kk, imag_fit))
Formulas that differ only by an overall factor (C8H8 vs C16H16) have the same
mass fractions and therefore the same cost — fix one count to set the scale.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

MODES = ("atoms", "fragments")


def _grid(cr):
    """int | [lo, hi] | [lo, hi, step] → list of ints (kk_stoichiometry_fit rule)."""
    if isinstance(cr, (int, np.integer)):
        return [int(cr)]
    cr = list(cr)
    step = int(cr[2]) if len(cr) == 3 else 1
    return list(range(int(cr[0]), int(cr[1]) + 1, step))


def _norm_counts(counts):
    out = []
    for c in counts:
        if isinstance(c, (int, np.integer)):
            out.append(int(c))
        else:
            c = [int(x) for x in c]
            if len(c) not in (2, 3) or c[1] < c[0]:
                raise ValueError(f"count range {c}: need [lo, hi] or [lo, hi, step]")
            out.append(c)
    return out


@dataclass
class StoichSpec:
    """
    mode="atoms":      atoms=['Si','O','C','H'], counts=[2, [1, 4], [1, 6], [2, 12, 2]]
    mode="fragments":  fragments=['SiO2', 'CH3'], counts=[[1, 3], [2, 8]]
    counts entries: int (fixed) | [lo, hi] | [lo, hi, step].
    merge_points: kkcalc splice energies (default: data min/max).
    energy_mask:  residual window(s): (lo, hi) or [(lo1, hi1), (lo2, hi2), ...]
                  (union; e.g. pre/post edge only).  The KK transform always
                  uses the full imaginary spectrum inside merge_points.
    refine_density: after the search, re-optimise the best formula's density
                    within ±refine_density g/cm³ (None → skip).
    """
    name: str
    source: str
    mode: str = "atoms"
    atoms: list | None = None
    fragments: list | None = None
    counts: list = field(default_factory=list)
    density_range: tuple = (0.8, 2.5)
    merge_points: list | None = None
    energy_mask: list | None = None
    refine_density: float | None = 0.1
    top_n: int = 10
    n_workers: int | None = None
    criteria: str = "best"
    description: str = ""

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        names = self.atoms if self.mode == "atoms" else self.fragments
        if not names:
            raise ValueError(f"mode '{self.mode}' needs "
                             f"{'atoms' if self.mode == 'atoms' else 'fragments'}=[...]")
        self.counts = _norm_counts(self.counts)
        if len(self.counts) != len(names):
            raise ValueError(f"{len(names)} {self.mode} but {len(self.counts)} count entries")
        lo, hi = (float(x) for x in self.density_range)
        if not 0 < lo < hi:
            raise ValueError(f"density_range {self.density_range}: need 0 < lo < hi")
        self.density_range = (lo, hi)
        if self.merge_points is not None:
            self.merge_points = [float(x) for x in self.merge_points]
        if self.energy_mask is not None:
            from kk_stoichiometry_fit import _mask_windows
            wins = _mask_windows(self.energy_mask)
            if any(hi <= lo for lo, hi in wins):
                raise ValueError(f"energy_mask {self.energy_mask}: each window needs lo < hi")
            self.energy_mask = list(wins[0]) if len(wins) == 1 else [list(w) for w in wins]

    @property
    def names(self):
        return self.atoms if self.mode == "atoms" else self.fragments

    def n_candidates(self):
        if self.mode == "fragments":
            from kk_stoichiometry_fit import _build_fragment_grid
            return len(_build_fragment_grid(self.fragments, self._kk_counts()))
        n = 1
        for c in self.counts:
            n *= len(_grid(c))
        return n

    def _kk_counts(self):
        return [c if isinstance(c, int) else tuple(c) for c in self.counts]

    def to_json(self):
        d = asdict(self)
        d["density_range"] = list(self.density_range)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_json(cls, d):
        return cls(**d)

    def describe(self):
        ranges = ", ".join(
            f"{n}={c}" if isinstance(c, int) else
            f"{n}∈[{c[0]}..{c[1]}{'' if len(c) == 2 else f' step {c[2]}'}]"
            for n, c in zip(self.names, self.counts))
        return (f"{self.name}: {self.mode} search on {self.source}  |  {ranges}  |  "
                f"ρ {self.density_range[0]:g}–{self.density_range[1]:g} g/cm³  |  "
                f"{self.n_candidates()} candidates"
                + (f"  |  merge {self.merge_points}" if self.merge_points else "")
                + (f"  |  residual window {_mask_str(self.energy_mask)} eV"
                   if self.energy_mask else ""))


def _mask_str(em):
    from kk_stoichiometry_fit import _mask_label
    return _mask_label(em)


# ---------------------------------------------------------------------------
# SLD sources
# ---------------------------------------------------------------------------

def resolve_source(project, source, criteria="best"):
    """→ (sld_3col sorted by energy, provenance str)."""
    if os.path.exists(source):
        from .materials import load_table
        arr = load_table(source)
        prov = f"file {source}"
    elif ":" in source:
        from .analysis import layer_sld_vs_energy
        if project is None:
            raise ValueError("a Model:Layer source needs a project")
        model, layer = source.split(":", 1)
        arr = layer_sld_vs_energy(project.h5_path, project.sample_name, model, layer,
                                  criteria)
        prov = f"{model} {layer} fitted SLD ({criteria} run, {len(arr)} energies)"
    else:
        if project is None:
            raise ValueError(f"'{source}' is not a file; material sources need a project")
        arr = np.array(project.material_array(source), dtype=float)
        prov = f"material {source} ({project.cfg['materials'][source].get('path', 'formula')})"
    arr = np.asarray(arr, dtype=float)[:, :3]
    arr = arr[np.argsort(arr[:, 0])]
    return arr, prov


CPU_FRACTION = 0.5          # never use more than half the machine's CPUs


def cpu_cap():
    """Maximum worker processes for stoichiometry searches (50% of CPUs)."""
    return max(1, int((os.cpu_count() or 2) * CPU_FRACTION))


def resolve_workers(requested, in_use=0):
    """
    Workers for one search: `requested` (None → all of the cap), clipped so
    that this search plus `in_use` workers of other running searches stays
    within cpu_cap().
    """
    room = max(1, cpu_cap() - int(in_use))
    return room if requested is None else max(1, min(int(requested), room))


_REGISTRY = os.path.expanduser("~/.cache/rsoxr_harness/cpu_reservations.json")


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _with_registry(fn):
    import fcntl
    os.makedirs(os.path.dirname(_REGISTRY), exist_ok=True)
    with open(_REGISTRY + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(_REGISTRY) as fh:
                reg = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            reg = {}
        reg = {p: w for p, w in reg.items() if _alive(p)}       # drop dead processes
        out = fn(reg)
        with open(_REGISTRY, "w") as fh:
            json.dump(reg, fh)
        return out


def cpus_in_use():
    """Workers reserved by running stoich searches on this machine (any project)."""
    return _with_registry(lambda reg: sum(reg.values()))


def reserve_workers(requested):
    """Atomically reserve workers within the machine-wide 50% cap → n."""
    def go(reg):
        n = resolve_workers(requested, sum(reg.values()))
        reg[str(os.getpid())] = n
        return n
    return _with_registry(go)


def release_workers():
    _with_registry(lambda reg: reg.pop(str(os.getpid()), None))


def _defaults(spec, sld):
    mp = spec.merge_points or [float(sld[:, 0].min()), float(sld[:, 0].max())]
    em = spec.energy_mask
    if em and len(em) == 2 and not isinstance(em[0], (list, tuple)):
        em = tuple(em)
    return mp, em


def _restrict(sld, mp):
    """Keep rows inside the merge window (kkcalc splices the data there)."""
    keep = (sld[:, 0] >= mp[0] - 1e-9) & (sld[:, 0] <= mp[1] + 1e-9)
    return sld[keep]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_search(spec, sld, out_dir, provenance="", verbose=True):
    """Search + optional density refinement; writes the result files."""
    import kk_stoichiometry_fit as K
    os.makedirs(out_dir, exist_ok=True)
    mp, mask = _defaults(spec, sld)
    sld = _restrict(sld, mp)
    np.savetxt(os.path.join(out_dir, "sld_input.csv"), sld, delimiter=",",
               header="Energy_eV,Real_SLD,Imag_SLD", comments="")
    with open(os.path.join(out_dir, "spec.json"), "w") as fh:
        json.dump({**spec.to_json(), "provenance": provenance,
                   "merge_points_used": mp, "energy_mask_used": mask}, fh, indent=2)

    t0 = time.time()
    # never hand None to kk_stoichiometry_fit (it would use every CPU)
    workers = resolve_workers(spec.n_workers)
    common = dict(density_range=spec.density_range, merge_points=mp, energy_mask=mask,
                  optimise_density=True, n_workers=workers, verbose=verbose)
    if spec.mode == "atoms":
        df, best = K.fit_stoichiometry_density(sld, atoms=spec.atoms,
                                               count_ranges=spec._kk_counts(), **common)
    else:
        df, best = K.fit_stoichiometry_density_fragments(sld, fragments=spec.fragments,
                                                         count_ranges=spec._kk_counts(),
                                                         **common)
    search_sec = time.time() - t0
    df = df.dropna(subset=["mse"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("every candidate failed (KK transform errors) — check "
                           "merge_points against the data range")
    best = {k: (float(v) if k != "formula" else v) for k, v in best.items()}
    best["counts"] = {n: int(df.iloc[0][f"n_{n}"]) for n in spec.names}

    refined = None
    if spec.refine_density:
        lo = max(spec.density_range[0], best["density"] - spec.refine_density)
        hi = min(spec.density_range[1], best["density"] + spec.refine_density)
        refined = _refine(sld, best["formula"], (lo, hi), mp, mask)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    final = refined or best
    kk = kk_sld(sld, final["formula"], final["density"], mp)
    np.savetxt(os.path.join(out_dir, "best_kk_sld.csv"), kk, delimiter=",",
               header="Energy_eV,Real_SLD,Imag_SLD", comments="")
    summary = {"name": spec.name, "mode": spec.mode, "provenance": provenance,
               "n_candidates": int(len(df)), "search_sec": round(search_sec, 1),
               "best": best, "refined": refined,
               "top": df.head(spec.top_n)[["formula", "density", "rmse"]]
               .to_dict(orient="records"),
               "merge_points": mp, "energy_mask": mask}
    with open(os.path.join(out_dir, "best.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary


def _refine(sld, formula, drange, mp, mask):
    """Bounded density optimisation for one fixed formula (one KK transform)."""
    import kk_stoichiometry_fit as K
    (_, e_fit, re_fit, im_fit, e_beta, _) = K._prepare_fit_data(sld, mask)
    asf, mass = K._asf_output_for_formula(e_beta, formula, mp, {})
    if asf is None:
        return None
    from scipy.optimize import minimize_scalar
    f = lambda rho: K._fwd_combined_mse(asf, mass, rho, e_fit, re_fit, im_fit)
    r = minimize_scalar(f, bounds=drange, method="bounded",
                        options={"xatol": 1e-5})
    return {"formula": formula, "density": float(r.x), "mse": float(r.fun),
            "rmse": float(np.sqrt(r.fun)), "density_window": [float(x) for x in drange]}


def kk_sld(sld, formula, density, merge_points):
    """KK-consistent [E, Re, Im] SLD for a formula/density over the data range."""
    from NEXAFS import EnergytoWavelength, imag_SLD_to_real_SLD
    s4 = np.column_stack((sld, EnergytoWavelength(sld[:, 0])))
    _, out = imag_SLD_to_real_SLD(s4, chemical_formula=formula, density=density,
                                  merge_points=merge_points)
    m = (out[:, 0] >= sld[:, 0].min()) & (out[:, 0] <= sld[:, 0].max())
    return np.asarray(out[m][:, :3], dtype=float)


def rmse_for(sld, formula, density, merge_points, energy_mask=None):
    """RMSE of one formula/density (no fitting) — same cost as the search."""
    import kk_stoichiometry_fit as K
    (_, e_fit, re_fit, im_fit, e_beta, _) = K._prepare_fit_data(sld, energy_mask or None)
    asf, mass = K._asf_output_for_formula(e_beta, formula, merge_points, {})
    if asf is None:
        return float("nan")
    return float(np.sqrt(K._fwd_combined_mse(asf, mass, density, e_fit, re_fit, im_fit)))


def load_result(out_dir):
    import pandas as pd
    with open(os.path.join(out_dir, "best.json")) as fh:
        summary = json.load(fh)
    with open(os.path.join(out_dir, "spec.json")) as fh:
        sp = json.load(fh)
    df = pd.read_csv(os.path.join(out_dir, "results.csv"))
    sld = np.loadtxt(os.path.join(out_dir, "sld_input.csv"), delimiter=",", skiprows=1)
    return summary, sp, df, sld


def final_best(summary):
    return summary.get("refined") or summary["best"]


def sld_from_jaxns_cache(cache_dir, layer, out_csv, stat="median", pattern="*jaxns_*.npz"):
    """
    Build an SLD(E) target table from per-energy JAXNS caches (notebook
    `save_jaxns` .npz files or harness uncertainty caches): the posterior
    `stat` (median | mean) of '<layer> - sld' / '<layer> - isld' at each
    energy, plus the posterior std.  Duplicate caches of one energy
    (e.g. *_270.npz and *_270.0.npz) must agree.  Writes
    Energy_eV,Real_SLD,Imag_SLD,Real_std,Imag_std → out_csv; returns the array.
    """
    import glob
    import re
    rows = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, pattern))):
        m = re.search(r"_([0-9]+(?:\.[0-9]+)?)(?:eV)?\.npz$", f)
        if not m:
            continue
        e = round(float(m.group(1)), 6)
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["param_names"]]
        if f"{layer} - sld" not in names:
            raise KeyError(f"{f}: no '{layer} - sld' (params: {names})")
        v = z[f"posterior_{stat}"]
        s = z["posterior_std"]
        i, j = names.index(f"{layer} - sld"), names.index(f"{layer} - isld")
        row = (e, float(v[i]), float(v[j]), float(s[i]), float(s[j]))
        if e in rows and not np.allclose(rows[e], row, rtol=1e-9):
            raise ValueError(f"duplicate caches for {e:g} eV disagree")
        rows[e] = row
    if not rows:
        raise FileNotFoundError(f"no JAXNS caches matching {pattern} in {cache_dir}")
    arr = np.array([rows[e] for e in sorted(rows)])
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    np.savetxt(out_csv, arr, delimiter=",",
               header="Energy_eV,Real_SLD,Imag_SLD,Real_std,Imag_std", comments="")
    return arr


# ---------------------------------------------------------------------------
# σ-weighted χ² and Δχ² profile intervals
# ---------------------------------------------------------------------------
#
# χ²(formula, ρ) = Σ_E [(Re_KK − Re)/σ_Re]² + [(Im_KK − Im)/σ_Im]²  over the
# residual window; ρ is profiled (minimised) per formula.  A quantity's
# profile interval at level Δ is its range over candidates with
# χ²_min(formula) − χ²_best ≤ Δ  (Δ = 1 → 68 %, Δ = 4 → 95 %, one parameter).
# "Birge-scaled" intervals divide Δχ² by s² = χ²_best/ν (ν = 2·N_E − n_atoms),
# i.e. they trust only the relative σ between energies — use these when σ are
# known to be mis-scaled (e.g. JAXNS run with normalize=True) or χ²_best/ν ≫ 1.

SIGMA_FLOOR = 1e-3


def load_sigma(source, energies):
    """σ_Re, σ_Im at `energies` from columns 4–5 of the source table."""
    from .materials import load_table
    if not os.path.exists(source):
        raise ValueError(f"σ-weighted χ² needs a table source with σ columns; got '{source}'")
    t = load_table(source)
    if t.shape[1] < 5:
        raise ValueError(f"{source} has no σ columns (need E, Re, Im, σ_Re, σ_Im)")
    idx = [int(np.argmin(np.abs(t[:, 0] - e))) for e in energies]
    if any(abs(t[i, 0] - e) > 1e-6 for i, e in zip(idx, energies)):
        raise ValueError("σ table energies do not match the fitted SLD energies")
    s = t[idx][:, 3:5]
    return np.maximum(s[:, 0], SIGMA_FLOOR), np.maximum(s[:, 1], SIGMA_FLOOR)


def _chi2_worker(args):
    """(formula, energy_beta, merge_points, ρ range, E, Re, Im, σRe, σIm) → record."""
    import kk_stoichiometry_fit as K
    from scipy.optimize import minimize_scalar
    formula, e_beta, mp, drange, e, re_, im_, s_re, s_im, rho_grid = args
    asf, mass = K._asf_output_for_formula(e_beta, formula, mp, {})
    rec = {"formula": formula, "density_w": np.nan, "chi2": np.inf,
           "grid": np.full(len(rho_grid), np.inf)}
    if asf is None:
        return rec

    def chi2(rho):
        r, i = K._asf_to_both_sld(asf, mass, rho, e)
        if r is None:
            return 1e300
        return float(np.sum(((r - re_) / s_re) ** 2 + ((i - im_) / s_im) ** 2))

    res = minimize_scalar(chi2, bounds=drange, method="bounded", options={"xatol": 1e-5})
    rec.update(density_w=float(res.x), chi2=float(res.fun),
               grid=np.array([chi2(r) for r in rho_grid]))
    return rec


def chi2_scan(project, name, workers=None, verbose=True):
    """
    Re-score every candidate of a finished search with the σ-weighted χ²
    (density re-optimised per formula).  Writes chi2.csv / chi2.json in the
    run directory and returns (df, info).  Uses the 50 % CPU budget.
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    import kk_stoichiometry_fit as K
    d = project.stoich_dir(name)
    summary, sp, df, sld = load_result(d)
    mask = summary["energy_mask"]
    (_, e_fit, re_fit, im_fit, e_beta, _) = K._prepare_fit_data(sld, mask)
    s_re, s_im = load_sigma(sp["source"], e_fit)
    formulas = list(dict.fromkeys(df.formula))
    rho_grid = np.linspace(*sp["density_range"], 401)       # χ²(ρ) per formula
    args = [(f, e_beta, summary["merge_points"], tuple(sp["density_range"]),
             e_fit, re_fit, im_fit, s_re, s_im, rho_grid) for f in formulas]
    n = reserve_workers(workers)
    t0 = time.time()
    try:
        if verbose:
            print(f"σ-weighted χ²: {len(formulas)} formulas, {len(e_fit)} energies × 2, "
                  f"{n} workers", flush=True)
        with ProcessPoolExecutor(max_workers=n, mp_context=K._SPAWN_CTX,
                                 initializer=K._spawn_worker_init,
                                 initargs=(__import__("sys").path,)) as pool:
            recs = list(pool.map(_chi2_worker, args, chunksize=4))
    finally:
        release_workers()
    grids = {r["formula"]: r.pop("grid") for r in recs}
    out = pd.DataFrame(recs).merge(df, on="formula", how="left")
    out = out.sort_values("chi2").reset_index(drop=True)
    n_atoms = len([c for c in df.columns if c.startswith("n_")])
    n_pts = 2 * len(e_fit)
    nu = max(1, n_pts - n_atoms)
    best = float(out.chi2.iloc[0])
    out["dchi2"] = out.chi2 - best
    info = {"chi2_best": best, "n_points": n_pts, "nu": nu, "birge_s2": best / nu,
            "energies": [float(x) for x in e_fit],
            "sigma_median": [float(np.median(s_re)), float(np.median(s_im))],
            "best_formula": out.formula.iloc[0], "best_density": float(out.density_w.iloc[0]),
            "seconds": round(time.time() - t0, 1)}
    out.to_csv(os.path.join(d, "chi2.csv"), index=False)
    np.savez_compressed(os.path.join(d, "chi2_grid.npz"), rho=rho_grid,
                        formulas=np.array(list(out.formula), dtype=object),
                        chi2=np.array([grids[f] for f in out.formula]))
    with open(os.path.join(d, "chi2.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    return out, info


def _composition(df):
    """Add element counts, mass fractions and ratios (vs the first element)."""
    from kk_stoichiometry_fit import _parse_formula
    from periodictable import elements
    comp = [_parse_formula(f) for f in df.formula]
    els = sorted({e for c in comp for e in c}, key=lambda e: ("CONHSi".find(e) % 99, e))
    for e in els:
        df[f"N_{e}"] = [c.get(e, 0) for c in comp]
    mass = sum(df[f"N_{e}"] * elements.symbol(e).mass for e in els)
    for e in els:
        df[f"w_{e}"] = df[f"N_{e}"] * elements.symbol(e).mass / mass
    for e in els[1:]:
        df[f"{e}:{els[0]}"] = df[f"N_{e}"] / df[f"N_{els[0]}"]
    if len(els) >= 2:
        df[f"{els[0]}:{els[1]}"] = df[f"N_{els[0]}"] / df[f"N_{els[1]}"].replace(0, np.nan)
    return df, els


def _density_interval(grid, thresh):
    """[ρ_lo, ρ_hi] where any formula's χ²(ρ) ≤ thresh (linear interpolation)."""
    rho, c2 = grid["rho"], grid["chi2"]
    env = np.min(c2, axis=0)                       # best formula at each ρ
    ok = env <= thresh
    if not ok.any():
        return (np.nan, np.nan)
    i0, i1 = np.argmax(ok), len(ok) - 1 - np.argmax(ok[::-1])

    def cross(a, b):                               # env crosses thresh between a and b
        if env[a] == env[b]:
            return rho[b]
        return rho[a] + (thresh - env[a]) * (rho[b] - rho[a]) / (env[b] - env[a])
    lo = rho[i0] if i0 == 0 else cross(i0 - 1, i0)
    hi = rho[i1] if i1 == len(rho) - 1 else cross(i1 + 1, i1)
    return (float(lo), float(hi))


def chi2_intervals(df, info, levels=(1.0, 4.0), scaled=False, grid=None):
    """
    Profile intervals {level: {quantity: (min, best, max, n_formulas)}}.
    Composition quantities range over formulas with Δχ² ≤ level; density
    uses the χ²(ρ) grid (chi2_grid.npz) so it is a continuous interval —
    'density' key; 'density_w' is each accepted formula's own optimum.
    scaled=True divides Δχ² by the Birge factor s² = χ²_best/ν.
    A range that reaches the search grid edge is only a lower bound.
    """
    df, els = _composition(df.copy())
    s2 = info["birge_s2"] if scaled else 1.0
    q = ["density_w"] + [f"w_{e}" for e in els] + [c for c in df.columns if ":" in c]
    best = df.iloc[0]
    out = {}
    for lv in levels:
        sub = df[df.dchi2 / s2 <= lv]
        out[lv] = {k: (float(sub[k].min()), float(best[k]), float(sub[k].max()), len(sub))
                   for k in q}
        if grid is not None:
            lo, hi = _density_interval(grid, info["chi2_best"] + lv * s2)
            out[lv] = {"density": (lo, float(best["density_w"]), hi, len(sub)), **out[lv]}
    return out, df, els


def load_chi2(d):
    """(df, info, grid) from a run directory after chi2_scan."""
    import pandas as pd
    df = pd.read_csv(os.path.join(d, "chi2.csv"))
    with open(os.path.join(d, "chi2.json")) as fh:
        info = json.load(fh)
    g = np.load(os.path.join(d, "chi2_grid.npz"), allow_pickle=True)
    return df, info, {"rho": g["rho"], "chi2": g["chi2"], "formulas": list(g["formulas"])}
