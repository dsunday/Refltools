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
    energy_mask:  (lo, hi) residual window (default: all data).
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
            self.energy_mask = [float(x) for x in self.energy_mask]

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
                + (f"  |  mask {self.energy_mask}" if self.energy_mask else ""))


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


def _defaults(spec, sld):
    mp = spec.merge_points or [float(sld[:, 0].min()), float(sld[:, 0].max())]
    return mp, (tuple(spec.energy_mask) if spec.energy_mask else None)


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
    common = dict(density_range=spec.density_range, merge_points=mp, energy_mask=mask,
                  optimise_density=True, n_workers=spec.n_workers, verbose=verbose)
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
    (_, e_fit, re_fit, im_fit, e_beta, _) = K._prepare_fit_data(
        sld, tuple(energy_mask) if energy_mask else None)
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
