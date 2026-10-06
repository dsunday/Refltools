"""
Kramers–Kronig consistency check of fitted optical constants.

The fitted imaginary SLD of a layer (or any SLD table) is KK-transformed into
a real SLD (kkcalc via NEXAFS.imag_SLD_to_real_SLD: the measured Im is spliced
into tabulated scattering factors of `formula` at `density` between the merge
points) and compared with the fitted real SLD at every energy.  The density is
either given or fitted (bounded, same combined Re+Im cost as the stoichiometry
search).  The match does not need to be perfect; the check flags energies and
regions where the fitted Re and Im are mutually inconsistent.

Results go to <project>/kk/<name>/:
    spec.json  kk_table.csv  kk_sld.csv  summary.json
"""
import json
import os
import re

import numpy as np

from .stoich import resolve_source, kk_sld, _refine, _restrict


def _in_windows(e, windows):
    if not windows:
        return np.ones_like(e, dtype=bool)
    if len(windows) == 2 and not isinstance(windows[0], (list, tuple)):
        windows = [windows]
    m = np.zeros_like(e, dtype=bool)
    for lo, hi in windows:
        m |= (e >= lo - 1e-9) & (e <= hi + 1e-9)
    return m


def default_name(source, formula):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{source}_{formula}").strip("_")


def kk_dir(project, name):
    return os.path.join(project.dir("kk"), name)


def kk_check(project, source, formula, density=None, density_range=(0.5, 2.5),
             merge_points=None, energy_mask=None, criteria="best", name=None,
             tol=None):
    """
    source   : 'Model:Layer' (fitted SLD from the project h5), a material name,
               or a CSV/table path ([E, Re, Im]).
    formula  : chemical formula used for the KK tails (e.g. 'C8H8').
    density  : g/cm³; None → fitted within density_range.
    merge_points : [lo, hi] eV where the data are spliced in (default: data range).
    energy_mask  : [lo, hi] or [[lo, hi], ...] — energies used for the density
                   fit and the summary statistics (all energies are tabulated).
    tol      : |ΔRe| flag threshold (1e-6 Å⁻²); default max(1, 10 % of the
               fitted Re range).
    → summary dict (also written to <project>/kk/<name>/summary.json).
    """
    import pandas as pd
    sld, prov = resolve_source(project, source, criteria)
    mp = list(merge_points) if merge_points else [float(sld[:, 0].min()),
                                                  float(sld[:, 0].max())]
    sld = _restrict(sld, mp)
    if len(sld) < 3:
        raise ValueError(f"only {len(sld)} energies inside merge points {mp}")
    E, re_fit, im_fit = sld[:, 0], sld[:, 1], sld[:, 2]
    use = _in_windows(E, energy_mask)

    em = energy_mask
    if em and len(em) == 2 and not isinstance(em[0], (list, tuple)):
        em = tuple(em)                       # single window: stoich-search convention
    fitted_density = density is None
    if fitted_density:
        r = _refine(sld, formula, tuple(density_range), mp, em or None)
        if r is None:
            raise RuntimeError(f"KK transform failed for {formula}")
        density = r["density"]
        at_edge = min(density - density_range[0], density_range[1] - density) \
            < 0.01 * (density_range[1] - density_range[0])
    else:
        at_edge = False

    kk = kk_sld(sld, formula, float(density), mp)
    re_kk = np.interp(E, kk[:, 0], kk[:, 1])
    im_kk = np.interp(E, kk[:, 0], kk[:, 2])
    d_re = re_fit - re_kk
    d_im = im_fit - im_kk
    span = float(np.ptp(re_fit[use])) if use.sum() > 1 else 0.0
    tol = float(tol) if tol is not None else max(1.0, 0.1 * span)

    table = pd.DataFrame({"energy_eV": E, "re_fit": re_fit, "re_kk": re_kk,
                          "d_re": d_re, "im_fit": im_fit, "im_kk": im_kk,
                          "d_im": d_im, "used": use,
                          "flag": use & (np.abs(d_re) > tol)})
    du = d_re[use]
    rms = float(np.sqrt(np.mean(du ** 2)))
    rel = rms / span if span > 0 else float("nan")
    verdict = ("consistent" if rel < 0.10 else
               "roughly consistent" if rel < 0.25 else "inconsistent")
    worst = table[use].reindex(table[use].d_re.abs().sort_values(ascending=False).index)
    name = name or default_name(source, formula)
    summary = {
        "name": name, "source": source, "provenance": prov, "formula": formula,
        "density": float(density), "density_fitted": fitted_density,
        "density_range": list(density_range) if fitted_density else None,
        "density_at_range_edge": bool(at_edge),
        "merge_points": mp, "energy_mask": energy_mask, "criteria": criteria,
        "n_energies": int(use.sum()),
        "rms_d_re": rms, "mean_d_re": float(np.mean(du)),
        "max_abs_d_re": float(np.max(np.abs(du))),
        "max_at_eV": float(worst.energy_eV.iloc[0]),
        "re_span": span, "rms_over_span": rel, "verdict": verdict,
        "rms_d_im": float(np.sqrt(np.mean(d_im[use] ** 2))),
        "tol": tol, "flagged_eV": [float(e) for e in table.energy_eV[table.flag]],
        "worst": worst.head(5)[["energy_eV", "re_fit", "re_kk", "d_re"]]
        .to_dict(orient="records"),
    }
    d = kk_dir(project, name)
    os.makedirs(d, exist_ok=True)
    table.to_csv(os.path.join(d, "kk_table.csv"), index=False)
    np.savetxt(os.path.join(d, "kk_sld.csv"), kk, delimiter=",",
               header="Energy_eV,Real_SLD_kk,Imag_SLD_kk", comments="")
    with open(os.path.join(d, "spec.json"), "w") as fh:
        json.dump({"source": source, "formula": formula,
                   "density": None if fitted_density else float(density),
                   "density_range": list(density_range), "merge_points": mp,
                   "energy_mask": energy_mask, "criteria": criteria, "tol": tol},
                  fh, indent=2)
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary


def load_kk(project, name):
    import pandas as pd
    d = kk_dir(project, name)
    if not os.path.exists(os.path.join(d, "summary.json")):
        raise KeyError(f"no KK check '{name}' in {project.dir('kk')}")
    with open(os.path.join(d, "summary.json")) as fh:
        summary = json.load(fh)
    table = pd.read_csv(os.path.join(d, "kk_table.csv"))
    kk = np.loadtxt(os.path.join(d, "kk_sld.csv"), delimiter=",", skiprows=1)
    return summary, table, kk


def kk_names(project):
    d = project.dir("kk")
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d)
                  if os.path.exists(os.path.join(d, n, "summary.json")))


def format_summary(s):
    dens = (f"{s['density']:.3f} g/cm³ (fitted in {s['density_range']}"
            + (", AT RANGE EDGE" if s["density_at_range_edge"] else "") + ")"
            if s["density_fitted"] else f"{s['density']:.3f} g/cm³ (fixed)")
    lines = [f"KK check '{s['name']}': {s['source']} vs KK({s['formula']}), {dens}",
             f"  {s['provenance']}; merge points {s['merge_points'][0]:g}–"
             f"{s['merge_points'][1]:g} eV; {s['n_energies']} energies",
             f"  ΔRe = Re_fit − Re_KK: rms {s['rms_d_re']:.2f}, mean {s['mean_d_re']:+.2f}, "
             f"max |ΔRe| {s['max_abs_d_re']:.2f} at {s['max_at_eV']:g} eV  (1e-6 Å⁻²)",
             f"  rms / Re span ({s['re_span']:.1f}) = {s['rms_over_span']:.2f} → "
             f"{s['verdict']}  (rough guide: <0.10 consistent, <0.25 roughly)",
             f"  flagged |ΔRe| > {s['tol']:.2f}: "
             + (", ".join(f"{e:g}" for e in s["flagged_eV"]) or "none")]
    return "\n".join(lines)
