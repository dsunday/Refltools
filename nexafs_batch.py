"""Batch NEXAFS peak fitting for the Brewer Nov2025 samples.

Wraps NEXAFS_BuildingBlock.fit_nexafs_spectrum (Gaussian peaks + erf/decay edge) so the same
seed models (nexafs_models.py, extracted from LARIAT/BrewerNEXAFS_Fits.ipynb) can be applied to
  * LARIAT surface NEXAFS spectra   (kind='lariat',  Energy,Processed_Intensity)
  * RSoXR-fitted optical constants  (kind='isld',    SLD_Fits CSV, SLD_Imag column)

isld grids are sparse (~30 points) so for kind='isld' the peak energies/widths are frozen
(from the LARIAT fit of the same sample, or the seed model) and only heights are fitted, plus a
trial energy shift between the RSoXR and NEXAFS energy axes.

Run as a script for the pilot:  python nexafs_batch.py pilot
"""
import copy
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nexafs_models as M
from NEXAFS_BuildingBlock import fit_nexafs_spectrum, gaussian, edge_bl_func, save_peak_fit_results

NOV = "/homes/dfs1/Reflectivity/Brewer/Nov2025"
FALL = "/homes/dfs1/LARIAT/Brewer_Spectra/March2026/FallBatch"
OUT = f"{NOV}/NEXAFS_Fits"


# ---------------------------------------------------------------- data
def load_source(kind, path):
    """Return (E, y) sorted by energy. kind: 'lariat' | 'isld' | 'sld_real'."""
    df = pd.read_csv(path)
    col = {"lariat": 1, "isld": 2, "sld_real": 1}[kind]
    E = df.iloc[:, 0].to_numpy(float)
    y = df.iloc[:, col].to_numpy(float)
    o = np.argsort(E)
    return E[o], y[o]


def normalise_isld(E, y, pre=(None, 282.0), post=(320.0, None), edge_height=1.43):
    """Map isld to the processed-NEXAFS convention (pre-edge 0, post-edge step = edge_height).

    Returns (y_norm, pre_level, scale) so that isld = pre_level + scale * y_norm.
    """
    mpre = (E <= pre[1]) if pre[0] is None else (E >= pre[0]) & (E <= pre[1])
    mpost = (E >= post[0])
    pre_level = float(np.mean(y[mpre]))
    step = float(np.mean(y[mpost])) - pre_level
    scale = step / edge_height
    return (y - pre_level) / scale, pre_level, scale


# ---------------------------------------------------------------- model helpers
def with_flags(model, free_energy=True, free_width=True, free_height=True, free_edge=False):
    """Deep copy of a model with fit flags overridden (None-safe: keeps flag if arg is None)."""
    m = copy.deepcopy(model)
    for k, b in list(m["peaks"].items()):
        if not k.endswith("_bounds"):
            continue
        want = {"energy": free_energy, "width": free_width, "height": free_height}[k.rsplit("_", 2)[1]]
        if want is not None:
            m["peaks"][k] = (b[0], b[1], bool(want) and b[2] if want == "keep" else bool(want))
    return m


def relaxed(model, height_max=6.0):
    """Notebook seeds were tuned per sample (e.g. peak-2 height frozen). Free the pi* peak
    (energy/height), the edge height/decay and widen height caps so a seed can transfer."""
    m = copy.deepcopy(model)
    for k, b in list(m["peaks"].items()):
        if k.endswith("_height_bounds"):
            m["peaks"][k] = (0.0, height_max, True)
    m["peaks"]["peak_2_energy_bounds"] = (285.2, 285.8, True)
    m["edge"]["height_bounds"] = (0.6, 2.2, True)
    m["edge"]["decay_bounds"] = (0.0, 0.05, True)
    return m


def frozen_from_fit(model, fitted_peaks, free_height=True):
    """Seed `model` with fitted energies/widths (frozen) and heights (free, bounds widened)."""
    m = copy.deepcopy(model)
    for k, v in fitted_peaks.items():
        if k in m["peaks"]:
            m["peaks"][k] = v
    for k, b in list(m["peaks"].items()):
        if k.endswith("_energy_bounds") or k.endswith("_width_bounds"):
            m["peaks"][k] = (b[0], b[1], False)
        elif k.endswith("_height_bounds"):
            m["peaks"][k] = (0.0, max(3.0, 3 * m["peaks"][k.replace("_bounds", "")]), free_height)
    return m


def fit_one(E, y, model, energy_range=(280.0, 330.0), method="trf"):
    """Fit one spectrum; returns dict(results, peaks DataFrame, model)."""
    data = np.column_stack([E, y])
    res = fit_nexafs_spectrum(data, model["peaks"], model["edge"], baseline=model["baseline"],
                              method=method, algorithm="curve_fit", energy_range=energy_range)
    fv = res["fit_result"]["fitted_values"]
    unc = res["fit_result"]["parameter_uncertainties"]
    rows = []
    n = len([k for k in model["peaks"] if k.endswith("_energy")])
    for i in range(1, n + 1):
        row = {"peak": i}
        for q in ("energy", "width", "height"):
            name = f"peak_{i}_{q}"
            row[q] = res["fitted_peak_params"][name]
            row[q + "_err"] = unc.get(name, np.nan)
            lo, hi, flag = model["peaks"][name + "_bounds"]
            row[q + "_free"] = bool(flag)
            row[q + "_at_bound"] = bool(flag) and (abs(row[q] - lo) < 1e-6 * max(1, abs(lo)) or abs(row[q] - hi) < 1e-6 * max(1, abs(hi)))
        rows.append(row)
    return dict(results=res, peaks=pd.DataFrame(rows), model=model)


def model_curve(E, fit):
    pk = fit["results"]["fitted_peak_params"]
    ed = fit["results"]["fitted_edge_params"]
    n = len([k for k in pk if k.endswith("_energy")])
    comps = [gaussian(E, pk[f"peak_{i}_height"], pk[f"peak_{i}_energy"], pk[f"peak_{i}_width"]) for i in range(1, n + 1)]
    edge = edge_bl_func(E, ed["location"], ed["height"], ed["width"], ed["decay"])
    return fit["model"]["baseline"] + edge + np.sum(comps, axis=0), comps, edge


def plot_fit(E, y, fit, title, path, energy_range, ylabel="Processed intensity"):
    Eg = np.linspace(energy_range[0], energy_range[1], 1500)
    tot, comps, edge = model_curve(Eg, fit)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    m = (E >= energy_range[0]) & (E <= energy_range[1])
    a.plot(E[m], y[m], "k.", ms=4, label="data")
    a.plot(Eg, tot, "r-", lw=1.5, label=f"fit  R²={fit['results']['r_squared']:.4f}")
    a.plot(Eg, edge, "--", c="gray", lw=1, label="edge")
    for c in comps:
        a.fill_between(Eg, 0, c, alpha=0.3)
    a.set_ylabel(ylabel); a.set_title(title); a.legend()
    ftot, _, _ = model_curve(E[m], fit)
    b.plot(E[m], y[m] - ftot, "k.-", ms=3, lw=0.5); b.axhline(0, c="r", lw=0.5)
    b.set_xlabel("Energy (eV)"); b.set_ylabel("resid")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_fit(fit, outdir, E, y, title, energy_range, extra=None, ylabel="Processed intensity"):
    os.makedirs(outdir, exist_ok=True)
    fit["peaks"].to_csv(f"{outdir}/peaks.csv", index=False)
    save_peak_fit_results(fit["results"], f"{outdir}/result.pkl", fit["model"]["baseline"])
    plot_fit(E, y, fit, title, f"{outdir}/fit.png", energy_range, ylabel)
    r = fit["results"]
    summ = dict(title=title, r_squared=r["r_squared"], rmse=r["rmse"], n_at_bound=int(fit["peaks"].filter(like="at_bound").to_numpy().sum()))
    summ.update(extra or {})
    pd.Series(summ).to_csv(f"{outdir}/summary.csv", header=False)
    return summ


# ---------------------------------------------------------------- isld with energy-shift scan
def fit_isld(E, isld, model, frozen_peaks=None, shifts=np.arange(-2.0, 2.01, 0.25), energy_range=(268.0, 335.0),
             tail_decay=None, tail_min_E=320.0):
    """Heights-only fit of normalised isld; scans E_rsoxr -> E_nexafs = E + shift. Returns best.

    tail_decay: if given, the edge decay is fixed to this value and the edge height is solved so the
    edge passes through the mean of the points above tail_min_E (high-energy data take priority;
    use when there are no points in the 300-310 eV gap to pin the broad peaks)."""
    yn, pre, scale = normalise_isld(E, isld, edge_height=model["edge"]["height"])
    base = frozen_from_fit(model, frozen_peaks) if frozen_peaks else frozen_from_fit(model, {
        k: v for k, v in model["peaks"].items() if not k.endswith("_bounds")})
    # normalisation to a fixed edge height misses the high-energy tail (exp decay of the edge),
    # so let edge height/decay float; peak heights stay free, energies/widths frozen
    base["edge"] = dict(base["edge"], height_bounds=(0.6, 6.0, True), decay_bounds=(0.0, 0.05, True))
    best = None
    for s in shifts:
        if tail_decay is not None:
            m = (E + s) >= tail_min_E
            H = yn[m].mean() / edge_bl_func(E[m] + s, base["edge"]["location"], 1.0, base["edge"]["width"], tail_decay).mean()
            base["edge"] = dict(base["edge"], height=float(H), decay=tail_decay)
            base["edge"].pop("height_bounds", None); base["edge"].pop("decay_bounds", None)
        try:
            f = fit_one(E + s, yn, base, energy_range=energy_range)
        except Exception as exc:  # a failed shift is just skipped
            print("  shift", s, "failed:", exc)
            continue
        rss = float(np.sum(f["results"]["residuals"] ** 2))
        if best is None or rss < best[0]:
            best = (rss, s, f)
    rss, s, f = best
    return f, dict(shift=s, pre_level=pre, scale=scale), E + s, yn


# ---------------------------------------------------------------- pilot
PILOT = {
    # name: (lariat file, isld csv, model, label)
    "Brewer1_SOC_UV": (f"{FALL}/Brewer1_SOC_UV_Carbon_processed.csv", f"{NOV}/SLD_Fits/Brewer1/Brewer1_SOC_UV_Carbon.csv", M.SOC_CARBON_UV),
    "Brewer3_SOG_UV200": (f"{FALL}/Brewer3_SOG_UV200_processed.csv", f"{NOV}/SLD_Fits/Brewer3/Brewer3_SOG_UV200_Carbon.csv", M.SOG_CARBON_BAKE),
}


# Tail anchoring is decided per sample/layer after looking at the data (never automatic):
# add an entry only where the 300-310 eV gap leaves the broad peaks undefined (see isld_coverage).
TAIL_DECAY = {"Brewer1_SOC_UV": 0.005}


def isld_coverage(E):
    """Points per high-energy window, to help decide whether the tail needs anchoring."""
    return {"n_300_310": int(np.sum((E >= 300) & (E <= 310))), "n_ge_310": int(np.sum(E > 310)), "n_ge_320": int(np.sum(E >= 320))}


def run_pilot():
    for name, (lar, sld, model) in PILOT.items():
        print("==", name)
        if not os.path.exists(lar):
            print("  missing", lar); continue
        E, y = load_source("lariat", lar)
        f0 = fit_one(E, y, model)
        print("  seed-as-is R2 = %.4f" % f0["results"]["r_squared"])
        f = fit_one(E, y, relaxed(model))
        s = save_fit(f, f"{OUT}/{name}/lariat", E, y, f"{name} LARIAT carbon", (280.0, 330.0))
        print("  LARIAT", s)
        print(f["peaks"][["peak", "energy", "width", "height", "energy_at_bound", "width_at_bound", "height_at_bound"]].round(3).to_string(index=False))
        Ei, ii = load_source("isld", sld)
        frozen = f["results"]["fitted_peak_params"]
        fi, info, Es, yn = fit_isld(Ei, ii, model, tail_decay=TAIL_DECAY.get(name), frozen_peaks={k: v for k, v in frozen.items() if not k.endswith("_bounds")})
        s = save_fit(fi, f"{OUT}/{name}/isld", Es, yn, f"{name} RSoXR isld (shift {info['shift']:+.2f} eV)",
                     (268.0, 335.0), extra=info, ylabel="isld, normalised")
        print("  isld", s, "coverage", isld_coverage(Ei + info["shift"]))


# ---------------------------------------------------------------- reference samples 1-6 (carbon edge)
REFS = {
    # name: (lariat file, SLD_Fits carbon csv, seed model, layer)
    "Brewer1_SOC_UV": (f"{FALL}/Brewer1_SOC_UV_Carbon_processed.csv", f"{NOV}/SLD_Fits/Brewer1/Brewer1_SOC_UV_Carbon.csv", M.SOC_CARBON_UV, "SOC"),
    "Brewer2_SOC_Bake": (f"{FALL}/Brewer2_SOC_Bake_processed.csv", f"{NOV}/SLD_Fits/Brewer2/Brewer2_SOC_Bake_Carbon.csv", M.SOC_CARBON_BAKE, "SOC"),
    "Brewer3_SOG_UV200": (f"{FALL}/Brewer3_SOG_UV200_processed.csv", f"{NOV}/SLD_Fits/Brewer3/Brewer3_SOG_UV200_Carbon.csv", M.SOG_CARBON_BAKE, "SOG"),
    "Brewer4_SOG_UV100": (f"{FALL}/Brewer4_SOG_UV100_processed.csv", f"{NOV}/SLD_Fits/Brewer4/Brewer4_SOG_UV100_Carbon.csv", M.SOG_CARBON_BAKE, "SOG"),
    "Brewer5_SOG_UV50": (f"{FALL}/Brewer5_SOG_UV50_processed.csv", f"{NOV}/SLD_Fits/Brewer5/Brewer5_SOG_UV50_Carbon.csv", M.SOG_CARBON_BAKE, "SOG"),
    "Brewer6_SOG_Bake": (f"{FALL}/Brewer6_SOG_Bake_processed.csv", f"{NOV}/SLD_Fits/Brewer6/Brewer6_SOG_Bake_Carbon.csv", M.SOG_CARBON_BAKE, "SOG"),
}


def flags_for(source, summ, peaks, cov=None, lariat_shift_ref=None):
    fl = []
    if summ["r_squared"] < 0.97 if source == "lariat" else summ["r_squared"] < 0.93:
        fl.append("low_R2")
    if summ["n_at_bound"] >= 6:
        fl.append(f"{summ['n_at_bound']}_params_at_bound")
    if cov is not None and cov["n_ge_320"] == 0:
        fl.append("no_pts_ge_320")
    if cov is not None and cov["n_300_310"] == 0:
        fl.append("gap_300_310_consider_tail_anchor")
    return fl


def run_refs(names=None):
    """LARIAT + isld free-edge fits for reference samples (no tail anchoring); writes NEXAFS_Fits/refs_summary.csv."""
    rows = []
    for name, (lar, sld, model, layer) in REFS.items():
        if names and name not in names:
            continue
        row = dict(sample=name, layer=layer)
        E, y = load_source("lariat", lar)
        f0 = fit_one(E, y, model, energy_range=(280.0, 330.0))
        f = fit_one(E, y, relaxed(model), energy_range=(280.0, 330.0))
        s = save_fit(f, f"{OUT}/{name}/lariat", E, y, f"{name} LARIAT carbon", (280.0, 330.0))
        row.update(lariat_R2_seed=f0["results"]["r_squared"], lariat_R2=s["r_squared"], lariat_at_bound=s["n_at_bound"],
                   lariat_edge_H=f["results"]["fitted_edge_params"]["height"], lariat_edge_decay=f["results"]["fitted_edge_params"]["decay"])
        flags = [f"lariat:{x}" for x in flags_for("lariat", s, f["peaks"])]
        Ei, ii = load_source("isld", sld)
        frozen = {k: v for k, v in f["results"]["fitted_peak_params"].items() if not k.endswith("_bounds")}
        fi, info, Es, yn = fit_isld(Ei, ii, model, frozen_peaks=frozen, tail_decay=TAIL_DECAY.get(name))
        cov = isld_coverage(Es)
        si = save_fit(fi, f"{OUT}/{name}/isld", Es, yn, f"{name} RSoXR isld (shift {info['shift']:+.2f} eV)",
                      (268.0, 335.0), extra=dict(info, **cov), ylabel="isld, normalised")
        ed = fi["results"]["fitted_edge_params"]
        row["tail_anchor_decay"] = TAIL_DECAY.get(name)
        row.update(isld_R2=si["r_squared"], isld_at_bound=si["n_at_bound"], shift=info["shift"], isld_edge_H=ed["height"],
                   isld_edge_decay=ed["decay"], **cov)
        flags += [f"isld:{x}" for x in flags_for("isld", si, fi["peaks"], cov)]
        if ed["decay"] <= 1e-4 and name not in TAIL_DECAY:
            flags.append("isld:edge_decay_at_0")
        if abs(info["shift"]) >= 2.0:
            flags.append("isld:shift_at_scan_limit")
        row["flags"] = ";".join(flags)
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/refs_summary.csv", index=False)
    return df



# ---------------------------------------------------------------- quantitative comparison of isld peaks
GAUSS_AREA = np.sqrt(np.pi / (4 * np.log(2)))  # area = height * FWHM * GAUSS_AREA


def compare_isld_peaks(names=None, outdir=OUT, tag=""):
    """Absolute isld peak heights/areas (1e-6 A^-2 [* eV]) for each sample from saved isld fits.

    peaks.csv holds heights in normalised units; isld = pre_level + scale * y_norm, so the absolute
    height is height * scale (summary.csv). Errors are conditional on the frozen energies/widths."""
    rows = []
    for name in (names or list(REFS)):
        d = f"{outdir}/{name}/isld"
        pk = pd.read_csv(f"{d}/peaks.csv")
        sm = pd.read_csv(f"{d}/summary.csv", header=None, index_col=0)[1]
        sc = float(sm["scale"])
        for _, r in pk.iterrows():
            rows.append(dict(sample=name, peak=int(r["peak"]), energy=r["energy"], fwhm=r["width"],
                             height_abs=r["height"] * sc, height_abs_err=r["height_err"] * sc,
                             area_abs=r["height"] * sc * r["width"] * GAUSS_AREA,
                             at_bound=bool(r["height_at_bound"])))
    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/isld_peak_comparison{tag}.csv", index=False)
    samples = df["sample"].unique()
    peaks = sorted(df["peak"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    w = 0.8 / len(samples)
    for ax, col, lab in ((axes[0], "height_abs", "peak height  (isld, 1e-6 Å$^{-2}$)"),
                         (axes[1], "area_abs", "peak area  (1e-6 Å$^{-2}$ eV)")):
        for j, sname in enumerate(samples):
            sub = df[df["sample"] == sname].set_index("peak")
            x = np.arange(len(peaks)) + (j - (len(samples) - 1) / 2) * w
            ax.bar(x, [sub.loc[p, col] for p in peaks], w, label=sname,
                   yerr=[sub.loc[p, "height_abs_err"] * (1 if col == "height_abs" else sub.loc[p, "fwhm"] * GAUSS_AREA) for p in peaks],
                   capsize=2, hatch=None)
        ax.set_ylabel(lab)
    ref = df[df["sample"] == samples[0]].set_index("peak")["energy"]
    axes[1].set_xticks(np.arange(len(peaks)))
    axes[1].set_xticklabels([f"{p}\n~{ref[p]:.1f}" for p in peaks])
    axes[1].set_xlabel("peak # (energy in first sample, eV; frozen from each sample's LARIAT fit)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title(f"RSoXR isld peak comparison {tag.strip('_')} (absolute units; peaks 7–9 poorly constrained)")
    fig.tight_layout(); fig.savefig(f"{outdir}/isld_peak_comparison{tag}.png", dpi=150); plt.close(fig)
    return df


def plot_isld_grid(names, path, outdir=OUT, xlim=(280.0, 335.0),
                   groups=(((1, 2), "285.0 + 285.2 eV (sum)"), ((5,), "288.8 eV"), ((6,), "291.5 eV"), ((7,), "293.3 eV"))):
    """Row 1: isld fits (absolute units, shared y, legend by peak energy).
    Row 2: one panel per peak group comparing absolute peak areas (1e-6 Å^-2 eV) across the samples.
    Area errors are conditional on frozen energies/widths; sum errors add in quadrature (peaks 1/2 are
    anticorrelated, so the sum error is overestimated)."""
    from NEXAFS_BuildingBlock import load_peak_fit_results
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.6), gridspec_kw=dict(height_ratios=[1.15, 1]))
    top, bot = axes
    for a_ in top[1:]:
        a_.sharey(top[0])
    for a_ in bot[1:]:
        a_.sharey(bot[0])
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    Eg = np.linspace(xlim[0], xlim[1], 1500)
    handles = {}
    areas = {}
    for ax, name in zip(top, names):
        d = f"{outdir}/{name}/isld"
        sm = pd.read_csv(f"{d}/summary.csv", header=None, index_col=0)[1]
        sh, pre, sc = float(sm["shift"]), float(sm["pre_level"]), float(sm["scale"])
        res = load_peak_fit_results(f"{d}/result.pkl")
        Ei, ii = load_source("isld", REFS[name][1])
        fit = dict(results=res, model=dict(baseline=0))
        tot, comps, edge = model_curve(Eg, fit)
        pk = res["fitted_peak_params"]
        ax.plot(Ei + sh, ii, "k.", ms=5, zorder=5)
        ax.plot(Eg, pre + sc * tot, "r-", lw=1.6, zorder=4)
        ax.plot(Eg, pre + sc * edge, "--", c="gray", lw=1)
        for i, c in enumerate(comps, 1):
            h = ax.fill_between(Eg, pre, pre + sc * c, color=cols[(i - 1) % 10], alpha=0.45, lw=0)
            handles[i] = (h, f"{pk[f'peak_{i}_energy']:.1f} eV")
        ax.set_title(f"{name.split('_', 1)[0]} {name.split('_', 1)[1]}\nR²={float(sm['r_squared']):.3f}, shift {sh:+.2f} eV", fontsize=10)
        ax.set_xlabel("Energy (eV)"); ax.set_xlim(*xlim)
        pdf = pd.read_csv(f"{d}/peaks.csv").set_index("peak")
        areas[name] = {i: (pdf.loc[i, "height"] * sc * pdf.loc[i, "width"] * GAUSS_AREA,
                           pdf.loc[i, "height_err"] * sc * pdf.loc[i, "width"] * GAUSS_AREA) for i in pdf.index}
    top[0].set_ylabel("isld (1e-6 Å$^{-2}$)")
    short = [nm.split("_", 1)[0].replace("Brewer", "B") + "\n" + nm.split("_", 1)[1] for nm in names]
    for ax, (grp, title) in zip(bot, groups):
        val = [sum(areas[nm][i][0] for i in grp) for nm in names]
        err = [np.sqrt(sum(areas[nm][i][1] ** 2 for i in grp)) for nm in names]
        ax.bar(range(n), val, yerr=err, capsize=3, color=[plt.cm.Set2(j) for j in range(n)])
        ax.set_xticks(range(n)); ax.set_xticklabels(short, fontsize=8)
        ax.set_title(title, fontsize=10); ax.axhline(0, c="k", lw=0.5)
    bot[0].set_ylabel("peak area (1e-6 Å$^{-2}$ eV)")
    hl = [(handles[i][0], handles[i][1]) for i in sorted(handles)]
    fig.legend([h for h, _ in hl], [l for _, l in hl], title=f"peaks ({names[0].split('_')[0]} energies)", loc="center left",
               bbox_to_anchor=(1.0, 0.62), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return areas


if __name__ == "__main__":
    if sys.argv[1:] == ["pilot"]:
        run_pilot()
    elif sys.argv[1:] == ["compare"]:
        for tag, grp in (("_SOC", [n for n, v in REFS.items() if v[3] == "SOC"]), ("_SOG", [n for n, v in REFS.items() if v[3] == "SOG"])):
            compare_isld_peaks(grp, tag=tag)
    elif sys.argv[1:] == ["sog_grid"]:
        plot_isld_grid([n for n, v in REFS.items() if v[3] == "SOG"], f"{OUT}/isld_fits_SOG.png")
    elif sys.argv[1:] == ["refs"]:
        run_refs()
