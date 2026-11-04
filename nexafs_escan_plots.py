"""Optical constants from an energy-scan fit (nexafs isld basis) shown with the NEXAFS peak components.

For each escan: isld(E) with every Gaussian (shaded from the baseline), the edge+baseline, and the total; sld(E)
below with the peak energies marked.  The fits live on the RSoXR energy axis; the NEXAFS peak model sits on the
NEXAFS axis (E_nexafs = E_rsoxr + shift, shift from the escan's nexafs entry), shown on a secondary top axis.
Use:  plot(fits, out_dir, ...) with fits = [(label, escan_dir, uncertainty_dir), ...]; figures and the optional
`<sample>_escan_optics_sigma.csv` are looked up in / written to out_dir.  (The Brewer1/2 call lives with the Brewer project:
NEXAFS_Fits/code/brewer12_escan_plots.py.)
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/homes/dfs1/Refltools")
sys.path.insert(0, "/homes/dfs1/Refltools/rsoxr_harness")
from rsoxr_harness import nexafs as NX


def load(d):
    s = json.load(open(f"{d}/summary.json"))
    nx = json.load(open(f"{d}/nexafs_SOC.json"))
    P = np.loadtxt(f"{d}/paths_SOC.csv", delimiter=",", skiprows=1)
    rho = json.load(open(f"{d}/../../project.json"))["escan"][os.path.basename(d)]["kk"][0]["density"]
    return s, nx, P, rho


def load_nuts(d, layer="SOC", method="nuts"):
    """Per-energy NUTS posterior of the SOC layer: median and 2.5/97.5 % (95 % interval) of sld and isld, χ²/ν (Birge factor), convergence.
    → dict of arrays sorted by energy.  Samples are (chains, draws, params) in physical units."""
    rows = []
    for f in sorted(glob.glob(f"{d}/{method}_*.json")):
        j = json.load(open(f))
        z = np.load(f.replace(".json", ".npz"), allow_pickle=True)
        names = [str(n) for n in z["param_names"]]
        if method == "nuts":
            S = z["samples_bounded"].reshape(-1, len(names))
            ok = bool(j["converged"]) and j["rhat_max"] < 1.05
        else:                                   # JAXNS: equal-weight posterior samples; ESS as the health check
            S = np.asarray(z["samples"]).reshape(-1, len(names))
            ok = bool(float(j.get("ESS", 0)) > 200)
        r = dict(E=j["energy"], ok=ok,
                 birge=float(np.sqrt(max(j["chi2_best"] / max(j["npoints"] - len(names), 1), 1.0))))
        for key, nm in (("sld", f"{layer} - sld"), ("isld", f"{layer} - isld")):
            v = S[:, names.index(nm)]
            r[key], r[key + "_lo"], r[key + "_hi"] = np.median(v), np.percentile(v, 2.5), np.percentile(v, 97.5)
        rows.append(r)
    rows.sort(key=lambda r: r["E"])
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def plot_points(ax, nu, key, method="nuts"):
    """Per-energy posterior medians with raw 95 % interval (dark) and Birge-scaled interval (light)."""
    m, lo, hi = nu[key], nu[key + "_lo"], nu[key + "_hi"]
    b = nu["birge"]
    for sc, kw in ((b, dict(ecolor="#f4a582", elinewidth=1.2, capsize=2.5, zorder=3)),
                   (1.0, dict(ecolor="k", elinewidth=1.0, capsize=0, zorder=4))):
        ax.errorbar(nu["E"], m, yerr=[sc * (m - lo), sc * (hi - m)], fmt="none", **kw)
    ok = nu["ok"]
    ax.plot(nu["E"][ok], m[ok], "o", ms=4.5, mfc="white", mec="k", mew=1.0, zorder=5)
    ax.plot(nu["E"][~ok], m[~ok], "x", ms=6, c="tab:red", mew=1.4, zorder=5)
    if method == "jaxns" and key == "isld":
        pass


def plot(fits, out_dir, path=None, xlim=(280.0, 335.0), method="nuts"):
    """method: 'nuts' | 'jaxns' per-energy posterior points (95 % intervals).  Also overlays the escan-fit 95 % band
    (±1.96 σ from the Gauss–Newton covariance, conditional on the NEXAFS model; usually thinner than the line) when
    `<sample>_escan_optics_sigma.csv` exists."""
    path = path or (f"{out_dir}/escan_optics_with_peaks.png" if method == "nuts" else f"{out_dir}/escan_optics_with_peaks_{method}.png")
    M = method.upper()
    os.makedirs(out_dir, exist_ok=True)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(2, len(fits), figsize=(7.2 * len(fits), 9), sharex=True)
    for j, (label, d, ud) in enumerate(fits):
        s, nx, P, rho = load(d)
        nu = load_nuts(ud, method=method)
        m, sh = nx["model"], nx["shift"]
        E, sld, isld = P[:, 0], P[:, 1], P[:, 2]
        keep = (E >= xlim[0] - 5) & (E <= xlim[1] + 5)
        Eg = np.linspace(xlim[0], xlim[1], 3000)
        tot, comp = NX.evaluate(m, Eg + sh, components=True)
        base = comp["baseline"]
        a, b = axes[0, j], axes[1, j]
        handles = []
        for i, p in enumerate(m["peaks"]):
            h = a.fill_between(Eg, base, base + comp[p["name"]], color=cols[i % 10], alpha=0.5, lw=0)
            handles.append((h, f"{p['energy']['value']:.2f} eV   FWHM {p['width']['value']:.2f}   h {p['height']['value']:.1f}"))
            b.axvline(p["energy"]["value"] - sh, color=cols[i % 10], lw=0.8, ls=":", alpha=0.8)
        a.plot(Eg, base + comp["edge"], "--", c="gray", lw=1.2, label="edge + baseline")
        a.plot(E[keep], isld[keep], "k-", lw=1.8, label="isld (KK-linked fit)")
        b.plot(E[keep], sld[keep], "k-", lw=1.8)
        inwin = (nu["E"] >= xlim[0]) & (nu["E"] <= xlim[1])
        nuw = {k: v[inwin] for k, v in nu.items()}
        plot_points(a, nuw, "isld", method)
        plot_points(b, nuw, "sld", method)
        sigf = f"{out_dir}/{label.split()[0]}_escan_optics_sigma.csv"
        if os.path.exists(sigf):
            sg = np.genfromtxt(sigf, delimiter=",", names=True)
            k2 = (sg["E_eV"] >= xlim[0] - 5) & (sg["E_eV"] <= xlim[1] + 5)
            a.fill_between(sg["E_eV"][k2], (sg["isld"] - 1.96 * sg["sigma_isld"])[k2], (sg["isld"] + 1.96 * sg["sigma_isld"])[k2],
                           color="tab:blue", alpha=0.35, lw=0, zorder=2)
            b.fill_between(sg["E_eV"][k2], (sg["sld"] - 1.96 * sg["sigma_sld"])[k2], (sg["sld"] + 1.96 * sg["sigma_sld"])[k2],
                           color="tab:blue", alpha=0.35, lw=0, zorder=2)
        b.axhline(0, c="k", lw=0.5)
        a.set_title(f"{label}   ρ = {rho:.3f} g/cm³ (C25O5H8N4),  cost {s['cost']:.0f}", fontsize=11)
        a.set_ylabel("isld (10$^{-6}$ Å$^{-2}$)"); b.set_ylabel("sld (10$^{-6}$ Å$^{-2}$)")
        b.set_xlabel("Energy on the RSoXR axis (eV)")
        sec = a.secondary_xaxis("top", functions=(lambda x, sh=sh: x + sh, lambda x, sh=sh: x - sh))
        sec.set_xlabel(f"NEXAFS axis (eV) = RSoXR {sh:+.2f}")
        leg1 = a.legend([h for h, _ in handles], [l for _, l in handles], title="NEXAFS peaks (energy, FWHM, height)",
                        loc="upper right", fontsize=7.5, title_fontsize=8, framealpha=0.9)
        a.add_artist(leg1)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        b.legend(handles=[Line2D([], [], c="k", lw=1.8, label="escan fit (KK-linked)"),
                          Line2D([], [], c="gray", ls="--", label="edge + baseline (isld panel)"),
                          Line2D([], [], marker="o", mfc="white", mec="k", ls="none", label=f"{M} median, per energy"),
                          Line2D([], [], marker="x", c="tab:red", ls="none", label=("NUTS not converged (R̂>1.05)" if method == "nuts" else "JAXNS ESS < 200")),
                          Line2D([], [], c="k", lw=1, label=f"{M} 95 % (raw)"),
                          Line2D([], [], c="#f4a582", lw=1.5, label=f"{M} 95 % × √(χ²/ν)"),
                          Patch(color="tab:blue", alpha=0.35, label="escan fit 95 % (model-conditional)")],
                 loc="lower right", fontsize=7.5, framealpha=0.9)
        a.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

