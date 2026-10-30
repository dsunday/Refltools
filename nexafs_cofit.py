"""Joint (co-)fit of several NEXAFS spectra with shared peak positions and widths.

Shared across the samples (but free to vary within bounds): peak energies and widths.
Per sample: peak heights, edge height and edge decay (edge location/width fixed at the seed values).

Data kinds: 'lariat' (processed surface NEXAFS) or 'isld' (RSoXR fitted isld, normalised with
nexafs_batch.normalise_isld and shifted by the per-sample shift from the independent isld fit).
Uncertainties come from the full covariance (J^T J)^-1 * RSS/dof, so shared-parameter correlations
(e.g. peaks 1/2) propagate into areas and ratios via the delta method.

Run:  python nexafs_cofit.py sog      (SOG references Brewer3-6, LARIAT + isld)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nexafs_batch as B
from NEXAFS_BuildingBlock import gaussian, edge_bl_func, load_peak_fit_results

G = B.GAUSS_AREA
OUT = f"{B.OUT}/CoFit"


# ---------------------------------------------------------------- data
def load_sample(name, kind):
    """Return dict(E, y, window, pre_level, scale) for one sample and data kind."""
    lar, sld, model, layer = B.REFS[name]
    if kind == "lariat":
        E, y = B.load_source("lariat", lar)
        m = (E >= 280.0) & (E <= 330.0)
        return dict(E=E[m], y=y[m], pre_level=0.0, scale=1.0, shift=0.0)
    sm = pd.read_csv(f"{B.OUT}/{name}/isld/summary.csv", header=None, index_col=0)[1]
    sh = float(sm["shift"])
    Ei, ii = B.load_source("isld", sld)
    yn, pre, sc = B.normalise_isld(Ei, ii)
    E = Ei + sh
    m = (E >= 268.0) & (E <= 335.0)
    return dict(E=E[m], y=yn[m], pre_level=pre, scale=sc, shift=sh)


# ---------------------------------------------------------------- model
class CoFit:
    """Parameter vector: [E_1..E_n, W_1..W_n, H_(s,i) (S x n), edgeH_s, edgeDecay_s]."""

    def __init__(self, names, kind, model, wide=False, drop=()):
        """wide: energy bounds +-0.5 eV beyond the seed bounds, width bounds x0.5 / x1.6.
        drop: peak numbers (1-based) removed from the model entirely."""
        self.names, self.kind, self.S = list(names), kind, len(names)
        self.m = B.relaxed(model)  # seed bounds for energy/width; heights/edge bounds widened
        n_all = len([k for k in self.m["peaks"] if k.endswith("_energy")])
        self.peaks = [i for i in range(1, n_all + 1) if i not in set(drop)]  # active peak numbers
        self.n = len(self.peaks)
        self.data = [load_sample(nm, kind) for nm in names]
        self.loc, self.ewid = self.m["edge"]["location"], self.m["edge"]["width"]
        n, S = self.n, self.S
        pb = self.m["peaks"]
        lo_e = [pb[f"peak_{i}_energy_bounds"][0] - (0.5 if wide else 0) for i in self.peaks]
        hi_e = [pb[f"peak_{i}_energy_bounds"][1] + (0.5 if wide else 0) for i in self.peaks]
        lo_w = [pb[f"peak_{i}_width_bounds"][0] * (0.5 if wide else 1) for i in self.peaks]
        hi_w = [pb[f"peak_{i}_width_bounds"][1] * (1.6 if wide else 1) for i in self.peaks]
        hmax = 6.0 if kind == "lariat" else 8.0
        hh = 3.0 if kind == "lariat" else 6.0
        self.lo = np.r_[lo_e, lo_w, np.zeros(S * n), np.full(S, 0.6), np.zeros(S)]
        self.hi = np.r_[hi_e, hi_w, np.full(S * n, hmax), np.full(S, hh), np.full(S, 0.05)]

    def unpack(self, th):
        n, S = self.n, self.S
        return th[:n], th[n:2 * n], th[2 * n:2 * n + S * n].reshape(S, n), th[2 * n + S * n:2 * n + S * n + S], th[2 * n + S * n + S:]

    def curve(self, th, s, E):
        e, w, h, eh, ed = self.unpack(th)
        comps = [gaussian(E, h[s, i], e[i], w[i]) for i in range(self.n)]
        edge = edge_bl_func(E, self.loc, eh[s], self.ewid, ed[s])
        return edge + np.sum(comps, axis=0), comps, edge

    def resid(self, th):
        return np.concatenate([d["y"] - self.curve(th, s, d["E"])[0] for s, d in enumerate(self.data)])

    def start(self):
        """Mean of the independent fits (energies/widths), each sample's own heights/edge."""
        n, S = self.n, self.S
        E, W, H, EH, ED = np.zeros(n), np.zeros(n), np.zeros((S, n)), np.zeros(S), np.zeros(S)
        for s, nm in enumerate(self.names):
            d = f"{B.OUT}/{nm}/{self.kind}"
            pk = pd.read_csv(f"{d}/peaks.csv").set_index("peak")
            pk = pk.loc[self.peaks]
            E += pk["energy"].to_numpy() / S
            W += pk["width"].to_numpy() / S
            H[s] = pk["height"].to_numpy()
            ed = load_peak_fit_results(f"{d}/result.pkl")["fitted_edge_params"]
            EH[s], ED[s] = ed["height"], ed["decay"]
        th = np.r_[E, W, H.ravel(), EH, ED]
        return np.clip(th, self.lo + 1e-9, self.hi - 1e-9)

    def fit(self):
        th0 = self.start()
        r = least_squares(self.resid, th0, bounds=(self.lo, self.hi), method="trf", x_scale="jac", max_nfev=5000)
        n_pts = sum(len(d["y"]) for d in self.data)
        dof = max(n_pts - len(r.x), 1)
        rss = float(np.sum(r.fun ** 2))
        J = r.jac
        cov = np.linalg.pinv(J.T @ J) * rss / dof
        self.res = dict(th=r.x, cov=cov, rss=rss, dof=dof, n_pts=n_pts, success=r.success, nfev=r.nfev)
        return self.res

    # ---- derived quantities with delta-method errors
    def propagate(self, fn, eps=1e-6):
        th, cov = self.res["th"], self.res["cov"]
        v0 = np.atleast_1d(fn(th))
        g = np.zeros((len(v0), len(th)))
        for k in range(len(th)):
            d = np.zeros_like(th)
            d[k] = eps * max(1.0, abs(th[k]))
            g[:, k] = (np.atleast_1d(fn(th + d)) - np.atleast_1d(fn(th - d))) / (2 * d[k])
        return v0, np.sqrt(np.maximum(np.einsum("ik,kl,il->i", g, cov, g), 0))

    def area_fn(self, groups, scale_abs=False):
        def fn(th):
            e, w, h, _, _ = self.unpack(th)
            out = []
            for s in range(self.S):
                sc = self.data[s]["scale"] if scale_abs else 1.0
                for grp in groups:
                    out.append(sum(h[s, self.peaks.index(i)] * w[self.peaks.index(i)] * G for i in grp if i in self.peaks) * sc)
            return np.array(out)
        return fn

    def at_bound(self):
        th = self.res["th"]
        tol = 1e-4 * (self.hi - self.lo)
        return (th - self.lo < tol) | (self.hi - th < tol)


# ---------------------------------------------------------------- run + report
GROUPS = (((1, 2), "285.0 + 285.2 eV (sum)"), ((5,), "288.8 eV"), ((6,), "291.5 eV"), ((7,), "293.3 eV"))


def independent_rss(names, kind):
    out = []
    for nm in names:
        res = load_peak_fit_results(f"{B.OUT}/{nm}/{kind}/result.pkl")
        out.append(float(np.sum(np.asarray(res["residuals"]) ** 2)))
    return out


def run(names, kind, model, tag, wide=False, drop=(), make_plot=True, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    cf = CoFit(names, kind, model, wide=wide, drop=drop)
    r = cf.fit()
    th, cov = r["th"], r["cov"]
    e, w, h, eh, ed = cf.unpack(th)
    err = np.sqrt(np.diag(cov))
    n, S = cf.n, cf.S
    ab = cf.at_bound()
    # shared peaks table
    sp = pd.DataFrame(dict(peak=cf.peaks, energy=e, energy_err=err[:n], energy_at_bound=ab[:n],
                           width=w, width_err=err[n:2 * n], width_at_bound=ab[n:2 * n]))
    sp.to_csv(f"{OUT}/{tag}_shared_peaks.csv", index=False)
    # per-sample heights, areas (absolute for isld) with covariance-propagated errors
    absu = kind == "isld"
    singles = tuple(((i,), f"peak {i}") for i in cf.peaks)
    grp = GROUPS + singles
    v, ve = cf.propagate(cf.area_fn([g for g, _ in grp], scale_abs=absu))
    rows = []
    for s, nm in enumerate(names):
        for gi, (g, lab) in enumerate(grp):
            rows.append(dict(sample=nm, group=lab, peaks="+".join(map(str, g)), area=v[s * len(grp) + gi], area_err=ve[s * len(grp) + gi]))
    pd.DataFrame(rows).to_csv(f"{OUT}/{tag}_areas.csv", index=False)
    # ratios to 288.8 eV (peak 5) with propagated errors
    def ratio_fn(th):
        a = cf.area_fn([g for g, _ in GROUPS], scale_abs=False)(th).reshape(S, len(GROUPS))
        return np.concatenate([a[:, [0, 2, 3]] / a[:, [1]]]).ravel()
    rv, re = cf.propagate(ratio_fn)
    rr = pd.DataFrame(rv.reshape(S, 3), index=names, columns=["sum(p1+p2)/p5", "p6/p5", "p7/p5"])
    rre = pd.DataFrame(re.reshape(S, 3), index=names, columns=[c + "_err" for c in rr.columns])
    pd.concat([rr, rre], axis=1).to_csv(f"{OUT}/{tag}_ratios.csv")
    # fit quality vs independent fits
    rss_ind = independent_rss(names, kind)
    per = []
    for s, (nm, d) in enumerate(zip(names, cf.data)):
        rs = float(np.sum((d["y"] - cf.curve(th, s, d["E"])[0]) ** 2))
        ss_tot = float(np.sum((d["y"] - d["y"].mean()) ** 2))
        per.append(dict(sample=nm, R2_cofit=1 - rs / ss_tot, RSS_cofit=rs, RSS_independent=rss_ind[s], edge_H=eh[s], edge_decay=ed[s]))
    qual = pd.DataFrame(per)
    qual.to_csv(f"{OUT}/{tag}_quality.csv", index=False)
    N, k = r["n_pts"], len(th)
    # AIC/BIC from the Gaussian-likelihood RSS (residuals are correlated -> indicative only)
    r.update(aic=N * np.log(r["rss"] / N) + 2 * k, bic=N * np.log(r["rss"] / N) + k * np.log(N), k=k, n_at_bound=int(ab.sum()))
    print(f"[{tag}] success={r['success']} RSS={r['rss']:.4f} (independent sum {sum(rss_ind):.4f}) "
          f"params={k} pts={N} at-bound={r['n_at_bound']} AIC={r['aic']:.1f} BIC={r['bic']:.1f}")
    if verbose:
        print(sp.round(3).to_string(index=False))
        print(qual.round(4).to_string(index=False))
        print(pd.concat([rr, rre], axis=1).round(3).to_string())
    if make_plot:
        plot(cf, names, tag, sp, v, ve, len(grp))
    return cf, sp, qual, pd.concat([rr, rre], axis=1)


def plot(cf, names, tag, sp, v, ve, ng):
    S, n = cf.S, cf.n
    th = cf.res["th"]
    kind = cf.kind
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    xlim = (280.0, 335.0) if kind == "isld" else (280.0, 330.0)
    Eg = np.linspace(*xlim, 1500)
    fig, axes = plt.subplots(2, S, figsize=(4.2 * S, 8.6), gridspec_kw=dict(height_ratios=[1.15, 1]))
    top, bot = axes
    for a in top[1:]:
        a.sharey(top[0])
    for a in bot[1:]:
        a.sharey(bot[0])
    handles = []
    for s, (ax, nm) in enumerate(zip(top, names)):
        d = cf.data[s]
        pre, sc = d["pre_level"], d["scale"]
        tot, comps, edge = cf.curve(th, s, Eg)
        f = (lambda y: pre + sc * y) if kind == "isld" else (lambda y: y)
        y0 = pre if kind == "isld" else 0.0
        ydata = f(d["y"])
        ax.plot(d["E"], ydata, "k.", ms=4 if kind == "lariat" else 5, zorder=5)
        ax.plot(Eg, f(tot), "r-", lw=1.5, zorder=4)
        ax.plot(Eg, f(edge), "--", c="gray", lw=1)
        for i, c in enumerate(comps):
            h = ax.fill_between(Eg, y0, y0 + (sc if kind == "isld" else 1.0) * c, color=cols[(cf.peaks[i] - 1) % 10], alpha=0.45, lw=0)
            if s == 0:
                handles.append((h, f"{sp.energy[i]:.2f} eV (FWHM {sp.width[i]:.2f})"))
        r2 = 1 - np.sum((d["y"] - cf.curve(th, s, d["E"])[0]) ** 2) / np.sum((d["y"] - d["y"].mean()) ** 2)
        ax.set_title(f"{nm.split('_', 1)[0]} {nm.split('_', 1)[1]}\nco-fit R²={r2:.3f}", fontsize=10)
        ax.set_xlabel("Energy (eV)"); ax.set_xlim(*xlim)
    top[0].set_ylabel("isld (1e-6 Å$^{-2}$)" if kind == "isld" else "processed intensity")
    short = [nm.split("_", 1)[0].replace("Brewer", "B") + "\n" + nm.split("_", 1)[1] for nm in names]
    shown = [gi for gi, (g, _) in enumerate(GROUPS) if all(i in cf.peaks for i in g)]  # skip groups whose peak was dropped
    for ax in bot[len(shown):]:
        ax.set_visible(False)
    en = dict(zip(sp.peak, sp.energy))
    for ax, gi in zip(bot, shown):
        g, _ = GROUPS[gi]
        lab = (" + ".join(f"{en[i]:.1f}" for i in g) + " eV (sum)") if len(g) > 1 else f"{en[g[0]]:.1f} eV (peak {g[0]})"
        val = [v[s * ng + gi] for s in range(S)]
        er = [ve[s * ng + gi] for s in range(S)]
        ax.bar(range(S), val, yerr=er, capsize=3, color=[plt.cm.Set2(j) for j in range(S)])
        ax.set_xticks(range(S)); ax.set_xticklabels(short, fontsize=8)
        ax.set_title(lab, fontsize=10); ax.axhline(0, c="k", lw=0.5)
    bot[0].set_ylabel("peak area (1e-6 Å$^{-2}$ eV)" if kind == "isld" else "peak area (norm. units eV)")
    fig.legend([h for h, _ in handles], [l for _, l in handles], title="shared peaks (energy, FWHM)", loc="center left",
               bbox_to_anchor=(1.0, 0.62), fontsize=9)
    fig.suptitle(f"Co-fit ({kind}): shared peak energies/widths, per-sample heights and edge", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def variants(names, kind, model, groupname="SOG"):
    """seed vs wide bounds, with vs without peak 6 (291.5 eV)."""
    rows, ratios = [], {}
    for wide in (False, True):
        for drop in ((), (6,)):
            tag = f"{groupname}_{kind}_{'wide' if wide else 'seed'}_{'no6' if drop else 'with6'}"
            cf, sp, qual, rat = run(names, kind, model, tag, wide=wide, drop=drop, verbose=False)
            r = cf.res
            e = dict(zip(sp.peak, sp.energy)); w = dict(zip(sp.peak, sp.width))
            row = dict(variant=tag, bounds="wide" if wide else "seed", peak6="dropped" if drop else "kept", RSS=r["rss"], k=r["k"],
                       at_bound=r["n_at_bound"], AIC=r["aic"], BIC=r["bic"], **{f"R2_{n_.split('_')[0]}": q for n_, q in zip(qual["sample"], qual["R2_cofit"])})
            for pk in (1, 2, 4, 5, 6, 7, 9):
                row[f"E{pk}"] = e.get(pk, np.nan); row[f"W{pk}"] = w.get(pk, np.nan)
            rows.append(row)
            ratios[tag] = rat
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{groupname}_{kind}_variants.csv", index=False)
    pd.concat(ratios, axis=0).to_csv(f"{OUT}/{groupname}_{kind}_variants_ratios.csv")
    return df, ratios


def soc_test(names, model):
    """SOC (Brewer1,2): seed vs wide bounds, all peaks kept (SOC peak 6 is ~290.4 eV, not 291.5)."""
    rows = []
    for kind in ("lariat", "isld"):
        for wide in (False, True):
            tag = f"SOC_{kind}_{'wide' if wide else 'seed'}"
            cf, sp, qual, rat = run(names, kind, model, tag, wide=wide, verbose=True)
            rows.append(dict(variant=tag, RSS=cf.res["rss"], k=cf.res["k"], at_bound=cf.res["n_at_bound"], AIC=cf.res["aic"], BIC=cf.res["bic"]))
    pd.DataFrame(rows).to_csv(f"{OUT}/SOC_variants.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["soc"]:
        import nexafs_models as M
        soc_test([n for n, v in B.REFS.items() if v[3] == "SOC"], M.SOC_CARBON_UV)
    elif sys.argv[1:] == ["sog_wide"]:
        names = [n for n, v in B.REFS.items() if v[3] == "SOG"]
        import nexafs_models as M
        for kind in ("lariat", "isld"):
            for drop in ((), (6,)):
                run(names, kind, M.SOG_CARBON_BAKE, f"SOG_{kind}_wide_{'no6' if drop else 'with6'}", wide=True, drop=drop, verbose=False)
    elif sys.argv[1:] == ["sog_variants"]:
        names = [n for n, v in B.REFS.items() if v[3] == "SOG"]
        import nexafs_models as M
        for kind in ("lariat", "isld"):
            df, rat = variants(names, kind, M.SOG_CARBON_BAKE)
            print(df.round(3).T.to_string())
    if sys.argv[1:] == ["sog"]:
        names = [n for n, v in B.REFS.items() if v[3] == "SOG"]
        import nexafs_models as M
        for kind in ("lariat", "isld"):
            run(names, kind, M.SOG_CARBON_BAKE, f"SOG_{kind}")
