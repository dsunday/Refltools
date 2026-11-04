"""O K-edge NEXAFS peak fits for the Brewer samples (LARIAT surface spectra): independent fits and a
joint co-fit with shared peak energies/widths (same idea as nexafs_cofit.py, which is carbon-specific).

Shared across samples (free within bounds): peak energies and widths.
Per sample: peak heights, edge height and decay; edge location/width either shared ("share_edge") or per sample.
Uncertainties: (J^T J)^-1 * RSS/dof, delta method for areas. LARIAT errors are statistical only.

Outputs under NEXAFS_Fits/Oxygen/.  Run via NEXAFS_Fits/Oxygen/scripts/*.py.
"""
import json
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
import nexafs_models as M
from NEXAFS_BuildingBlock import gaussian, edge_bl_func

G = B.GAUSS_AREA
OUT = f"{B.OUT}/Oxygen"
FIT_RANGE = (526.0, 565.0)  # residual climbs above ~566 eV (post-edge normalisation artefact >575 eV)

SOC_SAMPLES = {
    "Brewer1_SOC_UV": f"{B.FALL}/Brewer1_SOC_UV_Oxygen_processed.csv",
    "Brewer2_SOC_Bake": f"{B.FALL}/Brewer2_SOC_Bake_Oxygen_processed.csv",
}


def load_lariat(path, rng=FIT_RANGE):
    E, y = B.load_source("lariat", path)
    m = (E >= rng[0]) & (E <= rng[1])
    return E[m], y[m]


def independent(name, path, n_peaks, rng=FIT_RANGE, model_fn=None, **model_kw):
    """Independent wide-bounds fit; returns (fit, E, y). model_kw -> model_fn (default nexafs_models.soc_oxygen; SOG: nexafs_models.sog_oxygen)."""
    E, y = load_lariat(path, rng)
    fit = B.fit_one(E, y, (model_fn or M.soc_oxygen)(n_peaks, **model_kw), energy_range=rng)
    return fit, E, y


def rss_aic(res, n_pts, k):
    return float(np.sum(res ** 2)), n_pts * np.log(np.sum(res ** 2) / n_pts) + 2 * k


class OCoFit:
    """th = [E(n), W(n), H(S*n), edgeH(S), edgeDecay(S), edgeLoc(S|1), edgeWid(S|1)]."""

    def __init__(self, names, n_peaks, share_edge=False, rng=FIT_RANGE, paths=None, ordered=False, dmin=0.1, model_fn=None, **model_kw):
        """n_peaks = seed set (5-8); model_kw -> nexafs_models.soc_oxygen (merge56, wide_caps); self.n = peaks actually fitted."""
        self.names, self.S, self.share = list(names), len(names), share_edge
        self.ordered, self.dmin = ordered, dmin  # ordered: energies = E1 + cumulative gaps (>= dmin), so peaks can never cross their neighbours
        paths = paths or SOC_SAMPLES
        self.data = [dict(zip(("E", "y"), load_lariat(paths[nm], rng))) for nm in names]
        self.m = (model_fn or M.soc_oxygen)(n_peaks, **model_kw)
        n = self.n = len([k for k in self.m["peaks"] if k.endswith("_energy")])
        S = self.S
        pb = self.m["peaks"]
        ed = self.m["edge"]
        k = 1 if share_edge else S
        self.lo = np.r_[[pb[f"peak_{i}_energy_bounds"][0] for i in range(1, n + 1)],
                        [pb[f"peak_{i}_width_bounds"][0] for i in range(1, n + 1)],
                        np.zeros(S * n), np.full(S, ed["height_bounds"][0]), np.full(S, ed["decay_bounds"][0]),
                        np.full(k, ed["location_bounds"][0]), np.full(k, ed["width_bounds"][0])]
        self.hi = np.r_[[pb[f"peak_{i}_energy_bounds"][1] for i in range(1, n + 1)],
                        [pb[f"peak_{i}_width_bounds"][1] for i in range(1, n + 1)],
                        np.full(S * n, 4.0), np.full(S, ed["height_bounds"][1]), np.full(S, ed["decay_bounds"][1]),
                        np.full(k, ed["location_bounds"][1]), np.full(k, ed["width_bounds"][1])]
        self.e_lo = np.array([pb[f"peak_{i}_energy_bounds"][0] for i in range(1, n + 1)])  # per-peak energy windows (kept in ordered mode as a stiff hinge)
        self.e_hi = np.array([pb[f"peak_{i}_energy_bounds"][1] for i in range(1, n + 1)])
        self.box_lam = 100.0
        if ordered:  # th[0] = E1 (seed bounds), th[1:n] = gaps to the previous peak
            self.lo[1:n], self.hi[1:n] = dmin, 40.0
        self.labels = ([f"E{i}" for i in range(1, n + 1)] + [f"W{i}" for i in range(1, n + 1)]
                       + [f"H{s}_{i}" for s in range(S) for i in range(1, n + 1)] + [f"edgeH{s}" for s in range(S)]
                       + [f"edgeDecay{s}" for s in range(S)] + [f"edgeLoc{s}" for s in range(k)] + [f"edgeWid{s}" for s in range(k)])

    def unpack(self, th):
        n, S = self.n, self.S
        k = 1 if self.share else S
        o = 2 * n
        H = th[o:o + S * n].reshape(S, n); o += S * n
        eh = th[o:o + S]; o += S
        ed = th[o:o + S]; o += S
        el = th[o:o + k]; o += k
        ew = th[o:o + k]
        if self.share:
            el, ew = np.full(S, el[0]), np.full(S, ew[0])
        e = np.cumsum(th[:n]) if self.ordered else th[:n]
        return e, th[n:2 * n], H, eh, ed, el, ew

    def curve(self, th, s, E):
        e, w, h, eh, ed, el, ew = self.unpack(th)
        comps = [gaussian(E, h[s, i], e[i], w[i]) for i in range(self.n)]
        edge = edge_bl_func(E, el[s], eh[s], ew[s], ed[s])
        return edge + np.sum(comps, axis=0), comps, edge

    def resid_data(self, th):
        return np.concatenate([d["y"] - self.curve(th, s, d["E"])[0] for s, d in enumerate(self.data)])

    def resid(self, th):
        r = self.resid_data(th)
        if not self.ordered:
            return r
        e = self.unpack(th)[0]   # hinge penalty keeps every peak inside its own energy window (the cumulative-gap parameters carry no per-peak box)
        pen = self.box_lam * (np.maximum(self.e_lo - e, 0) + np.maximum(e - self.e_hi, 0))
        return np.r_[r, pen]

    def start(self, indep):
        """indep: list of fits from `independent` (same n_peaks) -> mean energies/widths, own heights/edge."""
        n, S = self.n, self.S
        E, W, H, EH, ED, EL, EW = np.zeros(n), np.zeros(n), np.zeros((S, n)), np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S)
        for s, f in enumerate(indep):
            pk = f["peaks"]
            E += pk["energy"].to_numpy() / S
            W += pk["width"].to_numpy() / S
            H[s] = pk["height"].to_numpy()
            ep = f["results"]["fitted_edge_params"]
            EH[s], ED[s], EL[s], EW[s] = ep["height"], ep["decay"], ep["location"], ep["width"]
        # order peaks by energy so the mean is over matching components (independent fits keep seed order)
        el, ew = (np.array([EL.mean()]), np.array([EW.mean()])) if self.share else (EL, EW)
        th = np.r_[E, W, H.ravel(), EH, ED, el, ew]
        return np.clip(th, self.lo + 1e-9, self.hi - 1e-9)

    def th_ordered_from(self, other):
        """Starting vector for an ordered fit from a fitted unordered OCoFit `other` (same peaks/samples): peaks sorted by energy, energies -> E1 + gaps."""
        e, w, h, eh, ed, el, ew = other.unpack(other.res["th"])
        o = np.argsort(e); e, w, h = e[o], w[o], h[:, o]
        gaps = np.r_[e[0], np.maximum(np.diff(e), self.dmin + 1e-6)]
        k = 1 if self.share else self.S
        th = np.r_[gaps, w, h.ravel(), eh, ed, (el[:1] if self.share else el), (ew[:1] if self.share else ew)]
        return np.clip(th, self.lo + 1e-9, self.hi - 1e-9)

    def fit(self, th0):
        r = least_squares(self.resid, th0, bounds=(self.lo, self.hi), method="trf", x_scale="jac", max_nfev=5000)
        n_pts = sum(len(d["y"]) for d in self.data)
        dof = max(n_pts - len(r.x), 1)
        rss = float(np.sum(self.resid_data(r.x) ** 2))
        cov = np.linalg.pinv(r.jac.T @ r.jac) * rss / dof
        tol = 1e-4 * (self.hi - self.lo)
        ab = (r.x - self.lo < tol) | (self.hi - r.x < tol)
        k = len(r.x)
        if self.ordered:   # a peak pressed against its own energy window shows up as at-bound
            e = self.unpack(r.x)[0]; ab[:self.n] = ab[:self.n] | np.r_[False, np.zeros(self.n - 1, bool)]
            self.win_bound = (e - self.e_lo < 1e-2) | (self.e_hi - e < 1e-2)
        self.res = dict(th=r.x, cov=cov, rss=rss, dof=dof, n_pts=n_pts, success=bool(r.success), nfev=int(r.nfev), at_bound=ab,
                        k=k, aic=n_pts * np.log(rss / n_pts) + 2 * k, bic=n_pts * np.log(rss / n_pts) + k * np.log(n_pts),
                        n_at_bound=int(ab.sum()))
        return self.res

    def propagate(self, fn, eps=1e-6):
        th, cov = self.res["th"], self.res["cov"]
        v0 = np.atleast_1d(fn(th))
        g = np.zeros((len(v0), len(th)))
        for k in range(len(th)):
            d = np.zeros_like(th)
            d[k] = eps * max(1.0, abs(th[k]))
            g[:, k] = (np.atleast_1d(fn(th + d)) - np.atleast_1d(fn(th - d))) / (2 * d[k])
        return v0, np.sqrt(np.maximum(np.einsum("ik,kl,il->i", g, cov, g), 0))

    def area_fn(self, th):
        e, w, h, *_ = self.unpack(th)
        return (h * w[None, :] * G).ravel()  # (S*n,)

    def tables(self):
        th = self.res["th"]
        e, w, h, eh, ed, el, ew = self.unpack(th)
        err = np.sqrt(np.diag(self.res["cov"]))
        n = self.n
        ab = self.res["at_bound"]
        if self.ordered:
            _, err_e = self.propagate(lambda x: self.unpack(x)[0])
            ab_e = ab[:n] | self.win_bound  # gap at its minimum (against its neighbour) or pressed on its own energy window
        else:
            err_e, ab_e = err[:n], ab[:n]
        sp = pd.DataFrame(dict(peak=range(1, n + 1), energy=e, energy_err=err_e, energy_at_bound=ab_e,
                               width=w, width_err=err[n:2 * n], width_at_bound=ab[n:2 * n]))
        v, ve = self.propagate(self.area_fn)
        rows = []
        for s, nm in enumerate(self.names):
            for i in range(n):
                rows.append(dict(sample=nm, peak=i + 1, energy=e[i], height=h[s, i], area=v[s * n + i], area_err=ve[s * n + i]))
        edge = pd.DataFrame(dict(sample=self.names, edge_height=eh, edge_decay=ed, edge_loc=el, edge_width=ew))
        return sp, pd.DataFrame(rows), edge


def plot_cofit(cf, tag, path, title=None):
    S, n = cf.S, cf.n
    th = cf.res["th"]
    fig, axs = plt.subplots(2, S, figsize=(7.5 * S, 8), gridspec_kw=dict(height_ratios=[3, 1]), squeeze=False)
    for s, nm in enumerate(cf.names):
        d = cf.data[s]
        tot, comps, edge = cf.curve(th, s, d["E"])
        ax, ar = axs[0, s], axs[1, s]
        ax.plot(d["E"], d["y"], "k.", ms=3, label="LARIAT O K-edge")
        ax.plot(d["E"], tot, "r", label="co-fit")
        ax.plot(d["E"], edge, "g--", label="edge")
        for i, g in enumerate(comps, 1):
            ax.plot(d["E"], g, lw=1, label=f"p{i} {cf.unpack(th)[0][i - 1]:.2f} eV")
        ax.set_title(f"{nm}  [{tag}]"); ax.legend(ncol=2, fontsize=7); ax.set_ylabel("Processed intensity")
        ar.plot(d["E"], d["y"] - tot, "k-", lw=0.8); ar.axhline(0, c="r"); ar.set_xlabel("Energy (eV)"); ar.set_ylabel("resid")
    if title:
        fig.suptitle(title)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close(fig)


def model_json(cf, s, path, fit_range=FIT_RANGE):
    """Dump one sample's co-fit as a plain dict (peak energies/widths/heights, edge) for later stages."""
    th = cf.res["th"]
    e, w, h, eh, ed, el, ew = cf.unpack(th)
    d = dict(sample=cf.names[s], fit_range=list(fit_range), peaks=[dict(peak=i + 1, energy=float(e[i]), width=float(w[i]),
             height=float(h[s, i])) for i in range(cf.n)], edge=dict(location=float(el[s]), height=float(eh[s]),
             width=float(ew[s]), decay=float(ed[s])), rss=cf.res["rss"], n_pts=cf.res["n_pts"])
    json.dump(d, open(path, "w"), indent=1)
    return d
