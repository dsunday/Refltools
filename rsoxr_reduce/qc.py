"""
qc.py
-----
Quality flags on stitched curves.  An energy is held for review only when a
problem could change the curve materially; noisy-but-consistent junctions
are reported as info.

Per junction (median-ratio stitch of section k onto sections < k):
    scale, n (overlap ratios), sigma = 1.25·1.48·MAD/median/√n (rel.),
    drift = change of the ratio between the lower and upper half of the
    overlap (shape disagreement: CEM non-linearity, misalignment).

Flags
    NO_OVERLAP     (review) no overlap ratios -> section unscaled; the scale
                   from neighbouring energies is proposed / used
    SCALE_OUTLIER  (review) |log(scale/trend)| > max(k·σ_tot, 15 %), trend =
                   running median over neighbouring energies of the same
                   junction (the PD/CEM sensitivity ratio varies smoothly with
                   energy)
    JUMP           (review) drift > qc.jump and > 3× its own noise
    NOISY_OVERLAP  (info)   sigma > qc.noisy_overlap but consistent with trend
    LOW_OVERLAP    (info)   n < min_overlap but consistent with trend
    NEG_AFTER_BG   (info)   points ≤ 0 after background subtraction, dropped
"""

import numpy as np


def junction_stats(pe, res):
    secs = pe["sections"]
    out = []
    for i, info in enumerate(res["sections"]):
        if i == 0:
            continue
        r = np.asarray(info["ratios"], float)
        st = {"pair": f"{secs[i - 1]['id']}->{secs[i]['id']}", "index": i,
              "scale": float(info["scale"]), "n": int(len(r)),
              "override": bool(info.get("override"))}
        if len(r) >= 2:
            med = np.median(r)
            mad = 1.4826 * np.median(np.abs(r - med))
            st["dispersion"] = float(mad / med)
            st["sigma"] = float(1.2533 * mad / med / np.sqrt(len(r)))
        if len(r) >= 6:
            h = len(r) // 2
            a, b = np.median(r[:h]), np.median(r[h:])
            st["drift"] = float(b / a - 1)
            st["drift_sigma"] = float(np.sqrt(2) * 1.2533 * st["dispersion"] / np.sqrt(h))
        out.append(st)
    return out


def energy_flags(pe, res, stats, settings):
    """Flags that need only this energy."""
    q, t = settings["qc"], settings["trim"]
    flags = []
    for i, info in enumerate(res["sections"]):
        if info["n_nonpositive"]:
            flags.append({"code": "NEG_AFTER_BG", "severity": "info",
                          "detail": f"{pe['sections'][i]['id']}: {info['n_nonpositive']} "
                                    f"point(s) ≤ 0 after background, dropped"})
    for st in stats:
        if st["n"] == 0 and not st["override"]:
            flags.append({"code": "NO_OVERLAP", "severity": "review",
                          "detail": f"{st['pair']}: no overlap, section not scaled"})
            continue
        if st["n"] < t["min_overlap"]:
            flags.append({"code": "LOW_OVERLAP", "severity": "info",
                          "detail": f"{st['pair']}: {st['n']} overlap ratio(s)"})
        if st.get("sigma", 0) > q["noisy_overlap"]:
            flags.append({"code": "NOISY_OVERLAP", "severity": "info",
                          "detail": f"{st['pair']}: scale ±{st['sigma']:.0%} "
                                    f"(ratio scatter {st['dispersion']:.0%}, n={st['n']})"})
        d = st.get("drift")
        if d is not None and abs(d) > q["jump"] and abs(d) > 3 * st["drift_sigma"]:
            flags.append({"code": "JUMP", "severity": "review",
                          "detail": f"{st['pair']}: overlap ratio changes {d:+.0%} "
                                    f"across the overlap (±{st['drift_sigma']:.0%})"})
    return flags


def trend(energies, values, k, window, max_de=None):
    """Robust local line through values[k]'s neighbours (excluding k),
    evaluated at energies[k].  Theil–Sen slope, median intercept.
    -> (prediction, residual scatter) or (None, None)."""
    lo, hi = max(0, k - window), min(len(values), k + window + 1)
    idx = [j for j in range(lo, hi) if j != k and np.isfinite(values[j])
           and (max_de is None or abs(energies[j] - energies[k]) <= max_de)]
    if len(idx) < 3:
        return None, None
    x = np.array([energies[j] for j in idx], float)
    y = np.array([values[j] for j in idx], float)
    e = energies[k]
    two_sided = (x < e).any() and (x > e).any()
    sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))
          if x[j] != x[i]]
    # never extrapolate a slope past the end of the energy range
    b = float(np.median(sl)) if (sl and two_sided) else 0.0
    a = float(np.median(y - b * x))
    res = y - (a + b * x)
    mad = float(1.4826 * np.median(np.abs(res - np.median(res))))
    return a + b * energies[k], mad


def trend_flags(per_energy, settings):
    """per_energy: list of (energy, stats) sorted by energy.

    Returns {energy: [flags]} and {energy: {pair: trend_scale}}.
    """
    q = settings["qc"]
    window = q["trend_window"]
    flags, trends = {}, {}
    pairs = sorted({s["pair"] for _, sts in per_energy for s in sts})
    for pair in pairs:
        pts = [(E, s) for E, sts in per_energy for s in sts if s["pair"] == pair]
        pts.sort(key=lambda p: p[0])
        # gain-normalised scale (PD gain is in decades)
        L = [np.log(s["scale_n"]) if (s["n"] > 0 and s["scale"] > 0) else np.nan
             for _, s in pts]
        for k, (E, s) in enumerate(pts):
            med, mad = trend([p[0] for p in pts], L, k, window, q["trend_max_de"])
            if med is not None:
                s["trend_sigma"] = float(max(mad, settings["qc"]["trend_sigma_min"]))
            if med is None:
                continue
            gf = 10.0 ** s["gain"] if s.get("gain") is not None else 1.0
            trends.setdefault(E, {})[pair] = float(np.exp(med) * gf)
            s["trend"] = float(np.exp(med) * gf)
            if not np.isfinite(L[k]):
                continue
            dev = L[k] - med
            s["trend_dev"] = float(np.exp(dev) - 1)
            tol = max(q["scale_outlier_mad"] * np.hypot(s.get("sigma", 0.0),
                                                        max(mad, q["trend_sigma_min"])),
                      np.log(1 + q["scale_outlier_min"]))
            if abs(dev) > tol:
                flags.setdefault(E, []).append({
                    "code": "SCALE_OUTLIER", "severity": "review",
                    "detail": f"{pair}: scale {s['scale']:.3g} is {np.exp(dev) - 1:+.0%} from "
                              f"neighbouring energies ({s['trend']:.3g} at the same PD gain; own ±"
                              f"{s.get('sigma', 0):.0%}, trend scatter {mad:.0%})"})
    return flags, trends


def gap_bridge_scale(prev, cur, m=4, max_gap=0.5):
    """Scale for ``cur`` (θ, y) onto ``prev`` (θ, y) when they do not overlap.

    Linear fits of log y vs θ over the last/first ``m`` positive points are
    extrapolated to the gap midpoint and matched.  -> (scale, gap_deg) or
    (None, gap_deg) if the gap is too wide or there are too few points.
    """
    p = prev[prev[:, 1] > 0]
    c = cur[cur[:, 1] > 0]
    if len(p) < m or len(c) < m:
        return None, None
    p = p[np.argsort(p[:, 0])][-m:]
    c = c[np.argsort(c[:, 0])][:m]
    gap = float(c[0, 0] - p[-1, 0])
    if gap > max_gap:
        return None, gap
    mid = 0.5 * (c[0, 0] + p[-1, 0])
    fp = np.polyfit(p[:, 0], np.log(p[:, 1]), 1)
    fc = np.polyfit(c[:, 0], np.log(c[:, 1]), 1)
    return float(np.exp(np.polyval(fp, mid) - np.polyval(fc, mid))), gap


def nearest_scale(per_energy, energy, pair, k=3):
    """Median scale of the k nearest energies whose junction had overlap."""
    pts = [(abs(E - energy), s["scale_n"]) for E, sts in per_energy for s in sts
           if s["pair"] == pair and s["n"] > 0 and not s["override"] and E != energy]
    if not pts:
        return None, []
    pts.sort()
    use = pts[:k]
    return float(np.median([p[1] for p in use])), [p[0] for p in use]


# ----------------------------------------------------------------------
# filter / aperture changes between sections
# ----------------------------------------------------------------------
def _contrast(theta, logy, var_noise, deg=3):
    """Fringe contrast: rms of log R about a smooth polynomial, with the
    counting-noise variance removed.  -> (contrast, residuals)"""
    if len(theta) < deg + 4:
        return None, None
    x = (theta - theta.mean()) / max(np.ptp(theta), 1e-9)
    res = logy - np.polyval(np.polyfit(x, logy, deg), x)
    v = np.var(res) - np.mean(var_noise)
    return float(np.sqrt(max(v, 0.0))), res


def filter_checks(pe, res, stats, i0, settings):
    """FILTER_CHANGE between adjacent sections, and the fringe comparison in
    their overlap.  Opening/removing the aperture has in practice damped the
    fringes (over-exposure / larger footprint), which also biases the stitch.

    FILTER_CHANGE   (review)  filter/aperture differs between adjacent sections
    FRINGE_DAMPING  (review)  fringe contrast of the upper section is below
                              (1 - qc.fringe_damping) × the lower one's
    FRINGES_MATCH   (info)    contrast and phase agree: the change looks harmless
    """
    q = settings["qc"]
    secs = pe["sections"]
    flags = []
    for i in range(1, len(secs)):
        a, b = secs[i - 1], secs[i]
        fa, fb = a.get("filter"), b.get("filter")
        if not fa or not fb or fa == fb:
            continue
        pair = f"{a['id']}->{b['id']}"
        flags.append({"code": "FILTER_CHANGE", "severity": "review",
                      "detail": f"{pair}: '{fa}' -> '{fb}'; check fringe damping and the "
                                f"stitch in the overlap"})
        lo, hi = res["sections"][i - 1], res["sections"][i]
        t0, t1 = max(lo["theta"].min(), hi["theta"].min()), min(lo["theta"].max(), hi["theta"].max())
        ml = (lo["theta"] >= t0) & (lo["theta"] <= t1) & (lo["y"] > 0)
        mh = (hi["theta"] >= t0) & (hi["theta"] <= t1) & (hi["y"] > 0)
        if ml.sum() < 8 or mh.sum() < 8:
            # fallback: fringe amplitude in adjacent windows either side of the
            # junction (weaker test: R itself changes across the windows)
            w = q["fringe_window_deg"]
            wl = (lo["theta"] >= lo["theta"].max() - w) & (lo["y"] > 0)
            wh = (hi["theta"] <= hi["theta"].min() + w) & (hi["y"] > 0)
            if wl.sum() < 8 or wh.sum() < 8:
                flags.append({"code": "FRINGE_CHECK_SKIPPED", "severity": "info",
                              "detail": f"{pair}: too few points to compare fringes"})
                continue

            def rms(th, y):
                x_ = (th - th.mean()) / max(np.ptp(th), 1e-9)
                v = np.log(y)
                r_ = v - np.polyval(np.polyfit(x_, v, 2), x_)
                return float(np.std(np.convolve(np.pad(r_, 1, mode="edge"),
                                                np.ones(3) / 3, mode="valid")))
            cl, ch = rms(lo["theta"][wl], lo["y"][wl]), rms(hi["theta"][wh], hi["y"][wh])
            ratio = ch / cl if cl > 0 else np.inf
            det = (f"{pair}: overlap too short ({mh.sum()} pts); fringe rms in {w:g}° windows "
                   f"either side {ch:.3f} ({b['id']}) vs {cl:.3f} ({a['id']}), ratio {ratio:.2f}")
            if cl >= q["fringe_min_contrast"] and ratio < 1 - q["fringe_damping"]:
                flags.append({"code": "FRINGE_DAMPING", "severity": "review",
                              "detail": det + " — possible damping (adjacent-window test)"})
            else:
                flags.append({"code": "FRINGE_CHECK_WEAK", "severity": "info", "detail": det})
            continue

        # common grid = the coarser section's points inside the overlap; detrend
        # log R with a cubic, smooth lightly (fringes span many points, noise does not)
        coarse = lo if ml.sum() <= mh.sum() else hi
        mc = ml if coarse is lo else mh
        grid = coarse["theta"][mc]
        ll = np.interp(grid, lo["theta"][ml], np.log(lo["y"][ml]))
        lh = np.interp(grid, hi["theta"][mh], np.log(hi["y"][mh]))
        x = (grid - grid.mean()) / max(np.ptp(grid), 1e-9)

        def resid(v):
            r_ = v - np.polyval(np.polyfit(x, v, 3), x)
            k_ = np.ones(3) / 3
            return np.convolve(np.pad(r_, 1, mode="edge"), k_, mode="valid")

        rl, rh = resid(ll), resid(lh)
        cl, ch = float(np.std(rl)), float(np.std(rh))
        if cl < q["fringe_min_contrast"] and ch < q["fringe_min_contrast"]:
            flags.append({"code": "FRINGE_CHECK_SKIPPED", "severity": "info",
                          "detail": f"{pair}: no measurable fringes in the overlap "
                                    f"(rms {cl:.3f}/{ch:.3f})"})
            continue
        corr = float(np.corrcoef(rl, rh)[0, 1]) if cl > 0 and ch > 0 else 0.0
        # damping = fringe amplitude of the upper section relative to the lower,
        # from the regression through the origin (noise in either lowers corr)
        ratio = float(np.dot(rl, rh) / np.dot(rl, rl)) if cl > 0 else np.inf
        for st in stats:
            if st["pair"] == pair:
                st.update(fringe_ratio=ratio, fringe_corr=corr, fringe_rms=(cl, ch))
        det = (f"{pair}: fringe amplitude {b['id']}/{a['id']} = {ratio:.2f} "
               f"(rms {ch:.3f} vs {cl:.3f} in log R), correlation {corr:.2f}")
        if corr >= q["fringe_min_corr"] and ratio < 1 - q["fringe_damping"]:
            flags.append({"code": "FRINGE_DAMPING", "severity": "review",
                          "detail": det + f" — {b['id']} fringes damped after the filter change"})
        elif corr < q["fringe_min_corr"]:
            flags.append({"code": "FRINGE_MISMATCH", "severity": "review",
                          "detail": det + " — fringes do not line up / too noisy to compare"})
        else:
            flags.append({"code": "FRINGES_MATCH", "severity": "info", "detail": det})
    return flags
