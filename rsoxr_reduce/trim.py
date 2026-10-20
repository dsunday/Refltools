"""
trim.py
-------
Rule-based trims, each returned with a human-readable reason.

Rules (all thresholds in settings['trim']):

PD start – beam clipping (per folder)
    The low-angle PD signal rises while the beam footprint overfills the
    sample, peaks, then follows R(θ).  Clipping is geometric, so one angle is
    used for the whole folder: the median over energies of the angle of the
    PD maximum below ``clip_search_deg``.  Each file starts at its first point
    at/above that angle.  Calibrated on Nov 2025: matches the manual start
    within one grid step in 82 % of carbon folders (oxygen manual choices
    scatter ±0.17°).  An energy whose own peak is > ``clip_shift_steps`` steps
    away is flagged CLIP_SHIFT.

PD background – dark offset (per folder)
    Where PD and CT1 overlap, PD_raw = bg + k·CT_rate.  The intercept of a
    robust regression is the PD dark offset, its residual scatter the PD noise
    σ.  The folder value is the median over well-conditioned energies
    (bg is a detector/session property).  Falls back to ``bg_default``.

Noise model
    PD: relative error σ / S(θ), S = smoothed (PD − bg).
    CEM: values are rates, counts = rate × count time, rel. error 1/√counts.

Section end at a junction (PD→CT1, CTk→CTk+1)
    Keep the lower section into the overlap only until the stitching scale
    factor is determined to ``scale_rel_target`` (σ_scale ≈ 1.25·rms(rel)/√n,
    median-of-ratios) with at least ``min_overlap`` points – beyond that the
    next (better-counting) section takes over – and never past the point
    where the lower section itself gets too noisy (smoothed SNR <
    ``snr_min`` for PD, counts < ``ct_min_counts`` for CEM).

PD floor points (PD_FLOOR_POINTS)
    When the PD section is kept past its SNR floor so the next section has
    ``min_overlap`` points (LOW_OVERLAP), single readings there can sit at the
    dark offset and become ~0 after background subtraction (Brewer2 carbon
    289.5 eV: R ×0.01 of its neighbours).  In that stretch (all points still
    enter the stitch scale):
      * points at/above the next section's first angle are not written – the
        next section covers them with better statistics;
      * below it (PD is the only data) a point is not written when
        (PD − bg) < ``floor_outlier_ratio`` × the 5-point running median of
        (PD − bg) – isolated dropouts only, so the kept noise stays unbiased
        (a threshold in σ would keep only the upward fluctuations).

CT start – saturation
    CEM rates above ``ct_max_rate`` are non-linear (Nov 2025: PD/CT ratio
    +3 % at 1–3e5 cps, +22 % at 3e5–1e6 cps); leading points above it are
    trimmed (CT_SATURATION).

Last section tail
    Trim trailing points whose smoothed counts < ``ct_min_counts`` (PD: SNR <
    ``snr_min``); the final point is always dropped (legacy convention).
"""

import numpy as np

from .settings import edge_value


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def smooth(y, w=5):
    """Running median (w) followed by running mean (3); same length."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return y.copy()
    h = w // 2
    med = np.array([np.median(y[max(0, i - h):i + h + 1]) for i in range(n)])
    k = np.ones(3) / 3
    pad = np.concatenate([[med[0]], med, [med[-1]]])
    return np.convolve(pad, k, mode="valid")


def legacy_end(last_idx, n):
    """Index of the last kept point -> legacy ``end`` value."""
    return last_idx + 1 - n if last_idx < n - 1 else n


def last_kept(n, end):
    return (n + end - 1) if end < 0 else (end - 1)


def step_of(theta):
    d = np.diff(np.sort(theta))
    d = d[d > 1e-6]
    return float(np.median(d)) if len(d) else 0.0


# ----------------------------------------------------------------------
# PD start (clipping)
# ----------------------------------------------------------------------
def pd_saturated(data, settings):
    """Indices of PD readings at the ADC limit."""
    return np.where(data[:, 1] >= settings["trim"]["pd_sat_level"])[0]


def pd_peak(data, edge, settings):
    lim = edge_value(settings["trim"]["clip_search_deg"], edge)
    if len(pd_saturated(data, settings)):
        return None, None            # the peak is hidden in the saturated plateau
    m = data[:, 0] <= lim
    if m.sum() < 3:
        return None, None
    idx = np.where(m)[0]
    ip = idx[int(np.argmax(data[m, 1]))]
    return float(data[ip, 0]), int(ip)


def folder_clip_angle(pd_arrays, edge, settings):
    """pd_arrays: {key: data}. -> (angle, {key: peak_angle})"""
    peaks = {}
    for k, d in pd_arrays.items():
        a, _ = pd_peak(d, edge, settings)
        if a is not None:
            peaks[k] = a
    if not peaks:
        return None, peaks
    return float(np.median(list(peaks.values()))), peaks


def pd_start(data, clip_angle, own_peak, settings):
    """-> (start_idx, reason, flags)"""
    t = settings["trim"]
    st = step_of(data[:, 0])
    flags = []
    sat = pd_saturated(data, settings)
    if len(sat):
        # saturation only happens at the bright low-angle end; start after it
        s_last = int(sat.max())
        lost = clip_angle is not None and data[s_last, 0] > clip_angle + 0.5 * st
        flags.append({"code": "PD_SATURATION", "severity": "review" if lost else "info",
                      "detail": f"PD at the ADC limit ({t['pd_sat_level']:g} V) for {len(sat)} "
                                f"point(s) up to {data[s_last, 0]:.2f}°"
                                + ("; data above the clip angle lost (lower the PD gain)"
                                   if lost else "")})
        idx = max(s_last + 1, t["min_start"])
        if clip_angle is not None:
            idx = max(idx, int(np.argmax(data[:, 0] >= clip_angle - 0.5 * st)))
        return idx, (f"PD saturated to {data[s_last, 0]:.2f}°: start after the plateau "
                     f"-> index {idx} ({data[idx, 0]:.3f}°)"), flags
    if clip_angle is None:
        return t["min_start"], "no clip peak found; dropped θ≈0 point only", [
            {"code": "NO_CLIP_PEAK", "severity": "review",
             "detail": "no PD maximum below the clip search angle"}]
    idx = int(np.argmax(data[:, 0] >= clip_angle - 0.5 * st))
    idx = max(idx, t["min_start"])
    if own_peak is not None and st > 0 and abs(own_peak - clip_angle) > t["clip_shift_steps"] * st:
        flags.append({"code": "CLIP_SHIFT", "severity": "info",
                      "detail": f"PD peak at {own_peak:.3f}° vs folder clip angle "
                                f"{clip_angle:.3f}°"})
    return idx, (f"beam clipping: start at folder clip angle {clip_angle:.3f}° "
                 f"(median PD low-angle peak) -> index {idx} ({data[idx, 0]:.3f}°)"), flags


# ----------------------------------------------------------------------
# PD background
# ----------------------------------------------------------------------
def fit_pd_background(pd, ct, count_time, settings):
    """Robust PD = bg + k·CT fit over the overlap. -> dict or None."""
    t = settings["trim"]
    lo, hi = ct[:, 0].min(), ct[:, 0].max()
    m = (pd[:, 0] >= lo) & (pd[:, 0] <= hi)
    if m.sum() < 6:
        return None
    th, y = pd[m, 0], pd[m, 1]
    x = np.interp(th, ct[:, 0], ct[:, 1])
    ok = (x * (count_time or 1.0) >= t["bg_fit_min_counts"]) & (x <= t["ct_max_rate"])
    if ok.sum() < 6:
        return None
    th, y, x = th[ok], y[ok], x[ok]
    w = np.ones(len(x), bool)
    bg = k = s = None
    for _ in range(6):
        if w.sum() < 5:
            return None
        A = np.vstack([np.ones(w.sum()), x[w]]).T
        (bg, k), *_ = np.linalg.lstsq(A, y[w], rcond=None)
        res = y - (bg + k * x)
        s = 1.4826 * np.median(np.abs(res[w] - np.median(res[w])))
        w = np.abs(res) < 3 * max(s, 1e-5)
    if k is None or k <= 0:
        return None
    span = (x[w].max() - x[w].min()) * k
    return {"bg": float(bg), "sigma": float(s), "k": float(k), "n": int(w.sum()),
            "signal_span": float(span)}


def folder_background(fits, edge, settings):
    """Combine per-energy fits -> dict(bg, sigma, source, n)."""
    t = settings["trim"]
    good = [f for f in fits if f and f["signal_span"] > 5 * f["sigma"]
            and f["sigma"] < t["bg_fit_max_sigma"]]
    if len(good) >= t["bg_fit_min_energies"]:
        bg = float(np.median([f["bg"] for f in good]))
        sig = float(np.median([f["sigma"] for f in good]))
        spread = float(1.4826 * np.median(np.abs([f["bg"] - bg for f in good])))
        return {"bg": bg, "sigma": sig, "source": "pd-vs-ct regression",
                "n": len(good), "spread": spread}
    bg = edge_value(t["bg_default"], edge)
    return {"bg": bg, "sigma": t["pd_sigma_default"], "source": "default (too few "
            f"well-conditioned PD/CT overlaps: {len(good)})", "n": len(good), "spread": None}


# ----------------------------------------------------------------------
# noise models
# ----------------------------------------------------------------------
def rel_err(kind, theta, y, bg=0.0, sigma=None, count_time=None):
    """Relative error per point (smoothed signal)."""
    if kind == "pd":
        s = smooth(y - bg)
        return sigma / np.maximum(s, 1e-12), s
    counts = smooth(y) * (count_time or 1.0)
    return 1.0 / np.sqrt(np.maximum(counts, 1.0)), counts


def _ok_mask(kind, s, sigma, settings):
    t = settings["trim"]
    if kind == "pd":
        return s >= t["snr_min"] * sigma
    return s >= t["ct_min_counts"]


# ----------------------------------------------------------------------
# section ends
# ----------------------------------------------------------------------
def junction_end(lo, hi, start_lo, start_hi, settings):
    """End of the lower section ``lo`` where it meets ``hi``.

    lo/hi: dicts(data, kind 'pd'|'cem', bg, sigma, count_time).
    Returns (last_idx, reason, flags, stats).
    """
    t = settings["trim"]
    dl, dh = lo["data"], hi["data"]
    n = len(dl)
    rl, sl = rel_err(lo["kind"], dl[:, 0], dl[:, 1], lo.get("bg", 0.0), lo.get("sigma"),
                     lo.get("count_time"))
    okl = _ok_mask(lo["kind"], sl, lo.get("sigma"), settings)
    hd = dh[start_hi:]
    rh, _ = rel_err(hi["kind"], hd[:, 0], hd[:, 1], hi.get("bg", 0.0), hi.get("sigma"),
                    hi.get("count_time"))
    rh_at = np.interp(dl[:, 0], hd[:, 0], rh)
    ov0 = hd[:, 0].min()
    # the noise cap: last index (after start) where lo is still usable
    usable = np.where(okl & (np.arange(n) >= start_lo))[0]
    cap = int(usable.max()) if len(usable) else start_lo
    first_ov = int(np.argmax(dl[:, 0] >= ov0)) if (dl[:, 0] >= ov0).any() else n
    flags, stats = [], {"cap_idx": cap, "cap_theta": float(dl[cap, 0]),
                        "overlap_start": float(ov0)}
    if first_ov >= n:
        flags.append({"code": "NO_OVERLAP", "severity": "review",
                      "detail": f"{hi['id']} starts at {ov0:.2f}° beyond the end of "
                                f"{lo['id']} ({dl[:, 0].max():.2f}°): cannot scale"})
        return n - 2, "no overlap with next section; kept all but last point", flags, stats
    chosen, sig = None, None
    j0 = max(first_ov, start_lo)
    rl_at_h = np.interp(hd[:, 0], dl[:, 0], rl)

    def overlap_stats(j):
        """Upper-section points inside [ov0, θ_lo[j]] -> (n, σ_scale)."""
        mh = (hd[:, 0] >= ov0) & (hd[:, 0] <= dl[j, 0])
        if not mh.any():
            return 0, np.inf
        rel = np.sqrt(rl_at_h[mh] ** 2 + rh[mh] ** 2)
        return int(mh.sum()), 1.2533 * np.sqrt(np.mean(rel ** 2)) / np.sqrt(mh.sum())

    for j in range(j0, n - 1):
        n_ov, sig = overlap_stats(j)
        if n_ov >= t["min_overlap"] and sig <= t["scale_rel_target"]:
            chosen = j
            break
        if j >= cap and n_ov >= t["min_overlap"]:
            break
    if chosen is not None and chosen <= cap:
        n_ov, sig = overlap_stats(chosen)
        stats.update(sigma_scale=float(sig), n_overlap=int(n_ov))
        return chosen, (f"overlap with {hi['id']}: {n_ov} {hi['id']} points "
                        f"{dl[j0, 0]:.2f}-{dl[chosen, 0]:.2f}° fix the scale "
                        f"to {sig:.1%} (target {t['scale_rel_target']:.0%})"), flags, stats
    # target not reached before the noise cap: keep to the cap, extended if
    # needed until the upper section has min_overlap points in the overlap
    last = cap
    while last < n - 2 and overlap_stats(last)[0] < t["min_overlap"]:
        last += 1
    last = min(max(last, j0), n - 2)
    n_ov, sig = overlap_stats(last)
    stats.update(sigma_scale=float(sig), n_overlap=int(n_ov))
    why = (f"{lo['id']} usable to {dl[cap, 0]:.2f}° "
           f"({'SNR' if lo['kind'] == 'pd' else 'counts'} floor)")
    if last > cap:
        flags.append({"code": "LOW_OVERLAP", "severity": "info",
                      "detail": f"{why}; extended to {dl[last, 0]:.2f}° for "
                                f"{n_ov} {hi['id']} points in the overlap"})
    if sig > 2 * t["scale_rel_target"]:
        flags.append({"code": "NOISY_OVERLAP", "severity": "info",
                      "detail": f"scale onto {hi['id']} only determined to {sig:.1%} "
                                f"from {n_ov} points"})
    return last, (f"{why}; overlap {n_ov} {hi['id']} points to {dl[last, 0]:.2f}°, "
                  f"scale ±{sig:.1%}"), flags, stats


def pd_floor_points(lo, hi, start_hi, last, cap, settings):
    """PD indices in (cap, last] not to write. -> (idx list, reason, flags)"""
    if lo["kind"] != "pd" or last <= cap:
        return [], None, []
    d = lo["data"]
    th_hi = float(hi["data"][start_hi, 0])
    k = settings["trim"]["floor_outlier_ratio"]
    s = d[:, 1] - lo.get("bg", 0.0)
    med = np.array([np.median(s[max(0, i - 2):i + 3]) for i in range(len(s))])
    covered = [i for i in range(cap + 1, last + 1) if d[i, 0] >= th_hi - 1e-6]
    dropout = [i for i in range(cap + 1, last + 1)
               if d[i, 0] < th_hi - 1e-6 and med[i] > 0 and s[i] < k * med[i]]
    if not covered and not dropout:
        return [], None, []
    parts = []
    if covered:
        parts.append(f"{len(covered)} past {th_hi:.2f}° left to {hi['id']}")
    if dropout:
        parts.append(f"{len(dropout)} dropout(s) < {k:g}× running median at "
                     + ", ".join(f"{d[i, 0]:.2f}°" for i in dropout))
    why = (f"PD past its SNR floor: {'; '.join(parts)} "
           "(used for the stitch scale, not written)")
    return sorted(covered + dropout), why, [{"code": "PD_FLOOR_POINTS", "severity": "info",
                                             "detail": f"{lo['id']}: {why}"}]


def tail_end(sec, start, settings):
    """Last section: trim the noise-floor tail. -> (last_idx, reason, flags)"""
    t = settings["trim"]
    d = sec["data"]
    n = len(d)
    _, s = rel_err(sec["kind"], d[:, 0], d[:, 1], sec.get("bg", 0.0), sec.get("sigma"),
                   sec.get("count_time"))
    ok = _ok_mask(sec["kind"], s, sec.get("sigma"), settings)
    good = np.where(ok & (np.arange(n) >= start))[0]
    last = min(int(good.max()) if len(good) else start, n - 1 - t["ct_end_min"])
    flags = []
    frac = (n - 1 - last) / max(n - start, 1)
    if frac > settings["qc"]["low_counts_tail_frac"]:
        flags.append({"code": "LOW_COUNTS_TAIL", "severity": "info",
                      "detail": f"{sec['id']}: last {n - 1 - last} of {n - start} points "
                                f"below the noise floor, trimmed from {d[last, 0]:.2f}°"})
    unit = "SNR" if sec["kind"] == "pd" else "counts"
    floor = t["snr_min"] if sec["kind"] == "pd" else t["ct_min_counts"]
    return last, (f"tail: last point with smoothed {unit} ≥ {floor:g} at "
                  f"{d[last, 0]:.2f}°" + (" (final point always dropped)"
                                          if last == n - 2 else "")), flags


def ct_start(sec, settings):
    """Trim leading saturated CEM points. -> (start, reason, flags)"""
    t = settings["trim"]
    d = sec["data"]
    sat = d[:, 1] > t["ct_max_rate"]
    flags = []
    if not sat.any():
        return 0, "no CEM saturation (all rates ≤ %.0e cps)" % t["ct_max_rate"], flags
    start = 0
    while start < len(d) and sat[start]:
        start += 1
    inner = int(sat[start:].sum())
    flags.append({"code": "CT_SATURATION", "severity": "info" if inner == 0 else "review",
                  "detail": f"{sec['id']}: {start} leading point(s) above "
                            f"{t['ct_max_rate']:.0e} cps trimmed"
                            + (f"; {inner} later point(s) also above it" if inner else "")})
    return start, f"CEM saturation: first {start} point(s) > {t['ct_max_rate']:.0e} cps", flags
