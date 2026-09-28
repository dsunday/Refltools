"""
recover.py
----------
Reconstruct the decisions behind an existing (manual, widget-made) reduced
file from the raw data, exactly.

The legacy pipeline writes, for every kept raw point i of section k,

    R_i = (raw_i - bg_k) * scale_k / I0(E)       at Q_i = 4π sin θ_i / (12399/E)

so each reduced point maps back to one raw point, and a two-parameter linear
fit per raw file (slope a = scale/I0, intercept -a*bg) reproduces the reduced
intensities to machine precision.  From the inlier indices we get the trims.

Used by calibrate.py to build a ground-truth table of manual trims,
backgrounds, scale factors and file choices.
"""

import numpy as np

from .inventory import read_raw

HC = 12399.0  # eV·Å, the constant used by RSoXRReduction (1239.9 nm·eV * 10)


def q_to_theta(q, energy):
    lam = HC / energy
    return np.degrees(np.arcsin(np.clip(q * lam / (4 * np.pi), -1, 1)))


def load_reduced(path):
    return np.loadtxt(path, skiprows=1, ndmin=2)


def _ransac_linear(x, y, rel_tol=1e-7, n_iter=200, seed=0):
    """Exact-relation search y = a*x + b among noisy candidates.
    Returns (a, b, inlier_mask) or (None, None, empty mask)."""
    n = len(x)
    best = (None, None, np.zeros(n, bool))
    if n == 0:
        return best
    rng = np.random.default_rng(seed)
    # b = 0 hypothesis first (CT sections, PD without background)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = y / x
    ok = np.isfinite(r) & (r > 0)
    if ok.any():
        a0 = np.median(r[ok])
        m = np.abs(a0 * x / y - 1) < 1e-6
        if m.sum() >= 2:
            a0 = np.median(r[m])
            m = np.abs(a0 * x / y - 1) < rel_tol * 10
            best = (a0, 0.0, m)
    if n < 2:
        return best
    idx = np.arange(n)
    pairs = [(i, i + 1) for i in range(n - 1)]
    if len(pairs) > n_iter:
        pairs = [pairs[i] for i in rng.choice(len(pairs), n_iter, replace=False)]
    for i, j in pairs:
        if x[i] == x[j]:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        b = y[i] - a * x[i]
        if a <= 0:
            continue
        m = np.abs((a * x + b) / y - 1) < 1e-6
        if m.sum() > best[2].sum() + 1:
            # refine on inliers
            A = np.vstack([x[m], np.ones(m.sum())]).T
            a, b = np.linalg.lstsq(A, y[m], rcond=None)[0]
            m = np.abs((a * x + b) / y - 1) < 1e-6
            best = (a, b, m)
    _ = idx
    return best


def recover_file(red, energy, raw_path, min_points=3):
    """Match one raw file against a reduced array.

    Returns dict or None: {file, n_raw, start, end (legacy negative index),
    first_idx, last_idx, n_used, n_interior_missing, a, bg, theta0, theta1,
    red_idx (indices into red)}.
    """
    _, d = read_raw(raw_path)
    if d.shape[0] < min_points:
        return None
    th_red = q_to_theta(red[:, 0], energy)
    raw_th = d[:, 0]
    order = np.argsort(raw_th)
    # nearest raw point for every reduced point
    pos = np.searchsorted(raw_th[order], th_red)
    pos = np.clip(pos, 1, len(order) - 1)
    lo, hi = order[pos - 1], order[pos]
    j = np.where(np.abs(raw_th[lo] - th_red) <= np.abs(raw_th[hi] - th_red), lo, hi)
    tol = 2e-4 + 3e-5 * np.abs(th_red)
    hit = np.abs(raw_th[j] - th_red) <= tol
    if hit.sum() < min_points:
        return None
    ri = np.where(hit)[0]
    x = d[j[hit], 1]
    y = red[ri, 1]
    a, b, m = _ransac_linear(x, y)
    if a is None or m.sum() < min_points:
        return None
    ri, jj = ri[m], j[hit][m]
    # one raw point may only be used once
    _, first = np.unique(jj, return_index=True)
    ri, jj = ri[first], jj[first]
    n = d.shape[0]
    i0, i1 = int(jj.min()), int(jj.max())
    return {"file": raw_path, "n_raw": n, "start": i0, "end": i1 + 1 - n,
            "first_idx": i0, "last_idx": i1, "n_used": int(len(jj)),
            "n_interior_missing": int(i1 - i0 + 1 - len(jj)),
            "a": float(a), "bg": float(-b / a) if a else 0.0,
            "theta0": float(d[i0, 0]), "theta1": float(d[i1, 0]),
            "red_idx": ri.tolist()}


def recover(reduced_path, energy, candidate_paths):
    """Recover all sections of one reduced file.

    Returns {'n_red': N, 'explained': k, 'sections': [recover_file dicts
    sorted by theta0]}.  Overlapping candidates that explain the same
    reduced points are resolved by keeping the one explaining more points.
    """
    red = load_reduced(reduced_path)
    hits = [h for h in (recover_file(red, energy, p) for p in candidate_paths) if h]
    hits.sort(key=lambda h: -h["n_used"])
    used = np.zeros(len(red), bool)
    keep = []
    for h in hits:
        idx = np.array(h["red_idx"], int)
        new = ~used[idx]
        if new.sum() < 0.8 * len(idx):
            continue      # explained by a better file already (duplicate scan)
        used[idx] = True
        keep.append(h)
    keep.sort(key=lambda h: h["theta0"])
    return {"n_red": int(len(red)), "explained": int(used.sum()), "sections": keep}
