"""
kk_fast_search.py
-----------------
Exact, batched stoichiometry / density search (forward KK direction), a drop-in
replacement for the per-candidate pool in kk_stoichiometry_fit.

Why it is exact
---------------
For a candidate F = sum_k n_k S_k (S_k = atoms, or fragment formulas) every step
of the legacy pipeline (kk_stoichiometry_fit._worker) is LINEAR in the counts n_k:
the tabulated scattering factors, the relativistic correction, the affine rescale
of the measured spectrum onto the tabulated tails at the merge points (a ratio of
two ASF differences, both linear in n) and the KK integral itself.  With the
conversion to SLD,

    SLD(E; F, rho) = s(E) * rho / M_F * f(E; F),      M_F = sum_k n_k M_k ,

the SLD at rho = 1 is therefore the mass-fraction-weighted sum of the per-species
SLDs p_k(E):  p(E; F) = sum_k w_k p_k(E),  w_k = n_k M_k / M_F.  The cost

    0.5 * MSE(rho p_re, SLD_re) + 0.5 * MSE(rho p_im, SLD_im)

is a quadratic in rho,  A rho^2 - B rho + c0, with
    A = 0.5 w^T (G_re + G_im) w,   B = w . (b_re + b_im),   c0 = 0.5 (<y_re^2> + <y_im^2>)
(G = Gram matrix of the per-species predictions over the residual energies, b =
their overlap with the data, <.> = mean over the finite points of each array,
as in K._mse), so rho* = clip(B / 2A, rho_lo, rho_hi) in closed form.  One KK
transform per SPECIES (K._asf_output_for_formula, the unchanged legacy code) plus
O(n_species^2) arithmetic per candidate replaces one KK transform + a Brent search
per candidate.  Candidates that are integer multiples of each other (C8H8, C16H16,
CH) have identical weights and therefore identical cost.

Validity (otherwise the caller falls back to the legacy path)
-------------------------------------------------------------
* density is optimised (not a grid scan);  * fix_distortions is off (it fits a
formula-dependent slope: not linear);  * every residual energy lies inside the
merge window, so each is a node of the KK grid and the legacy interpolation onto
the refined grid is irrelevant (the harness guarantees this via _restrict);
* every species' KK transform succeeds.
Violations raise FastPathUnsupported (expected, silent) or FastPathError (a
species failed; the caller warns).

Backends: 'numpy' (chunked) and 'jax' / 'jax-gpu' / 'jax-cpu' (jitted on-device
candidate enumeration, optional top-k so nothing per-candidate leaves the device).
JAX is imported only when a jax backend is requested (it enables float64 globally).
"""
import math
import os
import time
import warnings

import numpy as np
import pandas as pd

_CHUNK_NUMPY = 1 << 18
_CHUNK_JAX = 1 << 21
_RESCORE_CHUNK = 1 << 17


class FastPathError(RuntimeError):
    """The fast path cannot be used because something failed (the caller warns and falls back)."""


class FastPathUnsupported(FastPathError):
    """The request is outside the fast path's validity conditions (the caller falls back silently)."""


def _K():
    import kk_stoichiometry_fit as K          # lazy: K imports this module lazily as well
    return K


def fast_enabled(fast=None):
    """fast=None -> REFLTOOLS_KK_FAST (default on); True/False override."""
    if fast is not None:
        return bool(fast)
    return os.environ.get("REFLTOOLS_KK_FAST", "1").strip().lower() not in ("0", "false", "no", "off", "")


# --------------------------------------------------------------------------------------
# species-level quantities: the only KK transforms
# --------------------------------------------------------------------------------------
def species_statistics(sld_3col, species, merge_points, energy_mask=None, kk_kwargs=None):
    """One KK transform per species (formula string), via the legacy helpers.  Returns a dict with the species
    masses ``m``, Gram matrix ``G``, overlap vector ``b``, constant ``c0``, and the per-species SLD predictions at
    rho = 1 (``P1`` real, ``P2`` imaginary) with the data and finite masks (for the exact rescoring)."""
    K = _K()
    kk_kwargs = kk_kwargs or {}
    if kk_kwargs.get("fix_distortions"):
        raise FastPathUnsupported("fix_distortions=True is not linear in the counts")
    (_, e_fit, R, I, e_beta, _) = K._prepare_fit_data(sld_3col, energy_mask)
    if merge_points is None:                      # kkcalc then splices over the whole measured range
        lo, hi = float(e_beta[:, 0].min()), float(e_beta[:, 0].max())
    else:
        lo, hi = float(merge_points[0]), float(merge_points[1])
    if e_fit.min() < lo - 1e-9 or e_fit.max() > hi + 1e-9:
        raise FastPathUnsupported("residual energies lie outside the merge window (not KK grid nodes)")
    n_sp = len(species)
    P1, P2, m = np.zeros((n_sp, len(e_fit))), np.zeros((n_sp, len(e_fit))), np.zeros(n_sp)
    for k, f in enumerate(species):
        asf, mass = K._asf_output_for_formula(e_beta, f, merge_points, kk_kwargs)     # identical call to the legacy worker
        r = i = None
        if asf is not None:
            r, i = K._asf_to_both_sld(asf, mass, 1.0, e_fit)                           # SLD at rho = 1 g/cm^3
        if r is None or not (np.all(np.isfinite(r)) and np.all(np.isfinite(i))) or not (mass and mass > 0):
            raise FastPathError(f"KK transform failed for species '{f}'")
        P1[k], P2[k], m[k] = r, i, mass
    mr, mi = np.isfinite(R), np.isfinite(I)                                           # same per-array masks as K._mse
    if not mr.any() or not mi.any():
        raise FastPathUnsupported("no finite residual points")
    nr, ni = mr.sum(), mi.sum()
    G = (P1[:, mr] @ P1[:, mr].T) / nr + (P2[:, mi] @ P2[:, mi].T) / ni
    b = (P1[:, mr] @ R[mr]) / nr + (P2[:, mi] @ I[mi]) / ni
    c0 = 0.5 * (np.mean(R[mr] ** 2) + np.mean(I[mi] ** 2))
    return {"m": m, "G": G, "b": b, "c0": float(c0), "P1": P1, "P2": P2, "R": R, "I": I, "mr": mr, "mi": mi,
            "e_fit": e_fit, "species": list(species)}


# --------------------------------------------------------------------------------------
# candidate enumeration (same order as itertools.product)
# --------------------------------------------------------------------------------------
def _grids(count_ranges):
    K = _K()
    return [np.asarray(K._build_count_grid(cr), dtype=np.int64) for cr in count_ranges]


def _decode(idx, grids):
    shape = tuple(len(g) for g in grids)
    digits = np.unravel_index(idx, shape)
    return np.stack([g[d] for g, d in zip(grids, digits)], axis=1)          # (n, n_sp) int64


def dedupe_keep_smallest(counts):
    """Row indices of one representative (the smallest multiple) per class of integer-proportional count vectors,
    with the class multiplicity.  All-zero rows are dropped.  Rows are returned in ascending original order."""
    counts = np.asarray(counts)
    g = np.gcd.reduce(counts, axis=1)
    valid = g > 0
    red = np.zeros_like(counts)
    red[valid] = counts[valid] // g[valid, None]
    order = np.lexsort((g, *[red[:, k] for k in range(red.shape[1] - 1, -1, -1)]))   # group by reduced vector, smallest g first
    r = red[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = np.any(r[1:] != r[:-1], axis=1)
    cls = np.cumsum(new) - 1
    vo = valid[order]
    mult = np.bincount(cls[vo], minlength=cls.max() + 1 if len(cls) else 0)
    sel = new & vo
    keep, kmult = order[sel], mult[cls[sel]]
    o = np.argsort(keep)
    return keep[o], kmult[o]


# --------------------------------------------------------------------------------------
# quadratic form in rho
# --------------------------------------------------------------------------------------
def _quad_numpy(counts, stats, rho_lo, rho_hi):
    m, G, b, c0 = stats["m"], stats["G"], stats["b"], stats["c0"]
    mass = counts @ m
    with np.errstate(divide="ignore", invalid="ignore"):
        W = counts * m / mass[:, None]
        A = 0.5 * np.einsum("ni,ni->n", W @ G, W)
        B = W @ b
        rho = np.clip(B / (2.0 * A), rho_lo, rho_hi)
        mse = A * rho * rho - B * rho + c0
    bad = ~np.isfinite(mse) | (mass <= 0) | ~np.isfinite(rho)
    mse = np.where(bad, np.inf, np.maximum(mse, 0.0))
    return np.where(bad, np.nan, rho), mse


def rescore_exact(stats, counts, rho_lo, rho_hi):
    """Explicit energy-resolved cost with the closed-form density (no Gram-form cancellation), chunked."""
    m, P1, P2, R, I, mr, mi = stats["m"], stats["P1"], stats["P2"], stats["R"], stats["I"], stats["mr"], stats["mi"]
    counts = np.asarray(counts, dtype=float)
    rho_out = np.full(len(counts), np.nan); mse_out = np.full(len(counts), np.inf)
    Rm, Im = R[mr], I[mi]
    for s in range(0, len(counts), _RESCORE_CHUNK):
        c = counts[s:s + _RESCORE_CHUNK]
        mass = c @ m
        with np.errstate(divide="ignore", invalid="ignore"):
            W = c * m / mass[:, None]
            p1, p2 = (W @ P1)[:, mr], (W @ P2)[:, mi]
            num = np.mean(p1 * Rm, axis=1) + np.mean(p2 * Im, axis=1)
            den = 0.5 * (np.mean(p1 * p1, axis=1) + np.mean(p2 * p2, axis=1))
            rho = np.clip(num / (2.0 * den), rho_lo, rho_hi)
            mse = 0.5 * np.mean((rho[:, None] * p1 - Rm) ** 2, axis=1) + 0.5 * np.mean((rho[:, None] * p2 - Im) ** 2, axis=1)
        bad = ~np.isfinite(mse) | (mass <= 0) | ~np.isfinite(rho)
        rho_out[s:s + len(c)] = np.where(bad, np.nan, rho); mse_out[s:s + len(c)] = np.where(bad, np.inf, mse)
    return rho_out, mse_out


# --------------------------------------------------------------------------------------
# NumPy backend over an index range / over an explicit count matrix
# --------------------------------------------------------------------------------------
def _search_numpy_grid(stats, grids, rho_lo, rho_hi, top_k, chunk):
    total = int(np.prod([len(g) for g in grids]))
    rho_all, mse_all = [], []
    best_idx = best_mse = None
    for s in range(0, total, chunk):
        idx = np.arange(s, min(s + chunk, total))
        rho, mse = _quad_numpy(_decode(idx, grids).astype(float), stats, rho_lo, rho_hi)
        if top_k is None:
            rho_all.append(rho); mse_all.append(mse)
        else:
            k = min(top_k, len(mse)); sel = np.argpartition(mse, k - 1)[:k]
            ci, cm = idx[sel], mse[sel]
            best_idx = ci if best_idx is None else np.concatenate((best_idx, ci))
            best_mse = cm if best_mse is None else np.concatenate((best_mse, cm))
            if len(best_mse) > 4 * top_k:
                keep = np.argpartition(best_mse, top_k - 1)[:top_k]; best_idx, best_mse = best_idx[keep], best_mse[keep]
    if top_k is None:
        return np.arange(total), np.concatenate(rho_all), np.concatenate(mse_all)
    o = np.argsort(best_mse, kind="stable")[:top_k]
    idx = best_idx[o]
    rho, mse = _quad_numpy(_decode(idx, grids).astype(float), stats, rho_lo, rho_hi)
    return idx, rho, mse


# --------------------------------------------------------------------------------------
# JAX backend (CPU or GPU), atoms-style grids
# --------------------------------------------------------------------------------------
_JAX = {}


def _jax_setup():
    if _JAX:
        return _JAX
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from functools import partial

    @partial(jax.jit, static_argnames=("chunk", "top_k", "shape"))
    def run(start, total, grids_pad, m, G, b, c0, rho_lo, rho_hi, *, chunk, top_k, shape):
        idx = start + jnp.arange(chunk, dtype=jnp.int64)
        valid = idx < total
        idc = jnp.where(valid, idx, 0)
        cols = {}
        div = 1
        for k in range(len(shape) - 1, -1, -1):                      # mixed radix, last axis fastest (C order)
            cols[k] = grids_pad[k][(idc // div) % shape[k]]
            div *= shape[k]
        counts = jnp.stack([cols[k] for k in range(len(shape))], axis=1).astype(jnp.float64)
        mass = counts @ m
        W = counts * m / mass[:, None]
        A = 0.5 * jnp.sum((W @ G) * W, axis=1)
        B = W @ b
        rho = jnp.clip(B / (2.0 * A), rho_lo, rho_hi)
        mse = A * rho * rho - B * rho + c0
        bad = ~jnp.isfinite(mse) | (mass <= 0) | ~valid
        mse = jnp.where(bad, jnp.inf, jnp.maximum(mse, 0.0))
        rho = jnp.where(bad, jnp.nan, rho)
        if top_k:
            vals, pos = jax.lax.top_k(-mse, top_k)
            return idx[pos], rho[pos], -vals
        return idx, rho, mse

    _JAX.update(jax=jax, jnp=jnp, run=run)
    return _JAX


def _search_jax_grid(stats, grids, rho_lo, rho_hi, top_k, chunk, device=None):
    J = _jax_setup(); jnp = J["jnp"]; jax = J["jax"]
    shape = tuple(len(g) for g in grids); total = int(np.prod(shape))
    L = max(shape); gp = np.zeros((len(grids), L), dtype=np.int64)
    for k, g in enumerate(grids):
        gp[k, :len(g)] = g
    dev = jax.devices(device)[0] if device else jax.devices()[0]
    out_i, out_r, out_m = [], [], []
    with jax.default_device(dev):
        gpd = jnp.asarray(gp); m = jnp.asarray(stats["m"]); G = jnp.asarray(stats["G"]); b = jnp.asarray(stats["b"])
        chunk = min(chunk, 1 << max(10, math.ceil(math.log2(total))))
        k_top = min(top_k, chunk) if top_k else 0
        for s in range(0, total, chunk):
            i_, r_, m_ = J["run"](s, total, gpd, m, G, b, stats["c0"], rho_lo, rho_hi, chunk=chunk, top_k=k_top, shape=shape)
            out_i.append(np.asarray(i_)); out_r.append(np.asarray(r_)); out_m.append(np.asarray(m_))
    idx, rho, mse = np.concatenate(out_i), np.concatenate(out_r), np.concatenate(out_m)
    keep = idx < total
    idx, rho, mse = idx[keep], rho[keep], mse[keep]
    if top_k:
        o = np.argsort(mse, kind="stable")[:top_k]; idx, rho, mse = idx[o], rho[o], mse[o]
    return idx, rho, mse


# --------------------------------------------------------------------------------------
# shared driver
# --------------------------------------------------------------------------------------
def _is_jax(backend):
    return str(backend).startswith("jax")


def _fit(kind, sld_3col, species, count_ranges, density_range, merge_points, energy_mask, kk_kwargs,
         backend, top_k, dedupe, chunk, device, verbose):
    K = _K()
    t0 = time.perf_counter()
    rho_lo, rho_hi = float(density_range[0]), float(density_range[1])
    stats = species_statistics(sld_3col, species, merge_points, energy_mask, kk_kwargs)
    t_species = time.perf_counter() - t0
    grids = _grids(count_ranges)
    total = int(np.prod([len(g) for g in grids]))
    n_sp = len(species)
    t1 = time.perf_counter()

    if kind == "fragments":
        # legacy _build_fragment_grid: skip the empty tuple and keep the first of identical combined formulas
        frag_dicts = [K._parse_formula(f) for f in species]
        elems = sorted({e for d in frag_dicts for e in d})
        Fm = np.array([[d.get(e, 0) for e in elems] for d in frag_dicts], dtype=np.int64)
        counts_all = _decode(np.arange(total), grids)
        nz = np.nonzero(counts_all.any(axis=1))[0]
        comb = counts_all[nz] @ Fm
        _, first = np.unique(comb, axis=0, return_index=True)
        sel = nz[np.sort(first)]
        counts_sel = counts_all[sel]
        engine = "fast-numpy"
        rho, mse = _quad_numpy(counts_sel.astype(float), stats, rho_lo, rho_hi)
        counts = counts_sel
        n_raw = len(counts_sel)
    else:
        engine = "fast-jax" if _is_jax(backend) else "fast-numpy"
        n_raw = total
        want = top_k
        if top_k is not None and dedupe:
            want = min(total, max(20 * top_k, 1000))               # over-select, then keep the best distinct compositions
        while True:
            if _is_jax(backend):
                dev = {"jax": device, "jax-gpu": "gpu", "jax-cpu": "cpu"}.get(backend, device)
                idx, rho, mse = _search_jax_grid(stats, grids, rho_lo, rho_hi, want, chunk or _CHUNK_JAX, dev)
            else:
                idx, rho, mse = _search_numpy_grid(stats, grids, rho_lo, rho_hi, want, chunk or _CHUNK_NUMPY)
            counts = _decode(idx, grids)
            if top_k is None or not dedupe:
                break
            keep, mult = dedupe_keep_smallest(counts)
            if len(keep) >= top_k or want >= total:
                break
            want = min(total, want * 4)
        if top_k is not None and dedupe:
            o = np.argsort(mse[keep], kind="stable")
            keep = keep[o]; mult = mult[o]
            idx, rho, mse, counts = idx[keep], rho[keep], mse[keep], counts[keep]
            n_equiv = mult
        else:
            n_equiv = None

    # exact (energy-resolved) re-scoring: removes the quadratic-form cancellation error
    ok = np.isfinite(mse)
    if ok.any():
        r2, m2 = rescore_exact(stats, counts[ok], rho_lo, rho_hi)
        rho = rho.copy(); mse = mse.copy(); rho[ok] = r2; mse[ok] = m2
    if kind == "fragments":
        n_equiv = None
    if dedupe and not (kind == "atoms" and top_k is not None):
        keep, mult = dedupe_keep_smallest(counts)
        rho, mse, counts, n_equiv = rho[keep], mse[keep], counts[keep], mult

    if kind == "fragments":
        fd = [K._parse_formula(f) for f in species]
        formulas = [K._combine_fragments(fd, c) for c in counts]
    else:
        formulas = [K._build_formula(species, c) for c in counts]
    rmse = np.where(np.isfinite(mse), np.sqrt(np.where(np.isfinite(mse), mse, 0.0)), np.inf)
    df = pd.DataFrame({"formula": formulas, "density": rho, "mse": mse, "rmse": rmse})
    for k, a in enumerate(species):
        df[f"n_{a}"] = counts[:, k]
    if n_equiv is not None:
        df["multiplicity"] = n_equiv
    df = df.sort_values("mse", kind="stable").reset_index(drop=True)
    if top_k is not None:
        df = df.head(top_k)
    df.attrs.update({"engine": engine, "n_candidates_total": int(n_raw), "n_rows": int(len(df)),
                     "rho_lo": rho_lo, "rho_hi": rho_hi,
                     "t_species_s": round(t_species, 4), "t_search_s": round(time.perf_counter() - t1, 4)})
    if len(df) == 0:
        raise FastPathError("no candidates")
    best_row = df.iloc[0]
    best = {k: best_row[k] for k in ("formula", "density", "mse", "rmse")}
    if verbose:
        print(f"KK stoichiometry fit  [{kind}, forward direction, {engine}]")
        print(f"  species KK transforms : {n_sp} ({t_species:.2f} s);  candidates: {n_raw}"
              + (f";  rows kept: {len(df)}" if len(df) != n_raw else "") + f";  search {df.attrs['t_search_s']:.2f} s")
        print(f"   Best: {best['formula']}  density {best['density']:.4f} g/cm³  RMSE {best['rmse']:.6f}")
    return df, best


def fast_fit_atoms(sld_3col, atoms, count_ranges, density_range, merge_points, energy_mask=None, kk_kwargs=None,
                   backend="numpy", top_k=None, dedupe=False, chunk=None, device=None, verbose=False):
    """Fast equivalent of K.fit_stoichiometry_density (optimise_density=True).  Same columns; rows for every
    candidate (all-zero candidates get mse = inf, density = NaN, as failed candidates do in the legacy search)."""
    return _fit("atoms", sld_3col, list(atoms), count_ranges, density_range, merge_points, energy_mask, kk_kwargs,
                backend, top_k, dedupe, chunk, device, verbose)


def fast_fit_fragments(sld_3col, fragments, count_ranges, density_range, merge_points, energy_mask=None, kk_kwargs=None,
                       backend="numpy", top_k=None, dedupe=False, chunk=None, device=None, verbose=False):
    """Fast equivalent of K.fit_stoichiometry_density_fragments (same dedupe-by-formula and empty-tuple skip).
    Always evaluated on NumPy (fragment grids are small)."""
    return _fit("fragments", sld_3col, list(fragments), count_ranges, density_range, merge_points, energy_mask, kk_kwargs,
                "numpy", top_k, dedupe, chunk, device, verbose)


def fast_count_scan(sld_3col, species, fixed_counts, scan_idx, count_vals, density_range, merge_points,
                    energy_mask=None, kk_kwargs=None, skip_all_zero=False):
    """RMSE and best density for each value of one species' count, the others held fixed (the scan_* surface
    functions).  Returns (rmses, best_dens) with NaN where the cost is not finite."""
    rho_lo, rho_hi = float(density_range[0]), float(density_range[1])
    stats = species_statistics(sld_3col, list(species), merge_points, energy_mask, kk_kwargs)
    counts = []
    for c in count_vals:
        row = list(fixed_counts); row[scan_idx] = int(c); counts.append(row)
    counts = np.array(counts, dtype=np.int64)
    rho, mse = rescore_exact(stats, counts, rho_lo, rho_hi)
    if skip_all_zero:
        zero = ~counts.any(axis=1)
        mse = np.where(zero, np.inf, mse)
    ok = np.isfinite(mse)
    return np.where(ok, np.sqrt(np.where(ok, mse, 0.0)), np.nan), np.where(ok, rho, np.nan)
