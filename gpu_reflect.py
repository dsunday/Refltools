"""
GPU-accelerated reflectivity utilities for refnx.

Provides batched JAX reflectivity and chi-squared computation, plus
objective-to-slab conversion helpers for the GPU sweep optimizer.

All batched GPU functions accept numpy arrays as input and return numpy arrays.
JAX/CUDA computation is handled internally.

Smearing
--------
When a refnx Objective uses constant dQ/Q resolution smearing (model.dq > 0)
Gauss-Legendre pointwise quadrature is used.  This closely matches refnx's
_smeared_kernel_pointwise and gives near-identical chi-squared landscapes to
_smeared_kernel_constant (the difference is purely numerical).

Transform
---------
When a refnx Objective uses Transform('logY') the GPU chi-squared is computed
in log10 space with propagated uncertainties, matching objective.chisqr().
"""

import os
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap

# Grow GPU memory on demand instead of pre-allocating ~75% of VRAM at startup.
# Must be set before any JAX import triggers device initialisation.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax.config.update("jax_enable_x64", True)

from refnx.reflect._jax_reflect import jabeles, jabeles_scan
from refnx.reflect.reflect_model import gauss_legendre

# Resolution-smearing constants (must match refnx)
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5
TINY = 1e-30  # branch-cut guard used in Abeles calculation (matches _jax_reflect.py)


# ---------------------------------------------------------------------------
# Low-level batched GPU reflectivity — unsmeared
# ---------------------------------------------------------------------------

def _jabeles_pos(q, layers, scale, bkg):
    """jabeles with scale/bkg as positional args for vmap compatibility."""
    return jabeles(q, layers, scale=scale, bkg=bkg)


# JIT-compiled function: compute reflectivity for M slab configs at once.
# q: (n_q,)  layers: (M, n_rows, 4)  scale/bkg: (M,) → output: (M, n_q)
_batched_reflectivity_jit = jit(vmap(_jabeles_pos, in_axes=(None, 0, 0, 0)))

# Variant for multi-energy batching where each candidate has its own Q array.
# q: (M, max_nq)  layers: (M, n_rows, 4)  scale/bkg: (M,) → output: (M, max_nq)
_batched_reflectivity_padded_jit = jit(vmap(_jabeles_pos, in_axes=(0, 0, 0, 0)))


def batched_reflectivity_gpu(q, layers_batch, scale_batch, bkg_batch):
    """
    Compute reflectivity for M slab configurations simultaneously on GPU.

    Parameters
    ----------
    q            : array (n_q,)                  Q values (float64)
    layers_batch : array (M, n_layers+2, 4)      slab parameters
    scale_batch  : array (M,)                    scale factors
    bkg_batch    : array (M,)                    background values

    Returns
    -------
    reflectivity : ndarray (M, n_q)
    """
    return np.array(_batched_reflectivity_jit(
        jnp.array(q, dtype=jnp.float64),
        jnp.array(layers_batch, dtype=jnp.float64),
        jnp.array(scale_batch, dtype=jnp.float64),
        jnp.array(bkg_batch, dtype=jnp.float64),
    ))


def batched_chisqr_gpu(q, y, y_err, layers_batch, scale_batch, bkg_batch):
    """
    Compute chi-squared for M slab configurations simultaneously on GPU.

    Parameters
    ----------
    q, y, y_err  : array (n_q,)                 shared Q grid and data
    layers_batch : array (M, n_layers+2, 4)      slab parameters
    scale_batch  : array (M,)                    scale factors
    bkg_batch    : array (M,)                    background values

    Returns
    -------
    chisqr : ndarray (M,)   chi-squared value for each candidate
    """
    q_j = jnp.array(q, dtype=jnp.float64)
    y_j = jnp.array(y, dtype=jnp.float64)
    y_err_j = jnp.array(y_err, dtype=jnp.float64)
    l_j = jnp.array(layers_batch, dtype=jnp.float64)
    s_j = jnp.array(scale_batch, dtype=jnp.float64)
    b_j = jnp.array(bkg_batch, dtype=jnp.float64)

    model_batch = _batched_reflectivity_jit(q_j, l_j, s_j, b_j)  # (M, n_q)
    chi2 = jnp.sum(((y_j[None, :] - model_batch) / y_err_j[None, :]) ** 2, axis=1)
    return np.array(chi2)


# ---------------------------------------------------------------------------
# Gauss-Legendre smeared reflectivity — GPU
# ---------------------------------------------------------------------------

def compute_smear_params(q, dqvals_fwhm, quad_order=17):
    """
    Precompute Q quadrature grid and combined Gaussian×GL weights for smearing.

    Parameters
    ----------
    q            : (n_q,) Q values (float64)
    dqvals_fwhm  : (n_q,) or scalar — FWHM of resolution Gaussian at each Q
    quad_order   : Gauss-Legendre order (default 17, matching refnx)

    Returns
    -------
    q_quad        : ndarray (n_q, quad_order)  quadrature Q values
    combo_weights : ndarray (quad_order,)      Gaussian × GL weights
    """
    abscissa, weights = gauss_legendre(quad_order)
    abscissa = np.asarray(abscissa, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    prefactor = 1.0 / np.sqrt(2 * np.pi)
    gaussvals = prefactor * np.exp(-0.5 * (abscissa * _INTLIMIT) ** 2)
    combo_weights = (gaussvals * weights).astype(np.float64)

    dqvals_fwhm = np.asarray(dqvals_fwhm, dtype=np.float64)
    if dqvals_fwhm.ndim == 0:
        dqvals_fwhm = np.broadcast_to(dqvals_fwhm, q.shape)

    va = (q - _INTLIMIT * dqvals_fwhm / _FWHM)[:, None]
    vb = (q + _INTLIMIT * dqvals_fwhm / _FWHM)[:, None]
    # q_quad[i, k] = (abscissa[k] * (vb[i] - va[i]) + vb[i] + va[i]) / 2
    q_quad = (abscissa[None, :] * (vb - va) + vb + va) / 2.0
    return q_quad.astype(np.float64), combo_weights


def _jabeles_smeared_pos(q_quad, combo_weights, layers, scale, bkg):
    """
    Gauss-Legendre smeared reflectivity for a single slab set.

    q_quad        : (n_q, quad_order) — precomputed quadrature Q values
    combo_weights : (quad_order,)     — Gaussian × GL weights
    Returns smeared reflectivity (n_q,).
    """
    # jabeles flattens q and reshapes output → (n_q, quad_order)
    r_quad = jabeles(q_quad, layers, scale=1.0, bkg=0.0)
    smeared = jnp.sum(r_quad * combo_weights, axis=-1) * _INTLIMIT * scale + bkg
    return smeared


# M slab sets, shared q_quad and combo_weights → (M, n_q)
_batched_smeared_reflectivity_jit = jit(
    vmap(_jabeles_smeared_pos, in_axes=(None, None, 0, 0, 0))
)

# Padded variant: each candidate has its own Q-quad grid (for multi-energy batching)
# q_quad: (M, max_nq, quad_order)  layers: (M, n_rows, 4)  → output: (M, max_nq)
_batched_smeared_padded_jit = jit(
    vmap(_jabeles_smeared_pos, in_axes=(0, None, 0, 0, 0))
)

# ---------------------------------------------------------------------------
# Fused per-energy chi-squared functions for ModelsBatchBuilder
#
# These avoid tiling the Q-quad array across candidates each generation.
# Instead they use a nested vmap: outer over energies, inner over popsize.
# All constant arrays (q_quad_pad, y_log_pad, mask, …) are passed as
# persistent JAX device arrays, so no CPU→GPU transfer occurs.
#
# Input shapes (for the vmapped versions):
#   q_quad_pad / q_pad : (n_energies, max_nq, quad_order) or (n_energies, max_nq)
#   combo              : (quad_order,)
#   y_*_pad / mask     : (n_energies, max_nq)
#   layers             : (n_energies, popsize, n_slab, 4)
#   scale / bkg        : (n_energies, popsize)
# Output: (n_energies, popsize)
# ---------------------------------------------------------------------------

def _smeared_log_chi2_one_energy(q_quad_i, combo, y_log_i, y_err_log_i, mask_i,
                                  layers_e, scale_e, bkg_e):
    """Smeared log-chi2 for all candidates of a single energy."""
    model_e = vmap(_jabeles_smeared_pos, in_axes=(None, None, 0, 0, 0))(
        q_quad_i, combo, layers_e, scale_e, bkg_e
    )  # (popsize, max_nq)
    log_model = jnp.log10(jnp.maximum(model_e, 1e-100))
    return jnp.sum(mask_i * ((y_log_i - log_model) / y_err_log_i) ** 2, axis=-1)


_fused_smeared_log_jit = jit(
    vmap(_smeared_log_chi2_one_energy, in_axes=(0, None, 0, 0, 0, 0, 0, 0))
)


def _smeared_linear_chi2_one_energy(q_quad_i, combo, y_i, y_err_i, mask_i,
                                     layers_e, scale_e, bkg_e):
    """Smeared linear-chi2 for all candidates of a single energy."""
    model_e = vmap(_jabeles_smeared_pos, in_axes=(None, None, 0, 0, 0))(
        q_quad_i, combo, layers_e, scale_e, bkg_e
    )
    return jnp.sum(mask_i * ((y_i - model_e) / y_err_i) ** 2, axis=-1)


_fused_smeared_linear_jit = jit(
    vmap(_smeared_linear_chi2_one_energy, in_axes=(0, None, 0, 0, 0, 0, 0, 0))
)


def _unsmeared_log_chi2_one_energy(q_i, y_log_i, y_err_log_i, mask_i,
                                    layers_e, scale_e, bkg_e):
    """Unsmeared log-chi2 for all candidates of a single energy."""
    model_e = vmap(_jabeles_pos, in_axes=(None, 0, 0, 0))(
        q_i, layers_e, scale_e, bkg_e
    )
    log_model = jnp.log10(jnp.maximum(model_e, 1e-100))
    return jnp.sum(mask_i * ((y_log_i - log_model) / y_err_log_i) ** 2, axis=-1)


_fused_unsmeared_log_jit = jit(
    vmap(_unsmeared_log_chi2_one_energy, in_axes=(0, 0, 0, 0, 0, 0, 0))
)


def _unsmeared_linear_chi2_one_energy(q_i, y_i, y_err_i, mask_i,
                                       layers_e, scale_e, bkg_e):
    """Unsmeared linear-chi2 for all candidates of a single energy."""
    model_e = vmap(_jabeles_pos, in_axes=(None, 0, 0, 0))(
        q_i, layers_e, scale_e, bkg_e
    )
    return jnp.sum(mask_i * ((y_i - model_e) / y_err_i) ** 2, axis=-1)


_fused_unsmeared_linear_jit = jit(
    vmap(_unsmeared_linear_chi2_one_energy, in_axes=(0, 0, 0, 0, 0, 0, 0))
)


# ---------------------------------------------------------------------------
# float32 (complex64) Abeles path — ~8x faster than complex128 on GPU.
# Accuracy: max relative chi2 error ~0.1% vs float64; sufficient for CMA-ES
# population ranking. Final best-solution chi2 is re-evaluated in float64.
# ---------------------------------------------------------------------------

def _jabeles_scan_f32(q, layers, scale=1.0, bkg=0, threads=0):
    """jabeles_scan using complex64 arithmetic."""
    qvals = q.astype(jnp.float32)
    flatq = qvals.ravel()
    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    sld = jnp.zeros(nlayers + 2, jnp.complex64)
    sld = sld.at[1:].add(
        ((layers[1:, 1] - layers[0, 1]) + 1j * (jnp.abs(layers[1:, 2]) + TINY)).astype(
            jnp.complex64
        ) * jnp.float32(1.0e-6)
    )

    kn = jnp.sqrt(
        flatq[:, jnp.newaxis].astype(jnp.complex64) ** 2 / 4.0 - 4.0 * jnp.pi * sld
    )
    damping = jnp.exp(
        -2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3].astype(jnp.float32) ** 2
    )
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:]) * damping

    ones_row = jnp.ones((1, npnts), jnp.complex64)
    if nlayers:
        phase = jnp.exp(
            kn[:, 1:-1] * jnp.complex64(1j) * jnp.abs(layers[1:-1, 0].astype(jnp.float32))
        )
        mi00_stk = jnp.concatenate([ones_row, phase.T], axis=0)
    else:
        mi00_stk = ones_row

    mi11_stk = jnp.float32(1.0) / mi00_stk
    rj_T = rj.T
    mi01_stk = rj_T * mi00_stk
    mi10_stk = rj_T * mi11_stk

    def step(carry, x):
        m00, m01, m10, m11 = carry
        n00, n01, n10, n11 = x
        return (
            m00 * n00 + m01 * n10,
            m00 * n01 + m01 * n11,
            m10 * n00 + m11 * n10,
            m10 * n01 + m11 * n11,
        ), None

    init = (mi00_stk[0], mi01_stk[0], mi10_stk[0], mi11_stk[0])
    (m00, _, m10, _), _ = jax.lax.scan(
        step, init, (mi00_stk[1:], mi01_stk[1:], mi10_stk[1:], mi11_stk[1:])
    )
    r = m10 / m00
    return jnp.real(jnp.reshape(r * jnp.conj(r), qvals.shape)).astype(jnp.float64) * scale + bkg


def _jabeles_f32_pos(q, layers, scale, bkg):
    return _jabeles_scan_f32(q, layers, scale=scale, bkg=bkg)


def _jabeles_smeared_f32_pos(q_quad, combo_weights, layers, scale, bkg):
    r_quad = _jabeles_scan_f32(q_quad, layers, scale=1.0, bkg=0.0)
    return jnp.sum(r_quad * combo_weights, axis=-1) * _INTLIMIT * scale + bkg


def _smeared_log_chi2_f32_one_energy(q_quad_i, combo, y_log_i, y_err_log_i, mask_i,
                                      layers_e, scale_e, bkg_e):
    model_e = vmap(_jabeles_smeared_f32_pos, in_axes=(None, None, 0, 0, 0))(
        q_quad_i, combo, layers_e, scale_e, bkg_e
    )
    log_model = jnp.log10(jnp.maximum(model_e, 1e-100))
    return jnp.sum(mask_i * ((y_log_i - log_model) / y_err_log_i) ** 2, axis=-1)


_fused_smeared_log_f32_jit = jit(
    vmap(_smeared_log_chi2_f32_one_energy, in_axes=(0, None, 0, 0, 0, 0, 0, 0))
)


def _smeared_linear_chi2_f32_one_energy(q_quad_i, combo, y_i, y_err_i, mask_i,
                                         layers_e, scale_e, bkg_e):
    model_e = vmap(_jabeles_smeared_f32_pos, in_axes=(None, None, 0, 0, 0))(
        q_quad_i, combo, layers_e, scale_e, bkg_e
    )
    return jnp.sum(mask_i * ((y_i - model_e) / y_err_i) ** 2, axis=-1)


_fused_smeared_linear_f32_jit = jit(
    vmap(_smeared_linear_chi2_f32_one_energy, in_axes=(0, None, 0, 0, 0, 0, 0, 0))
)


def _unsmeared_log_chi2_f32_one_energy(q_i, y_log_i, y_err_log_i, mask_i,
                                        layers_e, scale_e, bkg_e):
    model_e = vmap(_jabeles_f32_pos, in_axes=(None, 0, 0, 0))(
        q_i, layers_e, scale_e, bkg_e
    )
    log_model = jnp.log10(jnp.maximum(model_e, 1e-100))
    return jnp.sum(mask_i * ((y_log_i - log_model) / y_err_log_i) ** 2, axis=-1)


_fused_unsmeared_log_f32_jit = jit(
    vmap(_unsmeared_log_chi2_f32_one_energy, in_axes=(0, 0, 0, 0, 0, 0, 0))
)


def _unsmeared_linear_chi2_f32_one_energy(q_i, y_i, y_err_i, mask_i,
                                           layers_e, scale_e, bkg_e):
    model_e = vmap(_jabeles_f32_pos, in_axes=(None, 0, 0, 0))(
        q_i, layers_e, scale_e, bkg_e
    )
    return jnp.sum(mask_i * ((y_i - model_e) / y_err_i) ** 2, axis=-1)


_fused_unsmeared_linear_f32_jit = jit(
    vmap(_unsmeared_linear_chi2_f32_one_energy, in_axes=(0, 0, 0, 0, 0, 0, 0))
)


def batched_smeared_chisqr_log_gpu(
    q_quad, combo_weights, y_log, y_err_log,
    layers_batch, scale_batch, bkg_batch,
):
    """
    Chi-squared in log10 space with Gauss-Legendre resolution smearing.

    Parameters
    ----------
    q_quad        : (n_q, quad_order)  precomputed quadrature Q values
    combo_weights : (quad_order,)      Gaussian × GL weights
    y_log         : (n_q,)             log10(data reflectivity)
    y_err_log     : (n_q,)             log-space uncertainties = |dy / (y * ln10)|
    layers_batch  : (M, n_slab, 4)
    scale_batch   : (M,)
    bkg_batch     : (M,)

    Returns
    -------
    chisqr : ndarray (M,)
    """
    model_batch = np.array(_batched_smeared_reflectivity_jit(
        jnp.array(q_quad, dtype=jnp.float64),
        jnp.array(combo_weights, dtype=jnp.float64),
        jnp.array(layers_batch, dtype=jnp.float64),
        jnp.array(scale_batch, dtype=jnp.float64),
        jnp.array(bkg_batch, dtype=jnp.float64),
    ))  # (M, n_q)

    log_model = np.log10(np.maximum(model_batch, 1e-100))
    chi2 = np.sum(
        ((y_log[None, :] - log_model) / y_err_log[None, :]) ** 2, axis=1
    )
    return chi2


def batched_chisqr_log_gpu(q, y_log, y_err_log, layers_batch, scale_batch, bkg_batch):
    """
    Chi-squared in log10 space — no resolution smearing.

    Parameters
    ----------
    q             : (n_q,)
    y_log         : (n_q,)   log10(data reflectivity)
    y_err_log     : (n_q,)   log-space uncertainties
    layers_batch  : (M, n_slab, 4)
    scale_batch   : (M,)
    bkg_batch     : (M,)

    Returns
    -------
    chisqr : ndarray (M,)
    """
    model_batch = np.array(_batched_reflectivity_jit(
        jnp.array(q, dtype=jnp.float64),
        jnp.array(layers_batch, dtype=jnp.float64),
        jnp.array(scale_batch, dtype=jnp.float64),
        jnp.array(bkg_batch, dtype=jnp.float64),
    ))  # (M, n_q)

    log_model = np.log10(np.maximum(model_batch, 1e-100))
    chi2 = np.sum(
        ((y_log[None, :] - log_model) / y_err_log[None, :]) ** 2, axis=1
    )
    return chi2


# ---------------------------------------------------------------------------
# Objective parameter utilities
# ---------------------------------------------------------------------------

def extract_free_params(objective, excluded_name=None):
    """
    Return free Parameter objects from objective, optionally excluding one by name.

    Returns them in the same traversal order as objective.parameters.flattened(),
    which is the order used by objective.setp().
    """
    return [
        p for p in objective.parameters.flattened()
        if p.vary and p.name != excluded_name
    ]


def get_bounds_array(free_params):
    """Return (n_free, 2) float64 array of [lb, ub] for each free Parameter."""
    bounds = []
    for p in free_params:
        lb = p.bounds.lb if hasattr(p.bounds, "lb") else float(p.bounds[0])
        ub = p.bounds.ub if hasattr(p.bounds, "ub") else float(p.bounds[1])
        bounds.append([lb, ub])
    return np.array(bounds, dtype=np.float64)


def get_free_param_values(objective, excluded_name=None):
    """Return current values of free parameters as a 1D float64 array."""
    return np.array(
        [p.value for p in extract_free_params(objective, excluded_name)],
        dtype=np.float64,
    )


def objective_data(objective):
    """Return (q, y, y_err) as float64 arrays from the objective's dataset."""
    d = objective.data
    return (
        d.x.astype(np.float64),
        d.y.astype(np.float64),
        d.y_err.astype(np.float64),
    )


def set_free_params(objective, values, excluded_name=None):
    """
    Set the free parameters of objective (excluding excluded_name) to values.

    Modifies the objective in-place. values must match the order from
    extract_free_params(objective, excluded_name).
    """
    for p, v in zip(extract_free_params(objective, excluded_name), values):
        p.value = float(v)


def get_slabs_scale_bkg(objective):
    """
    Return (slabs, scale, bkg) reflecting the current parameter state.

    slabs : ndarray (n_layers+2, 4)  — first 4 columns of structure.slabs()
    scale : float
    bkg   : float
    """
    slabs = objective.model.structure.slabs()[:, :4].copy()
    return slabs, float(objective.model.scale.value), float(objective.model.bkg.value)


def _build_param_map(obj, free_params):
    """
    Build a mapping from free-parameter index to slab/scale/bkg position.

    Perturbs each free parameter by a small amount and observes which entries
    of the slab matrix, scale, or bkg change. Works for any refnx model,
    including shared or constrained parameters.

    Returns
    -------
    param_map : list of lists
        param_map[k] = list of entries describing all positions affected by free_params[k].

        Slab entry format: ('slab', row, col, slope, intercept)
            slab_value = intercept + slope * param_value
        Scale/bkg entries: ('scale',) | ('bkg',)
            (scale and bkg parameters map 1:1 — no slope/intercept needed)

        The slope/intercept form correctly handles parameters whose value is NOT
        the slab column value (e.g. material density → SLD conversion). For
        thickness and roughness, slope=1 and intercept=0, so the formula reduces
        to slab_value = param_value as before.
    """
    base_slabs, base_scale, base_bkg = get_slabs_scale_bkg(obj)
    param_map = []
    for p in free_params:
        orig = p.value
        eps = 1e-6 * max(abs(orig), 1.0)
        p.value = orig + eps
        slabs2, scale2, bkg2 = get_slabs_scale_bkg(obj)
        p.value = orig

        mappings = []
        diff = slabs2 - base_slabs
        for row, col in zip(*np.where(np.abs(diff) > 1e-10 * eps)):
            # Linear model: slab_value = intercept + slope * param_value
            # slope = d(slab_col) / d(param)  (computed via finite difference)
            # intercept = slab_col_at_orig - slope * orig
            slope = float((slabs2[row, col] - base_slabs[row, col]) / eps)
            intercept = float(base_slabs[row, col] - slope * orig)
            mappings.append(('slab', int(row), int(col), slope, intercept))
        if abs(scale2 - base_scale) > 1e-10 * eps:
            mappings.append(('scale',))
        if abs(bkg2 - base_bkg) > 1e-10 * eps:
            mappings.append(('bkg',))
        param_map.append(mappings)
    return param_map


def _detect_objective_fitness_mode(objective):
    """
    Inspect an objective to determine which GPU fitness function to use.

    Returns
    -------
    use_log : bool
        True if objective uses a log10 transform (logY).
    dqvals_fwhm : ndarray or None
        Per-point FWHM resolution array if smearing is needed, else None.
    quad_order : int
        Gauss-Legendre order to use for smearing.
    """
    # --- transform ---
    transform = getattr(objective, "transform", None)
    use_log = (
        transform is not None and getattr(transform, "form", None) == "logY"
    )

    # --- smearing ---
    q = objective.data.x.astype(np.float64)
    x_err = getattr(objective.data, "x_err", None)
    quad_order = getattr(objective.model, "quad_order", 17)

    if x_err is not None and np.asarray(x_err).size == q.size:
        # per-point resolution from data file (FWHM values)
        dqvals_fwhm = np.asarray(x_err, dtype=np.float64)
    else:
        dq_val = 0.0
        if hasattr(objective.model, "dq"):
            dq_val = float(objective.model.dq.value)
        if dq_val > 0.5:
            dqvals_fwhm = q * (dq_val / 100.0)
        else:
            dqvals_fwhm = None

    return use_log, dqvals_fwhm, quad_order


# ---------------------------------------------------------------------------
# SweepBatchBuilder — pre-built sweep objectives with fast batch evaluation
# ---------------------------------------------------------------------------

class SweepBatchBuilder:
    """
    Manages n_sweep copies of a refnx objective (one per sweep parameter value)
    and builds GPU-ready slab batches for the evosax DE population.

    Automatically detects and matches the objective's transform (logY) and
    resolution smearing (constant dQ/Q or pointwise) so that the GPU
    chi-squared closely matches objective.chisqr().

    Usage
    -----
    builder = SweepBatchBuilder(objective, 'SOC - sld', sweep_values)
    # inside DE loop:
    fitness = builder.fitness(pop_matrix)  # (n_sweep, popsize)
    """

    def __init__(self, objective, swept_param_name, sweep_values,
                 normalize_by_n_q=False):
        """
        Parameters
        ----------
        objective        : refnx Objective at best-fit parameter values
        swept_param_name : name of the parameter to sweep (will be held fixed)
        sweep_values     : 1D sequence of values for the swept parameter
        normalize_by_n_q : if True, divide chi² by n_q (number of data points)
                           to compute a mean rather than a sum. Reduces gradient
                           magnitude for MCMC; default False (standard GF).
        """
        self.normalize_by_n_q = normalize_by_n_q
        self.swept_param_name = swept_param_name
        self.sweep_values = np.asarray(sweep_values, dtype=np.float64)
        self.n_sweep = len(self.sweep_values)

        # Make one deep copy per sweep value, with the swept param fixed
        self.objectives = []
        for v in self.sweep_values:
            obj_copy = copy.deepcopy(objective)
            for p in obj_copy.parameters.flattened():
                if p.name == swept_param_name:
                    p.value = float(v)
                    p.vary = False
                    break
            self.objectives.append(obj_copy)

        # Q, data, y_err are the same for all sweep points
        self.q, self.y, self.y_err = objective_data(objective)
        self.n_q = len(self.q)

        # Free parameters (excluding the swept one) — same for all sweep points
        self.free_params = extract_free_params(self.objectives[0], excluded_name=swept_param_name)
        self.n_free = len(self.free_params)
        self.bounds = get_bounds_array(self.free_params)  # (n_free, 2)

        # Slab shape from a sample evaluation
        slabs_0, _, _ = get_slabs_scale_bkg(self.objectives[0])
        self.n_slab_rows = slabs_0.shape[0]

        # Detect transform and smearing from objective
        use_log, dqvals_fwhm, quad_order = _detect_objective_fitness_mode(objective)
        self.use_log = use_log

        if dqvals_fwhm is not None:
            self.use_smearing = True
            self.q_quad, self.combo_weights = compute_smear_params(
                self.q, dqvals_fwhm, quad_order
            )
        else:
            self.use_smearing = False

        if use_log:
            self.y_log = np.log10(np.maximum(self.y, 1e-100))
            self.y_err_log = np.abs(self.y_err / (self.y * np.log(10)))

        # Precompute base slab arrays (one per sweep point) and param map.
        # base_slabs[i] = slab matrix with swept param fixed to sweep_values[i]
        # and all free params at their current best-fit values.
        self.base_slabs = np.stack(
            [get_slabs_scale_bkg(obj)[0] for obj in self.objectives]
        )  # (n_sweep, n_slab_rows, 4)
        self.base_scale = np.array(
            [get_slabs_scale_bkg(obj)[1] for obj in self.objectives]
        )  # (n_sweep,)
        self.base_bkg = np.array(
            [get_slabs_scale_bkg(obj)[2] for obj in self.objectives]
        )  # (n_sweep,)
        # Param map is the same for all sweep points (same model structure)
        self.param_map = _build_param_map(self.objectives[0], self.free_params)

        # Persistent GPU arrays — constant data transferred to device once
        self._q_gpu = jnp.array(self.q)
        if self.use_smearing:
            self._q_quad_gpu = jnp.array(self.q_quad)
            self._combo_gpu = jnp.array(self.combo_weights)
        if use_log:
            self._y_log_gpu = jnp.array(self.y_log)
            self._y_err_log_gpu = jnp.array(self.y_err_log)
        else:
            self._y_gpu = jnp.array(self.y)
            self._y_err_gpu = jnp.array(self.y_err)

    def build_batch(self, pop_matrix):
        """
        Construct slab arrays for all sweep-point/population combinations.

        Parameters
        ----------
        pop_matrix : ndarray (n_sweep, popsize, n_free)
            Free-parameter candidate values from the DE.

        Returns
        -------
        layers_batch : ndarray (n_sweep * popsize, n_slab_rows, 4)
        scale_batch  : ndarray (n_sweep * popsize,)
        bkg_batch    : ndarray (n_sweep * popsize,)
        """
        n_sweep, popsize, _ = pop_matrix.shape
        pop_flat = pop_matrix.reshape(n_sweep * popsize, pop_matrix.shape[2])

        # Broadcast per-sweep-point base values over popsize candidates
        layers_out = np.repeat(self.base_slabs, popsize, axis=0).copy()
        scale_out  = np.repeat(self.base_scale, popsize)
        bkg_out    = np.repeat(self.base_bkg,   popsize)

        # Apply free parameters via n_free NumPy array writes — no per-candidate loop
        # Slab entries use: slab_value = intercept + slope * param_value
        for k, mappings in enumerate(self.param_map):
            for entry in mappings:
                if entry[0] == 'slab':
                    slope = entry[3]
                    intercept = entry[4]
                    layers_out[:, entry[1], entry[2]] = intercept + slope * pop_flat[:, k]
                elif entry[0] == 'scale':
                    scale_out[:] = pop_flat[:, k]
                elif entry[0] == 'bkg':
                    bkg_out[:] = pop_flat[:, k]

        return layers_out, scale_out, bkg_out

    def fitness(self, pop_matrix):
        """
        Chi-squared fitness for all DE candidates across all sweep points.

        Matches objective.chisqr(): applies log10 transform if the objective
        uses Transform('logY'), and Gauss-Legendre smearing if dq > 0.

        Parameters
        ----------
        pop_matrix : ndarray (n_sweep, popsize, n_free)

        Returns
        -------
        chisqr : ndarray (n_sweep, popsize)
        """
        n_sweep, popsize, _ = pop_matrix.shape
        layers_batch, scale_batch, bkg_batch = self.build_batch(pop_matrix)
        l_j = jnp.array(layers_batch, dtype=jnp.float64)
        s_j = jnp.array(scale_batch,  dtype=jnp.float64)
        b_j = jnp.array(bkg_batch,    dtype=jnp.float64)

        if self.use_smearing and self.use_log:
            model = _batched_smeared_reflectivity_jit(
                self._q_quad_gpu, self._combo_gpu, l_j, s_j, b_j
            )
            log_model = jnp.log10(jnp.maximum(model, 1e-100))
            chi2_flat = np.array(jnp.sum(
                ((self._y_log_gpu - log_model) / self._y_err_log_gpu) ** 2, axis=-1
            ))
        elif self.use_smearing:
            model = _batched_smeared_reflectivity_jit(
                self._q_quad_gpu, self._combo_gpu, l_j, s_j, b_j
            )
            chi2_flat = np.array(jnp.sum(
                ((self._y_gpu - model) / self._y_err_gpu) ** 2, axis=-1
            ))
        elif self.use_log:
            model = _batched_reflectivity_jit(self._q_gpu, l_j, s_j, b_j)
            log_model = jnp.log10(jnp.maximum(model, 1e-100))
            chi2_flat = np.array(jnp.sum(
                ((self._y_log_gpu - log_model) / self._y_err_log_gpu) ** 2, axis=-1
            ))
        else:
            model = _batched_reflectivity_jit(self._q_gpu, l_j, s_j, b_j)
            chi2_flat = np.array(jnp.sum(
                ((self._y_gpu - model) / self._y_err_gpu) ** 2, axis=-1
            ))

        chi2 = chi2_flat.reshape(n_sweep, popsize)
        if self.normalize_by_n_q:
            chi2 = chi2 / self.n_q
        return chi2


# ---------------------------------------------------------------------------
# ModelsBatchBuilder — for batching across energies (batch_fit_selected_models_gpu)
# ---------------------------------------------------------------------------

class ModelsBatchBuilder:
    """
    Batches reflectivity evaluation across multiple energies for
    batch_fit_selected_models_gpu.

    Different energies may have different Q-grid lengths; they are zero-padded
    to max_nq and masked out of the chi-squared sum.

    Automatically detects and matches each objective's transform and smearing.
    Currently assumes all energies share the same transform type (logY or linear).
    For smearing: if objectives use constant dQ/Q, per-energy dqvals are computed
    from each energy's Q grid and used in per-candidate smeared reflectivity.
    """

    def __init__(self, objectives_dict, energy_list=None, use_f32_abeles=False,
                 normalize_by_n_q=False):
        """
        Parameters
        ----------
        objectives_dict  : dict {energy: refnx Objective}
            Objectives for each energy with parameters already configured.
        energy_list      : list, optional — subset of energies (default: all keys).
        use_f32_abeles   : use float32 Abeles for ~8x GPU throughput (default False).
        normalize_by_n_q : if True, divide each energy's chi² by its n_q to compute
                           a mean rather than a sum. Default False (standard GF).
        """
        self.normalize_by_n_q = normalize_by_n_q
        if energy_list is None:
            energy_list = sorted(objectives_dict.keys())
        self.energies = list(energy_list)
        self.n_energies = len(self.energies)
        self.objectives = [objectives_dict[e] for e in self.energies]
        self.use_f32_abeles = use_f32_abeles

        # Collect Q, data, y_err per energy
        self.q_list = []
        self.y_list = []
        self.y_err_list = []
        for obj in self.objectives:
            q, y, y_err = objective_data(obj)
            self.q_list.append(q)
            self.y_list.append(y)
            self.y_err_list.append(y_err)

        # Pad Q grids to max length
        self.n_q_list = [len(q) for q in self.q_list]
        self.max_nq = max(self.n_q_list)

        self.q_pad = np.zeros((self.n_energies, self.max_nq), dtype=np.float64)
        self.y_pad = np.zeros((self.n_energies, self.max_nq), dtype=np.float64)
        self.y_err_pad = np.ones((self.n_energies, self.max_nq), dtype=np.float64)  # avoid /0
        self.mask = np.zeros((self.n_energies, self.max_nq), dtype=np.float64)

        for i, (nq, q, y, y_err) in enumerate(
            zip(self.n_q_list, self.q_list, self.y_list, self.y_err_list)
        ):
            self.q_pad[i, :nq] = q
            self.y_pad[i, :nq] = y
            self.y_err_pad[i, :nq] = y_err
            self.mask[i, :nq] = 1.0

        # Detect transform and smearing from the first objective
        use_log, dqvals_fwhm_0, quad_order = _detect_objective_fitness_mode(
            self.objectives[0]
        )
        self.use_log = use_log

        # Build per-energy smearing (pads to max_nq)
        if dqvals_fwhm_0 is not None:
            self.use_smearing = True
            self._build_smear_pads(quad_order)
        else:
            self.use_smearing = False

        # Log-transform padded data arrays
        if use_log:
            self.y_log_pad = np.log10(np.maximum(self.y_pad, 1e-100))
            # padded entries get y_err_log = 1 (mask keeps them out of chi2)
            y_safe = np.where(self.mask > 0, self.y_pad, 1.0)
            self.y_err_log_pad = np.abs(self.y_err_pad / (y_safe * np.log(10)))

        # Free parameters — assumed identical structure across all energies
        self.free_params = extract_free_params(self.objectives[0])
        self.n_free = len(self.free_params)
        self.bounds = get_bounds_array(self.free_params)

        # Per-energy bounds: (n_energies, n_free, 2)
        # Used for correct clipping when bounds differ across energies (e.g. energy-dependent SLDs)
        self.per_energy_bounds = np.stack(
            [get_bounds_array(extract_free_params(obj)) for obj in self.objectives]
        )  # (n_energies, n_free, 2)

        # Slab shape
        slabs_0, _, _ = get_slabs_scale_bkg(self.objectives[0])
        self.n_slab_rows = slabs_0.shape[0]

        # Precompute base slab arrays (one per energy) and param map.
        # All energies share the same model structure (same free param positions).
        self.base_slabs = np.stack(
            [get_slabs_scale_bkg(obj)[0] for obj in self.objectives]
        )  # (n_energies, n_slab_rows, 4)
        self.base_scale = np.array(
            [get_slabs_scale_bkg(obj)[1] for obj in self.objectives]
        )  # (n_energies,)
        self.base_bkg = np.array(
            [get_slabs_scale_bkg(obj)[2] for obj in self.objectives]
        )  # (n_energies,)
        self.param_map = _build_param_map(self.objectives[0], self.free_params)

        # Persistent GPU arrays — constant data transferred to device once,
        # reused every generation without CPU→GPU round-trips.
        self._mask_gpu = jnp.array(self.mask)
        if self.use_smearing:
            self._q_quad_pad_gpu = jnp.array(self.q_quad_pad)
            self._combo_gpu = jnp.array(self.combo_weights)
        else:
            self._q_pad_gpu = jnp.array(self.q_pad)
        if use_log:
            self._y_log_pad_gpu = jnp.array(self.y_log_pad)
            self._y_err_log_pad_gpu = jnp.array(self.y_err_log_pad)
        else:
            self._y_pad_gpu = jnp.array(self.y_pad)
            self._y_err_pad_gpu = jnp.array(self.y_err_pad)

        # Persistent GPU arrays for JAX scatter (used by fitness_jax).
        self._base_slabs_j = jnp.array(self.base_slabs, dtype=jnp.float64)
        self._base_scale_j = jnp.array(self.base_scale, dtype=jnp.float64)
        self._base_bkg_j   = jnp.array(self.base_bkg,   dtype=jnp.float64)

        # Decompose param_map into scatter index lists for JAX.
        # Python for-loop here runs at trace time only — XLA fuses the at[].set() ops.
        self._slab_scatter = []   # [(k, row, col, slope, intercept), ...]
        self._scale_k = None
        self._bkg_k   = None
        for k, mappings in enumerate(self.param_map):
            for entry in mappings:
                if entry[0] == 'slab':
                    self._slab_scatter.append((k, entry[1], entry[2], entry[3], entry[4]))
                elif entry[0] == 'scale':
                    self._scale_k = k
                elif entry[0] == 'bkg':
                    self._bkg_k = k

        self._scatter_fn = self._make_scatter_fn()

    def _build_smear_pads(self, quad_order):
        """Precompute per-energy smearing Q-quad and weights, padded to max_nq."""
        abscissa, weights = gauss_legendre(quad_order)
        abscissa = np.asarray(abscissa, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        prefactor = 1.0 / np.sqrt(2 * np.pi)
        gaussvals = prefactor * np.exp(-0.5 * (abscissa * _INTLIMIT) ** 2)
        self.combo_weights = (gaussvals * weights).astype(np.float64)  # (quad_order,)

        n_quad = len(abscissa)
        self.q_quad_pad = np.zeros(
            (self.n_energies, self.max_nq, n_quad), dtype=np.float64
        )

        for i, obj in enumerate(self.objectives):
            q = self.q_list[i]
            _, dqvals_fwhm, _ = _detect_objective_fitness_mode(obj)
            if dqvals_fwhm is None:
                dqvals_fwhm = np.zeros_like(q)
            q_quad_i, _ = compute_smear_params(q, dqvals_fwhm, len(abscissa))
            nq = len(q)
            self.q_quad_pad[i, :nq, :] = q_quad_i
            # padded rows: duplicate last row (mask excludes them)
            if nq < self.max_nq:
                self.q_quad_pad[i, nq:, :] = q_quad_i[-1]

    def build_batch(self, pop_matrix):
        """
        Build slab batches for all energies and population members.

        Parameters
        ----------
        pop_matrix : ndarray (n_energies, popsize, n_free)

        Returns
        -------
        layers_batch : ndarray (n_energies * popsize, n_slab_rows, 4)
        scale_batch  : ndarray (n_energies * popsize,)
        bkg_batch    : ndarray (n_energies * popsize,)
        """
        n_energies, popsize, _ = pop_matrix.shape
        pop_flat = pop_matrix.reshape(n_energies * popsize, pop_matrix.shape[2])

        # Broadcast per-energy base values over popsize candidates
        layers_out = np.repeat(self.base_slabs, popsize, axis=0).copy()
        scale_out  = np.repeat(self.base_scale, popsize)
        bkg_out    = np.repeat(self.base_bkg,   popsize)

        # Apply free parameters via n_free NumPy array writes — no per-candidate loop
        # Slab entries use: slab_value = intercept + slope * param_value
        for k, mappings in enumerate(self.param_map):
            for entry in mappings:
                if entry[0] == 'slab':
                    slope = entry[3]
                    intercept = entry[4]
                    layers_out[:, entry[1], entry[2]] = intercept + slope * pop_flat[:, k]
                elif entry[0] == 'scale':
                    scale_out[:] = pop_flat[:, k]
                elif entry[0] == 'bkg':
                    bkg_out[:] = pop_flat[:, k]

        return layers_out, scale_out, bkg_out

    def _make_scatter_fn(self):
        """
        Return a @jax.jit function that maps (n_e, pop, n_free) → (layers, scale, bkg).

        The Python for-loop over slab_scatter is unrolled at JIT trace time;
        XLA fuses the resulting at[].set() operations. No NumPy or CPU work
        occurs after the first call, making it safe to use inside fori_loop.
        """
        base_slabs_j = self._base_slabs_j
        base_scale_j = self._base_scale_j
        base_bkg_j   = self._base_bkg_j
        n_slab_rows  = self.n_slab_rows
        slab_scatter = self._slab_scatter
        scale_k      = self._scale_k
        bkg_k        = self._bkg_k

        @jax.jit
        def scatter_fn(pop_matrix):
            n_e, pop_sz, _ = pop_matrix.shape
            layers = (base_slabs_j[:, None, :, :]
                      + jnp.zeros((n_e, pop_sz, n_slab_rows, 4), jnp.float64))
            for k, row, col, slope, intercept in slab_scatter:
                layers = layers.at[:, :, row, col].set(
                    intercept + slope * pop_matrix[:, :, k]
                )
            scale = base_scale_j[:, None] + jnp.zeros((n_e, pop_sz), jnp.float64)
            bkg   = base_bkg_j[:, None]   + jnp.zeros((n_e, pop_sz), jnp.float64)
            if scale_k is not None:
                scale = pop_matrix[:, :, scale_k]
            if bkg_k is not None:
                bkg = pop_matrix[:, :, bkg_k]
            return layers, scale, bkg

        return scatter_fn

    def fitness_jax(self, pop_matrix_jax):
        """
        Chi-squared accepting and returning JAX arrays (no NumPy conversion).

        Uses the JIT-compiled scatter_fn to assemble slab arrays entirely on
        GPU. Safe to call inside jax.lax.fori_loop.

        Parameters
        ----------
        pop_matrix_jax : JAX array (n_energies, popsize, n_free)

        Returns
        -------
        chi2 : JAX array (n_energies, popsize)
        """
        layers, scale, bkg = self._scatter_fn(pop_matrix_jax)

        if self.use_smearing and self.use_log:
            kernel = _fused_smeared_log_f32_jit if self.use_f32_abeles else _fused_smeared_log_jit
            return kernel(
                self._q_quad_pad_gpu, self._combo_gpu,
                self._y_log_pad_gpu, self._y_err_log_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        elif self.use_smearing:
            kernel = _fused_smeared_linear_f32_jit if self.use_f32_abeles else _fused_smeared_linear_jit
            chi2 = kernel(
                self._q_quad_pad_gpu, self._combo_gpu,
                self._y_pad_gpu, self._y_err_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        elif self.use_log:
            kernel = _fused_unsmeared_log_f32_jit if self.use_f32_abeles else _fused_unsmeared_log_jit
            chi2 = kernel(
                self._q_pad_gpu,
                self._y_log_pad_gpu, self._y_err_log_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        else:
            kernel = _fused_unsmeared_linear_f32_jit if self.use_f32_abeles else _fused_unsmeared_linear_jit
            chi2 = kernel(
                self._q_pad_gpu,
                self._y_pad_gpu, self._y_err_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        if self.normalize_by_n_q:
            n_q_j = jnp.array(self.n_q_list, dtype=jnp.float64).reshape(-1, 1)
            chi2 = chi2 / n_q_j
        return chi2

    def fitness(self, pop_matrix):
        """
        Chi-squared for all energies and population members.

        Matches objective.chisqr(): applies log10 transform and smearing
        based on objective configuration.

        Uses persistent GPU arrays for constant data (Q grids, y data, mask)
        to avoid CPU→GPU transfers every generation.

        Parameters
        ----------
        pop_matrix : ndarray (n_energies, popsize, n_free)

        Returns
        -------
        chisqr : ndarray (n_energies, popsize)
        """
        n_energies, popsize, _ = pop_matrix.shape
        layers_flat, scale_flat, bkg_flat = self.build_batch(pop_matrix)

        # Reshape to (n_energies, popsize, ...) for nested vmap
        layers = jnp.array(
            layers_flat.reshape(n_energies, popsize, self.n_slab_rows, 4),
            dtype=jnp.float64,
        )
        scale = jnp.array(scale_flat.reshape(n_energies, popsize), dtype=jnp.float64)
        bkg   = jnp.array(bkg_flat.reshape(n_energies, popsize),   dtype=jnp.float64)

        if self.use_smearing and self.use_log:
            kernel = _fused_smeared_log_f32_jit if self.use_f32_abeles else _fused_smeared_log_jit
            chi2 = kernel(
                self._q_quad_pad_gpu, self._combo_gpu,
                self._y_log_pad_gpu, self._y_err_log_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        elif self.use_smearing:
            kernel = _fused_smeared_linear_f32_jit if self.use_f32_abeles else _fused_smeared_linear_jit
            chi2 = kernel(
                self._q_quad_pad_gpu, self._combo_gpu,
                self._y_pad_gpu, self._y_err_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        elif self.use_log:
            kernel = _fused_unsmeared_log_f32_jit if self.use_f32_abeles else _fused_unsmeared_log_jit
            chi2 = kernel(
                self._q_pad_gpu,
                self._y_log_pad_gpu, self._y_err_log_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        else:
            kernel = _fused_unsmeared_linear_f32_jit if self.use_f32_abeles else _fused_unsmeared_linear_jit
            chi2 = kernel(
                self._q_pad_gpu,
                self._y_pad_gpu, self._y_err_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )

        chi2_np = np.array(chi2)  # (n_energies, popsize)
        if self.normalize_by_n_q:
            n_q_arr = np.array(self.n_q_list, dtype=np.float64).reshape(-1, 1)
            chi2_np = chi2_np / n_q_arr
        return chi2_np


# ---------------------------------------------------------------------------
# MultiEnergySweepBatchBuilder — batches n_energies × max_n_sweep simultaneously
# ---------------------------------------------------------------------------

class MultiEnergySweepBatchBuilder:
    """
    Batches n_energies × max_n_sweep optimization instances simultaneously.

    Combines per-energy Q/data/optical-constant variation (like ModelsBatchBuilder)
    with per-sweep-value parameter-fixing (like SweepBatchBuilder).  Total batch
    size: n_total = n_energies × max_n_sweep.

    For delta/step sweeps where different energies produce different n_sweep
    counts, slots beyond each energy's real sweep count are padded; padded
    slots return fitness = 1e30 and never become best solutions.

    fitness(pop_matrix) accepts (n_total, popsize, n_free)
    and returns (n_total, popsize).

    Results are indexed as: slot index = energy_index * max_n_sweep + sweep_index.
    """

    def __init__(self, objectives_dict, param_name, sweep_values_per_energy):
        """
        Parameters
        ----------
        objectives_dict          : {energy: refnx Objective} at best-fit values
        param_name               : str — parameter to sweep (held fixed per slot)
        sweep_values_per_energy  : {energy: array-like} — sweep values per energy
        """
        self.param_name = param_name
        self.energies = sorted(objectives_dict.keys())
        self.n_energies = len(self.energies)

        self.sweep_values_per_energy = {
            e: np.asarray(sweep_values_per_energy[e], dtype=np.float64)
            for e in self.energies
        }

        n_sweeps = [len(self.sweep_values_per_energy[e]) for e in self.energies]
        self.max_n_sweep = max(n_sweeps)
        self.n_total = self.n_energies * self.max_n_sweep

        # valid_mask[i] = True for real slots (flat index = e_idx * max_n_sweep + s_idx)
        self.valid_mask = np.zeros(self.n_total, dtype=bool)
        for i, ns in enumerate(n_sweeps):
            self.valid_mask[i * self.max_n_sweep : i * self.max_n_sweep + ns] = True

        # Build free params and bounds from first energy (structure shared across energies)
        ref_obj_copy = copy.deepcopy(objectives_dict[self.energies[0]])
        for p in ref_obj_copy.parameters.flattened():
            if p.name == param_name:
                p.vary = False
                break
        self.free_params = extract_free_params(ref_obj_copy, excluded_name=param_name)
        self.n_free = len(self.free_params)
        self.bounds = get_bounds_array(self.free_params)

        # Validate n_free is consistent across energies
        for e in self.energies[1:]:
            other_free = [
                p for p in objectives_dict[e].parameters.flattened()
                if p.vary and p.name != param_name
            ]
            if len(other_free) != self.n_free:
                raise ValueError(
                    f"Energy {e} has {len(other_free)} free parameters "
                    f"(excluding '{param_name}'), but {self.energies[0]} has "
                    f"{self.n_free}. All energies must have the same n_free."
                )

        # Slab shape from reference
        slabs_0, _, _ = get_slabs_scale_bkg(ref_obj_copy)
        self.n_slab_rows = slabs_0.shape[0]

        # Precompute base slabs for all n_total slots — one deep copy per energy
        self.base_slabs = np.empty((self.n_total, self.n_slab_rows, 4), dtype=np.float64)
        self.base_scale = np.empty(self.n_total, dtype=np.float64)
        self.base_bkg   = np.empty(self.n_total, dtype=np.float64)

        for i, e in enumerate(self.energies):
            obj_copy = copy.deepcopy(objectives_dict[e])
            sv = self.sweep_values_per_energy[e]
            ns = len(sv)
            swept_p = next(
                (p for p in obj_copy.parameters.flattened() if p.name == param_name),
                None,
            )
            for s in range(self.max_n_sweep):
                val = float(sv[min(s, ns - 1)])  # repeat last value for padding
                if swept_p is not None:
                    swept_p.value = val
                slabs, scale, bkg = get_slabs_scale_bkg(obj_copy)
                idx = i * self.max_n_sweep + s
                self.base_slabs[idx] = slabs
                self.base_scale[idx] = scale
                self.base_bkg[idx]   = bkg

        # param_map is the same for all instances (shared model structure)
        self.param_map = _build_param_map(ref_obj_copy, self.free_params)

        # Detect transform and smearing from first objective
        use_log, dqvals_fwhm_0, quad_order = _detect_objective_fitness_mode(
            objectives_dict[self.energies[0]]
        )
        self.use_log = use_log

        # Build Q/data padded arrays: each energy's arrays are tiled max_n_sweep times
        q_list, y_list, y_err_list, n_q_list = [], [], [], []
        for e in self.energies:
            q, y, y_err = objective_data(objectives_dict[e])
            q_list.append(q); y_list.append(y); y_err_list.append(y_err)
            n_q_list.append(len(q))

        self.max_nq = max(n_q_list)

        self.q_pad_all     = np.zeros((self.n_total, self.max_nq), dtype=np.float64)
        self.y_pad_all     = np.zeros((self.n_total, self.max_nq), dtype=np.float64)
        self.y_err_pad_all = np.ones( (self.n_total, self.max_nq), dtype=np.float64)
        self.mask_all      = np.zeros((self.n_total, self.max_nq), dtype=np.float64)

        for i, (nq, q, y, y_err) in enumerate(
            zip(n_q_list, q_list, y_list, y_err_list)
        ):
            rows = slice(i * self.max_n_sweep, (i + 1) * self.max_n_sweep)
            self.q_pad_all[rows, :nq]     = q
            self.y_pad_all[rows, :nq]     = y
            self.y_err_pad_all[rows, :nq] = y_err
            self.mask_all[rows, :nq]      = 1.0

        if dqvals_fwhm_0 is not None:
            self.use_smearing = True
            self._build_smear_pads(objectives_dict, quad_order)
        else:
            self.use_smearing = False

        if use_log:
            self.y_log_pad     = np.log10(np.maximum(self.y_pad_all, 1e-100))
            y_safe             = np.where(self.mask_all > 0, self.y_pad_all, 1.0)
            self.y_err_log_pad = np.abs(self.y_err_pad_all / (y_safe * np.log(10)))

        # Transfer constant arrays to GPU once
        self._mask_gpu = jnp.array(self.mask_all)
        if self.use_smearing:
            self._q_quad_pad_gpu = jnp.array(self.q_quad_pad)
            self._combo_gpu      = jnp.array(self.combo_weights)
        else:
            self._q_pad_gpu = jnp.array(self.q_pad_all)
        if use_log:
            self._y_log_pad_gpu     = jnp.array(self.y_log_pad)
            self._y_err_log_pad_gpu = jnp.array(self.y_err_log_pad)
        else:
            self._y_pad_gpu     = jnp.array(self.y_pad_all)
            self._y_err_pad_gpu = jnp.array(self.y_err_pad_all)

    def _build_smear_pads(self, objectives_dict, quad_order):
        """Precompute per-energy smearing Q-quad arrays, tiled max_n_sweep times."""
        abscissa, weights = gauss_legendre(quad_order)
        abscissa = np.asarray(abscissa, dtype=np.float64)
        weights  = np.asarray(weights,  dtype=np.float64)
        prefactor = 1.0 / np.sqrt(2 * np.pi)
        gaussvals = prefactor * np.exp(-0.5 * (abscissa * _INTLIMIT) ** 2)
        self.combo_weights = (gaussvals * weights).astype(np.float64)

        n_quad = len(abscissa)
        self.q_quad_pad = np.zeros(
            (self.n_total, self.max_nq, n_quad), dtype=np.float64
        )

        for i, e in enumerate(self.energies):
            obj = objectives_dict[e]
            q = obj.data.x.astype(np.float64)
            nq = len(q)
            _, dqvals_fwhm, _ = _detect_objective_fitness_mode(obj)
            if dqvals_fwhm is None:
                dqvals_fwhm = np.zeros_like(q)
            q_quad_i, _ = compute_smear_params(q, dqvals_fwhm, n_quad)
            rows = slice(i * self.max_n_sweep, (i + 1) * self.max_n_sweep)
            self.q_quad_pad[rows, :nq, :] = q_quad_i
            if nq < self.max_nq:
                self.q_quad_pad[rows, nq:, :] = q_quad_i[-1]

    def build_batch(self, pop_matrix):
        """
        Build slab batches for all n_total instances and population members.

        Parameters
        ----------
        pop_matrix : ndarray (n_total, popsize, n_free)

        Returns
        -------
        layers_batch : ndarray (n_total * popsize, n_slab_rows, 4)
        scale_batch  : ndarray (n_total * popsize,)
        bkg_batch    : ndarray (n_total * popsize,)
        """
        n_total, popsize, _ = pop_matrix.shape
        pop_flat = pop_matrix.reshape(n_total * popsize, self.n_free)

        layers_out = np.repeat(self.base_slabs, popsize, axis=0).copy()
        scale_out  = np.repeat(self.base_scale, popsize)
        bkg_out    = np.repeat(self.base_bkg,   popsize)

        for k, mappings in enumerate(self.param_map):
            for entry in mappings:
                if entry[0] == 'slab':
                    layers_out[:, entry[1], entry[2]] = pop_flat[:, k]
                elif entry[0] == 'scale':
                    scale_out[:] = pop_flat[:, k]
                elif entry[0] == 'bkg':
                    bkg_out[:] = pop_flat[:, k]

        return layers_out, scale_out, bkg_out

    def fitness(self, pop_matrix):
        """
        Chi-squared for all n_total instances and population members.
        Padded slots (outside each energy's real sweep range) return 1e30.

        Parameters
        ----------
        pop_matrix : ndarray (n_total, popsize, n_free)

        Returns
        -------
        chisqr : ndarray (n_total, popsize)
        """
        n_total, popsize, _ = pop_matrix.shape
        layers_flat, scale_flat, bkg_flat = self.build_batch(pop_matrix)

        layers = jnp.array(
            layers_flat.reshape(n_total, popsize, self.n_slab_rows, 4),
            dtype=jnp.float64,
        )
        scale = jnp.array(scale_flat.reshape(n_total, popsize), dtype=jnp.float64)
        bkg   = jnp.array(bkg_flat.reshape(n_total, popsize),   dtype=jnp.float64)

        if self.use_smearing and self.use_log:
            chi2 = _fused_smeared_log_jit(
                self._q_quad_pad_gpu, self._combo_gpu,
                self._y_log_pad_gpu, self._y_err_log_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        elif self.use_smearing:
            chi2 = _fused_smeared_linear_jit(
                self._q_quad_pad_gpu, self._combo_gpu,
                self._y_pad_gpu, self._y_err_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        elif self.use_log:
            chi2 = _fused_unsmeared_log_jit(
                self._q_pad_gpu,
                self._y_log_pad_gpu, self._y_err_log_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )
        else:
            chi2 = _fused_unsmeared_linear_jit(
                self._q_pad_gpu,
                self._y_pad_gpu, self._y_err_pad_gpu, self._mask_gpu,
                layers, scale, bkg,
            )

        chi2_np = np.array(chi2)  # (n_total, popsize)
        chi2_np[~self.valid_mask, :] = 1e30
        return chi2_np
