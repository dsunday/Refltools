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

import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap

jax.config.update("jax_enable_x64", True)

from refnx.reflect._jax_reflect import jabeles
from refnx.reflect.reflect_model import gauss_legendre

# Resolution-smearing constants (must match refnx)
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5


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

    def __init__(self, objective, swept_param_name, sweep_values):
        """
        Parameters
        ----------
        objective        : refnx Objective at best-fit parameter values
        swept_param_name : name of the parameter to sweep (will be held fixed)
        sweep_values     : 1D sequence of values for the swept parameter
        """
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
        n_sweep, popsize, n_free = pop_matrix.shape
        n_total = n_sweep * popsize

        layers_out = np.empty((n_total, self.n_slab_rows, 4), dtype=np.float64)
        scale_out = np.empty(n_total, dtype=np.float64)
        bkg_out = np.empty(n_total, dtype=np.float64)

        for i in range(n_sweep):
            obj_i = self.objectives[i]
            for j in range(popsize):
                set_free_params(obj_i, pop_matrix[i, j], self.swept_param_name)
                slabs, scale, bkg = get_slabs_scale_bkg(obj_i)
                idx = i * popsize + j
                layers_out[idx] = slabs
                scale_out[idx] = scale
                bkg_out[idx] = bkg

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
        layers_batch, scale_batch, bkg_batch = self.build_batch(pop_matrix)

        if self.use_smearing and self.use_log:
            chi2_flat = batched_smeared_chisqr_log_gpu(
                self.q_quad, self.combo_weights,
                self.y_log, self.y_err_log,
                layers_batch, scale_batch, bkg_batch,
            )
        elif self.use_smearing:
            chi2_flat = batched_chisqr_gpu(
                self.q, self.y, self.y_err,
                layers_batch, scale_batch, bkg_batch,
            )
        elif self.use_log:
            chi2_flat = batched_chisqr_log_gpu(
                self.q, self.y_log, self.y_err_log,
                layers_batch, scale_batch, bkg_batch,
            )
        else:
            chi2_flat = batched_chisqr_gpu(
                self.q, self.y, self.y_err,
                layers_batch, scale_batch, bkg_batch,
            )

        return chi2_flat.reshape(self.n_sweep, pop_matrix.shape[1])


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

    def __init__(self, objectives_dict, energy_list=None):
        """
        Parameters
        ----------
        objectives_dict : dict {energy: refnx Objective}
            Objectives for each energy with parameters already configured.
        energy_list : list, optional
            Subset of energies to use (default: all keys).
        """
        if energy_list is None:
            energy_list = sorted(objectives_dict.keys())
        self.energies = list(energy_list)
        self.n_energies = len(self.energies)
        self.objectives = [objectives_dict[e] for e in self.energies]

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

        # Slab shape
        slabs_0, _, _ = get_slabs_scale_bkg(self.objectives[0])
        self.n_slab_rows = slabs_0.shape[0]

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
        n_energies, popsize, n_free = pop_matrix.shape
        n_total = n_energies * popsize

        layers_out = np.empty((n_total, self.n_slab_rows, 4), dtype=np.float64)
        scale_out = np.empty(n_total, dtype=np.float64)
        bkg_out = np.empty(n_total, dtype=np.float64)

        for i in range(n_energies):
            obj_i = self.objectives[i]
            for j in range(popsize):
                set_free_params(obj_i, pop_matrix[i, j])
                slabs, scale, bkg = get_slabs_scale_bkg(obj_i)
                idx = i * popsize + j
                layers_out[idx] = slabs
                scale_out[idx] = scale
                bkg_out[idx] = bkg

        return layers_out, scale_out, bkg_out

    def fitness(self, pop_matrix):
        """
        Chi-squared for all energies and population members.

        Matches objective.chisqr(): applies log10 transform and smearing
        based on objective configuration.

        Parameters
        ----------
        pop_matrix : ndarray (n_energies, popsize, n_free)

        Returns
        -------
        chisqr : ndarray (n_energies, popsize)
        """
        n_energies, popsize, _ = pop_matrix.shape
        n_total = n_energies * popsize

        layers_batch, scale_batch, bkg_batch = self.build_batch(pop_matrix)
        energy_idx = np.repeat(np.arange(n_energies), popsize)

        if self.use_smearing:
            # Per-candidate Q-quad grid tiled from padded per-energy arrays
            q_quad_tiled = self.q_quad_pad[energy_idx]      # (n_total, max_nq, quad_order)
            mask_tiled = self.mask[energy_idx]              # (n_total, max_nq)

            # Smeared model: loop over candidates using padded Q-quad
            # Use the padded variant: vmap over (q_quad, layers, scale, bkg)
            q_quad_j = jnp.array(q_quad_tiled, dtype=jnp.float64)
            combo_j = jnp.array(self.combo_weights, dtype=jnp.float64)
            l_j = jnp.array(layers_batch, dtype=jnp.float64)
            s_j = jnp.array(scale_batch, dtype=jnp.float64)
            b_j = jnp.array(bkg_batch, dtype=jnp.float64)

            _batched_smeared_padded = jit(
                vmap(_jabeles_smeared_pos, in_axes=(0, None, 0, 0, 0))
            )
            model_batch = np.array(
                _batched_smeared_padded(q_quad_j, combo_j, l_j, s_j, b_j)
            )  # (n_total, max_nq)
        else:
            q_tiled = self.q_pad[energy_idx]
            model_batch = np.array(_batched_reflectivity_padded_jit(
                jnp.array(q_tiled, dtype=jnp.float64),
                jnp.array(layers_batch, dtype=jnp.float64),
                jnp.array(scale_batch, dtype=jnp.float64),
                jnp.array(bkg_batch, dtype=jnp.float64),
            ))  # (n_total, max_nq)

        mask_tiled = self.mask[energy_idx]  # (n_total, max_nq)

        if self.use_log:
            y_tiled = self.y_log_pad[energy_idx]
            y_err_tiled = self.y_err_log_pad[energy_idx]
            log_model = np.log10(np.maximum(model_batch, 1e-100))
            chi2_flat = np.sum(
                mask_tiled * ((y_tiled - log_model) / y_err_tiled) ** 2, axis=1
            )
        else:
            y_tiled = self.y_pad[energy_idx]
            y_err_tiled = self.y_err_pad[energy_idx]
            chi2_flat = np.sum(
                mask_tiled * ((y_tiled - model_batch) / y_err_tiled) ** 2, axis=1
            )

        return chi2_flat.reshape(n_energies, popsize)
