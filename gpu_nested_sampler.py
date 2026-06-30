"""
gpu_nested_sampler.py
---------------------
JAXNS nested sampling for refnx reflectometry objectives.

Nested sampling computes the Bayesian evidence (log Z) alongside posterior
samples, making it the right tool for comparing models with different numbers
of layers:

    log Bayes Factor = log Z(2-layer) - log Z(1-layer)

Typical usage
-------------
>>> from gpu_nested_sampler import run_nested_sampling, compare_evidence
>>>
>>> ns_1L = run_nested_sampling(obj_1L, num_live_points=500)
>>> ns_2L = run_nested_sampling(obj_2L, num_live_points=500)
>>> print(ns_1L.summary())
>>> compare_evidence({'1Layer': ns_1L, '2Layer': ns_2L})

The forward model reuses the same JAX-compiled reflectivity kernels as
gpu_mcmc.py (apply_params_jax, _jabeles_pos / _jabeles_smeared_pos) so
GPU acceleration is inherited automatically.
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Suppress JAXNS x64 warning (we enable it above)
warnings.filterwarnings("ignore", message="JAX x64 is not enabled")

_refltools = os.path.dirname(os.path.abspath(__file__))
if _refltools not in sys.path:
    sys.path.insert(0, _refltools)

from gpu_reflect import (
    extract_free_params,
    get_bounds_array,
    get_free_param_values,
    get_slabs_scale_bkg,
    set_free_params,
    set_global_params,
    _build_param_map,
    _detect_objective_fitness_mode,
    compute_smear_params,
    _jabeles_pos,
    _jabeles_smeared_pos,
)
from gpu_mcmc import apply_params_jax


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NestedSamplingResult:
    """
    Result of JAXNS nested sampling for a single refnx reflectometry model.

    The key quantity for model comparison is ``log_Z_mean`` (log marginal
    likelihood).  To compare two models::

        log_BF = result_2L.log_Z_mean - result_1L.log_Z_mean

    Positive log_BF means the 2-layer model is preferred.  Use
    ``compare_evidence({'1Layer': ns_1L, '2Layer': ns_2L})`` for a formatted
    summary with Jeffreys'-scale interpretation.

    Posterior samples (``samples``) are resampled to equal weights and stored
    as (n_posterior_samples, n_free) for downstream analysis.
    """

    log_Z_mean: float          # log marginal evidence E[log Z]
    log_Z_std: float           # uncertainty (log_Z_uncert from JAXNS)
    ESS: float                 # Kish effective sample size
    H_mean: float              # information gain (nats)
    samples: np.ndarray        # (n_posterior_samples, n_free) — equal-weight
    log_L_samples: np.ndarray  # (n_posterior_samples,) log-likelihood values
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    posterior_median: np.ndarray
    param_names: list
    run_seconds: float
    total_likelihood_evaluations: int

    def summary(self):
        """Return a pandas DataFrame with per-parameter posterior statistics."""
        import pandas as pd
        rows = []
        for i, name in enumerate(self.param_names):
            col = self.samples[:, i]
            rows.append({
                "param": name,
                "mean":   float(np.mean(col)),
                "std":    float(np.std(col)),
                "2.5%":   float(np.percentile(col, 2.5)),
                "median": float(np.median(col)),
                "97.5%":  float(np.percentile(col, 97.5)),
            })
        df = pd.DataFrame(rows).set_index("param")
        print(f"\nlog Z = {self.log_Z_mean:.3f} ± {self.log_Z_std:.3f}  "
              f"ESS = {self.ESS:.0f}  H = {self.H_mean:.2f} nats  "
              f"evals = {self.total_likelihood_evaluations}  "
              f"time = {self.run_seconds:.1f}s\n")
        return df

    def corner_plot(self, **kwargs):
        """Corner plot of posterior samples (requires the ``corner`` package)."""
        try:
            import corner
        except ImportError:
            raise ImportError("Install 'corner' to use corner_plot: pip install corner")
        import matplotlib.pyplot as plt
        fig = corner.corner(
            self.samples,
            labels=self.param_names,
            show_titles=True,
            **kwargs,
        )
        return fig

    def to_refnx_objective(self, objective):
        """Set objective's free parameters to the posterior median (in-place)."""
        set_free_params(objective, self.posterior_median)
        return objective

    def to_global_result(self, global_result):
        """
        Set a GlobalObjective's free parameters to the posterior median (in-place).

        Use this instead of to_refnx_objective() when the result comes from
        run_global_nested_sampling().  Uses varying_parameters() to avoid
        double-setting shared parameters.
        """
        set_global_params(global_result['global_objective'], self.posterior_median)
        return global_result['global_objective']

    def __repr__(self):
        return (
            f"NestedSamplingResult("
            f"log_Z={self.log_Z_mean:.3f}±{self.log_Z_std:.3f}, "
            f"ESS={self.ESS:.0f}, "
            f"n_params={len(self.param_names)}, "
            f"n_samples={len(self.samples)})"
        )


# ---------------------------------------------------------------------------
# Internal: build JAXNS Model from refnx Objective
# ---------------------------------------------------------------------------

def _build_jaxns_model(objective, use_log_space, normalize_by_n_q=False):
    """
    Build a JAXNS Model from a refnx reflectometry Objective.

    The prior is Uniform over each parameter's [lb, ub] bounds.
    The log-likelihood is the Gaussian chi-squared in either linear or
    log10 reflectivity space (matching what the refnx objective uses).

    Returns
    -------
    model : jaxns.Model
    param_names : list[str]
    """
    from jaxns import Prior, Model
    import tensorflow_probability.substrates.jax as tfp

    free_params = extract_free_params(objective)
    if not free_params:
        raise ValueError("Objective has no free parameters.")

    bounds = get_bounds_array(free_params)   # (n_free, 2)
    lb = bounds[:, 0]
    ub = bounds[:, 1]

    if not np.all(ub > lb):
        raise ValueError("All parameter upper bounds must exceed their lower bounds.")

    param_names = [p.name for p in free_params]
    base_slabs, base_scale, base_bkg = get_slabs_scale_bkg(objective)
    param_map = _build_param_map(objective, free_params)

    # Smearing detection (same logic as BlackjaxSampler)
    _, dqvals_fwhm, quad_order = _detect_objective_fitness_mode(objective)
    q = objective.data.x.astype(np.float64)

    if dqvals_fwhm is not None:
        q_quad_np, combo_weights_np = compute_smear_params(q, dqvals_fwhm, quad_order)
        q_quad_j       = jnp.array(q_quad_np,       dtype=jnp.float64)
        combo_weights_j = jnp.array(combo_weights_np, dtype=jnp.float64)
        smear = (q_quad_j, combo_weights_j)
    else:
        smear = None

    # Capture as JAX arrays in closure
    q_j          = jnp.array(q,          dtype=jnp.float64)
    y_j          = jnp.array(objective.data.y,     dtype=jnp.float64)
    dy_j         = jnp.array(objective.data.y_err, dtype=jnp.float64)
    base_slabs_j = jnp.array(base_slabs, dtype=jnp.float64)
    base_scale_f = float(base_scale)
    base_bkg_f   = float(base_bkg)

    lb_f = [float(v) for v in lb]
    ub_f = [float(v) for v in ub]

    def prior_model():
        params = []
        for i, name in enumerate(param_names):
            p = yield Prior(
                tfp.distributions.Uniform(
                    low=jnp.float64(lb_f[i]),
                    high=jnp.float64(ub_f[i]),
                ),
                name=name,
            )
            params.append(p)
        return jnp.stack(params)

    def log_likelihood(theta):
        layers, scale, bkg = apply_params_jax(
            base_slabs_j, base_scale_f, base_bkg_f, theta, param_map
        )
        if smear is not None:
            model_r = _jabeles_smeared_pos(smear[0], smear[1], layers, scale, bkg)
        else:
            model_r = _jabeles_pos(q_j, layers, scale, bkg)

        if use_log_space:
            log_m  = jnp.log10(jnp.maximum(model_r, 1e-100))
            log_d  = jnp.log10(jnp.maximum(y_j,     1e-100))
            dy_log = jnp.abs(dy_j / (jnp.maximum(y_j, 1e-100) * jnp.log(10.0)))
            ll = -0.5 * jnp.sum(((log_d - log_m) / dy_log) ** 2)
        else:
            ll = -0.5 * jnp.sum(((y_j - model_r) / dy_j) ** 2)
        return ll / y_j.shape[0] if normalize_by_n_q else ll

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    return model, param_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_nested_sampling(
    objective,
    max_samples=100_000,
    num_live_points=500,
    seed=0,
    difficult_model=False,
    n_posterior_samples=2000,
    use_log_space=True,
    verbose=True,
    normalize=False,
):
    """
    Run JAXNS nested sampling on a refnx reflectometry Objective.

    The objective should be at best-fit parameter values (e.g., after
    CMA-ES optimisation) — this doesn't affect the evidence calculation but
    ensures good JIT-compilation of the forward model.

    Parameters
    ----------
    objective          : refnx Objective
    max_samples        : max likelihood evaluations (default 100_000)
    num_live_points    : approx live points — higher = more accurate log Z (default 500)
    seed               : PRNG seed
    difficult_model    : if True, use more robust JAXNS settings (default False)
    n_posterior_samples: number of equally-weighted posterior draws (default 2000)
    use_log_space      : compute chi² in log10 reflectivity space (default True)
    verbose            : show JAXNS progress (default True)

    Returns
    -------
    NestedSamplingResult
        .log_Z_mean  — log marginal evidence (use for Bayes factors)
        .log_Z_std   — uncertainty
        .ESS         — effective sample size
        .samples     — (n_posterior_samples, n_free) posterior draws
        .summary()   — pandas DataFrame of posterior statistics
        .corner_plot()
        .to_refnx_objective(obj)
    """
    from jaxns import NestedSampler
    from jaxns.utils import resample

    model, param_names = _build_jaxns_model(objective, use_log_space,
                                            normalize_by_n_q=normalize)

    # Restrict to a single GPU. JAXNS's sharded executor requires all devices
    # to be of identical hardware type; mixed-GPU systems (e.g. A6000 + RTX 8000)
    # raise JaxRuntimeError when sharding across them.
    ns = NestedSampler(
        model,
        max_samples=max_samples,
        num_live_points=num_live_points,
        difficult_model=difficult_model,
        verbose=verbose,
        devices=[jax.devices()[0]],
    )

    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    termination_reason, state = ns(key)
    results = ns.to_results(termination_reason, state)
    elapsed = time.perf_counter() - t0

    # Resample to equal weights
    key2 = jax.random.PRNGKey(seed + 1)
    param_samples_dict = resample(
        key2,
        results.samples,
        results.log_dp_mean,
        S=n_posterior_samples,
        replace=True,
    )
    log_L_resampled = resample(
        jax.random.PRNGKey(seed + 2),
        results.log_L_samples,
        results.log_dp_mean,
        S=n_posterior_samples,
        replace=True,
    )

    # Stack named samples into a matrix (n_posterior_samples, n_free)
    if isinstance(param_samples_dict, dict):
        samples_arr = np.stack(
            [np.asarray(param_samples_dict[name]) for name in param_names], axis=-1
        )
    else:
        samples_arr = np.asarray(param_samples_dict)

    log_L_arr = np.asarray(log_L_resampled)

    return NestedSamplingResult(
        log_Z_mean   = float(results.log_Z_mean),
        log_Z_std    = float(results.log_Z_uncert),
        ESS          = float(results.ESS),
        H_mean       = float(results.H_mean),
        samples      = samples_arr,
        log_L_samples= log_L_arr,
        posterior_mean   = np.mean(samples_arr,  axis=0),
        posterior_std    = np.std(samples_arr,   axis=0),
        posterior_median = np.median(samples_arr, axis=0),
        param_names  = param_names,
        run_seconds  = elapsed,
        total_likelihood_evaluations = int(results.total_num_likelihood_evaluations),
    )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def _interpret_log_bayes_factor(log_bf):
    """Jeffreys' (1961) scale interpretation of log10 Bayes factor."""
    # Convert from natural log to log10
    log10_bf = log_bf / np.log(10.0)
    if log10_bf < 0:
        return f"Evidence *against* this model (log10 BF = {log10_bf:.2f})"
    elif log10_bf < 0.5:
        return f"Barely worth mentioning (log10 BF = {log10_bf:.2f})"
    elif log10_bf < 1.0:
        return f"Substantial evidence (log10 BF = {log10_bf:.2f})"
    elif log10_bf < 1.5:
        return f"Strong evidence (log10 BF = {log10_bf:.2f})"
    elif log10_bf < 2.0:
        return f"Very strong evidence (log10 BF = {log10_bf:.2f})"
    else:
        return f"Decisive evidence (log10 BF = {log10_bf:.2f})"


# ---------------------------------------------------------------------------
# Global nested sampling
# ---------------------------------------------------------------------------

def _build_global_jaxns_model(global_result, use_log_space, normalize_by_n_q=False):
    """
    Build a JAXNS Model from a GlobalObjective.

    The prior is Uniform over each global parameter's [lb, ub] bounds.
    The log-likelihood sums individual chi-squared contributions across all
    objectives, slicing the global theta vector for each objective via the
    per-objective global index mapping.

    Returns
    -------
    model : jaxns.Model
    param_names : list[str]
    """
    from jaxns import Prior, Model
    import tensorflow_probability.substrates.jax as tfp

    global_obj = global_result['global_objective']
    objectives = global_result['objectives']

    global_params = list(global_obj.varying_parameters())
    if not global_params:
        raise ValueError("GlobalObjective has no free parameters.")

    bounds = get_bounds_array(global_params)
    lb = bounds[:, 0]
    ub = bounds[:, 1]

    if not np.all(ub > lb):
        raise ValueError("All parameter upper bounds must exceed their lower bounds.")

    param_names = [p.name for p in global_params]
    lb_f = [float(v) for v in lb]
    ub_f = [float(v) for v in ub]

    # Per-objective global index mapping
    global_id_to_idx = {id(p): i for i, p in enumerate(global_params)}
    per_obj_global_idx = [
        [global_id_to_idx[id(p)] for p in extract_free_params(obj)]
        for obj in objectives
    ]

    # Per-objective JAX arrays
    per_obj_q, per_obj_y, per_obj_dy = [], [], []
    per_obj_base_slabs, per_obj_base_scale, per_obj_base_bkg = [], [], []
    per_obj_param_maps, per_obj_smear, per_obj_n_q = [], [], []

    for obj in objectives:
        local_free = extract_free_params(obj)
        q = obj.data.x.astype(np.float64)
        y = obj.data.y.astype(np.float64)
        dy = obj.data.y_err.astype(np.float64)
        base_slabs, base_scale, base_bkg = get_slabs_scale_bkg(obj)
        param_map = _build_param_map(obj, local_free)
        _, dqvals_fwhm, quad_order = _detect_objective_fitness_mode(obj)

        per_obj_q.append(jnp.array(q,  dtype=jnp.float64))
        per_obj_y.append(jnp.array(y,  dtype=jnp.float64))
        per_obj_dy.append(jnp.array(dy, dtype=jnp.float64))
        per_obj_base_slabs.append(jnp.array(base_slabs, dtype=jnp.float64))
        per_obj_base_scale.append(float(base_scale))
        per_obj_base_bkg.append(float(base_bkg))
        per_obj_param_maps.append(param_map)
        per_obj_n_q.append(len(q))

        if dqvals_fwhm is not None:
            q_quad, combo = compute_smear_params(q, dqvals_fwhm, quad_order)
            per_obj_smear.append((
                jnp.array(q_quad, dtype=jnp.float64),
                jnp.array(combo,  dtype=jnp.float64),
            ))
        else:
            per_obj_smear.append(None)

    n_objectives = len(objectives)
    use_log = use_log_space
    # Convert index lists to JAX arrays once — raw Python lists cannot be used
    # as indices inside jit-compiled functions in JAX ≥0.4.
    per_obj_idx_j = [jnp.array(idx, dtype=jnp.int32) for idx in per_obj_global_idx]

    def prior_model():
        params = []
        for i, name in enumerate(param_names):
            p = yield Prior(
                tfp.distributions.Uniform(
                    low=jnp.float64(lb_f[i]),
                    high=jnp.float64(ub_f[i]),
                ),
                name=name,
            )
            params.append(p)
        return jnp.stack(params)

    def log_likelihood(theta):
        # theta is already in bounded space (JAXNS handles the prior transform).
        # Python for-loop unrolled at trace time; constant-index gather is differentiable.
        ll = jnp.float64(0.0)

        for i in range(n_objectives):
            theta_i = theta[per_obj_idx_j[i]]
            layers, scale, bkg = apply_params_jax(
                per_obj_base_slabs[i],
                per_obj_base_scale[i],
                per_obj_base_bkg[i],
                theta_i,
                per_obj_param_maps[i],
            )

            smear_i = per_obj_smear[i]
            if smear_i is not None:
                model_r = _jabeles_smeared_pos(smear_i[0], smear_i[1], layers, scale, bkg)
            else:
                model_r = _jabeles_pos(per_obj_q[i], layers, scale, bkg)

            if use_log:
                log_m  = jnp.log10(jnp.maximum(model_r, 1e-100))
                log_d  = jnp.log10(jnp.maximum(per_obj_y[i], 1e-100))
                dy_log = jnp.abs(
                    per_obj_dy[i] / (jnp.maximum(per_obj_y[i], 1e-100) * jnp.log(10.0))
                )
                chi2_i = jnp.sum(((log_d - log_m) / dy_log) ** 2)
            else:
                chi2_i = jnp.sum(((per_obj_y[i] - model_r) / per_obj_dy[i]) ** 2)

            if normalize_by_n_q:
                chi2_i = chi2_i / per_obj_n_q[i]

            ll = ll - 0.5 * chi2_i

        return ll

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    return model, param_names


def run_global_nested_sampling(
    global_result,
    max_samples=100_000,
    num_live_points=500,
    seed=0,
    difficult_model=False,
    n_posterior_samples=2000,
    use_log_space=True,
    verbose=True,
    normalize=False,
):
    """
    Run JAXNS nested sampling on a GlobalObjective.

    Computes the Bayesian evidence log Z for the global model (all samples
    simultaneously constrained to share the same substrate / linked layers)
    and returns posterior samples over all n_global free parameters.

    Typical use: compare evidence between global models with different
    numbers of shared layers, or between global and independent fits::

        ns_global = run_global_nested_sampling(global_result, num_live_points=500)
        ns_indep  = run_nested_sampling(obj_sample1, num_live_points=500)
        # sum log Z across independent fits for comparison
        log_BF = ns_global.log_Z_mean - sum_indep_log_Z

    Parameters
    ----------
    global_result      : dict from create_global_reflectometry_models()
    max_samples        : max likelihood evaluations (default 100_000)
    num_live_points    : approximate live points — higher = more accurate log Z
    seed               : PRNG seed
    difficult_model    : if True, use more robust JAXNS settings
    n_posterior_samples: number of equally-weighted posterior draws (default 2000)
    use_log_space      : compute chi² in log10 reflectivity space (default True)
    verbose            : show JAXNS progress
    normalize          : if True, divide each objective's chi² by n_q

    Returns
    -------
    NestedSamplingResult
        .param_names  — global parameter names (n_global,)
        .samples      — (n_posterior_samples, n_global) posterior draws
        .log_Z_mean   — log marginal evidence for the global model
        Use result.to_global_result(global_result) to restore best params.
    """
    from jaxns import NestedSampler
    from jaxns.utils import resample

    model, param_names = _build_global_jaxns_model(
        global_result, use_log_space, normalize_by_n_q=normalize
    )

    ns = NestedSampler(
        model,
        max_samples=max_samples,
        num_live_points=num_live_points,
        difficult_model=difficult_model,
        verbose=verbose,
        devices=[jax.devices()[0]],
    )

    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    termination_reason, state = ns(key)
    results = ns.to_results(termination_reason, state)
    elapsed = time.perf_counter() - t0

    key2 = jax.random.PRNGKey(seed + 1)
    param_samples_dict = resample(
        key2,
        results.samples,
        results.log_dp_mean,
        S=n_posterior_samples,
        replace=True,
    )
    log_L_resampled = resample(
        jax.random.PRNGKey(seed + 2),
        results.log_L_samples,
        results.log_dp_mean,
        S=n_posterior_samples,
        replace=True,
    )

    if isinstance(param_samples_dict, dict):
        samples_arr = np.stack(
            [np.asarray(param_samples_dict[name]) for name in param_names], axis=-1
        )
    else:
        samples_arr = np.asarray(param_samples_dict)

    log_L_arr = np.asarray(log_L_resampled)

    return NestedSamplingResult(
        log_Z_mean    = float(results.log_Z_mean),
        log_Z_std     = float(results.log_Z_uncert),
        ESS           = float(results.ESS),
        H_mean        = float(results.H_mean),
        samples       = samples_arr,
        log_L_samples = log_L_arr,
        posterior_mean   = np.mean(samples_arr,   axis=0),
        posterior_std    = np.std(samples_arr,    axis=0),
        posterior_median = np.median(samples_arr, axis=0),
        param_names   = param_names,
        run_seconds   = elapsed,
        total_likelihood_evaluations = int(results.total_num_likelihood_evaluations),
    )


def compare_evidence(results, reference=None):
    """
    Print a Bayesian evidence comparison table for multiple models.

    Parameters
    ----------
    results   : dict {model_name: NestedSamplingResult}
    reference : name of the reference model for pairwise Bayes factors.
                Default: the model with the highest log Z (best evidence).

    Example
    -------
    >>> compare_evidence({'1Layer': ns_1L, '2Layer': ns_2L})

    Model       log Z        ± uncert    ESS      H (nats)
    -------     ------------ ----------- -------- ----------
    1Layer      -42.31       0.12        1850     12.4
    2Layer      -39.08       0.14        1720     15.1

    log Bayes Factor (2Layer vs 1Layer) = 3.23 nats
    → Strong evidence for 2Layer  (Jeffreys' scale, log10 BF = 1.40)
    """
    names = list(results.keys())
    if not names:
        print("No results to compare.")
        return

    if reference is None:
        reference = max(names, key=lambda n: results[n].log_Z_mean)

    # Header
    header = f"{'Model':<14} {'log Z':>12} {'± uncert':>11} {'ESS':>8} {'H (nats)':>10}"
    sep    = "-" * len(header)
    print(f"\n{header}")
    print(sep)
    for name in names:
        r = results[name]
        marker = " *" if name == reference else ""
        print(f"{name:<14} {r.log_Z_mean:>12.3f} {r.log_Z_std:>11.3f} "
              f"{r.ESS:>8.0f} {r.H_mean:>10.2f}{marker}")

    print(f"\n* = reference model ({reference})\n")

    # Pairwise Bayes factors vs reference
    ref = results[reference]
    for name in names:
        if name == reference:
            continue
        r = results[name]
        log_bf = r.log_Z_mean - ref.log_Z_mean
        sign = ">" if log_bf > 0 else "<"
        direction = name if log_bf > 0 else reference
        other     = reference if log_bf > 0 else name

        print(f"log BF ({name} vs {reference}) = {log_bf:+.3f} nats")
        print(f"  → {_interpret_log_bayes_factor(abs(log_bf))} "
              f"for {direction} over {other}")
        print()
