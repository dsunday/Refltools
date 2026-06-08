"""
GPU-accelerated Bayesian MCMC for refnx reflectivity objectives via BlackJAX.

Provides gradient-based posterior sampling (NUTS, MCLMC) that runs entirely on
the GPU using JAX.  Works with any refnx Objective accepted by the existing
gpu_reflect.py optimisation stack.

Bounded parameters are handled via a logit transform so the sampler works in
unconstrained ℝ^n space; the Jacobian correction is included in the log-prob.

Typical usage
-------------
>>> from gpu_mcmc import gpu_mcmc
>>> results = gpu_mcmc(objective, sampler='nuts', n_chains=4,
...                    n_warmup=1000, n_samples=2000)
>>> print(results.summary())
>>> results.trace_plot()
>>> results.to_refnx_objective(objective)   # set params to posterior median
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc import integrators as _integrators

jax.config.update("jax_enable_x64", True)

import sys
import os as _os
_refltools = _os.path.dirname(_os.path.abspath(__file__))
if _refltools not in sys.path:
    sys.path.insert(0, _refltools)

from gpu_reflect import (
    extract_free_params,
    get_bounds_array,
    get_free_param_values,
    objective_data,
    set_free_params,
    get_slabs_scale_bkg,
    _build_param_map,
    _detect_objective_fitness_mode,
    compute_smear_params,
    _jabeles_pos,
    _jabeles_smeared_pos,
)


# ---------------------------------------------------------------------------
# Parameter transform utilities
# ---------------------------------------------------------------------------

def _bounded_to_unconstrained(theta, lb, ub, eps=1e-6):
    """Logit transform: θ ∈ [lb, ub] → y ∈ ℝ."""
    s = jnp.clip((theta - lb) / (ub - lb), eps, 1.0 - eps)
    return jnp.log(s / (1.0 - s))


def _unconstrained_to_bounded(y, lb, ub):
    """Sigmoid inverse-logit: y ∈ ℝ → θ ∈ (lb, ub)."""
    return lb + jax.nn.sigmoid(y) * (ub - lb)


# ---------------------------------------------------------------------------
# Pure-JAX slab-matrix builder (differentiable)
# ---------------------------------------------------------------------------

def apply_params_jax(base_slabs, base_scale, base_bkg, theta, param_map):
    """
    Apply free parameters to slab/scale/bkg arrays — JAX-differentiable.

    param_map is a static Python list captured in a closure; the for-loop is
    unrolled at JIT trace time into a chain of .at[].set() functional updates,
    each of which is differentiable.

    Parameters
    ----------
    base_slabs  : jnp.ndarray (n_rows, 4)
    base_scale  : scalar float (Python or JAX)
    base_bkg    : scalar float (Python or JAX)
    theta       : jnp.ndarray (n_free,)   bounded parameter values
    param_map   : list of lists            static at trace time

    Returns
    -------
    (layers, scale, bkg) — JAX arrays / scalars
    """
    layers = base_slabs
    scale  = jnp.asarray(base_scale, dtype=jnp.float64)
    bkg    = jnp.asarray(base_bkg,   dtype=jnp.float64)
    for k, mappings in enumerate(param_map):
        for entry in mappings:
            if entry[0] == 'slab':
                # slab_value = intercept + slope * param_value
                # This correctly handles density→SLD and other indirect mappings.
                slope     = entry[3]
                intercept = entry[4]
                layers = layers.at[entry[1], entry[2]].set(intercept + slope * theta[k])
            elif entry[0] == 'scale':
                scale = theta[k]
            elif entry[0] == 'bkg':
                bkg = theta[k]
    return layers, scale, bkg


# ---------------------------------------------------------------------------
# Log-probability function factory
# ---------------------------------------------------------------------------

def make_log_prob_fn(
    q_j, y_j, y_err_j,
    base_slabs_j, base_scale, base_bkg,
    lb_j, ub_j,
    param_map,
    use_log=False,
    q_quad_j=None,
    combo_weights_j=None,
    normalize_by_n_q=False,
):
    """
    Build a pure-JAX log-probability function for BlackJAX samplers.

    The returned callable maps unconstrained ℝ^n_free → scalar log-probability
    and is fully differentiable (supports jax.grad, jax.jit).

    All array arguments are captured as JAX device arrays in the closure —
    no CPU→GPU transfer occurs at sample time.

    Parameters
    ----------
    q_j, y_j, y_err_j   : (n_q,) JAX float64  — Q grid and data
    base_slabs_j         : (n_rows, 4) JAX float64 — baseline slab matrix
    base_scale, base_bkg : Python floats — baseline instrument params
    lb_j, ub_j           : (n_free,) JAX float64 — parameter bounds
    param_map            : static Python list from _build_param_map
    use_log              : if True, compute chi² in log10 space
    q_quad_j             : (n_q, quad_order) JAX — smearing quadrature Q values
    combo_weights_j      : (quad_order,) JAX — Gaussian×GL weights
    """
    def log_prob(unconstrained):
        # 1. Unconstrained → bounded
        s     = jax.nn.sigmoid(unconstrained)
        theta = lb_j + s * (ub_j - lb_j)

        # 2. Build slab matrix
        layers, scale, bkg = apply_params_jax(
            base_slabs_j, base_scale, base_bkg, theta, param_map
        )

        # 3. Reflectivity
        if q_quad_j is not None:
            model = _jabeles_smeared_pos(q_quad_j, combo_weights_j, layers, scale, bkg)
        else:
            model = _jabeles_pos(q_j, layers, scale, bkg)

        # 4. Log-likelihood (optionally normalised by n_q to reduce gradient magnitude)
        if use_log:
            log_model = jnp.log10(jnp.maximum(model, 1e-100))
            log_data  = jnp.log10(jnp.maximum(y_j,  1e-100))
            y_err_log = jnp.abs(y_err_j / (jnp.maximum(y_j, 1e-100) * jnp.log(10.0)))
            ll = -0.5 * jnp.sum(((log_data - log_model) / y_err_log) ** 2)
        else:
            ll = -0.5 * jnp.sum(((y_j - model) / y_err_j) ** 2)
        if normalize_by_n_q:
            ll = ll / y_j.shape[0]

        # 5. Log-Jacobian correction for logit transform:
        #    dθ/dy = σ(y)·(1−σ(y))·(ub−lb)
        lj = jnp.sum(jnp.log(s * (1.0 - s) * (ub_j - lb_j)))

        return ll + lj

    return log_prob


# ---------------------------------------------------------------------------
# Chain initialisation helpers
# ---------------------------------------------------------------------------

def _make_init_positions(y_init, n_chains, key, perturbation_scale=0.1):
    """
    Generate n_chains starting positions in unconstrained space.

    Small Gaussian noise around y_init so chains start near the MAP
    estimate but are distinct.  scale=0.1 is small relative to the logit
    range ~[-5, 5].
    """
    keys   = jax.random.split(key, n_chains)
    noise  = jnp.stack([jax.random.normal(k, y_init.shape) for k in keys])
    return y_init[None, :] + noise * perturbation_scale


# ---------------------------------------------------------------------------
# Progress bar helper (chunked tqdm — avoids fastprogress dependency)
# ---------------------------------------------------------------------------

def _make_pbar(n_total, progress, desc=''):
    if not progress:
        return None
    try:
        from tqdm.auto import tqdm
        return tqdm(total=n_total, desc=desc)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# MCMCSamples result container
# ---------------------------------------------------------------------------

class MCMCSamples:
    """
    Container for BlackJAX MCMC results.

    Attributes
    ----------
    samples_bounded : ndarray (n_chains, n_samples, n_free)
        Posterior samples in the original bounded parameter space [lb, ub].
    samples_unconstrained : ndarray (n_chains, n_samples, n_free)
        Samples in the logit-transformed unconstrained ℝ^n space.
    param_names : list[str]
    n_chains, n_samples, n_free : int
    sampler_name : str   'nuts' or 'mclmc'
    extras : dict   sampler-specific diagnostic arrays
    """

    def __init__(self, samples_unconstrained, samples_bounded,
                 param_names, sampler_name, extras=None):
        self.samples_unconstrained = np.asarray(samples_unconstrained)
        self.samples_bounded       = np.asarray(samples_bounded)
        self.param_names           = list(param_names)
        self.sampler_name          = sampler_name
        self.extras                = dict(extras or {})
        self.n_chains, self.n_samples, self.n_free = self.samples_bounded.shape

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self):
        """
        Per-parameter summary: mean, std, 2.5%/50%/97.5%, R-hat, ESS.

        Returns a pandas DataFrame indexed by parameter name.
        R-hat < 1.05 indicates convergence; ESS > 100 × n_chains is healthy.
        """
        import pandas as pd

        s = jnp.array(self.samples_bounded)          # (n_chains, n_samples, n_free)
        s_flat = np.array(s).reshape(-1, self.n_free)  # (n_chains*n_samples, n_free)

        rows = []
        for i, name in enumerate(self.param_names):
            col    = s_flat[:, i]
            # rhat/ess expect (n_chains, n_samples, 1) or (n_chains, n_samples)
            chain_i = s[:, :, i]   # (n_chains, n_samples)
            rhat = float(blackjax.rhat(chain_i, chain_axis=0, sample_axis=1))
            ess  = float(blackjax.ess(chain_i,  chain_axis=0, sample_axis=1))
            rows.append({
                'parameter': name,
                'mean':    float(col.mean()),
                'std':     float(col.std()),
                '2.5%':    float(np.percentile(col, 2.5)),
                'median':  float(np.percentile(col, 50.0)),
                '97.5%':   float(np.percentile(col, 97.5)),
                'r_hat':   rhat,
                'ess':     ess,
            })
        return pd.DataFrame(rows).set_index('parameter')

    def acceptance_rate(self):
        """Mean acceptance rate across chains (NUTS only; None for MCLMC)."""
        ar = self.extras.get('acceptance_rate')
        if ar is not None:
            return float(np.asarray(ar).mean())
        return None

    def divergence_fraction(self):
        """Fraction of divergent transitions (NUTS only; None for MCLMC)."""
        d = self.extras.get('is_divergent')
        if d is not None:
            return float(np.asarray(d).mean())
        return None

    def energy_change_variance(self):
        """Variance of energy change per step (MCLMC only; None for NUTS)."""
        e = self.extras.get('energy_change')
        if e is not None:
            return float(np.asarray(e).var())
        return None

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def trace_plot(self, figsize=None):
        """
        Trace plot: one row per parameter, all chains overlaid.
        Returns a matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        n = self.n_free
        fig, axes = plt.subplots(n, 1, figsize=figsize or (10, 2.5 * n), squeeze=False)
        axes = axes[:, 0]
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            for c in range(self.n_chains):
                ax.plot(self.samples_bounded[c, :, i], alpha=0.6, lw=0.8)
            ax.set_ylabel(name, fontsize=8)
            ax.tick_params(labelsize=8)
        axes[-1].set_xlabel('Sample index')
        fig.suptitle(
            f'{self.sampler_name.upper()} — {self.n_chains} chains × {self.n_samples} samples'
        )
        fig.tight_layout()
        return fig

    def corner_plot(self, **kwargs):
        """
        Corner plot of all chains combined.  Requires the `corner` package.
        Returns a matplotlib Figure.
        """
        import corner
        flat = self.samples_bounded.reshape(-1, self.n_free)
        defaults = dict(labels=self.param_names, show_titles=True,
                        title_kwargs={"fontsize": 9})
        defaults.update(kwargs)
        return corner.corner(flat, **defaults)

    def to_arviz(self):
        """
        Convert to an arviz InferenceData object for extended diagnostics.
        Requires `arviz`.
        """
        import arviz as az
        d = {name: self.samples_bounded[:, :, i]
             for i, name in enumerate(self.param_names)}
        return az.from_dict(d)

    # ------------------------------------------------------------------
    # Integration with refnx
    # ------------------------------------------------------------------

    def to_refnx_objective(self, objective):
        """
        Set objective's free parameters to the posterior median.
        Modifies the objective in-place and returns it.
        """
        flat = self.samples_bounded.reshape(-1, self.n_free)
        median_vals = np.median(flat, axis=0)
        set_free_params(objective, median_vals)
        return objective

    def __repr__(self):
        return (
            f"MCMCSamples(sampler={self.sampler_name!r}, "
            f"n_chains={self.n_chains}, n_samples={self.n_samples}, "
            f"n_free={self.n_free})"
        )


# ---------------------------------------------------------------------------
# Main sampler class
# ---------------------------------------------------------------------------

class BlackjaxSampler:
    """
    GPU-accelerated Bayesian MCMC sampler for a refnx Objective.

    Uses BlackJAX (NUTS or MCLMC) with logit-transformed bounded parameters.
    Multiple chains run in parallel via jax.vmap.

    The Objective is snapshotted at construction time — changes to the
    objective's parameters after construction are not reflected.

    Parameters
    ----------
    objective : refnx Objective
        Should be at the best-fit parameter values.
    sampler : 'nuts' | 'mclmc'
    n_chains : int
    seed : int
    """

    def __init__(self, objective, sampler='nuts', n_chains=4, seed=0, normalize=False):
        if sampler not in ('nuts', 'mclmc'):
            raise ValueError(f"sampler must be 'nuts' or 'mclmc', got {sampler!r}")

        self.sampler_name = sampler
        self.n_chains     = n_chains
        self._master_key  = jax.random.PRNGKey(seed)

        # --- Extract parameter info ---
        self.free_params  = extract_free_params(objective)
        self.n_free       = len(self.free_params)
        if self.n_free == 0:
            raise ValueError("Objective has no free parameters.")
        if sampler == 'mclmc' and self.n_free < 2:
            raise ValueError(
                "MCLMC requires at least 2 free parameters. "
                "Use sampler='nuts' for single-parameter models."
            )

        bounds_np         = get_bounds_array(self.free_params)  # (n_free, 2)
        if not np.all(bounds_np[:, 1] > bounds_np[:, 0]):
            raise ValueError(
                "All parameter upper bounds must be strictly greater than lower bounds."
            )
        self.param_names  = [p.name for p in self.free_params]

        # --- JAX device arrays (uploaded once) ---
        self._lb = jnp.array(bounds_np[:, 0], dtype=jnp.float64)
        self._ub = jnp.array(bounds_np[:, 1], dtype=jnp.float64)

        q, y, y_err = objective_data(objective)
        self._q    = jnp.array(q,     dtype=jnp.float64)
        self._y    = jnp.array(y,     dtype=jnp.float64)
        self._yerr = jnp.array(y_err, dtype=jnp.float64)

        base_slabs, base_scale, base_bkg = get_slabs_scale_bkg(objective)
        self._base_slabs = jnp.array(base_slabs, dtype=jnp.float64)
        self._base_scale = float(base_scale)
        self._base_bkg   = float(base_bkg)

        # --- Smearing / log-space detection ---
        use_log, dqvals_fwhm, quad_order = _detect_objective_fitness_mode(objective)
        self._use_log = use_log

        if dqvals_fwhm is not None:
            q_quad, combo_weights = compute_smear_params(q, dqvals_fwhm, quad_order)
            self._q_quad = jnp.array(q_quad,       dtype=jnp.float64)
            self._combo  = jnp.array(combo_weights, dtype=jnp.float64)
        else:
            self._q_quad = None
            self._combo  = None

        # --- Param map ---
        self._param_map = _build_param_map(objective, self.free_params)

        # --- Initial unconstrained position ---
        theta_raw = np.array(get_free_param_values(objective), dtype=np.float64)

        # Detect and warn about parameters outside their bounds — then clip.
        # Out-of-bounds values produce extreme logit values that corrupt the
        # log-prob and gradient, causing NUTS/MCLMC to reject every proposal.
        out_of_bounds = [
            self.param_names[i]
            for i in range(self.n_free)
            if theta_raw[i] < bounds_np[i, 0] or theta_raw[i] > bounds_np[i, 1]
        ]
        if out_of_bounds:
            import warnings
            clipped = {}
            for i in range(self.n_free):
                name = self.param_names[i]
                if name in out_of_bounds:
                    clipped[name] = (
                        float(theta_raw[i]),
                        float(np.clip(theta_raw[i], bounds_np[i, 0] + 1e-10*(bounds_np[i,1]-bounds_np[i,0]),
                                                       bounds_np[i, 1] - 1e-10*(bounds_np[i,1]-bounds_np[i,0])))
                    )
            warnings.warn(
                f"Parameters outside their bounds will be clipped to the interior: "
                f"{clipped}. Fix the model's parameter values or bounds before sampling.",
                stacklevel=2,
            )
        theta_init = jnp.clip(
            jnp.array(theta_raw, dtype=jnp.float64),
            self._lb + 1e-10 * (self._ub - self._lb),
            self._ub - 1e-10 * (self._ub - self._lb),
        )
        self._y_init = _bounded_to_unconstrained(theta_init, self._lb, self._ub)

        # Warn if any parameter sits very close to a bound after clipping
        near_bound = [
            self.param_names[i]
            for i in range(self.n_free)
            if abs(float(self._y_init[i])) > 5.0
        ]
        if near_bound:
            import warnings
            warnings.warn(
                f"Parameters {near_bound} are near their bounds (|logit| > 5). "
                "Consider widening the bounds for better MCMC mixing.",
                stacklevel=2,
            )

        # --- Log-probability function ---
        self.log_prob_fn = make_log_prob_fn(
            self._q, self._y, self._yerr,
            self._base_slabs, self._base_scale, self._base_bkg,
            self._lb, self._ub,
            self._param_map,
            use_log=use_log,
            q_quad_j=self._q_quad,
            combo_weights_j=self._combo,
            normalize_by_n_q=normalize,
        )

        # Set by warmup()
        self._adapted_params = None
        self._adapted_state  = None

    # ------------------------------------------------------------------
    # Warmup / adaptation
    # ------------------------------------------------------------------

    def warmup(self, n_steps=1000, initial_position=None, verbose=True):
        """
        Run sampler adaptation on a single chain.

        NUTS: window_adaptation (dual averaging + Welford mass matrix).
        MCLMC: mclmc_find_L_and_step_size.

        Parameters
        ----------
        n_steps          : adaptation steps
        initial_position : (n_free,) unconstrained array, or None for MAP start
        verbose          : print adapted hyperparameters

        Returns
        -------
        self (for chaining)
        """
        y0 = (
            jnp.array(initial_position, dtype=jnp.float64)
            if initial_position is not None
            else self._y_init
        )

        self._master_key, warmup_key = jax.random.split(self._master_key)

        if self.sampler_name == 'nuts':
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                self.log_prob_fn,
                is_mass_matrix_diagonal=True,
                target_acceptance_rate=0.80,
                progress_bar=False,
            )
            (last_state, parameters), _ = warmup.run(warmup_key, y0, num_steps=n_steps)
            self._adapted_params = parameters
            self._adapted_state  = last_state
            if verbose:
                ss = float(parameters['step_size'])
                imm = np.asarray(parameters['inverse_mass_matrix'])
                print(
                    f"[NUTS warmup] step_size={ss:.5f}, "
                    f"inv_mass_matrix ∈ [{imm.min():.4f}, {imm.max():.4f}]"
                )

        elif self.sampler_name == 'mclmc':
            self._master_key, init_key = jax.random.split(self._master_key)
            initial_state = blackjax.mclmc.init(y0, self.log_prob_fn, init_key)

            # mclmc_find_L_and_step_size expects a FACTORY: imm → kernel
            # (it calls factory(imm) each adaptation step to rebuild the kernel)
            log_prob_fn = self.log_prob_fn
            def kernel_factory(inverse_mass_matrix):
                return blackjax.mclmc.build_kernel(
                    log_prob_fn,
                    inverse_mass_matrix,
                    _integrators.isokinetic_mclachlan,
                )

            tuned_state, params, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel_factory,
                num_steps=n_steps,
                state=initial_state,
                rng_key=warmup_key,
                diagonal_preconditioning=False,
            )
            self._adapted_params = params   # MCLMCAdaptationState(L, step_size, imm)
            self._adapted_state  = tuned_state
            if verbose:
                print(
                    f"[MCLMC warmup] L={float(params.L):.5f}, "
                    f"step_size={float(params.step_size):.5f}"
                )

        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples=2000, n_warmup=1000, initial_position=None,
               progress=True, chunk_size=100):
        """
        Draw posterior samples (running warmup first if not done).

        Parameters
        ----------
        n_samples        : post-warmup samples per chain
        n_warmup         : adaptation steps (ignored if warmup() already called)
        initial_position : unconstrained starting position (optional)
        progress         : show tqdm progress bar
        chunk_size       : scan chunk size for progress update granularity

        Returns
        -------
        MCMCSamples
        """
        if self._adapted_params is None:
            self.warmup(
                n_steps=n_warmup,
                initial_position=initial_position,
                verbose=progress,
            )

        self._master_key, init_key, sample_key = jax.random.split(self._master_key, 3)

        # Start chains from the warmup endpoint — it's guaranteed to be inside
        # the posterior typical set.  Perturbing the raw y_init can place chains
        # far from the typical set (especially when parameters are near their
        # bounds), causing 100 % proposal rejection.
        if initial_position is not None:
            y0 = jnp.array(initial_position, dtype=jnp.float64)
        elif self._adapted_state is not None:
            y0 = self._adapted_state.position   # warmup endpoint
        else:
            y0 = self._y_init
        init_positions = _make_init_positions(y0, self.n_chains, init_key)

        # ── Build sampler-specific states and one-step function ────────────
        if self.sampler_name == 'nuts':
            params = self._adapted_params
            kernel = blackjax.nuts(self.log_prob_fn, **params).step

            init_states = jax.vmap(
                lambda p: blackjax.nuts.init(p, self.log_prob_fn)
            )(init_positions)

            n_chains = self.n_chains

            @jax.jit
            def one_step_all_chains(states, rng_key):
                ck = jax.random.split(rng_key, n_chains)
                new_states, infos = jax.vmap(kernel)(ck, states)
                return new_states, (
                    new_states.position,
                    infos.acceptance_rate,
                    infos.is_divergent,
                    infos.num_integration_steps,
                )

            extra_keys = ('acceptance_rate', 'is_divergent', 'num_integration_steps')

        else:  # mclmc
            params    = self._adapted_params
            L         = params.L
            step_size = params.step_size
            imm       = params.inverse_mass_matrix

            final_kernel = blackjax.mclmc.build_kernel(
                self.log_prob_fn,
                imm,
                _integrators.isokinetic_mclachlan,
            )

            mclmc_init_keys = jax.random.split(init_key, self.n_chains)
            init_states = jax.vmap(
                lambda p, k: blackjax.mclmc.init(p, self.log_prob_fn, k)
            )(init_positions, mclmc_init_keys)

            n_chains = self.n_chains

            @jax.jit
            def one_step_all_chains(states, rng_key):
                ck = jax.random.split(rng_key, n_chains)
                new_states, infos = jax.vmap(
                    lambda k, s: final_kernel(k, s, L, step_size)
                )(ck, states)
                return new_states, (
                    new_states.position,
                    infos.energy_change,
                )

            extra_keys = ('energy_change',)

        # ── Sampling loop ──────────────────────────────────────────────────
        # NUTS uses lax.while_loop internally for tree-building.  Nesting
        # lax.scan(vmap(while_loop)) forces XLA to unroll the dynamic loop
        # across all scan iterations, producing a computation graph that can
        # take hours to compile even for short runs.  For NUTS we therefore
        # use a plain Python loop (effective chunk_size=1) so each JIT-compiled
        # step is a single NUTS proposal.  MCLMC has a fixed computation graph
        # so lax.scan compiles quickly and gives better GPU utilisation.
        _use_scan = (self.sampler_name != 'nuts')
        _chunk    = chunk_size if _use_scan else 1

        n_chunks     = n_samples // _chunk
        remainder    = n_samples % _chunk
        n_chunks_run = n_chunks + (1 if remainder else 0)
        n_total      = n_chunks_run * _chunk

        if _use_scan:
            @jax.jit
            def run_chunk(states, keys):
                return jax.lax.scan(one_step_all_chains, states, keys)
        else:
            # Single-step JIT — avoids scan(while_loop) compilation catastrophe
            one_step_jit = jax.jit(one_step_all_chains)

            def run_chunk(states, keys):
                # keys has shape (1, 2) — unwrap the single key
                single_key = keys[0]
                new_states, out = one_step_jit(states, single_key)
                # Wrap outputs to match the (chunk_size, ...) shape expected below
                return new_states, tuple(o[None] for o in out)

        all_positions = []
        all_extras    = {k: [] for k in extra_keys}
        states        = init_states

        pbar = _make_pbar(n_total, progress, desc=f'{self.sampler_name.upper()} sampling')
        try:
            for _ in range(n_chunks_run):
                sample_key, subkey = jax.random.split(sample_key)
                chunk_keys = jax.random.split(subkey, _chunk)
                states, chunk_out = run_chunk(states, chunk_keys)
                positions_chunk = chunk_out[0]
                positions_chunk.block_until_ready()
                all_positions.append(np.array(positions_chunk))
                for idx, k in enumerate(extra_keys):
                    all_extras[k].append(np.array(chunk_out[idx + 1]))
                if pbar is not None:
                    pbar.update(_chunk)
        finally:
            if pbar is not None:
                pbar.close()

        # ── Assemble results ──────────────────────────────────────────────
        # positions after scan: each element is (chunk_size, n_chains, n_free)
        # concatenate along axis 0 → (n_total, n_chains, n_free), trim, transpose
        positions = np.concatenate(all_positions, axis=0)[:n_samples]   # (n_samples, n_chains, n_free)
        positions = positions.transpose(1, 0, 2)                         # (n_chains, n_samples, n_free)

        extras = {
            k: np.concatenate(v, axis=0)[:n_samples]
            for k, v in all_extras.items()
        }

        # Inverse-logit back to bounded space: (n_chains, n_samples, n_free)
        bounded = np.array(
            self._lb.reshape(1, 1, -1)
            + jax.nn.sigmoid(jnp.array(positions))
            * (self._ub - self._lb).reshape(1, 1, -1)
        )

        return MCMCSamples(
            samples_unconstrained=positions,
            samples_bounded=bounded,
            param_names=self.param_names,
            sampler_name=self.sampler_name,
            extras=extras,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def gpu_mcmc(
    objective,
    sampler    = 'nuts',
    n_chains   = 4,
    n_warmup   = 1000,
    n_samples  = 2000,
    seed       = 0,
    progress   = True,
    chunk_size = 100,
    normalize  = False,
):
    """
    GPU-accelerated Bayesian MCMC for a refnx Objective.

    Samples the posterior distribution of the objective's free parameters
    using BlackJAX on GPU. The objective should be at best-fit values
    (e.g., after differential evolution or CMA-ES) for good warmup
    initialisation.

    Parameters
    ----------
    objective  : refnx Objective
    sampler    : 'nuts' (default) | 'mclmc'
        NUTS uses gradient-based HMC with auto-tuning (window_adaptation).
        MCLMC uses isokinetic Langevin dynamics, often faster for smooth
        posteriors.
    n_chains   : int  — parallel chains (typically 4)
    n_warmup   : int  — adaptation steps
    n_samples  : int  — posterior samples per chain
    seed       : int  — random seed
    progress   : bool — show tqdm progress bar
    chunk_size : int  — scan chunk size (100 gives ~1 Hz progress updates)

    Returns
    -------
    MCMCSamples
        .samples_bounded       ndarray (n_chains, n_samples, n_free)
        .summary()             pandas DataFrame — mean, std, CI, R-hat, ESS
        .trace_plot()          matplotlib Figure
        .corner_plot()         corner Figure  (requires `corner`)
        .to_arviz()            arviz InferenceData  (requires `arviz`)
        .to_refnx_objective()  set objective params to posterior median
        .acceptance_rate()     float (NUTS only)
        .divergence_fraction() float (NUTS only)
        .energy_change_variance() float (MCLMC only)

    Examples
    --------
    >>> results = gpu_mcmc(objective, sampler='nuts', n_chains=4,
    ...                    n_warmup=500, n_samples=1000)
    >>> print(results.summary())
    >>> results.trace_plot()
    >>> results.to_refnx_objective(objective)
    """
    s = BlackjaxSampler(objective, sampler=sampler, n_chains=n_chains, seed=seed,
                        normalize=normalize)
    return s.sample(
        n_samples=n_samples,
        n_warmup=n_warmup,
        progress=progress,
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Convergence-targeted sampler
# ---------------------------------------------------------------------------

def gpu_mcmc_converge(
    objective,
    sampler          = 'nuts',
    n_chains         = 4,
    n_warmup         = 500,
    samples_per_chunk= 200,
    min_samples      = 500,
    max_samples      = 5000,
    rhat_threshold   = 1.05,
    ess_min          = 200,
    seed             = 0,
    progress         = True,
    chunk_size       = 100,
    normalize        = False,
):
    """
    Run MCMC until R-hat convergence rather than a fixed sample count.

    Warmup runs once; then samples are drawn in chunks of ``samples_per_chunk``.
    After each chunk, R-hat and ESS are evaluated for every free parameter.
    Sampling stops when:

        max(R-hat) < rhat_threshold  AND  min(ESS) >= ess_min
        AND  total samples >= min_samples

    or when ``max_samples`` is reached.

    Parameters
    ----------
    objective         : refnx Objective at best-fit parameters (e.g., after CMA-ES)
    sampler           : 'nuts' | 'mclmc'
    n_chains          : parallel chains (default 4)
    n_warmup          : warmup / adaptation steps (default 500)
    samples_per_chunk : samples collected between convergence checks (default 200)
    min_samples       : minimum samples before checking convergence (default 500)
    max_samples       : hard upper limit on total samples (default 5000)
    rhat_threshold    : convergence criterion for R-hat (default 1.05)
    ess_min           : minimum per-parameter ESS to accept (default 200)
    seed              : PRNG seed
    progress          : show tqdm progress bar
    chunk_size        : internal lax.scan chunk size for MCLMC (ignored for NUTS)

    Returns
    -------
    MCMCSamples — same object as gpu_mcmc(), with an extra attribute
        .converged : bool — True if R-hat threshold was met before max_samples
    """
    import time

    s = BlackjaxSampler(objective, sampler=sampler, n_chains=n_chains, seed=seed,
                        normalize=normalize)

    # ── Warmup (once) ──────────────────────────────────────────────────────
    if progress:
        print(f"[{sampler.upper()}] Warmup ({n_warmup} steps)...")
    t0 = time.perf_counter()
    s.warmup(n_steps=n_warmup, verbose=progress)
    if progress:
        print(f"  warmup done in {time.perf_counter()-t0:.1f}s")

    # ── Build the one-step function (mirrors BlackjaxSampler.sample logic) ──
    sample_key = jax.random.PRNGKey(seed + 1)
    sample_key, init_key = jax.random.split(sample_key)
    y0 = s._adapted_state.position
    init_positions = _make_init_positions(y0, n_chains, init_key)

    if sampler == 'nuts':
        params = s._adapted_params
        kernel = blackjax.nuts(s.log_prob_fn, **params).step
        init_states = jax.vmap(
            lambda p: blackjax.nuts.init(p, s.log_prob_fn)
        )(init_positions)

        @jax.jit
        def _one_step(states, rng_key):
            ck = jax.random.split(rng_key, n_chains)
            new_states, infos = jax.vmap(kernel)(ck, states)
            return new_states, (
                new_states.position,
                infos.acceptance_rate,
                infos.is_divergent,
                infos.num_integration_steps,
            )
        extra_keys = ('acceptance_rate', 'is_divergent', 'num_integration_steps')

    else:  # mclmc
        from blackjax.mcmc import integrators as _int
        params    = s._adapted_params
        L         = params.L
        step_size = params.step_size
        imm       = params.inverse_mass_matrix

        final_kernel = blackjax.mclmc.build_kernel(
            s.log_prob_fn, imm, _int.isokinetic_mclachlan)
        mclmc_init_keys = jax.random.split(init_key, n_chains)
        init_states = jax.vmap(
            lambda p, k: blackjax.mclmc.init(p, s.log_prob_fn, k)
        )(init_positions, mclmc_init_keys)

        @jax.jit
        def _one_step(states, rng_key):
            ck = jax.random.split(rng_key, n_chains)
            new_states, infos = jax.vmap(
                lambda k, st: final_kernel(k, st, L, step_size)
            )(ck, states)
            return new_states, (new_states.position, infos.energy_change)
        extra_keys = ('energy_change',)

    # For MCLMC use lax.scan over chunks; for NUTS use single-step loop (avoids
    # scan(while_loop) XLA compilation catastrophe).
    _use_scan = (sampler != 'nuts')
    _chunk    = chunk_size if _use_scan else 1

    if _use_scan:
        @jax.jit
        def _run_chunk(states, keys):
            return jax.lax.scan(_one_step, states, keys)
    else:
        _one_step_jit = jax.jit(_one_step)

        def _run_chunk(states, keys):
            new_states, out = _one_step_jit(states, keys[0])
            return new_states, tuple(o[None] for o in out)

    # ── Convergence loop ───────────────────────────────────────────────────
    all_positions = []
    all_extras    = {k: [] for k in extra_keys}
    states        = init_states
    total_samples = 0
    converged     = False

    pbar = _make_pbar(max_samples, progress,
                      desc=f'{sampler.upper()} sampling (until convergence)')
    try:
        while total_samples < max_samples:
            # Draw one chunk of samples_per_chunk
            chunk_positions = []
            chunk_extras    = {k: [] for k in extra_keys}

            steps_this_chunk = 0
            while steps_this_chunk < samples_per_chunk:
                sample_key, subkey = jax.random.split(sample_key)
                batch_keys = jax.random.split(subkey, _chunk)
                states, out = _run_chunk(states, batch_keys)
                out[0].block_until_ready()
                chunk_positions.append(np.array(out[0]))   # (_chunk, n_chains, n_free)
                for idx, k in enumerate(extra_keys):
                    chunk_extras[k].append(np.array(out[idx + 1]))
                steps_this_chunk += _chunk
                if pbar is not None:
                    pbar.update(_chunk)

            all_positions.append(
                np.concatenate(chunk_positions, axis=0)[:samples_per_chunk])
            for k in extra_keys:
                all_extras[k].append(
                    np.concatenate(chunk_extras[k], axis=0)[:samples_per_chunk])

            total_samples += samples_per_chunk

            if total_samples < min_samples:
                continue

            # ── Check R-hat and ESS ──────────────────────────────────────
            # positions: list of (samples_per_chunk, n_chains, n_free) → stack
            pos_all = np.concatenate(all_positions, axis=0)  # (total, n_chains, n_free)
            # rearrange to (n_chains, total, n_free)
            pos_chains = pos_all.transpose(1, 0, 2)
            bounded_chains = (
                np.array(s._lb).reshape(1, 1, -1)
                + jax.nn.sigmoid(jnp.array(pos_chains))
                * (np.array(s._ub) - np.array(s._lb)).reshape(1, 1, -1)
            )

            rhats = []
            esss  = []
            for i in range(s.n_free):
                chain_i = jnp.array(bounded_chains[:, :, i])  # (n_chains, total)
                rhats.append(float(blackjax.rhat(chain_i, chain_axis=0, sample_axis=1)))
                esss.append(float(blackjax.ess(chain_i,  chain_axis=0, sample_axis=1)))

            rhat_max = max(rhats)
            ess_min_val = min(esss)

            if progress:
                print(f"  {total_samples} samples — "
                      f"R-hat max={rhat_max:.4f}  ESS min={ess_min_val:.0f}")

            if rhat_max < rhat_threshold and ess_min_val >= ess_min:
                converged = True
                break

    finally:
        if pbar is not None:
            pbar.close()

    if progress:
        status = "CONVERGED" if converged else f"MAX SAMPLES ({max_samples}) REACHED"
        print(f"[{sampler.upper()}] {status} — {total_samples} samples")

    # ── Assemble final MCMCSamples ─────────────────────────────────────────
    pos_all    = np.concatenate(all_positions, axis=0)[:total_samples]
    positions  = pos_all.transpose(1, 0, 2)       # (n_chains, total_samples, n_free)

    extras = {}
    for k in extra_keys:
        arr = np.concatenate(all_extras[k], axis=0)[:total_samples]
        extras[k] = arr.transpose(1, 0) if arr.ndim == 2 else arr

    bounded = np.array(
        s._lb.reshape(1, 1, -1)
        + jax.nn.sigmoid(jnp.array(positions))
        * (s._ub - s._lb).reshape(1, 1, -1)
    )

    result = MCMCSamples(
        samples_unconstrained=positions,
        samples_bounded=bounded,
        param_names=s.param_names,
        sampler_name=sampler,
        extras=extras,
    )
    result.converged = converged
    result.total_samples = total_samples
    return result
