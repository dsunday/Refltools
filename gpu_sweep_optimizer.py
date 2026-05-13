"""
GPU-accelerated parameter sweep optimizer for refnx.

Replaces the sequential run_parameter_sweep (scipy DE per sweep point) with
a batched evosax DE that optimizes all sweep points simultaneously on GPU.

The key operation:
  - n_sweep independent DE instances (one per sweep-parameter value)
  - Each generation evaluates n_sweep × popsize reflectivity curves in one GPU call
  - All populations are updated simultaneously via JAX vmap

Expected speedup over sequential scipy DE (8 workers):
  ~10-50× depending on n_sweep, popsize, model complexity, and GPU.
"""

import os
import copy
import time
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import jit, vmap

jax.config.update("jax_enable_x64", True)

from evosax.algorithms import DifferentialEvolution
from gpu_reflect import (
    ModelsBatchBuilder,
    MultiEnergySweepBatchBuilder,
    SweepBatchBuilder,
    extract_free_params,
    get_bounds_array,
    get_free_param_values,
    get_slabs_scale_bkg,
    set_free_params,
)


# ---------------------------------------------------------------------------
# Evosax DE helpers
# ---------------------------------------------------------------------------

def _make_strategy(n_free, popsize, scale_factor=0.8, crossover_prob=0.3):
    """
    Create a DifferentialEvolution strategy.

    evosax 0.2.0 API:
      strategy.init(key, init_pop, init_fitness, params)
      strategy.ask(key, state, params)
      strategy.tell(key, population, fitness, state, params)

    Note: bounds are NOT enforced by evosax; clip candidates after ask().
    """
    solution_prototype = np.zeros(n_free, dtype=np.float64)
    strategy = DifferentialEvolution(
        population_size=popsize,
        solution=solution_prototype,
    )
    params = strategy.default_params.replace(
        differential_weight=scale_factor,
        crossover_rate=crossover_prob,
    )
    return strategy, params


def _uniform_init_population(key, n_sweep, popsize, bounds):
    """
    Sample initial populations for n_sweep independent DE instances.

    Returns shape (n_sweep, popsize, n_free).
    Samples uniformly within parameter bounds.
    """
    n_free = bounds.shape[0]
    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])
    keys = jax.random.split(key, n_sweep)
    # (n_sweep, popsize, n_free)
    raw = jax.vmap(
        lambda k: jax.random.uniform(k, (popsize, n_free), dtype=jnp.float64)
    )(keys)
    return raw * (ub - lb) + lb


# ---------------------------------------------------------------------------
# Core GPU sweep optimizer
# ---------------------------------------------------------------------------

def gpu_parameter_sweep(
    objective,
    param_name,
    sweep_values,
    popsize=20,
    n_generations=200,
    seed=0,
    verbose=False,
):
    """
    Run all sweep-point DE optimizations simultaneously on GPU.

    For each value v in sweep_values:
      - Fix param_name = v
      - Optimize remaining free parameters to minimise chi-squared

    Parameters
    ----------
    objective    : refnx Objective at best-fit parameters
    param_name   : name of the parameter to sweep (str)
    sweep_values : 1D array-like of sweep values
    popsize      : DE population size per sweep point (default 20)
    n_generations: number of DE generations (default 200)
    seed         : random seed for reproducibility
    verbose      : if True, print progress every 20 generations

    Returns
    -------
    dict with keys:
      'sweep_values'   : array of sweep values
      'best_chisqr'    : (n_sweep,) best chi-squared achieved at each sweep value
      'best_params'    : (n_sweep, n_free) best free-parameter values
      'free_param_names': list of free parameter names
      'param_name'     : the swept parameter name
      'elapsed_sec'    : wall time in seconds
    """
    sweep_values = np.asarray(sweep_values, dtype=np.float64)
    n_sweep = len(sweep_values)

    # Build batch evaluator (deep-copies objectives, fixes swept param)
    builder = SweepBatchBuilder(objective, param_name, sweep_values)
    n_free = builder.n_free
    bounds = builder.bounds  # (n_free, 2)

    if n_free == 0:
        raise ValueError(
            f"No free parameters after excluding '{param_name}'. "
            "Check that other parameters have vary=True."
        )

    # Create n_sweep independent DE instances (vmapped)
    strategy, params = _make_strategy(n_free, popsize)
    lb = jnp.array(bounds[:, 0])  # (n_free,) lower bounds
    ub = jnp.array(bounds[:, 1])  # (n_free,) upper bounds

    v_init = jax.jit(jax.vmap(strategy.init, in_axes=(0, 0, 0, None)))
    v_ask = jax.jit(jax.vmap(strategy.ask, in_axes=(0, 0, None)))
    v_tell = jax.jit(jax.vmap(strategy.tell, in_axes=(0, 0, 0, 0, None)))

    # Initialise populations
    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)
    init_keys = jax.random.split(init_key, n_sweep)

    # Sample initial populations uniformly within bounds
    init_pops = np.array(
        _uniform_init_population(init_key, n_sweep, popsize, bounds)
    )

    # Evaluate initial fitness to initialise evosax state
    init_fitness = builder.fitness(init_pops)  # (n_sweep, popsize)

    states = v_init(
        init_keys,
        jnp.array(init_pops, dtype=jnp.float64),
        jnp.array(init_fitness, dtype=jnp.float64),
        params,
    )

    t_start = time.perf_counter()
    for gen in range(n_generations):
        # Generate fresh per-sweep keys each generation for DE randomisation
        master_key, gen_key = jax.random.split(master_key)
        keys = jax.random.split(gen_key, n_sweep)

        # Ask: propose new candidates for all sweep points simultaneously
        populations, states = v_ask(keys, states, params)
        # populations: (n_sweep, popsize, n_free)

        # Clip to bounds (evosax DE does not enforce bounds natively)
        populations = jnp.clip(populations, lb, ub)

        # Evaluate all n_sweep * popsize candidates on GPU
        pop_np = np.array(populations)  # (n_sweep, popsize, n_free)
        fitness = builder.fitness(pop_np)  # (n_sweep, popsize)

        # Tell: update all populations with fitness scores
        states, _ = v_tell(
            keys,
            populations,
            jnp.array(fitness, dtype=jnp.float64),
            states,
            params,
        )

        if verbose and (gen + 1) % 20 == 0:
            best = np.array(states.best_fitness)
            print(
                f"  gen {gen+1:3d}/{n_generations}  "
                f"mean_best={best.mean():.2f}  "
                f"min_best={best.min():.2f}"
            )

    elapsed = time.perf_counter() - t_start

    best_chisqr = np.array(states.best_fitness)
    best_solutions = np.array(jax.vmap(strategy.get_best_solution)(states))

    return {
        "sweep_values": sweep_values,
        "best_chisqr": best_chisqr,
        "best_params": best_solutions,
        "free_param_names": [p.name for p in builder.free_params],
        "param_name": param_name,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Sep-CMA-ES parameter sweep
# ---------------------------------------------------------------------------

def gpu_parameter_sweep_cmaes(
    objective,
    param_name,
    sweep_values,
    popsize=20,
    n_generations=300,
    seed=0,
    verbose=False,
    tol=1e-4,
    patience=5,
    check_every=10,
):
    """
    Run all sweep-point optimizations simultaneously using Sep-CMA-ES.

    Same interface as gpu_parameter_sweep but uses Sep-CMA-ES instead of DE,
    with optional early stopping. Typically reaches lower chi-squared in fewer
    generations on reflectometry problems.

    Parameters
    ----------
    objective    : refnx Objective at best-fit parameters
    param_name   : name of the parameter to sweep
    sweep_values : 1D array of sweep values
    popsize      : population size per sweep point (default 20)
    n_generations: maximum generations (default 300)
    seed         : random seed
    verbose      : print progress every check_every generations
    tol          : relative improvement threshold for early stopping (0 = off)
    patience     : consecutive non-improving checks before stopping
    check_every  : generations between convergence checks

    Returns
    -------
    Same dict as gpu_parameter_sweep:
      sweep_values, best_chisqr, best_params, free_param_names,
      param_name, elapsed_sec, generations_run, converged
    """
    from evosax.algorithms import Sep_CMA_ES

    sweep_values = np.asarray(sweep_values, dtype=np.float64)
    n_sweep = len(sweep_values)

    builder = SweepBatchBuilder(objective, param_name, sweep_values)
    n_free = builder.n_free
    bounds = builder.bounds

    if n_free == 0:
        raise ValueError(
            f"No free parameters after excluding '{param_name}'. "
            "Check that other parameters have vary=True."
        )

    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])
    mean_init = (lb + ub) / 2.0

    solution_proto = np.zeros(n_free, dtype=np.float64)
    strategy = Sep_CMA_ES(population_size=popsize, solution=solution_proto)
    params = strategy.default_params

    v_init = jax.jit(jax.vmap(strategy.init, in_axes=(0, None, None)))
    v_ask  = jax.jit(jax.vmap(strategy.ask,  in_axes=(0, 0, None)))
    v_tell = jax.jit(jax.vmap(strategy.tell, in_axes=(0, 0, 0, 0, None)))

    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)
    init_keys = jax.random.split(init_key, n_sweep)

    states = v_init(init_keys, mean_init, params)

    prev_mean_best = np.inf
    no_improve_count = 0
    converged = False
    generations_run = 0

    t_start = time.perf_counter()
    for gen in range(n_generations):
        master_key, gen_key = jax.random.split(master_key)
        keys = jax.random.split(gen_key, n_sweep)

        populations, states = v_ask(keys, states, params)
        populations = jnp.clip(populations, lb, ub)
        pop_np = np.array(populations)
        fitness = builder.fitness(pop_np)
        states, _ = v_tell(
            keys, populations,
            jnp.array(fitness, dtype=jnp.float64),
            states, params,
        )
        generations_run = gen + 1

        if (gen + 1) % check_every == 0:
            best = np.array(states.best_fitness)
            mean_best = float(best.mean())

            if verbose:
                print(
                    f"  gen {gen+1:4d}/{n_generations}  "
                    f"mean_χ²={mean_best:.2f}  "
                    f"min_χ²={float(best.min()):.2f}"
                )

            if tol > 0 and np.isfinite(prev_mean_best):
                rel_improvement = (prev_mean_best - mean_best) / max(prev_mean_best, 1e-12)
                if rel_improvement < tol:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        if verbose:
                            print(f"  Early stop at gen {gen+1}.")
                        converged = True
                        break
                else:
                    no_improve_count = 0
            prev_mean_best = mean_best

    elapsed = time.perf_counter() - t_start

    best_chisqr = np.array(states.best_fitness)
    best_solutions = np.array(states.best_solution)

    return {
        "sweep_values":    sweep_values,
        "best_chisqr":     best_chisqr,
        "best_params":     best_solutions,
        "free_param_names": [p.name for p in builder.free_params],
        "param_name":      param_name,
        "elapsed_sec":     elapsed,
        "generations_run": generations_run,
        "converged":       converged,
    }


# ---------------------------------------------------------------------------
# Multi-energy batched CMA-ES sweep
# ---------------------------------------------------------------------------

def gpu_parameter_sweep_cmaes_batched_energies(
    objectives_dict,
    param_name,
    sweep_values_per_energy,
    popsize=20,
    n_generations=300,
    seed=0,
    verbose=False,
    tol=1e-4,
    patience=5,
    check_every=10,
):
    """
    Run parameter sweep for all energies simultaneously using Sep-CMA-ES.

    Replaces N_energies sequential calls to gpu_parameter_sweep_cmaes with a
    single GPU run that batches n_energies × max_n_sweep optimization instances
    together.  Energies with different sweep grid sizes are zero-padded.

    Parameters
    ----------
    objectives_dict          : {energy: refnx Objective} at best-fit values
    param_name               : parameter to sweep (held fixed per slot)
    sweep_values_per_energy  : {energy: np.ndarray} — sweep grid per energy
    popsize                  : population size per instance (default 20)
    n_generations            : maximum generations (default 300)
    seed                     : random seed
    verbose                  : print progress every check_every generations
    tol                      : relative improvement threshold for early stopping
    patience                 : consecutive non-improving checks before stop
    check_every              : generations between convergence checks

    Returns
    -------
    dict {energy: result} where each result has the same keys as
    gpu_parameter_sweep_cmaes:
        sweep_values, best_chisqr, best_params, free_param_names,
        param_name, elapsed_sec, generations_run, converged
    """
    from evosax.algorithms import Sep_CMA_ES

    energies = sorted(objectives_dict.keys())

    builder = MultiEnergySweepBatchBuilder(
        objectives_dict, param_name, sweep_values_per_energy
    )
    n_total = builder.n_total
    n_free  = builder.n_free
    bounds  = builder.bounds

    if n_free == 0:
        raise ValueError(
            f"No free parameters after excluding '{param_name}'. "
            "Check that other parameters have vary=True."
        )

    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])
    mean_init = (lb + ub) / 2.0

    solution_proto = np.zeros(n_free, dtype=np.float64)
    strategy = Sep_CMA_ES(population_size=popsize, solution=solution_proto)
    params = strategy.default_params

    v_init = jax.jit(jax.vmap(strategy.init, in_axes=(0, None, None)))
    v_ask  = jax.jit(jax.vmap(strategy.ask,  in_axes=(0, 0, None)))
    v_tell = jax.jit(jax.vmap(strategy.tell, in_axes=(0, 0, 0, 0, None)))

    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)
    init_keys = jax.random.split(init_key, n_total)

    states = v_init(init_keys, mean_init, params)

    prev_mean_best  = np.inf
    no_improve_count = 0
    converged       = False
    generations_run = 0

    t_start = time.perf_counter()
    for gen in range(n_generations):
        master_key, gen_key = jax.random.split(master_key)
        keys = jax.random.split(gen_key, n_total)

        populations, states = v_ask(keys, states, params)
        populations = jnp.clip(populations, lb, ub)
        pop_np = np.array(populations)
        fitness = builder.fitness(pop_np)  # (n_total, popsize)
        states, _ = v_tell(
            keys, populations,
            jnp.array(fitness, dtype=jnp.float64),
            states, params,
        )
        generations_run = gen + 1

        if (gen + 1) % check_every == 0:
            best_all  = np.array(states.best_fitness)  # (n_total,)
            valid_best = best_all[builder.valid_mask]
            mean_best  = float(valid_best.mean())

            if verbose:
                print(
                    f"  gen {gen+1:4d}/{n_generations}  "
                    f"mean_χ²={mean_best:.2f}  "
                    f"min_χ²={float(valid_best.min()):.2f}  "
                    f"[{len(energies)} energies × {builder.max_n_sweep} sweep pts]"
                )

            if tol > 0 and np.isfinite(prev_mean_best):
                rel_imp = (prev_mean_best - mean_best) / max(prev_mean_best, 1e-12)
                if rel_imp < tol:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        if verbose:
                            print(f"  Early stop at gen {gen+1}.")
                        converged = True
                        break
                else:
                    no_improve_count = 0
            prev_mean_best = mean_best

    elapsed = time.perf_counter() - t_start

    best_chisqr_all   = np.array(states.best_fitness)   # (n_total,)
    best_solutions_all = np.array(states.best_solution)  # (n_total, n_free)

    results = {}
    for i, e in enumerate(energies):
        sv  = np.asarray(sweep_values_per_energy[e], dtype=np.float64)
        ns  = len(sv)
        start = i * builder.max_n_sweep
        results[e] = {
            "sweep_values":     sv,
            "best_chisqr":      best_chisqr_all[start : start + ns],
            "best_params":      best_solutions_all[start : start + ns],
            "free_param_names": [p.name for p in builder.free_params],
            "param_name":       param_name,
            "elapsed_sec":      elapsed,
            "generations_run":  generations_run,
            "converged":        converged,
        }
    return results


# ---------------------------------------------------------------------------
# Result conversion: GPU sweep result → refnx sweep_info format
# ---------------------------------------------------------------------------

def gpu_result_to_sweep_info(gpu_result, objective, param_name, sweep_values):
    """
    Convert gpu_parameter_sweep output to the sweep_info dict format used by
    get_best_fit_with_uncertainty() in parameter_sweep.py.

    Parameters
    ----------
    gpu_result    : dict returned by gpu_parameter_sweep
    objective     : the refnx Objective used in the sweep
    param_name    : swept parameter name
    sweep_values  : array of sweep values (same as gpu_result['sweep_values'])

    Returns
    -------
    sweep_info : dict compatible with get_best_fit_with_uncertainty()
    """
    sweep_values = np.asarray(sweep_values, dtype=np.float64)
    best_params = gpu_result["best_params"]   # (n_sweep, n_free)
    best_chisqr = gpu_result["best_chisqr"]   # (n_sweep,)

    best_idx = int(np.argmin(best_chisqr))

    # Reconstruct parameter values at each sweep point for storage
    obj_copy = copy.deepcopy(objective)
    sweep_data = []
    for i, v in enumerate(sweep_values):
        for p in obj_copy.parameters.flattened():
            if p.name == param_name:
                p.value = float(v)
                break
        set_free_params(obj_copy, best_params[i], excluded_name=param_name)
        all_params = {
            p.name: float(p.value)
            for p in obj_copy.parameters.flattened()
        }
        sweep_data.append({
            "param_value": float(v),
            "chisqr": float(best_chisqr[i]),
            "params": all_params,
        })

    # Format compatible with get_best_fit_with_uncertainty() in parameter_sweep.py
    sweep_info = {
        # Fields required by get_best_fit_with_uncertainty
        "param_name": param_name,
        "parameter_values": list(sweep_values),
        "goodness_of_fit": list(best_chisqr.astype(float)),
        "best_fit": {
            "index": best_idx,
            "value": float(sweep_values[best_idx]),
            "gof": float(best_chisqr[best_idx]),
            "model_name": f"gpu_{param_name}_{sweep_values[best_idx]:.3f}",
        },
        # Additional fields for convenience
        "objective": objective,
        "sweep_data": sweep_data,
    }
    return sweep_info


# ---------------------------------------------------------------------------
# Batch models optimizer (for batch_fit_selected_models_gpu)
# ---------------------------------------------------------------------------

def gpu_fit_models(
    objectives_dict,
    energy_list=None,
    popsize=20,
    n_generations=300,
    seed=0,
    verbose=False,
):
    """
    Fit reflectometry models for all energies simultaneously on GPU.

    Runs n_energies independent DE optimizations in parallel, one per energy.
    Each energy has its own objective with its own Q-grid and data.

    Parameters
    ----------
    objectives_dict : dict {energy: refnx Objective}
        Objectives ready for fitting (parameters configured with vary=True, bounds).
    energy_list     : list of energies to process (default: all keys)
    popsize         : DE population size (default 20)
    n_generations   : number of generations (default 300)
    seed            : random seed
    verbose         : print progress every 50 generations

    Returns
    -------
    dict with keys:
      'fitted_objectives'   : dict {energy: Objective at best-fit params}
      'best_chisqr'         : dict {energy: float}
      'energies'            : list of processed energies
      'elapsed_sec'         : wall time
    """
    if energy_list is None:
        energy_list = sorted(objectives_dict.keys())

    builder = ModelsBatchBuilder(objectives_dict, energy_list)
    n_energies = builder.n_energies
    n_free = builder.n_free
    bounds = builder.bounds

    strategy, params = _make_strategy(n_free, popsize)
    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])

    v_init = jax.jit(jax.vmap(strategy.init, in_axes=(0, 0, 0, None)))
    v_ask = jax.jit(jax.vmap(strategy.ask, in_axes=(0, 0, None)))
    v_tell = jax.jit(jax.vmap(strategy.tell, in_axes=(0, 0, 0, 0, None)))

    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)
    init_keys = jax.random.split(init_key, n_energies)
    init_pops = np.array(
        _uniform_init_population(init_key, n_energies, popsize, bounds)
    )
    init_fitness = builder.fitness(init_pops)

    states = v_init(
        init_keys,
        jnp.array(init_pops, dtype=jnp.float64),
        jnp.array(init_fitness, dtype=jnp.float64),
        params,
    )

    t_start = time.perf_counter()
    for gen in range(n_generations):
        master_key, gen_key = jax.random.split(master_key)
        keys = jax.random.split(gen_key, n_energies)
        populations, states = v_ask(keys, states, params)
        populations = jnp.clip(populations, lb, ub)
        pop_np = np.array(populations)
        fitness = builder.fitness(pop_np)
        states, _ = v_tell(
            keys,
            populations,
            jnp.array(fitness, dtype=jnp.float64),
            states,
            params,
        )
        if verbose and (gen + 1) % 50 == 0:
            best = np.array(states.best_fitness)
            print(
                f"  gen {gen+1:3d}/{n_generations}  "
                f"mean_chisqr={best.mean():.2f}"
            )

    elapsed = time.perf_counter() - t_start
    best_solutions = np.array(jax.vmap(strategy.get_best_solution)(states))
    best_chisqr_arr = np.array(states.best_fitness)

    # Reconstruct fitted objectives at best-fit parameters
    fitted_objectives = {}
    best_chisqr_dict = {}
    for i, energy in enumerate(builder.energies):
        obj_fitted = copy.deepcopy(builder.objectives[i])
        set_free_params(obj_fitted, best_solutions[i])
        fitted_objectives[energy] = obj_fitted
        best_chisqr_dict[energy] = float(best_chisqr_arr[i])

    return {
        "fitted_objectives": fitted_objectives,
        "best_chisqr": best_chisqr_dict,
        "energies": list(builder.energies),
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Sep-CMA-ES batch optimizer with early stopping
# ---------------------------------------------------------------------------

def gpu_fit_models_cmaes(
    objectives_dict,
    energy_list=None,
    popsize=20,
    n_generations=500,
    seed=0,
    verbose=False,
    tol=1e-4,
    patience=5,
    check_every=10,
):
    """
    Fit reflectometry models for all energies simultaneously using Sep-CMA-ES.

    Sep-CMA-ES (Separable CMA-ES) uses a diagonal covariance matrix and
    adapts its search distribution to the curvature of the fitness landscape.
    In benchmarks on 3-layer reflectometry problems it reaches ~2× lower
    chi-squared than DE in the same number of generations.

    Supports early stopping: the optimisation halts when the mean best
    chi-squared across all energies has not improved by more than `tol`
    (relative) over `patience` consecutive checks of `check_every` generations.

    Parameters
    ----------
    objectives_dict : dict {energy: refnx Objective}
        Objectives ready for fitting (vary=True, bounds set).
    energy_list     : list of energies to process (default: all keys)
    popsize         : population size per energy (default 20)
    n_generations   : maximum number of generations (default 500)
    seed            : random seed
    verbose         : print progress every check_every generations
    tol             : relative improvement threshold for early stopping.
                      Stop when (prev_best - cur_best) / prev_best < tol
                      for `patience` consecutive checks. Set to 0 to disable.
    patience        : consecutive non-improving checks before stopping (default 5)
    check_every     : generations between convergence checks (default 10)

    Returns
    -------
    dict with keys:
      'fitted_objectives'   : dict {energy: Objective at best-fit params}
      'best_chisqr'         : dict {energy: float}
      'energies'            : list of processed energies
      'elapsed_sec'         : wall time in seconds
      'generations_run'     : actual number of generations completed
      'converged'           : True if early stopping fired
    """
    from evosax.algorithms import Sep_CMA_ES

    if energy_list is None:
        energy_list = sorted(objectives_dict.keys())

    builder = ModelsBatchBuilder(objectives_dict, energy_list)
    n_energies = builder.n_energies
    n_free = builder.n_free
    bounds = builder.bounds
    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])
    mean_init = (lb + ub) / 2.0

    solution_proto = np.zeros(n_free, dtype=np.float64)
    strategy = Sep_CMA_ES(population_size=popsize, solution=solution_proto)
    params = strategy.default_params

    v_init = jax.jit(jax.vmap(strategy.init, in_axes=(0, None, None)))
    v_ask  = jax.jit(jax.vmap(strategy.ask,  in_axes=(0, 0, None)))
    v_tell = jax.jit(jax.vmap(strategy.tell, in_axes=(0, 0, 0, 0, None)))

    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)
    init_keys = jax.random.split(init_key, n_energies)

    states = v_init(init_keys, mean_init, params)

    prev_mean_best = np.inf
    no_improve_count = 0
    converged = False
    generations_run = 0

    t_start = time.perf_counter()
    for gen in range(n_generations):
        master_key, gen_key = jax.random.split(master_key)
        keys = jax.random.split(gen_key, n_energies)

        populations, states = v_ask(keys, states, params)
        populations = jnp.clip(populations, lb, ub)
        fitness = builder.fitness(np.array(populations))
        states, _ = v_tell(
            keys, populations,
            jnp.array(fitness, dtype=jnp.float64),
            states, params,
        )
        generations_run = gen + 1

        if (gen + 1) % check_every == 0:
            best = np.array(states.best_fitness)
            mean_best = float(best.mean())

            if verbose:
                print(
                    f"  gen {gen+1:4d}/{n_generations}  "
                    f"mean_χ²={mean_best:.2f}  "
                    f"min_χ²={float(best.min()):.2f}  "
                    f"max_χ²={float(best.max()):.2f}"
                )

            if tol > 0 and np.isfinite(prev_mean_best):
                rel_improvement = (prev_mean_best - mean_best) / max(prev_mean_best, 1e-12)
                if rel_improvement < tol:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        if verbose:
                            print(
                                f"  Early stop at gen {gen+1}: "
                                f"improvement {rel_improvement:.2e} < tol {tol:.0e} "
                                f"for {patience} consecutive checks."
                            )
                        converged = True
                        break
                else:
                    no_improve_count = 0

            prev_mean_best = mean_best

    elapsed = time.perf_counter() - t_start

    best_solutions = np.array(states.best_solution)
    best_chisqr_arr = np.array(states.best_fitness)

    fitted_objectives = {}
    best_chisqr_dict = {}
    for i, energy in enumerate(builder.energies):
        obj_fitted = copy.deepcopy(builder.objectives[i])
        set_free_params(obj_fitted, best_solutions[i])
        fitted_objectives[energy] = obj_fitted
        best_chisqr_dict[energy] = float(best_chisqr_arr[i])

    return {
        "fitted_objectives": fitted_objectives,
        "best_chisqr":       best_chisqr_dict,
        "energies":          list(builder.energies),
        "elapsed_sec":       elapsed,
        "generations_run":   generations_run,
        "converged":         converged,
    }
