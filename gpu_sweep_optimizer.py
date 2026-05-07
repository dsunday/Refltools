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

import copy
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap

jax.config.update("jax_enable_x64", True)

from evosax.algorithms import DifferentialEvolution
from gpu_reflect import (
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

def _make_strategy(n_free, popsize, scale_factor=0.8, crossover_prob=0.9):
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
    from gpu_reflect import ModelsBatchBuilder

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
