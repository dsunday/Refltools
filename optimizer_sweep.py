"""
Hyperparameter and algorithm sweep for GPU reflectometry optimizers.

Two sweeps:
  1. DE hyperparameters: F × CR grid, then popsize, using 3-layer model
  2. Algorithm comparison: DE, PSO, CMA-ES, Sep_CMA-ES, SNES, xNES, LM_MA-ES

Tracks convergence curve (best chi2 every CHECK_EVERY generations) so you
can compare speed-to-solution, not just final quality.

Run:
    python optimizer_sweep.py
"""

import os
import sys
import copy
import time
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, "/homes/dfs1/Refltools")

from refnx.reflect import SLD, ReflectModel
from refnx.dataset import ReflectDataset
from refnx.analysis import Objective, Transform

from evosax.algorithms import (
    DifferentialEvolution, PSO,
    CMA_ES, Sep_CMA_ES, SNES, xNES, LM_MA_ES,
)

from gpu_reflect import ModelsBatchBuilder
from gpu_sweep_optimizer import _uniform_init_population


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LAYERS      = 3        # benchmark problem complexity
N_GENERATIONS = 300
CHECK_EVERY   = 10       # record best chi2 every N generations
N_SEEDS       = 3
POPSIZE_REF   = 20       # reference population size for most sweeps

# Algorithms with population-based init (DE, PSO)
POP_BASED = {"DifferentialEvolution", "PSO"}


# ---------------------------------------------------------------------------
# Synthetic objective (same as benchmark_gpu.py)
# ---------------------------------------------------------------------------

def make_synthetic_objective(n_layers=3, seed=0, sld_substrate=2.07,
                              q_min=0.005, q_max=0.3, n_q=300, dq=1.6):
    rng = np.random.default_rng(seed)
    air = SLD(0.0, name="air")
    substrate = SLD(sld_substrate, name="substrate")
    inner_slabs = []
    for i in range(n_layers):
        sld_val   = rng.uniform(1.0, 5.0)
        thickness = rng.uniform(20.0, 80.0)
        roughness = rng.uniform(2.0, 6.0)
        layer = SLD(sld_val, name=f"layer{i+1}")
        inner_slabs.append(layer(thickness, roughness))
    structure = air(0, 0)
    for slab in inner_slabs:
        structure = structure | slab
    structure = structure | substrate(0, 3.0)
    model = ReflectModel(structure, bkg=1e-6, scale=1.0, dq=dq)
    q = np.linspace(q_min, q_max, n_q)
    r_true  = model(q)
    r_noisy = r_true * (1.0 + rng.normal(0.0, 0.02, n_q))
    r_noisy = np.maximum(r_noisy, 1e-10)
    r_err   = r_true * 0.02
    dataset = ReflectDataset([q, r_noisy, r_err])
    for slab in structure.components[1:-1]:
        slab.thick.setp(vary=True, bounds=(5.0, 150.0))
        slab.sld.real.setp(vary=True, bounds=(-1.0, 8.0))
        slab.rough.setp(vary=True, bounds=(1.0, 15.0))
    return Objective(model, dataset, transform=Transform("logY"))


# ---------------------------------------------------------------------------
# Generic optimizer runner
# ---------------------------------------------------------------------------

def run_optimizer(objective, algo_class, popsize, n_generations,
                  algo_kwargs=None, seed=0):
    """
    Run a single optimizer and return convergence curve.

    Parameters
    ----------
    objective   : refnx Objective
    algo_class  : evosax algorithm class
    popsize     : population size
    n_generations: number of generations
    algo_kwargs : dict of extra keyword args passed to algo_class.__init__
    seed        : random seed

    Returns
    -------
    curve   : ndarray (n_checkpoints,) — best chi2 at each checkpoint
    elapsed : float — wall time in seconds (after warmup)
    """
    if algo_kwargs is None:
        algo_kwargs = {}

    builder = ModelsBatchBuilder({"e0": objective}, ["e0"])
    bounds  = builder.bounds          # (n_free, 2)
    n_free  = builder.n_free
    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])
    mean_init = (lb + ub) / 2.0

    algo_name = algo_class.__name__

    solution_proto = np.zeros(n_free, dtype=np.float64)
    strategy = algo_class(population_size=popsize, solution=solution_proto,
                          **algo_kwargs)
    params = strategy.default_params

    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)

    # Initialise state
    if algo_name in POP_BASED:
        init_pop = np.array(
            _uniform_init_population(init_key, 1, popsize, bounds)
        )[0]
        init_fit = builder.fitness(init_pop[None])[0]
        state = jax.jit(strategy.init)(
            init_key,
            jnp.array(init_pop, dtype=jnp.float64),
            jnp.array(init_fit, dtype=jnp.float64),
            params,
        )
    else:
        state = jax.jit(strategy.init)(init_key, mean_init, params)

    ask_jit  = jax.jit(strategy.ask)
    tell_jit = jax.jit(strategy.tell)

    def _step(key, state):
        pop, new_state = ask_jit(key, state, params)
        pop_clipped = jnp.clip(pop, lb, ub)
        fit = builder.fitness(np.array(pop_clipped)[None])[0]
        new_state, _ = tell_jit(
            key, pop_clipped, jnp.array(fit, dtype=jnp.float64), new_state, params
        )
        return new_state

    # Warmup generation (triggers JIT compilation)
    master_key, wk = jax.random.split(master_key)
    state = _step(wk, state)

    curve   = []
    n_check = n_generations // CHECK_EVERY
    t0 = time.perf_counter()

    for block in range(n_check):
        for _ in range(CHECK_EVERY):
            master_key, gk = jax.random.split(master_key)
            state = _step(gk, state)
        best = float(np.array(state.best_fitness))
        curve.append(best)

    elapsed = time.perf_counter() - t0
    return np.array(curve), elapsed


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_seeds(objective, algo_class, popsize, n_generations,
              algo_kwargs=None, seeds=None):
    """Run multiple seeds; return (mean_curve, std_curve, mean_time)."""
    if seeds is None:
        seeds = list(range(N_SEEDS))
    curves, times = [], []
    for s in seeds:
        c, t = run_optimizer(
            copy.deepcopy(objective), algo_class, popsize, n_generations,
            algo_kwargs=algo_kwargs, seed=s
        )
        curves.append(c)
        times.append(t)
        print(f".", end="", flush=True)
    curves = np.stack(curves)
    return curves.mean(axis=0), curves.std(axis=0), np.mean(times)


# ---------------------------------------------------------------------------
# Sweep 1: DE — F × CR grid
# ---------------------------------------------------------------------------

def sweep_de_params(objective, popsize=POPSIZE_REF, n_generations=N_GENERATIONS,
                    seeds=None):
    F_vals  = [0.4, 0.6, 0.8, 1.0]
    CR_vals = [0.3, 0.5, 0.7, 0.9]

    print(f"\n{'='*72}")
    print(f"DE Parameter Sweep: F × CR  (popsize={popsize}, gens={n_generations})")
    print(f"{'='*72}")
    header = f"{'F':>5} {'CR':>5} | {'final χ²':>10} {'±':>8} | {'time(s)':>8}"
    print(header)
    print("-" * len(header))

    n_total = len(F_vals) * len(CR_vals)
    results = {}
    combo_i = 0
    for F in F_vals:
        for CR in CR_vals:
            combo_i += 1
            print(f"  [{combo_i:2d}/{n_total}] F={F:.1f} CR={CR:.1f} ...", end="", flush=True)
            mean_c, std_c, t = run_seeds_de(
                objective, popsize, n_generations,
                F=F, CR=CR, seeds=seeds
            )
            final_mean = mean_c[-1]
            final_std  = std_c[-1]
            print(f"\r  {F:>4.1f}  {CR:>4.1f} | {final_mean:>10.2f} {final_std:>8.2f} | {t:>8.1f}")
            results[(F, CR)] = (mean_c, std_c, t)

    best_key = min(results, key=lambda k: results[k][0][-1])
    print(f"\n  Best: F={best_key[0]}, CR={best_key[1]}"
          f"  (final χ²={results[best_key][0][-1]:.2f})")
    return results, best_key


def run_seeds_de(objective, popsize, n_generations, F, CR, seeds=None):
    """Specialised multi-seed runner for DE where F/CR are passed via params."""
    if seeds is None:
        seeds = list(range(N_SEEDS))

    builder = ModelsBatchBuilder({"e0": objective}, ["e0"])
    bounds  = builder.bounds
    n_free  = builder.n_free
    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])
    solution_proto = np.zeros(n_free, dtype=np.float64)
    strategy = DifferentialEvolution(population_size=popsize, solution=solution_proto)
    params = strategy.default_params.replace(
        differential_weight=F, crossover_rate=CR
    )

    ask_jit  = jax.jit(strategy.ask)
    tell_jit = jax.jit(strategy.tell)

    curves, times = [], []
    for s_i, s in enumerate(seeds):
        print(f".", end="", flush=True)
        master_key = jax.random.PRNGKey(s)
        master_key, init_key = jax.random.split(master_key)
        init_pop = np.array(_uniform_init_population(init_key, 1, popsize, bounds))[0]
        init_fit = builder.fitness(init_pop[None])[0]
        state = jax.jit(strategy.init)(
            init_key,
            jnp.array(init_pop, dtype=jnp.float64),
            jnp.array(init_fit, dtype=jnp.float64),
            params,
        )
        # Warmup
        master_key, wk = jax.random.split(master_key)
        pop, state = ask_jit(wk, state, params)
        pop = jnp.clip(pop, lb, ub)
        fit = builder.fitness(np.array(pop)[None])[0]
        state, _ = tell_jit(wk, pop, jnp.array(fit, dtype=jnp.float64), state, params)

        curve = []
        n_check = n_generations // CHECK_EVERY
        t0 = time.perf_counter()
        for _ in range(n_check):
            for __ in range(CHECK_EVERY):
                master_key, gk = jax.random.split(master_key)
                pop, state = ask_jit(gk, state, params)
                pop = jnp.clip(pop, lb, ub)
                fit = builder.fitness(np.array(pop)[None])[0]
                state, _ = tell_jit(gk, pop, jnp.array(fit, dtype=jnp.float64), state, params)
            curve.append(float(np.array(state.best_fitness)))
        times.append(time.perf_counter() - t0)
        curves.append(curve)

    curves = np.array(curves)
    return curves.mean(axis=0), curves.std(axis=0), np.mean(times)


# ---------------------------------------------------------------------------
# Sweep 2: popsize for DE (best F/CR)
# ---------------------------------------------------------------------------

def sweep_de_popsize(objective, F, CR, n_generations=N_GENERATIONS, seeds=None):
    popsizes = [10, 15, 20, 30, 50]

    print(f"\n{'='*72}")
    print(f"DE Popsize Sweep  (F={F}, CR={CR}, gens={n_generations})")
    print(f"{'='*72}")
    header = f"{'popsize':>8} | {'final χ²':>10} {'±':>8} | {'time(s)':>8}"
    print(header)
    print("-" * len(header))

    results = {}
    for i, ps in enumerate(popsizes):
        print(f"  [{i+1}/{len(popsizes)}] popsize={ps} ...", end="", flush=True)
        mean_c, std_c, t = run_seeds_de(
            objective, ps, n_generations, F=F, CR=CR, seeds=seeds
        )
        print(f"\r  {ps:>7} | {mean_c[-1]:>10.2f} {std_c[-1]:>8.2f} | {t:>8.1f}")
        results[ps] = (mean_c, std_c, t)

    best_ps = min(results, key=lambda k: results[k][0][-1])
    print(f"\n  Best popsize: {best_ps}  (final χ²={results[best_ps][0][-1]:.2f})")
    return results, best_ps


# ---------------------------------------------------------------------------
# Sweep 3: Algorithm comparison
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("DE",         DifferentialEvolution, {}),
    ("PSO",        PSO,                  {}),
    ("CMA-ES",     CMA_ES,               {}),
    ("Sep-CMA-ES", Sep_CMA_ES,           {}),
    ("SNES",       SNES,                 {}),
    ("xNES",       xNES,                 {}),
    ("LM-MA-ES",   LM_MA_ES,             {}),
]


def sweep_algorithms(objective, popsize=POPSIZE_REF, n_generations=N_GENERATIONS,
                     seeds=None):
    print(f"\n{'='*72}")
    print(f"Algorithm Comparison  (popsize={popsize}, gens={n_generations})")
    print(f"{'='*72}")
    header = (f"{'Algorithm':>12} | {'final χ²':>10} {'±':>8} | "
              f"{'gen50 χ²':>10} {'gen100 χ²':>10} | {'time(s)':>8}")
    print(header)
    print("-" * len(header))

    results = {}
    for i, (name, cls, kwargs) in enumerate(ALGORITHMS):
        print(f"  [{i+1}/{len(ALGORITHMS)}] {name} ...", end="", flush=True)
        try:
            mean_c, std_c, t = run_seeds(
                copy.deepcopy(objective), cls, popsize, n_generations,
                algo_kwargs=kwargs, seeds=seeds
            )
            idx50  = min(50  // CHECK_EVERY - 1, len(mean_c) - 1)
            idx100 = min(100 // CHECK_EVERY - 1, len(mean_c) - 1)
            print(f"\r  {name:>11} | {mean_c[-1]:>10.2f} {std_c[-1]:>8.2f} | "
                  f"{mean_c[idx50]:>10.2f} {mean_c[idx100]:>10.2f} | {t:>8.1f}")
            results[name] = (mean_c, std_c, t)
        except Exception as e:
            print(f"\r  {name:>11} | FAILED: {e}")
            results[name] = None

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print(f"Optimizer Sweep — {N_LAYERS}-layer model, {N_SEEDS} seeds, "
          f"{N_GENERATIONS} generations")
    print(f"  JAX devices: {jax.devices()}")
    print("=" * 72)

    obj = make_synthetic_objective(N_LAYERS, seed=42)

    # --- Sweep 1: DE F × CR ---
    de_grid_results, best_F_CR = sweep_de_params(obj)
    best_F, best_CR = best_F_CR

    # --- Sweep 2: DE popsize ---
    de_ps_results, best_ps = sweep_de_popsize(obj, F=best_F, CR=best_CR)

    # --- Sweep 3: Algorithm comparison ---
    algo_results = sweep_algorithms(obj, popsize=POPSIZE_REF)

    # --- Convergence summary ---
    print(f"\n{'='*72}")
    print("Convergence at key generation counts (mean over seeds):")
    print(f"{'='*72}")
    header2 = f"{'Algorithm':>12} | " + " ".join(
        f"gen{g:>3}" for g in range(CHECK_EVERY, N_GENERATIONS + 1, 50)
    )
    print(header2)
    print("-" * len(header2))
    checkpoints = list(range(CHECK_EVERY, N_GENERATIONS + 1, 50))
    indices = [g // CHECK_EVERY - 1 for g in checkpoints]
    for name, cls, _ in ALGORITHMS:
        if algo_results.get(name) is not None:
            mean_c = algo_results[name][0]
            vals = " ".join(
                f"{mean_c[min(i, len(mean_c)-1)]:>7.1f}" for i in indices
            )
            print(f"  {name:>11} | {vals}")

    print("\nDone.")
