"""
GPU timing benchmarks for reflectometry fitting.

Test 1: Algorithm comparison (1-5 inner layers)
  - CPU: scipy differential_evolution (8 workers)
  - GPU: evosax DifferentialEvolution (JAX/CUDA)
  - GPU: evosax CMA_ES (JAX/CUDA)

Test 2: Concurrency comparison (10 energies, 1 layer each)
  - CPU: sequential scipy DE for each energy
  - GPU: all energies simultaneously via gpu_fit_models

Run: python benchmark_gpu.py
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
from refnx.analysis import Objective, Transform, CurveFitter

from evosax.algorithms import DifferentialEvolution, CMA_ES

from gpu_reflect import ModelsBatchBuilder
from gpu_sweep_optimizer import _make_strategy, _uniform_init_population, gpu_fit_models


# ---------------------------------------------------------------------------
# Synthetic model builder
# ---------------------------------------------------------------------------

def make_synthetic_objective(n_layers, seed=0, sld_substrate=2.07,
                              q_min=0.005, q_max=0.3, n_q=300, dq=1.6):
    """
    Build a synthetic n-layer reflectometry model with noisy forward-model data.

    Inner layer parameters (thickness, real SLD, roughness) are set free with
    physically reasonable bounds so optimizers have a non-trivial task.
    """
    rng = np.random.default_rng(seed)

    air = SLD(0.0, name="air")
    substrate = SLD(sld_substrate, name="substrate")

    inner_slabs = []
    for i in range(n_layers):
        sld_val = rng.uniform(1.0, 5.0)
        thickness = rng.uniform(20.0, 80.0)
        roughness = rng.uniform(2.0, 6.0)
        layer = SLD(sld_val, name=f"layer{i + 1}")
        inner_slabs.append(layer(thickness, roughness))

    structure = air(0, 0)
    for slab in inner_slabs:
        structure = structure | slab
    structure = structure | substrate(0, 3.0)

    model = ReflectModel(structure, bkg=1e-6, scale=1.0, dq=dq)

    q = np.linspace(q_min, q_max, n_q)
    r_true = model(q)
    r_noisy = r_true * (1.0 + rng.normal(0.0, 0.02, n_q))
    r_noisy = np.maximum(r_noisy, 1e-10)
    r_err = r_true * 0.02
    dataset = ReflectDataset([q, r_noisy, r_err])

    for slab in structure.components[1:-1]:
        slab.thick.setp(vary=True, bounds=(5.0, 150.0))
        slab.sld.real.setp(vary=True, bounds=(-1.0, 8.0))
        slab.rough.setp(vary=True, bounds=(1.0, 15.0))

    return Objective(model, dataset, transform=Transform("logY"))


# ---------------------------------------------------------------------------
# CPU baseline
# ---------------------------------------------------------------------------

def time_cpu_de(objective, maxiter=300, popsize=20, workers=8, seed=42):
    fitter = CurveFitter(objective)
    t0 = time.perf_counter()
    fitter.fit(
        "differential_evolution",
        maxiter=maxiter,
        popsize=popsize,
        tol=0,
        workers=workers,
        seed=seed,
    )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# GPU: evosax DE (single objective)
# ---------------------------------------------------------------------------

def time_gpu_de(objective, popsize=20, n_generations=300, seed=0):
    builder = ModelsBatchBuilder({"e0": objective}, ["e0"])
    bounds = builder.bounds
    n_free = builder.n_free
    strategy, params = _make_strategy(n_free, popsize)
    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])

    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)
    init_pop = np.array(_uniform_init_population(init_key, 1, popsize, bounds))[0]
    init_fit = builder.fitness(init_pop[None])[0]

    ask_jit = jax.jit(strategy.ask)
    tell_jit = jax.jit(strategy.tell)

    state = jax.jit(strategy.init)(
        init_key,
        jnp.array(init_pop, dtype=jnp.float64),
        jnp.array(init_fit, dtype=jnp.float64),
        params,
    )

    # Warmup: one generation to trigger JIT compilation before timing
    master_key, gen_key = jax.random.split(master_key)
    pop, state = ask_jit(gen_key, state, params)
    pop = jnp.clip(pop, lb, ub)
    fit = builder.fitness(np.array(pop)[None])[0]
    state, _ = tell_jit(gen_key, pop, jnp.array(fit, dtype=jnp.float64), state, params)

    t0 = time.perf_counter()
    for _ in range(n_generations):
        master_key, gen_key = jax.random.split(master_key)
        pop, state = ask_jit(gen_key, state, params)
        pop = jnp.clip(pop, lb, ub)
        fit = builder.fitness(np.array(pop)[None])[0]
        state, _ = tell_jit(gen_key, pop, jnp.array(fit, dtype=jnp.float64), state, params)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# GPU: evosax CMA-ES (single objective)
# ---------------------------------------------------------------------------

def time_gpu_cmaes(objective, popsize=20, n_generations=300, seed=0):
    builder = ModelsBatchBuilder({"e0": objective}, ["e0"])
    bounds = builder.bounds
    n_free = builder.n_free
    lb = jnp.array(bounds[:, 0])
    ub = jnp.array(bounds[:, 1])
    mean_init = (lb + ub) / 2.0

    cma = CMA_ES(population_size=popsize, solution=np.zeros(n_free, dtype=np.float64))
    p = cma.default_params

    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)
    state = jax.jit(cma.init)(init_key, mean_init, p)

    ask_jit = jax.jit(cma.ask)
    tell_jit = jax.jit(cma.tell)

    # Warmup
    master_key, gen_key = jax.random.split(master_key)
    pop, state = ask_jit(gen_key, state, p)
    pop_clipped = jnp.clip(pop, lb, ub)
    fit = builder.fitness(np.array(pop_clipped)[None])[0]
    state, _ = tell_jit(
        gen_key, pop_clipped, jnp.array(fit, dtype=jnp.float64), state, p
    )

    t0 = time.perf_counter()
    for _ in range(n_generations):
        master_key, gen_key = jax.random.split(master_key)
        pop, state = ask_jit(gen_key, state, p)
        pop_clipped = jnp.clip(pop, lb, ub)
        fit = builder.fitness(np.array(pop_clipped)[None])[0]
        state, _ = tell_jit(
            gen_key, pop_clipped, jnp.array(fit, dtype=jnp.float64), state, p
        )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Test 1: Algorithm comparison across 1–5 inner layers
# ---------------------------------------------------------------------------

def run_test1(n_layers_list=(1, 2, 3, 4, 5), popsize=20, n_generations=300,
              cpu_maxiter=300, cpu_workers=8):
    header = (
        f"{'Layers':>6} | {'n_free':>6} | {'CPU DE (s)':>10} | "
        f"{'GPU DE (s)':>10} | {'GPU CMA (s)':>11} | "
        f"{'DE ×':>7} | {'CMA ×':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in n_layers_list:
        n_free = n * 3
        obj = make_synthetic_objective(n, seed=n)

        t_cpu = time_cpu_de(
            copy.deepcopy(obj),
            maxiter=cpu_maxiter,
            popsize=popsize,
            workers=cpu_workers,
        )
        t_de = time_gpu_de(copy.deepcopy(obj), popsize=popsize, n_generations=n_generations)
        t_cma = time_gpu_cmaes(copy.deepcopy(obj), popsize=popsize, n_generations=n_generations)

        speedup_de  = t_cpu / t_de  if t_de  > 0 else float("nan")
        speedup_cma = t_cpu / t_cma if t_cma > 0 else float("nan")

        print(
            f"{n:>6} | {n_free:>6} | {t_cpu:>10.2f} | {t_de:>10.2f} | "
            f"{t_cma:>11.2f} | {speedup_de:>6.1f}x | {speedup_cma:>6.1f}x"
        )
        rows.append((n, n_free, t_cpu, t_de, t_cma, speedup_de, speedup_cma))

    print(
        f"\n  CPU: scipy DE, {cpu_workers} workers, maxiter={cpu_maxiter}, popsize={popsize}"
    )
    print(f"  GPU: evosax, {n_generations} generations, popsize={popsize}")
    return rows


# ---------------------------------------------------------------------------
# Test 2: Concurrency — 10 energies, 1 inner layer each
# ---------------------------------------------------------------------------

def time_cpu_sequential(objectives_dict, maxiter=300, popsize=20, workers=8):
    t0 = time.perf_counter()
    for obj in objectives_dict.values():
        fitter = CurveFitter(obj)
        fitter.fit(
            "differential_evolution",
            maxiter=maxiter,
            popsize=popsize,
            tol=0,
            workers=workers,
            seed=42,
        )
    return time.perf_counter() - t0


def time_gpu_concurrent(objectives_dict, popsize=20, n_generations=300, seed=0):
    t0 = time.perf_counter()
    gpu_fit_models(
        objectives_dict,
        popsize=popsize,
        n_generations=n_generations,
        seed=seed,
    )
    return time.perf_counter() - t0


def run_test2(n_energies=10, popsize=20, n_generations=300,
              cpu_maxiter=300, cpu_workers=8):
    sld_vals = np.linspace(1.5, 3.5, n_energies)
    print(
        f"  Building {n_energies} single-layer objectives"
        f" (substrate SLD {sld_vals[0]:.2f}–{sld_vals[-1]:.2f} ×10⁻⁶ Å⁻²)"
    )
    objectives = {
        f"E{i}": make_synthetic_objective(1, seed=i, sld_substrate=float(sv))
        for i, sv in enumerate(sld_vals)
    }

    t_cpu = time_cpu_sequential(
        copy.deepcopy(objectives),
        maxiter=cpu_maxiter,
        popsize=popsize,
        workers=cpu_workers,
    )
    t_gpu = time_gpu_concurrent(
        copy.deepcopy(objectives),
        popsize=popsize,
        n_generations=n_generations,
    )

    speedup = t_cpu / t_gpu if t_gpu > 0 else float("nan")
    print(f"\n  CPU sequential ({n_energies} energies): {t_cpu:.2f} s")
    print(f"  GPU concurrent ({n_energies} energies): {t_gpu:.2f} s")
    print(f"  Speedup:                          {speedup:.1f}x")
    print(
        f"\n  CPU: scipy DE, {cpu_workers} workers, maxiter={cpu_maxiter}, popsize={popsize}"
    )
    print(f"  GPU: evosax DE, {n_generations} generations, popsize={popsize}")
    return t_cpu, t_gpu, speedup


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("GPU Timing Benchmarks — refnx reflectometry fitting")
    print(f"  JAX devices: {jax.devices()}")
    print("=" * 72)

    print("\n--- Test 1: Algorithm Comparison (1–5 inner layers) ---\n")
    run_test1()

    print("\n--- Test 2: Concurrency Comparison (10 energies, 1 layer) ---\n")
    run_test2()

    print("\nDone.")
