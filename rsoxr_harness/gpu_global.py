"""
gpu_global.py
-------------
Fast χ² for energy-linked global fits.

gpu_reflect.GlobalModelsBatchBuilder assembles every energy's slab arrays in
NumPy and calls the GPU once per energy — ~45 s per 500-member generation for
28 energies × 340 parameters.  Here the global parameter vector is gathered
onto the per-energy layout (n_energies, popsize, n_free) with one index array
and handed to ModelsBatchBuilder's fused, jitted kernel (the path the
per-energy CMA-ES uses), then summed over energies.
"""

import numpy as np


class FastGlobalFitness:
    """
    g: dict from FitProject.build_global.  All energies must share one free-
    parameter layout (true for recipe-built objectives: same layers, same
    vary flags).  fitness(pop (P, n_global)) → total χ² (P,).
    """

    def __init__(self, g, normalize_by_n_q=False, chunk=250, f32=True):
        import jax
        import jax.numpy as jnp
        from gpu_reflect import ModelsBatchBuilder, extract_free_params, get_bounds_array
        objs, elist = g["objectives"], g["energy_list"]
        gparams = list(g["global_objective"].varying_parameters())
        gid = {id(p): i for i, p in enumerate(gparams)}
        local = [extract_free_params(o) for o in objs]
        n = {len(l) for l in local}
        if len(n) != 1:
            raise ValueError(f"energies have different numbers of free parameters {n}")
        names0 = [p.name for p in local[0]]
        for e, l in zip(elist, local):
            if [p.name for p in l] != names0:
                raise ValueError(f"{e:g} eV free-parameter layout differs from {elist[0]:g} eV")
        self.idx = np.array([[gid[id(p)] for p in l] for l in local], dtype=np.int32)
        self.n_global = len(gparams)
        self.param_names = [p.name for p in gparams]
        self.bounds = get_bounds_array(gparams)
        # float32 Abeles, as the per-energy CMA-ES uses (fp64 is a small
        # fraction of fp32 throughput on these GPUs); reported χ² always
        # comes from refnx in float64
        self.mb = ModelsBatchBuilder(dict(zip(elist, objs)), elist,
                                     use_f32_abeles=f32,
                                     normalize_by_n_q=normalize_by_n_q)
        idx_j = jnp.asarray(self.idx)
        mb = self.mb

        # gather in its own jit; χ² kernels as the per-energy CMA-ES calls them
        @jax.jit
        def _gather(pop):                    # (P, G) → (E, P, n_free)
            return jnp.transpose(pop[:, idx_j], (1, 0, 2))

        def _fit(pop):
            # population chunks bound the smeared (E, chunk, n_quad_q) complex
            # intermediates (fp64 at 500 × 28 energies ran out of GPU memory)
            parts = [jnp.sum(mb.fitness_jax(_gather(pop[i:i + chunk])), axis=0)
                     for i in range(0, pop.shape[0], chunk)]
            return jnp.concatenate(parts)

        self._fit = _fit

    def fitness_jax(self, pop):
        return self._fit(pop)

    def fitness(self, pop):
        import jax.numpy as jnp
        return np.asarray(self._fit(jnp.asarray(pop, dtype=jnp.float64)))
