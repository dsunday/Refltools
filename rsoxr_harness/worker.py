"""
worker.py
---------
Runs one job inside the background process launched by jobs.launch.
"""

import copy
import json
import os
import time
import traceback

from . import jobs as J
from .project import FitProject


def run_job(project_path, job_id):
    project = FitProject.open(project_path)
    job = J.update_job(project, job_id, status="running", pid=os.getpid(),
                       started=J._now())
    print(f"[harness] job {job_id} ({job['kind']}) started {job['started']} "
          f"pid={os.getpid()} gpu={job['gpu']}", flush=True)
    try:
        if job["kind"] == "fit":
            outputs = _run_fit(project, job)
        elif job["kind"] in ("nuts", "jaxns"):
            outputs = _run_sampling(project, job)
        elif job["kind"] == "stoich":
            outputs = _run_stoich(project, job)
        else:
            raise ValueError(f"unknown job kind '{job['kind']}'")
    except BaseException as exc:          # includes KeyboardInterrupt / SIGTERM
        tb = traceback.format_exc()
        print(tb, flush=True)
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        current = J.read_job(project, job_id)
        if current["status"] != "cancelled":
            J.update_job(project, job_id, status=status, finished=J._now(), error=tb)
        return 1
    J.update_job(project, job_id, status="done", finished=J._now(), outputs=outputs)
    print(f"[harness] job {job_id} done", flush=True)
    return 0


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def _run_fit(project, job):
    from batch import batch_fit_selected_models_cmaes
    from h5io import save_batch_to_h5

    p = job["params"]
    energies = job["energies"]
    outputs = {}
    for model in job["models"]:
        recipe = project.get_model(model)
        if recipe.is_linked:
            outputs[model] = _run_global_fit(project, job, recipe)
            J.update_job(project, job["id"], outputs=outputs)
            continue
        print(f"\n[harness] {model}: building objectives at {len(energies)} "
              f"energies", flush=True)
        built = project.build_objectives(recipe, energies)
        objs = built["objectives"]
        originals = {e: copy.deepcopy(o) for e, o in objs.items()}
        chi0 = {e: float(o.chisqr()) for e, o in objs.items()}

        t0 = time.time()
        res = batch_fit_selected_models_cmaes(
            objectives_dict=objs, energy_list=energies,
            popsize=p["popsize"], n_generations=p["n_generations"],
            seed=p.get("seed", 0), tol=p["tol"], patience=p["patience"],
            n_restarts=p.get("n_restarts", 1), normalize=p.get("normalize", False),
            sample_name=project.sample_name, model_name=model,
            h5_filepath=None, verbose=True)
        elapsed = time.time() - t0

        # fill what the CMA-ES path leaves out, so h5 has initial values too
        res["original_objectives"] = originals
        res["original_structures"] = {e: o.model.structure for e, o in originals.items()}
        for e, d in res["individual_results"].items():
            d["initial_chi_squared"] = chi0[e]

        with J.h5_lock(project.h5_path):
            save_batch_to_h5(res, project.sample_name, model, project.h5_path,
                             energy_list=energies)
            runs = _tag_runs(project, model, energies, recipe, job)

        chi = {e: float(res["individual_results"][e]["chisqr"]) for e in energies}
        npts = {e: int(objs[e].npoints) for e in energies}
        outputs[model] = {
            "runs": runs,
            "chi2": {str(e): chi[e] for e in energies},
            "chi2_initial": {str(e): chi0[e] for e in energies},
            "chi2_reduced_mean": sum(chi[e] / npts[e] for e in energies) / len(energies),
            "generations_run": int(res["generations_run"]),
            "converged": bool(res["converged"]),
            "elapsed_sec": round(elapsed, 1),
        }
        J.update_job(project, job["id"], outputs=outputs)
        print(f"[harness] {model}: saved runs {sorted(set(runs.values()))} "
              f"in {elapsed:.0f}s; mean chi2/N = "
              f"{outputs[model]['chi2_reduced_mean']:.4g}", flush=True)
    return outputs


def global_cmaes(builder, x0, popsize, n_generations, seed=0, tol=1e-4,
                 patience=5, check_every=10, n_restarts=1, verbose=True):
    """
    Sep-CMA-ES over a global fitness's parameter vector (FastGlobalFitness;
    same loop as the LAMS7 global notebooks).  Restart 0 starts at x0; later restarts
    start uniformly inside the bounds.  Returns (best_x, best_chi2, info).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from evosax.algorithms import Sep_CMA_ES
    lb, ub = builder.bounds[:, 0], builder.bounds[:, 1]
    lb_j, ub_j = jnp.array(lb), jnp.array(ub)
    rng = np.random.default_rng(seed)
    best = (None, np.inf)
    history = []
    gens_total = 0
    for k in range(n_restarts):
        start = np.clip(x0, lb, ub) if k == 0 else rng.uniform(lb, ub)
        strategy = Sep_CMA_ES(population_size=popsize,
                              solution=np.zeros(builder.n_global, dtype=np.float64))
        params = strategy.default_params
        key = jax.random.PRNGKey(seed + k)
        key, ik = jax.random.split(key)
        state = strategy.init(ik, jnp.array(start), params)
        prev, bad, converged, gen = np.inf, 0, False, 0
        for gen in range(n_generations):
            key, gk = jax.random.split(key)
            pop, state = strategy.ask(gk, state, params)
            pop = jnp.clip(pop, lb_j, ub_j)
            chi = jnp.array(builder.fitness(np.array(pop, dtype=np.float64)),
                            dtype=jnp.float64)
            state, _ = strategy.tell(gk, pop, chi, state, params)
            if (gen + 1) % check_every == 0:
                b = float(state.best_fitness)
                if verbose and (gen + 1) % (check_every * 10) == 0:
                    print(f"  restart {k} gen {gen + 1:5d}  best χ²={b:.6g}", flush=True)
                if np.isfinite(prev) and (prev - b) / max(prev, 1e-12) < tol:
                    bad += 1
                    if bad >= patience:
                        converged = True
                        break
                else:
                    bad = 0
                prev = b
        x = np.clip(np.array(state.best_solution), lb, ub)
        c = float(builder.fitness(x[None, :])[0])
        history.append(c)
        gens_total += gen + 1
        if verbose:
            print(f"  restart {k}: χ²={c:.6g} after {gen + 1} generations"
                  f"{' (converged)' if converged else ''}", flush=True)
        if c < best[1]:
            best = (x, c)
    return best[0], best[1], {"restart_chi2": history, "generations_run": gens_total,
                              "converged": converged}


def _run_global_fit(project, job, recipe):
    """One energy-linked CMA-ES fit; saved per energy under one run index."""
    import h5py
    import numpy as np
    from gpu_reflect import set_global_params
    from .gpu_global import FastGlobalFitness
    from h5io import save_batch_to_h5

    p = job["params"]
    model = recipe.name
    g = project.build_global(recipe)
    elist, objs, gobj = g["energy_list"], g["objectives"], g["global_objective"]
    ck = g["checks"]
    print(f"\n[harness] {model}: GLOBAL fit over {ck['n_energies']} energies; "
          f"{ck['n_global']} free parameters ({ck['n_shared']} shared: "
          + ", ".join(f"{l} {pn}" for l, pn in g["shared_params"]) + ")", flush=True)
    print(f"[harness] checks OK: parameter count {ck['n_global']} = "
          f"{ck['n_independent']} - ({ck['n_energies']}-1)x{ck['n_shared']}; "
          f"shared identity; χ² additivity", flush=True)
    originals = {e: copy.deepcopy(o) for e, o in zip(elist, objs)}
    chi0 = {e: float(o.chisqr()) for e, o in zip(elist, objs)}
    x0 = np.array([q.value for q in gobj.varying_parameters()])
    builder = FastGlobalFitness(g, normalize_by_n_q=p.get("normalize", False))
    c_start = float(builder.fitness(x0[None, :])[0])

    t0 = time.time()
    x, c, info = global_cmaes(builder, x0, p["popsize"], p["n_generations"],
                              seed=p.get("seed", 0), tol=p["tol"],
                              patience=p["patience"], n_restarts=p.get("n_restarts", 1))
    kept_start = False
    if c > c_start:                         # never return worse than the start
        x, c, kept_start = x0, c_start, True
        print("[harness] CMA-ES ended above its start point; keeping the start", flush=True)
    elapsed = time.time() - t0
    set_global_params(gobj, x)
    chi = {e: float(o.chisqr()) for e, o in zip(elist, objs)}
    total = float(sum(chi.values()))

    res = {"fitted_objectives": dict(zip(elist, objs)),
           "individual_results": {e: {"objective": o, "chisqr": chi[e],
                                      "initial_chi_squared": chi0[e]}
                                  for e, o in zip(elist, objs)},
           "original_objectives": originals,
           "original_structures": {e: o.model.structure for e, o in originals.items()},
           "fitted_energies": list(elist), "non_fitted_energies": [],
           "summary_stats": {"algorithm": "Sep-CMA-ES (global, energy-linked)",
                             "final_chi_squared_total": total,
                             "initial_chi_squared_total": sum(chi0.values())},
           "elapsed_sec": elapsed, "generations_run": info["generations_run"],
           "converged": info["converged"]}
    with J.h5_lock(project.h5_path):
        idx = 0
        if os.path.exists(project.h5_path):
            with h5py.File(project.h5_path, "r") as f:
                for e in elist:
                    path = f"{project.sample_name}/{float(e)}/{model}"
                    if path in f:
                        idx = max([idx] + [int(k.split("_")[1]) + 1 for k in f[path]
                                           if k.startswith("run_")])
        save_batch_to_h5(res, project.sample_name, model, project.h5_path,
                         energy_list=elist, run_index=idx)
        runs = _tag_runs(project, model, elist, recipe, job)
        shared = {f"{l} - {'thick' if pn == 'thickness' else 'rough'}": float(sp.value)
                  for (l, pn), sp in g["shared_params"].items()}
        with h5py.File(project.h5_path, "a") as f:
            for e in elist:
                rg = f[f"{project.sample_name}/{float(e)}/{model}/run_{idx}"]
                rg.attrs["global_chi2_total"] = total
                rg.attrs["global_n_params"] = ck["n_global"]
                rg.attrs["global_energies"] = json.dumps(list(elist))
                rg.attrs["global_shared"] = json.dumps(shared)
    assert set(runs.values()) == {idx}, runs
    npts = {e: int(o.npoints) for e, o in zip(elist, objs)}
    n_data = sum(npts.values())
    out = {"runs": runs, "global": True, "run_index": idx,
           "chi2": {str(e): chi[e] for e in elist},
           "chi2_initial": {str(e): chi0[e] for e in elist},
           "chi2_total": total, "chi2_total_initial": sum(chi0.values()),
           "n_global_params": ck["n_global"], "n_data": n_data,
           "chi2_reduced_global": total / (n_data - ck["n_global"]),
           "chi2_reduced_mean": sum(chi[e] / npts[e] for e in elist) / len(elist),
           "shared": shared, "restart_chi2": info["restart_chi2"],
           "kept_start": kept_start,
           "generations_run": info["generations_run"], "converged": info["converged"],
           "elapsed_sec": round(elapsed, 1)}
    print(f"[harness] {model}: global run_{idx} saved in {elapsed:.0f}s; total χ² "
          f"{sum(chi0.values()):.6g} → {total:.6g}; χ²/(N-P) = "
          f"{out['chi2_reduced_global']:.4g}; mean χ²/N = {out['chi2_reduced_mean']:.4g}",
          flush=True)
    for k, v in shared.items():
        print(f"[harness]   shared {k} = {v:.4g}", flush=True)
    return out


def _tag_runs(project, model, energies, recipe, job):
    """Stamp the newest run_N of each energy with recipe hash / job id."""
    import h5py
    runs = {}
    rj = json.dumps(recipe.to_json())
    with h5py.File(project.h5_path, "a") as f:
        for e in energies:
            g = f[f"{project.sample_name}/{float(e)}/{model}"]
            n = max(int(k.split("_")[1]) for k in g if k.startswith("run_"))
            run = g[f"run_{n}"]
            run.attrs["recipe_hash"] = recipe.physics_hash()
            run.attrs["recipe_json"] = rj
            run.attrs["job_id"] = job["id"]
            run.attrs["fit_params"] = json.dumps(job["params"])
            runs[str(e)] = n
    return runs


# ---------------------------------------------------------------------------
# uncertainty: NUTS (BlackJAX) / JAXNS
# ---------------------------------------------------------------------------

def unc_dir_sample(job, project):
    """Cache namespace: project fits → sample name; external h5 → ext-<sample>."""
    p = job["params"]
    return f"ext-{p['sample']}" if p.get("h5") else project.sample_name


def _run_sampling(project, job):
    import numpy as np
    from Plotting_Refl import save_mcmc, save_jaxns
    from gpu_reflect import extract_free_params

    method = job["kind"]
    p = dict(job["params"])
    h5, sample = p.pop("h5", None), p.pop("sample", None)
    force = p.pop("force", False)
    criteria = p.pop("criteria", "best")
    model = job["models"][0]
    ns = unc_dir_sample(job, project)
    energies = job["energies"]

    todo = [e for e in energies
            if force or not os.path.exists(project.unc_path(model, method, e, ns))]
    skipped = [e for e in energies if e not in todo]
    if skipped:
        print(f"[harness] cached, skipping: {', '.join(f'{e:g}' for e in skipped)} eV "
              f"(use --force to rerun)", flush=True)
    outputs = {"skipped_cached": skipped, "energies": {}}
    if not todo:
        return outputs

    best = project.load_best(model, todo, criteria=criteria, h5=h5, sample=sample)
    recipe_hash = (project.get_model(model).physics_hash()
                   if best["source"] == "recipe" else None)
    for e in todo:
        obj = best["objectives"][e]
        chi_best = float(obj.chisqr())
        names = [q.name for q in extract_free_params(obj)]
        print(f"\n[harness] {method} {model} {e:g} eV: {len(names)} free params, "
              f"best chi2={chi_best:.6g} ({best['runs'][e]}, {best['source']})",
              flush=True)
        t0 = time.time()
        if method == "nuts":
            from gpu_mcmc import gpu_mcmc_converge
            res = gpu_mcmc_converge(obj, sampler="nuts", progress=False, **p)
            summ = res.summary()
            stats = {"converged": bool(getattr(res, "converged", False)),
                     "total_samples": int(getattr(res, "total_samples", res.n_samples)),
                     "rhat_max": float(summ["r_hat"].max()),
                     "ess_min": float(summ["ess"].min()),
                     "acceptance_rate": res.acceptance_rate(),
                     "divergence_fraction": res.divergence_fraction()}
            median = summ["median"].to_dict()
        else:
            from gpu_nested_sampler import run_nested_sampling
            res = run_nested_sampling(obj, verbose=True, **p)
            stats = {"log_Z_mean": float(res.log_Z_mean),
                     "log_Z_std": float(res.log_Z_std), "ESS": float(res.ESS),
                     "H_mean": float(res.H_mean)}
            median = dict(zip(res.param_names, map(float, res.posterior_median)))
        elapsed = time.time() - t0
        path = project.unc_path(model, method, e, ns)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        (save_mcmc if method == "nuts" else save_jaxns)(path, res)
        meta = {"method": method, "model": model, "energy": e, "sample": ns,
                "params": job["params"], "job_id": job["id"],
                "fit_run": best["runs"][e], "fit_source": best["source"],
                "h5": best["h5"], "recipe_hash": recipe_hash,
                "chi2_best": chi_best, "npoints": int(obj.npoints),
                "param_names": names, "posterior_median": median,
                "elapsed_sec": round(elapsed, 1), **stats}
        with open(os.path.splitext(path)[0] + ".json", "w") as fh:
            json.dump(meta, fh, indent=2, default=float)
        outputs["energies"][str(e)] = {k: meta[k] for k in stats} | {
            "elapsed_sec": meta["elapsed_sec"], "file": path}
        J.update_job(project, job["id"], outputs=outputs)
        print(f"[harness] {method} {model} {e:g} eV done in {elapsed:.0f}s: "
              + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in stats.items()), flush=True)
    return outputs


# ---------------------------------------------------------------------------
# KK stoichiometry search (CPU, process pool)
# ---------------------------------------------------------------------------

def _run_stoich(project, job):
    from . import stoich as S
    outputs = {}
    for name in job["models"]:
        spec = project.get_stoich(name)
        spec.n_workers = S.reserve_workers(job["params"].get("n_workers") or spec.n_workers)
        print(f"[harness] using {spec.n_workers} worker processes (cap {S.cpu_cap()} = 50% "
              f"of {os.cpu_count()} CPUs, shared by all stoich searches)", flush=True)
        sld, prov = S.resolve_source(project, spec.source, spec.criteria)
        print(f"\n[harness] stoich {spec.describe()}", flush=True)
        print(f"[harness] source: {prov}", flush=True)
        try:
            summ = S.run_search(spec, sld, project.stoich_dir(name), prov, verbose=True)
        finally:
            S.release_workers()
        b = S.final_best(summ)
        outputs[name] = {"best": b, "n_candidates": summ["n_candidates"],
                         "search_sec": summ["search_sec"]}
        J.update_job(project, job["id"], outputs=outputs)
        print(f"[harness] stoich {name}: best {b['formula']}  ρ={b['density']:.4f} g/cm³  "
              f"RMSE={b['rmse']:.5f}  ({summ['n_candidates']} candidates, "
              f"{summ['search_sec']:.0f}s)", flush=True)
    return outputs
