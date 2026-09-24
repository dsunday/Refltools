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
