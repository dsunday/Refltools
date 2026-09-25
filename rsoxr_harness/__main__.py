"""
python -m rsoxr_harness <command> --project DIR ...

Commands:
    init       create a project from a data folder
    material   register a material SLD source
    models     list models + lineage
    show       layer table of a model (+ diff vs parent or --diff MODEL)
    build      build objectives; --check prints free parameters per energy
    run fit    batch CMA-ES fit of one or more models (background job)
    status     job table (+ --tail N log lines)
    cancel     stop a running job
    fits       chi² summary of fits in the project h5
    run nuts / run jaxns   posterior sampling around the best fit (cached per energy)
    unc        summary of cached NUTS/JAXNS results (+ summed evidence)
    plot       chi2 | chi2-energy | sld-energy | param | refl | profiles |
               uncertainty | posterior | evidence | corner | materials  → PNG
    export-sld save a layer's fitted SLD(E) as CSV (optionally register it)
"""

import argparse
import json
import os
import sys

import numpy as np


def _proj(a):
    from . import FitProject
    return FitProject.open(a.project)


def cmd_init(a):
    from . import FitProject
    p = FitProject.create(a.project, a.sample, a.data, file_type=a.file_type,
                          q_max=a.q_max, description=a.description or "")
    e = p.energies
    print(f"project {p.path}: {len(e)} energies {e[0]:g}–{e[-1]:g} eV")


def cmd_material(a):
    p = _proj(a)
    kw = {"kind": a.kind}
    if a.path:
        kw["path"] = a.path
    if a.formula:
        kw["formula"], kw["density"] = a.formula, a.density
    if a.description:
        kw["description"] = a.description
    src = p.add_material(a.name, overwrite=a.overwrite, **kw)
    print(f"material {a.name}: {src.kind} {src.label()}")


def cmd_models(a):
    p = _proj(a)
    print(p.models_table() if p.model_names else "(no models)")


def cmd_show(a):
    p = _proj(a)
    for m in a.model:
        print(p.show_model(m, diff_from=a.diff or "parent"))
        print()


def cmd_build(a):
    from .project import free_parameter_table
    p = _proj(a)
    energies = p.select_energies(a.energies, a.erange)
    for m in a.model:
        res = p.build_objectives(m, energies, verbose=a.verbose)
        objs = res["objectives"]
        nfree = {e: len(o.varying_parameters()) for e, o in objs.items()}
        print(f"{m}: built {len(objs)} energies, free params per energy: "
              f"{sorted(set(nfree.values()))}")
        sr = res.get("seed_report")
        if sr:
            print(f"  seeded from {sr['model']}: {len(sr['seeded'])} energies"
                  + (f"; no {sr['model']} fit at {sr['unseeded_energies']} (tabulated start)"
                     if sr["unseeded_energies"] else ""))
            for e, cl in sr["clipped"].items():
                for n, v, lo, hi in cl:
                    print(f"  clipped @ {e:g} eV: {n} = {v:.4g} outside [{lo:.4g}, {hi:.4g}]")
        if a.check:
            for e in (res["energies"] if a.all else res["energies"][:1]):
                o = objs[e]
                print(f"  {e:g} eV  chi2/N={o.chisqr() / o.npoints:.3g}")
                for n, v, lo, hi in free_parameter_table(o):
                    print(f"    {n:<22s} {v:>11.4g}   [{lo:.4g}, {hi:.4g}]")


def _energies(p, a):
    return p.select_energies(a.energies, a.erange)


def cmd_run(a):
    from . import jobs as J
    p = _proj(a)
    models = a.models
    k = a.kind
    if k in ("nuts", "jaxns") and len(models) != 1:
        print(f"error: run {k} takes one model per job", file=sys.stderr)
        return 2
    if a.h5 is None:
        for m in models:
            p.get_model(m)                  # fail early on unknown names
        if k in ("nuts", "jaxns") and p.get_model(models[0]).is_linked:
            print(f"error: {models[0]} is energy-linked; global {k} is not built yet "
                  f"(M6 step 2)", file=sys.stderr)
            return 2
    if k == "fit":
        if a.h5:
            print("error: --h5 is only for nuts/jaxns", file=sys.stderr)
            return 2
        energies = _energies(p, a)
        linked = [m for m in models if p.get_model(m).is_linked]
        if linked and (a.energies or a.erange):
            print(f"error: {', '.join(linked)} {'is' if len(linked) == 1 else 'are'} "
                  f"energy-linked and always fit their own energy set "
                  f"(change it with the link_energies op)", file=sys.stderr)
            return 2
        if linked and len(linked) == len(models):
            energies = sorted({e for m in linked for e in p.link_energy_list(m)})
    else:
        from .analysis import h5_energies, filter_energies
        h5 = a.h5 or p.h5_path
        fitted = h5_energies(h5, a.sample or p.sample_name, models[0]) \
            if os.path.exists(h5) else []
        if not fitted:
            print(f"error: {models[0]} has no fits in {h5}", file=sys.stderr)
            return 2
        energies = filter_energies(fitted, a.energies, a.erange)
    params = p.settings(k)
    if k == "fit":
        over = {"popsize": a.popsize, "n_generations": a.gens, "tol": a.tol,
                "patience": a.patience, "n_restarts": a.restarts, "seed": a.seed}
    elif k == "jaxns":
        over = {"num_live_points": a.live, "max_samples": a.max_samples,
                "seed": a.seed}
    else:
        over = {"n_chains": a.chains, "max_samples": a.max_samples,
                "n_warmup": a.warmup, "seed": a.seed}
    params.update({kk: v for kk, v in over.items() if v is not None})
    if a.normalize:
        params["normalize"] = True
    if a.no_normalize:
        params["normalize"] = False
    if k != "fit":
        params.update(force=a.force, criteria=_crit(a.criteria))
        if a.h5:
            params.update(h5=os.path.abspath(a.h5), sample=a.sample or p.sample_name)
    gpu = "cpu" if a.cpu else a.gpu
    if gpu is None:
        print("error: pass --gpu N or --cpu", file=sys.stderr)
        return 2
    if gpu in J.busy_gpus(p):
        print(f"warning: another harness job is already running on gpu {gpu}")
    job = J.new_job(p, k, models, energies, params, gpu)
    print(f"job {job['id']}: {k} {' '.join(models)} on {len(energies)} "
          f"energies ({energies[0]:g}–{energies[-1]:g} eV), gpu={gpu}, "
          f"params={params}")
    print(f"log: {job['log']}")
    proc = J.launch(p, job, detach=a.detach)
    if a.detach:
        print(f"launched pid {proc.pid}; check with: python -m rsoxr_harness "
              f"status -p {p.path}")
        return 0
    job = J.read_job(p, job["id"])
    print(f"\njob {job['id']}: {job['status']}")
    return 0 if job["status"] == "done" else 1


def cmd_stoich(a):
    from . import jobs as J
    from . import stoich as S
    p = _proj(a)
    if a.action == "list":
        print(p.stoich_table() if p.stoich_names else "(no stoich searches)")
        return 0
    if not a.name:
        raise SystemExit(f"stoich {a.action} needs a NAME")
    if a.action == "show":
        for n in a.name:
            sp = p.get_stoich(n)
            print(sp.describe())
            d = p.stoich_dir(n)
            if os.path.exists(os.path.join(d, "best.json")):
                summ, _, df, _ = S.load_result(d)
                b = summ["best"]
                print(f"  source: {summ['provenance']}")
                print(f"  search best: {b['formula']}  ρ={b['density']:.4f}  RMSE={b['rmse']:.5f}")
                if summ.get("refined"):
                    r = summ["refined"]
                    print(f"  refined ρ:   {r['density']:.5f} in {r['density_window']}  "
                          f"RMSE={r['rmse']:.5f}")
                print(df.head(a.top).to_string(index=False, float_format="{:.4f}".format))
            else:
                print("  (not run yet)")
            print()
        return 0
    if a.action == "chi2":
        for n in a.name:
            df, info = S.chi2_scan(p, n, workers=a.workers)
            print(f"\n{n}: χ²_best = {info['chi2_best']:.2f} ({info['best_formula']}, "
                  f"ρ={info['best_density']:.4f}); points {info['n_points']}, ν = {info['nu']}, "
                  f"s² = χ²/ν = {info['birge_s2']:.2f}  ({info['seconds']:.0f}s)")
            grid = S.load_chi2(p.stoich_dir(n))[2]
            for scaled in (False, True):
                iv, _, els = S.chi2_intervals(df, info, scaled=scaled, grid=grid)
                print(f"  {'Birge-scaled' if scaled else 'raw σ'} profile intervals:")
                for lv in (1.0, 4.0):
                    parts = [f"{k}: {v[0]:.3f}–{v[2]:.3f}" for k, v in iv[lv].items()]
                    n_c = next(iter(iv[lv].values()))[3]
                    print(f"    Δχ²≤{lv:g} ({n_c} formulas)  " + " | ".join(parts))
        return 0
    # run
    for n in a.name:
        p.get_stoich(n)
    # 50 % CPU budget shared by all running stoich searches on this machine
    # (the worker re-checks and reserves atomically when it starts)
    in_use = S.cpus_in_use()                  # machine-wide, any project
    req = a.workers or max(p.get_stoich(n).n_workers or 0 for n in a.name) or None
    workers = S.resolve_workers(req, in_use)
    if req is not None and workers < req:
        print(f"note: {req} workers requested; capped at {workers} "
              f"(50% of {os.cpu_count()} CPUs, {in_use} in use by other stoich jobs)")
    params = {"n_workers": workers}
    job = J.new_job(p, "stoich", list(a.name), [], params, "cpu")
    print(f"job {job['id']}: stoich {' '.join(a.name)}  ("
          + ", ".join(f"{n}: {p.get_stoich(n).n_candidates()} candidates" for n in a.name)
          + ")")
    print(f"log: {job['log']}")
    proc = J.launch(p, job, detach=a.detach)
    if a.detach:
        print(f"launched pid {proc.pid}; check with: python -m rsoxr_harness "
              f"status -p {p.path}")
        return 0
    job = J.read_job(p, job["id"])
    print(f"\njob {job['id']}: {job['status']}")
    return 0 if job["status"] == "done" else 1


def cmd_unc(a):
    import pandas as pd
    from . import analysis as A
    p = _proj(a)
    ns = A.unc_namespace(p, a.h5, a.sample)
    models = a.models
    with pd.option_context("display.width", 200, "display.max_rows", 500,
                           "display.float_format", "{:.4g}".format):
        for m in models:
            for method in ("nuts", "jaxns"):
                metas = A.cached_uncertainty(p, m, method, ns)
                if not metas:
                    continue
                keys = (["log_Z_mean", "log_Z_std", "ESS", "H_mean"] if method == "jaxns"
                        else ["converged", "rhat_max", "ess_min", "divergence_fraction"])
                df = pd.DataFrame({e: {k: md.get(k) for k in keys + ["chi2_best", "elapsed_sec"]}
                                   for e, md in metas.items()}).T
                df.index.name = "energy"
                norm = {md["params"].get("normalize") for md in metas.values()}
                print(f"\n{m} {method} ({len(metas)} energies, normalize={sorted(norm, key=str)})")
                print(df)
        if len(models) > 1:
            s = A.sum_log_evidence(p, models, ns)
            if not s.empty:
                print("\nsummed JAXNS evidence over common energies")
                print(s)


def cmd_status(a):
    from . import jobs as J
    print(J.status_table(_proj(a), a.job, a.tail))


def cmd_cancel(a):
    from . import jobs as J
    job = J.cancel(_proj(a), a.job)
    print(f"{job['id']}: {job['status']}")


def cmd_fits(a):
    import os
    import pandas as pd
    from .analysis import chi2_table, fits_summary
    p = _proj(a)
    h5 = a.h5 or p.h5_path
    sample = a.sample or p.sample_name
    if not os.path.exists(h5):
        print("(no fits yet)")
        return
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.float_format", "{:.4g}".format):
        if a.per_energy:
            print(chi2_table(h5, sample, a.models, a.criteria,
                             energies=_energies(p, a) if (a.energies or a.erange) else None))
        else:
            print(fits_summary(h5, sample, a.models, a.criteria))


def cmd_plot(a):
    from . import plots as P
    p = _proj(a)
    common = dict(criteria=_crit(a.criteria), h5=a.h5, sample=a.sample, out=a.out)
    sel = dict(energies=a.energies, erange=a.erange)
    k = a.what
    if k == "chi2":
        path = P.plot_chi2_bar(p, a.models, metric=a.metric, log=a.log, **sel, **common)
    elif k == "chi2-energy":
        path = P.plot_chi2_vs_energy(p, a.models, metric=a.metric, **sel, **common)
    elif k == "sld-energy":
        if not a.layer:
            raise SystemExit("plot sld-energy needs --layer")
        comps = {"both": ("real", "imag"), "real": ("real",), "imag": ("imag",)}[a.component]
        path = P.plot_sld_vs_energy(p, a.layer, a.models, references=a.ref or (),
                                    components=comps, show_bounds=a.bounds,
                                    bounds_from=a.bounds_from or (), **sel, **common)
    elif k == "param":
        if not a.param:
            raise SystemExit("plot param needs --param (e.g. 'SOG - thick')")
        path = P.plot_parameter(p, a.param, a.models, **sel, **common)
    elif k == "refl":
        path = P.plot_reflectivity_grid(p, a.models, ncols=a.ncols, q4=a.q4,
                                        residuals=not a.no_residuals, **sel, **common)
    elif k == "profiles":
        path = P.plot_sld_profiles(p, a.models, references=a.ref or (),
                                   profile_xlim=a.xlim, **sel, **common)
    elif k.startswith("stoich-"):
        sims = []
        for x in a.sim or []:
            f_, r_ = x.rsplit(":", 1)
            sims.append((f_, float(r_)))
        if k == "stoich-overlay":
            path = P.plot_stoich_overlay(p, a.name, formulas=sims, source=a.source,
                                         energy_mask=a.erange, out=a.out)
        elif not a.name:
            raise SystemExit(f"plot {k} needs --name")
        elif k == "stoich-results":
            path = P.plot_stoich_results(p, a.name, top_n=a.top, out=a.out)
        elif k == "stoich-chi2":
            path = P.plot_stoich_chi2(p, a.name, out=a.out)
        elif k == "stoich-accept":
            path = P.plot_stoich_accept(p, a.name, tol_pct=a.tol_pct, out=a.out)
        elif k == "stoich-density":
            path = P.plot_stoich_density(p, a.name, formula=a.formula, out=a.out)
        else:
            base = None
            if a.baseline:
                base = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.baseline}
            path = P.plot_stoich_counts(p, a.name, baseline=base, out=a.out)
    elif k == "materials":
        path = P.plot_materials(p, a.names, erange=a.erange,
                                model=a.models[0] if a.models else None, out=a.out)
    elif k == "uncertainty":
        path = P.plot_uncertainty(p, _one(a.models), a.method, h5=a.h5,
                                  sample=a.sample, out=a.out, **sel)
    elif k == "posterior":
        path = P.plot_posterior_vs_energy(p, a.models, a.method, layer=a.layer,
                                          param=a.param, references=a.ref or (),
                                          h5=a.h5, sample=a.sample, out=a.out)
    elif k == "evidence":
        path = P.plot_evidence(p, a.models, energies=a.energies, h5=a.h5,
                               sample=a.sample, out=a.out)
    elif k == "corner":
        if not a.energies or len(a.energies) != 1:
            raise SystemExit("plot corner needs exactly one --energies value")
        path = P.plot_corner(p, _one(a.models), a.method, a.energies[0], h5=a.h5,
                             sample=a.sample, out=a.out)
    print(path)


def _one(models):
    if not models or len(models) != 1:
        raise SystemExit("this plot takes exactly one --model")
    return models[0]


def cmd_export_sld(a):
    from . import plots as P
    p = _proj(a)
    path = P.export_sld(p, a.model, a.layer, out=a.out, criteria=_crit(a.criteria),
                        register_as=a.register_as, h5=a.h5, sample=a.sample)
    print(path + (f"  (registered as material '{a.register_as}')" if a.register_as else ""))


def _crit(c):
    return int(c) if str(c).lstrip("-").isdigit() else c


def cmd_worker(a):
    import signal
    from .worker import run_job

    def _term(*_):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, _term)
    return run_job(a.project, a.job)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rsoxr_harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--project", "-p", required=True)
        s.set_defaults(fn=fn)
        return s

    s = add("init", cmd_init, "create a project")
    s.add_argument("--sample", required=True)
    s.add_argument("--data", required=True, help="folder of reflectivity .dat files")
    s.add_argument("--file-type", default="smoothed", choices=["smoothed", "raw", "all"])
    s.add_argument("--q-max", type=float)
    s.add_argument("--description")

    s = add("material", cmd_material, "register a material")
    s.add_argument("name")
    s.add_argument("--kind", default="file", choices=["file", "fit_csv", "formula"])
    s.add_argument("--path")
    s.add_argument("--formula")
    s.add_argument("--density", type=float)
    s.add_argument("--description")
    s.add_argument("--overwrite", action="store_true")

    add("models", cmd_models, "list models")

    s = add("show", cmd_show, "show model recipe(s)")
    s.add_argument("model", nargs="+")
    s.add_argument("--diff", help="diff against this model instead of the parent")

    s = add("build", cmd_build, "build objectives")
    s.add_argument("model", nargs="+")
    s.add_argument("--energies", type=float, nargs="+")
    s.add_argument("--erange", type=float, nargs=2)
    s.add_argument("--check", action="store_true", help="print free parameters")
    s.add_argument("--all", action="store_true", help="--check every energy")
    s.add_argument("--verbose", action="store_true")

    s = add("run", cmd_run, "launch a background job")
    s.add_argument("kind", choices=["fit", "nuts", "jaxns"])
    s.add_argument("--models", "--model", nargs="+", required=True)
    s.add_argument("--energies", type=float, nargs="+")
    s.add_argument("--erange", type=float, nargs=2)
    s.add_argument("--gpu", type=int)
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--popsize", type=int)
    s.add_argument("--gens", type=int)
    s.add_argument("--tol", type=float)
    s.add_argument("--patience", type=int)
    s.add_argument("--restarts", type=int)
    s.add_argument("--seed", type=int)
    s.add_argument("--live", type=int, help="jaxns live points")
    s.add_argument("--max-samples", type=int)
    s.add_argument("--chains", type=int, help="nuts chains")
    s.add_argument("--warmup", type=int, help="nuts warmup steps")
    s.add_argument("--normalize", action="store_true")
    s.add_argument("--no-normalize", action="store_true")
    s.add_argument("--force", action="store_true", help="ignore cached results")
    s.add_argument("--criteria", default="best", help="which fit run to sample around")
    s.add_argument("--h5", help="sample around fits in another h5 (nuts/jaxns)")
    s.add_argument("--sample", help="sample name inside --h5")
    s.add_argument("--detach", action="store_true",
                   help="return immediately (otherwise stream the log and wait)")

    s = add("stoich", cmd_stoich, "KK stoichiometry searches: list | show | run | chi2")
    s.add_argument("action", choices=["list", "show", "run", "chi2"])
    s.add_argument("name", nargs="*")
    s.add_argument("--workers", type=int,
                   help="process-pool size (default and maximum: 50%% of CPUs, "
                        "shared with other running stoich jobs)")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--tol-pct", type=float, default=5.0,
                   help="stoich-accept: RMSE acceptance cut, %% above the best")
    s.add_argument("--detach", action="store_true")

    s = add("unc", cmd_unc, "summarise cached NUTS/JAXNS results")
    s.add_argument("--models", "--model", nargs="+", required=True)
    s.add_argument("--h5")
    s.add_argument("--sample")

    s = add("status", cmd_status, "job status")
    s.add_argument("--job")
    s.add_argument("--tail", type=int, default=0)

    s = add("cancel", cmd_cancel, "cancel a job")
    s.add_argument("--job", required=True)

    s = add("fits", cmd_fits, "chi² summary from the h5")
    s.add_argument("--models", nargs="+")
    s.add_argument("--criteria", default="best")
    s.add_argument("--per-energy", action="store_true")
    s.add_argument("--energies", type=float, nargs="+")
    s.add_argument("--erange", type=float, nargs=2)
    s.add_argument("--h5", help="read another h5 (e.g. a notebook's) instead")
    s.add_argument("--sample", help="sample name inside --h5")

    s = add("plot", cmd_plot, "make a figure (PNG path is printed)")
    s.add_argument("what", choices=["chi2", "chi2-energy", "sld-energy", "param",
                                    "refl", "profiles", "uncertainty", "posterior",
                                    "evidence", "corner", "materials",
                                    "stoich-results", "stoich-density", "stoich-counts",
                                    "stoich-accept", "stoich-chi2",
                                    "stoich-overlay"])
    s.add_argument("--name", help="stoich search name")
    s.add_argument("--formula", help="stoich-density: formula (default: best)")
    s.add_argument("--sim", nargs="+", metavar="FORMULA:RHO",
                   help="stoich-overlay: extra/simulated formula:density pairs")
    s.add_argument("--source", help="stoich-overlay without a run: material, Model:Layer or CSV")
    s.add_argument("--baseline", nargs="+", metavar="NAME=COUNT",
                   help="stoich-counts: hold these counts instead of the best")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--tol-pct", type=float, default=5.0,
                   help="stoich-accept: RMSE acceptance cut, %% above the best")
    s.add_argument("--method", default="jaxns", choices=["jaxns", "nuts"])
    s.add_argument("--names", nargs="+", help="materials to plot (plot materials)")
    s.add_argument("--models", "--model", nargs="+")
    s.add_argument("--layer")
    s.add_argument("--param")
    s.add_argument("--ref", nargs="+", help="reference materials or CSV paths")
    s.add_argument("--component", default="both", choices=["both", "real", "imag"])
    s.add_argument("--bounds", action="store_true", help="shade parameter bounds")
    s.add_argument("--bounds-from", nargs="+", metavar="MODEL",
                   help="sld-energy: overlay these models' SLD windows (fitted or not)")
    s.add_argument("--metric", default="reduced", choices=["reduced", "raw"])
    s.add_argument("--log", action="store_true")
    s.add_argument("--q4", action="store_true")
    s.add_argument("--no-residuals", action="store_true")
    s.add_argument("--ncols", type=int, default=4)
    s.add_argument("--xlim", type=float, nargs=2)
    s.add_argument("--energies", type=float, nargs="+")
    s.add_argument("--erange", type=float, nargs=2)
    s.add_argument("--criteria", default="best")
    s.add_argument("--h5")
    s.add_argument("--sample")
    s.add_argument("--out")

    s = add("export-sld", cmd_export_sld, "export fitted layer SLD to CSV")
    s.add_argument("--model", required=True)
    s.add_argument("--layer", required=True)
    s.add_argument("--register-as")
    s.add_argument("--criteria", default="best")
    s.add_argument("--h5")
    s.add_argument("--sample")
    s.add_argument("--out")

    s = add("_worker", cmd_worker, argparse.SUPPRESS)
    s.add_argument("--job", required=True)

    a = ap.parse_args(argv)
    from .project import ProjectError
    try:
        rc = a.fn(a)
    except (ProjectError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
