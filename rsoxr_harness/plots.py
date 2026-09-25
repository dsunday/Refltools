"""
plots.py
--------
Headless (Agg) figures saved as PNG into <project>/figures/, each with a
sidecar .json recording the call so it can be regenerated.

Every function reads fits from the project's h5 by default; pass h5= and
sample= to plot fits from another file (e.g. an existing notebook h5).
"""

import contextlib
import io
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analysis as A  # noqa: E402
from .materials import load_table  # noqa: E402

_LS = ["--", ":", "-.", (0, (5, 1, 1, 1))]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _src(project, h5, sample):
    h5 = h5 or project.h5_path
    sample = sample or project.sample_name
    if not os.path.exists(h5):
        raise FileNotFoundError(f"no fits yet ({h5} does not exist)")
    return h5, sample


def _models(h5, sample, models):
    have = A.h5_models(h5, sample)
    if not models:
        return have
    missing = [m for m in models if m not in have]
    if missing:
        raise KeyError(f"no fits for {missing} in {h5}; available: {have}")
    return list(models)


def _save(project, fig, kind, models, args, out=None, tag=""):
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mtag = "-".join(models) if len(models) <= 4 else f"{models[0]}+{len(models) - 1}"
        name = "_".join(x for x in (kind, tag, mtag, stamp) if x)
        out = os.path.join(project.dir("figures"), name + ".png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump({"kind": kind, "created": datetime.now().isoformat(timespec="seconds"),
                   "args": args}, fh, indent=2, default=str)
    return out


def reference_array(project, ref):
    """A reference SLD table given a project material name or a file path."""
    if ref in project.cfg["materials"]:
        return project.material_array(ref)
    if os.path.exists(ref):
        return load_table(ref)[:, :3]
    raise KeyError(f"reference '{ref}' is neither a registered material "
                   f"({sorted(project.cfg['materials'])}) nor a file")


def _ref_label(ref):
    return os.path.splitext(os.path.basename(ref))[0] if os.sep in ref else ref


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _load_objectives(h5, sample, model, energies, criteria):
    from h5io import load_h5_objectives
    with _quiet():
        objs, structs = load_h5_objectives(h5, sample, model, criteria=criteria,
                                           energy_list=energies)
    return objs, structs


# ---------------------------------------------------------------------------
# chi² summaries
# ---------------------------------------------------------------------------

def plot_chi2_bar(project, models=None, metric="reduced", criteria="best",
                  energies=None, erange=None, log=False, h5=None, sample=None,
                  out=None):
    """
    Bar chart of mean χ² per model (best run per energy).
    metric='reduced' divides each energy's χ² by its number of points (fair
    across energies with different q ranges); 'raw' matches the notebooks.
    Only energies fitted by *every* shown model are averaged, so bars compare
    like with like; the count is printed on each bar.
    """
    h5, sample = _src(project, h5, sample)
    models = _models(h5, sample, models)
    t = A.chi2_table(h5, sample, models, criteria, reduced=(metric == "reduced"))
    t = t.loc[A.filter_energies(t.index, energies, erange)]
    common = t.dropna()
    use = common if len(common) else t
    mean, med = use.mean(), use.median()

    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(models) + 2), 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(models))
    bars = ax.bar(x, [mean[m] for m in models],
                  color=[colors[i % len(colors)] for i in range(len(models))])
    ax.plot(x, [med[m] for m in models], "k_", ms=18, mew=2, label="median")
    for b, m in zip(bars, models):
        ax.annotate(f"{mean[m]:.3g}\n(n={int(use[m].notna().sum())})",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, models, rotation=45 if len(models) > 5 else 0)
    ax.set_ylabel(("mean χ²/N" if metric == "reduced" else "mean χ²")
                  + f" ({criteria} run per energy)")
    ax.set_title(f"{sample}: model comparison")
    if log:
        ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * (1.6 if log else 1.15))
    ax.legend(fontsize=8)
    fig.tight_layout()
    args = dict(models=models, metric=metric, criteria=criteria,
                energies=[float(e) for e in use.index], h5=h5, sample=sample)
    return _save(project, fig, "chi2bar", models, args, out)


def plot_chi2_vs_energy(project, models=None, metric="reduced", criteria="best",
                        energies=None, erange=None, log=True, h5=None, sample=None,
                        out=None):
    h5, sample = _src(project, h5, sample)
    models = _models(h5, sample, models)
    t = A.chi2_table(h5, sample, models, criteria, reduced=(metric == "reduced"))
    t = t.loc[A.filter_energies(t.index, energies, erange)]
    fig, ax = plt.subplots(figsize=(10, 4))
    markers = "osD^vP*X"
    for i, m in enumerate(models):
        s = t[m].dropna()
        ax.plot(s.index, s.values, marker=markers[i % len(markers)], ms=4, label=m)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("χ²/N" if metric == "reduced" else "χ²")
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=min(len(models), 5))
    ax.set_title(f"{sample}: goodness of fit vs energy")
    fig.tight_layout()
    return _save(project, fig, "chi2energy", models,
                 dict(models=models, metric=metric, criteria=criteria, h5=h5,
                      sample=sample), out)


# ---------------------------------------------------------------------------
# SLD vs energy (wraps h5io.plot_parameter_vs_energy)
# ---------------------------------------------------------------------------

def plot_sld_vs_energy(project, layer, models=None, references=(),
                       components=("real", "imag"), criteria="best",
                       energies=None, erange=None, show_bounds=False,
                       show_chi2=True, bounds_from=(), h5=None, sample=None,
                       out=None):
    """
    Fitted SLD of one layer vs energy for several models, with reference SLD
    tables (project material names or CSV paths) overlaid as black lines.
    bounds_from: project model names (fitted or not) whose SLD windows, as
    built from their recipes, are overlaid in their own colour; energies where
    a window differs from the first fitted model's stored bounds are marked.
    """
    from h5io import plot_parameter_vs_energy
    h5, sample = _src(project, h5, sample)
    models = _models(h5, sample, models)
    pt = A.param_table(h5, sample, models, criteria)
    with_layer = set(pt.loc[pt.param == f"{layer} - sld", "model"])
    have = [m for m in models if m in with_layer]
    if not have:
        raise KeyError(f"none of {models} has a layer '{layer}'")
    avail = A.h5_energies(h5, sample)
    elist = A.filter_energies(avail, energies, erange) if (energies or erange) else None
    refs = [(r, reference_array(project, r)) for r in references]
    extra = {}                                  # model → {E: {pname: (lo, hi)}}
    if bounds_from:
        eb = elist or [e for e in project.energies if e in set(avail)]
        for m in bounds_from:
            with _quiet():
                objs = project.build_objectives(m, eb)["objectives"]
            extra[m] = {e: {q.name: (float(q.bounds.lb), float(q.bounds.ub))
                            for q in o.parameters.flattened()
                            if q.name.startswith(f"{layer} - ") and q.vary}
                        for e, o in objs.items()}

    n = len(components) + (1 if show_chi2 else 0)
    ratios = [3] * len(components) + ([1.3] if show_chi2 else [])
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 + 2.6 * len(components)),
                             sharex=True, gridspec_kw={"height_ratios": ratios})
    axes = np.atleast_1d(axes)
    for ax, comp in zip(axes, components):
        pname = f"{layer} - {'sld' if comp == 'real' else 'isld'}"
        with _quiet():
            plot_parameter_vs_energy(h5, sample, pname, have, criteria=criteria,
                                     energy_list=elist, show_bounds=show_bounds,
                                     ax=ax)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        stored = pt[(pt.model == have[0]) & (pt.param == pname)].set_index("energy")
        for i, (m, eb) in enumerate(extra.items()):
            c = colors[(len(have) + 1 + i) % len(colors)]
            E = sorted(e for e in eb if pname in eb[e])
            if not E:
                continue
            lo_, hi_ = (np.array([eb[e][pname][k] for e in E]) for k in (0, 1))
            ax.fill_between(E, lo_, hi_, color=c, alpha=0.12, lw=0)
            ax.plot(E, lo_, color=c, ls="--", lw=1.4, label=f"{m} bounds")
            ax.plot(E, hi_, color=c, ls="--", lw=1.4)
            ch = [(e, l, h) for e, l, h in zip(E, lo_, hi_) if e in stored.index
                  and not (np.isclose(l, stored.at[e, "lb"]) and np.isclose(h, stored.at[e, "ub"]))]
            for j, (e, l, h) in enumerate(ch):
                wid = h if not np.isclose(h, stored.at[e, "ub"]) else l
                ax.plot(e, wid, marker="^" if wid == h else "v", ms=8, color=c,
                        mec="k", mew=0.6, ls="none", zorder=5,
                        label=f"{m} widened" if j == 0 else "_nolegend_")
        col = 1 if comp == "real" else 2
        lo, hi = ax.get_xlim()
        for i, (r, arr) in enumerate(refs):
            ax.plot(arr[:, 0], arr[:, col], color="k", ls=_LS[i % len(_LS)], lw=1.4,
                    label=_ref_label(r))
        ax.set_xlim(lo, hi)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel(f"{layer} {'Re' if comp == 'real' else 'Im'} SLD\n(10⁻⁶ Å⁻²)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    if show_chi2:
        ax = axes[-1]
        t = A.chi2_table(h5, sample, have, criteria)
        if elist:
            t = t.loc[[e for e in elist if e in t.index]]
        for m in have:
            s = t[m].dropna()
            ax.plot(s.index, s.values, marker="o", ms=3, lw=1, label=m)
        ax.set_yscale("log")
        ax.set_ylabel("χ²/N")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Energy (eV)")
    fig.suptitle(f"{sample}: {layer} SLD vs energy", y=0.995)
    fig.tight_layout()
    return _save(project, fig, "sldenergy", have,
                 dict(layer=layer, models=have, references=list(references),
                      bounds_from=list(bounds_from),
                      components=list(components), criteria=criteria, h5=h5,
                      sample=sample), out, tag=layer)


def plot_parameter(project, param, models=None, criteria="best", energies=None,
                   erange=None, show_bounds=True, h5=None, sample=None, out=None):
    """Any stored parameter (e.g. 'SOG - thick', 'scale') vs energy."""
    from h5io import plot_parameter_vs_energy
    h5, sample = _src(project, h5, sample)
    models = _models(h5, sample, models)
    avail = A.h5_energies(h5, sample)
    elist = A.filter_energies(avail, energies, erange) if (energies or erange) else None
    fig, ax = plt.subplots(figsize=(10, 4))
    with _quiet():
        plot_parameter_vs_energy(h5, sample, param, models, criteria=criteria,
                                 energy_list=elist, show_bounds=show_bounds, ax=ax)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(project, fig, "param", models,
                 dict(param=param, models=models, criteria=criteria, h5=h5,
                      sample=sample), out, tag=param.replace(" - ", "-").replace(" ", ""))


# ---------------------------------------------------------------------------
# reflectivity
# ---------------------------------------------------------------------------

def plot_reflectivity_grid(project, models=None, energies=None, erange=None,
                           criteria="best", ncols=4, residuals=True, q4=False,
                           h5=None, sample=None, out=None):
    """Data + best fit of each model per energy, with normalised residuals."""
    h5, sample = _src(project, h5, sample)
    models = _models(h5, sample, models)
    elist = A.filter_energies(A.h5_energies(h5, sample), energies, erange)
    loaded = {m: _load_objectives(h5, sample, m, elist, criteria)[0] for m in models}
    elist = [e for e in elist if any(e in loaded[m] for m in models)]
    n = len(elist)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    height = (4.0 if residuals else 3.0) * nrows
    fig = plt.figure(figsize=(4.2 * ncols, height))
    # fixed ~0.6 in top margin; the default fraction leaves a big gap on tall grids
    outer = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.28,
                             top=1 - 0.6 / height)
    for k, e in enumerate(elist):
        cell = outer[k // ncols, k % ncols]
        if residuals:
            sub = cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax, axr = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        else:
            ax, axr = fig.add_subplot(cell), None
        first = next(loaded[m][e] for m in models if e in loaded[m])
        q, r = first.data.x, first.data.y
        dr = first.data.y_err
        w = q ** 4 if q4 else 1.0
        ax.errorbar(q, r * w, yerr=None if dr is None else dr * w, fmt="o", ms=2,
                    color="k", ecolor="0.6", elinewidth=0.6, label="data", zorder=1)
        txt = []
        for i, m in enumerate(models):
            o = loaded[m].get(e)
            if o is None:
                continue
            c = colors[i % len(colors)]
            fit = o.model(q, x_err=o.data.x_err)
            ax.plot(q, fit * w, color=c, lw=1.2, label=m, zorder=2)
            txt.append(f"{m}: {o.chisqr() / len(q):.3g}")
            if axr is not None and dr is not None:
                axr.plot(q, (r - fit) / dr, color=c, lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{e:g} eV", fontsize=10)
        ax.text(0.97, 0.97, "χ²/N\n" + "\n".join(txt), transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="lower left")
        if axr is not None:
            axr.axhline(0, color="k", lw=0.6)
            axr.set_xlim(ax.get_xlim())
            ax.tick_params(labelbottom=False)
            axr.set_xlabel("q (Å⁻¹)", fontsize=8)
            if k % ncols == 0:
                axr.set_ylabel("Δ/σ", fontsize=8)
        else:
            ax.set_xlabel("q (Å⁻¹)", fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel("R·q⁴" if q4 else "R", fontsize=8)
    fig.suptitle(f"{sample}: reflectivity ({criteria} fits)", y=1 - 0.15 / height)
    return _save(project, fig, "refl", models,
                 dict(models=models, energies=elist, criteria=criteria, q4=q4,
                      h5=h5, sample=sample), out)


# ---------------------------------------------------------------------------
# depth profiles (wraps Plotting_Refl.plot_energy_model_grid)
# ---------------------------------------------------------------------------

def plot_sld_profiles(project, models=None, energies=None, erange=None,
                      references=(), criteria="best", max_energies=6,
                      profile_xlim=None, h5=None, sample=None, out=None):
    """
    One row per energy: reflectivity fits (left) and real (solid) / imag
    (dashed) SLD depth profiles (right).  references: material names whose
    tabulated SLD is drawn as horizontal lines where a layer of that name
    exists.  Without energies, up to max_energies evenly spaced are shown.
    """
    from Plotting_Refl import plot_energy_model_grid
    h5, sample = _src(project, h5, sample)
    models = _models(h5, sample, models)
    avail = A.h5_energies(h5, sample)
    if energies or erange:
        elist = A.filter_energies(avail, energies, erange)
    else:
        idx = np.unique(np.linspace(0, len(avail) - 1, min(max_energies, len(avail))).round())
        elist = [avail[int(i)] for i in idx]
    refs = {_ref_label(r): reference_array(project, r) for r in references}
    with _quiet():
        fig, _ = plot_energy_model_grid(h5, sample, models, elist,
                                        mox_sld_arrays=refs, criteria=criteria,
                                        row_height=3.2, profile_xlim=profile_xlim)
    return _save(project, fig, "profiles", models,
                 dict(models=models, energies=elist, references=list(references),
                      criteria=criteria, h5=h5, sample=sample), out)


# ---------------------------------------------------------------------------
# SLD export
# ---------------------------------------------------------------------------

def export_sld(project, model, layer, out=None, criteria="best", register_as=None,
               h5=None, sample=None):
    """
    Save a layer's fitted SLD(E) to CSV (default <project>/sld_exports/).
    register_as='Name' also registers it as a fit_csv material in the project.
    """
    h5, sample = _src(project, h5, sample)
    out = out or os.path.join(project.path, "sld_exports",
                              f"{sample}_{model}_{layer}.csv")
    A.export_sld_csv(h5, sample, model, layer, out, criteria)
    if register_as:
        project.add_material(register_as, kind="fit_csv", path=out,
                             description=f"{sample} {model} {layer} fit ({criteria})",
                             overwrite=True)
    return out


# ---------------------------------------------------------------------------
# uncertainty (NUTS / JAXNS caches)
# ---------------------------------------------------------------------------

def _flat_samples(result, method):
    if method == "nuts":
        s = result.samples_bounded
        return s.reshape(-1, s.shape[-1])
    return result.samples


def _aligned(result, method, obj):
    """Posterior samples reordered to the objective's free-parameter order."""
    from gpu_reflect import extract_free_params
    names = [p.name for p in extract_free_params(obj)]
    flat = _flat_samples(result, method)
    rn = list(result.param_names)
    if rn == names:
        return flat, names
    if sorted(rn) != sorted(names):
        raise ValueError(f"posterior parameters {rn} do not match the objective's "
                         f"free parameters {names}")
    idx = [rn.index(n) for n in names]
    return flat[:, idx], names


def _unc_setup(project, model, method, energies, erange, h5, sample, max_energies=6):
    ns = A.unc_namespace(project, h5, sample)
    have = sorted(A.cached_uncertainty(project, model, method, ns))
    if not have:
        raise KeyError(f"no cached {method} results for {model} "
                       f"(run: python -m rsoxr_harness run {method} --model {model} ...)")
    if energies or erange:
        elist = A.filter_energies(have, energies, erange)
    else:
        idx = np.unique(np.linspace(0, len(have) - 1, min(max_energies, len(have))).round())
        elist = [have[int(i)] for i in idx]
    res = A.load_uncertainty(project, model, method, elist, ns)
    best = project.load_best(model, elist, h5=h5, sample=sample)["objectives"]
    return ns, elist, res, best


def plot_uncertainty(project, model, method="jaxns", energies=None, erange=None,
                     ci=(2.5, 97.5), n_curves=200, seed=0, h5=None, sample=None,
                     out=None):
    """
    Per energy: data + best fit + posterior band on R(q) (left), and the real
    (solid) / imaginary (dashed) SLD depth profile bands (right).
    """
    from types import SimpleNamespace
    from Plotting_Refl import sample_posterior_curves
    ns, elist, res, best = _unc_setup(project, model, method, energies, erange, h5, sample)
    n = len(elist)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.3 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    for row, e in enumerate(elist):
        obj, r = best[e], res[e]
        flat, _ = _aligned(r, method, obj)
        axr, axs = axes[row]
        d = obj.data
        q = np.linspace(d.x.min(), d.x.max(), 400)
        curves = sample_posterior_curves(obj, SimpleNamespace(samples=flat), "jaxns",
                                         q, n_curves=n_curves, seed=seed)
        axr.errorbar(d.x, d.y, yerr=d.y_err, fmt="k.", ms=2, elinewidth=0.5,
                     ecolor="0.6", label="data", zorder=1)
        lo, med, hi = np.percentile(curves["R"], [ci[0], 50, ci[1]], axis=0)
        axr.fill_between(q, lo, hi, color="C0", alpha=0.35, lw=0,
                         label=f"{ci[1] - ci[0]:g}% CI", zorder=2)
        axr.plot(q, obj.model(q), "C3--", lw=0.9, label="best fit", zorder=3)
        axr.set_yscale("log")
        axr.set_ylabel("R")
        axr.set_title(f"{model} {e:g} eV — {method}", fontsize=10)
        if row == n - 1:
            axr.set_xlabel("q (Å⁻¹)")
        axr.legend(fontsize=7, loc="upper right")
        z = curves["z"]
        for key, ls, lab in (("SLD_real", "-", "Re"), ("SLD_imag", "--", "Im")):
            lo, med, hi = np.percentile(curves[key], [ci[0], 50, ci[1]], axis=0)
            c = "C0" if key == "SLD_real" else "C1"
            axs.fill_between(z, lo, hi, color=c, alpha=0.3, lw=0)
            axs.plot(z, med, color=c, ls=ls, lw=1, label=f"{lab} SLD (median)")
        axs.set_ylabel("SLD (10⁻⁶ Å⁻²)")
        if row == n - 1:
            axs.set_xlabel("depth from surface (Å)")
        axs.legend(fontsize=7)
        axs.grid(alpha=0.3)
    fig.tight_layout()
    return _save(project, fig, "uncertainty", [model],
                 dict(model=model, method=method, energies=elist, ci=list(ci),
                      n_curves=n_curves, namespace=ns), out, tag=method)


def plot_posterior_vs_energy(project, models, method="jaxns", layer=None,
                             param=None, references=(), show_fit=True,
                             h5=None, sample=None, out=None):
    """
    Posterior median with 95% interval vs energy for a layer's SLD (real and
    imag panels) or any single parameter, several models overlaid.  CMA-ES
    best fits drawn as open markers; references as black lines.
    """
    if (layer is None) == (param is None):
        raise ValueError("give exactly one of layer= or param=")
    ns = A.unc_namespace(project, h5, sample)
    pt = A.posterior_table(project, models, method, ns)
    if pt.empty:
        raise KeyError(f"no cached {method} results for {models}")
    pnames = [f"{layer} - sld", f"{layer} - isld"] if layer else [param]
    h5p, smp = _src(project, h5, sample)
    fit = A.param_table(h5p, smp, models) if show_fit else None
    refs = [(r, reference_array(project, r)) for r in references]
    fig, axes = plt.subplots(len(pnames), 1, figsize=(10, 3.4 * len(pnames)),
                             sharex=True, squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, pn in zip(axes[:, 0], pnames):
        for i, m in enumerate(models):
            s = pt[(pt.model == m) & (pt.param == pn)].sort_values("energy")
            if s.empty:
                continue
            c = colors[i % len(colors)]
            ax.errorbar(s.energy, s["median"], yerr=[s["median"] - s.lo, s.hi - s["median"]],
                        fmt="o-", ms=3, lw=1, capsize=2, color=c,
                        label=f"{m} ({method} median, 95%)")
            if fit is not None:
                f = fit[(fit.model == m) & (fit.param == pn)]
                ax.plot(f.energy, f.value, "o", mfc="none", mec=c, ms=6,
                        label=f"{m} best fit")
        if layer:
            col = 1 if pn.endswith(" - sld") else 2
            xl = ax.get_xlim()
            for j, (r, arr) in enumerate(refs):
                ax.plot(arr[:, 0], arr[:, col], color="k", ls=_LS[j % len(_LS)],
                        lw=1.3, label=_ref_label(r))
            ax.set_xlim(xl)
        ax.set_ylabel(pn)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("Energy (eV)")
    fig.suptitle(f"{project.sample_name}: posterior {'SLD of ' + layer if layer else param}",
                 y=0.995)
    fig.tight_layout()
    return _save(project, fig, "posterior", list(models),
                 dict(models=list(models), method=method, layer=layer, param=param,
                      references=list(references), namespace=ns), out,
                 tag=(layer or param).replace(" - ", "-").replace(" ", ""))


def plot_evidence(project, models, energies=None, h5=None, sample=None, out=None):
    """
    Left: summed log Z (JAXNS) per model relative to the best, over energies
    every model has.  Right: per-energy log Z relative to the per-energy mean.
    """
    ns = A.unc_namespace(project, h5, sample)
    summ = A.sum_log_evidence(project, models, ns, energies)
    zt, st = A.evidence_table(project, models, ns, energies)
    if summ.empty or not summ["n_energies"].iloc[0]:
        raise KeyError(f"no energy has cached JAXNS results for all of {models}")
    common = zt.dropna().index
    norm = {m["params"].get("normalize") for mdl in models
            for m in A.cached_uncertainty(project, mdl, "jaxns", ns).values()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2),
                                 gridspec_kw={"width_ratios": [1, 1.6]})
    order = [m for m in models if m in summ.index]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cmap = {m: colors[i % len(colors)] for i, m in enumerate(models)}
    a1.bar(order, summ.loc[order, "delta_vs_best"], yerr=summ.loc[order, "log_Z_err"],
           color=[cmap[m] for m in order], capsize=3)
    a1.axhline(0, color="k", lw=0.8)
    a1.set_ylabel("Σ log Z − best (nats)")
    a1.set_title(f"summed over {len(common)} energies")
    rel = zt.loc[common].sub(zt.loc[common].mean(axis=1), axis=0)
    for m in order:
        a2.errorbar(rel.index, rel[m], yerr=st.loc[common, m], fmt="o-", ms=3,
                    lw=1, capsize=2, color=cmap[m], label=m)
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xlabel("Energy (eV)")
    a2.set_ylabel("log Z − mean over models")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    nl = ("normalize=True (log L ÷ N points: tempered, compare only like-for-like)"
          if norm == {True} else f"normalize={sorted(norm, key=str)}")
    fig.suptitle(f"{project.sample_name}: JAXNS evidence — {nl}", fontsize=10)
    fig.tight_layout()
    return _save(project, fig, "evidence", list(models),
                 dict(models=list(models), energies=[float(e) for e in common],
                      namespace=ns, summary=summ.reset_index().to_dict("records")), out)


def plot_corner(project, model, method, energy, h5=None, sample=None, out=None):
    import corner
    ns, elist, res, best = _unc_setup(project, model, method, [energy], None, h5, sample)
    e = elist[0]
    obj = best[e]
    flat, names = _aligned(res[e], method, obj)
    from gpu_reflect import extract_free_params
    truths = [p.value for p in extract_free_params(obj)]
    fig = corner.corner(flat, labels=names, truths=truths, truth_color="C3",
                        show_titles=True, title_kwargs={"fontsize": 8},
                        label_kwargs={"fontsize": 8}, quantiles=[0.025, 0.5, 0.975])
    fig.suptitle(f"{model} {e:g} eV — {method} (red: best fit)", y=1.01)
    return _save(project, fig, "corner", [model],
                 dict(model=model, method=method, energy=e, namespace=ns), out,
                 tag=f"{method}-{e:g}eV")


# ---------------------------------------------------------------------------
# registered optical constants
# ---------------------------------------------------------------------------

def plot_materials(project, names=None, erange=None, mark_energies=True,
                   model=None, out=None):
    """
    Real and imaginary SLD vs energy of registered materials (two panels,
    shared energy axis).  Data energies are marked on each curve where the
    fit will interpolate them.  model= restricts to that model's materials.
    Curves that are identical (same source) are noted in the legend.
    """
    if model is not None:
        names = project.get_model(model).materials_used()
    names = list(names) if names else sorted(project.cfg["materials"])
    arrays = {n: project.material_array(n) for n in names}
    E = np.array(project.energies)
    lo, hi = (min(E) - 5, max(E) + 5) if erange is None else sorted(erange)
    # identical sources → one curve, labelled with every name
    groups = []
    for n in names:
        for g in groups:
            a, b = arrays[g[0]], arrays[n]
            if a.shape == b.shape and np.allclose(a, b):
                g.append(n)
                break
        else:
            groups.append([n])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, g in enumerate(groups):
        arr = arrays[g[0]]
        m = (arr[:, 0] >= lo) & (arr[:, 0] <= hi)
        c, ls = colors[i % len(colors)], styles[i % len(styles)]
        label = " = ".join(g)
        for ax, col in zip(axes, (1, 2)):
            ax.plot(arr[m, 0], arr[m, col], color=c, ls=ls, lw=1.6, label=label)
            if mark_energies:
                Ein = E[(E >= lo) & (E <= hi)]
                ax.plot(Ein, np.interp(Ein, arr[:, 0], arr[:, col]), "o", ms=3.5,
                        color=c, mec="white", mew=0.6)
    for ax, lab in zip(axes, ("Re SLD (10⁻⁶ Å⁻²)", "Im SLD (10⁻⁶ Å⁻²)")):
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="0.5", lw=0.6)
    axes[0].legend(fontsize=8, ncol=min(len(groups), 5), loc="best")
    axes[1].set_xlabel("Energy (eV)" + ("   (dots: data energies)" if mark_energies else ""))
    axes[1].set_xlim(lo, hi)
    title = f"{project.sample_name}: loaded optical constants"
    if model:
        title += f" ({model})"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return _save(project, fig, "materials", names,
                 dict(names=names, erange=[lo, hi], model=model,
                      sources={n: project.cfg["materials"][n] for n in names}), out)
