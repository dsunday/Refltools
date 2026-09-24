"""
analysis.py
-----------
Read-only summaries of fit results in the h5 (h5io layout).
"""

import numpy as np
import pandas as pd


def _pick_run(model_grp, criteria):
    runs = [k for k in model_grp if k.startswith("run_")]
    if not runs:
        return None
    if criteria == "best":
        return min(runs, key=lambda k: model_grp[k].attrs.get("chi_sq_final", np.inf))
    if criteria == "last":
        return max(runs, key=lambda k: int(k.split("_")[1]))
    name = f"run_{int(criteria)}"
    return name if name in model_grp else None


def chi2_table(h5_path, sample_name, models=None, criteria="best",
               energies=None, reduced=True):
    """
    DataFrame [energy × model] of chi² (reduced by the number of data points
    when reduced=True) for the chosen run per energy.  NaN where absent.
    """
    import h5py
    rows = {}
    with h5py.File(h5_path, "r") as f:
        if sample_name not in f:
            raise KeyError(f"sample '{sample_name}' not in {h5_path}")
        g = f[sample_name]
        for ek in g:
            try:
                e = float(ek)
            except ValueError:
                continue
            if energies is not None and not any(abs(e - x) < 1e-6 for x in energies):
                continue
            eg = g[ek]
            for m in (models or list(eg)):
                if m not in eg:
                    continue
                run = _pick_run(eg[m], criteria)
                if run is None:
                    continue
                chi = float(eg[m][run].attrs["chi_sq_final"])
                if reduced:
                    chi /= len(eg[m][run]["data/q"])
                rows.setdefault(e, {})[m] = chi
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "energy_eV"
    if models:
        df = df.reindex(columns=[m for m in models if m in df.columns])
    return df


def fits_summary(h5_path, sample_name, models=None, criteria="best"):
    """Per model: n energies fitted, mean / median reduced chi²."""
    t = chi2_table(h5_path, sample_name, models, criteria)
    return pd.DataFrame({"n_energies": t.notna().sum(),
                         "mean_chi2_red": t.mean(),
                         "median_chi2_red": t.median()})


def natural_models(names):
    """Sort Model1, Model2, …, Model10 numerically; other names after."""
    import re

    def key(n):
        m = re.fullmatch(r"(.*?)(\d+)", n)
        return (0, m.group(1), int(m.group(2))) if m else (1, n, 0)
    return sorted(names, key=key)


def h5_models(h5_path, sample_name):
    """All model names present under a sample (any energy)."""
    import h5py
    out = set()
    with h5py.File(h5_path, "r") as f:
        g = f[sample_name]
        for ek in g:
            try:
                float(ek)
            except ValueError:
                continue
            out.update(g[ek].keys())
    return natural_models(out)


def h5_energies(h5_path, sample_name, model=None):
    import h5py
    with h5py.File(h5_path, "r") as f:
        g = f[sample_name]
        es = []
        for ek in g:
            try:
                e = float(ek)
            except ValueError:
                continue
            if model is None or model in g[ek]:
                es.append(e)
    return sorted(es)


def filter_energies(available, energies=None, erange=None, tol=1e-3):
    """Subset of available energies by explicit list and/or inclusive range."""
    available = sorted(available)
    if energies is None and erange is None:
        return available
    out = set()
    for e in energies or []:
        hits = [a for a in available if abs(a - float(e)) <= tol]
        if not hits:
            raise KeyError(f"no fits at {e} eV; available: "
                           f"{', '.join(f'{a:g}' for a in available)}")
        out.update(hits)
    if erange is not None:
        lo, hi = sorted(map(float, erange))
        out.update(a for a in available if lo - tol <= a <= hi + tol)
    return sorted(out)


def param_table(h5_path, sample_name, models=None, criteria="best",
                energies=None):
    """
    Long DataFrame of fitted parameters for the chosen run per energy:
    energy, model, run, param, value, lb, ub, vary, stderr, chi2, npts.
    Names are exact h5 names, e.g. 'SOG - sld' ('SOG' never matches 'SOG2').
    """
    import h5py
    rows = []
    with h5py.File(h5_path, "r") as f:
        g = f[sample_name]
        for ek in g:
            try:
                e = float(ek)
            except ValueError:
                continue
            if energies is not None and not any(abs(e - x) < 1e-6 for x in energies):
                continue
            eg = g[ek]
            for m in (models or list(eg)):
                if m not in eg:
                    continue
                run = _pick_run(eg[m], criteria)
                if run is None:
                    continue
                rg = eg[m][run]
                pg = rg["parameters"]
                names = [n.decode() if isinstance(n, bytes) else n for n in pg["names"][()]]
                se = pg["stderr"][()] if "stderr" in pg else np.full(len(names), np.nan)
                chi = float(rg.attrs["chi_sq_final"])
                npts = len(rg["data/q"])
                for i, n in enumerate(names):
                    rows.append(dict(energy=e, model=m, run=run, param=n,
                                     value=float(pg["final_values"][i]),
                                     lb=float(pg["final_lb"][i]),
                                     ub=float(pg["final_ub"][i]),
                                     vary=bool(pg["final_vary"][i]),
                                     stderr=float(se[i]), chi2=chi, npts=npts))
    return pd.DataFrame(rows)


def layer_sld_vs_energy(h5_path, sample_name, model, layer, criteria="best"):
    """ndarray [E, Re, Im] of a layer's fitted SLD for one model."""
    t = param_table(h5_path, sample_name, [model], criteria)
    if t.empty:
        raise KeyError(f"no fits for {model} in {h5_path}")
    re_ = t[t.param == f"{layer} - sld"].set_index("energy")["value"]
    im_ = t[t.param == f"{layer} - isld"].set_index("energy")["value"]
    if re_.empty:
        layers = sorted({p.split(" - ")[0] for p in t.param if " - sld" in p})
        raise KeyError(f"{model} has no layer '{layer}'; layers: {layers}")
    both = pd.concat([re_, im_], axis=1, keys=["re", "im"]).dropna().sort_index()
    return np.column_stack([both.index.values, both["re"].values, both["im"].values])


def export_sld_csv(h5_path, sample_name, model, layer, out_path, criteria="best"):
    """
    Write a layer's fitted SLD(E) as Energy_eV,Real_SLD,Imag_SLD – the format
    Model_Setup.load_material_sld_array reads, so it can seed the next sample.
    """
    import os
    arr = layer_sld_vs_energy(h5_path, sample_name, model, layer, criteria)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savetxt(out_path, arr, delimiter=",", header="Energy_eV,Real_SLD,Imag_SLD",
               comments="")
    return out_path


# ---------------------------------------------------------------------------
# uncertainty caches  (<project>/uncertainty/<sample-or-ext>/<model>/<method>_<E>eV.npz)
# ---------------------------------------------------------------------------

def unc_namespace(project, h5=None, sample=None):
    return f"ext-{sample or project.sample_name}" if h5 else project.sample_name


def cached_uncertainty(project, model, method, ns=None):
    """{energy: meta dict} for cached results of one model/method."""
    import glob
    import json
    import os
    ns = ns or project.sample_name
    d = os.path.join(project.dir("uncertainty"), ns, model)
    out = {}
    for js in glob.glob(os.path.join(d, f"{method}_*eV.json")):
        with open(js) as fh:
            meta = json.load(fh)
        if os.path.exists(os.path.splitext(js)[0] + ".npz"):
            out[float(meta["energy"])] = meta
    return dict(sorted(out.items()))


def load_uncertainty(project, model, method, energies=None, ns=None):
    """{energy: loaded result} (Plotting_Refl.load_mcmc / load_jaxns)."""
    from Plotting_Refl import load_mcmc, load_jaxns
    loader = load_mcmc if method == "nuts" else load_jaxns
    metas = cached_uncertainty(project, model, method, ns)
    if energies is not None:
        metas = {e: m for e, m in metas.items()
                 if any(abs(e - float(x)) < 1e-3 for x in energies)}
    return {e: loader(project.unc_path(model, method, e, ns or project.sample_name))
            for e in metas}


def posterior_table(project, models, method, ns=None, energies=None):
    """
    Long DataFrame: model, energy, param, median, lo (2.5%), hi (97.5%), std
    from cached posterior samples.
    """
    rows = []
    for m in models:
        for e, r in load_uncertainty(project, m, method, energies, ns).items():
            flat = (r.samples_bounded.reshape(-1, r.samples_bounded.shape[-1])
                    if method == "nuts" else r.samples)
            for i, name in enumerate(r.param_names):
                col = flat[:, i]
                lo, med, hi = np.percentile(col, [2.5, 50, 97.5])
                rows.append(dict(model=m, energy=e, param=name, median=med,
                                 lo=lo, hi=hi, std=float(col.std())))
    return pd.DataFrame(rows)


def evidence_table(project, models, ns=None, energies=None):
    """DataFrame [energy × model] of JAXNS log Z, and one of its std."""
    z, s = {}, {}
    for m in models:
        for e, meta in cached_uncertainty(project, m, "jaxns", ns).items():
            if energies is not None and not any(abs(e - float(x)) < 1e-3 for x in energies):
                continue
            z.setdefault(e, {})[m] = meta["log_Z_mean"]
            s.setdefault(e, {})[m] = meta["log_Z_std"]
    zt = pd.DataFrame.from_dict(z, orient="index").sort_index()
    st = pd.DataFrame.from_dict(s, orient="index").sort_index()
    return zt.reindex(columns=[m for m in models if m in zt.columns]), \
        st.reindex(columns=[m for m in models if m in st.columns])


def sum_log_evidence(project, models, ns=None, energies=None):
    """
    Total log Z per model summed over the energies *every* model has
    (errors in quadrature), with Δ log Z vs the best model.
    """
    zt, st = evidence_table(project, models, ns, energies)
    common = zt.dropna().index
    tot = zt.loc[common].sum()
    err = np.sqrt((st.loc[common] ** 2).sum())
    df = pd.DataFrame({"log_Z": tot, "log_Z_err": err,
                       "n_energies": len(common)})
    df["delta_vs_best"] = df["log_Z"] - df["log_Z"].max()
    return df.sort_values("log_Z", ascending=False)
