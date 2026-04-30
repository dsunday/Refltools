import numpy as np
from copy import deepcopy
import pandas as pd
import re
import os
import pickle
from datetime import datetime
from scipy.interpolate import interp1d

from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel, MaterialSLD
from refnx.reflect.structure import isld_profile

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

CLASSICAL_ELECTRON_RADIUS = 2.8179403227e-15  # metres
AVOGADRO_NUMBER = 6.022140857e23              # mol^-1
HC_EV_NM = 1239.8                             # hc in eV·nm


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_reflectometry_model(materials_list, layer_params, layer_order=None,
                                ignore_layers=None, sample_name=None,
                                energy=None, wavelength=None, probe="x-ray"):
    """
    Create a complete reflectometry model.  Supports both SLD-based and
    density-based material definitions, detected automatically.

    Materials list entries use either:
        SLD approach  : {'name': ..., 'real': ..., 'imag': ...}
        Density approach: {'name': ..., 'formula': ..., 'density': ...}

    layer_params entries may contain any of:
        thickness           – initial value (Å)
        roughness           – initial value (Å)
        thickness_bounds    – (lower, upper, vary)
        roughness_bounds    – (lower, upper, vary)
        sld_real_bounds     – (lower, upper, vary)   [SLD approach only]
        sld_imag_bounds     – (lower, upper, vary)   [SLD approach only]
        density_bounds      – (lower, upper, vary)   [density approach only]

    Args:
        materials_list : list of material dicts
        layer_params   : dict of {name: param_dict}
        layer_order    : list of layer names, top to bottom
        ignore_layers  : layer names excluded from model-name generation
                         (default: ["Si", "SiO2"])
        sample_name    : label used in the generated model name
        energy         : X-ray energy in eV (converted to wavelength internally)
        wavelength     : wavelength in Å (alternative to energy)
        probe          : "x-ray" (default) or "neutron"

    Returns:
        (materials, layers, structure, model_name)
    """
    if ignore_layers is None:
        ignore_layers = ["Si", "SiO2"]

    # --- validate probe / wavelength ------------------------------------------
    if probe not in ("x-ray", "neutron"):
        raise ValueError(f"Invalid probe type '{probe}'. Must be 'x-ray' or 'neutron'.")

    if probe == "x-ray":
        if energy is not None and wavelength is None:
            wavelength = 12.398 / energy          # keV → Å  (12.398 eV·Å ≡ hc/1000)
        # wavelength stays None if neither given – only matters for density approach
    elif probe == "neutron" and wavelength is None:
        raise ValueError("For neutron measurements wavelength must be provided.")

    # --- detect approach from first material -----------------------------------
    def _detect(info):
        if 'real' in info and 'imag' in info:
            return "SLD"
        if 'density' in info:
            return "density"
        raise ValueError(
            f"Material '{info.get('name','?')}' must have 'real'/'imag' (SLD) "
            f"or 'density' (density) keys."
        )

    approach = _detect(materials_list[0])

    # --- build material objects ------------------------------------------------
    materials = {}
    density_params = {}  # only used in density approach

    for info in materials_list:
        name = info['name']
        mode = _detect(info)
        if mode != approach:
            raise ValueError(
                "Mixed material approaches detected. All materials must use "
                "either SLD (real/imag) or density (formula/density)."
            )

        if mode == "SLD":
            real = info['real']
            imag = info['imag']
            materials[name] = SLD(real + (imag if isinstance(imag, complex) else imag * 1j),
                                  name=name)
        else:  # density
            density = info['density']
            dp = Parameter(density, name=f"{name}_density")
            density_params[name] = dp
            if 'formula' in info:
                if wavelength is None:
                    raise ValueError(
                        "energy or wavelength must be supplied when using "
                        "the density approach with X-ray probe."
                    )
                materials[name] = MaterialSLD(info['formula'], density=density,
                                              probe=probe, wavelength=wavelength,
                                              name=name)
            else:
                sld_r = info.get('sld_real', 0)
                sld_i = info.get('sld_imag', 0)
                materials[name] = SLD(sld_r + sld_i * 1j, name=name)

    # --- build layers and apply bounds ----------------------------------------
    Layer = {}
    varying_params = {"T": set(), "Rg": set()}
    if approach == "SLD":
        varying_params.update({"R": set(), "I": set()})
    else:
        varying_params["D"] = set()

    materials_varying = set()

    for name, params in layer_params.items():
        if name not in materials:
            continue

        thickness = params.get("thickness", 0)
        roughness = params.get("roughness", 0)
        Layer[name] = materials[name](thickness, roughness)

        has_varying = False

        if approach == "SLD":
            if "sld_real_bounds" in params:
                lo, hi, vary = params["sld_real_bounds"]
                Layer[name].sld.real.setp(vary=vary, bounds=(lo, hi))
                if vary and name not in ignore_layers:
                    varying_params["R"].add(name)
                    has_varying = True

            if "sld_imag_bounds" in params:
                lo, hi, vary = params["sld_imag_bounds"]
                Layer[name].sld.imag.setp(vary=vary, bounds=(lo, hi))
                if vary and name not in ignore_layers:
                    varying_params["I"].add(name)
                    has_varying = True

        else:  # density
            if "density_bounds" in params:
                lo, hi, vary = params["density_bounds"]
                if name in density_params and isinstance(materials[name], MaterialSLD):
                    dp = density_params[name]
                    dp.setp(bounds=(lo, hi), vary=vary)
                    materials[name].density = dp
                    if vary and name not in ignore_layers:
                        varying_params["D"].add(name)
                        has_varying = True

        if "thickness_bounds" in params:
            lo, hi, vary = params["thickness_bounds"]
            Layer[name].thick.setp(bounds=(lo, hi), vary=vary)
            if vary and name not in ignore_layers:
                varying_params["T"].add(name)
                has_varying = True

        if "roughness_bounds" in params:
            lo, hi, vary = params["roughness_bounds"]
            Layer[name].rough.setp(bounds=(lo, hi), vary=vary)
            if vary and name not in ignore_layers:
                varying_params["Rg"].add(name)
                has_varying = True

        if has_varying:
            materials_varying.add(name)

    # --- assemble structure ----------------------------------------------------
    structure = None
    if layer_order:
        structure = Layer[layer_order[0]]
        for lname in layer_order[1:]:
            structure = structure | Layer[lname]

    # --- generate model name --------------------------------------------------
    if sample_name is None:
        sample_name = "Unknown"
    energy_str = str(energy) if energy is not None else "Unknown"
    active = [l for l in (layer_order or [])
              if l not in ignore_layers and l != "air"]
    model_name = f"{sample_name}_{energy_str}_{len(active)}Layers"

    return materials, Layer, structure, model_name


# ---------------------------------------------------------------------------
# Objective creation
# ---------------------------------------------------------------------------

def create_model_and_objective(structure, data, model_name=None, scale=1.0,
                                bkg=None, dq=1.6, q_offset=0.0,
                                vary_scale=True, vary_bkg=True,
                                vary_dq=False, vary_qoffset=False,
                                scale_bounds=(0.1, 10), bkg_bounds=(0.01, 10),
                                dq_bounds=(0.5, 2.0),
                                qoffset_bounds=(-0.01, 0.01),
                                transform='logY'):
    """
    Create a ReflectModel and Objective from a structure and dataset.

    Args:
        structure      : refnx Structure object
        data           : ReflectDataset or Data1D
        model_name     : optional string label
        scale          : initial scale factor
        bkg            : initial background (auto-set to data minimum if None)
        dq             : resolution smearing (%)
        q_offset       : initial Q offset
        vary_scale/bkg/dq/qoffset : whether each instrument parameter varies
        scale_bounds   : (factor_low, factor_high) multiplied by scale value
        bkg_bounds     : (factor_low, factor_high) multiplied by bkg value
        dq_bounds      : absolute (lower, upper) for dq
        qoffset_bounds : absolute (lower, upper) for q_offset
        transform      : 'logY', 'YX4', etc.

    Returns:
        (model, objective)
    """
    if bkg is None:
        try:
            bkg = float(min(data.y))
        except (AttributeError, TypeError):
            try:
                bkg = float(min(data.data[:, 1]))
            except Exception:
                bkg = 1e-6
        print(f"Auto-setting background to {bkg:.3e}")

    model = ReflectModel(structure, scale=scale, bkg=bkg, dq=dq,
                         q_offset=q_offset, name=model_name)

    if vary_scale:
        model.scale.setp(bounds=(scale * scale_bounds[0],
                                 scale * scale_bounds[1]), vary=True)
    if vary_bkg:
        model.bkg.setp(bounds=(bkg * bkg_bounds[0],
                               bkg * bkg_bounds[1]), vary=True)
    if vary_dq:
        model.dq.setp(bounds=dq_bounds, vary=True)
    if vary_qoffset:
        model.q_offset.setp(bounds=qoffset_bounds, vary=True)

    objective = Objective(model, data, transform=Transform(transform))
    return model, objective


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def run_fitting(objective, method='differential_evolution',
                workers=-1, popsize=15, steps=1000, burn=500,
                nthin=1, nwalkers=100,
                sampler='emcee', sampler_kws=None,
                save_dir=None, save_objective=False, save_results=False,
                structure=None, model_name=None, verbose=False):
    """
    Optimise and optionally MCMC-sample a refnx Objective.

    Args:
        objective    : refnx Objective
        method       : scipy optimisation method (default 'differential_evolution')
        workers      : parallel workers for differential evolution (-1 = all cores)
        popsize      : population size for differential evolution
        steps        : MCMC steps (0 to skip MCMC).
                       For pymc: draws per chain.  For dynesty: not used.
        burn         : burn-in steps to discard.
                       For pymc: passed as nburn to process_chain.
                       For dynesty: not used.
        nthin        : thinning factor.
                       For pymc: passed as nthin to process_chain.
                       For dynesty: not used.
        nwalkers     : number of MCMC walkers (emcee) or chains (pymc).
                       Not used by dynesty.
        sampler      : 'emcee' (default), 'pymc', or 'dynesty'
        sampler_kws  : extra keyword arguments forwarded to the sampler call.
                       For emcee → fitter.sample(); for pymc → pm.sample();
                       for dynesty → DynamicNestedSampler().
                       For pymc, 'step' key overrides the default DEMetropolis step.
        save_dir     : directory to write pickle files (None = no saving)
        save_objective: save objective to <model_name>_objective.pkl
        save_results : save combined results+structure to
                       <model_name>_results_structure.pkl
        structure    : Structure associated with this objective (for saving)
        model_name   : override model name (defaults to objective.model.name)
        verbose      : print progress

    Returns:
        results dict with keys:
            objective, structure, model_name,
            initial_chi_squared, optimized_chi_squared,
            optimized_parameters,
            mcmc_samples (array or None),
            mcmc_stats   (dict or None),
            log_evidence (float or None — dynesty only)
    """
    if model_name is None:
        model_name = getattr(objective.model, 'name', 'unnamed_model')

    if verbose:
        print(f"Fitting model: {model_name}")

    results = {
        'objective':            objective,
        'structure':            structure,
        'model_name':           model_name,
        'initial_chi_squared':  objective.chisqr(),
        'optimized_parameters': None,
        'optimized_chi_squared': None,
        'mcmc_samples':         None,
        'mcmc_stats':           None,
        'log_evidence':         None,
    }

    # --- optimisation ---------------------------------------------------------
    fitter = CurveFitter(objective)
    if verbose:
        print(f"Optimising with {method}...")

    if method == 'differential_evolution':
        fitter.fit(method, workers=workers, popsize=popsize)
    else:
        fitter.fit(method)

    results['optimized_parameters'] = objective.parameters.pvals.copy()
    results['optimized_chi_squared'] = objective.chisqr()

    if verbose:
        print(f"Optimisation done. χ² = {results['optimized_chi_squared']:.6g}")

    # --- MCMC -----------------------------------------------------------------
    if steps > 0:
        _skws = dict(sampler_kws or {})
        if verbose:
            print(f"MCMC ({sampler}): {steps} steps, {burn} burn-in, "
                  f"{nwalkers} walkers/chains …")
        try:
            if sampler == 'emcee':
                fitter.sample(steps, nthin=nthin, **_skws)
                chain = fitter.chain

                if chain is not None and burn > 0:
                    if chain.ndim == 3:
                        b = min(burn, chain.shape[1])
                        chain = chain[:, b:, :]
                    elif chain.ndim == 2:
                        b = min(burn, chain.shape[0])
                        chain = chain[b:, :]
                    if verbose:
                        print(f"Removed {b} burn-in steps. Chain shape: {chain.shape}")

                results['mcmc_samples'] = chain
                if chain is not None:
                    results['mcmc_stats'] = calculate_mcmc_statistics(
                        chain, objective.parameters)

            elif sampler == 'pymc':
                from refnx.analysis import pymc_model
                from refnx.analysis import process_chain as _process_chain
                import pymc as pm
                step_fn = _skws.pop('step', pm.DEMetropolis())
                nvary = len(objective.varying_parameters())
                # DEMetropolis requires ≥ ndim+1 chains; silently bump if needed
                n_chains = nwalkers
                if isinstance(step_fn, pm.DEMetropolis):
                    min_chains = nvary + 1
                    if n_chains < min_chains:
                        print(f"  PyMC DEMetropolis requires ≥{min_chains} chains "
                              f"for {nvary} parameters; bumping nwalkers "
                              f"{n_chains} → {min_chains}")
                        n_chains = min_chains
                with pymc_model(objective) as _model:
                    starter = {f"p{n}": p.value
                               for n, p in enumerate(objective.varying_parameters())}
                    trace = pm.sample(draws=steps, chains=n_chains,
                                      initvals=starter, step=step_fn,
                                      **_skws)
                raw = np.stack([trace.posterior[f"p{i}"].data
                                for i in range(nvary)], axis=-1)  # (chains, draws, nvary)
                chain = raw.transpose(1, 0, 2)                    # (draws, chains, nvary)
                _process_chain(objective, chain, nburn=burn, nthin=nthin)
                results['mcmc_samples'] = chain[burn::nthin] if burn < chain.shape[0] else chain
                results['mcmc_stats'] = _stats_from_param_chains(objective)

            elif sampler == 'dynesty':
                import dynesty
                from refnx.analysis import process_chain as _process_chain
                ndim = len(objective.varying_parameters())
                ns = dynesty.DynamicNestedSampler(
                    objective.logl, objective.prior_transform,
                    ndim=ndim, **_skws)
                ns.run_nested()
                chain = ns.results.samples_equal()  # (nsamples, ndims)
                _process_chain(objective, chain[:, None, :])
                results['mcmc_samples'] = chain
                results['log_evidence'] = float(ns.results.logz[-1])
                results['mcmc_stats'] = _stats_from_param_chains(objective)

            else:
                raise ValueError(
                    f"sampler must be 'emcee', 'pymc', or 'dynesty' — got {sampler!r}")

        except Exception as exc:
            print(f"MCMC ({sampler}) failed: {exc}")

    # --- optional file saving -------------------------------------------------
    if save_dir is not None:
        save_fitting_files(results, save_dir, model_name,
                          save_objective, save_results, structure)

    if verbose:
        print(f"Fitting complete: {model_name}")

    return results


# ---------------------------------------------------------------------------
# MCMC statistics
# ---------------------------------------------------------------------------

def calculate_mcmc_statistics(chain, parameters):
    """
    Compute per-parameter statistics from an MCMC chain.

    Args:
        chain      : ndarray, shape (nwalkers, nsteps, nparams) or (nsteps, nparams)
        parameters : refnx Parameters object

    Returns:
        dict mapping parameter name → {'mean', 'median', 'std', 'percentiles'}
    """
    stats = {}
    try:
        flat = chain.reshape(-1, chain.shape[-1]) if chain.ndim == 3 else chain
        names = [p.name for p in parameters.flattened() if p.vary]
        for i, name in enumerate(names):
            if i < flat.shape[1]:
                s = flat[:, i]
                stats[name] = {
                    'mean':        float(np.mean(s)),
                    'median':      float(np.median(s)),
                    'std':         float(np.std(s)),
                    'percentiles': np.percentile(s, [16, 50, 84]),
                }
    except Exception as exc:
        print(f"Error calculating MCMC statistics: {exc}")
    return stats


def _stats_from_param_chains(objective):
    """Build mcmc_stats from param.chain set by refnx process_chain()."""
    stats = {}
    for p in objective.varying_parameters():
        chain = getattr(p, 'chain', None)
        if chain is None or chain.size == 0:
            continue
        stats[p.name] = {
            'mean':        float(np.mean(chain)),
            'median':      float(np.median(chain)),
            'std':         float(np.std(chain)),
            'percentiles': np.percentile(chain, [16, 50, 84]),
        }
    return stats


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def get_param_type(param_name):
    """
    Classify a refnx parameter name into a human-readable type string.

    Returns one of: 'density', 'thickness', 'roughness', 'scale',
    'background', 'sld_real', 'sld_imag', 'sld', 'resolution', 'other'.
    """
    n = param_name.lower()
    if '_density' in n:
        return 'density'
    if 'thick' in n:
        return 'thickness'
    if 'rough' in n:
        return 'roughness'
    if 'scale' in n:
        return 'scale'
    if 'bkg' in n:
        return 'background'
    if 'isld' in n:
        return 'sld_imag'
    if 'sld' in n:
        return 'sld_real' if 'real' in n else ('sld_imag' if 'imag' in n else 'sld')
    if 'dq' in n:
        return 'resolution'
    return 'other'


def print_fit_results(objective, filter_parameters=None, show_substrate=False,
                      show_other=False):
    """
    Print current parameter values for an Objective with colour coding.

    Colour key
    ----------
    Red   – varying parameter within 1 % of a bound
    Green – varying parameter not near bounds
    Plain – fixed parameter

    Args:
        objective         : refnx Objective
        filter_parameters : string or list of strings to include (substring match)
        show_substrate    : include air / Si / SiO2 parameters (default False)
        show_other        : include parameters of type 'other' (default False)
    """
    RED, GREEN, BLUE, RESET = '\033[91m', '\033[92m', '\033[94m', '\033[0m'

    rows = []
    for param in objective.parameters.flattened():
        name = param.name
        ptype = get_param_type(name)
        value = param.value
        vary  = getattr(param, 'vary', False)
        stderr = getattr(param, 'stderr', None)

        bl = bh = None
        try:
            bounds = getattr(param, 'bounds', None)
            if bounds is not None:
                if hasattr(bounds, 'lb'):
                    bl, bh = bounds.lb, bounds.ub
                elif isinstance(bounds, tuple) and len(bounds) == 2:
                    bl, bh = bounds
        except Exception:
            pass

        rows.append(dict(parameter=name, param_type=ptype, value=value,
                         stderr=stderr, vary=vary, bound_low=bl, bound_high=bh))

    if not rows:
        print("No parameters found.")
        return

    df = pd.DataFrame(rows)

    # --- substrate filter -----------------------------------------------------
    if not show_substrate:
        mask = (df['parameter'].str.contains('air', case=False) |
                df['parameter'].str.contains(r'^Si -', case=False) |
                df['parameter'].str.contains('SiO2 -', case=False))
        df = df[~mask]

    # --- 'other' filter -------------------------------------------------------
    if not show_other:
        df = df[df['param_type'] != 'other']

    # --- user filter ----------------------------------------------------------
    if filter_parameters:
        if isinstance(filter_parameters, str):
            filter_parameters = [filter_parameters]
        mask = pd.Series([False] * len(df), index=df.index)
        for f in filter_parameters:
            mask |= df['parameter'].str.contains(f, case=False)
        df = df[mask]

    if df.empty:
        print("No matching parameters found.")
        return

    gof = objective.chisqr()
    print(f"χ² = {gof:.6g}\n")

    df_sorted = df.sort_values(['param_type', 'vary', 'parameter'],
                               ascending=[True, False, True])
    current_type = None

    for _, row in df_sorted.iterrows():
        if row['param_type'] != current_type:
            current_type = row['param_type']
            print(f"\n{BLUE}--- {current_type.upper()} ---{RESET}")

        # near-bounds check
        near = False
        if (row['bound_low'] is not None and row['bound_high'] is not None and
                row['value'] is not None):
            span = row['bound_high'] - row['bound_low']
            thresh = 0.01 * span
            near = (abs(row['value'] - row['bound_low']) < thresh or
                    abs(row['bound_high'] - row['value']) < thresh)

        # format value
        pt = row['param_type']
        if pt == 'density':
            val_str = f"{row['value']:.4f} g/cm³"
            err_str = f" ± {row['stderr']:.4f} g/cm³" if row['vary'] and row['stderr'] else ""
            bnd_str = (f" (bounds: {row['bound_low']:.4f}–{row['bound_high']:.4f} g/cm³)"
                       if row['bound_low'] is not None else "")
        elif pt in ('thickness', 'roughness'):
            val_str = f"{row['value']:.2f} Å"
            err_str = f" ± {row['stderr']:.2f} Å" if row['vary'] and row['stderr'] else ""
            bnd_str = (f" (bounds: {row['bound_low']:.2f}–{row['bound_high']:.2f} Å)"
                       if row['bound_low'] is not None else "")
        elif pt in ('sld_real', 'sld_imag', 'sld'):
            val_str = f"{row['value']:.6g} ×10⁻⁶ Å⁻²"
            err_str = f" ± {row['stderr']:.6g}" if row['vary'] and row['stderr'] else ""
            bnd_str = (f" (bounds: {row['bound_low']:.6g}–{row['bound_high']:.6g})"
                       if row['bound_low'] is not None else "")
        else:
            val_str = f"{row['value']:.6g}"
            err_str = f" ± {row['stderr']:.6g}" if row['vary'] and row['stderr'] else ""
            bnd_str = (f" (bounds: {row['bound_low']:.6g}–{row['bound_high']:.6g})"
                       if row['bound_low'] is not None else "")

        vary_str = " (varying)" if row['vary'] else " (fixed)"
        line = f"  {row['parameter']}: {val_str}{err_str}{bnd_str}{vary_str}"

        if near:
            print(f"{RED}{line}{RESET}")
        elif row['vary']:
            print(f"{GREEN}{line}{RESET}")
        else:
            print(line)


def density_summary(objective):
    """
    Print a concise table of density parameters from an Objective.

    Args:
        objective : refnx Objective

    Returns:
        pandas DataFrame with columns Material, Density (g/cm³), Uncertainty, Varying
    """
    rows = []
    for param in objective.parameters.flattened():
        if get_param_type(param.name) == 'density':
            material = param.name.replace('_density', '')
            rows.append({
                'Material':          material,
                'Density (g/cm³)':   param.value,
                'Uncertainty':       getattr(param, 'stderr', None),
                'Varying':           getattr(param, 'vary', False),
            })

    if not rows:
        print("No density parameters found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('Density (g/cm³)', ascending=False)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------

def get_parameter_info(objective, parameter_pattern=None):
    """
    Return a DataFrame of parameter values and bounds for an Objective.

    Args:
        objective         : refnx Objective
        parameter_pattern : optional regex string to filter by name

    Returns:
        DataFrame with columns: name, value, stderr, vary,
                                bound_low, bound_high, near_bounds, bound_pct
    """
    rows = []
    for param in objective.parameters.flattened():
        if parameter_pattern and not re.search(parameter_pattern, param.name, re.IGNORECASE):
            continue

        bl = bh = None
        try:
            b = getattr(param, 'bounds', None)
            if b is not None:
                if hasattr(b, 'lb'):
                    bl, bh = b.lb, b.ub
                elif isinstance(b, tuple) and len(b) == 2:
                    bl, bh = b
        except Exception:
            pass

        near = False
        if bl is not None and bh is not None and bh > bl:
            thresh = 0.02 * (bh - bl)
            near = (abs(param.value - bl) < thresh or
                    abs(bh - param.value) < thresh)

        rows.append(dict(name=param.name, value=param.value,
                         stderr=getattr(param, 'stderr', None),
                         vary=getattr(param, 'vary', None),
                         bound_low=bl, bound_high=bh,
                         near_bounds=near))

    if not rows:
        return pd.DataFrame(columns=['name', 'value', 'stderr', 'vary',
                                     'bound_low', 'bound_high',
                                     'near_bounds', 'bound_pct'])

    df = pd.DataFrame(rows)
    df['bound_pct'] = None
    mask = (df['bound_high'].notnull() & df['bound_low'].notnull() &
            (df['bound_high'] > df['bound_low']))
    df.loc[mask, 'bound_pct'] = (
        (df.loc[mask, 'value'] - df.loc[mask, 'bound_low']) /
        (df.loc[mask, 'bound_high'] - df.loc[mask, 'bound_low']) * 100
    )
    return df


def update_parameter_bounds(objective, parameter_name, new_bounds, verbose=True):
    """
    Update bounds on a single named parameter in an Objective.

    Args:
        objective      : refnx Objective (modified in-place)
        parameter_name : exact parameter name string
        new_bounds     : (lower, upper) tuple
        verbose        : print confirmation

    Returns:
        True if the parameter was found and updated, False otherwise.
    """
    if not isinstance(new_bounds, tuple) or len(new_bounds) != 2:
        if verbose:
            print("new_bounds must be a (lower, upper) tuple.")
        return False
    lo, hi = new_bounds
    if lo >= hi:
        if verbose:
            print("Lower bound must be less than upper bound.")
        return False

    for param in objective.parameters.flattened():
        if param.name == parameter_name:
            old = None
            try:
                b = getattr(param, 'bounds', None)
                if b is not None:
                    old = (b.lb, b.ub) if hasattr(b, 'lb') else b
            except Exception:
                pass
            param.bounds = new_bounds
            if verbose:
                print(f"Updated {parameter_name}: {old} → {new_bounds}")
            return True

    if verbose:
        print(f"Parameter '{parameter_name}' not found.")
    return False


def update_multiple_parameter_bounds(objective, parameters_dict, verbose=True):
    """
    Update bounds for several parameters at once.

    Args:
        objective       : refnx Objective
        parameters_dict : {name: (lower, upper), ...}
        verbose         : print per-parameter confirmations

    Returns:
        dict {name: bool} indicating success per parameter
    """
    results = {name: update_parameter_bounds(objective, name, bounds, verbose)
               for name, bounds in parameters_dict.items()}
    if verbose:
        n = sum(results.values())
        print(f"Updated {n}/{len(parameters_dict)} parameters.")
    return results


def expand_sld_parameter_bounds(objective, parameter_names=None,
                                 expansion_factor=1.5, min_bound=None,
                                 filter_pattern='sld', verbose=True):
    """
    Expand bounds on SLD parameters (or any pattern match) by a scale factor.

    Args:
        objective         : refnx Objective
        parameter_names   : explicit list of names; if None, use filter_pattern
        expansion_factor  : new range = old range × this factor
        min_bound         : floor for lower bound (applied to 'isld' params)
        filter_pattern    : regex applied when parameter_names is None
        verbose           : print info

    Returns:
        dict {name: bool} – same as update_multiple_parameter_bounds
    """
    if parameter_names is None:
        parameter_names = [p.name for p in objective.parameters.flattened()
                           if re.search(filter_pattern, p.name, re.IGNORECASE)]
    if verbose:
        print(f"Found {len(parameter_names)} parameters to expand.")

    to_update = {}
    for name in parameter_names:
        param = next((p for p in objective.parameters.flattened()
                      if p.name == name), None)
        if param is None:
            continue
        try:
            b = getattr(param, 'bounds', None)
            if b is None:
                continue
            lo = b.lb if hasattr(b, 'lb') else b[0]
            hi = b.ub if hasattr(b, 'ub') else b[1]
        except Exception:
            continue

        mid = param.value if lo <= param.value <= hi else (lo + hi) / 2
        new_range = (hi - lo) * expansion_factor
        new_lo = mid - new_range / 2
        new_hi = mid + new_range / 2
        if min_bound is not None and 'isld' in name.lower():
            new_lo = max(new_lo, min_bound)
        to_update[name] = (new_lo, new_hi)

    return update_multiple_parameter_bounds(objective, to_update, verbose=verbose)


def update_objective(objectives_dict, energy, material, updates):
    """
    Update parameters for a material at a specific energy in an objectives dict.

    Args:
        objectives_dict : {energy: Objective}
        energy          : key into the dict
        material        : material name (matched as substring in parameter names)
        updates         : dict with any of:
                            thickness, roughness, sld_real, sld_imag (values)
                            thickness_bounds, roughness_bounds,
                            sld_real_bounds, sld_imag_bounds,
                            density_bounds  (lower, upper, vary) tuples

    Returns:
        updated copy of objectives_dict
    """
    updated = deepcopy(objectives_dict)
    if energy not in updated:
        print(f"Warning: energy {energy} not in objectives_dict.")
        return updated

    objective = updated[energy]
    made = []

    for param in objective.parameters.flattened():
        pn = param.name.lower()
        if f"{material.lower()} - " not in pn:
            continue

        def _apply_value(key, label):
            if key in updates:
                old = param.value
                param.value = updates[key]
                made.append(f"{label} value: {old} → {param.value}")

        def _apply_bounds(key, label):
            if key in updates:
                t = updates[key]
                if len(t) == 1:
                    param.setp(vary=t[0])
                    made.append(f"{label} vary → {t[0]}")
                elif len(t) == 3:
                    lo, hi, vary = t
                    param.setp(bounds=(lo, hi), vary=vary)
                    made.append(f"{label} bounds → ({lo}, {hi}), vary={vary}")

        if 'thick' in pn:
            _apply_value('thickness', 'thickness')
            _apply_bounds('thickness_bounds', 'thickness')
        elif 'rough' in pn:
            _apply_value('roughness', 'roughness')
            _apply_bounds('roughness_bounds', 'roughness')
        elif 'isld' in pn:
            _apply_value('sld_imag', 'sld_imag')
            _apply_bounds('sld_imag_bounds', 'sld_imag')
        elif 'sld' in pn:
            _apply_value('sld_real', 'sld_real')
            _apply_bounds('sld_real_bounds', 'sld_real')
        elif '_density' in pn:
            _apply_value('density', 'density')
            _apply_bounds('density_bounds', 'density')

    if made:
        print(f"Updated {material} at {energy} eV:")
        for m in made:
            print(f"  {m}")
    else:
        print(f"No matching parameters for {material} at {energy} eV.")

    return updated


def update_objective_with_plotting(objectives_dict, energy, material, updates,
                                    plot=False, figsize=(12, 8),
                                    profile_shift=-20, save_plot=False,
                                    plot_filename=None, structures_dict=None):
    """
    update_objective with optional before/after comparison plot.

    Additional args vs update_objective:
        plot           : generate plots after update
        figsize        : (width, height)
        profile_shift  : depth shift for SLD profile
        save_plot      : save figure to file
        plot_filename  : filename for saved figure (auto-generated if None)
        structures_dict: {energy: Structure} for SLD profile plots

    Returns:
        updated objectives_dict
    """
    updated = update_objective(objectives_dict, energy, material, updates)

    if plot:
        _create_comparison_plot(updated, energy, material, figsize,
                                profile_shift, save_plot, plot_filename,
                                structures_dict)
    return updated


# ---------------------------------------------------------------------------
# SLD / optical utilities
# ---------------------------------------------------------------------------

def DeltaBetatoSLD(DeltaBeta):
    """Convert Delta/Beta optical constants to SLD array."""
    wl = energy_to_wavelength(DeltaBeta[:, 0])
    SLD_out = np.zeros([len(DeltaBeta[:, 0]), 4])
    SLD_out[:, 0] = DeltaBeta[:, 0]
    SLD_out[:, 3] = wl
    SLD_out[:, 1] = 2 * np.pi * DeltaBeta[:, 1] / np.power(wl, 2) * 1e6
    SLD_out[:, 2] = 2 * np.pi * DeltaBeta[:, 2] / np.power(wl, 2) * 1e6
    return SLD_out


def SLDinterp(Energy, SLDarray):
    """Interpolate real and imaginary SLD at a given energy."""
    real = np.interp(Energy, SLDarray[:, 0], SLDarray[:, 1])
    imag = np.interp(Energy, SLDarray[:, 0], SLDarray[:, 2]) * 1j
    return real, imag


def energy_to_wavelength(energy_ev):
    """Convert photon energy in eV to wavelength in nm."""
    return HC_EV_NM / energy_ev


def interpolate_sld_from_file(file_path, target_energy):
    """
    Interpolate SLD at a target energy from a three-column file
    (energy_eV, real, imag).

    Returns:
        (sld_real, sld_imag) or None on error
    """
    try:
        data = pd.read_csv(file_path, sep=r'\s+', header=None,
                           names=['energy_eV', 'real', 'imag'])
    except Exception:
        try:
            data = pd.read_csv(file_path, header=None,
                               names=['energy_eV', 'real', 'imag'])
        except Exception as exc:
            print(f"Could not read {file_path}: {exc}")
            return None

    data = data.sort_values('energy_eV')
    lo, hi = data['energy_eV'].min(), data['energy_eV'].max()
    if not (lo <= target_energy <= hi):
        print(f"Warning: {target_energy} eV outside file range [{lo}, {hi}].")

    r_fn = interp1d(data['energy_eV'], data['real'], kind='cubic',
                    bounds_error=False, fill_value='extrapolate')
    i_fn = interp1d(data['energy_eV'], data['imag'], kind='cubic',
                    bounds_error=False, fill_value='extrapolate')
    return float(r_fn(target_energy)), float(i_fn(target_energy))


def generate_sld_array_from_material(formula, density, energy_list, probe="x-ray"):
    """
    Compute SLD vs energy for a material using refnx MaterialSLD.

    Args:
        formula     : chemical formula string
        density     : density in g/cm³
        energy_list : energies in eV
        probe       : "x-ray" or "neutron"

    Returns:
        ndarray, shape (n, 3) – columns [Energy_eV, Real_SLD, Imag_SLD]
    """
    rows = []
    for ev in energy_list:
        wl = 12398.0 / ev
        mat = MaterialSLD(formula, density=density, probe=probe, wavelength=wl)
        c = complex(mat)
        rows.append([ev, c.real, c.imag])
    return np.array(rows)


# ---------------------------------------------------------------------------
# SLD profile helper
# ---------------------------------------------------------------------------

def get_sld_profile(structure):
    """
    Extract depth and SLD arrays from a refnx Structure.

    Returns:
        (z_real, sld_real, z_imag, sld_imag)
    """
    try:
        z, sld = structure.sld_profile()
        if hasattr(sld, 'real') and hasattr(sld, 'imag'):
            return z, sld.real, z, sld.imag
        return z, sld, z, np.zeros_like(z)
    except Exception:
        try:
            z = np.linspace(-10, 300, 1000)
            sld = structure.sld_profile(z)
            if hasattr(sld, 'real'):
                return z, sld.real, z, sld.imag
            return z, sld, z, np.zeros_like(z)
        except Exception:
            z = np.linspace(0, 300, 1000)
            return z, np.zeros_like(z), z, np.zeros_like(z)


# alias used by modelcomparisonplot
profileflip = get_sld_profile


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

def batch_fit_selected_models_v2(objectives_dict, structures_dict,
                                  energy_list=None,
                                  method='differential_evolution',
                                  workers=-1, popsize=15,
                                  steps=1000, burn=500,
                                  nthin=1, nwalkers=100,
                                  sampler='emcee', sampler_kws=None,
                                  save_dir=None,
                                  save_objectives=False,
                                  save_results=False,
                                  preserve_originals=True,
                                  verbose=True,
                                  model_name=None):
    """
    Fit a collection of Objectives (one per energy) in sequence.

    All objectives are included in the returned 'fitted_objectives' dict:
    energies in energy_list receive fitted objectives; others are passed
    through unchanged.

    Args:
        objectives_dict   : {energy: Objective}
        structures_dict   : {energy: Structure}
        energy_list       : energies to fit (None = all available)
        method / workers / popsize / steps / burn / nthin / nwalkers :
                            passed directly to run_fitting
        sampler           : 'emcee' (default), 'pymc', or 'dynesty'
        sampler_kws       : extra kwargs forwarded to the sampler (see run_fitting)
        save_dir          : directory for per-energy pickle files
        save_objectives   : save each objective
        save_results      : save each results dict
        preserve_originals: keep deep copies of input objectives/structures
        verbose           : progress printing
        model_name        : label applied to all fits (default "Model1")

    Returns:
        dict with keys:
            fitted_objectives, individual_results, summary_stats,
            fitted_energies, non_fitted_energies,
            original_objectives (if preserve_originals),
            original_structures (if preserve_originals)
    """
    print("=" * 60)
    print("BATCH FITTING")
    print("=" * 60)

    fitted_objectives = deepcopy(objectives_dict)

    # determine which energies to process
    if energy_list is None:
        to_fit = sorted(set(objectives_dict) & set(structures_dict))
    else:
        to_fit = []
        for e in energy_list:
            if e in objectives_dict and e in structures_dict:
                to_fit.append(e)
            else:
                print(f"Warning: {e} eV not in both dicts – skipping.")

    if not to_fit:
        print("No valid energies found.")
        return None

    all_energies    = set(objectives_dict)
    fitted_set      = set(to_fit)
    non_fitted_set  = all_energies - fitted_set

    print(f"Total: {len(all_energies)}  |  To fit: {len(fitted_set)}  |  Pass-through: {len(non_fitted_set)}")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    original_objectives = {}
    original_structures = {}
    individual_results  = {}
    successful = failed = 0
    chi_init_total = chi_final_total = 0.0

    for i, energy in enumerate(to_fit):
        print(f"\n--- {i+1}/{len(to_fit)}: {energy} eV ---")
        try:
            orig_obj  = objectives_dict[energy]
            structure = structures_dict[energy]

            if preserve_originals:
                original_objectives[energy] = deepcopy(orig_obj)
                original_structures[energy] = deepcopy(structure)

            working = deepcopy(orig_obj)
            ename   = model_name if model_name else "Model1"

            chi_init = working.chisqr()
            chi_init_total += chi_init
            if verbose:
                print(f"  Initial χ²: {chi_init:.6g}")

            results = run_fitting(
                objective=working,
                method=method,
                workers=workers,
                popsize=popsize,
                steps=steps,
                burn=burn,
                nthin=nthin,
                nwalkers=nwalkers,
                sampler=sampler,
                sampler_kws=dict(sampler_kws or {}),
                save_dir=save_dir,
                save_objective=save_objectives,
                save_results=save_results,
                structure=structure,
                model_name=ename,
                verbose=verbose,
            )

            fitted_objectives[energy] = working
            individual_results[energy] = results

            chi_final = working.chisqr()
            chi_final_total += chi_final
            if chi_init > 0:
                pct = (chi_init - chi_final) / chi_init * 100
                print(f"  Final χ²: {chi_final:.6g}  (improvement: {pct:.1f}%)")
            else:
                print(f"  Final χ²: {chi_final:.6g}")
            successful += 1

        except Exception as exc:
            print(f"  ERROR at {energy} eV: {exc}")
            failed += 1

    # summary
    print(f"\n{'='*60}")
    print(f"Successful: {successful}  |  Failed: {failed}  |  Pass-through: {len(non_fitted_set)}")
    if successful > 0 and chi_init_total > 0:
        overall = (chi_init_total - chi_final_total) / chi_init_total * 100
        print(f"Overall χ² improvement: {overall:.1f}%")

    summary = {
        'total_models':              len(to_fit),
        'total_objectives_in_output': len(fitted_objectives),
        'successful_fits':           successful,
        'failed_fits':               failed,
        'non_fitted_count':          len(non_fitted_set),
        'initial_chi_squared_total': chi_init_total,
        'final_chi_squared_total':   chi_final_total,
        'overall_improvement_percent': (
            (chi_init_total - chi_final_total) / chi_init_total * 100
            if chi_init_total > 0 else 0),
        'save_directory': save_dir,
    }

    ret = {
        'fitted_objectives':  fitted_objectives,
        'individual_results': individual_results,
        'summary_stats':      summary,
        'fitted_energies':    sorted(fitted_set),
        'non_fitted_energies': sorted(non_fitted_set),
    }
    if preserve_originals:
        ret['original_objectives'] = original_objectives
        ret['original_structures'] = original_structures

    return ret


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_fitting_files(results, save_dir, model_name,
                       save_objective, save_results, structure):
    """
    Persist fitting results to <save_dir>.

    Writes:
        <model_name>_objective.pkl          (if save_objective)
        <model_name>_results_structure.pkl  (if save_results)
        <model_name>_mcmc_samples.npy       (if save_results and MCMC present)
    """
    os.makedirs(save_dir, exist_ok=True)

    if save_objective:
        path = os.path.join(save_dir, f"{model_name}_objective.pkl")
        try:
            with open(path, 'wb') as fh:
                pickle.dump(results['objective'], fh)
            print(f"Saved objective → {path}")
        except Exception as exc:
            print(f"Error saving objective: {exc}")

    if save_results:
        copy = results.copy()
        copy['objective'] = None
        combined = {
            'results': copy,
            'structure': structure,
            'objective': results['objective'] if save_objective else None,
            'model_name': model_name,
        }
        path = os.path.join(save_dir, f"{model_name}_results_structure.pkl")
        try:
            with open(path, 'wb') as fh:
                pickle.dump(combined, fh)
            print(f"Saved results+structure → {path}")
        except Exception as exc:
            print(f"Error saving results: {exc}")

        if results.get('mcmc_samples') is not None:
            path = os.path.join(save_dir, f"{model_name}_mcmc_samples.npy")
            try:
                np.save(path, results['mcmc_samples'])
                print(f"Saved MCMC samples → {path}")
            except Exception as exc:
                print(f"Error saving MCMC samples: {exc}")


def load_fitting_file(filename):
    """
    Load a pickle file saved by save_fitting_files or run_fitting.

    Returns a normalised dict with keys:
        objective, structure, results, mcmc_samples, mcmc_stats,
        model_name, has_objective, has_structure,
        has_mcmc_samples, file_type
    """
    result = dict(objective=None, structure=None, results=None,
                  mcmc_samples=None, mcmc_stats=None,
                  model_name='unknown', has_objective=False,
                  has_structure=False, has_mcmc_samples=False,
                  file_type='unknown')
    try:
        with open(filename, 'rb') as fh:
            data = pickle.load(fh)
    except Exception as exc:
        print(f"Error loading {filename}: {exc}")
        return None

    if isinstance(data, dict) and 'results' in data and 'structure' in data:
        result['file_type']  = 'combined'
        result['structure']  = data.get('structure')
        result['objective']  = data.get('objective')
        result['model_name'] = data.get('model_name', 'unknown')
        if isinstance(data.get('results'), dict):
            result['results']      = data['results']
            result['mcmc_samples'] = data['results'].get('mcmc_samples')
            result['mcmc_stats']   = data['results'].get('mcmc_stats')

    elif hasattr(data, 'model') and hasattr(data, 'chisqr'):
        result['file_type']  = 'objective'
        result['objective']  = data
        result['model_name'] = getattr(data.model, 'name', 'unnamed')

    elif isinstance(data, dict) and 'mcmc_samples' in data:
        result['file_type']    = 'results'
        result['results']      = data
        result['mcmc_samples'] = data.get('mcmc_samples')
        result['mcmc_stats']   = data.get('mcmc_stats')
        if data.get('objective') is not None:
            result['objective']  = data['objective']
            result['model_name'] = getattr(data['objective'].model, 'name', 'unnamed')
        result['structure'] = data.get('structure')

    elif isinstance(data, np.ndarray):
        result['file_type']    = 'mcmc_samples'
        result['mcmc_samples'] = data

    else:
        print("Unknown file format.")
        return {'raw_data': data, 'file_type': 'unknown'}

    result['has_objective']    = result['objective']    is not None
    result['has_structure']    = result['structure']    is not None
    result['has_mcmc_samples'] = result['mcmc_samples'] is not None

    print(f"Loaded {result['file_type']} | model: {result['model_name']} | "
          f"obj={result['has_objective']} struct={result['has_structure']} "
          f"mcmc={result['has_mcmc_samples']}")
    return result


def extract_structure(loaded_data):
    """Return structure from a load_fitting_file result dict."""
    if loaded_data is None or not loaded_data.get('has_structure', False):
        print("No structure in loaded data.")
        return None
    return loaded_data.get('structure')


def extract_objective(loaded_data):
    """Return objective from a load_fitting_file result dict."""
    if loaded_data is None or not loaded_data.get('has_objective', False):
        print("No objective in loaded data.")
        return None
    return loaded_data.get('objective')


def extract_results(loaded_data):
    """Return results dict from a load_fitting_file result dict."""
    if loaded_data is None:
        return None
    return loaded_data.get('results')


def extract_mcmc_samples(loaded_data):
    """Return MCMC samples array from a load_fitting_file result dict."""
    if loaded_data is None or not loaded_data.get('has_mcmc_samples', False):
        print("No MCMC samples in loaded data.")
        return None
    return loaded_data.get('mcmc_samples')


def get_parameter_summary(loaded_data):
    """
    Build a DataFrame of MCMC parameter statistics from a loaded file.

    Returns DataFrame with columns: parameter, median, mean, std,
                                    upper_error, lower_error, value, stderr
    """
    results = loaded_data.get('results') if loaded_data else None
    if results is None or results.get('mcmc_stats') is None:
        print("No MCMC statistics available.")
        return None

    rows = []
    for name, s in results['mcmc_stats'].items():
        if 'percentiles' in s and s.get('median') is not None:
            p = s['percentiles']
            rows.append(dict(
                parameter=name,
                median=s['median'], mean=s.get('mean'), std=s.get('std'),
                upper_error=p[2] - s['median'],
                lower_error=s['median'] - p[0],
                value=s.get('value'), stderr=s.get('stderr'),
            ))
    return pd.DataFrame(rows)


def save_batch_fit_results(batch_results, filename, save_dir=None,
                            include_objectives=True, include_mcmc_samples=True):
    """
    Save the dict returned by batch_fit_selected_models_v2 to a pickle file.

    Returns:
        Full path to the saved file.
    """
    required = ['fitted_objectives', 'individual_results', 'summary_stats']
    missing  = [k for k in required if k not in batch_results]
    if missing:
        raise ValueError(f"batch_results missing keys: {missing}")

    fname = filename if filename.endswith('.pkl') else f"{filename}.pkl"
    path  = os.path.join(save_dir, fname) if save_dir else fname
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    save = {
        'file_type':    'batch_fit_results_v2',
        'summary_stats': batch_results['summary_stats'].copy(),
        'metadata':      {'include_objectives': include_objectives,
                          'include_mcmc_samples': include_mcmc_samples},
        'fitted_objectives':  batch_results['fitted_objectives'] if include_objectives else None,
        'original_objectives': batch_results.get('original_objectives') if include_objectives else None,
        'original_structures': batch_results.get('original_structures'),
    }

    if include_mcmc_samples:
        save['individual_results'] = batch_results['individual_results']
    else:
        stripped = {}
        for e, r in batch_results['individual_results'].items():
            rc = r.copy() if isinstance(r, dict) else r
            if isinstance(rc, dict):
                rc['mcmc_samples'] = None
            stripped[e] = rc
        save['individual_results'] = stripped

    with open(path, 'wb') as fh:
        pickle.dump(save, fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"Saved batch results → {path}  ({size_mb:.2f} MB)")
    return path


def load_batch_fit_results(file_path, verbose=True):
    """
    Load a file saved by save_batch_fit_results.

    Returns:
        dict matching the structure of batch_fit_selected_models_v2 output.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, 'rb') as fh:
        data = pickle.load(fh)

    if verbose:
        print(f"Loaded: {file_path}")
        if data.get('file_type') != 'batch_fit_results_v2':
            print("Warning: unexpected file format.")

    result = {
        'summary_stats':      data.get('summary_stats', {}),
        'individual_results': data.get('individual_results', {}),
    }
    for key in ('fitted_objectives', 'original_objectives',
                'original_structures'):
        if data.get(key) is not None:
            result[key] = data[key]
            if verbose:
                print(f"  {key}: {len(data[key])} entries")

    if verbose:
        n = len(result['individual_results'])
        mcmc_n = sum(1 for r in result['individual_results'].values()
                     if isinstance(r, dict) and r.get('mcmc_samples') is not None)
        print(f"  Energies: {sorted(result['individual_results'].keys())}")
        print(f"  MCMC available for {mcmc_n}/{n} energies")

    return result


def get_batch_results_info(file_path):
    """
    Summarise a batch results file without returning the full data.

    Returns:
        dict with keys: file_path, summary_stats, n_energies,
                        has_objectives, has_mcmc_samples, energies
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, 'rb') as fh:
        data = pickle.load(fh)

    ind = data.get('individual_results', {})
    return {
        'file_path':       file_path,
        'summary_stats':   data.get('summary_stats', {}),
        'n_energies':      len(ind),
        'has_objectives':  data.get('fitted_objectives') is not None,
        'has_mcmc_samples': any(
            isinstance(r, dict) and r.get('mcmc_samples') is not None
            for r in ind.values()),
        'energies':        sorted(ind.keys()),
    }


def save_material_sld(objectives_dict, material_name, filename, save_dir=None,
                      include_header=True, plot=False, figsize=(10, 6)):
    """
    Extract SLD vs energy for a material from fitted objectives and save to CSV.

    Args:
        objectives_dict : {energy: Objective}
        material_name   : material to extract (e.g. 'MOX')
        filename        : output CSV filename (with or without .csv extension)
        save_dir        : optional directory (None = use filename as-is)
        include_header  : write column header row
        plot            : show a plot of the SLD vs energy after saving
        figsize         : figure size if plot=True

    Returns:
        ndarray, shape (n, 3) – [Energy_eV, Real_SLD, Imag_SLD]
    """
    rows = []
    for energy in sorted(objectives_dict):
        obj = objectives_dict[energy]
        real_val = imag_val = None
        for param in obj.parameters.flattened():
            pn = param.name.lower()
            if f'{material_name.lower()} - ' not in pn:
                continue
            if 'isld' in pn:
                imag_val = param.value
            elif 'sld' in pn:
                real_val = param.value

        if real_val is not None and imag_val is not None:
            rows.append([energy, real_val, imag_val])
        else:
            print(f'Warning: could not find SLD params for {material_name} at {energy} eV')

    if not rows:
        print(f'No SLD data found for {material_name}.')
        return None

    arr = np.array(rows, dtype=float)

    # save
    fname = filename if filename.endswith('.csv') else f'{filename}.csv'
    path  = os.path.join(save_dir, fname) if save_dir else fname
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    header = f'{material_name} SLD data\nEnergy_eV,Real_SLD,Imag_SLD' if include_header else ''
    np.savetxt(path, arr, delimiter=',', header=header, comments='')
    print(f'Saved {material_name} SLD ({len(arr)} energies) → {path}')

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(arr[:, 0], arr[:, 1], 'o-', color='C0')
        axes[0].set_xlabel('Energy (eV)')
        axes[0].set_ylabel(r'Real SLD ($10^{-6}$ Å$^{-2}$)')
        axes[0].set_title(f'{material_name} – Real SLD')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(arr[:, 0], arr[:, 2], 'o-', color='C1')
        axes[1].set_xlabel('Energy (eV)')
        axes[1].set_ylabel(r'Imag SLD ($10^{-6}$ Å$^{-2}$)')
        axes[1].set_title(f'{material_name} – Imaginary SLD')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return arr


def load_material_sld_array(file_path, has_header=True, verbose=True):
    """
    Load a three-column (Energy, Real_SLD, Imag_SLD) CSV saved by save_material_sld.

    Returns:
        ndarray, shape (n, 3)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    arr = np.loadtxt(file_path, delimiter=',', skiprows=(2 if has_header else 0))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns, got {arr.shape[1]}.")

    if verbose:
        print(f"Loaded SLD data from {file_path}  shape={arr.shape}")
        print(f"  Energy: {arr[:,0].min():.1f}–{arr[:,0].max():.1f} eV")
    return arr


def save_reflectivity_data(obj, save_path, model_name=None):
    """
    Save experimental and simulated reflectivity from an Objective.

    Writes:
        <save_path>_data.txt
        <save_path>_simulated.txt
        <save_path>_metadata.pkl

    Returns:
        metadata dict
    """
    x = obj.data.x
    y = obj.data.y
    ye = (obj.data.y_err
          if hasattr(obj.data, 'y_err') and obj.data.y_err is not None
          else np.zeros_like(y))
    ys = obj.model(x)

    np.savetxt(f"{save_path}_data.txt", np.column_stack([x, y, ye]),
               header='Q (1/Angstrom)    R    R_err', fmt='%.8e')
    np.savetxt(f"{save_path}_simulated.txt", np.column_stack([x, ys]),
               header='Q (1/Angstrom)    R_sim', fmt='%.8e')

    meta = dict(model_name=model_name, n_points=len(x),
                q_min=float(x.min()), q_max=float(x.max()),
                data_file=f"{save_path}_data.txt",
                sim_file=f"{save_path}_simulated.txt")
    with open(f"{save_path}_metadata.pkl", 'wb') as fh:
        pickle.dump(meta, fh)

    print(f"Saved reflectivity data to {save_path}_{{data,simulated,metadata}}.*")
    return meta


def load_reflectivity_data(save_path):
    """
    Load data saved by save_reflectivity_data.

    Returns:
        dict with keys: data (Q,R,R_err), simulated (Q,R_sim), metadata
    """
    return {
        'data':      np.loadtxt(f"{save_path}_data.txt"),
        'simulated': np.loadtxt(f"{save_path}_simulated.txt"),
        'metadata':  pickle.load(open(f"{save_path}_metadata.pkl", 'rb')),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reflectivity_data(loaded_data, figsize=(10, 6), log_y=True,
                            save_path=None):
    """
    Plot experimental and simulated reflectivity from load_reflectivity_data output.

    Returns:
        (fig, ax)
    """
    d  = loaded_data['data']
    s  = loaded_data['simulated']
    m  = loaded_data['metadata']

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(d[:, 0], d[:, 1], yerr=d[:, 2],
                fmt='o', markersize=4, capsize=2, alpha=0.7,
                label='Experimental', zorder=1)
    ax.plot(s[:, 0], s[:, 1], '-', linewidth=2, label='Simulated', zorder=2)
    ax.set_xlabel('Q (1/Å)', fontsize=12)
    ax.set_ylabel('Reflectivity', fontsize=12)
    if log_y:
        ax.set_yscale('log')
    title = f"Reflectivity: {m['model_name']}" if m.get('model_name') else "Reflectivity"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure → {save_path}")
    return fig, ax


def plot_material_sld(sld_df, material_name, plot_type='both', figsize=(10, 6),
                      save_path=None):
    """
    Plot Real and/or Imaginary SLD vs energy for a material from a DataFrame.

    sld_df must have columns: Energy, Real_SLD, Imag_SLD (or similar).
    """
    fig, ax = plt.subplots(figsize=figsize)
    if plot_type in ('real', 'both'):
        ax.plot(sld_df.iloc[:, 0], sld_df.iloc[:, 1], label='Real SLD')
    if plot_type in ('imag', 'both'):
        ax.plot(sld_df.iloc[:, 0], sld_df.iloc[:, 2], '--', label='Imag SLD')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('SLD (×10⁻⁶ Å⁻²)')
    ax.set_title(f'SLD: {material_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_sld_comparison(sld_df, materials_comparison, figsize=(14, 10),
                         save_path=None):
    """
    Overlay SLD profiles for multiple materials.

    materials_comparison : list of material name strings to plot from sld_df
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for mat in materials_comparison:
        col_r = f"{mat}_real"
        col_i = f"{mat}_imag"
        if col_r in sld_df.columns:
            axes[0].plot(sld_df.iloc[:, 0], sld_df[col_r], label=mat)
        if col_i in sld_df.columns:
            axes[1].plot(sld_df.iloc[:, 0], sld_df[col_i], '--', label=mat)
    for i, lbl in enumerate(('Real SLD', 'Imag SLD')):
        axes[i].set_xlabel('Energy (eV)')
        axes[i].set_ylabel(f'{lbl} (×10⁻⁶ Å⁻²)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, axes


def extract_sld_from_objectives(objectives_dict, materials_filter=None,
                                 verbose=True):
    """
    Extract fitted SLD values across energies from an objectives dict.

    Args:
        objectives_dict  : {energy: Objective}
        materials_filter : list of material name substrings to include (None = all)
        verbose          : print progress

    Returns:
        DataFrame with columns: energy, parameter, value, stderr, vary
    """
    rows = []
    for energy, obj in sorted(objectives_dict.items()):
        for param in obj.parameters.flattened():
            ptype = get_param_type(param.name)
            if ptype not in ('sld_real', 'sld_imag', 'density'):
                continue
            if materials_filter and not any(
                    m.lower() in param.name.lower() for m in materials_filter):
                continue
            rows.append(dict(energy=energy, parameter=param.name,
                             value=param.value,
                             stderr=getattr(param, 'stderr', None),
                             vary=getattr(param, 'vary', False)))
        if verbose:
            print(f"Processed {energy} eV ({len(rows)} params so far)")
    return pd.DataFrame(rows)


def modelcomparisonplot(obj_list, structure_list, shade_start=None,
                         fig_size_w=16, colors=None, profile_shift=-10,
                         xlim=None, zoom_xlim=None, zoom_ylim=None):
    """
    Three-row comparison plot: full reflectivity, zoomed reflectivity, SLD profile.

    Args:
        obj_list       : list of Objective objects
        structure_list : list of Structure objects (parallel to obj_list)
        shade_start    : list of starting positions for layer shading (or None)
        fig_size_w     : figure width
        colors         : list of shading colours
        profile_shift  : depth offset applied to SLD profiles
        xlim           : [min, max] for SLD x-axis (None = auto)
        zoom_xlim      : (min, max) for zoomed reflectivity (default (0, 0.05))
        zoom_ylim      : (min, max) for zoomed y-axis (default auto from scale)

    Returns:
        (fig, axes)
    """
    n = len(obj_list)
    if colors is None:
        colors = ['silver', 'grey', 'blue', 'violet', 'orange',
                  'purple', 'red', 'green', 'yellow']
    if zoom_xlim is None:
        zoom_xlim = (0, 0.05)

    fig, axes = plt.subplots(3, n if n > 1 else 1, figsize=(fig_size_w, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    chi = np.array([o.chisqr() for o in obj_list])
    rel_chi = np.round(chi / chi[0], 2)

    for i in range(n):
        ax_r  = axes[0, i]
        ax_rz = axes[1, i]
        ax_s  = axes[2, i]

        data = obj_list[i].data
        q, r_obs = data.data[0], data.data[1]
        r_model = obj_list[i].model(q)

        # full reflectivity
        ax_r.plot(q, r_obs, 'o', markersize=3, label='Data')
        ax_r.plot(q, r_model, '-', label='Model')
        ax_r.set_yscale('log')
        ax_r.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_r.set_ylabel('Reflectivity')
        ax_r.text(0.5, 0.98, f'Rel. GF {rel_chi[i]}',
                  transform=ax_r.transAxes, ha='center', va='top', fontsize=10)
        ax_r.legend(fontsize=8)

        # zoom
        ax_rz.plot(q, r_obs, 'o', markersize=3, label='Data')
        ax_rz.plot(q, r_model, '-', label='Model')
        ax_rz.set_xlim(zoom_xlim)
        ax_rz.set_yscale('linear')
        mask = (q >= zoom_xlim[0]) & (q <= zoom_xlim[1])
        if np.any(mask):
            all_y = np.concatenate([r_obs[mask], r_model[mask]])
            if zoom_ylim is not None:
                ax_rz.set_ylim(zoom_ylim)
            else:
                ax_rz.set_ylim(all_y.min() * 0.9, all_y.max() * 1.1)
        ax_rz.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_rz.set_ylabel('Reflectivity (linear)')

        # SLD profile
        z, real_sld, _, imag_sld = get_sld_profile(structure_list[i])
        ax_s.plot(z + profile_shift, real_sld, color='blue',
                  label='Real SLD', zorder=2)
        ax_s.plot(z + profile_shift, imag_sld, linestyle='--',
                  color='blue', label='Imag SLD', zorder=2)
        if xlim is not None:
            ax_s.set_xlim(xlim)

        # layer shading
        slabs = structure_list[i].slabs()
        pvals = obj_list[i].parameters.pvals
        start = (shade_start[i] if (shade_start and len(shade_start) > i) else 0)
        thicknesses = [start]
        for j in range(1, len(slabs)):
            idx = (len(slabs) - j - 1) * 5 + 9
            if idx < len(pvals):
                thicknesses.append(thicknesses[-1] + pvals[idx])
            else:
                thicknesses.append(thicknesses[-1] + slabs[j]['thickness'])
        if thicknesses:
            ax_s.axvspan(0, thicknesses[0], color='silver', alpha=0.3, zorder=0)
        for j in range(len(thicknesses) - 1):
            ax_s.axvspan(thicknesses[j], thicknesses[j + 1],
                         color=colors[min(j, len(colors) - 1)],
                         alpha=0.2, zorder=1)
        ax_s.legend(fontsize=8)
        ax_s.set_xlabel(r'Distance from Si ($\AA$)')
        ax_s.set_ylabel(r'SLD $(10^{-6})$ $\AA^{-2}$')

    plt.tight_layout()
    return fig, axes


def create_interactive_fit_explorer(batch_results, figsize=(16, 12),
                                     default_material=None, fitted_only=False):
    """
    Interactive ipywidgets explorer for batch fit results.

    Requires ipywidgets and a Jupyter environment.

    Args:
        batch_results    : output from batch_fit_selected_models_v2
        figsize          : figure dimensions
        default_material : pre-select a material in the dropdown
        fitted_only      : if True, hide original (pre-fit) curves

    Returns:
        ipywidgets VBox widget
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets is required for the interactive explorer.")
        return None

    individual_results   = batch_results.get('individual_results', {})
    original_objectives  = batch_results.get('original_objectives', {})
    original_structures  = batch_results.get('original_structures', {})
    fitted_objectives    = batch_results.get('fitted_objectives', {})

    energies = sorted(individual_results.keys())
    if not energies:
        print("No results to display.")
        return None

    # collect material names from parameter names
    all_materials = set()
    for e in energies:
        obj = fitted_objectives.get(e)
        if obj:
            for p in obj.parameters.flattened():
                if ' - ' in p.name:
                    mat = p.name.split(' - ')[0].strip()
                    if mat.lower() not in ('air', 'si', 'sio2'):
                        all_materials.add(mat)
    materials = sorted(all_materials)

    material_dropdown  = widgets.Dropdown(
        options=materials,
        value=default_material if default_material in materials else (materials[0] if materials else None),
        description='Material:')
    parameter_radio    = widgets.RadioButtons(
        options=['SLD Real', 'SLD Imag', 'Thickness', 'Roughness'],
        description='Parameter:')
    energy_range       = widgets.SelectionRangeSlider(
        options=energies, index=(0, len(energies) - 1),
        description='Energy range:',
        layout=widgets.Layout(width='440px'))
    status_label       = widgets.HTML("<b>Click a point to compare fits.</b>")
    plot_output        = widgets.Output()

    def update_plots(*_):
        with plot_output:
            plot_output.clear_output(wait=True)
            mat    = material_dropdown.value
            param  = parameter_radio.value
            e_min, e_max = energy_range.value
            sel_e  = [e for e in energies if e_min <= e <= e_max]

            param_map = {'SLD Real': 'sld', 'SLD Imag': 'isld',
                         'Thickness': 'thick', 'Roughness': 'rough'}
            kw = param_map.get(param, 'sld')

            vals, errs, ens = [], [], []
            for e in sel_e:
                obj = fitted_objectives.get(e)
                if obj is None:
                    continue
                for p in obj.parameters.flattened():
                    if (f"{mat.lower()} - " in p.name.lower() and
                            kw in p.name.lower()):
                        vals.append(p.value)
                        errs.append(getattr(p, 'stderr', None) or 0)
                        ens.append(e)
                        break

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.errorbar(ens, vals, yerr=errs, fmt='o-', capsize=3)
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel(param)
            ax.set_title(f'{mat} – {param}')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    material_dropdown.observe(update_plots, names='value')
    parameter_radio.observe(update_plots, names='value')
    energy_range.observe(update_plots, names='value')

    controls = widgets.HBox([
        widgets.VBox([material_dropdown], layout=widgets.Layout(width='250px')),
        widgets.VBox([parameter_radio],   layout=widgets.Layout(width='220px')),
        widgets.VBox([energy_range],      layout=widgets.Layout(width='460px')),
    ])
    widget = widgets.VBox([controls, plot_output, status_label])
    update_plots()
    return widget


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_comparison_plot(objectives_dict, energy, material, figsize,
                             profile_shift, save_plot, plot_filename,
                             structures_dict):
    """Helper: reflectivity + SLD profile + parameter table for one energy."""
    obj = objectives_dict[energy]
    chi = obj.chisqr()

    has_struct = structures_dict and energy in structures_dict
    if has_struct:
        fig, ((ax_r, ax_s), (ax_p, ax_e)) = plt.subplots(2, 2, figsize=figsize)
        ax_e.axis('off')
    else:
        fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=figsize)
        ax_s = None

    data = obj.data
    ax_r.plot(data.data[0], data.data[1], 'o', markersize=4,
              color='black', alpha=0.7, label='Experimental')
    ax_r.plot(data.data[0], obj.model(data.data[0]), '-',
              linewidth=2, color='red', label='Simulated')
    ax_r.set_yscale('log')
    ax_r.set_xlabel(r'Q ($\AA^{-1}$)')
    ax_r.set_ylabel('Reflectivity')
    ax_r.set_title(f'{energy} eV  χ²={chi:.4f}')
    ax_r.legend()
    ax_r.grid(True, alpha=0.3)

    if ax_s is not None:
        structure = structures_dict[energy]
        try:
            z, rsl, _, isl = get_sld_profile(structure)
            ax_s.plot(z + profile_shift, rsl, 'b-',  label='Real SLD')
            ax_s.plot(z + profile_shift, isl, 'r--', label='Imag SLD')
            ax_s.set_xlabel('Depth (Å)')
            ax_s.set_ylabel('SLD (×10⁻⁶ Å⁻²)')
            ax_s.set_title(f'SLD Profile {energy} eV')
            ax_s.legend()
            ax_s.grid(True, alpha=0.3)
        except Exception as exc:
            ax_s.text(0.5, 0.5, f'SLD error:\n{exc}',
                      transform=ax_s.transAxes, ha='center', va='center')

    # parameter table
    tdata = []
    for p in obj.parameters.flattened():
        if f"{material.lower()} - " in p.name.lower():
            label = p.name.split(' - ')[-1] if ' - ' in p.name else p.name
            b = getattr(p, 'bounds', None)
            if b is not None:
                bstr = (f"({b.lb:.3f}, {b.ub:.3f})" if hasattr(b, 'lb')
                        else str(b))
            else:
                bstr = "None"
            tdata.append([label, f"{p.value:.4f}", bstr,
                           str(getattr(p, 'vary', False))])
    if tdata:
        ax_p.axis('tight')
        ax_p.axis('off')
        tbl = ax_p.table(cellText=tdata,
                         colLabels=['Parameter', 'Value', 'Bounds', 'Vary'],
                         cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.5)
        ax_p.set_title(f'{material} @ {energy} eV', pad=20)
    else:
        ax_p.text(0.5, 0.5, f'No {material} params', ha='center', va='center',
                  transform=ax_p.transAxes)

    plt.tight_layout()
    if save_plot:
        fname = plot_filename or f"{material}_{energy}eV_comparison.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Plot saved → {fname}")
    plt.show()
