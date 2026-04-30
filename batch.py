"""
batch.py
--------
RSoXR batch pipeline: loading data, building energy-dependent models from
SLD arrays, running batch fits, and extracting/exporting results.

Typical workflow
----------------
1. import_batch_reflectivity      – load .dat files → {energy: ReflectDataset}
2. generate_batch_models           – interpolate SLD arrays, build objectives
3. batch_fit_selected_models       – fit all (or selected) energies
4. extract_results_from_objectives – pull fitted parameters into a DataFrame
5. export_best_parameters          – save comparison to CSV
"""

import os
import re
import copy
import pickle

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from refnx.dataset import ReflectDataset

from Model_Setup import (
    create_reflectometry_model,
    create_model_and_objective,
    batch_fit_selected_models_v2,
    run_fitting,
)
from h5io import (save_batch_to_h5, save_sweep_results_to_h5,
                  load_h5_objectives, get_h5_info,
                  plot_parameter_vs_energy, plot_reflectivity,
                  save_nexafs_to_h5, update_nexafs_sld,
                  load_nexafs_from_h5, list_nexafs_spectra)
from parameter_sweep import (setup_parameter_sweep, run_parameter_sweep,
                              get_best_fit_with_uncertainty)
 

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def import_batch_reflectivity(folder_path, file_type='smoothed'):
    """
    Load a batch of reflectivity .dat files from a folder.

    Args:
        folder_path : path to the folder containing reflectivity files
        file_type   : 'raw'      → *raw.dat
                      'smoothed' → *smoothed.dat  (default)
                      'all'      → all .dat files

    Returns:
        (data_dict, energy_list)
            data_dict   – {energy: ReflectDataset}
            energy_list – sorted list of energies found
    """
    patterns = {
        'raw':      r'.*_([0-9.]+).*raw\.dat$',
        'smoothed': r'.*_([0-9.]+).*smoothed\.dat$',
        'all':      r'.*_([0-9.]+).*\.dat$',
    }
    pattern = patterns.get(file_type, patterns['smoothed'])

    folder_path = Path(folder_path)
    data_dict   = {}
    failed      = []

    for fp in folder_path.glob('*.dat'):
        m = re.match(pattern, fp.name)
        if m:
            try:
                energy = float(m.group(1))
                data_dict[energy] = ReflectDataset(str(fp))
                print(f"Loaded {fp.name}  ({energy} eV)")
            except Exception as exc:
                failed.append((fp.name, str(exc)))

    if failed:
        print("\nFiles matched but failed to load:")
        for name, err in failed:
            print(f"  {name}: {err}")

    energy_list = sorted(data_dict)
    print(f"\nLoaded {len(data_dict)} datasets  "
          f"({min(energy_list) if energy_list else 'N/A'} – "
          f"{max(energy_list) if energy_list else 'N/A'} eV)")
    return data_dict, energy_list


# ---------------------------------------------------------------------------
# Model building helpers
# ---------------------------------------------------------------------------

def generate_materials_from_sld_arrays(energy_list, material_sld_arrays,
                                        constant_materials=None):
    """
    Build per-energy materials lists by interpolating SLD arrays.

    Args:
        energy_list          : energies (eV) to generate materials for
        material_sld_arrays  : {name: ndarray} where each array has columns
                               [Energy_eV, Real_SLD, Imag_SLD]
        constant_materials   : {name: {'real': …, 'imag': …}} for materials
                               with energy-independent SLD (default: air = 0)

    Returns:
        {energy: materials_list}  – each materials_list is suitable for
        passing directly to create_reflectometry_model
    """
    if constant_materials is None:
        constant_materials = {'air': {'real': 0.0, 'imag': 0.0}}

    # Build interpolators
    interp_funcs = {}
    for name, arr in material_sld_arrays.items():
        try:
            if arr.shape[1] < 3:
                print(f"Warning: array for '{name}' needs ≥3 columns.")
                continue
            r_fn = interp1d(arr[:, 0], arr[:, 1],
                            bounds_error=False, fill_value='extrapolate')
            i_fn = interp1d(arr[:, 0], arr[:, 2],
                            bounds_error=False, fill_value='extrapolate')
            interp_funcs[name] = (r_fn, i_fn, arr[:, 0].min(), arr[:, 0].max())
            print(f"Interpolator for '{name}':  "
                  f"{arr[:, 0].min():.1f}–{arr[:, 0].max():.1f} eV  "
                  f"({len(arr)} points)")
        except Exception as exc:
            print(f"Error building interpolator for '{name}': {exc}")

    energy_materials = {}
    for energy in energy_list:
        mlist = []
        for name, (r_fn, i_fn, lo, hi) in interp_funcs.items():
            if not (lo <= energy <= hi):
                print(f"Warning: {energy} eV outside range for '{name}' "
                      f"({lo:.1f}–{hi:.1f} eV) – extrapolating.")
            mlist.append({
                'name': name,
                'real': float(r_fn(energy)),
                'imag': float(i_fn(energy)) * 1j,
            })
        for name, vals in constant_materials.items():
            mlist.append({'name': name, 'real': vals['real'], 'imag': vals['imag']})
        energy_materials[energy] = mlist

    print(f"\nGenerated material lists for {len(energy_list)} energies.")
    return energy_materials


def generate_layer_params_with_flexible_bounds(energy_materials, base_layer_params,
                                                sld_offset_bounds=None):
    """
    Create per-energy layer parameter dicts with SLD bounds centred on
    interpolated values.

    Explicit bounds already present in base_layer_params take priority.
    If no bounds are supplied anywhere the parameter is fixed.
    Imaginary SLD lower bounds are clamped to zero.

    Args:
        energy_materials   : output of generate_materials_from_sld_arrays
        base_layer_params  : {name: param_dict} template
        sld_offset_bounds  : {name: {'real': (min_off, max_off, vary),
                                     'imag': (min_off, max_off, vary)}}

    Returns:
        {energy: layer_params_dict}
    """
    if sld_offset_bounds is None:
        sld_offset_bounds = {}

    energy_layer_params = {}

    for energy, mlist in energy_materials.items():
        lp = copy.deepcopy(base_layer_params)
        sld_map = {m['name']: m for m in mlist}

        for mat_name, params in lp.items():
            if mat_name not in sld_map:
                continue

            real_sld = sld_map[mat_name]['real']
            imag_sld = abs(sld_map[mat_name]['imag'])

            # --- real SLD ---
            if 'sld_real_bounds' not in params:
                if mat_name in sld_offset_bounds and 'real' in sld_offset_bounds[mat_name]:
                    lo_off, hi_off, vary = sld_offset_bounds[mat_name]['real']
                    params['sld_real_bounds'] = (real_sld + lo_off,
                                                  real_sld + hi_off, vary)
                else:
                    params['sld_real_bounds'] = (real_sld * 0.999,
                                                  real_sld * 1.001, False)

            # --- imag SLD ---
            if 'sld_imag_bounds' not in params:
                if mat_name in sld_offset_bounds and 'imag' in sld_offset_bounds[mat_name]:
                    lo_off, hi_off, vary = sld_offset_bounds[mat_name]['imag']
                    params['sld_imag_bounds'] = (max(0.0, imag_sld + lo_off),
                                                  imag_sld + hi_off, vary)
                else:
                    params['sld_imag_bounds'] = (max(0.0, imag_sld * 0.999),
                                                  imag_sld * 1.001, False)

        energy_layer_params[energy] = lp

    return energy_layer_params


def generate_batch_models(data_dict, energy_list, material_sld_arrays,
                           constant_materials, base_layer_params, layer_order,
                           sld_offset_bounds=None, sample_name='Sample',
                           scale=1.0, bkg=None,
                           scale_bounds=(0.1, 10), bkg_bounds=(0.01, 10),
                           dq_bounds=(1.0, 2.0), vary_scale=True,
                           vary_bkg=True, vary_dq=False, dq=1.6,
                           verbose=True):
    """
    Full pipeline: SLD arrays → per-energy models and objectives.

    Args:
        data_dict           : {energy: ReflectDataset}
        energy_list         : energies to build models for
        material_sld_arrays : {name: ndarray [Energy, Real_SLD, Imag_SLD]}
        constant_materials  : {name: {'real': …, 'imag': …}}
        base_layer_params   : layer parameter template
        layer_order         : layer names top→bottom
        sld_offset_bounds   : optional SLD offset bounds (see
                              generate_layer_params_with_flexible_bounds)
        sample_name         : label prefix for model names
        scale / bkg / dq    : initial instrument parameters
        *_bounds            : instrument parameter bounds
        vary_scale/bkg/dq   : whether each instrument parameter is free
        verbose             : print progress

    Returns:
        (models_dict, structures_dict, objectives_dict)
    """
    if verbose:
        print("Step 1: interpolating SLD arrays …")
    energy_materials = generate_materials_from_sld_arrays(
        energy_list, material_sld_arrays, constant_materials)

    if verbose:
        print("\nStep 2: building layer parameters …")
    energy_layer_params = generate_layer_params_with_flexible_bounds(
        energy_materials, base_layer_params, sld_offset_bounds)

    if verbose:
        print("\nStep 3: assembling models …")
    models_dict     = {}
    structures_dict = {}
    objectives_dict = {}

    for energy in energy_list:
        if (energy not in data_dict or energy not in energy_materials
                or energy not in energy_layer_params):
            if verbose:
                print(f"  Skipping {energy} eV – missing data/materials/params.")
            continue
        try:
            _, _, structure, model_name = create_reflectometry_model(
                materials_list=energy_materials[energy],
                layer_params=energy_layer_params[energy],
                layer_order=layer_order,
                sample_name=sample_name,
                energy=energy,
            )
            model, objective = create_model_and_objective(
                structure=structure,
                data=data_dict[energy],
                model_name=model_name,
                scale=scale, bkg=bkg, dq=dq,
                vary_scale=vary_scale, vary_bkg=vary_bkg, vary_dq=vary_dq,
                scale_bounds=scale_bounds, bkg_bounds=bkg_bounds,
                dq_bounds=dq_bounds,
            )
            models_dict[energy]     = model
            structures_dict[energy] = structure
            objectives_dict[energy] = objective
            if verbose:
                print(f"  {energy} eV → {model_name}")
        except Exception as exc:
            if verbose:
                print(f"  Error at {energy} eV: {exc}")

    if verbose:
        print(f"\nBuilt {len(models_dict)} models.")
    return models_dict, structures_dict, objectives_dict


def simulate_reflectivity_profiles(energy_list, material_sld_arrays,
                                    constant_materials, base_layer_params,
                                    layer_order, q_values, sample_name='Sample',
                                    dq=1.6, scale=1.0, bkg=0.0,
                                    return_models=False, return_structures=False,
                                    verbose=True):
    """
    Forward-simulate reflectivity for a set of energies without fitting.

    Args:
        energy_list / material_sld_arrays / constant_materials /
        base_layer_params / layer_order : same as generate_batch_models
        q_values        : 1-D Q array (Å⁻¹) for the simulation
        dq / scale / bkg: instrument parameters (fixed)
        return_models   : include ReflectModel objects in output
        return_structures: include Structure objects in output
        verbose         : print progress

    Returns:
        (reflectivity_dict, structures_dict, models_dict)
        Non-requested dicts are returned as None.
    """
    from refnx.reflect import ReflectModel

    q_array = np.asarray(q_values, dtype=float)
    if q_array.ndim != 1 or q_array.size == 0:
        raise ValueError("q_values must be a non-empty 1-D array.")

    energy_materials   = generate_materials_from_sld_arrays(
        energy_list, material_sld_arrays, constant_materials)
    energy_layer_params = generate_layer_params_with_flexible_bounds(
        energy_materials, base_layer_params, sld_offset_bounds=None)

    reflectivity_dict = {}
    structures_dict   = {} if return_structures else None
    models_dict       = {} if return_models     else None

    for energy in energy_list:
        if energy not in energy_materials or energy not in energy_layer_params:
            if verbose:
                print(f"  Skipping {energy} eV – missing data.")
            continue
        try:
            _, _, structure, model_name = create_reflectometry_model(
                materials_list=energy_materials[energy],
                layer_params=energy_layer_params[energy],
                layer_order=layer_order,
                sample_name=sample_name,
                energy=energy,
            )
            model = ReflectModel(structure, scale=scale, bkg=bkg,
                                 dq=dq, name=model_name)
            model.scale.vary = model.bkg.vary = model.dq.vary = False

            reflectivity_dict[energy] = {
                'model_name':  model_name,
                'q':           q_array.copy(),
                'reflectivity': np.asarray(model(q_array), dtype=float),
            }
            if structures_dict is not None:
                structures_dict[energy] = structure
            if models_dict is not None:
                models_dict[energy] = model
            if verbose:
                print(f"  Simulated {energy} eV")
        except Exception as exc:
            if verbose:
                print(f"  Error at {energy} eV: {exc}")

    return reflectivity_dict, structures_dict, models_dict


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

def batch_fit_selected_models(objectives_dict, structures_dict,
                               energy_list=None,
                               method='differential_evolution',
                               workers=8, popsize=20,
                               steps=500, burn=200,
                               nthin=1, nwalkers=100,
                               sampler='emcee', sampler_kws=None,
                               save_dir=None, save_objectives=False,
                               save_results=False,
                               preserve_originals=True,
                               verbose=True,
                               model_name=None,
                               sample_name=None,
                               h5_filepath=None,
                               run_index=None):
    """
    Fit a batch of RSoXR objectives, one per energy.

    Args:
        objectives_dict  : {energy: Objective}
        structures_dict  : {energy: Structure}
        energy_list      : energies to fit (None = all available)
        method           : optimisation method
        workers / popsize: differential_evolution settings
        steps / burn     : MCMC total steps and burn-in steps
        nthin / nwalkers : MCMC thinning and walker count
        sampler          : 'emcee' (default), 'pymc', or 'dynesty'
        sampler_kws      : extra kwargs forwarded to the sampler (see run_fitting)
        save_dir         : directory for per-energy pickle outputs
        save_objectives  : write <model>_objective.pkl per energy
        save_results     : write <model>_results_structure.pkl per energy
        preserve_originals: keep deep copies of inputs in output dict
        verbose          : progress printing
        model_name       : label applied to all fits (default 'Model1')
        sample_name      : HDF5 sample label (e.g. 'Brewer1'); required with h5_filepath
        h5_filepath      : path to .h5 file; if given, each energy is saved
                           immediately after it finishes so a crash loses at
                           most one energy's work
        run_index        : explicit HDF5 run index (default: auto-increment)

    Returns:
        dict with keys:
            fitted_objectives, individual_results, summary_stats,
            fitted_energies, non_fitted_energies,
            original_objectives (if preserve_originals),
            original_structures (if preserve_originals)
    """
    save_to_h5 = h5_filepath is not None and sample_name is not None
    if h5_filepath is not None and sample_name is None:
        print("  Warning: h5_filepath provided without sample_name — "
              "per-energy HDF5 saving disabled.")

    print("=" * 60)
    print("BATCH FITTING")
    print("=" * 60)

    fitted_objectives = copy.deepcopy(objectives_dict)

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

    all_energies   = set(objectives_dict)
    fitted_set     = set(to_fit)
    non_fitted_set = all_energies - fitted_set

    print(f"Total: {len(all_energies)}  |  To fit: {len(fitted_set)}  |  "
          f"Pass-through: {len(non_fitted_set)}")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ename = model_name if model_name else "Model1"
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
                original_objectives[energy] = copy.deepcopy(orig_obj)
                original_structures[energy] = copy.deepcopy(structure)

            working  = copy.deepcopy(orig_obj)
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

            if save_to_h5:
                energy_batch = {
                    'fitted_objectives':  {energy: working},
                    'individual_results': {energy: results},
                    'fitted_energies':    [energy],
                }
                if preserve_originals:
                    energy_batch['original_objectives'] = {
                        energy: original_objectives[energy]}
                    energy_batch['original_structures'] = {
                        energy: original_structures[energy]}
                save_batch_to_h5(
                    energy_batch,
                    sample_name=sample_name,
                    model_name=ename,
                    filepath=h5_filepath,
                    energy_list=[energy],
                    run_index=run_index,
                )

        except Exception as exc:
            print(f"  ERROR at {energy} eV: {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Successful: {successful}  |  Failed: {failed}  |  "
          f"Pass-through: {len(non_fitted_set)}")
    if successful > 0 and chi_init_total > 0:
        overall = (chi_init_total - chi_final_total) / chi_init_total * 100
        print(f"Overall χ² improvement: {overall:.1f}%")

    summary = {
        'total_models':               len(to_fit),
        'total_objectives_in_output': len(fitted_objectives),
        'successful_fits':            successful,
        'failed_fits':                failed,
        'non_fitted_count':           len(non_fitted_set),
        'initial_chi_squared_total':  chi_init_total,
        'final_chi_squared_total':    chi_final_total,
        'overall_improvement_percent': (
            (chi_init_total - chi_final_total) / chi_init_total * 100
            if chi_init_total > 0 else 0),
        'save_directory': save_dir,
    }

    ret = {
        'fitted_objectives':   fitted_objectives,
        'individual_results':  individual_results,
        'summary_stats':       summary,
        'fitted_energies':     sorted(fitted_set),
        'non_fitted_energies': sorted(non_fitted_set),
    }
    if preserve_originals:
        ret['original_objectives'] = original_objectives
        ret['original_structures'] = original_structures

    return ret


# ---------------------------------------------------------------------------
# Results extraction
# ---------------------------------------------------------------------------

def extract_results_from_objectives(results_dict, energy_list=None):
    """
    Extract fitted parameter values from an objectives-based results dict
    into a tidy DataFrame.

    Args:
        results_dict : {energy: {'objective': Objective, ...}}
                       (e.g. the 'individual_results' key from
                       batch_fit_selected_models output)
        energy_list  : energies to include (None = all)

    Returns:
        DataFrame with columns:
            energy, model_name, goodness_of_fit,
            parameter, value, stderr, bound_low, bound_high, vary
    """
    if energy_list is None:
        energy_list = list(results_dict.keys())

    rows = []
    for energy in energy_list:
        if energy not in results_dict:
            print(f"No results for {energy} eV.")
            continue
        result = results_dict[energy]
        if 'objective' not in result:
            print(f"No objective for {energy} eV.")
            continue

        obj        = result['objective']
        model_name = getattr(obj.model, 'name', f'Model_{energy}eV')
        try:
            gof = obj.chisqr()
        except Exception:
            gof = None

        for param in obj.parameters.flattened():
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

            rows.append(dict(
                energy=energy,
                model_name=model_name,
                goodness_of_fit=gof,
                parameter=param.name,
                value=param.value,
                stderr=getattr(param, 'stderr', None),
                bound_low=bl,
                bound_high=bh,
                vary=getattr(param, 'vary', False),
            ))

    if not rows:
        print("No parameter rows extracted.")
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parameter bound utilities
# ---------------------------------------------------------------------------

def batch_update_parameter_bounds(objectives_dict, material_name='PS',
                                   param_types=None,
                                   bound_expansion_factor=1.5,
                                   energy_list=None, save_dir=None):
    """
    Expand parameter bounds by a uniform factor across all objectives.

    Args:
        objectives_dict        : {energy: Objective}
        material_name          : material whose parameters are updated
        param_types            : list of type strings to match
                                 (default: ['sld', 'isld', 'thick', 'rough'])
        bound_expansion_factor : new_range = old_range × factor
        energy_list            : energies to process (None = all)
        save_dir               : directory to pickle updated objectives
                                 (default './batch_updated')

    Returns:
        {energy: updated_Objective}
    """
    if param_types is None:
        param_types = ['sld', 'isld', 'thick', 'rough']
    if save_dir is None:
        save_dir = './batch_updated'
    os.makedirs(save_dir, exist_ok=True)

    energies = (sorted(e for e in energy_list if e in objectives_dict)
                if energy_list else sorted(objectives_dict))
    if not energies:
        print("No valid energies.")
        return {}

    param_strings = [f"{material_name} - {pt}" for pt in param_types]
    updated = {e: copy.deepcopy(objectives_dict[e]) for e in energies}

    for energy, obj in updated.items():
        count = 0
        for param in obj.parameters.flattened():
            if not any(ps in param.name for ps in param_strings):
                continue
            try:
                b = getattr(param, 'bounds', None)
                if b is None:
                    continue
                lo = b.lb if hasattr(b, 'lb') else b[0]
                hi = b.ub if hasattr(b, 'ub') else b[1]
                mid      = (lo + hi) / 2
                new_half = (hi - lo) * bound_expansion_factor / 2
                new_lo   = mid - new_half
                new_hi   = mid + new_half
                if 'isld' in param.name.lower():
                    new_lo = max(0.0, new_lo)
                param.bounds = (new_lo, new_hi)
                count += 1
            except Exception as exc:
                print(f"  Error updating {param.name} at {energy} eV: {exc}")
        print(f"{energy} eV: updated {count} parameters.")

        path = os.path.join(save_dir, f"batch_updated_objective_{energy}eV.pkl")
        with open(path, 'wb') as fh:
            pickle.dump(obj, fh)

    print(f"Updated objectives saved to {save_dir}")
    return updated


def export_parameter_bounds(objectives_dict, material_name='PS',
                             param_types=None, energy_list=None,
                             output_file='parameter_bounds.csv'):
    """
    Write a CSV summary of parameter bounds for review.

    Args:
        objectives_dict : {energy: Objective}
        material_name   : material to inspect
        param_types     : types to include (default: sld, isld, thick, rough)
        energy_list     : energies to include (None = all)
        output_file     : CSV path (None to skip writing)

    Returns:
        DataFrame with columns: energy, parameter, value, bound_low,
                                bound_high, bound_range, position_in_range,
                                near_bound, vary, chi_squared
    """
    if param_types is None:
        param_types = ['sld', 'isld', 'thick', 'rough']

    energies = (sorted(e for e in energy_list if e in objectives_dict)
                if energy_list else sorted(objectives_dict))
    if not energies:
        print("No valid energies.")
        return pd.DataFrame()

    param_strings = [f"{material_name} - {pt}" for pt in param_types]
    rows = []

    for energy in energies:
        obj = objectives_dict[energy]
        try:
            chi2 = obj.chisqr()
        except Exception:
            chi2 = None

        for param in obj.parameters.flattened():
            if not any(ps in param.name for ps in param_strings):
                continue
            bl = bh = None
            try:
                b = getattr(param, 'bounds', None)
                if b is not None:
                    bl = b.lb if hasattr(b, 'lb') else b[0]
                    bh = b.ub if hasattr(b, 'ub') else b[1]
            except Exception:
                pass

            span = (bh - bl) if (bl is not None and bh is not None) else None
            pos  = ((param.value - bl) / span
                    if span is not None and span > 0 else None)
            near = (abs(param.value - bl) < 0.02 * span or
                    abs(bh - param.value) < 0.02 * span) if span else False

            rows.append(dict(
                energy=energy, parameter=param.name,
                value=param.value, bound_low=bl, bound_high=bh,
                bound_range=span, position_in_range=pos,
                near_bound=near,
                vary=getattr(param, 'vary', False),
                chi_squared=chi2,
            ))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['energy', 'parameter'])
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved parameter bounds → {output_file}")
    return df


def export_best_parameters(results_dict, original_objectives_dict,
                            output_file=None):
    """
    Compare fitted vs original parameters across energies and write a CSV.

    Args:
        results_dict             : {energy: {'objective': Objective, …}}
        original_objectives_dict : {energy: Objective} (pre-fit)
        output_file              : CSV path (None to skip writing)

    Returns:
        DataFrame with columns: energy, parameter, original_value,
                                fitted_value, stderr, percent_change,
                                varied, original_chi2, fitted_chi2,
                                improvement_percent
    """
    fitted = {e: r['objective']
              for e, r in results_dict.items() if 'objective' in r}
    rows = []

    for energy in sorted(fitted):
        if energy not in original_objectives_dict:
            continue
        fit_obj  = fitted[energy]
        orig_obj = original_objectives_dict[energy]
        orig_chi = orig_obj.chisqr()
        fit_chi  = fit_obj.chisqr()
        pct_imp  = (orig_chi - fit_chi) / orig_chi * 100 if orig_chi else None

        orig_vals = {p.name: p.value
                     for p in orig_obj.parameters.flattened()}

        for param in fit_obj.parameters.flattened():
            orig_val = orig_vals.get(param.name)
            pct_chg  = None
            if orig_val is not None and orig_val != 0:
                pct_chg = (param.value - orig_val) / abs(orig_val) * 100
            rows.append(dict(
                energy=energy,
                parameter=param.name,
                original_value=orig_val,
                fitted_value=param.value,
                stderr=getattr(param, 'stderr', None),
                percent_change=pct_chg,
                varied=getattr(param, 'vary', False),
                original_chi2=orig_chi,
                fitted_chi2=fit_chi,
                improvement_percent=pct_imp,
            ))

    df = pd.DataFrame(rows)
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved best parameters → {output_file}")
    return df


# ---------------------------------------------------------------------------
# Parameter sweep uncertainty estimation
# ---------------------------------------------------------------------------

def batch_fit_sweep(batch_results, sweep_params, sample_name, model_name,
                    h5_filepath, run_criteria='best', uncertainty_percent=10,
                    optimization_method='differential_evolution',
                    opt_workers=8, opt_popsize=20,
                    energy_list=None, verbose=True):
    """
    Estimate parameter uncertainties via sweep fitting across a batch of energies.

    For each energy and each specified parameter, fixes that parameter at a
    series of values, optimises all others, and defines the confidence interval
    as all points within *uncertainty_percent*% of the best-fit chi-squared.
    Results are saved to the HDF5 file after each energy completes, so a crash
    loses at most one energy's sweep time.

    Parameters
    ----------
    batch_results : dict
        Output of batch_fit_selected_models. Must contain 'fitted_objectives'.
    sweep_params : list of dict
        Each dict describes one parameter to sweep:
            {'param': 'SOC - sld', 'delta': 1.0, 'step': 0.2}
            {'param': 'SOC - sld', 'values': [4.0, 4.5, 5.0, 5.5]}
        'delta'/'step' auto-generates sweep_values as best_fit ± delta in
        steps of step, clamped to the parameter's bounds.
    sample_name : str
        HDF5 sample label, e.g. 'Brewer1'.
    model_name : str
        HDF5 model label, e.g. 'Model1'.
    h5_filepath : str or Path
        Existing .h5 file (must have been written by save_batch_to_h5 first).
    run_criteria : 'best' | 'last' | int
        Which run_N group to attach sweep results to per energy.
    uncertainty_percent : float
        GOF threshold: models within this % of the best chi-squared define the CI.
    optimization_method : str
        Optimisation method passed to run_parameter_sweep.
    opt_workers / opt_popsize : int
        Differential evolution settings.
    energy_list : list of float, optional
        Subset of energies to process; defaults to batch_results['fitted_energies'].
    verbose : bool

    Returns
    -------
    dict with keys:
        sweep_results     – {energy: {param_name: {'sweep_info': …, 'ci_result': …}}}
        ci_summary        – DataFrame (energy, param_name, best_value, best_gof,
                            ci_lower, ci_upper, uncertainty_width, uncertainty_percent)
        swept_energies    – list of energies successfully saved
        skipped_energies  – list of (energy, reason) tuples
        h5_filepath       – str path
        sample_name       – str
        model_name        – str
    """
    if 'fitted_objectives' not in batch_results:
        raise ValueError("batch_results must contain 'fitted_objectives'.")

    for i, spec in enumerate(sweep_params):
        if 'param' not in spec:
            raise ValueError(f"sweep_params[{i}] missing 'param' key.")
        if 'values' not in spec and not ('delta' in spec and 'step' in spec):
            raise ValueError(
                f"sweep_params[{i}]: must have either 'values' or both "
                f"'delta' and 'step'.")

    if energy_list is not None:
        missing = [e for e in energy_list
                   if e not in batch_results['fitted_objectives']]
        if missing:
            print(f"  Warning: energies not in fitted_objectives: {missing}")
        working_energies = [e for e in energy_list
                            if e in batch_results['fitted_objectives']]
    else:
        working_energies = batch_results.get(
            'fitted_energies',
            sorted(batch_results['fitted_objectives'].keys()))

    all_sweep_results = {}
    swept_energies    = []
    skipped_energies  = []
    ci_rows           = []

    for energy in sorted(working_energies):
        objective = batch_results['fitted_objectives'].get(energy)
        if objective is None:
            skipped_energies.append((energy, 'not in fitted_objectives'))
            continue

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  Energy: {energy} eV")

        energy_sweep_data = {}

        for spec in sweep_params:
            param_name = spec['param']

            if 'values' in spec:
                sweep_values = np.asarray(spec['values'], dtype=float)
                lb = ub = None
            else:
                delta = float(spec['delta'])
                step  = float(spec['step'])

                best_val = lb = ub = None
                for p in objective.parameters.flattened(unique=True):
                    if p.name == param_name:
                        best_val = float(p.value)
                        b = getattr(p, 'bounds', None)
                        if b is not None:
                            lb = float(b.lb) if hasattr(b, 'lb') else float(b[0])
                            ub = float(b.ub) if hasattr(b, 'ub') else float(b[1])
                        break

                if best_val is None:
                    print(f"  Warning: parameter '{param_name}' not found in "
                          f"objective for {energy} eV — skipping sweep.")
                    continue

                raw_lo = best_val - delta
                raw_hi = best_val + delta
                clamped_lo = (max(raw_lo, lb) if lb is not None and np.isfinite(lb)
                              else raw_lo)
                clamped_hi = (min(raw_hi, ub) if ub is not None and np.isfinite(ub)
                              else raw_hi)

                if clamped_lo != raw_lo or clamped_hi != raw_hi:
                    print(f"  Warning: sweep range for '{param_name}' at {energy} eV "
                          f"clamped from [{raw_lo:.4g}, {raw_hi:.4g}] to "
                          f"[{clamped_lo:.4g}, {clamped_hi:.4g}].")

                sweep_values = np.arange(clamped_lo, clamped_hi + step / 2, step)
                if len(sweep_values) == 0:
                    print(f"  Warning: no sweep values generated for '{param_name}' "
                          f"at {energy} eV after clamping — skipping.")
                    continue

            if verbose:
                print(f"    Sweeping '{param_name}': {len(sweep_values)} points "
                      f"[{sweep_values[0]:.4g} → {sweep_values[-1]:.4g}]")

            # Deep-copy so each sweep starts from the clean best-fit state
            obj_copy = copy.deepcopy(objective)

            try:
                sweep_info = setup_parameter_sweep(obj_copy, param_name, sweep_values)
                sweep_info, _ = run_parameter_sweep(
                    sweep_info,
                    optimization_method=optimization_method,
                    opt_workers=opt_workers,
                    opt_popsize=opt_popsize,
                    save_dir=None,
                    save_intermediate=False,
                    save_combined=False,
                )
                ci_result = get_best_fit_with_uncertainty(
                    sweep_info, uncertainty_percent)
            except ValueError as exc:
                print(f"  Warning: sweep failed for '{param_name}' at "
                      f"{energy} eV: {exc}")
                continue
            except Exception as exc:
                print(f"  Warning: unexpected error sweeping '{param_name}' at "
                      f"{energy} eV: {exc}")
                continue

            energy_sweep_data[param_name] = {
                'sweep_info': sweep_info,
                'ci_result':  ci_result,
            }

            lo, hi = ci_result['uncertainty_range']
            ci_rows.append(dict(
                energy=energy,
                param_name=param_name,
                best_value=ci_result['best_value'],
                best_gof=ci_result['best_gof'],
                ci_lower=lo,
                ci_upper=hi,
                uncertainty_width=ci_result['uncertainty_width'],
                uncertainty_percent=ci_result['uncertainty_percent'],
            ))

            if verbose:
                print(f"      CI ({uncertainty_percent}%): [{lo:.4g}, {hi:.4g}]  "
                      f"width={ci_result['uncertainty_width']:.4g}")

        if energy_sweep_data:
            save_sweep_results_to_h5(
                energy_sweep_data, sample_name, model_name, h5_filepath,
                energy, run_criteria=run_criteria,
                uncertainty_percent=uncertainty_percent,
            )
            all_sweep_results[energy] = energy_sweep_data
            swept_energies.append(energy)
        else:
            skipped_energies.append(
                (energy, 'no sweep parameters produced results'))

    ci_summary = (pd.DataFrame(ci_rows) if ci_rows
                  else pd.DataFrame(columns=[
                      'energy', 'param_name', 'best_value', 'best_gof',
                      'ci_lower', 'ci_upper', 'uncertainty_width',
                      'uncertainty_percent']))

    print(f"\nBatch sweep complete: {len(swept_energies)} energies saved "
          f"→ {h5_filepath}")

    return {
        'sweep_results':   all_sweep_results,
        'ci_summary':      ci_summary,
        'swept_energies':  swept_energies,
        'skipped_energies': skipped_energies,
        'h5_filepath':     str(h5_filepath),
        'sample_name':     sample_name,
        'model_name':      model_name,
    }
