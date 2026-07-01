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
                  extract_sld_from_h5, set_final_model,
                  plot_parameter_vs_energy, plot_reflectivity,
                  plot_stacked_reflectivity_h5,
                  plot_reflectivity_comparison, plot_material_comparison,
                  save_nexafs_to_h5, update_nexafs_sld,
                  load_nexafs_from_h5, list_nexafs_spectra)
from parameter_sweep import (setup_parameter_sweep, run_parameter_sweep,
                              get_best_fit_with_uncertainty)
 

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def import_batch_reflectivity(folder_path, file_type='smoothed', q_max=None):
    """
    Load a batch of reflectivity .dat files from a folder.

    Args:
        folder_path : path to the folder containing reflectivity files
        file_type   : 'raw'      → *raw.dat
                      'smoothed' → *smoothed.dat  (default)
                      'all'      → all .dat files
        q_max       : if given, mask out points with Q > q_max in every
                       loaded dataset (default None → no truncation)

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
    n_total_points  = 0
    n_masked_points = 0

    for fp in folder_path.glob('*.dat'):
        m = re.match(pattern, fp.name)
        if m:
            try:
                energy = float(m.group(1))
                ds = ReflectDataset(str(fp))
                if q_max is not None:
                    n_total_points += ds.x.size
                    ds.mask = ds.x <= q_max
                    n_masked_points += ds.x.size
                data_dict[energy] = ds
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
    if q_max is not None:
        print(f"Truncated to Q <= {q_max}: kept {n_masked_points}/{n_total_points} points")
    return data_dict, energy_list


def plot_data_comparison(data_dicts, labels, energies,
                         tolerance=0.6, log_y=True,
                         figsize_per_panel=(6, 4), colors=None,
                         show_errorbars=False, xlim=None, ylim=None,
                         scales=None, ncols=1, save_path=None):
    """
    Overlay reflectivity data from multiple samples at a set of energies.

    All samples are drawn on the same axes for each energy.  Panels are
    arranged in a grid of ncols columns (energies fill left-to-right,
    top-to-bottom).

    Args:
        data_dicts         : list of data_dict objects from import_batch_reflectivity
        labels             : list of sample names matching data_dicts
        energies           : list of target energies (eV) to plot
        tolerance          : max eV distance for matching keys in each dict (default 0.6)
        log_y              : log scale on R axis (default True)
        figsize_per_panel  : (width, height) for each individual panel (default (6, 4))
        colors             : list of colours, one per sample (default matplotlib cycle)
        show_errorbars     : draw R error bars (default False)
        xlim               : (qmin, qmax) or None
        ylim               : (rmin, rmax) or None
        scales             : multiplicative scale factors for R, one of:
                               - None            → no scaling (default)
                               - float           → same factor for all samples/energies
                               - list of float   → one factor per sample (same at all energies)
                               - list of lists   → scales[sample_idx][energy_idx]
        ncols              : number of columns in the panel grid (default 1)
        save_path          : file path to save the figure (e.g. 'fig.png'); None = no save

    Returns:
        (fig, axes)  – axes is a 2-D array of shape (n_rows, ncols)
    """
    import math
    import matplotlib.pyplot as plt

    if len(data_dicts) != len(labels):
        raise ValueError("data_dicts and labels must have the same length.")

    n_samples = len(data_dicts)
    n_energies = len(energies)
    n_rows = math.ceil(n_energies / ncols)

    # Normalise scales into a 2-D list [sample_idx][energy_idx]
    if scales is None:
        _scales = [[1.0] * n_energies for _ in range(n_samples)]
    elif isinstance(scales, (int, float)):
        _scales = [[float(scales)] * n_energies for _ in range(n_samples)]
    elif isinstance(scales[0], (int, float)):
        _scales = [[float(s)] * n_energies for s in scales]
    else:
        _scales = [[float(s) for s in row] for row in scales]

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = [prop_cycle[i % len(prop_cycle)] for i in range(n_samples)]

    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for idx, target_e in enumerate(energies):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        found_any = False

        for si, (dd, label, color) in enumerate(zip(data_dicts, labels, colors)):
            keys = list(dd.keys())
            if not keys:
                continue
            closest = min(keys, key=lambda e: abs(e - target_e))
            if abs(closest - target_e) > tolerance:
                continue

            ds = dd[closest]
            q = ds.x
            scale = _scales[si][idx]
            r = ds.y * scale
            dr = ds.y_err * scale if ds.y_err is not None else None

            scale_str = f' ×{scale}' if scale != 1.0 else ''
            leg_label = f"{label} ({closest:.2f} eV){scale_str}"

            if show_errorbars and dr is not None and np.any(dr > 0):
                ax.errorbar(q, r, yerr=dr, fmt='-', color=color,
                            label=leg_label, linewidth=1.2,
                            elinewidth=0.6, capsize=2)
            else:
                ax.plot(q, r, '-', color=color, label=leg_label, linewidth=1.2)
            found_any = True

        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel(r'Q ($\AA^{-1}$)')
        ax.set_ylabel('Reflectivity')
        ax.set_title(f'{target_e} eV')
        ax.legend(fontsize=8)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if not found_any:
            ax.text(0.5, 0.5, f'No data within {tolerance} eV of {target_e} eV',
                    transform=ax.transAxes, ha='center', va='center', color='grey')

    # Hide any unused panels in the last row
    for idx in range(n_energies, n_rows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, axes


def plot_intensity_vs_energy(data_dicts, labels, q_targets,
                              colors=None, figsize_per_panel=(7, 4),
                              log_y=True, scales=None,
                              xlim=None, ylim=None, ncols=1,
                              save_path=None):
    """
    Plot intensity (R) vs energy at fixed Q values for multiple samples.

    For each target Q the closest available Q point is used from each
    dataset at each energy.  All samples are overlaid per panel.  Panels
    are arranged in a grid of ncols columns (Q values fill left-to-right,
    top-to-bottom).

    Args:
        data_dicts        : list of data_dicts from import_batch_reflectivity
        labels            : list of sample names matching data_dicts
        q_targets         : list of Q values (Å⁻¹) to extract
        colors            : list of colours, one per sample (default cycle)
        figsize_per_panel : (width, height) for each individual panel (default (7, 4))
        log_y             : log scale on intensity axis (default True)
        scales            : multiplicative scale factors for R — same format
                            as plot_data_comparison (None, float, list of
                            float, or list of lists [sample][q_index])
        xlim              : (emin, emax) or None
        ylim              : (imin, imax) or None
        ncols             : number of columns in the panel grid (default 1)
        save_path         : file path to save the figure (e.g. 'fig.png'); None = no save

    Returns:
        (fig, axes)  – axes is a 2-D array of shape (n_rows, ncols)
    """
    import math
    import matplotlib.pyplot as plt

    if len(data_dicts) != len(labels):
        raise ValueError("data_dicts and labels must have the same length.")

    n_samples = len(data_dicts)
    n_q = len(q_targets)
    n_rows = math.ceil(n_q / ncols)

    # Normalise scales into 2-D list [sample_idx][q_idx]
    if scales is None:
        _scales = [[1.0] * n_q for _ in range(n_samples)]
    elif isinstance(scales, (int, float)):
        _scales = [[float(scales)] * n_q for _ in range(n_samples)]
    elif isinstance(scales[0], (int, float)):
        _scales = [[float(s)] * n_q for s in scales]
    else:
        _scales = [[float(s) for s in row] for row in scales]

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = [prop_cycle[i % len(prop_cycle)] for i in range(n_samples)]

    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for qi, q_target in enumerate(q_targets):
        row, col = divmod(qi, ncols)
        ax = axes[row, col]

        for si, (dd, label, color) in enumerate(zip(data_dicts, labels, colors)):
            energies_sorted = sorted(dd.keys())
            if not energies_sorted:
                continue

            scale = _scales[si][qi]
            plot_energies, plot_intensities, actual_qs = [], [], []

            for energy in energies_sorted:
                ds = dd[energy]
                q_arr = ds.x
                if q_arr is None or len(q_arr) == 0:
                    continue
                idx = int(np.argmin(np.abs(q_arr - q_target)))
                plot_energies.append(energy)
                plot_intensities.append(ds.y[idx] * scale)
                actual_qs.append(q_arr[idx])

            if not plot_energies:
                continue

            q_mean = np.mean(actual_qs)
            scale_str = f' ×{scale}' if scale != 1.0 else ''
            leg_label = f"{label} (Q≈{q_mean:.4f} Å⁻¹){scale_str}"
            ax.plot(plot_energies, plot_intensities, '-o', color=color,
                    label=leg_label, linewidth=1.2, markersize=3)

        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity (R)')
        ax.set_title(f'Q = {q_target} Å⁻¹')
        ax.legend(fontsize=8)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    # Hide unused panels
    for qi in range(n_q, n_rows * ncols):
        row, col = divmod(qi, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, axes


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


# ---------------------------------------------------------------------------
# GPU-accelerated batch fitting functions
# ---------------------------------------------------------------------------

def batch_fit_selected_models_gpu(
    objectives_dict,
    energy_list=None,
    popsize=20,
    n_generations=300,
    seed=0,
    verbose=True,
    sample_name=None,
    model_name=None,
    h5_filepath=None,
    run_index=None,
):
    """
    Fit reflectometry models for all energies simultaneously on GPU.

    Replaces the sequential per-energy scipy DE loop in
    batch_fit_selected_models with a batched JAX/evosax DE that optimises
    all energies in parallel.  No MCMC sampling is performed.

    Parameters
    ----------
    objectives_dict : dict {energy: refnx Objective}
        Objectives ready for fitting (parameters configured with vary=True
        and bounds set).
    energy_list     : list of float, optional
        Subset of energies to fit (default: all keys in objectives_dict).
    popsize         : int
        DE population size per energy (default 20).
    n_generations   : int
        Number of DE generations (default 300).
    seed            : int
        Random seed for reproducibility.
    verbose         : bool
        Print progress every 50 generations.
    sample_name     : str, optional
        HDF5 sample label; required if h5_filepath is given.
    model_name      : str, optional
        HDF5 model label (default 'Model1').
    h5_filepath     : str or Path, optional
        Save fitted objectives to this HDF5 file after all energies finish.
    run_index       : int, optional
        Explicit HDF5 run index (default: auto-increment).

    Returns
    -------
    dict with keys matching batch_fit_selected_models output:
        fitted_objectives  – {energy: Objective at best-fit params}
        individual_results – {energy: {'objective': Objective, 'chisqr': float}}
        summary_stats      – summary dict
        fitted_energies    – sorted list of fitted energies
        non_fitted_energies – []
        elapsed_sec        – GPU wall time
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from gpu_sweep_optimizer import gpu_fit_models

    if energy_list is None:
        energy_list = sorted(objectives_dict.keys())

    to_fit = [e for e in energy_list if e in objectives_dict]
    if not to_fit:
        print("batch_fit_selected_models_gpu: no valid energies found.")
        return None

    ename = model_name or "Model1"

    print("=" * 60)
    print("BATCH FITTING (GPU)")
    print(f"  Energies: {len(to_fit)}  |  popsize: {popsize}  |  "
          f"generations: {n_generations}")
    print("=" * 60)

    # Capture initial chi-squared before objectives are modified in-place
    chi_init = {e: objectives_dict[e].chisqr() for e in to_fit}

    gpu_result = gpu_fit_models(
        objectives_dict=objectives_dict,
        energy_list=to_fit,
        popsize=popsize,
        n_generations=n_generations,
        seed=seed,
        verbose=verbose,
    )

    fitted_objectives = gpu_result["fitted_objectives"]
    best_chisqr_dict = gpu_result["best_chisqr"]

    individual_results = {}
    chi_init_total = chi_final_total = 0.0

    for energy in to_fit:
        ci = chi_init[energy]
        chi_init_total += ci
        fitted_obj = fitted_objectives[energy]
        chi_final = float(best_chisqr_dict[energy])
        chi_final_total += chi_final
        individual_results[energy] = {
            "objective": fitted_obj,
            "chisqr": chi_final,
        }
        if verbose:
            pct = (ci - chi_final) / ci * 100 if ci > 0 else 0.0
            print(f"  {energy} eV: χ² {ci:.4g} → {chi_final:.4g}  "
                  f"({pct:.1f}% improvement)")

    summary = {
        "total_models": len(to_fit),
        "total_objectives_in_output": len(fitted_objectives),
        "successful_fits": len(to_fit),
        "failed_fits": 0,
        "non_fitted_count": 0,
        "initial_chi_squared_total": chi_init_total,
        "final_chi_squared_total": chi_final_total,
        "overall_improvement_percent": (
            (chi_init_total - chi_final_total) / chi_init_total * 100
            if chi_init_total > 0 else 0.0
        ),
        "elapsed_sec": gpu_result["elapsed_sec"],
        "gpu_accelerated": True,
    }

    print(f"\nGPU fitting complete in {gpu_result['elapsed_sec']:.1f}s  "
          f"(χ² improvement: {summary['overall_improvement_percent']:.1f}%)")

    ret = {
        "fitted_objectives":   fitted_objectives,
        "individual_results":  individual_results,
        "summary_stats":       summary,
        "fitted_energies":     sorted(to_fit),
        "non_fitted_energies": [],
        "elapsed_sec":         gpu_result["elapsed_sec"],
    }

    if h5_filepath is not None and sample_name is not None:
        save_batch_to_h5(
            ret,
            sample_name=sample_name,
            model_name=ename,
            filepath=h5_filepath,
            energy_list=to_fit,
            run_index=run_index,
        )
        print(f"Saved to {h5_filepath}")

    return ret


def batch_fit_selected_models_cmaes(
    objectives_dict,
    energy_list=None,
    popsize=20,
    n_generations=500,
    seed=0,
    verbose=True,
    tol=1e-4,
    patience=5,
    check_every=10,
    sample_name=None,
    model_name=None,
    h5_filepath=None,
    run_index=None,
    normalize=False,
):
    """
    Fit reflectometry models for all energies simultaneously using Sep-CMA-ES.

    Drop-in replacement for batch_fit_selected_models_gpu that uses the
    Separable CMA-ES algorithm instead of Differential Evolution.  In
    benchmarks on reflectometry problems Sep-CMA-ES reaches ~2× lower
    chi-squared in the same number of generations, and supports early
    stopping when the fit has converged.

    Parameters
    ----------
    objectives_dict : dict {energy: refnx Objective}
    energy_list     : subset of energies to fit (default: all)
    popsize         : population size per energy (default 20)
    n_generations   : maximum generations; actual run may be shorter if
                      early stopping fires (default 500)
    seed            : random seed
    verbose         : print per-checkpoint progress
    tol             : relative improvement threshold for early stopping
                      (default 1e-4 = 0.01%); set to 0 to disable
    patience        : consecutive non-improving checks before stopping (default 5)
    check_every     : generations between convergence checks and GPU→CPU syncs
                      (default 10); each block of check_every generations is
                      compiled as a single XLA program on GPU
    sample_name     : HDF5 sample label (required with h5_filepath)
    model_name      : HDF5 model label (default 'Model1')
    h5_filepath     : save fitted objectives here after completion
    run_index       : explicit HDF5 run index (default: auto-increment)

    Returns
    -------
    Same dict format as batch_fit_selected_models_gpu, with two extra keys:
      'generations_run' – actual generations completed
      'converged'       – True if early stopping fired
    """
    from gpu_sweep_optimizer import gpu_fit_models_cmaes

    if energy_list is None:
        energy_list = sorted(objectives_dict.keys())

    to_fit = [e for e in energy_list if e in objectives_dict]
    if not to_fit:
        print("batch_fit_selected_models_cmaes: no valid energies found.")
        return None

    ename = model_name or "Model1"

    print("=" * 60)
    print("BATCH FITTING (GPU — Sep-CMA-ES)")
    print(f"  Energies: {len(to_fit)}  |  popsize: {popsize}  |  "
          f"max generations: {n_generations}  |  tol: {tol:.0e}  |  "
          f"check_every: {check_every}")
    print("=" * 60)

    chi_init = {e: objectives_dict[e].chisqr() for e in to_fit}

    gpu_result = gpu_fit_models_cmaes(
        objectives_dict=objectives_dict,
        energy_list=to_fit,
        popsize=popsize,
        n_generations=n_generations,
        seed=seed,
        verbose=verbose,
        tol=tol,
        patience=patience,
        check_every=check_every,
        normalize=normalize,
    )

    fitted_objectives = gpu_result["fitted_objectives"]
    best_chisqr_dict  = gpu_result["best_chisqr"]

    individual_results = {}
    chi_init_total = chi_final_total = 0.0
    for energy in to_fit:
        ci = chi_init[energy]
        chi_init_total += ci
        fitted_obj  = fitted_objectives[energy]
        chi_final   = float(best_chisqr_dict[energy])
        chi_final_total += chi_final
        individual_results[energy] = {"objective": fitted_obj, "chisqr": chi_final}
        if verbose:
            pct = (ci - chi_final) / ci * 100 if ci > 0 else 0.0
            print(f"  {energy} eV: χ² {ci:.4g} → {chi_final:.4g}  ({pct:.1f}% improvement)")

    summary = {
        "total_models": len(to_fit),
        "total_objectives_in_output": len(fitted_objectives),
        "successful_fits": len(to_fit),
        "failed_fits": 0,
        "non_fitted_count": 0,
        "initial_chi_squared_total": chi_init_total,
        "final_chi_squared_total": chi_final_total,
        "overall_improvement_percent": (
            (chi_init_total - chi_final_total) / chi_init_total * 100
            if chi_init_total > 0 else 0.0
        ),
        "elapsed_sec": gpu_result["elapsed_sec"],
        "generations_run": gpu_result["generations_run"],
        "converged": gpu_result["converged"],
        "gpu_accelerated": True,
        "algorithm": "Sep-CMA-ES",
    }

    conv_str = (f"converged at gen {gpu_result['generations_run']}"
                if gpu_result["converged"] else
                f"ran full {gpu_result['generations_run']} generations")
    print(f"\nGPU fitting complete in {gpu_result['elapsed_sec']:.1f}s  "
          f"({conv_str}, χ² improvement: {summary['overall_improvement_percent']:.1f}%)")

    ret = {
        "fitted_objectives":   fitted_objectives,
        "individual_results":  individual_results,
        "summary_stats":       summary,
        "fitted_energies":     sorted(to_fit),
        "non_fitted_energies": [],
        "elapsed_sec":         gpu_result["elapsed_sec"],
        "generations_run":     gpu_result["generations_run"],
        "converged":           gpu_result["converged"],
    }

    if h5_filepath is not None and sample_name is not None:
        save_batch_to_h5(
            ret,
            sample_name=sample_name,
            model_name=ename,
            filepath=h5_filepath,
            energy_list=to_fit,
            run_index=run_index,
        )
        print(f"Saved to {h5_filepath}")

    return ret


def batch_fit_sweep_gpu(
    batch_results,
    sweep_params,
    sample_name,
    model_name,
    h5_filepath,
    run_criteria='best',
    uncertainty_percent=10,
    popsize=20,
    n_generations=200,
    energy_list=None,
    verbose=True,
):
    """
    GPU-accelerated parameter sweep fitting across a batch of energies.

    Drop-in replacement for batch_fit_sweep that uses JAX/evosax to run all
    sweep-point optimizations simultaneously on GPU instead of sequentially
    with scipy DE.

    For each energy and each specified parameter, evaluates all sweep values
    in a single batched GPU run, then extracts confidence intervals.

    Parameters
    ----------
    batch_results       : dict
        Output of batch_fit_selected_models or batch_fit_selected_models_gpu.
        Must contain 'fitted_objectives'.
    sweep_params        : list of dict
        Each dict describes one parameter to sweep:
            {'param': 'SOC - sld', 'delta': 1.0, 'step': 0.2}
            {'param': 'SOC - sld', 'values': [4.0, 4.5, 5.0, 5.5]}
    sample_name         : str
        HDF5 sample label.
    model_name          : str
        HDF5 model label.
    h5_filepath         : str or Path
        Existing .h5 file to append sweep results to.
    run_criteria        : 'best' | 'last' | int
        Which run_N group to attach sweep results to per energy.
    uncertainty_percent : float
        GOF threshold for CI calculation.
    popsize             : int
        DE population size per sweep point (default 20).
    n_generations       : int
        DE generations per sweep (default 200).
    energy_list         : list of float, optional
        Subset of energies to process.
    verbose             : bool

    Returns
    -------
    Same dict structure as batch_fit_sweep:
        sweep_results, ci_summary, swept_energies, skipped_energies,
        h5_filepath, sample_name, model_name
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from gpu_sweep_optimizer import gpu_parameter_sweep, gpu_result_to_sweep_info

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
        working_energies = [
            e for e in energy_list
            if e in batch_results['fitted_objectives']
        ]
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
                print(f"    Sweeping '{param_name}' (GPU): {len(sweep_values)} points "
                      f"[{sweep_values[0]:.4g} → {sweep_values[-1]:.4g}]")

            obj_copy = copy.deepcopy(objective)

            try:
                gpu_result = gpu_parameter_sweep(
                    obj_copy,
                    param_name=param_name,
                    sweep_values=sweep_values,
                    popsize=popsize,
                    n_generations=n_generations,
                    verbose=False,
                )
                sweep_info = gpu_result_to_sweep_info(
                    gpu_result, obj_copy, param_name, sweep_values
                )
                ci_result = get_best_fit_with_uncertainty(
                    sweep_info, uncertainty_percent
                )
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
                      f"width={ci_result['uncertainty_width']:.4g}  "
                      f"best_gof={ci_result['best_gof']:.4g}  "
                      f"elapsed={gpu_result['elapsed_sec']:.1f}s")

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

    print(f"\nGPU sweep complete: {len(swept_energies)} energies saved "
          f"→ {h5_filepath}")

    return {
        'sweep_results':    all_sweep_results,
        'ci_summary':       ci_summary,
        'swept_energies':   swept_energies,
        'skipped_energies': skipped_energies,
        'h5_filepath':      str(h5_filepath),
        'sample_name':      sample_name,
        'model_name':       model_name,
    }


def batch_fit_sweep_cmaes(
    batch_results,
    sweep_params,
    sample_name,
    model_name,
    h5_filepath,
    run_criteria='best',
    uncertainty_percent=10,
    popsize=20,
    n_generations=300,
    tol=1e-4,
    patience=5,
    energy_list=None,
    verbose=True,
):
    """
    GPU parameter sweep using Sep-CMA-ES — drop-in replacement for batch_fit_sweep_gpu.

    Runs the sweep-point optimizations with Sep-CMA-ES instead of DE.
    Supports early stopping: each energy's sweep halts once convergence is
    detected across sweep points, rather than always running to n_generations.

    Parameters
    ----------
    batch_results       : output of batch_fit_selected_models_cmaes or _gpu
    sweep_params        : list of dicts, each describing one parameter to sweep:
                            {'param': 'layer1 - thick', 'delta': 20.0, 'step': 2.0}
                            {'param': 'layer1 - thick', 'values': [30, 40, 50, 60]}
    sample_name         : HDF5 sample label
    model_name          : HDF5 model label
    h5_filepath         : existing .h5 file to append sweep results to
    run_criteria        : 'best' | 'last' | int — which run to attach results to
    uncertainty_percent : GOF threshold for CI calculation (default 10%)
    popsize             : population size per sweep point (default 20)
    n_generations       : maximum generations per sweep (default 300)
    tol                 : early stopping relative improvement threshold (default 1e-4)
    patience            : consecutive non-improving checks before stopping (default 5)
    energy_list         : subset of energies to process (default: all fitted)
    verbose             : print per-energy progress

    Returns
    -------
    Same dict as batch_fit_sweep_gpu:
        sweep_results, ci_summary, swept_energies, skipped_energies,
        h5_filepath, sample_name, model_name
    """
    from gpu_sweep_optimizer import (
        gpu_parameter_sweep_cmaes_batched_energies,
        gpu_result_to_sweep_info,
    )

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
        working_energies = [
            e for e in energy_list
            if e in batch_results['fitted_objectives']
        ]
    else:
        working_energies = batch_results.get(
            'fitted_energies',
            sorted(batch_results['fitted_objectives'].keys()))

    all_sweep_results = {}
    swept_energies    = []
    skipped_energies  = []
    ci_rows           = []

    # Accumulate per-energy sweep data across all sweep params before saving to HDF5.
    # {energy: {param_name: {'sweep_info': ..., 'ci_result': ...}}}
    energy_sweep_data_all = {}

    for spec in sweep_params:
        param_name = spec['param']

        # ── Phase 1: build per-energy objectives and sweep grids ──────────────
        objectives_for_sweep    = {}
        sweep_values_per_energy = {}

        for energy in sorted(working_energies):
            objective = batch_results['fitted_objectives'].get(energy)
            if objective is None:
                if not any(e == energy for e, _ in skipped_energies):
                    skipped_energies.append((energy, 'not in fitted_objectives'))
                continue

            if 'values' in spec:
                sv = np.asarray(spec['values'], dtype=float)
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
                          f"objective for {energy} eV — skipping.")
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

                sv = np.arange(clamped_lo, clamped_hi + step / 2, step)
                if len(sv) == 0:
                    print(f"  Warning: no sweep values generated for '{param_name}' "
                          f"at {energy} eV after clamping — skipping.")
                    continue

            objectives_for_sweep[energy]    = copy.deepcopy(objective)
            sweep_values_per_energy[energy] = sv

        if not objectives_for_sweep:
            continue

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  Sweeping '{param_name}' (Sep-CMA-ES, batched): "
                  f"{len(objectives_for_sweep)} energies")
            for energy, sv in sweep_values_per_energy.items():
                print(f"    {energy} eV: {len(sv)} points "
                      f"[{sv[0]:.4g} → {sv[-1]:.4g}]")

        # ── Phase 2: one batched GPU call for all energies ────────────────────
        try:
            results_by_energy = gpu_parameter_sweep_cmaes_batched_energies(
                objectives_for_sweep,
                param_name,
                sweep_values_per_energy,
                popsize=popsize,
                n_generations=n_generations,
                tol=tol,
                patience=patience,
                verbose=verbose,
            )
        except ValueError as exc:
            print(f"  Warning: batched sweep failed for '{param_name}': {exc}")
            continue
        except Exception as exc:
            print(f"  Warning: unexpected error during batched sweep of "
                  f"'{param_name}': {exc}")
            continue

        # ── Phase 3: post-process per energy ──────────────────────────────────
        sample_result = next(iter(results_by_energy.values()))
        conv_str = (
            f"converged gen {sample_result['generations_run']}"
            if sample_result['converged']
            else f"gen {sample_result['generations_run']}"
        )

        for energy, gpu_result in results_by_energy.items():
            obj_copy = objectives_for_sweep[energy]
            sv = sweep_values_per_energy[energy]

            try:
                sweep_info = gpu_result_to_sweep_info(
                    gpu_result, obj_copy, param_name, sv
                )
                ci_result = get_best_fit_with_uncertainty(
                    sweep_info, uncertainty_percent
                )
            except Exception as exc:
                print(f"  Warning: post-processing failed for '{param_name}' at "
                      f"{energy} eV: {exc}")
                continue

            if energy not in energy_sweep_data_all:
                energy_sweep_data_all[energy] = {}
            energy_sweep_data_all[energy][param_name] = {
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
                print(f"    {energy} eV  CI ({uncertainty_percent}%): "
                      f"[{lo:.4g}, {hi:.4g}]  "
                      f"width={ci_result['uncertainty_width']:.4g}  "
                      f"best_gof={ci_result['best_gof']:.4g}  "
                      f"elapsed={gpu_result['elapsed_sec']:.1f}s  ({conv_str})")

    # ── Save per energy to HDF5 (all sweep params combined) ───────────────────
    for energy in sorted(working_energies):
        energy_sweep_data = energy_sweep_data_all.get(energy, {})
        if energy_sweep_data:
            save_sweep_results_to_h5(
                energy_sweep_data, sample_name, model_name, h5_filepath,
                energy, run_criteria=run_criteria,
                uncertainty_percent=uncertainty_percent,
            )
            all_sweep_results[energy] = energy_sweep_data
            swept_energies.append(energy)
        elif not any(e == energy for e, _ in skipped_energies):
            skipped_energies.append(
                (energy, 'no sweep parameters produced results'))

    ci_summary = (pd.DataFrame(ci_rows) if ci_rows
                  else pd.DataFrame(columns=[
                      'energy', 'param_name', 'best_value', 'best_gof',
                      'ci_lower', 'ci_upper', 'uncertainty_width',
                      'uncertainty_percent']))

    print(f"\nSep-CMA-ES sweep complete: {len(swept_energies)} energies saved "
          f"→ {h5_filepath}")

    return {
        'sweep_results':    all_sweep_results,
        'ci_summary':       ci_summary,
        'swept_energies':   swept_energies,
        'skipped_energies': skipped_energies,
        'h5_filepath':      str(h5_filepath),
        'sample_name':      sample_name,
        'model_name':       model_name,
    }
