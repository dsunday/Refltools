"""
h5io.py
-------
HDF5 I/O for batch RSoXR fitting results.

HDF5 layout
-----------
/{sample_name}/
  /{energy_str}/                     e.g. "270.0"
    attrs: energy_eV
    /{model_name}/                   e.g. "Model1"
      /run_0/
        attrs: chi_sq_initial, chi_sq_final, timestamp, run_index, transform
        /parameters/
          names           vlen-str  [n_params]
          initial_values  float64   [n_params]
          initial_lb      float64   [n_params]   (-inf if unconstrained)
          initial_ub      float64   [n_params]   (+inf if unconstrained)
          initial_vary    int8      [n_params]
          final_values    float64   [n_params]
          final_lb        float64   [n_params]
          final_ub        float64   [n_params]
          final_vary      int8      [n_params]
          stderr          float64   [n_params]   (NaN where unavailable)
        /data/
          q    float64 [n_points]
          R    float64 [n_points]
          dR   float64 [n_points]   (NaN where unavailable)
        layer_names             vlen-str [n_layers]
        structure_slabs_initial float64  [n_layers, 5]
        structure_slabs_final   float64  [n_layers, 5]
      /run_1/
        ...

The schema is append-only: future versions may add new datasets or attributes
inside run_N groups without breaking existing load code.
"""

import datetime
import numpy as np
import h5py

from refnx.reflect import SLD, Structure, ReflectModel
from refnx.analysis import Objective, Transform
from refnx.dataset import ReflectDataset


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_param_arrays(objective):
    """
    Extract parameter metadata from a refnx Objective.

    Returns
    -------
    names   : list of str
    values  : float64 array [n]
    lb, ub  : float64 arrays [n]  (-inf / +inf for unconstrained)
    vary    : int8 array [n]
    stderr  : float64 array [n]  (NaN where unavailable)
    """
    params = objective.parameters.flattened(unique=True)
    names, values, lbs, ubs, varys, stderrs = [], [], [], [], [], []

    for p in params:
        names.append(str(p.name) if p.name is not None else '')
        values.append(float(p.value))
        varys.append(1 if p.vary else 0)
        stderrs.append(float(p.stderr) if p.stderr is not None else np.nan)

        b = p.bounds
        if b is not None and hasattr(b, 'lb') and hasattr(b, 'ub'):
            lbs.append(float(b.lb))
            ubs.append(float(b.ub))
        else:
            lbs.append(-np.inf)
            ubs.append(np.inf)

    return (
        names,
        np.array(values, dtype=np.float64),
        np.array(lbs,    dtype=np.float64),
        np.array(ubs,    dtype=np.float64),
        np.array(varys,  dtype=np.int8),
        np.array(stderrs, dtype=np.float64),
    )


def _get_layer_names(structure):
    """
    Extract ordered material names from a refnx Structure.
    Falls back to 'layer_N' for components without a recognisable name.
    """
    names = []
    for i, component in enumerate(structure.data):
        if hasattr(component, 'sld') and hasattr(component.sld, 'name'):
            names.append(str(component.sld.name))
        elif hasattr(component, 'name') and component.name:
            names.append(str(component.name))
        else:
            names.append(f'layer_{i}')
    return names


def _transform_form(objective):
    """Return the transform form string from an Objective ('logY', 'lin', etc.)."""
    t = getattr(objective, 'transform', None)
    if t is None:
        return 'logY'
    if hasattr(t, 'form'):
        return str(t.form) if t.form is not None else 'logY'
    return 'logY'


def _decode_strings(arr):
    """Convert an h5py string array to a plain Python list of str."""
    return [s if isinstance(s, str) else s.decode('utf-8') for s in arr]


def _reconstruct_objective(layer_names, final_slabs,
                            param_names, final_values, final_lb, final_ub, final_vary,
                            q, R, dR, transform='logY'):
    """
    Rebuild a refnx (Objective, Structure) pair from stored numerical data.

    Parameters
    ----------
    layer_names : sequence of str
    final_slabs : ndarray, shape (n_layers, 5)
        Columns: thick, SLD.real, SLD.imag, rough, vfsolv
    param_names : sequence of str
    final_values, final_lb, final_ub : float64 arrays
    final_vary : int8 array
    q, R, dR : float64 arrays  (dR may be None or all-NaN)
    transform : str

    Returns
    -------
    objective : refnx.analysis.Objective
    structure : refnx.reflect.Structure
    """
    pmap = {
        name: {'value': float(val), 'lb': float(lb), 'ub': float(ub), 'vary': bool(vary)}
        for name, val, lb, ub, vary in zip(
            param_names, final_values, final_lb, final_ub, final_vary)
    }

    def _apply(param, key):
        if key not in pmap:
            return
        d = pmap[key]
        param.setp(value=d['value'], vary=d['vary'], bounds=(d['lb'], d['ub']))

    # Build layers from stored slabs and parameter map
    components = []
    for i, lname in enumerate(layer_names):
        thick_val = float(final_slabs[i, 0])
        sld_real  = float(final_slabs[i, 1])
        sld_imag  = float(final_slabs[i, 2])
        rough_val = float(final_slabs[i, 3])

        mat  = SLD(complex(sld_real, sld_imag), name=lname)
        slab = mat(thick_val, rough_val)

        _apply(slab.thick, f'{lname} - thick')
        _apply(slab.rough, f'{lname} - rough')
        _apply(mat.real,   f'{lname} - sld')
        _apply(mat.imag,   f'{lname} - isld')

        components.append(slab)

    structure = Structure()
    for c in components:
        structure |= c

    # Instrument parameters
    def _pval(key, default):
        return pmap[key]['value'] if key in pmap else default

    model = ReflectModel(
        structure,
        scale=_pval('scale', 1.0),
        bkg=_pval('bkg', 1e-7),
        dq=_pval('dq - resolution', 1.6),
        q_offset=_pval('q_offset', 0.0),
    )
    _apply(model.scale,    'scale')
    _apply(model.bkg,      'bkg')
    _apply(model.dq,       'dq - resolution')
    _apply(model.q_offset, 'q_offset')

    # Experimental data  — ReflectDataset([x, y, y_err]) takes a list of 1-D arrays
    valid_dR = (dR is not None and len(dR) == len(q) and not np.all(np.isnan(dR)))
    data = ReflectDataset([q, R, dR] if valid_dR else [q, R])

    objective = Objective(model, data, transform=Transform(transform))
    return objective, structure


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_batch_to_h5(batch_results, sample_name, model_name, filepath,
                     energy_list=None, run_index=None):
    """
    Save batch fitting results to an HDF5 file.

    Results are stored under:
        /{sample_name}/{energy}/{model_name}/run_{N}/

    Successive calls for the same sample/model auto-increment run_N, so
    multiple fitting rounds accumulate as run_0, run_1, etc.

    Parameters
    ----------
    batch_results : dict
        Return value of batch_fit_selected_models.
    sample_name : str
        Top-level sample label, e.g. 'Brewer1'.
    model_name : str
        Model label, e.g. 'Model1'.
    filepath : str or Path
        Path to .h5 file; opened in append mode ('a').
    energy_list : list of float, optional
        Energies to save; defaults to batch_results['fitted_energies'].
    run_index : int, optional
        Explicit run index; defaults to auto-increment per energy.
    """
    required = ['fitted_objectives', 'individual_results']
    missing = [k for k in required if k not in batch_results]
    if missing:
        raise ValueError(f"batch_results missing keys: {missing}")

    if energy_list is None:
        energies = batch_results.get(
            'fitted_energies',
            sorted(batch_results['fitted_objectives'].keys()))
    else:
        energies = energy_list

    fitted_objectives   = batch_results['fitted_objectives']
    original_objectives = batch_results.get('original_objectives', {})
    original_structures = batch_results.get('original_structures', {})
    individual_results  = batch_results.get('individual_results', {})
    str_dt = h5py.string_dtype()

    saved = 0
    with h5py.File(filepath, 'a') as f:
        for energy in energies:
            if energy not in fitted_objectives:
                print(f"  Warning: {energy} eV not in fitted_objectives — skipping.")
                continue

            fitted_obj = fitted_objectives[energy]
            energy_key = str(float(energy))
            model_path = f"{sample_name}/{energy_key}/{model_name}"

            # Auto-increment run index
            if model_path in f and run_index is None:
                existing = [int(k.split('_')[1]) for k in f[model_path].keys()
                            if k.startswith('run_')]
                cur_run = (max(existing) + 1) if existing else 0
            elif run_index is not None:
                cur_run = run_index
            else:
                cur_run = 0

            run_grp = f.require_group(f"{model_path}/run_{cur_run}")
            f[f"{sample_name}/{energy_key}"].attrs['energy_eV'] = float(energy)

            # Scalar metadata
            indiv = individual_results.get(energy, {})
            chi_init  = (indiv.get('initial_chi_squared', np.nan)
                         if isinstance(indiv, dict) else np.nan)
            chi_final = float(fitted_obj.chisqr())

            run_grp.attrs['chi_sq_initial'] = chi_init
            run_grp.attrs['chi_sq_final']   = chi_final
            run_grp.attrs['timestamp']       = datetime.datetime.now().isoformat()
            run_grp.attrs['run_index']       = cur_run
            run_grp.attrs['transform']       = _transform_form(fitted_obj)

            # Parameters
            pg = run_grp.require_group('parameters')
            f_names, f_vals, f_lb, f_ub, f_vary, f_stderr = _extract_param_arrays(fitted_obj)
            n = len(f_vals)

            pg.create_dataset('names',        data=np.array(f_names, dtype=object), dtype=str_dt)
            pg.create_dataset('final_values', data=f_vals)
            pg.create_dataset('final_lb',     data=f_lb)
            pg.create_dataset('final_ub',     data=f_ub)
            pg.create_dataset('final_vary',   data=f_vary)
            pg.create_dataset('stderr',       data=f_stderr)

            orig_obj = original_objectives.get(energy)
            if orig_obj is not None:
                _, i_vals, i_lb, i_ub, i_vary, _ = _extract_param_arrays(orig_obj)
                pg.create_dataset('initial_values', data=i_vals)
                pg.create_dataset('initial_lb',     data=i_lb)
                pg.create_dataset('initial_ub',     data=i_ub)
                pg.create_dataset('initial_vary',   data=i_vary)
            else:
                nan_arr = np.full(n, np.nan)
                pg.create_dataset('initial_values', data=nan_arr)
                pg.create_dataset('initial_lb',     data=nan_arr.copy())
                pg.create_dataset('initial_ub',     data=nan_arr.copy())
                pg.create_dataset('initial_vary',   data=np.full(n, -1, dtype=np.int8))

            # Experimental data
            dg = run_grp.require_group('data')
            dg.create_dataset('q', data=np.asarray(fitted_obj.data.x,     dtype=np.float64))
            dg.create_dataset('R', data=np.asarray(fitted_obj.data.y,     dtype=np.float64))
            y_err = fitted_obj.data.y_err
            dR_arr = (np.asarray(y_err, dtype=np.float64)
                      if y_err is not None
                      else np.full(len(fitted_obj.data.x), np.nan))
            dg.create_dataset('dR', data=dR_arr)

            # Structure slabs and layer names
            fitted_structure = fitted_obj.model.structure
            layer_names = _get_layer_names(fitted_structure)
            final_slabs = np.asarray(fitted_structure.slabs(), dtype=np.float64)

            run_grp.create_dataset('layer_names',
                                   data=np.array(layer_names, dtype=object),
                                   dtype=str_dt)
            run_grp.create_dataset('structure_slabs_final',   data=final_slabs)

            orig_struct = original_structures.get(energy)
            init_slabs = (np.asarray(orig_struct.slabs(), dtype=np.float64)
                          if orig_struct is not None
                          else np.full_like(final_slabs, np.nan))
            run_grp.create_dataset('structure_slabs_initial', data=init_slabs)

            saved += 1

    print(f"Saved {saved}/{len(energies)} energies → {filepath}")


def load_h5_objectives(filepath, sample_name, model_name,
                       criteria='best', energy_list=None):
    """
    Load and reconstruct refnx Objectives from an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    model_name : str
    criteria : 'best' | 'last' | int
        'best'  – run with the lowest chi_sq_final per energy
        'last'  – run with the highest run index
        int     – that specific run index
    energy_list : list of float, optional
        Subset of energies to load; defaults to all under sample_name.

    Returns
    -------
    objectives_dict : {float energy: Objective}
    structures_dict : {float energy: Structure}
    """
    objectives_dict = {}
    structures_dict = {}

    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]

        available = list(sample_grp.keys())
        if energy_list is not None:
            wanted = {str(float(e)) for e in energy_list}
            available = [k for k in available if k in wanted]

        for ekey in available:
            energy_val = float(ekey)
            energy_grp = sample_grp[ekey]

            if model_name not in energy_grp:
                continue
            model_grp = energy_grp[model_name]

            run_keys = sorted(
                [k for k in model_grp.keys() if k.startswith('run_')],
                key=lambda k: int(k.split('_')[1]))
            if not run_keys:
                continue

            if criteria == 'best':
                run_key = min(run_keys,
                              key=lambda k: model_grp[k].attrs.get('chi_sq_final', np.inf))
            elif criteria == 'last':
                run_key = run_keys[-1]
            elif isinstance(criteria, int):
                run_key = f'run_{criteria}'
                if run_key not in model_grp:
                    print(f"  {energy_val} eV: run_{criteria} not found — skipping.")
                    continue
            else:
                raise ValueError(
                    f"criteria must be 'best', 'last', or int — got {criteria!r}")

            rg = model_grp[run_key]
            transform = rg.attrs.get('transform', 'logY')

            pg = rg['parameters']
            param_names  = _decode_strings(pg['names'][:])
            final_values = pg['final_values'][:]
            final_lb     = pg['final_lb'][:]
            final_ub     = pg['final_ub'][:]
            final_vary   = pg['final_vary'][:]

            dg  = rg['data']
            q   = dg['q'][:]
            R   = dg['R'][:]
            dR  = dg['dR'][:] if 'dR' in dg else None

            layer_names = _decode_strings(rg['layer_names'][:])
            final_slabs = rg['structure_slabs_final'][:]

            try:
                obj, struct = _reconstruct_objective(
                    layer_names, final_slabs,
                    param_names, final_values, final_lb, final_ub, final_vary,
                    q, R, dR, transform=transform)
                objectives_dict[energy_val] = obj
                structures_dict[energy_val] = struct
            except Exception as exc:
                print(f"  Warning: could not reconstruct {energy_val} eV: {exc}")

    print(f"Loaded {len(objectives_dict)} objectives  "
          f"[{sample_name}/{model_name}, criteria={criteria!r}]  ← {filepath}")
    return objectives_dict, structures_dict


def _near_bound(value, lb, ub, tol_pct):
    """Return True if *value* is within *tol_pct*% of either finite bound.

    For a fully finite interval [lb, ub] the percentage is relative to the
    interval width.  For a one-sided bound it is relative to that bound's
    absolute value.  Infinite bounds are ignored.
    """
    if tol_pct <= 0:
        return False
    fin_lb, fin_ub = np.isfinite(lb), np.isfinite(ub)
    if fin_lb and fin_ub:
        span = ub - lb
        if span <= 0:
            return False
        return ((value - lb) / span < tol_pct / 100
                or (ub - value) / span < tol_pct / 100)
    elif fin_lb:
        ref = max(abs(lb), 1e-10)
        return abs(value - lb) / ref < tol_pct / 100
    elif fin_ub:
        ref = max(abs(ub), 1e-10)
        return abs(ub - value) / ref < tol_pct / 100
    return False


def plot_parameter_vs_energy(
    filepath,
    sample_name,
    param_name,
    model_names,
    criteria='best',
    energy_list=None,
    show_bounds=False,
    bound_tol_pct=10.0,
    figsize=None,
    xlim=None,
    ylim=None,
    ax=None,
):
    """
    Plot a fitted parameter value vs. energy from an HDF5 results file.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    param_name : str
        Parameter name as stored in HDF5, e.g. ``'PS - sld'``, ``'scale'``.
        Use :func:`get_h5_info` to inspect available names.
    model_names : str or list of str
        One or more model labels to compare on the same axes.
    criteria : 'best' | 'last' | int
        Run selection per energy. ``'best'`` (default) picks the run with the
        lowest ``chi_sq_final``.
    energy_list : list of float, optional
        Subset of energies to include. Defaults to all energies in the file.
    show_bounds : bool
        If True, overlay the parameter bounds as dashed lines with a shaded
        fill between them (only where both lb and ub are finite). Default False.
    bound_tol_pct : float
        Points within this percentage of either bound are overlaid in red.
        For a finite [lb, ub] interval the percentage is relative to
        (ub − lb); for a one-sided bound it is relative to that bound's
        absolute value.  Set to 0 to disable.  Default 10.
    figsize : (width, height), optional
    xlim : (xmin, xmax), optional
    ylim : (ymin, ymax), optional
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; if None a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if isinstance(model_names, str):
        model_names = [model_names]

    # ---- collect data from HDF5 (file closed before any plotting) ----------
    model_data = {}
    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]

        energy_keys = sorted(sample_grp.keys(), key=float)
        if energy_list is not None:
            wanted = {str(float(e)) for e in energy_list}
            energy_keys = [k for k in energy_keys if k in wanted]

        for mname in model_names:
            mdata = {'energies': [], 'values': [], 'lbs': [], 'ubs': [],
                     'near_bound': []}

            for ekey in energy_keys:
                energy_grp = sample_grp[ekey]
                if mname not in energy_grp:
                    continue
                model_grp = energy_grp[mname]

                run_keys = sorted(
                    [k for k in model_grp.keys() if k.startswith('run_')],
                    key=lambda k: int(k.split('_')[1]))
                if not run_keys:
                    continue

                if criteria == 'best':
                    rkey = min(
                        run_keys,
                        key=lambda k: model_grp[k].attrs.get('chi_sq_final', np.inf))
                elif criteria == 'last':
                    rkey = run_keys[-1]
                elif isinstance(criteria, int):
                    rkey = f'run_{criteria}'
                    if rkey not in model_grp:
                        continue
                else:
                    raise ValueError(
                        f"criteria must be 'best', 'last', or int — got {criteria!r}")

                pg = model_grp[rkey]['parameters']
                pnames = _decode_strings(pg['names'][:])
                if param_name not in pnames:
                    continue
                idx = pnames.index(param_name)

                val = float(pg['final_values'][idx])
                lb  = float(pg['final_lb'][idx])
                ub  = float(pg['final_ub'][idx])

                mdata['energies'].append(float(ekey))
                mdata['values'].append(val)
                mdata['lbs'].append(lb)
                mdata['ubs'].append(ub)
                mdata['near_bound'].append(_near_bound(val, lb, ub, bound_tol_pct))

            model_data[mname] = mdata

    # ---- plot ---------------------------------------------------------------
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    colors   = plt.rcParams['axes.prop_cycle'].by_key()['color']
    nb_label_used = False

    for m_idx, mname in enumerate(model_names):
        mdata = model_data[mname]
        if not mdata['energies']:
            print(f"  Warning: '{param_name}' not found for model '{mname}'.")
            continue

        color  = colors[m_idx % len(colors)]
        marker = markers[m_idx % len(markers)]

        E    = np.array(mdata['energies'])
        V    = np.array(mdata['values'])
        LB   = np.array(mdata['lbs'])
        UB   = np.array(mdata['ubs'])
        flag = np.array(mdata['near_bound'], dtype=bool)

        # Main line connecting all points in model colour
        ax.plot(E, V, marker=marker, linestyle='-', color=color,
                label=mname, zorder=2)

        # Near-bound points overlaid in red (single legend entry)
        if bound_tol_pct > 0 and np.any(flag):
            nb_label = '_nolegend_' if nb_label_used else 'near bound'
            ax.scatter(E[flag], V[flag], c='red', marker=marker,
                       zorder=3, s=80, label=nb_label)
            nb_label_used = True

        # Parameter bounds
        if show_bounds:
            fin_lb   = np.isfinite(LB)
            fin_ub   = np.isfinite(UB)
            both     = fin_lb & fin_ub
            b_label  = f'{mname} bounds'

            if np.any(fin_lb):
                ax.plot(E[fin_lb], LB[fin_lb], linestyle='--',
                        color=color, alpha=0.6, zorder=1, label=b_label)
                b_label = '_nolegend_'
            if np.any(fin_ub):
                ax.plot(E[fin_ub], UB[fin_ub], linestyle='--',
                        color=color, alpha=0.6, zorder=1, label=b_label)
            if np.any(both):
                ax.fill_between(E[both], LB[both], UB[both],
                                color=color, alpha=0.1, zorder=0)

    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel(param_name)
    ax.set_title(f'{sample_name}  —  {param_name}')
    ax.legend()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


def plot_reflectivity(
    filepath,
    sample_name,
    model_names,
    criteria='best',
    energy_list=None,
    energy_range=None,
    ncols=4,
    show_errorbars=True,
    yscale='log',
    figsize=None,
    xlim=None,
    ylim=None,
):
    """
    Plot simulated vs. experimental reflectivity for a selection of energies.

    Each energy gets its own panel arranged in a grid.  Multiple models appear
    as separate curves on the same panel alongside the experimental data.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    model_names : str or list of str
        One or more model labels to include on each panel.
    criteria : 'best' | 'last' | int
        Run selection per energy. Default 'best'.
    energy_list : list of float, optional
        Explicit energies to plot.  Takes precedence over *energy_range*.
    energy_range : (float, float), optional
        Inclusive ``(emin, emax)`` interval; all stored energies in range are
        plotted.  Ignored when *energy_list* is provided.
    ncols : int
        Columns in the subplot grid.  Default 4.
    show_errorbars : bool
        Plot experimental dR uncertainties as error bars when available.
        Default True.
    yscale : 'log' | 'linear'
        Y-axis scale.  Default 'log'.
    figsize : (width, height), optional
        Total figure size.  Defaults to ``(ncols * 4, nrows * 3.5)``.
    xlim : (xmin, xmax), optional
    ylim : (ymin, ymax), optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes  (one per plotted energy, in energy order)
    """
    import math
    import matplotlib.pyplot as plt

    if isinstance(model_names, str):
        model_names = [model_names]

    # ---- collect data from HDF5 (file closed before any plotting) ----------
    # panel_data: {energy_float: {'exp': (q, R, dR), 'models': {mname: R_sim}}}
    panel_data = {}

    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]

        all_keys = sorted(sample_grp.keys(), key=float)

        if energy_list is not None:
            wanted = {str(float(e)) for e in energy_list}
            energy_keys = [k for k in all_keys if k in wanted]
        elif energy_range is not None:
            emin, emax = float(energy_range[0]), float(energy_range[1])
            energy_keys = [k for k in all_keys if emin <= float(k) <= emax]
        else:
            energy_keys = all_keys

        for ekey in energy_keys:
            energy_val = float(ekey)
            energy_grp = sample_grp[ekey]
            entry = {'exp': None, 'models': {}}

            for mname in model_names:
                if mname not in energy_grp:
                    continue
                model_grp = energy_grp[mname]

                run_keys = sorted(
                    [k for k in model_grp.keys() if k.startswith('run_')],
                    key=lambda k: int(k.split('_')[1]))
                if not run_keys:
                    continue

                if criteria == 'best':
                    rkey = min(
                        run_keys,
                        key=lambda k: model_grp[k].attrs.get('chi_sq_final', np.inf))
                elif criteria == 'last':
                    rkey = run_keys[-1]
                elif isinstance(criteria, int):
                    rkey = f'run_{criteria}'
                    if rkey not in model_grp:
                        continue
                else:
                    raise ValueError(
                        f"criteria must be 'best', 'last', or int — got {criteria!r}")

                rg = model_grp[rkey]
                pg = rg['parameters']
                dg = rg['data']

                q  = dg['q'][:]
                R  = dg['R'][:]
                dR = dg['dR'][:] if 'dR' in dg else None

                # Experimental data is the same across models; store once
                if entry['exp'] is None:
                    entry['exp'] = (q, R, dR)

                # Reconstruct model and evaluate simulated reflectivity
                try:
                    obj, _ = _reconstruct_objective(
                        _decode_strings(rg['layer_names'][:]),
                        rg['structure_slabs_final'][:],
                        _decode_strings(pg['names'][:]),
                        pg['final_values'][:],
                        pg['final_lb'][:],
                        pg['final_ub'][:],
                        pg['final_vary'][:],
                        q, R, dR,
                        transform=rg.attrs.get('transform', 'logY'))
                    entry['models'][mname] = obj.model(q)
                except Exception as exc:
                    print(f"  Warning: could not reconstruct "
                          f"{energy_val} eV / {mname}: {exc}")

            if entry['exp'] is not None and entry['models']:
                panel_data[energy_val] = entry

    if not panel_data:
        raise ValueError("No data found for the specified energies / models.")

    # ---- subplot grid -------------------------------------------------------
    energies_sorted = sorted(panel_data.keys())
    n_panels = len(energies_sorted)
    ncols    = min(ncols, n_panels)
    nrows    = math.ceil(n_panels / ncols)

    if figsize is None:
        figsize = (ncols * 4, nrows * 3.5)

    if len(model_names) == 1:
        suptitle = f'{sample_name}  —  {model_names[0]}'
    else:
        suptitle = f'{sample_name}  —  ' + ' vs '.join(model_names)

    fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes_arr.flatten().tolist()

    # ---- plot ---------------------------------------------------------------
    colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']
    exp_color = 'black'
    legend_done = False

    for panel_idx, energy_val in enumerate(energies_sorted):
        ax = axes_flat[panel_idx]
        q, R, dR = panel_data[energy_val]['exp']

        valid_dR = (dR is not None
                    and len(dR) == len(q)
                    and not np.all(np.isnan(dR)))

        # Experimental data
        exp_label = 'Exp' if not legend_done else '_nolegend_'
        if show_errorbars and valid_dR:
            ax.errorbar(q, R, yerr=dR, fmt='o', color=exp_color,
                        ms=3, lw=0.8, capsize=2, label=exp_label)
        else:
            ax.plot(q, R, 'o', color=exp_color, ms=3, label=exp_label)

        # Simulated curves (one per model)
        for m_idx, mname in enumerate(model_names):
            if mname not in panel_data[energy_val]['models']:
                continue
            R_sim = panel_data[energy_val]['models'][mname]
            color = colors[m_idx % len(colors)]
            ax.plot(q, R_sim, '-', color=color, lw=1.5,
                    label=mname if not legend_done else '_nolegend_')

        ax.set_yscale(yscale)
        ax.set_title(f'{energy_val:.2f} eV', fontsize=9)
        ax.set_xlabel('q (Å⁻¹)', fontsize=8)
        ax.set_ylabel('R', fontsize=8)
        ax.tick_params(labelsize=7)

        if not legend_done:
            ax.legend(fontsize=7, loc='upper right')
            legend_done = True

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    # Hide unused axes in the last row
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig, axes_flat[:n_panels]


def get_h5_info(filepath, sample_name=None):
    """
    Inspect an HDF5 results file without loading array data.

    Returns
    -------
    dict  {sample: {energy_float: {model: [run_info_dict, ...]}}}

    Each run_info_dict contains:
        run_index, chi_sq_initial, chi_sq_final, timestamp, n_params, n_layers
    """
    info = {}
    with h5py.File(filepath, 'r') as f:
        sample_keys = [sample_name] if sample_name and sample_name in f else list(f.keys())
        for skey in sample_keys:
            if skey not in f:
                continue
            info[skey] = {}
            for ekey, energy_grp in f[skey].items():
                energy_val = float(ekey)
                info[skey][energy_val] = {}
                for mkey, model_grp in energy_grp.items():
                    runs = []
                    for rkey in sorted(model_grp.keys(),
                                       key=lambda k: int(k.split('_')[1])
                                       if k.startswith('run_') else -1):
                        if not rkey.startswith('run_'):
                            continue
                        rg = model_grp[rkey]
                        n_params = len(rg['parameters/names']) if 'parameters/names' in rg else 0
                        n_layers = len(rg['layer_names']) if 'layer_names' in rg else 0
                        runs.append({
                            'run_index':      rg.attrs.get('run_index',
                                                           int(rkey.split('_')[1])),
                            'chi_sq_initial': rg.attrs.get('chi_sq_initial', np.nan),
                            'chi_sq_final':   rg.attrs.get('chi_sq_final', np.nan),
                            'timestamp':      rg.attrs.get('timestamp', ''),
                            'n_params':       n_params,
                            'n_layers':       n_layers,
                        })
                    info[skey][energy_val][mkey] = runs
    return info
