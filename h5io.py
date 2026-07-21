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
        attrs: chi_sq_initial, chi_sq_final, timestamp, run_index, transform,
               has_mcmc (bool)
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
          ci_lower        float64   [n_params]   (NaN where unavail; 2.5th pct, MCMC only)
          ci_upper        float64   [n_params]   (NaN where unavail; 97.5th pct, MCMC only)
          rhat            float64   [n_params]   (NaN where unavail; Gelman-Rubin, MCMC only)
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

import os
import datetime
import numpy as np
import h5py

from refnx.reflect import SLD, Structure, ReflectModel
from refnx.analysis import Objective, Transform
from refnx.dataset import ReflectDataset

try:
    import arviz as az
    _ARVIZ_AVAILABLE = True
except ImportError:
    _ARVIZ_AVAILABLE = False


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


def _compute_mcmc_diagnostics(objective, param_names):
    """
    Compute 95% CI and Gelman-Rubin R-hat for each parameter from MCMC chains.

    Uses ``param.chain`` set on each Parameter by refnx's ``process_chain``
    after ``CurveFitter.sample()``.  Chain shape is ``(steps, nwalkers)`` for
    standard emcee.

    Parameters
    ----------
    objective : refnx.analysis.Objective
    param_names : list of str
        Names in the same order as ``_extract_param_arrays`` returns.

    Returns
    -------
    ci_lower, ci_upper, rhat : float64 arrays of length len(param_names)
        NaN for fixed or un-sampled parameters.
    """
    n = len(param_names)
    ci_lower = np.full(n, np.nan)
    ci_upper = np.full(n, np.nan)
    rhat     = np.full(n, np.nan)

    params = list(objective.parameters.flattened(unique=True))
    if len(params) != n:
        return ci_lower, ci_upper, rhat

    for i, p in enumerate(params):
        chain = getattr(p, 'chain', None)
        if not p.vary or chain is None or chain.size == 0:
            continue

        # Flatten all walkers/steps into a 1-D sample for percentiles
        flat = chain.ravel()
        lo, hi = np.percentile(flat, [2.5, 97.5])
        ci_lower[i] = lo
        ci_upper[i] = hi

        # R-hat requires the un-flattened (chains, draws) shape
        if _ARVIZ_AVAILABLE and chain.ndim == 2:
            # chain shape: (steps, nwalkers) → transpose to (nwalkers, steps)
            n_chains = chain.shape[1]
            if n_chains >= 2:
                try:
                    result = az.rhat({"x": chain.T})
                    rhat[i] = float(result["x"].values)
                except Exception:
                    pass
        elif chain.ndim > 2:
            # Parallel tempering — R-hat not straightforwardly defined here
            pass

    return ci_lower, ci_upper, rhat


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


def _is_energy_key(key):
    """Return True if *key* is a valid float string (energy group), not 'nexafs' etc."""
    try:
        float(key)
        return True
    except (ValueError, TypeError):
        return False


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
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
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

            # MCMC diagnostics — only written when sampling was performed
            _has_mcmc = any(
                p.vary
                and getattr(p, 'chain', None) is not None
                and p.chain.size > 0
                for p in fitted_obj.parameters.flattened(unique=True)
            )
            if _has_mcmc:
                ci_lo, ci_hi, rhat_vals = _compute_mcmc_diagnostics(fitted_obj, f_names)
                pg.create_dataset('ci_lower', data=ci_lo)
                pg.create_dataset('ci_upper', data=ci_hi)
                pg.create_dataset('rhat',     data=rhat_vals)
            run_grp.attrs['has_mcmc'] = _has_mcmc

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


def save_sweep_results_to_h5(energy_sweep_data, sample_name, model_name,
                              filepath, energy, run_criteria='best',
                              uncertainty_percent=10):
    """
    Attach parameter-sweep uncertainty results to an existing run in the HDF5 file.

    Results are written under the selected run_N group:
        /{sample_name}/{energy}/{model_name}/run_N/
            attrs: has_sweep, sweep_timestamp, sweep_uncertainty_percent
            /parameters/
                sweep_ci_lower  [n_params]  NaN for non-swept params
                sweep_ci_upper  [n_params]  NaN for non-swept params
            /sweep_results/
                /{param_key}/
                    attrs: param_name, best_value, best_gof, ci_lower, ci_upper,
                           gof_threshold, uncertainty_percent, n_sweep_points
                    sweep_values  [n_sweep_points]
                    gof_values    [n_sweep_points]

    Re-running overwrites any existing sweep_results subgroup for that run.

    Parameters
    ----------
    energy_sweep_data : dict
        {param_name: {'sweep_info': ..., 'ci_result': ...}}
        As produced by batch_fit_sweep for a single energy.
    sample_name : str
    model_name : str
    filepath : str or Path
        Existing .h5 file; opened in append mode.
    energy : float
    run_criteria : 'best' | 'last' | int
        Which run_N group to update.
    uncertainty_percent : float
        Threshold used for CI calculation (stored as metadata).
    """
    energy_key = str(float(energy))
    model_path = f"{sample_name}/{energy_key}/{model_name}"

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with h5py.File(filepath, 'a') as f:
        if model_path not in f:
            raise KeyError(
                f"No existing run found at '{model_path}' in {filepath}. "
                f"Call save_batch_to_h5 before saving sweep results."
            )

        model_grp = f[model_path]
        run_keys = sorted(
            [k for k in model_grp.keys() if k.startswith('run_')],
            key=lambda k: int(k.split('_')[1])
        )
        if not run_keys:
            raise KeyError(f"No run groups found at '{model_path}'.")

        if run_criteria == 'best':
            run_key = min(
                run_keys,
                key=lambda k: model_grp[k].attrs.get('chi_sq_final', np.inf))
        elif run_criteria == 'last':
            run_key = run_keys[-1]
        elif isinstance(run_criteria, int):
            run_key = f'run_{run_criteria}'
            if run_key not in model_grp:
                raise KeyError(
                    f"run_{run_criteria} not found at '{model_path}'.")
        else:
            raise ValueError(
                f"run_criteria must be 'best', 'last', or int — got {run_criteria!r}")

        run_grp = model_grp[run_key]

        # Update run-level attrs
        run_grp.attrs['has_sweep'] = True
        run_grp.attrs['sweep_timestamp'] = datetime.datetime.now().isoformat()
        run_grp.attrs['sweep_uncertainty_percent'] = float(uncertainty_percent)

        # Aligned CI arrays (NaN for non-swept params)
        pg = run_grp['parameters']
        param_names = _decode_strings(pg['names'][:])
        n = len(param_names)
        sweep_ci_lower = np.full(n, np.nan)
        sweep_ci_upper = np.full(n, np.nan)

        # Overwrite sweep_results subgroup so re-runs are clean
        if 'sweep_results' in run_grp:
            del run_grp['sweep_results']
        sweep_grp = run_grp.create_group('sweep_results')

        for pname, data in energy_sweep_data.items():
            sweep_info = data['sweep_info']
            ci_result  = data['ci_result']

            key = pname.replace(' ', '_').replace('-', '_')
            param_grp = sweep_grp.create_group(key)
            param_grp.attrs['param_name'] = pname

            param_grp.create_dataset(
                'sweep_values',
                data=np.array(sweep_info['parameter_values'], dtype=np.float64))
            param_grp.create_dataset(
                'gof_values',
                data=np.array(sweep_info['goodness_of_fit'], dtype=np.float64))

            ci_lo = float(ci_result['uncertainty_range'][0])
            ci_hi = float(ci_result['uncertainty_range'][1])
            param_grp.attrs['best_value']         = float(ci_result['best_value'])
            param_grp.attrs['best_gof']           = float(ci_result['best_gof'])
            param_grp.attrs['ci_lower']           = ci_lo
            param_grp.attrs['ci_upper']           = ci_hi
            param_grp.attrs['gof_threshold']      = float(ci_result['gof_threshold'])
            param_grp.attrs['uncertainty_percent'] = float(ci_result['uncertainty_percent'])
            param_grp.attrs['n_sweep_points']     = len(sweep_info['parameter_values'])

            if pname in param_names:
                idx = param_names.index(pname)
                sweep_ci_lower[idx] = ci_lo
                sweep_ci_upper[idx] = ci_hi

        # Write aligned CI datasets (overwrite if already present)
        for dset_name, arr in [('sweep_ci_lower', sweep_ci_lower),
                                ('sweep_ci_upper', sweep_ci_upper)]:
            if dset_name in pg:
                del pg[dset_name]
            pg.create_dataset(dset_name, data=arr)


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

        available = [k for k in sample_grp.keys() if _is_energy_key(k)]
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


def set_final_model(
    filepath,
    sample_name,
    base_model,
    overrides=None,
    final_model_name='Final',
    criteria='best',
    overwrite=False,
    verbose=True,
):
    """
    Assemble a composite 'Final' model in the HDF5 file from pieces of existing models.

    The *base_model* provides fitted parameters for every energy at which it
    has data.  Each entry in *overrides* then replaces parameters for a
    specific energy range or list with those from a different model.  Overrides
    are applied in order, so later entries take precedence over earlier ones.

    The resulting 'Final' (or *final_model_name*) group is written as a new
    sibling of the source models — one ``run_0`` per energy, copied verbatim
    from the winning source run.  Each copied run carries two extra attributes
    recording provenance: ``source_model`` and ``source_run``.  The original
    model data is never modified.

    Because the copy is verbatim, energies that come from models with more
    layers will have extra parameters (e.g. ``UL - sld``) that energies from
    simpler models lack — this is fine and intentional.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    base_model : str
        Model that provides the starting set of energies and parameters.
    overrides : list of dict, optional
        Each dict must contain ``'model'`` (str) plus either:

        * ``'energy_range': (emin, emax)`` — all stored energies in
          ``[emin, emax]`` (inclusive) are sourced from this model instead.
        * ``'energy_list': [e1, e2, ...]`` — specific energies to override.

        Example::

            overrides = [
                {'model': 'Model2', 'energy_range': (280, 282)},
                {'model': 'Model3', 'energy_list': [285.0]},
            ]

    final_model_name : str
        Name of the group to create.  Default ``'Final'``.
    criteria : 'best' | 'last' | int
        Run selection applied to each source model at each energy.
        Default ``'best'`` (lowest ``chi_sq_final``).
    overwrite : bool
        If True, delete any existing *final_model_name* group at each energy
        before writing.  If False (default), raise ``ValueError`` when the
        Final group already exists at any energy.
    verbose : bool
        Print a summary line per energy.  Default True.

    Returns
    -------
    dict
        ``{energy: source_model_name}`` — the winning source model for every
        energy written to the Final group.
    """
    from pathlib import Path as _Path

    filepath = str(_Path(filepath).expanduser())

    if overrides is None:
        overrides = []

    def _pick_run(model_grp):
        run_keys = sorted(
            [k for k in model_grp.keys() if k.startswith('run_')],
            key=lambda k: int(k.split('_')[1]))
        if not run_keys:
            return None
        if criteria == 'best':
            return min(run_keys,
                       key=lambda k: model_grp[k].attrs.get('chi_sq_final', np.inf))
        if criteria == 'last':
            return run_keys[-1]
        if isinstance(criteria, int):
            rkey = f'run_{criteria}'
            return rkey if rkey in model_grp else None
        raise ValueError(f"criteria must be 'best', 'last', or int — got {criteria!r}")

    provenance = {}

    with h5py.File(filepath, 'a') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]

        # All energies present in the file
        all_keys = sorted([k for k in sample_grp.keys() if _is_energy_key(k)], key=float)

        # Build energy → source_model mapping starting from base_model
        source_map = {}   # ekey (str) → model name
        for ekey in all_keys:
            if base_model in sample_grp[ekey]:
                source_map[ekey] = base_model

        # Apply overrides in order (later entries win)
        for ov in overrides:
            ov_model = ov['model']
            if 'energy_range' in ov:
                emin, emax = float(ov['energy_range'][0]), float(ov['energy_range'][1])
                affected = [k for k in all_keys if emin <= float(k) <= emax]
            elif 'energy_list' in ov:
                wanted = {str(float(e)) for e in ov['energy_list']}
                affected = [k for k in all_keys if k in wanted]
            else:
                raise ValueError(
                    f"Each override must have 'energy_range' or 'energy_list': {ov!r}")

            for ekey in affected:
                if ov_model in sample_grp[ekey]:
                    source_map[ekey] = ov_model
                else:
                    print(f"  Warning: '{ov_model}' not found at {ekey} eV — skipping override.")

        if not source_map:
            raise ValueError(
                f"No energies found for base model '{base_model}' in '{sample_name}'.")

        # Check for existing Final groups before writing anything
        if not overwrite:
            conflicts = [ekey for ekey in source_map
                         if final_model_name in sample_grp[ekey]]
            if conflicts:
                raise ValueError(
                    f"'{final_model_name}' already exists at energies: "
                    f"{[float(k) for k in conflicts]}. "
                    f"Pass overwrite=True to replace.")

        # Write Final model
        for ekey in sorted(source_map, key=float):
            src_model = source_map[ekey]
            energy_grp = sample_grp[ekey]
            model_grp  = energy_grp[src_model]

            src_rkey = _pick_run(model_grp)
            if src_rkey is None:
                print(f"  Warning: no runs found for '{src_model}' at {ekey} eV — skipping.")
                continue

            # Remove existing Final group at this energy if overwriting
            if final_model_name in energy_grp:
                del energy_grp[final_model_name]

            final_grp = energy_grp.require_group(final_model_name)

            # Copy the entire source run verbatim into Final/run_0
            f.copy(model_grp[src_rkey], final_grp, name='run_0')

            # Record provenance
            final_grp['run_0'].attrs['source_model'] = src_model
            final_grp['run_0'].attrs['source_run']   = src_rkey

            provenance[float(ekey)] = src_model
            if verbose:
                chi = model_grp[src_rkey].attrs.get('chi_sq_final', float('nan'))
                print(f"  {float(ekey):7.2f} eV  ←  {src_model}/{src_rkey}"
                      f"  (χ² = {chi:.4g})")

    n = len(provenance)
    n_models = len(set(provenance.values()))
    print(f"\n'{final_model_name}' written: {n} energies from {n_models} source model(s).")
    return provenance


def extract_sld_from_h5(
    filepath,
    sample_name,
    model_names,
    criteria='best',
    materials_filter=None,
    energy_list=None,
    energy_range=None,
    verbose=True,
    save_path=None,
):
    """
    Extract fitted SLD parameters from an HDF5 results file.

    Mirrors the output of ``Model_Setup.extract_sld_from_objectives`` so the
    two can be used interchangeably.  Parameters are filtered to only those
    whose name indicates an SLD or density value (same logic as
    ``get_param_type`` returning ``'sld_real'``, ``'sld_imag'``, or
    ``'density'``).

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    model_names : str or list of str
        One or more model labels to extract.  A ``model`` column is always
        included in the output so results from different models can be
        distinguished when multiple names are given.
    criteria : 'best' | 'last' | int
        Run selection per energy.  ``'best'`` (default) picks the run with
        the lowest ``chi_sq_final``.
    materials_filter : str or list of str, optional
        Only include parameters whose name contains at least one of these
        substrings (case-insensitive).  Matches the *materials_filter*
        argument of ``extract_sld_from_objectives``.  Default None = all SLD
        parameters.
    energy_list : list of float, optional
        Subset of energies to extract.  Default = all energies in the file.
    energy_range : (emin, emax), optional
        Inclusive energy interval.  Ignored when *energy_list* is given.
    verbose : bool
        Print progress.  Default True.
    save_path : str or Path, optional
        When provided, saves a CSV of ``[Energy_eV, Real_SLD, Imag_SLD]``
        in the same format as ``Model_Setup.save_material_sld``.  Works best
        when ``model_names`` is a single model and ``materials_filter``
        targets one material.  Skipped with a warning if multiple models are
        requested.

    Returns
    -------
    pandas.DataFrame
        Columns: ``energy``, ``model``, ``parameter``, ``value``,
        ``stderr``, ``vary``.  Compatible with ``plot_material_sld`` and
        ``save_material_sld`` from ``Model_Setup``.
    """
    import pandas as pd
    from pathlib import Path as _Path

    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(materials_filter, str):
        materials_filter = [materials_filter]

    filepath = str(_Path(filepath).expanduser())

    def _is_sld_param(name):
        n = name.lower()
        return 'isld' in n or 'sld' in n or '_density' in n

    rows = []

    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]

        all_keys = sorted([k for k in sample_grp.keys() if _is_energy_key(k)], key=float)
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

                pg     = model_grp[rkey]['parameters']
                pnames = _decode_strings(pg['names'][:])
                vals   = pg['final_values'][:]
                vary   = pg['final_vary'][:].astype(bool)
                stderr = pg['stderr'][:] if 'stderr' in pg else np.full(len(pnames), np.nan)

                for i, pname in enumerate(pnames):
                    if not _is_sld_param(pname):
                        continue
                    if materials_filter and not any(
                            m.lower() in pname.lower() for m in materials_filter):
                        continue
                    rows.append(dict(
                        energy=energy_val,
                        model=mname,
                        parameter=pname,
                        value=float(vals[i]),
                        stderr=float(stderr[i]) if np.isfinite(stderr[i]) else None,
                        vary=bool(vary[i]),
                    ))

            if verbose:
                print(f"Processed {energy_val} eV ({len(rows)} params so far)")

    df = pd.DataFrame(rows)

    # ---- optional save ------------------------------------------------------
    if save_path is not None:
        if len(model_names) > 1:
            print("  Warning: save_path is ignored when multiple models are "
                  "requested — save separately per model.")
        else:
            import os
            from pathlib import Path as _Path2
            sub = df.copy()
            real = sub[sub['parameter'].str.lower().str.contains('sld') &
                        ~sub['parameter'].str.lower().str.contains('isld')]
            imag = sub[sub['parameter'].str.lower().str.contains('isld')]
            real = real[['energy', 'value']].rename(columns={'value': 'Real_SLD'})
            imag = imag[['energy', 'value']].rename(columns={'value': 'Imag_SLD'})
            merged = real.merge(imag, on='energy').sort_values('energy')
            arr = merged[['energy', 'Real_SLD', 'Imag_SLD']].to_numpy(dtype=float)

            mat_label = ('+'.join(materials_filter)
                         if materials_filter else 'SLD')
            sp = str(_Path2(save_path).expanduser())
            if not sp.endswith('.csv'):
                sp += '.csv'
            os.makedirs(os.path.dirname(sp) or '.', exist_ok=True)
            header = 'Energy_eV,Real_SLD,Imag_SLD'
            np.savetxt(sp, arr, delimiter=',', header=header, comments='')
            print(f'Saved {mat_label} SLD ({len(arr)} energies) → {sp}')

    return df


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
    show_gof=False,
    reference_data=None,
    reference_label='Reference',
    nexafs_spectrum=None,
    nexafs_component='real',
    nexafs_label=None,
    refl_energy_shift=0.0,
    nexafs_energy_shift=0.0,
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
    show_gof : bool
        If True, add a linked subplot below the main axes showing chi_sq_final
        vs. energy for each model (x-axes are shared so ticks align).
        A new figure is always created when show_gof is True; the *ax* argument
        is ignored in that case.  Default False.
    reference_data : array-like (N, 2), str, Path, or DataFrame, optional
        Reference values overlaid on the main plot as a dashed black line.
        Accepted formats:
        * str or Path — CSV file; first column energy, second column value.
        * ndarray / list — shape (N, 2), column 0 energy, column 1 value.
        * pandas DataFrame — first two columns used.
        Default None (no reference line).
    reference_label : str
        Legend label for the reference line.  Default ``'Reference'``.
    nexafs_spectrum : str, optional
        Name of a NEXAFS spectrum stored in the same HDF5 file under
        ``/{sample_name}/nexafs/``.  If given, the selected SLD component is
        overlaid on the main plot.  Default None (no overlay).
    nexafs_component : 'real' | 'imag'
        Which SLD component to plot.  Default ``'real'``.
    nexafs_label : str, optional
        Legend label for the NEXAFS line.  Defaults to
        ``'{nexafs_spectrum} (real SLD)'`` or ``'{nexafs_spectrum} (imag SLD)'``.
    refl_energy_shift : float
        Energy offset in eV added to the fitted reflectivity data points before
        plotting.  Positive values shift points to higher energies.  Default 0.
    nexafs_energy_shift : float
        Energy offset in eV added to the NEXAFS SLD overlay curve before
        plotting.  Default 0.
    figsize : (width, height), optional
        Total figure size.  When show_gof is True defaults to ``(10, 6)``.
    xlim : (xmin, xmax), optional
    ylim : (ymin, ymax), optional
        Applied to the main (parameter) axes only.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; ignored when show_gof is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The main parameter axes.  Returned when show_gof is False.
    (ax_main, ax_gof) : tuple of matplotlib.axes.Axes
        Main axes and goodness-of-fit axes.  Returned when show_gof is True.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from pathlib import Path as _Path

    if isinstance(model_names, str):
        model_names = [model_names]

    # ---- parse reference data -----------------------------------------------
    ref_E = ref_V = None
    if reference_data is not None:
        if isinstance(reference_data, (str, _Path)):
            import pandas as _pd
            _df = _pd.read_csv(reference_data)
            ref_E = _df.iloc[:, 0].to_numpy(dtype=float)
            ref_V = _df.iloc[:, 1].to_numpy(dtype=float)
        elif hasattr(reference_data, 'iloc'):
            ref_E = reference_data.iloc[:, 0].to_numpy(dtype=float)
            ref_V = reference_data.iloc[:, 1].to_numpy(dtype=float)
        else:
            _arr  = np.asarray(reference_data, dtype=float)
            ref_E, ref_V = _arr[:, 0], _arr[:, 1]

    # ---- collect data from HDF5 (file closed before any plotting) ----------
    model_data = {}
    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]

        energy_keys = sorted([k for k in sample_grp.keys() if _is_energy_key(k)], key=float)
        if energy_list is not None:
            wanted = {str(float(e)) for e in energy_list}
            energy_keys = [k for k in energy_keys if k in wanted]

        for mname in model_names:
            mdata = {'energies': [], 'values': [], 'lbs': [], 'ubs': [],
                     'near_bound': [], 'chi_sq': [],
                     'ci_lower': [], 'ci_upper': []}

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

                rg = model_grp[rkey]
                pg = rg['parameters']
                pnames = _decode_strings(pg['names'][:])
                if param_name not in pnames:
                    continue
                idx = pnames.index(param_name)

                val   = float(pg['final_values'][idx])
                lb    = float(pg['final_lb'][idx])
                ub    = float(pg['final_ub'][idx])
                ci_lo = ci_hi = np.nan
                if 'ci_lower' in pg:
                    ci_lo = float(pg['ci_lower'][idx])
                    ci_hi = float(pg['ci_upper'][idx]) if 'ci_upper' in pg else np.nan
                if np.isnan(ci_lo) and 'sweep_ci_lower' in pg:
                    ci_lo = float(pg['sweep_ci_lower'][idx])
                    ci_hi = float(pg['sweep_ci_upper'][idx]) if 'sweep_ci_upper' in pg else np.nan

                mdata['energies'].append(float(ekey))
                mdata['values'].append(val)
                mdata['lbs'].append(lb)
                mdata['ubs'].append(ub)
                mdata['near_bound'].append(_near_bound(val, lb, ub, bound_tol_pct))
                mdata['chi_sq'].append(float(rg.attrs.get('chi_sq_final', np.nan)))
                mdata['ci_lower'].append(ci_lo)
                mdata['ci_upper'].append(ci_hi)

            model_data[mname] = mdata

        # NEXAFS SLD overlay — read while the file is still open
        nexafs_E = nexafs_V = None
        if nexafs_spectrum is not None:
            if nexafs_component not in ('real', 'imag'):
                raise ValueError(
                    f"nexafs_component must be 'real' or 'imag' — got {nexafs_component!r}")
            npath = f'{sample_name}/nexafs/{nexafs_spectrum}/sld'
            if npath in f:
                sg       = f[npath]
                comp_key = 'sld_real' if nexafs_component == 'real' else 'sld_imag'
                nexafs_E = sg['energy'][:]
                nexafs_V = sg[comp_key][:]
            else:
                print(f"  Warning: NEXAFS '{nexafs_spectrum}' not found or SLD not "
                      f"computed in {filepath} — overlay skipped.")

    # ---- figure / axes setup ------------------------------------------------
    if show_gof:
        if figsize is None:
            figsize = (10, 6)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_gof  = fig.add_subplot(gs[1], sharex=ax_main)
        plt.setp(ax_main.get_xticklabels(), visible=False)
    else:
        if ax is None:
            _, ax_main = plt.subplots(figsize=figsize)
        else:
            ax_main = ax
        ax_gof = None

    # ---- main panel ---------------------------------------------------------
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

        E     = np.array(mdata['energies']) + refl_energy_shift
        V     = np.array(mdata['values'])
        LB    = np.array(mdata['lbs'])
        UB    = np.array(mdata['ubs'])
        flag  = np.array(mdata['near_bound'], dtype=bool)
        CI_LO = np.array(mdata['ci_lower'])
        CI_HI = np.array(mdata['ci_upper'])
        has_ci = np.isfinite(CI_LO) & np.isfinite(CI_HI)

        if np.any(has_ci):
            yerr = np.array([
                np.where(has_ci, V - CI_LO, 0.0),
                np.where(has_ci, CI_HI - V, 0.0),
            ])
            ax_main.errorbar(E, V, yerr=yerr,
                             marker=marker, linestyle='-', color=color,
                             label=mname, zorder=2,
                             capsize=3, capthick=1, elinewidth=1)
        else:
            ax_main.plot(E, V, marker=marker, linestyle='-', color=color,
                         label=mname, zorder=2)

        if bound_tol_pct > 0 and np.any(flag):
            nb_label = '_nolegend_' if nb_label_used else 'near bound'
            ax_main.scatter(E[flag], V[flag], c='red', marker=marker,
                            zorder=3, s=80, label=nb_label)
            nb_label_used = True

        if show_bounds:
            fin_lb  = np.isfinite(LB)
            fin_ub  = np.isfinite(UB)
            both    = fin_lb & fin_ub
            b_label = f'{mname} bounds'

            if np.any(fin_lb):
                ax_main.plot(E[fin_lb], LB[fin_lb], linestyle='--',
                             color=color, alpha=0.6, zorder=1, label=b_label)
                b_label = '_nolegend_'
            if np.any(fin_ub):
                ax_main.plot(E[fin_ub], UB[fin_ub], linestyle='--',
                             color=color, alpha=0.6, zorder=1, label=b_label)
            if np.any(both):
                ax_main.fill_between(E[both], LB[both], UB[both],
                                     color=color, alpha=0.1, zorder=0)

        # GoF subplot — same colour/marker as main, no duplicate legend needed
        if ax_gof is not None:
            chi = np.array(mdata['chi_sq'])
            ax_gof.plot(E, chi, marker=marker, linestyle='-', color=color,
                        label=mname if len(model_names) > 1 else '_nolegend_',
                        zorder=2)

    # Reference line
    if ref_E is not None:
        ax_main.plot(ref_E, ref_V, 'k--', lw=1.5, label=reference_label, zorder=1)

    # NEXAFS SLD overlay
    if nexafs_E is not None:
        _comp_str = 'real SLD' if nexafs_component == 'real' else 'imag SLD'
        _nx_label = nexafs_label or f'{nexafs_spectrum} ({_comp_str})'
        ax_main.plot(nexafs_E + nexafs_energy_shift, nexafs_V,
                     linestyle='-.', color='dimgray', lw=1.5,
                     label=_nx_label, zorder=1)

    ax_main.set_ylabel(param_name)
    ax_main.set_title(f'{sample_name}  —  {param_name}')
    ax_main.legend()

    if xlim is not None:
        ax_main.set_xlim(xlim)
    if ylim is not None:
        ax_main.set_ylim(ylim)

    # Build xlabel — append shift notes on a second line when active
    _shift_notes = []
    if refl_energy_shift != 0:
        _shift_notes.append(f'reflectivity data shifted {refl_energy_shift:+.3g} eV')
    if nexafs_energy_shift != 0:
        _shift_notes.append(f'NEXAFS shifted {nexafs_energy_shift:+.3g} eV')
    _xlabel = ('Energy (eV)\n' + ',  '.join(_shift_notes)
               if _shift_notes else 'Energy (eV)')

    if ax_gof is not None:
        ax_gof.set_xlabel(_xlabel)
        ax_gof.set_ylabel('χ²', fontsize=9)
        ax_gof.tick_params(labelsize=8)
        if len(model_names) > 1:
            ax_gof.legend(fontsize=7)
        if xlim is not None:
            ax_gof.set_xlim(xlim)
        return ax_main, ax_gof

    ax_main.set_xlabel(_xlabel)
    return ax_main


# ---------------------------------------------------------------------------
# Private helpers for plot_material_comparison
# ---------------------------------------------------------------------------

def _plot_material_comparison_plotly(
        model_data, model_names, material_name, sample_name,
        show_pct_diff, figsize, xlim):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.colors as pc

    gf_title = 'Δχ²/min(χ²) [%]' if show_pct_diff else 'χ² (open = better fit)'
    subplot_titles = ['SLD', 'iSLD', 'Thickness', 'Roughness', gf_title]
    ylabels = [
        f'{material_name} - sld (×10⁻⁶ Å⁻²)',
        f'{material_name} - isld (×10⁻⁶ Å⁻²)',
        f'{material_name} - thick (Å)',
        f'{material_name} - rough (Å)',
        gf_title,
    ]

    # 5-row × 1-col grid with shared x-axis so GF aligns with parameter points
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=['SLD', 'iSLD', 'Thickness', 'Roughness', gf_title],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )
    colors       = pc.qualitative.Plotly
    base_markers = ['circle', 'square']
    param_keys   = ['sld', 'isld', 'thick', 'rough']

    # ---- main 4 parameter panels ----------------------------------------
    for m_idx, mname in enumerate(model_names):
        color      = colors[m_idx % len(colors)]
        marker_sym = base_markers[m_idx % len(base_markers)]
        mdata      = model_data[mname]
        if len(mdata['energies']) == 0:
            print(f"  Warning: no data found for model '{mname}'.")
            continue
        E = mdata['energies'].tolist()

        for row, pkey in enumerate(param_keys, start=1):
            V = mdata[pkey].tolist()
            fig.add_trace(
                go.Scatter(
                    x=E, y=V,
                    mode='lines+markers',
                    name=mname,
                    legendgroup=mname,
                    showlegend=(row == 1),
                    marker=dict(symbol=marker_sym, color=color, size=8),
                    line=dict(color=color),
                ),
                row=row, col=1,
            )

    # ---- GF panel (row 5) -----------------------------------------------
    mname_A, mname_B = model_names
    dA, dB = model_data[mname_A], model_data[mname_B]
    eA_map = {e: i for i, e in enumerate(dA['energies'].tolist())}
    eB_map = {e: i for i, e in enumerate(dB['energies'].tolist())}

    if show_pct_diff:
        common_e = sorted(set(eA_map) & set(eB_map))
        if common_e:
            chi_A    = np.array([dA['chi_sq'][eA_map[e]] for e in common_e])
            chi_B    = np.array([dB['chi_sq'][eB_map[e]] for e in common_e])
            pct_diff = (chi_A - chi_B) / np.minimum(chi_A, chi_B) * 100
            fig.add_trace(
                go.Scatter(
                    x=common_e, y=pct_diff.tolist(),
                    mode='lines+markers',
                    name='Δχ²%',
                    marker=dict(symbol='diamond', color='purple', size=8),
                    line=dict(color='purple'),
                ),
                row=5, col=1,
            )
            fig.add_hline(y=0, line_dash='dash', line_color='gray', row=5, col=1)
    else:
        for m_idx, (mname, mdata, mmap, other_map, other_data) in enumerate([
            (mname_A, dA, eA_map, eB_map, dB),
            (mname_B, dB, eB_map, eA_map, dA),
        ]):
            color    = colors[m_idx % len(colors)]
            base_sym = base_markers[m_idx % len(base_markers)]
            E   = mdata['energies'].tolist()
            chi = mdata['chi_sq'].tolist()

            symbols = []
            for e, c in zip(E, chi):
                if e in other_map:
                    other_c = float(other_data['chi_sq'][other_map[e]])
                    symbols.append(f'{base_sym}-open' if c <= other_c else base_sym)
                else:
                    symbols.append(base_sym)

            fig.add_trace(
                go.Scatter(
                    x=E, y=chi,
                    mode='lines+markers',
                    name=mname,
                    legendgroup=mname,
                    showlegend=False,
                    marker=dict(symbol=symbols, color=color, size=8),
                    line=dict(color=color),
                ),
                row=5, col=1,
            )

    # ---- axis labels --------------------------------------------------------
    for row, ylabel in enumerate(ylabels, start=1):
        fig.update_yaxes(title_text=ylabel, row=row, col=1)
        if xlim is not None:
            fig.update_xaxes(range=list(xlim), row=row, col=1)
    fig.update_xaxes(title_text='Energy (eV)', row=5, col=1)

    w, h = figsize if figsize else (700, 1100)
    fig.update_layout(
        width=w, height=h,
        title_text=f'{sample_name}  —  {material_name}  (model comparison)',
    )
    return fig


def _plot_material_comparison_mpl(
        model_data, model_names, material_name, sample_name,
        show_pct_diff, figsize, xlim):
    import matplotlib.pyplot as plt

    gf_title = 'Δχ²/min(χ²) [%]' if show_pct_diff else 'χ² (open = better fit)'
    titles   = ['SLD', 'iSLD', 'Thickness', 'Roughness', gf_title]
    ylabels  = [
        f'{material_name} - sld (×10⁻⁶ Å⁻²)',
        f'{material_name} - isld (×10⁻⁶ Å⁻²)',
        f'{material_name} - thick (Å)',
        f'{material_name} - rough (Å)',
        gf_title,
    ]

    fig, axes = plt.subplots(5, 1, figsize=figsize or (8, 14), sharex=True)
    colors      = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mpl_markers = ['o', 's']
    param_keys  = ['sld', 'isld', 'thick', 'rough']

    # ---- main 4 parameter panels ----------------------------------------
    for m_idx, mname in enumerate(model_names):
        color  = colors[m_idx % len(colors)]
        marker = mpl_markers[m_idx % len(mpl_markers)]
        mdata  = model_data[mname]
        if len(mdata['energies']) == 0:
            print(f"  Warning: no data found for model '{mname}'.")
            continue
        E = mdata['energies']
        for p_idx, pkey in enumerate(param_keys):
            axes[p_idx].plot(E, mdata[pkey], marker=marker, linestyle='-',
                             color=color, label=mname)

    # ---- 5th panel: GF or % diff ----------------------------------------
    mname_A, mname_B = model_names
    dA, dB = model_data[mname_A], model_data[mname_B]
    eA_map = {e: i for i, e in enumerate(dA['energies'].tolist())}
    eB_map = {e: i for i, e in enumerate(dB['energies'].tolist())}

    if show_pct_diff:
        common_e = sorted(set(eA_map) & set(eB_map))
        if common_e:
            chi_A    = np.array([dA['chi_sq'][eA_map[e]] for e in common_e])
            chi_B    = np.array([dB['chi_sq'][eB_map[e]] for e in common_e])
            pct_diff = (chi_A - chi_B) / np.minimum(chi_A, chi_B) * 100
            axes[4].plot(common_e, pct_diff, 'D-', color='purple',
                         label=f'Δχ²%  ({mname_A} vs {mname_B})')
            axes[4].axhline(0, color='gray', linestyle='--', lw=1)
    else:
        for m_idx, (mname, mdata, mmap, other_map, other_data) in enumerate([
            (mname_A, dA, eA_map, eB_map, dB),
            (mname_B, dB, eB_map, eA_map, dA),
        ]):
            color  = colors[m_idx % len(colors)]
            marker = mpl_markers[m_idx % len(mpl_markers)]
            E   = mdata['energies']
            chi = mdata['chi_sq']

            is_better = np.array([
                e in other_map and chi[i] <= float(other_data['chi_sq'][other_map[e]])
                for i, e in enumerate(E.tolist())
            ], dtype=bool)

            axes[4].plot(E, chi, linestyle='-', color=color, label='_nolegend_')
            if np.any(is_better):
                axes[4].plot(E[is_better], chi[is_better],
                             marker=marker, linestyle='', color=color,
                             markerfacecolor='none', markersize=8,
                             label=f'{mname} (better)')
            if np.any(~is_better):
                axes[4].plot(E[~is_better], chi[~is_better],
                             marker=marker, linestyle='', color=color,
                             markerfacecolor=color, markersize=8,
                             label=f'{mname} (worse)')

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        if xlim is not None:
            ax.set_xlim(xlim)
    axes[-1].set_xlabel('Energy (eV)', fontsize=9)

    fig.suptitle(f'{sample_name}  —  {material_name}  (model comparison)')
    fig.tight_layout()
    return axes


def plot_material_comparison(
    filepath,
    sample_name,
    material_name,
    model_names,
    criteria='best',
    energy_list=None,
    show_pct_diff=False,
    figsize=None,
    xlim=None,
    interactive=True,
):
    """
    Compare two fitted models side-by-side for all parameters of one material.

    Produces five panels: SLD, iSLD, thickness, roughness (one per fitted
    parameter), and a goodness-of-fit panel. In the GF panel the model with
    the lower chi_sq at each energy is shown with an **open** marker; the
    worse-fitting model uses a filled marker. Set ``show_pct_diff=True`` to
    replace the GF panel with the percentage difference in chi_sq between
    the two models.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    material_name : str
        Material prefix as stored in HDF5 parameter names, e.g. ``'MOX'``
        reads ``'MOX - sld'``, ``'MOX - isld'``, ``'MOX - thick'``,
        ``'MOX - rough'``.
    model_names : list of str
        Exactly two model labels to compare.
    criteria : 'best' | 'last' | int
        Run selection per energy.  ``'best'`` (default) picks the run with
        the lowest ``chi_sq_final``.
    energy_list : list of float, optional
        Subset of energies to include.  Defaults to all energies in the file.
    show_pct_diff : bool
        If True, the 5th panel shows the percentage difference in chi_sq:
        ``(chi_A − chi_B) / min(chi_A, chi_B) × 100``.  Positive means
        model A fits worse.  Default False.
    figsize : (width, height), optional
        Figure size.  Defaults to ``(700, 1100)`` pixels for plotly or
        ``(8, 14)`` inches for matplotlib.
    xlim : (xmin, xmax), optional
        Applied to all subplots.
    interactive : bool
        If True (default) returns a plotly Figure with zoom/pan.
        If False returns an ndarray of 5 matplotlib Axes.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Returned when ``interactive=True``.
    axes : ndarray of matplotlib.axes.Axes, shape (5,)
        Returned when ``interactive=False``.
    """
    import h5py
    from pathlib import Path as _Path

    if len(model_names) != 2:
        raise ValueError(
            f"plot_material_comparison requires exactly 2 model names, "
            f"got {len(model_names)}: {model_names!r}")

    filepath = str(_Path(filepath).expanduser())
    param_suffixes = [
        ('sld',   f'{material_name} - sld'),
        ('isld',  f'{material_name} - isld'),
        ('thick', f'{material_name} - thick'),
        ('rough', f'{material_name} - rough'),
    ]

    # ---- collect data from HDF5 (file closed before any plotting) ----------
    model_data = {
        mname: {'energies': [], 'sld': [], 'isld': [],
                'thick': [], 'rough': [], 'chi_sq': []}
        for mname in model_names
    }

    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]

        energy_keys = sorted(
            [k for k in sample_grp.keys() if _is_energy_key(k)], key=float)
        if energy_list is not None:
            wanted = {str(float(e)) for e in energy_list}
            energy_keys = [k for k in energy_keys if k in wanted]

        for mname in model_names:
            mdata = model_data[mname]
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

                rg           = model_grp[rkey]
                pg           = rg['parameters']
                pnames       = _decode_strings(pg['names'][:])
                final_values = pg['final_values'][:]

                mdata['energies'].append(float(ekey))
                for key, pname in param_suffixes:
                    mdata[key].append(
                        float(final_values[pnames.index(pname)])
                        if pname in pnames else np.nan)
                mdata['chi_sq'].append(float(rg.attrs.get('chi_sq_final', np.nan)))

    for mname in model_names:
        d = model_data[mname]
        for k in d:
            d[k] = np.array(d[k])

    if interactive:
        return _plot_material_comparison_plotly(
            model_data, model_names, material_name, sample_name,
            show_pct_diff, figsize, xlim)
    return _plot_material_comparison_mpl(
        model_data, model_names, material_name, sample_name,
        show_pct_diff, figsize, xlim)


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

        all_keys = sorted([k for k in sample_grp.keys() if _is_energy_key(k)], key=float)

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


def plot_stacked_reflectivity_h5(
    filepath,
    sample_name,
    model_name,
    energies_to_plot=None,
    criteria='best',
    spacing=10,
    colormap='viridis',
    figsize=(10, 12),
    title=None,
    show_legend=True,
    sim_color=None,
    label_color='black',
    label_fontsize=9,
    label_fontfamily=None,
    label_fontweight='bold',
    save_path=None,
    save_dir=None,
    save_name=None,
    save_format='png',
    save_dpi=300,
    save_bbox_inches='tight',
    show_save_ui=False,
):
    """
    Stacked reflectivity plot (exp + sim) sourced from an HDF5 results file.

    Drop-in equivalent of ``plot_stacked_reflectivity`` for HDF5 data.
    Experimental data and fitted parameters are read directly from the file;
    the simulated curve is reconstructed from the stored layer structure and
    final parameter values.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    model_name : str
        Model label to plot (e.g. ``'Model1'`` or ``'Final'``).
    energies_to_plot : None | list of float | (emin, emax) | list of (emin, emax)
        * ``None`` — all energies in the file.
        * ``[e1, e2, ...]`` — explicit energy values.
        * ``(emin, emax)`` — inclusive energy range.
        * ``[(emin1, emax1), (emin2, emax2), ...]`` — union of ranges.
    criteria : 'best' | 'last' | int
        Run selection per energy.  Default ``'best'``.
    spacing : float
        Multiplicative offset between successive curves.  Default 10.
    colormap : str
        Matplotlib colormap name.  Default ``'viridis'``.
    figsize : (width, height)
        Figure size in inches.  Default ``(10, 12)``.
    title : str, optional
        Plot title.  Defaults to a generic stacked-reflectivity title.
    show_legend : bool
        Show legend.  Default True.
    sim_color : color, optional
        Fixed colour for all simulated lines.  Default: same as exp colour.
    label_color : str
        Colour of the inline energy labels.  Default ``'black'``.
    label_fontsize : int
        Font size for energy labels.  Default 9.
    label_fontfamily : str, optional
        Font family for energy labels.
    label_fontweight : str
        Font weight for energy labels.  Default ``'bold'``.
    save_path : str, optional
        Full path to save the figure directly.
    save_dir : str, optional
        Directory for save_dir + save_name saves.
    save_name : str, optional
        Filename (extension optional) for save_dir + save_name saves.
    save_format : str
        Format used when save_name has no extension.  Default ``'png'``.
    save_dpi : int
        DPI for raster formats.  Default 300.
    save_bbox_inches : str
        Passed to ``fig.savefig``.  Default ``'tight'``.
    show_save_ui : bool
        Display an ipywidgets save UI after the figure.  Default False.

    Returns
    -------
    (fig, ax) : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from pathlib import Path as _Path

    filepath = str(_Path(filepath).expanduser())

    # ---- collect data from HDF5 (file closed before plotting) --------------
    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]
        all_keys = sorted([k for k in sample_grp.keys() if _is_energy_key(k)], key=float)
        all_energies = [float(k) for k in all_keys]

        # --- energy selection (mirrors plot_stacked_reflectivity logic) -----
        if energies_to_plot is None:
            selected_energies = all_energies
        elif (isinstance(energies_to_plot, (list, tuple))
              and len(energies_to_plot) == 2
              and isinstance(energies_to_plot[0], (int, float))):
            emin, emax = float(energies_to_plot[0]), float(energies_to_plot[1])
            selected_energies = [e for e in all_energies if emin <= e <= emax]
            if not selected_energies:
                raise ValueError(f"No energies found in range ({emin}, {emax}).")
        else:
            energies_to_plot = list(energies_to_plot)
            if energies_to_plot and isinstance(energies_to_plot[0], (list, tuple)):
                selected_energies = []
                for rng in energies_to_plot:
                    mn, mx = float(rng[0]), float(rng[1])
                    selected_energies.extend(e for e in all_energies if mn <= e <= mx)
                selected_energies = sorted(set(selected_energies))
                if not selected_energies:
                    raise ValueError("No energies found in the specified ranges.")
            else:
                wanted = {float(e) for e in energies_to_plot}
                selected_energies = [e for e in all_energies if e in wanted]
                missing = wanted - set(selected_energies)
                if missing:
                    print(f"  Warning: energies not found in file: {sorted(missing)}")
                if not selected_energies:
                    raise ValueError("None of the specified energies were found.")

        selected_energies = sorted(selected_energies)

        # --- reconstruct exp + sim for each selected energy ------------------
        panel_data = {}
        ekey_map = {float(k): k for k in all_keys}

        for energy in selected_energies:
            ekey = ekey_map.get(energy)
            if ekey is None or model_name not in sample_grp[ekey]:
                print(f"  Warning: '{model_name}' not found at {energy} eV — skipping.")
                continue
            model_grp = sample_grp[ekey][model_name]

            run_keys = sorted(
                [k for k in model_grp.keys() if k.startswith('run_')],
                key=lambda k: int(k.split('_')[1]))
            if not run_keys:
                continue

            if criteria == 'best':
                rkey = min(run_keys,
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
                panel_data[energy] = (q, R, obj.model(q))
            except Exception as exc:
                print(f"  Warning: could not reconstruct {energy} eV: {exc}")

    if not panel_data:
        raise ValueError("No data could be loaded for the selected energies / model.")

    selected_energies = sorted(panel_data.keys())
    n_energies = len(selected_energies)

    # ---- colours and offsets (identical to plot_stacked_reflectivity) -------
    offsets = [spacing ** (n_energies - 1 - i) for i in range(n_energies)]

    try:
        cmap = cm.get_cmap(colormap)
    except ValueError:
        raise ValueError(f"Colormap '{colormap}' not found.")

    color_values = [0.5] if n_energies == 1 else [i / (n_energies - 1) for i in range(n_energies)]
    colors = [cmap(v) for v in color_values]

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    for i, energy in enumerate(selected_energies):
        q, R_exp, R_sim = panel_data[energy]
        offset = offsets[i]
        color  = colors[i]

        ax.plot(q, R_exp * offset, 'o',
                markerfacecolor='none', markeredgecolor=color,
                markersize=4, label=f'{energy} eV')
        ax.plot(q, R_sim * offset, '-',
                color=sim_color if sim_color is not None else color,
                linewidth=1.5)

    ax.set_yscale('log')
    ax.set_xlabel(r'$q$ ($\AA^{-1}$)', fontsize=16)
    ax.set_ylabel('Reflectivity (a.u.)', fontsize=16)
    ax.set_title(title or 'Stacked Reflectivity: Experimental vs Simulated', fontsize=14)
    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    # inline energy labels at right end of simulated curve
    for i, energy in enumerate(selected_energies):
        q, _, R_sim = panel_data[energy]
        offset = offsets[i]
        end_q  = q[np.argmax(q)]
        end_R  = R_sim[np.argmax(q)] * offset

        text_kw = dict(verticalalignment='center', horizontalalignment='left',
                       fontsize=label_fontsize, color=label_color, weight=label_fontweight)
        if label_fontfamily is not None:
            text_kw['fontfamily'] = label_fontfamily
        ax.text(end_q, end_R, f' {energy} eV', **text_kw)

    max_q_all = max(panel_data[e][0].max() for e in selected_energies)
    cur_xlim  = ax.get_xlim()
    ax.set_xlim(cur_xlim[0], max(cur_xlim[1], max_q_all * 1.15))

    plt.tight_layout()

    # ---- saving (identical helpers to plot_stacked_reflectivity) ------------
    def _norm_name(name):
        if name is None or str(name).strip() == '':
            return None
        name = str(name).strip()
        if os.path.splitext(name)[1] == '':
            name = f'{name}.{save_format}'
        return name

    def _save_to(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        fig.savefig(path, dpi=save_dpi, bbox_inches=save_bbox_inches)
        print(f'Saved → {os.path.abspath(path)}')
        return path

    if save_path is not None:
        _save_to(os.path.abspath(save_path))
    else:
        nname = _norm_name(save_name)
        if save_dir is not None and nname is not None:
            _save_to(os.path.abspath(os.path.join(save_dir, nname)))

    if show_save_ui:
        import ipywidgets as widgets
        from IPython.display import display as _display

        start_dir  = os.path.abspath(save_dir or os.getcwd())
        dir_text   = widgets.Text(value=start_dir, description='Dir:',
                                  layout=widgets.Layout(width='600px'))
        refresh_btn = widgets.Button(description='Refresh folders')
        folder_dd  = widgets.Dropdown(description='Folder:',
                                      layout=widgets.Layout(width='600px'))
        name_text  = widgets.Text(
            value=_norm_name(save_name) or f'stacked_reflectivity.{save_format}',
            description='File:', layout=widgets.Layout(width='600px'))
        save_btn   = widgets.Button(description='Save figure', button_style='success')
        out        = widgets.Output()

        def _list_folders(base):
            try:
                base = os.path.abspath(os.path.expanduser(base))
                return sorted(p for n in os.listdir(base)
                              if os.path.isdir(p := os.path.join(base, n)))
            except Exception:
                return []

        def refresh(_=None):
            folders = _list_folders(dir_text.value)
            folder_dd.options = folders
            if folders:
                folder_dd.value = folders[0]

        def on_save(_):
            chosen = folder_dd.value or dir_text.value
            fname  = _norm_name(name_text.value)
            with out:
                out.clear_output()
                if not fname:
                    print('Please enter a filename.')
                    return
                try:
                    _save_to(os.path.join(chosen, fname))
                except Exception as e:
                    print(f'Save failed: {e}')

        refresh_btn.on_click(refresh)
        save_btn.on_click(on_save)
        refresh()

        _display(widgets.VBox([
            widgets.HBox([dir_text, refresh_btn]),
            folder_dd, name_text, save_btn, out,
        ]))

    return fig, ax


def plot_reflectivity_comparison(
    filepath,
    sample_name,
    model_names,
    criteria='best',
    energy_list=None,
    energy_range=None,
    figsize=None,
    xlim=None,
    ylim=None,
):
    """
    Interactive two-panel widget comparing two model fits across energies.

    Returns an ``ipywidgets.VBox`` containing:

    * **GF panel** — chi_sq vs energy for both models (full plotly zoom/pan).
      Open marker = better fit at that energy, same convention as
      ``plot_material_comparison``.
    * **Energy selector** — ``SelectMultiple`` list.  Each entry is annotated
      with which model fits better at that energy.  Ctrl/Cmd-click or
      Shift-click to select multiple energies.
    * **Reflectivity panel** — updates automatically when the selection
      changes, showing exp data (black dots), Model A (solid line), and
      Model B (dashed line).  Each selected energy gets its own colour.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    model_names : list of str
        Exactly two model labels to compare.
    criteria : 'best' | 'last' | int
        Run selection per energy.  Default ``'best'``.
    energy_list : list of float, optional
        Explicit energies to include.  Takes precedence over *energy_range*.
    energy_range : (emin, emax), optional
        Inclusive energy interval.  Ignored when *energy_list* is provided.
    figsize : (width, height), optional
        Reflectivity panel size in pixels.  Defaults to ``(700, 480)``.
        The GF panel always spans the full width (``width + 220`` px).
    xlim : (xmin, xmax), optional
        q-axis limits on the reflectivity panel.
    ylim : (ymin, ymax), optional
        R-axis limits on the reflectivity panel (linear values).

    Returns
    -------
    widget : ipywidgets.VBox
        Display with the variable name alone on the last line of a Jupyter
        cell, or call ``IPython.display.display(widget)``.
    """
    import ipywidgets as widgets
    import plotly.graph_objects as go
    import plotly.colors as pc
    from pathlib import Path as _Path

    if isinstance(model_names, str):
        model_names = [model_names]
    if len(model_names) != 2:
        raise ValueError(
            f"plot_reflectivity_comparison requires exactly 2 model names, "
            f"got {len(model_names)}: {model_names!r}")
    mname_A, mname_B = model_names

    filepath = str(_Path(filepath).expanduser())

    # ---- collect data from HDF5 (file closed before any plotting) ----------
    panel_data = {}

    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        sample_grp = f[sample_name]

        all_keys = sorted([k for k in sample_grp.keys() if _is_energy_key(k)], key=float)
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
            entry = {'q': None, 'R_exp': None, 'dR': None,
                     'R_A': None, 'R_B': None,
                     'chi_A': np.nan, 'chi_B': np.nan}

            for mname, r_key, chi_key in [(mname_A, 'R_A', 'chi_A'),
                                           (mname_B, 'R_B', 'chi_B')]:
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

                if entry['q'] is None:
                    entry['q']     = q
                    entry['R_exp'] = R
                    entry['dR']    = dR

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
                    entry[r_key]   = obj.model(q)
                    entry[chi_key] = float(rg.attrs.get('chi_sq_final', np.nan))
                except Exception as exc:
                    print(f"  Warning: could not reconstruct "
                          f"{energy_val} eV / {mname}: {exc}")

            if entry['q'] is not None:
                panel_data[energy_val] = entry

    if not panel_data:
        raise ValueError("No data found for the specified energies / models.")

    energies = sorted(panel_data.keys())
    colors   = pc.qualitative.Plotly
    # persistent colour per energy index so colours stay stable across selections
    e_colors = {e: colors[i % len(colors)] for i, e in enumerate(energies)}

    refl_w, refl_h = figsize if figsize else (700, 480)
    gf_w = refl_w + 220  # GF spans the full widget width

    # ---- GF figure (always visible, full plotly interactivity) --------------
    chi_A_arr = np.array([panel_data[e]['chi_A'] for e in energies])
    chi_B_arr = np.array([panel_data[e]['chi_B'] for e in energies])

    def _sym(base, chi_self, chi_other):
        if np.isnan(chi_self) or np.isnan(chi_other):
            return base
        return f'{base}-open' if chi_self <= chi_other else base

    syms_A = [_sym('circle', chi_A_arr[i], chi_B_arr[i]) for i in range(len(energies))]
    syms_B = [_sym('square', chi_B_arr[i], chi_A_arr[i]) for i in range(len(energies))]

    gf_fig = go.Figure()
    gf_fig.add_trace(go.Scatter(
        x=energies, y=chi_A_arr.tolist(),
        mode='lines+markers', name=mname_A,
        marker=dict(symbol=syms_A, color=colors[0], size=10),
        line=dict(color=colors[0]),
    ))
    gf_fig.add_trace(go.Scatter(
        x=energies, y=chi_B_arr.tolist(),
        mode='lines+markers', name=mname_B,
        marker=dict(symbol=syms_B, color=colors[1], size=10),
        line=dict(color=colors[1]),
    ))
    gf_fig.update_layout(
        width=gf_w, height=280,
        title_text=(f'{sample_name}  —  {mname_A} vs {mname_B}'
                    '  |  open marker = better fit'),
        xaxis_title='Energy (eV)',
        yaxis_title='χ²',
        margin=dict(t=50, b=40),
    )

    gf_out = widgets.Output()
    with gf_out:
        gf_fig.show()

    # ---- Energy selector ----------------------------------------------------
    selector_options = []
    for e in energies:
        cA, cB = panel_data[e]['chi_A'], panel_data[e]['chi_B']
        if np.isnan(cA) and np.isnan(cB):
            tag = ''
        elif np.isnan(cB) or cA <= cB:
            tag = f'  [{mname_A}✓]'
        else:
            tag = f'  [{mname_B}✓]'
        selector_options.append((f'{e:.2f} eV{tag}', e))

    selector = widgets.SelectMultiple(
        options=selector_options,
        rows=min(18, len(energies)),
        layout=widgets.Layout(width='210px'),
    )
    selector_box = widgets.VBox([
        widgets.Label('Select energies  (Ctrl/Shift for multi):'),
        selector,
    ])

    # ---- Reflectivity output ------------------------------------------------
    refl_out = widgets.Output(
        layout=widgets.Layout(width=f'{refl_w + 20}px'))

    def _make_refl_fig(selected):
        fig = go.Figure()
        for energy in sorted(selected):
            entry   = panel_data[energy]
            e_color = e_colors[energy]
            e_label = f'{energy:.2f} eV'
            q       = entry['q']
            R_exp   = entry['R_exp']
            dR      = entry['dR']
            valid_dR = (dR is not None
                        and len(dR) == len(q)
                        and not np.all(np.isnan(dR)))

            exp_kw = dict(mode='markers', marker=dict(color='black', size=4),
                          name=f'{e_label} exp', legendgroup=e_label,
                          legendgrouptitle_text=e_label)
            if valid_dR:
                fig.add_trace(go.Scatter(
                    x=q.tolist(), y=R_exp.tolist(),
                    error_y=dict(type='data', array=dR.tolist(),
                                 visible=True, thickness=0.8, width=2),
                    **exp_kw))
            else:
                fig.add_trace(go.Scatter(
                    x=q.tolist(), y=R_exp.tolist(), **exp_kw))

            R_A = entry['R_A']
            if R_A is not None:
                fig.add_trace(go.Scatter(
                    x=q.tolist(), y=R_A.tolist(), mode='lines',
                    line=dict(color=e_color, width=2),
                    name=f'{e_label} {mname_A}', legendgroup=e_label))

            R_B = entry['R_B']
            if R_B is not None:
                fig.add_trace(go.Scatter(
                    x=q.tolist(), y=R_B.tolist(), mode='lines',
                    line=dict(color=e_color, width=2, dash='dash'),
                    name=f'{e_label} {mname_B}', legendgroup=e_label))

        fig.update_yaxes(type='log', title_text='R')
        fig.update_xaxes(title_text='q (Å⁻¹)')
        fig.update_layout(width=refl_w, height=refl_h,
                          legend=dict(groupclick='toggleitem'),
                          margin=dict(t=30))
        if xlim is not None:
            fig.update_xaxes(range=list(xlim))
        if ylim is not None:
            fig.update_yaxes(
                range=[np.log10(float(ylim[0])), np.log10(float(ylim[1]))])
        return fig

    def _on_select(change):
        with refl_out:
            refl_out.clear_output(wait=True)
            if selector.value:
                _make_refl_fig(selector.value).show()

    selector.observe(_on_select, names='value')

    return widgets.VBox([
        gf_out,
        widgets.HBox([selector_box, refl_out]),
    ])


# ---------------------------------------------------------------------------
# NEXAFS private helpers
# ---------------------------------------------------------------------------

def _parse_nexafs_input(nexafs_data):
    """
    Normalise *nexafs_data* to (energy_arr, intensity_arr, source_str).

    Accepted input types mirror those used by *reference_data* elsewhere:
    * str / Path → CSV file, first two columns used
    * DataFrame  → first two columns used
    * array-like → shape (N, 2), column 0 = energy, column 1 = intensity
    """
    from pathlib import Path as _Path
    if isinstance(nexafs_data, (str, _Path)):
        import pandas as _pd
        _df = _pd.read_csv(nexafs_data)
        return (_df.iloc[:, 0].to_numpy(dtype=float),
                _df.iloc[:, 1].to_numpy(dtype=float),
                str(nexafs_data))
    elif hasattr(nexafs_data, 'iloc'):
        return (nexafs_data.iloc[:, 0].to_numpy(dtype=float),
                nexafs_data.iloc[:, 1].to_numpy(dtype=float),
                '')
    else:
        arr = np.asarray(nexafs_data, dtype=float)
        return arr[:, 0], arr[:, 1], ''


def _write_sld_deltabeta(spec_grp, DeltaBeta, SLD,
                          chemical_formula, density, x_min, x_max):
    """Write fresh /sld/ and /deltabeta/ sub-groups into an open HDF5 group."""
    sld_g = spec_grp.create_group('sld')
    sld_g.attrs['chemical_formula'] = str(chemical_formula)
    sld_g.attrs['density']          = float(density)
    sld_g.attrs['x_min']            = float(x_min) if x_min is not None else np.nan
    sld_g.attrs['x_max']            = float(x_max) if x_max is not None else np.nan
    sld_g.create_dataset('energy',     data=SLD[:, 0].astype(np.float64))
    sld_g.create_dataset('sld_real',   data=SLD[:, 1].astype(np.float64))
    sld_g.create_dataset('sld_imag',   data=SLD[:, 2].astype(np.float64))
    sld_g.create_dataset('wavelength', data=SLD[:, 3].astype(np.float64))

    db_g = spec_grp.create_group('deltabeta')
    db_g.attrs['chemical_formula'] = str(chemical_formula)
    db_g.attrs['density']          = float(density)
    db_g.attrs['x_min']            = float(x_min) if x_min is not None else np.nan
    db_g.attrs['x_max']            = float(x_max) if x_max is not None else np.nan
    db_g.create_dataset('energy', data=DeltaBeta[:, 0].astype(np.float64))
    db_g.create_dataset('delta',  data=DeltaBeta[:, 1].astype(np.float64))
    db_g.create_dataset('beta',   data=DeltaBeta[:, 2].astype(np.float64))


# ---------------------------------------------------------------------------
# NEXAFS public API
# ---------------------------------------------------------------------------

def save_nexafs_to_h5(filepath, sample_name, nexafs_data, spectrum_name,
                      chemical_formula, density,
                      x_min=None, x_max=None,
                      beamline=None, date=None, method=None,
                      overwrite=False):
    """
    Save a NEXAFS spectrum to HDF5 and auto-compute its SLD and DeltaBeta.

    The spectrum is stored under::

        /{sample_name}/nexafs/{spectrum_name}/

    which sits alongside the fitting-results energy groups and never conflicts
    with them (``"nexafs"`` is not a valid float string).

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    nexafs_data : str, Path, array-like (N, 2), or DataFrame
        Raw NEXAFS data.  If a file path is given it is passed directly to
        ``process_nexafs_to_SLD``; otherwise the data is stacked into an
        (N, 2) array that ``calculate_refractive_index`` accepts.
    spectrum_name : str
        Unique label for this spectrum under the sample, e.g. ``'SOC_UV'``.
    chemical_formula : str
        Chemical formula used for the KK transform, e.g. ``'C8H8'``.
    density : float
        Density in g/cc used for the KK transform.
    x_min, x_max : float, optional
        Energy window passed to ``process_nexafs_to_SLD``.
    beamline, date, method : str, optional
        Freeform metadata stored as HDF5 attributes.
    overwrite : bool
        If True, replace an existing spectrum with the same name.
        Default False (raises ValueError on conflict).
    """
    energy_arr, intensity_arr, source_str = _parse_nexafs_input(nexafs_data)
    spec_path = f'{sample_name}/nexafs/{spectrum_name}'

    with h5py.File(filepath, 'a') as f:
        if spec_path in f:
            if not overwrite:
                raise ValueError(
                    f"Spectrum '{spectrum_name}' already exists under "
                    f"'{sample_name}/nexafs/'.  Use overwrite=True to replace.")
            del f[spec_path]

        spec_grp = f.require_group(spec_path)
        spec_grp.attrs['timestamp']   = datetime.datetime.now().isoformat()
        spec_grp.attrs['source_file'] = source_str
        spec_grp.attrs['beamline']    = beamline or ''
        spec_grp.attrs['date']        = date     or ''
        spec_grp.attrs['method']      = method   or ''

        raw_g = spec_grp.create_group('raw')
        raw_g.create_dataset('energy',    data=energy_arr.astype(np.float64))
        raw_g.create_dataset('intensity', data=intensity_arr.astype(np.float64))

        # Lazy import — avoids hard dependency on kkcalc at module load time
        from NEXAFS import process_nexafs_to_SLD
        input_data = (source_str
                      if source_str
                      else np.column_stack([energy_arr, intensity_arr]))
        DeltaBeta, SLD = process_nexafs_to_SLD(
            input_data, chemical_formula, density, x_min, x_max)

        _write_sld_deltabeta(spec_grp, DeltaBeta, SLD,
                             chemical_formula, density, x_min, x_max)

    print(f"Saved NEXAFS '{spectrum_name}' → {filepath}")
    print(f"  raw: {len(energy_arr)} pts | sld: {len(DeltaBeta)} pts | "
          f"formula={chemical_formula}, density={density} g/cc")


def update_nexafs_sld(filepath, sample_name, spectrum_name,
                      chemical_formula, density,
                      x_min=None, x_max=None):
    """
    Recompute and overwrite the SLD / DeltaBeta for an existing NEXAFS spectrum.

    The raw data stored in the file is reused; only the /sld/ and /deltabeta/
    sub-groups are replaced.  The spectrum's other metadata (beamline, date,
    etc.) is unchanged.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    spectrum_name : str
    chemical_formula : str
        New chemical formula for the KK transform.
    density : float
        New density in g/cc.
    x_min, x_max : float, optional
        Energy window for the KK transform.  Pass None to use the full range.
    """
    spec_path = f'{sample_name}/nexafs/{spectrum_name}'

    with h5py.File(filepath, 'a') as f:
        if spec_path not in f:
            raise KeyError(
                f"Spectrum '{spectrum_name}' not found under "
                f"'{sample_name}/nexafs/'.  Save it first with save_nexafs_to_h5.")

        spec_grp      = f[spec_path]
        energy_arr    = spec_grp['raw/energy'][:]
        intensity_arr = spec_grp['raw/intensity'][:]

        for sub in ('sld', 'deltabeta'):
            if sub in spec_grp:
                del spec_grp[sub]

        from NEXAFS import process_nexafs_to_SLD
        arr = np.column_stack([energy_arr, intensity_arr])
        DeltaBeta, SLD = process_nexafs_to_SLD(
            arr, chemical_formula, density, x_min, x_max)

        _write_sld_deltabeta(spec_grp, DeltaBeta, SLD,
                             chemical_formula, density, x_min, x_max)

    print(f"Updated SLD for '{spectrum_name}' in {filepath}")
    print(f"  formula={chemical_formula}, density={density} g/cc | "
          f"{len(DeltaBeta)} pts")


def load_nexafs_from_h5(filepath, sample_name, spectrum_name=None):
    """
    Load NEXAFS data from an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
    sample_name : str
    spectrum_name : str, optional
        Name of a specific spectrum.  If None, all spectra are returned.

    Returns
    -------
    dict
        When *spectrum_name* is given: a single result dict.
        When *spectrum_name* is None: ``{name: result_dict}`` for all spectra.

    Each result dict has the keys::

        {
          'spectrum_name': str,
          'raw':           {'energy': ndarray, 'intensity': ndarray},
          'sld':           {'energy', 'sld_real', 'sld_imag', 'wavelength'},
          'deltabeta':     {'energy', 'delta', 'beta'},
          'metadata':      {'chemical_formula', 'density', 'x_min', 'x_max',
                            'beamline', 'date', 'method',
                            'timestamp', 'source_file'},
        }

    ``sld`` and ``deltabeta`` are None if not yet computed.
    """
    def _read_spectrum(spec_grp, name):
        result = {'spectrum_name': name,
                  'raw': None, 'sld': None, 'deltabeta': None, 'metadata': {}}

        if 'raw' in spec_grp:
            result['raw'] = {
                'energy':    spec_grp['raw/energy'][:],
                'intensity': spec_grp['raw/intensity'][:],
            }

        if 'sld' in spec_grp:
            sg = spec_grp['sld']
            result['sld'] = {
                'energy':     sg['energy'][:],
                'sld_real':   sg['sld_real'][:],
                'sld_imag':   sg['sld_imag'][:],
                'wavelength': sg['wavelength'][:],
            }

        if 'deltabeta' in spec_grp:
            dg = spec_grp['deltabeta']
            result['deltabeta'] = {
                'energy': dg['energy'][:],
                'delta':  dg['delta'][:],
                'beta':   dg['beta'][:],
            }

        meta = {k: spec_grp.attrs.get(k, '')
                for k in ('beamline', 'date', 'method', 'timestamp', 'source_file')}
        sld_src = spec_grp.get('sld') or spec_grp.get('deltabeta')
        if sld_src is not None:
            meta['chemical_formula'] = sld_src.attrs.get('chemical_formula', '')
            meta['density']          = float(sld_src.attrs.get('density', np.nan))
            meta['x_min']            = float(sld_src.attrs.get('x_min',    np.nan))
            meta['x_max']            = float(sld_src.attrs.get('x_max',    np.nan))
        else:
            meta.update({'chemical_formula': '', 'density': np.nan,
                         'x_min': np.nan, 'x_max': np.nan})
        result['metadata'] = meta
        return result

    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}.")
        nexafs_path = f'{sample_name}/nexafs'
        if nexafs_path not in f:
            raise KeyError(f"No NEXAFS data found under '{sample_name}' in {filepath}.")
        nexafs_grp = f[nexafs_path]

        if spectrum_name is not None:
            if spectrum_name not in nexafs_grp:
                raise KeyError(
                    f"Spectrum '{spectrum_name}' not found under "
                    f"'{sample_name}/nexafs/'.")
            return _read_spectrum(nexafs_grp[spectrum_name], spectrum_name)

        return {name: _read_spectrum(nexafs_grp[name], name)
                for name in nexafs_grp.keys()}


def list_nexafs_spectra(filepath, sample_name):
    """
    List stored NEXAFS spectra for a sample without loading array data.

    Returns
    -------
    list of dict
        One dict per spectrum with keys:
        ``spectrum_name, chemical_formula, density, x_min, x_max,
        beamline, date, method, timestamp, source_file,
        n_points_raw, n_points_sld``
    """
    results = []
    with h5py.File(filepath, 'r') as f:
        nexafs_path = f'{sample_name}/nexafs'
        if nexafs_path not in f:
            return results
        for name, spec_grp in f[nexafs_path].items():
            entry = {'spectrum_name': name}
            for k in ('beamline', 'date', 'method', 'timestamp', 'source_file'):
                entry[k] = spec_grp.attrs.get(k, '')

            sld_src = spec_grp.get('sld') or spec_grp.get('deltabeta')
            if sld_src is not None:
                entry['chemical_formula'] = sld_src.attrs.get('chemical_formula', '')
                entry['density']          = float(sld_src.attrs.get('density', np.nan))
                entry['x_min']            = float(sld_src.attrs.get('x_min',    np.nan))
                entry['x_max']            = float(sld_src.attrs.get('x_max',    np.nan))
                entry['n_points_sld'] = (len(sld_src['energy'])
                                         if 'energy' in sld_src else 0)
            else:
                entry.update({'chemical_formula': '', 'density': np.nan,
                              'x_min': np.nan, 'x_max': np.nan, 'n_points_sld': 0})

            entry['n_points_raw'] = (len(spec_grp['raw/energy'])
                                     if 'raw/energy' in spec_grp else 0)
            results.append(entry)
    return results


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
                if not _is_energy_key(ekey):
                    continue
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
                            'has_mcmc':       bool(rg.attrs.get('has_mcmc', False)),
                        })
                    info[skey][energy_val][mkey] = runs
    return info


def print_parameters(filepath, sample_name, energy, model_name=None, run_index=-1,
                      tol_eV=0.5, criteria=None):
    """
    Print fitted parameters stored in an HDF5 results file.

    Parameters
    ----------
    filepath    : str   – path to the .h5 file
    sample_name : str   – top-level sample group
    energy      : float – energy in eV to look up (matched to nearest key within tol_eV)
    model_name  : str or None – if None, print all models at that energy
    run_index   : int   – which run to display; -1 means the last run (default).
                          Ignored if `criteria` is given.
    tol_eV      : float – how close the energy key must be (default 0.5 eV)
    criteria    : 'best' | 'last' | int | None
                          Run-selection override, same convention as
                          `load_h5_objectives`/`extract_sld_from_h5`:
                          'best' picks the run with the lowest chi_sq_final,
                          'last' the highest run index, an int an explicit
                          run index. Default None keeps the legacy
                          `run_index`-based selection.
    """
    with h5py.File(filepath, 'r') as f:
        if sample_name not in f:
            raise KeyError(f"Sample '{sample_name}' not found in {filepath}")
        sgrp = f[sample_name]

        energy_keys = [k for k in sgrp.keys() if _is_energy_key(k)]
        if not energy_keys:
            raise KeyError(f"No energy groups found under '{sample_name}'")
        closest = min(energy_keys, key=lambda k: abs(float(k) - energy))
        if abs(float(closest) - energy) > tol_eV:
            raise KeyError(
                f"No energy within {tol_eV} eV of {energy}. "
                f"Available: {sorted(float(k) for k in energy_keys)}"
            )
        egrp = sgrp[closest]
        print(f"\n{'='*60}")
        print(f"  Sample : {sample_name}   Energy : {float(closest):.1f} eV")
        print(f"{'='*60}")

        model_keys = [model_name] if model_name else list(egrp.keys())
        for mkey in model_keys:
            if mkey not in egrp:
                print(f"  Model '{mkey}' not found at this energy.")
                continue
            mgrp = egrp[mkey]
            run_keys = sorted(
                [k for k in mgrp.keys() if k.startswith('run_')],
                key=lambda k: int(k.split('_')[1])
            )
            if not run_keys:
                print(f"  Model '{mkey}': no runs found.")
                continue
            if criteria is not None:
                if criteria == 'best':
                    rkey = min(
                        run_keys,
                        key=lambda k: mgrp[k].attrs.get('chi_sq_final', np.inf))
                elif criteria == 'last':
                    rkey = run_keys[-1]
                elif isinstance(criteria, int):
                    rkey = f'run_{criteria}'
                    if rkey not in mgrp:
                        print(f"  Model '{mkey}': run_{criteria} not found.")
                        continue
                else:
                    raise ValueError(
                        f"criteria must be 'best', 'last', or int — got {criteria!r}")
            else:
                rkey = run_keys[run_index]
            rgrp = mgrp[rkey]

            chi_i = rgrp.attrs.get('chi_sq_initial', np.nan)
            chi_f = rgrp.attrs.get('chi_sq_final',   np.nan)
            has_mcmc = bool(rgrp.attrs.get('has_mcmc', False))
            print(f"\n  Model : {mkey}   Run : {rkey}")
            print(f"  chi² initial={chi_i:.4g}   final={chi_f:.4g}"
                  + ("   [MCMC]" if has_mcmc else ""))
            print()

            pg = rgrp['parameters']
            names  = _decode_strings(pg['names'][:])
            values = pg['final_values'][:]
            lb     = pg['final_lb'][:]
            ub     = pg['final_ub'][:]
            vary   = pg['final_vary'][:]
            stderr = pg['stderr'][:]
            ci_lo  = pg['ci_lower'][:] if 'ci_lower' in pg else np.full(len(names), np.nan)
            ci_hi  = pg['ci_upper'][:] if 'ci_upper' in pg else np.full(len(names), np.nan)

            col_w = max(len(n) for n in names) + 2
            header = f"  {'Parameter':<{col_w}}  {'Value':>14}  {'Stderr':>10}  {'Bounds':>22}  Vary"
            if has_mcmc:
                header += "   95% CI"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for name, val, se, lo, hi, v, clo, chi in zip(
                    names, values, stderr, lb, ub, vary, ci_lo, ci_hi):
                lo_str = f"{lo:.4g}" if np.isfinite(lo) else "-inf"
                hi_str = f"{hi:.4g}" if np.isfinite(hi) else "+inf"
                se_str = f"{se:.4g}" if np.isfinite(se) else "n/a"
                line = (f"  {name:<{col_w}}  {val:>14.6g}  {se_str:>10}  "
                        f"[{lo_str}, {hi_str}]  {'yes' if v else 'no ':>3}")
                if has_mcmc:
                    if np.isfinite(clo) and np.isfinite(chi):
                        line += f"   [{clo:.4g}, {chi:.4g}]"
                    else:
                        line += "   n/a"
                print(line)
        print()
