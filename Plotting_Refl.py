"""
Plotting_Refl.py
----------------
Reflectometry plotting utilities.

Public API
----------
profileflip(structure, depth_shift=0)
    Extract real and imaginary SLD profiles from a refnx Structure,
    flipped so depth runs from surface into the substrate.

modelcomparisonplot(obj_list, structure_list, ...)
    Three-row comparison plot: full reflectivity (log), zoomed reflectivity
    (linear), and SLD profile with layer shading.

plot_energy_model_grid(filepath, sample_name, model_names, energy_list, ...)
    Grid of reflectivity + SLD structure comparison across models fitted to
    an HDF5 file, one row per energy.

save_mcmc/load_mcmc(path, result) , save_jaxns/load_jaxns(path, result)
    Save/reload BlackJAX NUTS and JAXNS nested-sampling uncertainty results
    to/from compressed .npz files.

sample_posterior_curves(objective, result, method, q_plot, ...)
    Draw posterior parameter samples from a NUTS/JAXNS result, evaluate
    reflectivity + real/imaginary SLD profile for each draw.

plot_reflectivity_uncertainty_grid(energies, objectives_dict, uncertainty_results, ...)
    One row per energy: data, best fit, and a NUTS/JAXNS posterior-predictive
    reflectivity uncertainty band (energies missing uncertainty data are
    plotted best-fit-only).
"""

import numpy as np
import pandas as pd
from types import SimpleNamespace
from refnx.reflect.structure import sld_profile as _sld_profile
import matplotlib.pyplot as plt
import blackjax

from h5io import load_h5_objectives
from Model_Setup import SLDinterp
from gpu_reflect import extract_free_params, set_free_params


# ---------------------------------------------------------------------------
# Core profile helper
# ---------------------------------------------------------------------------

def profileflip(structure, depth_shift=0):
    """
    Extract SLD profiles from a refnx Structure, oriented so depth
    increases away from the substrate surface.

    Args:
        structure   : refnx Structure object
        depth_shift : constant added to the depth axis after flipping
                      (useful for sub-nm alignment tweaks)

    Returns:
        (Real_depth, Real_SLD, Imag_depth, Imag_SLD)
        All arrays are 1-D numpy arrays.
    """
    Real_depth, Real_SLD = structure.sld_profile()
    # isld_profile was removed from refnx; compute imaginary profile by
    # placing the iSLD column (col 2) into the real column (col 1) before calling
    slabs_imag = structure.slabs().copy()
    slabs_imag[:, 1] = slabs_imag[:, 2]
    Imag_depth, Imag_SLD = _sld_profile(slabs_imag)

    Real_depth = (Real_depth - Real_depth.max()) * -1 - depth_shift
    Imag_depth = (Imag_depth - Imag_depth.max()) * -1 - depth_shift

    return Real_depth, Real_SLD, Imag_depth, Imag_SLD


# ---------------------------------------------------------------------------
# Main comparison plot
# ---------------------------------------------------------------------------

def modelcomparisonplot(obj_list, structure_list, shade_start=None,
                         fig_size_w=16, colors=None, profile_shift=-10,
                         xlim=None, zoom_xlim=None, zoom_ylim=None):
    """
    Three-row comparison plot for one or more reflectometry models.

    Row 1 – full reflectivity on a log scale
    Row 2 – zoomed reflectivity (linear scale, low-Q region)
    Row 3 – SLD profile with layer shading

    Args:
        obj_list       : list of refnx Objective objects
        structure_list : list of refnx Structure objects (same order)
        shade_start    : list of depth offsets for layer shading
                         (None → all start at 0)
        fig_size_w     : figure width in inches
        colors         : list of shading colours (default palette provided)
        profile_shift  : constant depth shift applied to the SLD profile
        xlim           : [min, max] for the SLD depth axis (None = auto)
        zoom_xlim      : (min, max) Q range for the zoomed panel
                         (default (0, 0.05) Å⁻¹)
        zoom_ylim      : (min, max) for zoomed reflectivity y-axis
                         (default: auto from data in the zoom region)

    Returns:
        (fig, axes) – axes has shape (3, n) for n models
    """
    n = len(obj_list)
    if colors is None:
        colors = ['silver', 'grey', 'blue', 'violet', 'orange',
                  'purple', 'red', 'green', 'yellow']
    if zoom_xlim is None:
        zoom_xlim = (0.0, 0.05)

    fig, axes = plt.subplots(3, n if n > 1 else 1,
                              figsize=(fig_size_w, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    chi = np.array([o.chisqr() for o in obj_list])
    rel = np.round(chi / chi[0], 2)

    for i in range(n):
        ax_r  = axes[0, i]
        ax_rz = axes[1, i]
        ax_s  = axes[2, i]

        data    = obj_list[i].data
        q       = data.data[0]
        r_obs   = data.data[1]
        r_model = obj_list[i].model(q)

        # --- full reflectivity ---
        ax_r.plot(q, r_obs,   'o', markersize=3, label='Data')
        ax_r.plot(q, r_model, '-', label='Model')
        ax_r.set_yscale('log')
        ax_r.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_r.set_ylabel('Reflectivity')
        ax_r.text(0.5, 0.98, f'Rel. GF {rel[i]}',
                  transform=ax_r.transAxes,
                  ha='center', va='top', fontsize=9)
        ax_r.legend(fontsize=8)

        # --- zoomed reflectivity ---
        ax_rz.plot(q, r_obs,   'o', markersize=3, label='Data')
        ax_rz.plot(q, r_model, '-', label='Model')
        ax_rz.set_xlim(zoom_xlim)
        mask = (q >= zoom_xlim[0]) & (q <= zoom_xlim[1])
        if np.any(mask):
            all_y = np.concatenate([r_obs[mask], r_model[mask]])
            if zoom_ylim is not None:
                ax_rz.set_ylim(zoom_ylim)
            else:
                ax_rz.set_ylim(all_y.min() * 0.9, all_y.max() * 1.1)
        ax_rz.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_rz.set_ylabel('Reflectivity (linear)')

        # --- SLD profile ---
        Real_depth, Real_SLD, Imag_depth, Imag_SLD = profileflip(
            structure_list[i])
        ax_s.plot(Real_depth + profile_shift, Real_SLD,
                  color='blue', label='Real SLD', zorder=2)
        ax_s.plot(Imag_depth + profile_shift, Imag_SLD,
                  linestyle='--', color='blue', label='Imag SLD', zorder=2)
        if xlim is not None:
            ax_s.set_xlim(xlim)

        # layer shading
        slabs  = structure_list[i].slabs()
        pvals  = obj_list[i].parameters.pvals
        start  = (shade_start[i]
                  if shade_start and len(shade_start) > i else 0)
        thicknesses = [start]
        for j in range(1, len(slabs)):
            idx = (len(slabs) - j - 1) * 5 + 9
            t = pvals[idx] if idx < len(pvals) else slabs[j]['thickness']
            thicknesses.append(thicknesses[-1] + t)

        if thicknesses:
            ax_s.axvspan(0, thicknesses[0], color='silver',
                         alpha=0.3, zorder=0)
        for j in range(len(thicknesses) - 1):
            ax_s.axvspan(thicknesses[j], thicknesses[j + 1],
                         color=colors[min(j, len(colors) - 1)],
                         alpha=0.2, zorder=1)

        ax_s.legend(fontsize=8)
        ax_s.set_xlabel(r'Distance from Si ($\AA$)')
        ax_s.set_ylabel(r'SLD $(10^{-6})\ \AA^{-2}$')

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Energy-grid comparison plot
# ---------------------------------------------------------------------------

_DEFAULT_REF_COLORS = ['green', 'purple', 'saddlebrown', 'teal', 'magenta']


def plot_energy_model_grid(filepath, sample_name, model_names, energy_list,
                            mox_sld_arrays=None, criteria='best',
                            colors=None, fig_size_w=14, row_height=4,
                            profile_xlim=None, ref_colors=None):
    """
    Grid comparison of reflectivity + SLD structure across models, one row
    per energy.

    Loads the best (or otherwise selected) fitted Objective/Structure for
    each (model_name, energy) pair from an HDF5 fit results file via
    load_h5_objectives, then for each energy plots:
        left  – measured reflectivity with each model's fit curve overlaid
        right – each model's SLD-vs-depth structure profile overlaid, with
                horizontal reference lines for MOXA/MOXB-style materials
                (from mox_sld_arrays) drawn only where that layer is present.

    Args:
        filepath       : HDF5 fit results path
        sample_name    : sample label in the H5 file
        model_names    : list of model labels, e.g. ['Model1', 'Model2']
        energy_list    : energies to plot, one row each
        mox_sld_arrays : {material_name: ndarray[N,3]} reference SLD arrays
                         (Energy, Real_SLD, Imag_SLD), e.g.
                         {'MOXA': MOXA_SLD, 'MOXB': MOXB_SLD}. A material's
                         horizontal reference lines are only drawn on a row
                         if that layer is present in at least one loaded
                         structure for that energy.
        criteria       : passed to load_h5_objectives ('best' | 'last' | int)
        colors         : per-model line colors (default: matplotlib color
                         cycle)
        fig_size_w     : figure width in inches
        row_height     : height in inches allotted per energy row
        profile_xlim   : shared x-limits for the structure (right) column
        ref_colors     : per-material reference-line colors (default:
                         a small fixed palette, cycled over
                         mox_sld_arrays.keys())

    Returns:
        (fig, axes) – axes shape (len(energy_list), 2)
    """
    if mox_sld_arrays is None:
        mox_sld_arrays = {}

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [colors[i % len(colors)] for i in range(len(model_names))]

    if ref_colors is None:
        ref_colors = {
            mat: _DEFAULT_REF_COLORS[i % len(_DEFAULT_REF_COLORS)]
            for i, mat in enumerate(mox_sld_arrays)
        }

    model_data = {
        name: load_h5_objectives(filepath, sample_name, name,
                                  criteria=criteria, energy_list=energy_list)
        for name in model_names
    }

    n = len(energy_list)
    fig, axes = plt.subplots(n, 2, figsize=(fig_size_w, row_height * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, energy in enumerate(energy_list):
        ax_r, ax_s = axes[i, 0], axes[i, 1]
        data_plotted = False
        chi_lines = []
        present = set()

        for model_name, color in zip(model_names, colors):
            objectives_dict, structures_dict = model_data[model_name]
            obj = objectives_dict.get(energy)
            if obj is not None:
                q, r_obs = obj.data.data[0], obj.data.data[1]
                if not data_plotted:
                    ax_r.plot(q, r_obs, 'o', markersize=3, color='black',
                              label='Data')
                    data_plotted = True
                ax_r.plot(q, obj.model(q), '-', color=color, label=model_name)
                chi_lines.append(f'{model_name}: χ²={obj.chisqr():.3g}')

            structure = structures_dict.get(energy)
            if structure is not None:
                Real_depth, Real_SLD, Imag_depth, Imag_SLD = profileflip(structure)
                ax_s.plot(Real_depth, Real_SLD, '-', color=color, label=model_name)
                ax_s.plot(Imag_depth, Imag_SLD, '--', color=color)
                present.update(c.name for c in structure.components)

        ax_r.set_yscale('log')
        ax_r.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_r.set_ylabel('Reflectivity')
        ax_r.set_title(f'{energy} eV')
        if chi_lines:
            ax_r.text(0.98, 0.98, '\n'.join(chi_lines),
                      transform=ax_r.transAxes, ha='right', va='top', fontsize=8)
        ax_r.legend(fontsize=8)

        for material, sld_array in mox_sld_arrays.items():
            if material not in present:
                continue
            real, imag = SLDinterp(energy, sld_array)
            color = ref_colors[material]
            ax_s.axhline(real, linestyle='-', color=color, alpha=0.6,
                         linewidth=1, label=f'{material} ref (real)')
            ax_s.axhline(imag.imag, linestyle=':', color=color, alpha=0.6,
                         linewidth=1, label=f'{material} ref (imag)')

        if profile_xlim is not None:
            ax_s.set_xlim(profile_xlim)
        ax_s.set_xlabel(r'Distance from Si ($\AA$)')
        ax_s.set_ylabel(r'SLD $(10^{-6})\ \AA^{-2}$')
        ax_s.legend(fontsize=7)

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# BlackJAX NUTS / JAXNS uncertainty result I/O
# ---------------------------------------------------------------------------
# Ported from the per-notebook cells in BrewerN_Fit_NB/*_BlackJAX_JAXNS.ipynb so
# they're importable instead of copy-pasted; reads/writes the same .npz layout
# already on disk under BrewerN_Fit_NB/ALS_Fits/BrewerN/Uncertainty_M1/.

def save_mcmc(path, result):
    """Save a BlackJAX NUTS result (from gpu_mcmc.gpu_mcmc_converge) to .npz."""
    np.savez_compressed(path,
        samples_bounded    = result.samples_bounded,
        param_names        = np.array(result.param_names, dtype=object),
        sampler_name        = np.array([result.sampler_name]),
        converged           = np.array([getattr(result, 'converged', False)]),
        total_samples       = np.array([getattr(result, 'total_samples', result.n_samples)]),
        acceptance_rate     = np.array([result.acceptance_rate()    or np.nan]),
        divergent_fraction  = np.array([result.divergence_fraction() or np.nan]))


def _mcmc_summary(samples_bounded, param_names):
    """Mirrors MCMCSamples.summary() in gpu_mcmc.py, computed from reloaded samples."""
    n_free = samples_bounded.shape[-1]
    s_flat = samples_bounded.reshape(-1, n_free)
    rows = []
    for i, name in enumerate(param_names):
        col = s_flat[:, i]
        chain_i = samples_bounded[:, :, i]
        rows.append({
            'parameter': name,
            'mean':   float(col.mean()),
            'std':    float(col.std()),
            '2.5%':   float(np.percentile(col, 2.5)),
            'median': float(np.percentile(col, 50.0)),
            '97.5%':  float(np.percentile(col, 97.5)),
            'r_hat':  float(blackjax.rhat(chain_i, chain_axis=0, sample_axis=1)),
            'ess':    float(blackjax.ess(chain_i,  chain_axis=0, sample_axis=1)),
        })
    return pd.DataFrame(rows).set_index('parameter')


def load_mcmc(path):
    """Reload a BlackJAX NUTS result saved by save_mcmc."""
    d = np.load(path, allow_pickle=True)
    samples_bounded = d['samples_bounded']
    param_names = list(d['param_names'])
    return SimpleNamespace(
        samples_bounded     = samples_bounded,
        param_names         = param_names,
        sampler_name        = str(d['sampler_name'][0]),
        converged            = bool(d['converged'][0]),
        total_samples        = int(d['total_samples'][0]),
        acceptance_rate      = lambda: float(d['acceptance_rate'][0]),
        divergence_fraction  = lambda: float(d['divergent_fraction'][0]),
        summary              = lambda: _mcmc_summary(samples_bounded, param_names))


def save_jaxns(path, result):
    """Save a JAXNS nested-sampling result (from gpu_nested_sampler.run_nested_sampling) to .npz."""
    np.savez_compressed(path,
        samples          = result.samples,
        log_L_samples    = result.log_L_samples,
        log_Z_mean       = np.array([result.log_Z_mean]),
        log_Z_std        = np.array([result.log_Z_std]),
        ESS              = np.array([result.ESS]),
        H_mean           = np.array([result.H_mean]),
        posterior_mean   = result.posterior_mean,
        posterior_std    = result.posterior_std,
        posterior_median = result.posterior_median,
        param_names      = np.array(result.param_names, dtype=object),
        run_seconds      = np.array([result.run_seconds]))


def load_jaxns(path):
    """Reload a JAXNS result saved by save_jaxns."""
    d = np.load(path, allow_pickle=True)
    return SimpleNamespace(
        samples=d['samples'], log_L_samples=d['log_L_samples'],
        log_Z_mean=float(d['log_Z_mean'][0]), log_Z_std=float(d['log_Z_std'][0]),
        ESS=float(d['ESS'][0]), H_mean=float(d['H_mean'][0]),
        posterior_mean=d['posterior_mean'], posterior_std=d['posterior_std'],
        posterior_median=d['posterior_median'], param_names=list(d['param_names']),
        run_seconds=float(d['run_seconds'][0]))


# ---------------------------------------------------------------------------
# Posterior-predictive sampling + reflectivity uncertainty grid
# ---------------------------------------------------------------------------

def _sld_profile_complex(structure_obj, z=None):
    """
    Structure.sld_profile() only ever returns the REAL SLD (refnx's low-level
    sld_profile() ignores slab column 2, the imaginary/absorptive SLD). Compute
    the imaginary profile by swapping that column in and re-running the same
    erf-smoothing on the same z-grid.
    """
    slabs = structure_obj.slabs()
    zed, real_prof = _sld_profile(slabs, z=z)
    slabs_imag = np.copy(slabs)
    slabs_imag[:, 1] = slabs[:, 2]
    _, imag_prof = _sld_profile(slabs_imag, z=zed)
    return zed, real_prof, imag_prof


def sample_posterior_curves(objective, result, method, q_plot, z_ref=None,
                             n_curves=300, seed=0):
    """
    Draw random posterior parameter samples from a NUTS/JAXNS result, set them
    into `objective` one at a time, and evaluate reflectivity R(q_plot) and the
    real/imaginary SLD depth profile for each draw. Restores objective's
    original parameter values before returning.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        Best-fit objective whose free parameters match `result`'s param_names.
    result : SimpleNamespace
        A load_mcmc() result (method='blackjax') or load_jaxns() result
        (method='jaxns').
    method : 'blackjax' | 'jaxns'
    q_plot : ndarray
        Q values at which to evaluate the reflectivity model.
    z_ref : ndarray, optional
        Fixed depth grid for the SLD profile; if None, taken from the first
        posterior draw and reused for every subsequent draw.
    n_curves : int
        Number of posterior draws to evaluate.
    seed : int
        RNG seed for draw selection.

    Returns
    -------
    dict with keys 'q', 'z', 'R', 'SLD_real', 'SLD_imag' — the last three are
    (n_curves, n_points) arrays.
    """
    if method == 'blackjax':
        flat = result.samples_bounded.reshape(-1, result.samples_bounded.shape[-1])
    elif method == 'jaxns':
        flat = result.samples
    else:
        raise ValueError(f"method must be 'blackjax' or 'jaxns', got {method!r}")

    rng = np.random.default_rng(seed)
    model = objective.model
    structure = model.structure
    orig_vals = np.array([p.value for p in extract_free_params(objective)])

    idx = rng.choice(len(flat), min(n_curves, len(flat)), replace=False)
    R_curves, SLD_real_curves, SLD_imag_curves = [], [], []
    z_out = z_ref
    for s in flat[idx]:
        set_free_params(objective, s)
        R_curves.append(model(q_plot))
        z_i, sld_real_i, sld_imag_i = _sld_profile_complex(structure, z=z_out)
        if z_out is None:
            z_out = z_i
        SLD_real_curves.append(sld_real_i)
        SLD_imag_curves.append(sld_imag_i)

    set_free_params(objective, orig_vals)

    return {
        'q': q_plot,
        'z': z_out,
        'R': np.array(R_curves),
        'SLD_real': np.array(SLD_real_curves),
        'SLD_imag': np.array(SLD_imag_curves),
    }


def plot_reflectivity_uncertainty_grid(energies, objectives_dict, uncertainty_results,
                                        method='jaxns', n_curves=300, seed=0,
                                        row_height=3, fig_size_w=7, ci=(2.5, 97.5)):
    """
    One row per energy: experimental reflectivity data, best-fit curve, and a
    posterior-predictive uncertainty band from BlackJAX NUTS or JAXNS samples.

    Reflectivity-only counterpart of the two-panel (R + SLD profile) grid
    originally inlined in Brewer1_Carbon_Batch_Nov2025_BlackJAX_JAXNS.ipynb —
    the SLD side is instead covered by plot_sld_four_panel's uncertainty bands.

    Parameters
    ----------
    energies : list of float
        Energies to plot, one row each.
    objectives_dict : {float energy: refnx.analysis.Objective}
        Best-fit objective per energy (e.g. from h5io.load_h5_objectives).
    uncertainty_results : {float energy: SimpleNamespace}
        Loaded NUTS (load_mcmc) or JAXNS (load_jaxns) result per energy.
        Energies absent from this dict (or from objectives_dict) are plotted
        with whatever is available and annotated accordingly, rather than
        raising — partial uncertainty coverage across energies is expected.
    method : 'blackjax' | 'jaxns'
        Which uncertainty_results were loaded from.
    n_curves : int
        Posterior draws sampled per energy for the percentile band.
    seed : int
        RNG seed for draw selection.
    row_height, fig_size_w : float
        Figure sizing.
    ci : (low, high)
        Percentile bounds for the shaded band. Default (2.5, 97.5) = 95% CI.

    Returns
    -------
    (fig, axes)
    """
    n_rows = len(energies)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_size_w, row_height * n_rows),
                              squeeze=False)
    axes = axes[:, 0]

    for row, energy in enumerate(energies):
        ax = axes[row]
        obj = objectives_dict.get(energy)
        if obj is None:
            ax.set_title(f'{energy} eV — no best fit available')
            ax.axis('off')
            continue

        data = obj.data
        q_plot = np.linspace(data.x.min(), data.x.max(), 300)
        ax.errorbar(data.x, data.y, yerr=data.y_err, fmt='k.', ms=3, label='Data')
        ax.plot(q_plot, obj.model(q_plot), 'k--', lw=1, label='Best fit')

        result = uncertainty_results.get(energy)
        if result is not None:
            curves = sample_posterior_curves(obj, result, method, q_plot,
                                              n_curves=n_curves, seed=seed)
            lo, hi = np.percentile(curves['R'], ci, axis=0)
            ax.fill_between(q_plot, lo, hi, alpha=0.3, color='C0',
                             label=f'{ci[1] - ci[0]:.0f}% CI ({method})')
            ax.plot(q_plot, np.median(curves['R'], axis=0), color='C0', lw=1.2,
                    label='Median')
        else:
            ax.text(0.97, 0.9, 'no uncertainty data', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, style='italic', color='gray')

        ax.set_yscale('log')
        ax.set_xlabel(r'Q ($\AA^{-1}$)')
        ax.set_ylabel('Reflectivity')
        ax.set_title(f'{energy} eV')
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig, axes
