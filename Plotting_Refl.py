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
"""

import numpy as np
from refnx.reflect.structure import sld_profile as _sld_profile
import matplotlib.pyplot as plt

from h5io import load_h5_objectives
from Model_Setup import SLDinterp


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
