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
"""

import numpy as np
from refnx.reflect.structure import isld_profile
import matplotlib.pyplot as plt


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
    Imag_depth, Imag_SLD = isld_profile(structure.slabs())

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
