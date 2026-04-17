"""
NEXAFS_RSoXR.py
---------------
Visualisation and interactive exploration tools for RSoXR fitting results.

All data-loading, model-building, and batch-fitting pipeline functions
live in batch.py.  This module focuses purely on plotting and interactive
widgets that operate on already-built objectives / structures dicts.

Public API
----------
Static plots
    plot_simulated_reflectivity_and_sld
    plot_optical_constants_comparison
    visualize_batch_models
    visualize_fit_results
    visualize_before_after_fitting
    analyze_model_parameters_by_energy
    plot_parameter_energy_trends

Interactive widgets (require ipywidgets + Jupyter)
    create_interactive_model_visualizer
    create_multi_energy_comparison
    interactive_parameter_explorer
    create_vscode_sld_explorer
    create_vscode_parameter_explorer
    create_vscode_before_after_explorer
    create_interactive_parameter_updater_v3
"""

import os
import copy
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from NEXAFS import binary_contrast_numpy
from Plotting_Refl import profileflip, modelcomparisonplot
from Model_Setup import create_reflectometry_model, create_model_and_objective


# ---------------------------------------------------------------------------
# Static plotting
# ---------------------------------------------------------------------------

def plot_simulated_reflectivity_and_sld(reflectivity_dict, structures_dict,
                                         energies=None, profile_shift=-20,
                                         fig_size_w=14, save_path=None,
                                         sld_ylim=None, energy_tolerance=0.1):
    """
    Plot simulated reflectivity (top row) and SLD profiles (bottom row).

    Args:
        reflectivity_dict : {energy: {'q': …, 'reflectivity': …}} from
                            batch.simulate_reflectivity_profiles
        structures_dict   : {energy: Structure}
        energies          : energies to plot (None = all in intersection)
        profile_shift     : depth offset for SLD profiles
        fig_size_w        : figure width
        save_path         : save figure here if given
        sld_ylim          : (min, max) for SLD y-axis
        energy_tolerance  : max allowed mismatch when matching requested energies

    Returns:
        (fig, axes)  – axes shape (2, n_energies)
    """
    available = sorted(set(reflectivity_dict) & set(structures_dict))

    if energies is None:
        matched = available
    else:
        req = energies if isinstance(energies, (list, tuple, np.ndarray)) else [energies]
        matched = []
        for e in req:
            if not available:
                continue
            closest = min(available, key=lambda x: abs(x - e))
            if abs(closest - e) > energy_tolerance:
                print(f"Warning: {e} eV → using closest {closest} eV "
                      f"(Δ = {abs(closest-e):.3f} eV).")
            if closest not in matched:
                matched.append(closest)

    if not matched:
        print("No overlapping energies.")
        return None, None

    n   = len(matched)
    fig, axes = plt.subplots(2, n, figsize=(fig_size_w, 6), sharey=False)
    axes = np.array(axes).reshape(2, n)

    for idx, energy in enumerate(matched):
        entry     = reflectivity_dict[energy]
        structure = structures_dict[energy]
        ax_r = axes[0, idx]
        ax_s = axes[1, idx]

        ax_r.semilogy(entry['q'], entry['reflectivity'], color='C0')
        ax_r.set_xlabel('Q (Å⁻¹)')
        if idx == 0:
            ax_r.set_ylabel('Reflectivity')
        ax_r.set_title(f'{energy} eV')
        ax_r.grid(True, which='both', alpha=0.2)

        rd, rsl, id_, isl = profileflip(structure, depth_shift=0)
        ax_s.plot(rd + profile_shift, rsl, color='C1', label='Real SLD')
        ax_s.plot(id_ + profile_shift, isl, color='C1',
                  linestyle='--', label='Imag SLD')
        ax_s.set_xlabel('Distance from Si (Å)')
        if idx == 0:
            ax_s.set_ylabel(r'SLD ($10^{-6}$ Å$^{-2}$)')
        if sld_ylim is not None:
            ax_s.set_ylim(*sld_ylim)
        ax_s.grid(True, alpha=0.2)
        if idx == n - 1:
            ax_s.legend(loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_optical_constants_comparison(n1_array, n2_array=None, energy_list=None,
                                       bound_threshold=0.02, figsize=(15, 5),
                                       save_path=None, x_limits=None, labels=None):
    """
    Three-panel plot: real component, imaginary component, binary contrast.

    Args:
        n1_array       : (N, 3) array [Energy, Delta, Beta]
        n2_array       : optional second material array (same shape)
        energy_list    : energies to plot (default: all in n1_array)
        bound_threshold: passed to binary_contrast_numpy
        figsize        : figure dimensions
        save_path      : save path (optional)
        x_limits       : (min, max) for x-axis (optional)
        labels         : material labels

    Returns:
        (fig, axes)
    """
    if n1_array is None or len(n1_array) == 0:
        raise ValueError("n1_array must be a non-empty array.")

    energy = n1_array[:, 0]
    if labels is None:
        labels = ['Material 1', 'Material 2'] if n2_array is not None else ['Material 1']
    else:
        labels = list(labels)
        if n2_array is None:
            labels = labels[:1]
        elif len(labels) < 2:
            labels.append('Material 2')

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(energy, n1_array[:, 1], color='tab:blue', label=labels[0])
    if n2_array is not None:
        axes[0].plot(n2_array[:, 0], n2_array[:, 1],
                     color='tab:cyan', linestyle='--', label=labels[1])
    axes[0].set_title('Real Component')
    axes[0].set_xlabel('Energy (eV)')
    axes[0].set_ylabel('Delta (Real SLD)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(energy, n1_array[:, 2], color='tab:orange', label=labels[0])
    if n2_array is not None:
        axes[1].plot(n2_array[:, 0], n2_array[:, 2],
                     color='tab:red', linestyle='--', label=labels[1])
    axes[1].set_title('Imaginary Component')
    axes[1].set_xlabel('Energy (eV)')
    axes[1].set_ylabel('Beta (Imag SLD)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    contrast = binary_contrast_numpy(n1_array, n2_array, plot=False)
    axes[2].plot(contrast[:, 0], contrast[:, 1], color='tab:green')
    axes[2].set_title('Binary Contrast')
    axes[2].set_xlabel('Energy (eV)')
    axes[2].set_ylabel('Contrast')
    axes[2].grid(True, alpha=0.3)

    if x_limits is not None:
        for ax in axes:
            ax.set_xlim(x_limits)
        mask = ((contrast[:, 0] >= x_limits[0]) &
                (contrast[:, 0] <= x_limits[1]))
        if np.any(mask):
            seg = contrast[mask, 1]
            ymin, ymax = seg.min(), seg.max()
            if np.isfinite(ymin) and np.isfinite(ymax):
                margin = abs(ymax - ymin) * 0.05 if not np.isclose(ymin, ymax) else abs(ymin) * 0.05 or 1e-6
                axes[2].set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def visualize_batch_models(objectives_dict, structures_dict, energies=None,
                            shade_start=None, profile_shift=-20, xlim=None,
                            fig_size_w=16, colors=None, save_path=None):
    """
    Plot reflectometry models for selected energies using modelcomparisonplot.

    Args:
        objectives_dict : {energy: Objective}
        structures_dict : {energy: Structure}
        energies        : energy or list of energies to plot (None = all)
        shade_start     : list of layer-shading start positions (or scalar)
        profile_shift   : depth axis offset
        xlim            : [min, max] for SLD depth axis
        fig_size_w      : figure width
        colors          : layer shading colour list
        save_path       : save path (optional)

    Returns:
        (fig, axes) or (None, None) on failure
    """
    if energies is None:
        to_plot = sorted(set(objectives_dict) & set(structures_dict))
    elif isinstance(energies, (int, float)):
        to_plot = [energies]
    else:
        to_plot = sorted(energies)

    to_plot = [e for e in to_plot
               if e in objectives_dict and e in structures_dict]
    if not to_plot:
        print("No valid energies to plot.")
        return None, None

    obj_list  = [objectives_dict[e] for e in to_plot]
    s_list    = [structures_dict[e]  for e in to_plot]
    n         = len(to_plot)
    ss = ([0] * n if shade_start is None
          else [shade_start] * n if isinstance(shade_start, (int, float))
          else shade_start)

    try:
        fig, axes = modelcomparisonplot(
            obj_list=obj_list, structure_list=s_list,
            shade_start=ss, profile_shift=profile_shift,
            xlim=xlim, fig_size_w=fig_size_w, colors=colors)

        if n == 1:
            axes[0, 0].set_title(f'Reflectivity – {to_plot[0]} eV')
            axes[2, 0].set_title(f'SLD Profile – {to_plot[0]} eV')
        else:
            for i, e in enumerate(to_plot):
                axes[0, i].set_title(f'Reflectivity – {e} eV')
                axes[2, i].set_title(f'SLD Profile – {e} eV')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved → {save_path}")
        return fig, axes

    except Exception as exc:
        print(f"Error generating plots: {exc}")
        return None, None


def visualize_fit_results(results_dict, structures_dict, energies=None,
                           shade_start=None, profile_shift=-20, xlim=None,
                           fig_size_w=16, colors=None, save_path=None):
    """
    Visualise fitted results by extracting objectives from results_dict
    and forwarding to visualize_batch_models.

    Args:
        results_dict    : {energy: {'objective': …, 'optimized_chi_squared': …}}
        structures_dict : {energy: Structure}
        (other args same as visualize_batch_models)

    Returns:
        (fig, axes)
    """
    objectives = {e: r['objective']
                  for e, r in results_dict.items() if 'objective' in r}
    chi2       = {e: r.get('optimized_chi_squared')
                  for e, r in results_dict.items()}

    fig, axes = visualize_batch_models(
        objectives_dict=objectives, structures_dict=structures_dict,
        energies=energies, shade_start=shade_start,
        profile_shift=profile_shift, xlim=xlim,
        fig_size_w=fig_size_w, colors=colors)

    if fig is not None and axes is not None:
        plot_es = (sorted(objectives) if energies is None
                   else [e for e in energies if e in objectives])
        n = len(plot_es)
        if n == 1:
            e = plot_es[0]
            if chi2.get(e) is not None:
                axes[0, 0].set_title(
                    f'Reflectivity – {e} eV  (χ²: {chi2[e]:.4f})')
        else:
            for i, e in enumerate(plot_es):
                if i < axes.shape[1] and chi2.get(e) is not None:
                    axes[0, i].set_title(
                        f'Reflectivity – {e} eV  (χ²: {chi2[e]:.4f})')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes






# ---------------------------------------------------------------------------
# Interactive widgets
# ---------------------------------------------------------------------------

def create_interactive_model_visualizer(objectives_dict, structures_dict,
                                         energy_list=None, shade_start=None,
                                         profile_shift=-20, xlim=None,
                                         fig_size_w=14, colors=None):
    """
    Dropdown widget to inspect a single energy model at a time.

    Requires ipywidgets and a Jupyter/JupyterLab environment.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output

    avail = (sorted(set(objectives_dict) & set(structures_dict))
             if energy_list is None
             else [e for e in energy_list
                   if e in objectives_dict and e in structures_dict])
    if not avail:
        return widgets.HTML('<b>No valid energies.</b>')

    energy_dd   = widgets.Dropdown(
        options=[(f'{e} eV', e) for e in avail], description='Energy:')
    prev_btn    = widgets.Button(description='◀ Prev')
    next_btn    = widgets.Button(description='Next ▶')
    chi_label   = widgets.Label('χ² = N/A')
    output      = widgets.Output()

    def plot(energy):
        obj = objectives_dict[energy]
        s   = structures_dict[energy]
        try:
            chi_label.value = f'χ² = {obj.chisqr():.4f}'
        except Exception:
            chi_label.value = 'χ² = N/A'
        with output:
            clear_output(wait=True)
            fig = plt.figure(figsize=(fig_size_w, 8))
            gs  = GridSpec(2, 1, figure=fig)
            ax_r = fig.add_subplot(gs[0])
            ax_s = fig.add_subplot(gs[1])

            data = obj.data
            ax_r.semilogy(data.data[0], data.data[1], '.', markersize=3,
                          label='Data')
            ax_r.semilogy(data.data[0], obj.model(data.data[0]), '-',
                          label='Model')
            ax_r.set_xlabel(r'Q ($\AA^{-1}$)')
            ax_r.set_ylabel('Reflectivity')
            ax_r.set_title(f'{energy} eV  (χ²: {obj.chisqr():.4f})')
            ax_r.legend()

            rd, rsl, id_, isl = profileflip(s, depth_shift=0)
            ss = (shade_start if isinstance(shade_start, (int, float))
                  else 0)
            ax_s.plot(rd + profile_shift + ss, rsl,
                      color='blue', label='Real SLD')
            ax_s.plot(id_ + profile_shift + ss, isl,
                      linestyle='--', color='blue', label='Imag SLD')
            if xlim is not None:
                ax_s.set_xlim(xlim)
            ax_s.set_xlabel(r'Distance from Si ($\AA$)')
            ax_s.set_ylabel(r'SLD $(10^{-6})\ \AA^{-2}$')
            ax_s.legend()
            plt.tight_layout()
            plt.show()

    def on_dd(change):
        if change['name'] == 'value':
            plot(change['new'])

    def on_prev(_):
        idx = avail.index(energy_dd.value)
        if idx > 0:
            energy_dd.value = avail[idx - 1]

    def on_next(_):
        idx = avail.index(energy_dd.value)
        if idx < len(avail) - 1:
            energy_dd.value = avail[idx + 1]

    energy_dd.observe(on_dd, names='value')
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    plot(avail[0])

    return widgets.VBox([
        widgets.HBox([prev_btn, energy_dd, next_btn, chi_label]),
        output,
    ])


def create_multi_energy_comparison(objectives_dict, structures_dict,
                                    energy_list=None, shade_start=None,
                                    profile_shift=-20, xlim=None,
                                    fig_size_w=16, colors=None,
                                    max_energies=4):
    """
    Widget with a multi-select list to compare up to max_energies models
    side-by-side.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output

    avail = (sorted(set(objectives_dict) & set(structures_dict))
             if energy_list is None
             else [e for e in energy_list
                   if e in objectives_dict and e in structures_dict])
    if not avail:
        return widgets.HTML('<b>No valid energies.</b>')

    sel    = widgets.SelectMultiple(
        options=[(f'{e} eV', e) for e in avail],
        description='Energies:',
        rows=min(10, len(avail)))
    btn    = widgets.Button(description='Update Plot')
    info   = widgets.Label(f'Select up to {max_energies} energies')
    output = widgets.Output()

    def update(_):
        chosen = list(sel.value)
        if not chosen:
            info.value = 'Select at least one energy.'
            return
        if len(chosen) > max_energies:
            chosen = chosen[:max_energies]
        info.value = f'Comparing {len(chosen)} energies'
        obj_list = [objectives_dict[e] for e in chosen]
        s_list   = [structures_dict[e]  for e in chosen]
        ss = ([0] * len(chosen) if shade_start is None
              else [shade_start] * len(chosen)
              if isinstance(shade_start, (int, float)) else shade_start)
        with output:
            clear_output(wait=True)
            fig, axes = modelcomparisonplot(
                obj_list=obj_list, structure_list=s_list,
                shade_start=ss, profile_shift=profile_shift,
                xlim=xlim, fig_size_w=fig_size_w, colors=colors)
            n = len(chosen)
            if n == 1:
                axes[0, 0].set_title(f'Reflectivity – {chosen[0]} eV')
            else:
                for i, e in enumerate(chosen):
                    axes[0, i].set_title(
                        f'Reflectivity – {e} eV  (χ²: {obj_list[i].chisqr():.4f})')
            plt.tight_layout()
            plt.show()

    btn.on_click(update)
    return widgets.HBox([
        widgets.VBox([sel, btn, info]),
        output,
    ])


def interactive_parameter_explorer(objectives_dict, structures_dict,
                                    energy_list=None):
    """
    Combined widget: parameter-vs-energy trend plots plus a per-energy
    reflectivity/SLD viewer driven by clicking on trend points.

    Uses matplotlib pick events; requires %matplotlib widget in Jupyter.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output

    avail = (sorted(set(objectives_dict) & set(structures_dict))
             if energy_list is None
             else [e for e in energy_list
                   if e in objectives_dict and e in structures_dict])
    if not avail:
        return widgets.HTML('<b>No valid energies.</b>')

    # collect varying parameters
    param_names = []
    for e in avail:
        for p in objectives_dict[e].parameters.flattened():
            if p.vary and p.name not in param_names:
                param_names.append(p.name)

    param_dd = widgets.Dropdown(
        options=param_names or ['(none)'], description='Parameter:')
    output   = widgets.Output()
    detail   = widgets.Output()

    def plot_trend(param):
        with output:
            clear_output(wait=True)
            vals, errs = [], []
            for e in avail:
                for p in objectives_dict[e].parameters.flattened():
                    if p.name == param:
                        vals.append(p.value)
                        errs.append(getattr(p, 'stderr', None) or 0)
                        break
                else:
                    vals.append(None)
                    errs.append(0)
            valid = [(e, v, er) for e, v, er in zip(avail, vals, errs)
                     if v is not None]
            if not valid:
                return
            es, vs, es_err = zip(*valid)
            fig, ax = plt.subplots(figsize=(10, 4))
            sc = ax.errorbar(es, vs, yerr=es_err, fmt='o-', capsize=3,
                             picker=5)
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel(param)
            ax.set_title(param)
            ax.grid(True, alpha=0.3)

            def on_pick(event):
                if event.artist == sc.lines[0]:
                    idx = event.ind[0]
                    energy = es[idx]
                    with detail:
                        clear_output(wait=True)
                        obj = objectives_dict[energy]
                        s   = structures_dict[energy]
                        fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
                        d = obj.data
                        a1.semilogy(d.data[0], d.data[1], '.', markersize=3)
                        a1.semilogy(d.data[0], obj.model(d.data[0]), '-')
                        a1.set_title(f'{energy} eV  χ²={obj.chisqr():.4f}')
                        a1.set_xlabel(r'Q ($\AA^{-1}$)')
                        rd, rsl, id_, isl = profileflip(s)
                        a2.plot(rd, rsl, label='Real')
                        a2.plot(id_, isl, '--', label='Imag')
                        a2.set_xlabel(r'Depth ($\AA$)')
                        a2.legend()
                        plt.tight_layout()
                        plt.show()

            fig.canvas.mpl_connect('pick_event', on_pick)
            plt.tight_layout()
            plt.show()

    def on_dd(change):
        if change['name'] == 'value':
            plot_trend(change['new'])

    param_dd.observe(on_dd, names='value')
    if param_names:
        plot_trend(param_names[0])

    return widgets.VBox([param_dd, output, detail])


def create_vscode_sld_explorer(objectives_dict, structures_dict,
                                energy_list=None, profile_shift=-20,
                                xlim=None, fig_size_w=14, colors=None):
    """
    Keyboard-navigable SLD explorer suitable for VS Code notebooks.
    Use left/right arrow keys to step through energies.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output

    avail = (sorted(set(objectives_dict) & set(structures_dict))
             if energy_list is None
             else [e for e in energy_list
                   if e in objectives_dict and e in structures_dict])
    if not avail:
        print('No valid energies.')
        return None

    state  = {'idx': 0}
    output = widgets.Output()

    def plot(idx):
        energy = avail[idx]
        obj    = objectives_dict[energy]
        s      = structures_dict[energy]
        with output:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(fig_size_w, 5))
            data = obj.data
            axes[0].semilogy(data.data[0], data.data[1], '.', markersize=3,
                             label='Data')
            axes[0].semilogy(data.data[0], obj.model(data.data[0]), '-',
                             label='Model')
            axes[0].set_xlabel(r'Q ($\AA^{-1}$)')
            axes[0].set_ylabel('Reflectivity')
            axes[0].set_title(f'{energy} eV  χ²={obj.chisqr():.4f}')
            axes[0].legend()

            rd, rsl, id_, isl = profileflip(s, depth_shift=0)
            axes[1].plot(rd + profile_shift, rsl, label='Real SLD')
            axes[1].plot(id_ + profile_shift, isl, '--', label='Imag SLD')
            if xlim:
                axes[1].set_xlim(xlim)
            axes[1].set_xlabel(r'Distance from Si ($\AA$)')
            axes[1].set_ylabel(r'SLD $(10^{-6})\ \AA^{-2}$')
            axes[1].legend()

            def on_key(event):
                if event.key == 'right' and state['idx'] < len(avail) - 1:
                    state['idx'] += 1
                    plot(state['idx'])
                elif event.key == 'left' and state['idx'] > 0:
                    state['idx'] -= 1
                    plot(state['idx'])

            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.tight_layout()
            plt.show()

    plot(0)
    print('Use ← → arrow keys to navigate energies.')
    return output


def create_vscode_parameter_explorer(objectives_dict, structures_dict,
                                      energy_list=None, material_name='PS',
                                      figsize=(14, 12), xlim=None,
                                      bound_threshold=0.02):
    """
    Interactive parameter-vs-energy explorer for VS Code / Jupyter.

    Shows five trend panels (Real SLD, Imag SLD, Thickness, Roughness, χ²)
    for a chosen material alongside a live reflectivity and SLD profile panel
    that updates when you click any data point.

    Red points and shading indicate parameters within bound_threshold of
    their bounds.  Click any point to inspect that energy.

    Args:
        objectives_dict  : {energy: Objective}
        structures_dict  : {energy: Structure}
        energy_list      : energies to include (None = all)
        material_name    : material whose parameters are trended
        figsize          : (width, height) of the figure
        xlim             : (min, max) energy axis limits (None = auto)
        bound_threshold  : fraction of bound range used for near-bound
                           highlighting (default 0.02 = 2 %)

    Returns:
        matplotlib Figure
    """
    avail = (sorted(set(objectives_dict) & set(structures_dict))
             if energy_list is None
             else [e for e in energy_list
                   if e in objectives_dict and e in structures_dict])
    if not avail:
        print('No valid energies found.')
        return None

    param_strings = [
        f'{material_name} - sld',
        f'{material_name} - isld',
        f'{material_name} - thick',
        f'{material_name} - rough',
    ]
    param_colors = {'sld': 'blue', 'isld': 'orange',
                    'thick': 'green', 'rough': 'purple', 'chi': 'red'}

    # ── collect data ──────────────────────────────────────────────────────
    param_data = {ps: [] for ps in param_strings}
    chi_data   = []

    for energy in avail:
        obj  = objectives_dict[energy]
        chi2 = obj.chisqr()
        chi_data.append({'energy': energy, 'chi_squared': chi2})

        for ps in param_strings:
            found = False
            for param in obj.parameters.flattened():
                if param.name == ps:
                    bl = bh = None
                    try:
                        b = getattr(param, 'bounds', None)
                        if b is not None:
                            bl = b.lb if hasattr(b, 'lb') else b[0]
                            bh = b.ub if hasattr(b, 'ub') else b[1]
                    except Exception:
                        pass
                    near = False
                    if bl is not None and bh is not None and bh > bl:
                        span = bh - bl
                        near = (abs(param.value - bl) < bound_threshold * span
                                or abs(bh - param.value) < bound_threshold * span)
                    param_data[ps].append({
                        'energy': energy,
                        'value':  param.value,
                        'stderr': getattr(param, 'stderr', None) or 0,
                        'bound_low': bl, 'bound_high': bh,
                        'near_bound': near, 'chi_squared': chi2,
                    })
                    found = True
                    break
            if not found:
                print(f'  {ps} not found at {energy} eV')

    param_dfs  = {ps: pd.DataFrame(d).sort_values('energy')
                  for ps, d in param_data.items() if d}
    chi_df     = pd.DataFrame(chi_data).sort_values('energy')

    # ── build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 1.5])

    ax_sld   = fig.add_subplot(gs[0, :2])
    ax_isld  = fig.add_subplot(gs[1, :2])
    ax_thick = fig.add_subplot(gs[2, :2])
    ax_rough = fig.add_subplot(gs[3, :2])
    ax_chi   = fig.add_subplot(gs[4, :2])
    ax_refl  = fig.add_subplot(gs[3:5, 2])
    ax_prof  = fig.add_subplot(gs[0:3, 2])

    scatter_plots   = {}
    highlight_pts   = {}
    energy_vlines   = {}

    def _plot_param(ax, ps, ptype):
        if ps not in param_dfs:
            return None
        df = param_dfs[ps]
        c  = param_colors[ptype]

        normal  = df[~df['near_bound']]
        nearb   = df[ df['near_bound']]

        sc_norm = ax.scatter(normal['energy'], normal['value'],
                             s=64, color=c, picker=5) if not normal.empty else None
        sc_near = ax.scatter(nearb['energy'],  nearb['value'],
                             s=64, color='red', picker=5) if not nearb.empty else None

        # bound shading
        valid = df['bound_low'].notna() & df['bound_high'].notna()
        if valid.any():
            bd = df[valid]
            ax.fill_between(bd['energy'], bd['bound_low'], bd['bound_high'],
                            color=c, alpha=0.10)
            ax.plot(bd['energy'], bd['bound_low'],  '--', color=c, alpha=0.5, lw=1)
            ax.plot(bd['energy'], bd['bound_high'], '--', color=c, alpha=0.5, lw=1)

        ax.errorbar(df['energy'], df['value'], yerr=df['stderr'],
                    fmt='none', ecolor='gray', alpha=0.5)
        ax.plot(df['energy'], df['value'], '-', color=c, alpha=0.5)

        return sc_norm or sc_near

    scatter_plots['sld']   = _plot_param(ax_sld,   f'{material_name} - sld',   'sld')
    scatter_plots['isld']  = _plot_param(ax_isld,  f'{material_name} - isld',  'isld')
    scatter_plots['thick'] = _plot_param(ax_thick, f'{material_name} - thick', 'thick')
    scatter_plots['rough'] = _plot_param(ax_rough, f'{material_name} - rough', 'rough')

    ax_sld.set_ylabel('Real SLD\n(10⁻⁶ Å⁻²)')
    ax_sld.set_title(f'Real SLD – {material_name}')
    ax_sld.grid(True, alpha=0.3)

    ax_isld.set_ylabel('Imag SLD\n(10⁻⁶ Å⁻²)')
    ax_isld.set_title(f'Imaginary SLD – {material_name}')
    ax_isld.grid(True, alpha=0.3)

    ax_thick.set_ylabel('Thickness (Å)')
    ax_thick.set_title(f'Thickness – {material_name}')
    ax_thick.grid(True, alpha=0.3)

    ax_rough.set_ylabel('Roughness (Å)')
    ax_rough.set_title(f'Roughness – {material_name}')
    ax_rough.grid(True, alpha=0.3)

    scatter_plots['chi'] = ax_chi.scatter(
        chi_df['energy'], chi_df['chi_squared'], s=64, color='red', picker=5)
    ax_chi.plot(chi_df['energy'], chi_df['chi_squared'], 'r-', alpha=0.5)
    ax_chi.set_xlabel('Energy (eV)')
    ax_chi.set_ylabel('χ²')
    ax_chi.set_title('Goodness of Fit')
    ax_chi.grid(True, alpha=0.3)
    if (chi_df['chi_squared'].max() / (chi_df['chi_squared'].min() + 1e-30)) > 10:
        ax_chi.set_yscale('log')

    # vertical energy markers and highlight points
    for ptype, ax_obj, ps in [
        ('sld',   ax_sld,   f'{material_name} - sld'),
        ('isld',  ax_isld,  f'{material_name} - isld'),
        ('thick', ax_thick, f'{material_name} - thick'),
        ('rough', ax_rough, f'{material_name} - rough'),
        ('chi',   ax_chi,   None),
    ]:
        energy_vlines[ptype] = ax_obj.axvline(
            x=avail[0], color='red', linestyle='--', alpha=0.7)
        if ps and ps in param_dfs:
            df = param_dfs[ps]
            highlight_pts[ptype], = ax_obj.plot(
                df['energy'].iloc[0], df['value'].iloc[0],
                'o', color='red', markersize=12, alpha=0.7)

    if xlim:
        for ax_obj in [ax_sld, ax_isld, ax_thick, ax_rough, ax_chi]:
            ax_obj.set_xlim(xlim)

    # text annotations at the bottom
    ann = {k: fig.text(x, 0.02, '', fontsize=9, transform=fig.transFigure)
           for k, x in [('energy', 0.02), ('sld', 0.18), ('isld', 0.34),
                         ('thick', 0.50), ('rough', 0.66), ('gof', 0.82)]}

    # ── update callback ───────────────────────────────────────────────────
    def update(energy):
        obj = objectives_dict[energy]
        s   = structures_dict[energy]

        for ptype in energy_vlines:
            energy_vlines[ptype].set_xdata([energy, energy])

        def _get(ps):
            if ps in param_dfs:
                row = param_dfs[ps][param_dfs[ps]['energy'] == energy]
                if not row.empty:
                    return row['value'].iloc[0], row['near_bound'].iloc[0]
            return None, False

        sv, sn   = _get(f'{material_name} - sld')
        iv, in_  = _get(f'{material_name} - isld')
        tv, tn   = _get(f'{material_name} - thick')
        rv, rn   = _get(f'{material_name} - rough')

        for ptype, val in [('sld', sv), ('isld', iv),
                            ('thick', tv), ('rough', rv)]:
            if ptype in highlight_pts and val is not None:
                highlight_pts[ptype].set_data([energy], [val])

        chi_row = chi_df[chi_df['energy'] == energy]
        chi2    = chi_row['chi_squared'].iloc[0] if not chi_row.empty else None

        # reflectivity
        ax_refl.clear()
        data = obj.data
        ax_refl.semilogy(data.data[0], data.data[1], 'o',
                         markersize=3, label='Data')
        ax_refl.semilogy(data.data[0], obj.model(data.data[0]), '-',
                         label='Model')
        ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_refl.set_ylabel('Reflectivity')
        ax_refl.set_title(f'{energy} eV' +
                          (f'  χ²={chi2:.4f}' if chi2 is not None else ''))
        ax_refl.grid(True, alpha=0.3)
        ax_refl.legend(loc='best', fontsize=8)

        # SLD profile
        ax_prof.clear()
        rd, rsl, id_, isl = profileflip(s, depth_shift=0)
        ax_prof.plot(rd - 20, rsl, 'b-',  label='Real SLD')
        ax_prof.plot(id_ - 20, isl, 'b--', label='Imag SLD')
        ax_prof.set_xlabel(r'Distance from Si ($\AA$)')
        ax_prof.set_ylabel(r'SLD $(10^{-6})\ \AA^{-2}$')
        ax_prof.set_title(f'SLD Profile – {energy} eV')
        ax_prof.grid(True, alpha=0.3)
        ax_prof.legend(loc='best', fontsize=8)

        # text footer
        def _fmt(label, val, near):
            return f'{label}: {val:.4g}{"*" if near else ""}' if val is not None else ''

        ann['energy'].set_text(f'Energy: {energy} eV')
        ann['sld'].set_text(_fmt('SLD',   sv, sn))
        ann['isld'].set_text(_fmt('iSLD',  iv, in_))
        ann['thick'].set_text(_fmt('Thick', tv, tn) + (' Å' if tv is not None else ''))
        ann['rough'].set_text(_fmt('Rough', rv, rn) + (' Å' if rv is not None else ''))
        ann['gof'].set_text(f'χ²: {chi2:.4g}' if chi2 is not None else '')

        fig.canvas.draw_idle()

    def on_pick(event):
        if hasattr(event, 'ind') and len(event.ind) > 0:
            xdata = event.artist.get_offsets()[:, 0]
            update(xdata[event.ind[0]])

    fig.canvas.mpl_connect('pick_event', on_pick)

    fig.text(0.5, 0.005,
             f'Click any point to inspect that energy.  '
             f'Red = within {bound_threshold*100:.0f}% of bounds.  '
             f'* in footer = near bound.',
             ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    update(avail[0])
    print('Click any point to update the reflectivity and SLD profile panels.')
    return fig


def create_vscode_before_after_explorer(results_dict, original_objectives_dict,
                                         structures_dict, energy_list=None,
                                         profile_shift=-20, fig_size_w=16):
    """
    Keyboard-navigable before/after comparison for VS Code.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output

    fitted = {e: r['objective']
              for e, r in results_dict.items() if 'objective' in r}

    avail = (sorted(set(original_objectives_dict) & set(fitted)
                    & set(structures_dict))
             if energy_list is None
             else [e for e in energy_list
                   if e in original_objectives_dict
                   and e in fitted and e in structures_dict])
    if not avail:
        print('No valid energies.')
        return None

    state  = {'idx': 0}
    output = widgets.Output()

    def plot(idx):
        energy = avail[idx]
        orig   = original_objectives_dict[energy]
        fit    = fitted[energy]
        s      = structures_dict[energy]
        with output:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(fig_size_w, 5))
            data = orig.data
            q    = data.data[0]

            axes[0].semilogy(q, data.data[1], 'k.', markersize=3,
                             label='Data', alpha=0.6)
            axes[0].semilogy(q, orig.model(q), 'r-', label='Before')
            axes[0].semilogy(q, fit.model(q),  'b-', label='After')
            axes[0].set_xlabel(r'Q ($\AA^{-1}$)')
            axes[0].set_ylabel('Reflectivity')
            axes[0].set_title(f'{energy} eV\n'
                              f'χ² {orig.chisqr():.3g} → {fit.chisqr():.3g}')
            axes[0].legend()

            rd, rsl, id_, isl = profileflip(s, depth_shift=0)
            axes[1].plot(rd + profile_shift, rsl, label='Real')
            axes[1].plot(id_ + profile_shift, isl, '--', label='Imag')
            axes[1].set_xlabel(r'Distance from Si ($\AA$)')
            axes[1].legend()

            # residuals
            axes[2].plot(q, (data.data[1] - orig.model(q)) / data.data[1],
                         'r-', label='Before residual', alpha=0.7)
            axes[2].plot(q, (data.data[1] - fit.model(q)) / data.data[1],
                         'b-', label='After residual', alpha=0.7)
            axes[2].axhline(0, color='k', linewidth=0.8)
            axes[2].set_xlabel(r'Q ($\AA^{-1}$)')
            axes[2].set_ylabel('Relative residual')
            axes[2].legend()

            def on_key(event):
                if event.key == 'right' and state['idx'] < len(avail) - 1:
                    state['idx'] += 1
                    plot(state['idx'])
                elif event.key == 'left' and state['idx'] > 0:
                    state['idx'] -= 1
                    plot(state['idx'])

            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.tight_layout()
            plt.show()

    plot(0)
    print('Use ← → arrow keys to navigate energies.')
    return output


def create_interactive_parameter_updater_v3(objectives_dict, structures_dict,
                                             energy_list=None,
                                             material_names=None,
                                             figsize=(18, 24), xlim=None,
                                             bound_threshold=0.02,
                                             save_dir=None):
    """
    Full-featured interactive widget for viewing and editing parameter bounds.

    Shows all four parameter types (sld, isld, thick, rough) simultaneously
    for one material at a time.  Use the energy dropdown and material tabs to
    navigate.  Bounds can be updated live; a Save button pickles the updated
    objective.

    Requires ipywidgets and a Jupyter environment.
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    avail = (sorted(set(objectives_dict) & set(structures_dict))
             if energy_list is None
             else [e for e in energy_list
                   if e in objectives_dict and e in structures_dict])
    if not avail:
        return widgets.HTML('<b>No valid energies.</b>')

    # collect material names from parameter names
    if material_names is None:
        mat_set = set()
        for e in avail:
            for p in objectives_dict[e].parameters.flattened():
                if ' - ' in p.name:
                    mat_set.add(p.name.split(' - ')[0].strip())
        material_names = sorted(mat_set)

    if not material_names:
        return widgets.HTML('<b>No materials with named parameters found.</b>')

    param_types = ['sld', 'isld', 'thick', 'rough']

    energy_dd  = widgets.Dropdown(
        options=[(f'{e} eV', e) for e in avail], description='Energy:')
    mat_tabs   = widgets.Tab()
    plot_out   = widgets.Output()
    status     = widgets.HTML('<i>Ready.</i>')

    # build per-material, per-param-type bound widgets
    all_widgets = {}
    tab_children = []
    for mat in material_names:
        mat_boxes = {}
        mat_rows  = []
        for pt in param_types:
            lo_w = widgets.FloatText(description=f'{pt} lo:', layout=widgets.Layout(width='180px'))
            hi_w = widgets.FloatText(description='hi:', layout=widgets.Layout(width='180px'))
            vr_w = widgets.Checkbox(description='vary', layout=widgets.Layout(width='100px'))
            up_w = widgets.Button(description='Update', button_style='info',
                                   layout=widgets.Layout(width='90px'))
            rs_w = widgets.Button(description='Reset',
                                   layout=widgets.Layout(width='80px'))
            mat_boxes[pt] = dict(lo=lo_w, hi=hi_w, vary=vr_w,
                                  update=up_w, reset=rs_w)
            mat_rows.append(widgets.HBox([lo_w, hi_w, vr_w, up_w, rs_w]))
        all_widgets[mat] = mat_boxes
        tab_children.append(widgets.VBox(mat_rows))

    mat_tabs.children = tab_children
    for i, m in enumerate(material_names):
        mat_tabs.set_title(i, m)

    save_btn = widgets.Button(description='Save Objective',
                               button_style='success')

    def get_current():
        return energy_dd.value, material_names[mat_tabs.selected_index]

    def refresh_widgets():
        energy, mat = get_current()
        obj = objectives_dict[energy]
        for pt in param_types:
            ww = all_widgets[mat][pt]
            for p in obj.parameters.flattened():
                if f'{mat} - {pt}' in p.name.lower():
                    try:
                        b = getattr(p, 'bounds', None)
                        if b is not None:
                            ww['lo'].value = b.lb if hasattr(b, 'lb') else b[0]
                            ww['hi'].value = b.ub if hasattr(b, 'ub') else b[1]
                    except Exception:
                        pass
                    ww['vary'].value = bool(getattr(p, 'vary', False))
                    break

    def refresh_plot():
        energy, _ = get_current()
        obj = objectives_dict[energy]
        s   = structures_dict[energy]
        with plot_out:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            data = obj.data
            axes[0].semilogy(data.data[0], data.data[1], '.', markersize=3,
                             label='Data')
            axes[0].semilogy(data.data[0], obj.model(data.data[0]), '-',
                             label='Model')
            axes[0].set_title(f'{energy} eV  χ²={obj.chisqr():.4f}')
            axes[0].set_xlabel(r'Q ($\AA^{-1}$)')
            axes[0].legend()
            rd, rsl, id_, isl = profileflip(s, depth_shift=0)
            axes[1].plot(rd, rsl, label='Real')
            axes[1].plot(id_, isl, '--', label='Imag')
            axes[1].set_xlabel(r'Depth ($\AA$)')
            axes[1].legend()
            if xlim:
                axes[1].set_xlim(xlim)
            plt.tight_layout()
            plt.show()

    def make_update_handler(mat, pt):
        def handler(_):
            energy, _ = get_current()
            ww = all_widgets[mat][pt]
            obj = objectives_dict[energy]
            for p in obj.parameters.flattened():
                if f'{mat} - {pt}' in p.name.lower():
                    p.setp(bounds=(ww['lo'].value, ww['hi'].value),
                           vary=ww['vary'].value)
                    status.value = (f'<b>Updated</b> {p.name} at {energy} eV: '
                                    f'[{ww["lo"].value}, {ww["hi"].value}] '
                                    f'vary={ww["vary"].value}')
                    break
            refresh_plot()
        return handler

    def make_reset_handler(mat, pt):
        def handler(_):
            refresh_widgets()
            status.value = f'<i>Reset {mat} – {pt}.</i>'
        return handler

    for mat in material_names:
        for pt in param_types:
            all_widgets[mat][pt]['update'].on_click(make_update_handler(mat, pt))
            all_widgets[mat][pt]['reset'].on_click(make_reset_handler(mat, pt))

    def on_energy(change):
        if change['name'] == 'value':
            refresh_widgets()
            refresh_plot()

    def on_tab(change):
        if change['name'] == 'selected_index':
            refresh_widgets()

    energy_dd.observe(on_energy, names='value')
    mat_tabs.observe(on_tab, names='selected_index')

    def on_save(_):
        energy, _ = get_current()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'updated_objective_{energy}eV.pkl')
        else:
            path = f'updated_objective_{energy}eV.pkl'
        with open(path, 'wb') as fh:
            pickle.dump(objectives_dict[energy], fh)
        status.value = f'<b>Saved</b> objective for {energy} eV → {path}'

    save_btn.on_click(on_save)

    refresh_widgets()
    refresh_plot()

    return widgets.VBox([
        widgets.HBox([energy_dd, save_btn]),
        mat_tabs,
        plot_out,
        status,
    ])
