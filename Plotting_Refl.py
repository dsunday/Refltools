import numpy as np
from copy import deepcopy

from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel, MaterialSLD
from refnx.reflect.structure import isld_profile

import matplotlib.pyplot as plt
import pickle




def plot_parameter_results(results_df, param_type='sld', model_names=None, figsize=(10, 6), show_substrate=False):
    """
    Plot parameters from fitting results by parameter type.
    
    Args:
        results_df: DataFrame containing fitting results
        param_type: Type of parameter to plot ('sld', 'isld', 'thick', 'rough', or 'model')
        model_names: List of model names to include (None for all)
        figsize: Figure size as a tuple (width, height)
        show_substrate: Whether to include substrate layers (Si, SiO2) in the plot
        
    Returns:
        matplotlib figure and axes
    """
    # Define filter patterns based on parameter type
    if param_type == 'sld':
        filter_pattern = ' sld'  # Space before sld to avoid matching isld
        title = 'Real Component (SLD) Parameters'
    elif param_type == 'isld':
        filter_pattern = 'isld'
        title = 'Imaginary Component (iSLD) Parameters'
    elif param_type == 'thick':
        filter_pattern = 'thick'
        title = 'Thickness Parameters'
    elif param_type == 'rough':
        filter_pattern = 'rough'
        title = 'Roughness Parameters'
    elif param_type == 'model':
        # Model parameters (scale, bkg, dq)
        filter_pattern = '(scale|bkg|dq)'
        title = 'Model Parameters'
    else:
        raise ValueError("param_type must be 'sld', 'isld', 'thick', 'rough', or 'model'")
    
    # Filter parameters based on pattern
    filtered_params = results_df[results_df['parameter'].str.contains(filter_pattern, case=False, regex=True)]
    
    # Filter by model names if specified
    if model_names:
        filtered_params = filtered_params[filtered_params['model_name'].isin(model_names)]
    
    # Remove substrate and air layers unless specifically requested
    # Always remove air layers
    filtered_params = filtered_params[~filtered_params['parameter'].str.contains('air', case=False)]
    
    # Remove substrate layers (Si, SiO2) unless requested
    if not show_substrate:
        substrate_mask = (
            filtered_params['parameter'].str.contains('Si -', case=False) |
            filtered_params['parameter'].str.contains('SiO2 -', case=False)
        )
        filtered_params = filtered_params[~substrate_mask]
    
    # Exit if no data to plot
    if filtered_params.empty:
        print(f"No {param_type} parameters found in the results DataFrame (after filtering substrate layers).")
        return None, None
    
    # Group by model_name and parameter
    grouped = filtered_params.groupby(['model_name', 'parameter'])
    
    # Determine number of unique model/parameter combinations
    n_params = len(grouped)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up x-positions and labels
    x_pos = np.arange(n_params)
    x_labels = []
    
    # Colors for different models
    model_colors = plt.cm.tab10.colors
    model_names_unique = filtered_params['model_name'].unique()
    model_color_map = {name: model_colors[i % len(model_colors)] 
                       for i, name in enumerate(model_names_unique)}
    
    # Plot data points and error bars
    for i, ((model_name, param_name), group) in enumerate(grouped):
        # Get parameter info from the most recent entry for this model/parameter
        param_data = group.iloc[-1]
        
        # Parameter value, bounds, and error
        value = param_data['value']
        lower_bound = param_data['bound_low']
        upper_bound = param_data['bound_high']
        stderr = param_data['stderr'] or 0
        
        # Color based on model
        color = model_color_map[model_name]
        
        # Calculate bound distances for error bars (matplotlib expects distances not absolute values)
        lower_distance = value - lower_bound if lower_bound is not None else 0
        upper_distance = upper_bound - value if upper_bound is not None else 0
        
        # Plot point with error bars
        ax.errorbar(
            x=i, 
            y=value,
            yerr=stderr,
            fmt='o',
            markersize=8,
            capsize=5,
            color=color,
            label=model_name if model_name not in [l.get_label() for l in ax.get_lines()] else ""
        )
        
        # Plot bounds as a shaded region
        if lower_bound is not None and upper_bound is not None:
            ax.fill_between(
                [i-0.25, i+0.25],
                lower_bound,
                upper_bound,
                alpha=0.2,
                color=color
            )
        
        # Create label for x-axis
        if '.' in param_name:
            material = param_name.split('.')[0]
            short_param = param_name.split('.')[-1]
            x_labels.append(f"{material}\n{short_param}")
        elif ' - ' in param_name:
            # Handle "PR - sld" style parameter names
            parts = param_name.split(' - ')
            material = parts[0]
            short_param = parts[1]
            x_labels.append(f"{material}\n{short_param}")
        else:
            x_labels.append(param_name)
    
    # Set x-axis labels and positions
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add labels and title
    ax.set_ylabel('Parameter Value')
    ax.set_title(title + ' with Bounds and Errors')
    
    # Add grid and legend if multiple models
    ax.grid(alpha=0.3)
    if len(model_names_unique) > 1:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    
    return fig, ax

def plot_all_parameter_types(results_df, model_names=None, figsize=(12, 10), show_substrate=False):
    """
    Create a set of subplots for all parameter types.
    
    Args:
        results_df: DataFrame containing fitting results
        model_names: List of model names to include (None for all)
        figsize: Figure size as a tuple (width, height)
        show_substrate: Whether to include substrate layers (Si, SiO2) in the plots
        
    Returns:
        matplotlib figure and axes
    """
    # Define parameter types to plot
    param_types = ['sld', 'isld', 'thick', 'rough', 'model']
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(param_types), 1, figsize=figsize, sharex=False)
    
    # Plot each parameter type
    for i, param_type in enumerate(param_types):
        # Filter parameters based on pattern
        if param_type == 'sld':
            filter_pattern = ' sld'  # Space before sld to avoid matching isld
            title = 'Real Component (SLD) Parameters'
        elif param_type == 'isld':
            filter_pattern = 'isld'
            title = 'Imaginary Component (iSLD) Parameters'
        elif param_type == 'thick':
            filter_pattern = 'thick'
            title = 'Thickness Parameters'
        elif param_type == 'rough':
            filter_pattern = 'rough'
            title = 'Roughness Parameters'
        elif param_type == 'model':
            # Model parameters (scale, bkg, dq)
            filter_pattern = '(scale|bkg|dq)'
            title = 'Model Parameters'
        
        # Filter parameters based on pattern
        filtered_params = results_df[results_df['parameter'].str.contains(filter_pattern, case=False, regex=True)]
        
        # Filter by model names if specified
        if model_names:
            filtered_params = filtered_params[filtered_params['model_name'].isin(model_names)]
        
        # Remove air layers always
        filtered_params = filtered_params[~filtered_params['parameter'].str.contains('air', case=False)]
        
        # Remove substrate layers (Si, SiO2) unless requested
        if not show_substrate:
            substrate_mask = (
                filtered_params['parameter'].str.contains('Si -', case=False) |
                filtered_params['parameter'].str.contains('SiO2 -', case=False)
            )
            filtered_params = filtered_params[~substrate_mask]
        
        # Check if we have data to plot
        if filtered_params.empty:
            axes[i].text(0.5, 0.5, f"No {param_type} parameters found", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(title)
            continue
        
        # Group by model_name and parameter
        grouped = filtered_params.groupby(['model_name', 'parameter'])
        
        # Set up x-positions and labels
        n_params = len(grouped)
        x_pos = np.arange(n_params)
        x_labels = []
        
        # Colors for different models
        model_colors = plt.cm.tab10.colors
        model_names_unique = filtered_params['model_name'].unique()
        model_color_map = {name: model_colors[j % len(model_colors)] 
                           for j, name in enumerate(model_names_unique)}
        
        # Plot data points and error bars
        for j, ((model_name, param_name), group) in enumerate(grouped):
            # Get parameter info from the most recent entry
            param_data = group.iloc[-1]
            
            # Parameter value, bounds, and error
            value = param_data['value']
            lower_bound = param_data['bound_low']
            upper_bound = param_data['bound_high']
            stderr = param_data['stderr'] or 0
            
            # Color based on model
            color = model_color_map[model_name]
            
            # Plot point with error bars
            axes[i].errorbar(
                x=j, 
                y=value,
                yerr=stderr,
                fmt='o',
                markersize=8,
                capsize=5,
                color=color,
                label=model_name if model_name not in [l.get_label() for l in axes[i].get_lines()] else ""
            )
            
            # Plot bounds as a shaded region
            if lower_bound is not None and upper_bound is not None:
                axes[i].fill_between(
                    [j-0.25, j+0.25],
                    lower_bound,
                    upper_bound,
                    alpha=0.2,
                    color=color
                )
            
            # Create label for x-axis
            if '.' in param_name:
                material = param_name.split('.')[0]
                short_param = param_name.split('.')[-1]
                x_labels.append(f"{material}\n{short_param}")
            elif ' - ' in param_name:
                # Handle "PR - sld" style parameter names
                parts = param_name.split(' - ')
                material = parts[0]
                short_param = parts[1]
                x_labels.append(f"{material}\n{short_param}")
            else:
                x_labels.append(param_name)
        
        # Set x-axis labels and positions
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Add labels and title
        axes[i].set_ylabel('Value')
        axes[i].set_title(title)
        
        # Add grid and legend if multiple models
        axes[i].grid(alpha=0.3)
        if len(model_names_unique) > 1:
            handles, labels = axes[i].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[i].legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    return fig, axes


def modelcomparisonplot(obj_list, structure_list, shade_start=None, 
                        fig_size_w=16, colors=None, profile_shift=-10, xlim=None):
    """
    Create a flexible, comprehensive comparison plot for multiple reflectometry models.
    
    Args:
        obj_list: List of objective functions for each model (data will be extracted from these)
        structure_list: List of structure objects for each model
        shade_start: List of starting positions for layer shading (None for auto-detection)
        fig_size_w: Width of the figure (height is fixed at 8)
        colors: List of colors for layer shading (None for default colors)
        profile_shift: Shift applied to depth profiles (default: -10)
        xlim: Custom x-axis limits for SLD plots as [min, max] (None for auto)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Determine number of plots
    plot_number = len(obj_list)
    
    # Set up default colors if not provided
    if colors is None:
        colors = ['silver','grey','blue', 'violet','orange', 'purple',  'red', 'green', 'yellow']
    
    # Create figure and axes
    if plot_number == 1:
        fig, axes = plt.subplots(2, 1, figsize=(fig_size_w, 8))
        # For single plot, axes will be a 1D array with 2 elements
        ax_refl = axes[0]
        ax_sld = axes[1]
    else:
        fig, axes = plt.subplots(2, plot_number, figsize=(fig_size_w, 8))
    
    # Calculate chi-squared values and relative goodness of fit
    chi_values = np.zeros((plot_number, 2))
    for i in range(plot_number):
        chi_values[i, 0] = obj_list[i].chisqr()
    
    # Calculate relative goodness of fit
    chi_values[:, 1] = np.round(chi_values[:, 0] / chi_values[0, 0], 2)
    
    # Process each model
    for i in range(plot_number):
        # Get the current axes (handle both 1D and 2D arrays)
        if plot_number == 1:
            # Already assigned above for the single plot case
            pass
        else:
            ax_refl = axes[0, i]
            ax_sld = axes[1, i]
        
        # Extract data from objective
        data = obj_list[i].data
        
        # Plot reflectivity data
        ax_refl.plot(data.data[0], data.data[1], label='Data')
        ax_refl.set_yscale('log')
        ax_refl.plot(data.data[0], obj_list[i].model(data.data[0]), label='Simulation')
        ax_refl.legend(loc='upper right')
        ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_refl.set_ylabel('Reflectivity (a.u)')
        ax_refl.text(0.125, 1, f'RelGF {chi_values[i, 1]}', size=12, 
                   horizontalalignment='center', verticalalignment='bottom')
        
        # Plot SLD profiles
        Real_depth, Real_SLD, Imag_Depth, Imag_SLD = profileflip(structure_list[i])
        
        # Set initial plot to determine axis ranges
        ax_sld.plot(Real_depth + profile_shift, Real_SLD, color='blue', label='Real SLD', zorder=2)
        ax_sld.plot(Imag_Depth + profile_shift, Imag_SLD, linestyle='dashed', color='blue', label='Im SLD', zorder=2)
        
        # Set custom xlim if provided
        if xlim is not None:
            ax_sld.set_xlim(xlim)
        
        # Shade layers
        slabs = structure_list[i].slabs()
        num_layers = len(slabs)
        
        # Get parameter values and calculate thicknesses
        pvals = obj_list[i].parameters.pvals
        
        # Auto-calculate shade_start if not provided
        if shade_start is None or len(shade_start) <= i:
            current_shade_start = 0  # Start at 0 instead of the left edge
        else:
            current_shade_start = shade_start[i]
        
        # Calculate layer positions
        thicknesses = [current_shade_start]
        
        # Extract thicknesses from slabs or parameter values
        for j in range(1, num_layers):
            # Access thickness parameters consistently regardless of layer count
            thickness_index = (num_layers - j - 1) * 5 + 9  # Based on your pattern
            
            if thickness_index < len(pvals):
                thicknesses.append(thicknesses[-1] + pvals[thickness_index])
            else:
                # Fallback to using slab thickness if parameter index is out of range
                thicknesses.append(thicknesses[-1] + slabs[j]['thickness'])
        
        # Add silver shading between 0 and the first layer
        if len(thicknesses) > 0:
            ax_sld.axvspan(0, thicknesses[0], color='silver', alpha=0.3, zorder=0)
        
        # Shade each layer (starting from the first layer, not the top)
        for j in range(len(thicknesses) - 1):
            color_index = min(j, len(colors) - 1)
            ax_sld.axvspan(thicknesses[j], thicknesses[j + 1], 
                          color=colors[color_index], alpha=0.2, zorder=1)
        
        # # Add dark grey shading for substrate
        # if len(thicknesses) > 0:
        #     substrate_start = thicknesses[-1]  # Start at the end of the last layer
        #     substrate_end = ax_sld.get_xlim()[1]  # End at the right edge of the plot
        #     ax_sld.axvspan(substrate_start, substrate_end, color='darkgrey', alpha=0.3, zorder=0)
        
        # Add legend and axis labels
        ax_sld.legend(loc='upper right')
        ax_sld.set_xlabel(r'Distance from Si ($\AA$)')
        ax_sld.set_ylabel(r'SLD $(10^{-6})$ $\AA^{-2}$')
    
    plt.tight_layout()
    
    # Return the appropriate axes structure
    return fig, axes

def profileflip(structure, depth_shift=0):

    Real_depth,Real_SLD = structure.sld_profile()
    Imag_depth,Imag_SLD = isld_profile(structure.slabs())

    Real_depth=(Real_depth-max(Real_depth))*-1-depth_shift
    Imag_depth=(Imag_depth-max(Imag_depth))*-1-depth_shift

    return Real_depth,Real_SLD, Imag_depth, Imag_SLD