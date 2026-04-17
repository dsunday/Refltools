import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rc, gridspec
import os
import json
import re
from typing import Dict, Tuple, Optional, Union, List
import matplotlib.colors as mcolors
import pandas as pd

from NEXAFS import load_spectrum_data,  edge_bl_func, gaussian    #plot_simulated_nexafs_spectrum,  simulate_nexafs_spectrum_for_plotting
from scipy.optimize import curve_fit, least_squares, minimize, differential_evolution, dual_annealing

# Module-level dictionary to store peak energy to color mappings for synchronized plotting
_peak_energy_color_map = {}  # Maps energy (rounded) -> color hex string


def parse_peak_params(peak_params: Dict) -> Tuple[Dict, Dict]:
    """
    Parse the peak parameters dictionary to separate values from bounds/fit flags.
    
    Parameters:
    peak_params : dict
        Dictionary containing peak parameters in the format:
        {"peak_1_width": 10, "peak_1_width_bounds": (5,15,True), ...}
    
    Returns:
    tuple
        (values_dict, bounds_dict) where values_dict contains the parameter values
        and bounds_dict contains bounds and fit flags
    """
    values = {}
    bounds = {}
    
    for key, value in peak_params.items():
        if key.endswith('_bounds'):
            # This is a bounds specification
            param_name = key[:-7]  # Remove '_bounds' suffix
            bounds[param_name] = value
        else:
            # This is a parameter value
            values[key] = value
    
    return values, bounds

def get_peak_names(peak_params: Dict) -> list:
    """
    Extract unique peak names from the parameter dictionary.
    
    Parameters:
    peak_params : dict
        Dictionary containing peak parameters
    
    Returns:
    list
        List of peak names (e.g., ['peak_1', 'peak_2', ...])
    """
    peak_names = set()
    for key in peak_params.keys():
        if not key.endswith('_bounds'):
            # Extract peak name (everything before the last underscore)
            parts = key.split('_')
            if len(parts) >= 3:  # peak_X_parameter format
                peak_name = '_'.join(parts[:-1])
                peak_names.add(peak_name)
    
    return sorted(list(peak_names))

def validate_peak_params(peak_params: Dict) -> bool:
    """
    Validate that all required parameters are present for each peak.
    
    Parameters:
    peak_params : dict
        Dictionary containing peak parameters
    
    Returns:
    bool
        True if valid, raises ValueError if invalid
    """
    peak_names = get_peak_names(peak_params)
    required_params = ['width', 'height', 'energy']
    
    for peak_name in peak_names:
        for param in required_params:
            param_key = f"{peak_name}_{param}"
            if param_key not in peak_params:
                raise ValueError(f"Missing required parameter: {param_key}")
    
    return True



def simulate_nexafs_spectrum(x: np.ndarray = None, 
                           peak_params: Dict = None,
                           edge_params: Optional[Dict] = None,
                           baseline: float = 0.0,
                           experimental_intensity: Optional[np.ndarray] = None,
                           data: Optional[np.ndarray] = None,
                           plot: bool = False,
                           plot_kwargs: Optional[Dict] = None,
                           xlim: Optional[Tuple[float, float]] = None,
                           ylim: Optional[Tuple[float, float]] = None) -> Union[Tuple[np.ndarray, np.ndarray], plt.Figure]:
    """
    Simulate a NEXAFS spectrum with Gaussian peaks and optional step edge.
    Automatically uses high-density energy axis if input has sparse spacing.
    
    Parameters:
    x : numpy array, optional
        Energy values (experimental energy axis). If not provided, will be extracted from 'data' parameter.
    peak_params : dict, optional
        Peak parameters dictionary. If not provided, will be extracted from 'data' parameter.
    edge_params : dict, optional
        Edge parameters dictionary
    baseline : float
        Baseline value
    experimental_intensity : numpy array, optional
        Experimental intensity data for plotting comparison. If not provided, will be extracted from 'data' parameter.
    data : numpy array, optional
        2D array with shape (N, 2) where first column is energy and second column is intensity.
        If provided, will override x and experimental_intensity parameters.
    plot : bool
        If True, automatically create plot and return figure instead of data
    plot_kwargs : dict, optional
        Additional keyword arguments for plotting function
    xlim : tuple, optional
        Tuple of (xmin, xmax) to set the x-axis limits. If None, uses automatic scaling.
    ylim : tuple, optional
        Tuple of (ymin, ymax) to set the y-axis limits. If None, uses automatic scaling.
        
    Returns:
    tuple or matplotlib.figure.Figure
        If plot=False: (energy_sim, spectrum_sim) tuple
        If plot=True: matplotlib Figure object
    """
    # Handle data parameter - extract energy and intensity if provided
    if data is not None:
        if data.shape[1] != 2:
            raise ValueError("Data array must have 2 columns (energy, intensity)")
        x = data[:, 0]
        experimental_intensity = data[:, 1]
    
    # Validate required parameters
    if x is None:
        raise ValueError("Either 'x' parameter or 'data' parameter must be provided")
    if peak_params is None:
        raise ValueError("peak_params must be provided")
    # Check energy spacing
    energy_spacing = np.median(np.diff(x))
    
    if energy_spacing > 0.1:
        # Create high-density energy axis with 0.1 eV steps
        energy_sim = np.arange(x.min(), x.max() + 0.1, 0.1)
        print(f"Using high-density energy axis: {energy_spacing:.3f} eV -> 0.1 eV spacing")
        print(f"Points: {len(x)} -> {len(energy_sim)}")
    else:
        # Use original energy axis
        energy_sim = x.copy()
    
    # Initialize spectrum with baseline
    spectrum = np.full_like(energy_sim, baseline, dtype=float)
    
    # Add step edge if provided
    if edge_params is not None:
        edge_contribution = edge_bl_func(energy_sim, 
                                       edge_params['location'],
                                       edge_params['height'], 
                                       edge_params['width'],
                                       edge_params['decay'])
        spectrum += edge_contribution
    
    # Add Gaussian peaks
    peak_names = get_peak_names(peak_params)
    for peak_name in peak_names:
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]
        energy = peak_params[f"{peak_name}_energy"]
        
        # Add Gaussian peak to spectrum
        peak_contribution = gaussian(energy_sim, height, energy, width)
        spectrum += peak_contribution
    
    # If plotting requested, create plot
    if plot:
        # Determine what intensity data to use for plotting
        if experimental_intensity is not None:
            # Use original experimental data without any interpolation
            intensity_for_plot = experimental_intensity
            energy_for_plot = x  # Original experimental energy axis
        else:
            # For pure simulation, interpolate simulated spectrum back to original energy axis
            intensity_for_plot = np.interp(x, energy_sim, spectrum)
            energy_for_plot = x
        
        # Set up plotting arguments
        if plot_kwargs is None:
            plot_kwargs = {}
        
        # Set default title if not provided
        if 'title' not in plot_kwargs:
            if experimental_intensity is not None:
                plot_kwargs['title'] = "NEXAFS Spectrum with Initial Parameters"
            else:
                plot_kwargs['title'] = "Simulated NEXAFS Spectrum"
        
        # Add xlim and ylim to plot_kwargs if provided
        if xlim is not None:
            plot_kwargs['xlim'] = xlim
        if ylim is not None:
            plot_kwargs['ylim'] = ylim
        
        # Call plotting function with original experimental data
        fig = plot_simulated_nexafs_spectrum(
            energy_for_plot, intensity_for_plot, peak_params, edge_params, baseline,
            use_experimental_data=True, **plot_kwargs
        )
        
        return fig
    
    else:
        # Return simulation data
        return energy_sim, spectrum

def get_fit_parameters(peak_params: Dict, edge_params: Optional[Dict] = None) -> Dict:
    """
    Extract parameters that should be fitted (where bounds indicate fit=True).
    
    Parameters:
    peak_params : dict
        Dictionary containing peak parameters
    edge_params : dict, optional
        Dictionary containing edge parameters with bounds
    
    Returns:
    dict
        Dictionary of parameters to fit with their current values and bounds
    """
    values, bounds = parse_peak_params(peak_params)
    fit_params = {}
    
    # Check peak parameters
    for param_name, bound_info in bounds.items():
        if len(bound_info) >= 3 and bound_info[2]:  # Third element is fit flag
            if param_name in values:
                fit_params[param_name] = {
                    'value': values[param_name],
                    'bounds': (bound_info[0], bound_info[1]),
                    'fit': True
                }
    
    # Check edge parameters if provided
    if edge_params is not None:
        edge_values, edge_bounds = parse_edge_params(edge_params)
        for param_name, bound_info in edge_bounds.items():
            if len(bound_info) >= 3 and bound_info[2]:  # Third element is fit flag
                if param_name in edge_values:
                    fit_params[f"edge_{param_name}"] = {
                        'value': edge_values[param_name],
                        'bounds': (bound_info[0], bound_info[1]),
                        'fit': True
                    }
    
    return fit_params

def parse_edge_params(edge_params: Dict) -> Tuple[Dict, Dict]:
    """
    Parse edge parameters dictionary to separate values from bounds/fit flags.
    
    Parameters:
    edge_params : dict
        Dictionary containing edge parameters and bounds
    
    Returns:
    tuple
        (values_dict, bounds_dict)
    """
    values = {}
    bounds = {}
    
    for key, value in edge_params.items():
        if key.endswith('_bounds'):
            # This is a bounds specification
            param_name = key[:-7]  # Remove '_bounds' suffix
            bounds[param_name] = value
        else:
            # This is a parameter value
            values[key] = value
    
    return values, bounds

def create_fitting_function(peak_params: Dict, edge_params: Optional[Dict] = None, 
                          baseline: float = 0.0):
    """
    Create a fitting function for scipy.optimize with the current parameter structure.
    This function uses the experimental energy axis exactly as provided.
    
    Parameters:
    peak_params : dict
        Dictionary containing all peak parameters
    edge_params : dict, optional
        Dictionary containing all edge parameters
    baseline : float
        Baseline value
    
    Returns:
    function
        Fitting function that takes (x, *fit_params) and returns spectrum
    """
    # Get all parameter dictionaries
    peak_values, peak_bounds = parse_peak_params(peak_params)
    edge_values = {}
    edge_bounds = {}
    
    if edge_params is not None:
        edge_values, edge_bounds = parse_edge_params(edge_params)
    
    # Get list of parameters to fit
    fit_param_names = []
    
    # Peak parameters to fit
    for param_name, bound_info in peak_bounds.items():
        if len(bound_info) >= 3 and bound_info[2]:  # fit flag is True
            fit_param_names.append(param_name)
    
    # Edge parameters to fit
    for param_name, bound_info in edge_bounds.items():
        if len(bound_info) >= 3 and bound_info[2]:  # fit flag is True
            fit_param_names.append(f"edge_{param_name}")
    
    def fitting_function(x, *fit_values):
        """
        Function that scipy.optimize will call during fitting.
        Uses experimental energy axis exactly as provided.
        
        Parameters:
        x : array
            Energy values (experimental axis)
        *fit_values : tuple
            Values for the parameters being fitted
        
        Returns:
        array
            Calculated spectrum at experimental energy points
        """
        # Create copies of parameter dictionaries
        current_peak_params = peak_params.copy()
        current_edge_params = edge_params.copy() if edge_params is not None else None
        
        # Update with fitted values
        for i, param_name in enumerate(fit_param_names):
            if param_name.startswith('edge_'):
                # This is an edge parameter
                edge_param = param_name[5:]  # Remove 'edge_' prefix
                if current_edge_params is not None:
                    current_edge_params[edge_param] = fit_values[i]
            else:
                # This is a peak parameter
                current_peak_params[param_name] = fit_values[i]
        
        # Calculate spectrum using experimental energy axis (no high-density modification)
        return simulate_nexafs_spectrum_for_fitting(x, current_peak_params, 
                                                  current_edge_params, baseline)
    
    return fitting_function, fit_param_names


def fit_nexafs_spectrum(data: Union[str, np.ndarray, pd.DataFrame],
                       peak_params: Dict,
                       edge_params: Optional[Dict] = None,
                       baseline: float = 0.0,
                       method: str = 'lm',
                       max_iterations: int = 1000,
                       tolerance: float = 1e-8,
                       algorithm: str = 'curve_fit',
                       global_opt_params: Optional[Dict] = None,
                       energy_range: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Dict:
    """
    Fit NEXAFS spectrum parameters to experimental data with multiple optimization algorithms.
    Fitting always uses experimental energy density, not high-density simulation axis.
    
    Parameters:
    data : str, numpy array, or pandas DataFrame
        Experimental data with columns [energy, intensity]
    peak_params : dict
        Dictionary containing peak parameters with bounds and fit flags
    edge_params : dict, optional
        Dictionary containing edge parameters with bounds and fit flags
    baseline : float
        Baseline value
    method : str
        Fitting method for curve_fit ('lm', 'trf', 'dogbox')
    max_iterations : int
        Maximum number of fitting iterations
    tolerance : float
        Fitting tolerance
    algorithm : str
        Optimization algorithm to use:
        - 'curve_fit': scipy.optimize.curve_fit (Levenberg-Marquardt or Trust Region)
        - 'least_squares': scipy.optimize.least_squares (more robust)
        - 'minimize': scipy.optimize.minimize (general optimization)
        - 'differential_evolution': Global optimization using differential evolution
        - 'dual_annealing': Global optimization using dual annealing
    global_opt_params : dict, optional
        Additional parameters for global optimization algorithms
    energy_range : tuple, optional
        Optional (emin, emax) energy window in eV to restrict which points are used for fitting.
        Either bound may be None to indicate no bound on that side. Window is inclusive.
    
    Returns:
    dict
        Dictionary containing fitting results
    """
    
    # Load experimental data
    spectrum_data = load_spectrum_data(data)
    energy_exp_full = spectrum_data[:, 0]
    intensity_exp_full = spectrum_data[:, 1]

    # Restrict fit to an energy window if requested
    fit_energy_range = None
    if energy_range is not None:
        if not isinstance(energy_range, (tuple, list)) or len(energy_range) != 2:
            raise ValueError("energy_range must be a tuple/list of length 2: (emin, emax)")
        emin, emax = energy_range
        if emin is not None and emax is not None and emin > emax:
            raise ValueError(f"energy_range has emin > emax ({emin} > {emax})")

        mask = np.ones_like(energy_exp_full, dtype=bool)
        if emin is not None:
            mask &= (energy_exp_full >= float(emin))
        if emax is not None:
            mask &= (energy_exp_full <= float(emax))

        energy_exp = energy_exp_full[mask]
        intensity_exp = intensity_exp_full[mask]
        fit_energy_range = (emin, emax)
    else:
        energy_exp = energy_exp_full
        intensity_exp = intensity_exp_full
    
    # Create fitting function - this will use experimental energy axis for fitting
    fitting_func, fit_param_names = create_fitting_function(peak_params, edge_params, baseline)
    
    # Get initial values and bounds for fitting parameters
    fit_params_info = get_fit_parameters(peak_params, edge_params)
    
    if len(fit_params_info) == 0:
        raise ValueError("No parameters marked for fitting (set fit flag to True)")
    
    # Prepare initial values and bounds
    initial_values = []
    lower_bounds = []
    upper_bounds = []
    
    for param_name in fit_param_names:
        if param_name in fit_params_info:
            initial_values.append(fit_params_info[param_name]['value'])
            bounds = fit_params_info[param_name]['bounds']
            lower_bounds.append(bounds[0])
            upper_bounds.append(bounds[1])
        else:
            raise ValueError(f"Parameter {param_name} not found in fit parameters")
    
    initial_values = np.array(initial_values)

    # Validate fit data size (particularly important when restricting energy range)
    if len(energy_exp) < 3:
        raise ValueError(f"Not enough data points to fit ({len(energy_exp)} points).")
    if len(energy_exp) <= len(initial_values):
        raise ValueError(
            f"Not enough degrees of freedom: {len(energy_exp)} points for {len(initial_values)} fit parameters. "
            "Widen energy_range or reduce fitted parameters."
        )

    if len(energy_exp) >= 2:
        energy_spacing = float(np.median(np.diff(energy_exp)))
        spacing_str = f"{energy_spacing:.3f} eV"
    else:
        spacing_str = "N/A"

    if fit_energy_range is not None:
        print(
            f"Experimental data (fit window {fit_energy_range} eV): {len(energy_exp)} points "
            f"(from {len(energy_exp_full)} total), energy spacing: {spacing_str}"
        )
    else:
        print(f"Experimental data: {len(energy_exp)} points, energy spacing: {spacing_str}")
    
    # Set default global optimization parameters
    if global_opt_params is None:
        global_opt_params = {}
    
    try:
        if algorithm == 'curve_fit':
            bounds = (lower_bounds, upper_bounds)
            
            popt, pcov = curve_fit(
                fitting_func, 
                energy_exp,
                intensity_exp,
                p0=initial_values,
                bounds=bounds,
                method=method,
                maxfev=max_iterations,
                gtol=tolerance,
                ftol=tolerance,
                xtol=tolerance
            )
            
            # Calculate fitted spectrum using experimental energy axis
            fitted_spectrum = fitting_func(energy_exp, *popt)
            fit_success = True
            
        elif algorithm == 'differential_evolution':
            def objective_func(params):
                try:
                    predicted = fitting_func(energy_exp, *params)
                    return np.sum((intensity_exp - predicted)**2)
                except:
                    return np.inf
            
            bounds = list(zip(lower_bounds, upper_bounds))
            
            de_params = {
                'popsize': 15,
                'maxiter': max_iterations,
                'tol': tolerance,
                'seed': None,
                'workers': 1
            }
            de_params.update(global_opt_params)
            
            result = differential_evolution(
                objective_func,
                bounds,
                **de_params
            )
            
            popt = result.x
            pcov = None
            fitted_spectrum = fitting_func(energy_exp, *popt)
            fit_success = result.success
            
        # ... other algorithms would go here
        
        else:
            raise ValueError(f"Algorithm {algorithm} not implemented in this version")
        
        # Calculate residuals and statistics
        residuals = intensity_exp - fitted_spectrum
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((intensity_exp - np.mean(intensity_exp))**2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Update parameter dictionaries with fitted values
        fitted_peak_params = peak_params.copy()
        fitted_edge_params = edge_params.copy() if edge_params is not None else None
        
        for i, param_name in enumerate(fit_param_names):
            if param_name.startswith('edge_'):
                edge_param = param_name[5:]
                if fitted_edge_params is not None:
                    fitted_edge_params[edge_param] = popt[i]
            else:
                fitted_peak_params[param_name] = popt[i]
        
        # Calculate parameter uncertainties
        param_uncertainties = {}
        if pcov is not None:
            param_errors = np.sqrt(np.diag(pcov))
            for i, param_name in enumerate(fit_param_names):
                param_uncertainties[param_name] = param_errors[i]
        else:
            for param_name in fit_param_names:
                param_uncertainties[param_name] = 0.0
        
        # Prepare results dictionary
        fit_result = {
            'success': fit_success,
            'algorithm': algorithm,
            'fitted_values': dict(zip(fit_param_names, popt)),
            'parameter_uncertainties': param_uncertainties,
            'covariance_matrix': pcov,
            'r_squared': r_squared,
            'rmse': rmse,
            'residual_sum_squares': ss_res,
            'degrees_of_freedom': len(energy_exp) - len(popt)
        }
        
        results = {
            'fitted_peak_params': fitted_peak_params,
            'fitted_edge_params': fitted_edge_params,
            'fit_result': fit_result,
            'fitted_spectrum': fitted_spectrum,
            'residuals': residuals,
            'r_squared': r_squared,
            'rmse': rmse,
            'experimental_energy': energy_exp,
            'experimental_intensity': intensity_exp,
            'experimental_energy_full': energy_exp_full,
            'experimental_intensity_full': intensity_exp_full,
            'fit_energy_range': fit_energy_range
        }
        
        print(f"Fitting completed successfully using experimental energy axis ({len(energy_exp)} points)")
        return results
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        results = {
            'fitted_peak_params': peak_params,
            'fitted_edge_params': edge_params,
            'fit_result': {
                'success': False,
                'algorithm': algorithm,
                'error_message': str(e),
                'fitted_values': {},
                'parameter_uncertainties': {},
                'r_squared': 0,
                'rmse': np.inf
            },
            'fitted_spectrum': None,
            'residuals': None,
            'r_squared': 0,
            'rmse': np.inf,
            'experimental_energy': energy_exp,
            'experimental_intensity': intensity_exp,
            'experimental_energy_full': energy_exp_full,
            'experimental_intensity_full': intensity_exp_full,
            'fit_energy_range': fit_energy_range
        }
        return results

def estimate_parameter_uncertainties(fitting_func, energy_exp, intensity_exp, 
                                   popt, param_names, delta_frac=0.01):
    """
    Estimate parameter uncertainties using finite differences for algorithms
    that don't provide covariance matrices.
    
    Parameters:
    fitting_func : function
        The fitting function
    energy_exp : array
        Experimental energy values
    intensity_exp : array  
        Experimental intensity values
    popt : array
        Optimized parameter values
    param_names : list
        List of parameter names
    delta_frac : float
        Fractional change for finite difference calculation
    
    Returns:
    dict
        Dictionary of parameter uncertainties
    """
    uncertainties = {}
    
    # Calculate baseline residual
    fitted_spectrum = fitting_func(energy_exp, *popt)
    base_residual = np.sum((intensity_exp - fitted_spectrum)**2)
    
    for i, param_name in enumerate(param_names):
        # Calculate finite difference
        delta = abs(popt[i] * delta_frac) if popt[i] != 0 else delta_frac
        
        # Perturb parameter up
        popt_up = popt.copy()
        popt_up[i] += delta
        
        try:
            fitted_up = fitting_func(energy_exp, *popt_up)
            residual_up = np.sum((intensity_exp - fitted_up)**2)
            
            # Estimate uncertainty from change in residual
            # This is a rough approximation
            sensitivity = abs(residual_up - base_residual) / delta
            if sensitivity > 0:
                uncertainties[param_name] = np.sqrt(2 * base_residual / len(energy_exp)) / np.sqrt(sensitivity)
            else:
                uncertainties[param_name] = 0.0
                
        except:
            uncertainties[param_name] = 0.0
    
    return uncertainties


def plot_fit_results(results: Dict,
                    initial_peak_params: Optional[Dict] = None,
                    initial_edge_params: Optional[Dict] = None,
                    baseline: float = 0.0,
                    figsize: Tuple[float, float] = (12, 8),
                    peak_alpha: float = 0.3,
                    peak_colors: Optional[List[str]] = None,
                    save_path: Optional[str] = None,
                    dpi: int = 300) -> plt.Figure:
    """
    Plot NEXAFS fitting results with residuals and statistics.
    
    Parameters:
    results : dict
        Results dictionary from fit_nexafs_spectrum
    initial_peak_params : dict, optional
        Initial peak parameters for comparison
    initial_edge_params : dict, optional  
        Initial edge parameters for comparison
    baseline : float
        Baseline value
    figsize : tuple
        Figure size (width, height)
    peak_alpha : float
        Transparency for peak fill areas (0-1)
    peak_colors : list, optional
        List of colors for peaks
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure
        
    Returns:
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Extract data from results
    # `fit_nexafs_spectrum` may restrict the fit to an energy window; in that case
    # `experimental_energy`/`experimental_intensity` are the fit-subset, while
    # `*_full` contain the complete experimental arrays.
    energy_exp = results['experimental_energy']
    intensity_exp = results['experimental_intensity']
    energy_exp_full = results.get('experimental_energy_full', energy_exp)
    intensity_exp_full = results.get('experimental_intensity_full', intensity_exp)
    fit_energy_range = results.get('fit_energy_range', None)
    fitted_spectrum = results['fitted_spectrum']
    residuals = results['residuals']
    fitted_peak_params = results.get('fitted_peak_params')
    fitted_edge_params = results.get('fitted_edge_params')
    
    # Create figure with residuals subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                  gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    
    # Get peak names and set up colors
    peak_names = get_peak_names(fitted_peak_params)
    n_peaks = len(peak_names)
    
    if peak_colors is None:
        if n_peaks <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_peaks, 3)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_peaks))
        peak_colors = [mcolors.to_hex(color) for color in colors]
    
    # Plot experimental data (prefer full spectrum if available)
    ax1.plot(energy_exp_full, intensity_exp_full, 'ko', markersize=3, alpha=0.7, 
             label='Experimental Data', zorder=1)

    # Highlight fit window if used
    if fit_energy_range is not None:
        # Use the actual fit-subset extents for highlighting (handles None bounds)
        ax1.axvspan(float(np.min(energy_exp)), float(np.max(energy_exp)),
                    color='lightgrey', alpha=0.25, zorder=0,
                    label='Fit Window')
    
    # Plot initial spectrum if provided (use experimental energy axis)
    if initial_peak_params is not None:
        initial_spectrum = simulate_nexafs_spectrum_for_fitting(energy_exp, initial_peak_params, initial_edge_params, baseline)
        ax1.plot(energy_exp, initial_spectrum, 'g--', linewidth=2, alpha=0.7,
                 label='Initial Guess', zorder=3)
    
    # Plot fitted spectrum
    if fitted_spectrum is not None:
        ax1.plot(energy_exp, fitted_spectrum, 'r-', linewidth=2, 
                 label='Fitted Spectrum', zorder=5)
    
    # Get high-density energy axis for smooth component plotting
    energy_sim, _ = simulate_nexafs_spectrum_for_plotting(energy_exp, fitted_peak_params, fitted_edge_params, baseline)
    
    # Plot baseline if non-zero
    if baseline != 0:
        ax1.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Baseline = {baseline:.3f}', zorder=2)
    
    # Plot step edge if present (using high-density axis)
    if fitted_edge_params is not None:
        edge_spectrum = edge_bl_func(energy_sim, 
                                   fitted_edge_params['location'],
                                   fitted_edge_params['height'],
                                   fitted_edge_params['width'], 
                                   fitted_edge_params['decay'])
        edge_spectrum += baseline
        ax1.plot(energy_sim, edge_spectrum, 'b-', linewidth=2, alpha=0.8,
                 label='Step Edge', zorder=3)
        
        # Shade under the step edge with light grey
        ax1.fill_between(energy_sim, baseline, edge_spectrum,
                        color='lightgrey', alpha=0.4, zorder=1)
    
    # Plot individual fitted peaks (using high-density axis)
    peak_total = np.full_like(energy_sim, baseline)
    if fitted_edge_params is not None:
        peak_total += edge_bl_func(energy_sim,
                                 fitted_edge_params['location'],
                                 fitted_edge_params['height'],
                                 fitted_edge_params['width'],
                                 fitted_edge_params['decay'])
    
    for i, peak_name in enumerate(peak_names):
        # Get fitted peak parameters
        width = fitted_peak_params[f"{peak_name}_width"]
        height = fitted_peak_params[f"{peak_name}_height"]  
        peak_energy = fitted_peak_params[f"{peak_name}_energy"]
        
        # Calculate individual peak using high-density axis
        peak_spectrum = gaussian(energy_sim, height, peak_energy, width)
        peak_spectrum_with_baseline = peak_spectrum + peak_total
        
        # Plot peak line
        color = peak_colors[i % len(peak_colors)]
        peak_label = f'{peak_energy:.1f} eV'
        ax1.plot(energy_sim, peak_spectrum_with_baseline, '-', 
                 color=color, linewidth=1.5, label=peak_label, zorder=4)
        
        # Fill area under peak
        ax1.fill_between(energy_sim, peak_total, peak_spectrum_with_baseline,
                        color=color, alpha=peak_alpha, zorder=2)
    
    # Formatting for main plot
    ax1.set_xlabel('Energy (eV)', fontsize=12)
    ax1.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax1.set_title('NEXAFS Spectrum Fitting Results', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add fitting statistics as text box
    if 'r_squared' in results:
        r_squared = results['r_squared']
        rmse = results['rmse']
        algorithm = results.get('fit_result', {}).get('algorithm', 'N/A')
        
        stats_text = f'R² = {r_squared:.4f}\nRMSE = {rmse:.4f}\nAlgorithm: {algorithm}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Plot residuals
    if residuals is not None:
        ax2.plot(energy_exp, residuals, 'ko-', markersize=2, linewidth=0.5, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.7)
        ax2.set_xlabel('Energy (eV)', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Fit Residuals', fontsize=12)
        
        # Calculate and display residual statistics
        if len(residuals) > 0:
            res_std = np.std(residuals)
            res_mean = np.mean(residuals)
            ax2.axhline(y=res_mean + 2*res_std, color='orange', linestyle='--', alpha=0.7, label='±2σ')
            ax2.axhline(y=res_mean - 2*res_std, color='orange', linestyle='--', alpha=0.7)
            ax2.text(0.02, 0.95, f'σ = {res_std:.4f}', transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_nexafs_spectrum(energy: np.ndarray,
                        intensity: np.ndarray,
                        peak_params: Dict,
                        edge_params: Optional[Dict] = None,
                        baseline: float = 0.0,
                        figsize: Tuple[float, float] = (12, 8),
                        peak_alpha: float = 0.3,
                        peak_colors: Optional[List[str]] = None,
                        title: str = "NEXAFS Spectrum",
                        save_path: Optional[str] = None,
                        dpi: int = 300) -> plt.Figure:
    """
    Plot NEXAFS spectrum with peak decomposition - works with either initial or fitted parameters.
    
    This general function can plot:
    - Initial spectrum with starting parameter guesses
    - Fitted spectrum with optimized parameters
    - Any spectrum with given peak and edge parameters
    
    Parameters:
    energy : numpy array
        Energy values
    intensity : numpy array
        Intensity values (experimental data or simulated spectrum)
    peak_params : dict
        Peak parameters dictionary (either initial guesses or fitted values)
        Format: {"peak_1_width": 1.5, "peak_1_height": 2.0, "peak_1_energy": 285.0, ...}
    edge_params : dict, optional
        Edge parameters dictionary (either initial guesses or fitted values)
        Format: {"location": 284.0, "height": 1.0, "width": 2.0, "decay": 0.1}
    baseline : float
        Baseline value
    figsize : tuple
        Figure size (width, height)
    peak_alpha : float
        Transparency for peak fill areas (0-1)
    peak_colors : list, optional
        List of colors for peaks. If None, uses default colormap
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure
        
    Returns:
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get peak names and set up colors
    peak_names = get_peak_names(peak_params)
    n_peaks = len(peak_names)
    
    if peak_colors is None:
        # Use a colormap to generate distinct colors
        if n_peaks <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_peaks, 3)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_peaks))
        peak_colors = [mcolors.to_hex(color) for color in colors]
    
    # Plot experimental/input data
    ax.plot(energy, intensity, 'ko', markersize=3, alpha=0.7, 
             label='Data', zorder=1)
    
    # Calculate and plot total spectrum
    total_spectrum = simulate_nexafs_spectrum(energy, peak_params, edge_params, baseline)
    ax.plot(energy, total_spectrum, 'r-', linewidth=2, 
             label='Total Spectrum', zorder=5)
    
    # Plot baseline if non-zero
    if baseline != 0:
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Baseline = {baseline:.3f}', zorder=2)
    
    # Plot step edge if present
    if edge_params is not None:
        edge_spectrum = edge_bl_func(energy, 
                                   edge_params['location'],
                                   edge_params['height'],
                                   edge_params['width'], 
                                   edge_params['decay'])
        edge_spectrum += baseline
        ax.plot(energy, edge_spectrum, 'b-', linewidth=2, alpha=0.8,
                 label='Step Edge', zorder=3)
    
    # Plot individual peaks
    peak_total = np.full_like(energy, baseline)
    if edge_params is not None:
        peak_total += edge_bl_func(energy,
                                 edge_params['location'],
                                 edge_params['height'],
                                 edge_params['width'],
                                 edge_params['decay'])
    
    for i, peak_name in enumerate(peak_names):
        # Get peak parameters
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]  
        peak_energy = peak_params[f"{peak_name}_energy"]
        
        # Calculate individual peak
        peak_spectrum = gaussian(energy, height, peak_energy, width)
        peak_spectrum_with_baseline = peak_spectrum + peak_total
        
        # Plot peak line
        color = peak_colors[i % len(peak_colors)]
        peak_label = f'{peak_name.replace("_", " ").title()}'
        ax.plot(energy, peak_spectrum_with_baseline, '-', 
                 color=color, linewidth=1.5, label=peak_label, zorder=4)
        
        # Fill area under peak
        ax.fill_between(energy, peak_total, peak_spectrum_with_baseline,
                        color=color, alpha=peak_alpha, zorder=2)
        
        # Add peak info text
        peak_center_idx = np.argmin(np.abs(energy - peak_energy))
        peak_y = peak_spectrum_with_baseline[peak_center_idx]
        ax.annotate(f'{peak_energy:.1f} eV\nFWHM: {width:.2f}', 
                    xy=(peak_energy, peak_y), xytext=(5, 10),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    ha='left', va='bottom')
    
    # Formatting
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def plot_nexafs_fit(results: Dict, 
                   peak_params: Dict,
                   edge_params: Optional[Dict] = None,
                   baseline: float = 0.0,
                   figsize: Tuple[float, float] = (12, 8),
                   show_residuals: bool = True,
                   peak_alpha: float = 0.3,
                   peak_colors: Optional[List[str]] = None,
                   save_path: Optional[str] = None,
                   dpi: int = 300) -> plt.Figure:
    """
    Plot NEXAFS spectrum fitting results with individual peak components.
    
    Parameters:
    results : dict
        Results dictionary from fit_nexafs_spectrum containing:
        - 'experimental_energy': Energy values
        - 'experimental_intensity': Intensity values
        - 'fitted_spectrum': Fitted spectrum
        - 'residuals': Residuals (data - fit)
        - 'fitted_peak_params': Fitted peak parameters
        - 'fitted_edge_params': Fitted edge parameters
        - 'r_squared': R-squared value
        - 'rmse': Root mean square error
        - 'fit_result': Dictionary with algorithm info
    peak_params : dict
        Peak parameters dictionary (used if fitted params not available)
    edge_params : dict, optional
        Edge parameters dictionary (used if fitted params not available)
    baseline : float
        Baseline value
    figsize : tuple
        Figure size (width, height)
    show_residuals : bool
        Whether to show residuals subplot
    peak_alpha : float
        Transparency for peak fill areas (0-1)
    peak_colors : list, optional
        List of colors for peaks. If None, uses default colormap
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure
        
    Returns:
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Extract data from results
    energy_exp = results['experimental_energy']
    intensity_exp = results['experimental_intensity']
    fitted_spectrum = results['fitted_spectrum']
    residuals = results['residuals']
    
    # Use fitted parameters if available, otherwise use input parameters
    if results.get('fitted_peak_params') is not None:
        plot_peak_params = results['fitted_peak_params']
    else:
        plot_peak_params = peak_params
        
    if results.get('fitted_edge_params') is not None:
        plot_edge_params = results['fitted_edge_params']
    else:
        plot_edge_params = edge_params
    
    # Create figure and subplots
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    
    # Get peak names and set up colors
    peak_names = get_peak_names(plot_peak_params)
    n_peaks = len(peak_names)
    
    if peak_colors is None:
        # Use a colormap to generate distinct colors
        if n_peaks <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_peaks, 3)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_peaks))
        peak_colors = [mcolors.to_hex(color) for color in colors]
    
    # Plot experimental data
    ax1.plot(energy_exp, intensity_exp, 'ko', markersize=3, alpha=0.7, 
             label='Experimental Data', zorder=1)
    
    # Plot fitted spectrum
    if fitted_spectrum is not None:
        ax1.plot(energy_exp, fitted_spectrum, 'r-', linewidth=2, 
                 label='Fitted Spectrum', zorder=5)
    
    # Plot baseline if non-zero
    if baseline != 0:
        ax1.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Baseline = {baseline:.3f}', zorder=2)
    
    # Plot step edge if present
    if plot_edge_params is not None:
        edge_spectrum = edge_bl_func(energy_exp, 
                                   plot_edge_params['location'],
                                   plot_edge_params['height'],
                                   plot_edge_params['width'], 
                                   plot_edge_params['decay'])
        edge_spectrum += baseline
        ax1.plot(energy_exp, edge_spectrum, 'b-', linewidth=2, alpha=0.8,
                 label='Step Edge', zorder=3)
    
    # Plot individual peaks
    peak_total = np.full_like(energy_exp, baseline)
    if plot_edge_params is not None:
        peak_total += edge_bl_func(energy_exp,
                                 plot_edge_params['location'],
                                 plot_edge_params['height'],
                                 plot_edge_params['width'],
                                 plot_edge_params['decay'])
    
    for i, peak_name in enumerate(peak_names):
        # Get peak parameters
        width = plot_peak_params[f"{peak_name}_width"]
        height = plot_peak_params[f"{peak_name}_height"]  
        energy = plot_peak_params[f"{peak_name}_energy"]
        
        # Calculate individual peak
        peak_spectrum = gaussian(energy_exp, height, energy, width)
        peak_spectrum_with_baseline = peak_spectrum + peak_total
        
        # Plot peak line
        color = peak_colors[i % len(peak_colors)]
        peak_label = f'{peak_name.replace("_", " ").title()}'
        ax1.plot(energy_exp, peak_spectrum_with_baseline, '-', 
                 color=color, linewidth=1.5, label=peak_label, zorder=4)
        
        # Fill area under peak
        ax1.fill_between(energy_exp, peak_total, peak_spectrum_with_baseline,
                        color=color, alpha=peak_alpha, zorder=2)
        
        # Add peak info text
        peak_center_idx = np.argmin(np.abs(energy_exp - energy))
        peak_y = peak_spectrum_with_baseline[peak_center_idx]
        ax1.annotate(f'{energy:.1f} eV\nFWHM: {width:.2f}', 
                    xy=(energy, peak_y), xytext=(5, 10),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    ha='left', va='bottom')
    
    # Formatting for main plot
    ax1.set_xlabel('Energy (eV)', fontsize=12)
    ax1.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax1.set_title('NEXAFS Spectrum Fitting Results', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add fitting statistics as text box
    if 'r_squared' in results:
        r_squared = results['r_squared']
        rmse = results['rmse']
        algorithm = results.get('fit_result', {}).get('algorithm', 'N/A')
        
        stats_text = f'R² = {r_squared:.4f}\nRMSE = {rmse:.4f}\nAlgorithm: {algorithm}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Plot residuals if requested
    if show_residuals and ax2 is not None and residuals is not None:
        ax2.plot(energy_exp, residuals, 'ko-', markersize=2, linewidth=0.5, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.7)
        ax2.set_xlabel('Energy (eV)', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Fit Residuals', fontsize=12)
        
        # Calculate and display residual statistics
        if len(residuals) > 0:
            res_std = np.std(residuals)
            res_mean = np.mean(residuals)
            ax2.axhline(y=res_mean + 2*res_std, color='orange', linestyle='--', alpha=0.7, label='±2σ')
            ax2.axhline(y=res_mean - 2*res_std, color='orange', linestyle='--', alpha=0.7)
            ax2.text(0.02, 0.95, f'σ = {res_std:.4f}', transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def print_fit_results(fitted_peak_params, fitted_edge_params, fitted_baseline, 
                     tolerance_percent=10, color_threshold_percent=5):
    """
    Print fit results in a formatted table with color-coded text based on bound proximity.
    
    Parameters:
    -----------
    fitted_peak_params : dict
        Dictionary containing fitted peak parameters with bounds
    fitted_edge_params : dict
        Dictionary containing fitted edge parameters
    fitted_baseline : float
        Fitted baseline value
    tolerance_percent : float, default 10
        Percentage threshold for orange warning (fitted value within X% of bounds)
    color_threshold_percent : float, default 5
        Percentage threshold for red warning (fitted value very close to bounds)
    """
    
    def get_color_code(value, lower_bound, upper_bound, fit_flag, tolerance=tolerance_percent, threshold=color_threshold_percent):
        """Determine color based on proximity to bounds and fit status"""
        # If parameter is not being fitted, color it grey
        if not fit_flag:
            return "grey"
        
        range_size = upper_bound - lower_bound
        if range_size == 0:
            return "red"  # No range, at bounds
        
        # Calculate how close the value is to each bound
        lower_distance = abs(value - lower_bound) / range_size * 100
        upper_distance = abs(value - upper_bound) / range_size * 100
        
        # Check if within threshold of either bound
        if lower_distance <= threshold or upper_distance <= threshold:
            return "red"
        elif lower_distance <= tolerance or upper_distance <= tolerance:
            return "orange"
        else:
            return "green"
    
    def colorize_text(text, color):
        """Apply color formatting to text"""
        color_codes = {
            "red": "\033[91m",      # Red
            "orange": "\033[93m",    # Yellow/Orange
            "green": "\033[92m",     # Green
            "grey": "\033[90m",      # Grey/Dark Gray
            "reset": "\033[0m"      # Reset
        }
        return f"{color_codes[color]}{text}{color_codes['reset']}"
    
    print("=" * 120)
    print("FIT RESULTS SUMMARY")
    print("=" * 120)
    print(f"Fitted Baseline: {fitted_baseline:.2f}")
    print()
    
    # Print header with proper spacing - headers are 6 characters each
    print(f"{'Peak':<6} {'Energy (eV)':<30} {'Width':<30} {'Intensity':<30}")
    print(f"{'#':<6} {'Lower':<6} {'Fitted':<7} {'Upper':<6} {'Lower':<6} {'Fitted':<7} {'Upper':<6} {'Lower':<6} {'Fitted':<7} {'Upper':<6}")
    print("-" * 120)
    
    # Extract and sort peaks by energy
    peak_data = []
    for key in fitted_peak_params.keys():
        if key.endswith('_energy'):
            peak_num = key.split('_')[1]
            energy_val = fitted_peak_params[f"peak_{peak_num}_energy"]
            peak_data.append((peak_num, energy_val))
    
    # Sort by energy value
    peak_data.sort(key=lambda x: x[1])
    
    # Print each peak with proper alignment
    for peak_num, energy_val in peak_data:
        # Get energy data
        energy_val = fitted_peak_params[f"peak_{peak_num}_energy"]
        energy_bounds = fitted_peak_params[f"peak_{peak_num}_energy_bounds"]
        energy_lower, energy_upper, energy_fit_flag = energy_bounds
        
        # Get intensity data
        intensity_val = fitted_peak_params[f"peak_{peak_num}_height"]
        intensity_bounds = fitted_peak_params[f"peak_{peak_num}_height_bounds"]
        intensity_lower, intensity_upper, intensity_fit_flag = intensity_bounds
        
        # Get width data
        width_val = fitted_peak_params[f"peak_{peak_num}_width"]
        width_bounds = fitted_peak_params[f"peak_{peak_num}_width_bounds"]
        width_lower, width_upper, width_fit_flag = width_bounds
        
        # Determine colors
        energy_color = get_color_code(energy_val, energy_lower, energy_upper, energy_fit_flag)
        intensity_color = get_color_code(intensity_val, intensity_lower, intensity_upper, intensity_fit_flag)
        width_color = get_color_code(width_val, width_lower, width_upper, width_fit_flag)
        
        # Format values with proper alignment - pad to match header width (6 characters)
        energy_lower_str = f"{energy_lower:6.1f}"
        energy_val_str = f"{energy_val:7.2f}"
        energy_upper_str = f"{energy_upper:6.1f}"
        
        intensity_lower_str = f"{intensity_lower:6.1f}"
        intensity_val_str = f"{intensity_val:7.2f}"
        intensity_upper_str = f"{intensity_upper:6.1f}"
        
        width_lower_str = f"{width_lower:6.1f}"
        width_val_str = f"{width_val:7.2f}"
        width_upper_str = f"{width_upper:6.1f}"
        
        # Apply colors after formatting
        energy_lower_colored = colorize_text(energy_lower_str, energy_color)
        energy_val_colored = colorize_text(energy_val_str, energy_color)
        energy_upper_colored = colorize_text(energy_upper_str, energy_color)
        
        width_lower_colored = colorize_text(width_lower_str, width_color)
        width_val_colored = colorize_text(width_val_str, width_color)
        width_upper_colored = colorize_text(width_upper_str, width_color)
        
        intensity_lower_colored = colorize_text(intensity_lower_str, intensity_color)
        intensity_val_colored = colorize_text(intensity_val_str, intensity_color)
        intensity_upper_colored = colorize_text(intensity_upper_str, intensity_color)
        
        # Print with proper spacing using the pre-formatted strings
        print(f"{peak_num:<6} {energy_lower_colored} {energy_val_colored} {energy_upper_colored} {width_lower_colored} {width_val_colored} {width_upper_colored} {intensity_lower_colored} {intensity_val_colored} {intensity_upper_colored}")
    
    print("-" * 120)
    
    # Print edge parameters
    print("\nEDGE PARAMETERS:")
    print("-" * 60)
    edge_location = fitted_edge_params["location"]
    edge_height = fitted_edge_params["height"]
    edge_width = fitted_edge_params["width"]
    edge_decay = fitted_edge_params["decay"]
    
    print(f"Location: {edge_location:.2f}")
    print(f"Height:   {edge_height:.2f}")
    print(f"Width:    {edge_width:.2f}")
    print(f"Decay:    {edge_decay:.3f}")
    
    print("\n" + "=" * 120)
    print("COLOR LEGEND:")
    print(f"{colorize_text('Green', 'green')}: Fitted value well within bounds")
    print(f"{colorize_text('Orange', 'orange')}: Fitted value within {tolerance_percent}% of bounds")
    print(f"{colorize_text('Red', 'red')}: Fitted value within {color_threshold_percent}% of bounds (at bounds)")
    print(f"{colorize_text('Grey', 'grey')}: Parameter not being fitted (fixed)")
    print("=" * 120)


def _get_color_for_peak_energy(peak_energy: float, 
                               energy_tolerance: Optional[float] = None,
                               low_energy_threshold: float = 292.0,
                               low_energy_tolerance: float = 0.1,
                               high_energy_tolerance: float = 0.3) -> str:
    """
    Get color for a peak energy, matching to existing colors if within tolerance.
    If no match found, assigns a new color from the colormap.
    
    Uses different tolerances based on energy:
    - Below low_energy_threshold (default 292 eV): low_energy_tolerance (default 0.1 eV)
    - Above low_energy_threshold: high_energy_tolerance (default 0.3 eV)
    
    Parameters:
    -----------
    peak_energy : float
        Energy value of the peak (eV)
    energy_tolerance : float, optional
        Override tolerance for matching peak energies (eV). If None, uses energy-based tolerance.
    low_energy_threshold : float, default 292.0
        Energy threshold (eV) below which low_energy_tolerance is used
    low_energy_tolerance : float, default 0.1
        Tolerance for matching peak energies below threshold (eV)
    high_energy_tolerance : float, default 0.3
        Tolerance for matching peak energies above threshold (eV)
        
    Returns:
    --------
    str
        Color hex string for the peak
    """
    global _peak_energy_color_map
    
    # Determine tolerance based on energy if not explicitly provided
    if energy_tolerance is None:
        if peak_energy < low_energy_threshold:
            energy_tolerance = low_energy_tolerance
        else:
            energy_tolerance = high_energy_tolerance
    
    # Check if any existing mapped energy is within tolerance
    # For matching, we need to check both the mapped energy's tolerance region
    for mapped_energy, color in _peak_energy_color_map.items():
        # Determine the appropriate tolerance for the mapped energy
        if mapped_energy < low_energy_threshold:
            mapped_tolerance = low_energy_tolerance
        else:
            mapped_tolerance = high_energy_tolerance
        
        # Use the larger tolerance to ensure symmetric matching
        effective_tolerance = max(energy_tolerance, mapped_tolerance)
        
        if abs(peak_energy - mapped_energy) <= effective_tolerance:
            return color
    
    # No match found - assign new color
    # Count how many unique energy positions we have
    n_unique_energies = len(_peak_energy_color_map)
    
    # Use tab10 for up to 10 unique energies, tab20 for more
    if n_unique_energies < 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        color = colors[n_unique_energies % 10]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        color = colors[n_unique_energies % 20]
    
    color_hex = mcolors.to_hex(color)
    
    # Store the mapping using the actual peak energy (not rounded)
    # This allows for precise matching within tolerance
    _peak_energy_color_map[peak_energy] = color_hex
    
    return color_hex


def reset_peak_color_map():
    """
    Reset the global peak energy to color mapping.
    Useful when starting a new analysis session or when you want to reset color assignments.
    """
    global _peak_energy_color_map
    _peak_energy_color_map = {}


def plot_simulated_nexafs_spectrum(energy: np.ndarray,
                                 intensity: np.ndarray,
                                 peak_params: Dict,
                                 edge_params: Optional[Dict] = None,
                                 baseline: float = 0.0,
                                 figsize: Tuple[float, float] = (12, 8),
                                 peak_alpha: float = 0.3,
                                 peak_colors: Optional[List[str]] = None,
                                 title: str = "NEXAFS Spectrum",
                                 save_path: Optional[str] = None,
                                 dpi: int = 300,
                                 use_experimental_data: bool = True,
                                 xlim: Optional[Tuple[float, float]] = None,
                                 ylim: Optional[Tuple[float, float]] = None,
                                 show_peaks_subplot: bool = False,
                                 peaks_subplot_height: float = 0.8) -> plt.Figure:
    """
    Plot NEXAFS spectrum with peak decomposition using separate energy axes for 
    experimental data and simulated components.
    
    Parameters:
    energy : numpy array
        Energy values (experimental data axis)
    intensity : numpy array
        Intensity values (experimental data - plotted as-is if use_experimental_data=True)
    peak_params : dict
        Peak parameters dictionary (either initial guesses or fitted values)
    edge_params : dict, optional
        Edge parameters dictionary (either initial guesses or fitted values)
    baseline : float
        Baseline value
    figsize : tuple
        Figure size (width, height)
    peak_alpha : float
        Transparency for peak fill areas (0-1)
    peak_colors : list, optional
        List of colors for peaks. If None, uses default colormap
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure
    use_experimental_data : bool
        If True, plot intensity as experimental data without recalculation
        If False, treat intensity as simulated and recalculate components
    xlim : tuple, optional
        Tuple of (xmin, xmax) to set the x-axis limits. If None, uses automatic scaling.
    ylim : tuple, optional
        Tuple of (ymin, ymax) to set the y-axis limits. If None, uses automatic scaling.
    show_peaks_subplot : bool, default False
        If True, add a subplot below the main plot showing only the Gaussian peaks
        (no baseline and no step edge contribution).
    peaks_subplot_height : float, default 0.8
        Height ratio for the peaks-only subplot relative to the main plot.
        
    Returns:
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Create figure
    if show_peaks_subplot:
        fig, (ax, ax_peaks) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={'height_ratios': [3, peaks_subplot_height], 'hspace': 0.05}
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax_peaks = None
    
    # Get peak names and set up colors
    peak_names = get_peak_names(peak_params)
    n_peaks = len(peak_names)
    
    if peak_colors is None:
        # Use a colormap to generate distinct colors
        if n_peaks <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_peaks, 3)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_peaks))
        peak_colors = [mcolors.to_hex(color) for color in colors]
    
    # Plot experimental data using original energy axis (no recalculation)
    if use_experimental_data:
        ax.plot(energy, intensity, 'ko--', markersize=3, alpha=0.7, 
                 label='Experimental Data', zorder=1)
    else:
        ax.plot(energy, intensity, 'ko--', markersize=3, alpha=0.7, 
                 label='Data', zorder=1)
    
    # Create high-density energy axis for smooth simulation curves
    energy_spacing = np.median(np.diff(energy))
    if energy_spacing > 0.1:
        energy_sim = np.arange(energy.min(), energy.max() + 0.1, 0.1)
    else:
        energy_sim = energy.copy()
    
    # Calculate simulated components using high-density axis
    # Initialize spectrum with baseline
    total_spectrum = np.full_like(energy_sim, baseline, dtype=float)
    
    # Add step edge if provided
    if edge_params is not None:
        edge_contribution = edge_bl_func(energy_sim, 
                                       edge_params['location'],
                                       edge_params['height'], 
                                       edge_params['width'],
                                       edge_params['decay'])
        total_spectrum += edge_contribution
    
    # Add Gaussian peaks to total
    peak_names = get_peak_names(peak_params)
    for peak_name in peak_names:
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]
        peak_energy = peak_params[f"{peak_name}_energy"]
        
        # Add Gaussian peak to total spectrum
        peak_contribution = gaussian(energy_sim, height, peak_energy, width)
        total_spectrum += peak_contribution
    
    # Plot total spectrum using high-density axis
    ax.plot(energy_sim, total_spectrum, 'r-', linewidth=2, 
             label='Total Spectrum', zorder=5)
    
    # Plot baseline if non-zero
    if baseline != 0:
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Baseline = {baseline:.3f}', zorder=2)
    
    # Plot step edge if present (using high-density axis)
    if edge_params is not None:
        edge_spectrum = edge_bl_func(energy_sim, 
                                   edge_params['location'],
                                   edge_params['height'],
                                   edge_params['width'], 
                                   edge_params['decay'])
        edge_spectrum += baseline
        ax.plot(energy_sim, edge_spectrum, 'b-', linewidth=2, alpha=0.8,
                 label='Step Edge', zorder=3)
        
        # Shade under the step edge with light grey
        ax.fill_between(energy_sim, baseline, edge_spectrum,
                        color='lightgrey', alpha=0.4, zorder=1)
    
    # Plot individual peaks (using high-density axis)
    peak_total = np.full_like(energy_sim, baseline)
    if edge_params is not None:
        peak_total += edge_bl_func(energy_sim,
                                 edge_params['location'],
                                 edge_params['height'],
                                 edge_params['width'],
                                 edge_params['decay'])
    
    for i, peak_name in enumerate(peak_names):
        # Get peak parameters
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]  
        peak_energy = peak_params[f"{peak_name}_energy"]
        
        # Calculate individual peak using high-density axis
        peak_spectrum = gaussian(energy_sim, height, peak_energy, width)
        peak_spectrum_with_baseline = peak_spectrum + peak_total
        
        # Plot peak line
        color = peak_colors[i % len(peak_colors)]
        peak_label = f'{peak_energy:.1f} eV'
        ax.plot(energy_sim, peak_spectrum_with_baseline, '-', 
                 color=color, linewidth=1.5, label=peak_label, zorder=4)
        
        # Fill area under peak
        ax.fill_between(energy_sim, peak_total, peak_spectrum_with_baseline,
                        color=color, alpha=peak_alpha, zorder=2)

        # Peaks-only subplot (no baseline / no edge)
        if ax_peaks is not None:
            ax_peaks.plot(energy_sim, peak_spectrum, '-', color=color, linewidth=1.2, zorder=2)
            ax_peaks.fill_between(energy_sim, 0.0, peak_spectrum, color=color, alpha=max(0.0, peak_alpha * 0.6), zorder=1)
    
    # Formatting
    if ax_peaks is None:
        ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10) #bbox_to_anchor=(1.05, 1),

    if ax_peaks is not None:
        ax_peaks.set_xlabel('Energy (eV)', fontsize=12)
        ax_peaks.set_ylabel('Peaks', fontsize=11)
        ax_peaks.grid(True, alpha=0.2)
        ax_peaks.set_ylim(bottom=0.0)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add information about energy axes
    # energy_spacing_exp = np.median(np.diff(energy))
    # energy_spacing_sim = np.median(np.diff(energy_sim))
    # info_text = f'Exp. spacing: {energy_spacing_exp:.3f} eV\nSim. spacing: {energy_spacing_sim:.3f} eV'
    # ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
    #         fontsize=9, verticalalignment='bottom', horizontalalignment='right',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def plot_simulated_nexafs_spectrum_synced(energy: np.ndarray,
                                 intensity: np.ndarray,
                                 peak_params: Dict,
                                 edge_params: Optional[Dict] = None,
                                 baseline: float = 0.0,
                                 figsize: Tuple[float, float] = (12, 8),
                                 peak_alpha: float = 0.3,
                                 peak_colors: Optional[List[str]] = None,
                                 title: str = "NEXAFS Spectrum",
                                 save_path: Optional[str] = None,
                                 dpi: int = 300,
                                 use_experimental_data: bool = True,
                                 energy_tolerance: Optional[float] = None,
                                 low_energy_threshold: float = 292.0,
                                 low_energy_tolerance: float = 0.1,
                                 high_energy_tolerance: float = 0.3,
                                 reset_color_map: bool = False,
                                 show_components_in_legend: bool = False,
                                 xlim: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """
    Plot NEXAFS spectrum with peak decomposition using synchronized colors based on peak energy positions.
    This function maintains color consistency across multiple plots by assigning the same color to peaks
    at similar energies (within tolerance) across different plots.
    
    Uses different tolerances based on energy:
    - Below low_energy_threshold (default 292 eV): low_energy_tolerance (default 0.1 eV)
    - Above low_energy_threshold: high_energy_tolerance (default 0.3 eV)
    
    Parameters:
    energy : numpy array
        Energy values (experimental data axis)
    intensity : numpy array
        Intensity values (experimental data - plotted as-is if use_experimental_data=True)
    peak_params : dict
        Peak parameters dictionary (either initial guesses or fitted values)
    edge_params : dict, optional
        Edge parameters dictionary (either initial guesses or fitted values)
    baseline : float
        Baseline value
    figsize : tuple
        Figure size (width, height)
    peak_alpha : float
        Transparency for peak fill areas (0-1)
    peak_colors : list, optional
        List of colors for peaks. If None, uses synchronized color mapping based on energy.
        Note: This parameter is ignored when using synchronized colors - colors are assigned by energy.
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure
    use_experimental_data : bool
        If True, plot intensity as experimental data without recalculation
        If False, treat intensity as simulated and recalculate components
    energy_tolerance : float, optional
        Override tolerance for matching peak energies (eV). If None, uses energy-based tolerance.
        When None, peaks below low_energy_threshold use low_energy_tolerance,
        and peaks above use high_energy_tolerance.
    low_energy_threshold : float, default 292.0
        Energy threshold (eV) below which low_energy_tolerance is used
    low_energy_tolerance : float, default 0.1
        Tolerance for matching peak energies below threshold (eV)
    high_energy_tolerance : float, default 0.3
        Tolerance for matching peak energies above threshold (eV)
    reset_color_map : bool, default False
        If True, reset the global color mapping before plotting this spectrum.
    show_components_in_legend : bool, default False
        If True, include individual Gaussian peaks and step edges in the legend.
        If False (default), only show "Measured Spectrum" and "Simulated Spectrum" in the legend.
    xlim : tuple, optional
        Tuple of (xmin, xmax) to set the x-axis limits. If None, uses automatic scaling.
        
    Returns:
    matplotlib.figure.Figure
        The created figure object
    """
    global _peak_energy_color_map
    
    # Reset color map if requested
    if reset_color_map:
        _peak_energy_color_map = {}
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get peak names and extract energies
    peak_names = get_peak_names(peak_params)
    
    # Create list of (peak_name, peak_energy) tuples and sort by energy
    peak_data = []
    for peak_name in peak_names:
        peak_energy = peak_params[f"{peak_name}_energy"]
        peak_data.append((peak_name, peak_energy))
    
    # Sort peaks by energy to ensure consistent ordering
    peak_data.sort(key=lambda x: x[1])
    
    # Plot experimental data using original energy axis (no recalculation)
    if use_experimental_data:
        ax.plot(energy, intensity, 'ko--', markersize=3, alpha=0.7, 
                 label='Measured Spectrum', zorder=1)
    else:
        ax.plot(energy, intensity, 'ko--', markersize=3, alpha=0.7, 
                 label='Data', zorder=1)
    
    # Create high-density energy axis for smooth simulation curves
    energy_spacing = np.median(np.diff(energy))
    if energy_spacing > 0.1:
        energy_sim = np.arange(energy.min(), energy.max() + 0.1, 0.1)
    else:
        energy_sim = energy.copy()
    
    # Calculate simulated components using high-density axis
    # Initialize spectrum with baseline
    total_spectrum = np.full_like(energy_sim, baseline, dtype=float)
    
    # Add step edge if provided
    if edge_params is not None:
        edge_contribution = edge_bl_func(energy_sim, 
                                       edge_params['location'],
                                       edge_params['height'], 
                                       edge_params['width'],
                                       edge_params['decay'])
        total_spectrum += edge_contribution
    
    # Add Gaussian peaks to total
    for peak_name, peak_energy in peak_data:
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]
        
        # Add Gaussian peak to total spectrum
        peak_contribution = gaussian(energy_sim, height, peak_energy, width)
        total_spectrum += peak_contribution
    
    # Plot total spectrum using high-density axis
    ax.plot(energy_sim, total_spectrum, 'r-', linewidth=2, 
             label='Simulated Spectrum', zorder=5)
    
    # Plot baseline if non-zero
    if baseline != 0:
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Baseline = {baseline:.3f}', zorder=2)
    
    # Plot step edge if present (using high-density axis)
    if edge_params is not None:
        edge_spectrum = edge_bl_func(energy_sim, 
                                   edge_params['location'],
                                   edge_params['height'],
                                   edge_params['width'], 
                                   edge_params['decay'])
        edge_spectrum += baseline
        edge_label = 'Step Edge' if show_components_in_legend else ''
        ax.plot(energy_sim, edge_spectrum, 'b-', linewidth=2, alpha=0.8,
                 label=edge_label, zorder=3)
        
        # Shade under the step edge with light grey
        ax.fill_between(energy_sim, baseline, edge_spectrum,
                        color='lightgrey', alpha=0.4, zorder=1)
    
    # Plot individual peaks (using high-density axis)
    peak_total = np.full_like(energy_sim, baseline)
    if edge_params is not None:
        peak_total += edge_bl_func(energy_sim,
                                 edge_params['location'],
                                 edge_params['height'],
                                 edge_params['width'],
                                 edge_params['decay'])
    
    # Plot peaks in sorted order by energy, using synchronized colors
    for peak_name, peak_energy in peak_data:
        # Get peak parameters
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]
        
        # Calculate individual peak using high-density axis
        peak_spectrum = gaussian(energy_sim, height, peak_energy, width)
        peak_spectrum_with_baseline = peak_spectrum + peak_total
        
        # Get color based on peak energy (synchronized across plots)
        # Uses energy-based tolerance (different for low/high energy regions)
        color = _get_color_for_peak_energy(
            peak_energy, 
            energy_tolerance=energy_tolerance,
            low_energy_threshold=low_energy_threshold,
            low_energy_tolerance=low_energy_tolerance,
            high_energy_tolerance=high_energy_tolerance
        )
        peak_label = f'{peak_energy:.1f} eV' if show_components_in_legend else ''
        ax.plot(energy_sim, peak_spectrum_with_baseline, '-', 
                 color=color, linewidth=1.5, label=peak_label, zorder=4)
        
        # Fill area under peak
        ax.fill_between(energy_sim, peak_total, peak_spectrum_with_baseline,
                        color=color, alpha=peak_alpha, zorder=2)
    
    # Formatting
    ax.set_xlabel('Energy (eV)', fontsize=18)
    ax.set_ylabel('Intensity (a.u.)', fontsize=18)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Set x-axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def plot_simulated_nexafs_spectrum_synced_with_peaks(energy: np.ndarray,
                                 intensity: np.ndarray,
                                 peak_params: Dict,
                                 edge_params: Optional[Dict] = None,
                                 baseline: float = 0.0,
                                 figsize: Tuple[float, float] = (12, 10),
                                 peak_alpha: float = 0.3,
                                 peak_colors: Optional[List[str]] = None,
                                 title: str = "NEXAFS Spectrum",
                                 save_path: Optional[str] = None,
                                 dpi: int = 300,
                                 use_experimental_data: bool = True,
                                 energy_tolerance: Optional[float] = None,
                                 low_energy_threshold: float = 292.0,
                                 low_energy_tolerance: float = 0.1,
                                 high_energy_tolerance: float = 0.3,
                                 reset_color_map: bool = False,
                                 show_components_in_legend: bool = False,
                                 xlim: Optional[Tuple[float, float]] = None,
                                 subplot_height_ratio: Tuple[float, float] = (3, 1)) -> plt.Figure:
    """
    Plot NEXAFS spectrum with peak decomposition using synchronized colors, with two subplots:
    - Top subplot: Full fitted spectrum with measured data, simulated spectrum, edge, and individual peaks
    - Bottom subplot: Only individual Gaussian peaks
    
    This function maintains color consistency across multiple plots by assigning the same color to peaks
    at similar energies (within tolerance) across different plots.
    
    Uses different tolerances based on energy:
    - Below low_energy_threshold (default 292 eV): low_energy_tolerance (default 0.1 eV)
    - Above low_energy_threshold: high_energy_tolerance (default 0.3 eV)
    
    Parameters:
    energy : numpy array
        Energy values (experimental data axis)
    intensity : numpy array
        Intensity values (experimental data - plotted as-is if use_experimental_data=True)
    peak_params : dict
        Peak parameters dictionary (either initial guesses or fitted values)
    edge_params : dict, optional
        Edge parameters dictionary (either initial guesses or fitted values)
    baseline : float
        Baseline value
    figsize : tuple
        Figure size (width, height)
    peak_alpha : float
        Transparency for peak fill areas (0-1)
    peak_colors : list, optional
        List of colors for peaks. If None, uses synchronized color mapping based on energy.
        Note: This parameter is ignored when using synchronized colors - colors are assigned by energy.
    title : str
        Plot title (applied to top subplot)
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure
    use_experimental_data : bool
        If True, plot intensity as experimental data without recalculation
        If False, treat intensity as simulated and recalculate components
    energy_tolerance : float, optional
        Override tolerance for matching peak energies (eV). If None, uses energy-based tolerance.
        When None, peaks below low_energy_threshold use low_energy_tolerance,
        and peaks above use high_energy_tolerance.
    low_energy_threshold : float, default 292.0
        Energy threshold (eV) below which low_energy_tolerance is used
    low_energy_tolerance : float, default 0.1
        Tolerance for matching peak energies below threshold (eV)
    high_energy_tolerance : float, default 0.3
        Tolerance for matching peak energies above threshold (eV)
    reset_color_map : bool, default False
        If True, reset the global color mapping before plotting this spectrum.
    show_components_in_legend : bool, default False
        If True, include individual Gaussian peaks and step edges in the legend.
        If False (default), only show "Measured Spectrum" and "Simulated Spectrum" in the legend.
    xlim : tuple, optional
        Tuple of (xmin, xmax) to set the x-axis limits. If None, uses automatic scaling.
    subplot_height_ratio : tuple, default (3, 1)
        Ratio of heights for top and bottom subplots (top, bottom)
        
    Returns:
    matplotlib.figure.Figure
        The created figure object
    """
    global _peak_energy_color_map
    
    # Reset color map if requested
    if reset_color_map:
        _peak_energy_color_map = {}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': subplot_height_ratio, 'hspace': 0.1})
    
    # Get peak names and extract energies
    peak_names = get_peak_names(peak_params)
    
    # Create list of (peak_name, peak_energy) tuples and sort by energy
    peak_data = []
    for peak_name in peak_names:
        peak_energy = peak_params[f"{peak_name}_energy"]
        peak_data.append((peak_name, peak_energy))
    
    # Sort peaks by energy to ensure consistent ordering
    peak_data.sort(key=lambda x: x[1])
    
    # Create high-density energy axis for smooth simulation curves
    energy_spacing = np.median(np.diff(energy))
    if energy_spacing > 0.1:
        energy_sim = np.arange(energy.min(), energy.max() + 0.1, 0.1)
    else:
        energy_sim = energy.copy()
    
    # Calculate simulated components using high-density axis
    # Initialize spectrum with baseline
    total_spectrum = np.full_like(energy_sim, baseline, dtype=float)
    
    # Add step edge if provided
    if edge_params is not None:
        edge_contribution = edge_bl_func(energy_sim, 
                                       edge_params['location'],
                                       edge_params['height'], 
                                       edge_params['width'],
                                       edge_params['decay'])
        total_spectrum += edge_contribution
    
    # Add Gaussian peaks to total
    for peak_name, peak_energy in peak_data:
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]
        
        # Add Gaussian peak to total spectrum
        peak_contribution = gaussian(energy_sim, height, peak_energy, width)
        total_spectrum += peak_contribution
    
    # ========== TOP SUBPLOT: Full Spectrum ==========
    # Plot experimental data using original energy axis
    if use_experimental_data:
        ax1.plot(energy, intensity, 'ko--', markersize=3, alpha=0.7, 
                 label='Measured Spectrum', zorder=1)
    else:
        ax1.plot(energy, intensity, 'ko--', markersize=3, alpha=0.7, 
                 label='Data', zorder=1)
    
    # Plot total spectrum using high-density axis
    ax1.plot(energy_sim, total_spectrum, 'r-', linewidth=2, 
             label='Simulated Spectrum', zorder=5)
    
    # Plot baseline if non-zero
    if baseline != 0:
        ax1.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Baseline = {baseline:.3f}', zorder=2)
    
    # Plot step edge if present (using high-density axis)
    if edge_params is not None:
        edge_spectrum = edge_bl_func(energy_sim, 
                                   edge_params['location'],
                                   edge_params['height'],
                                   edge_params['width'], 
                                   edge_params['decay'])
        edge_spectrum += baseline
        edge_label = 'Step Edge' if show_components_in_legend else ''
        ax1.plot(energy_sim, edge_spectrum, 'b-', linewidth=2, alpha=0.8,
                 label=edge_label, zorder=3)
        
        # Shade under the step edge with light grey
        ax1.fill_between(energy_sim, baseline, edge_spectrum,
                        color='lightgrey', alpha=0.4, zorder=1)
    
    # Plot individual peaks (using high-density axis)
    peak_total = np.full_like(energy_sim, baseline)
    if edge_params is not None:
        peak_total += edge_bl_func(energy_sim,
                                 edge_params['location'],
                                 edge_params['height'],
                                 edge_params['width'],
                                 edge_params['decay'])
    
    # Plot peaks in sorted order by energy, using synchronized colors
    for peak_name, peak_energy in peak_data:
        # Get peak parameters
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]
        
        # Calculate individual peak using high-density axis
        peak_spectrum = gaussian(energy_sim, height, peak_energy, width)
        peak_spectrum_with_baseline = peak_spectrum + peak_total
        
        # Get color based on peak energy (synchronized across plots)
        color = _get_color_for_peak_energy(
            peak_energy, 
            energy_tolerance=energy_tolerance,
            low_energy_threshold=low_energy_threshold,
            low_energy_tolerance=low_energy_tolerance,
            high_energy_tolerance=high_energy_tolerance
        )
        peak_label = f'{peak_energy:.1f} eV' if show_components_in_legend else ''
        ax1.plot(energy_sim, peak_spectrum_with_baseline, '-', 
                 color=color, linewidth=1.5, label=peak_label, zorder=4)
        
        # Fill area under peak
        ax1.fill_between(energy_sim, peak_total, peak_spectrum_with_baseline,
                        color=color, alpha=peak_alpha, zorder=2)
    
    # Formatting for top subplot
    ax1.set_ylabel('Intensity (a.u.)', fontsize=18)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Set x-axis limits if provided
    if xlim is not None:
        ax1.set_xlim(xlim)
    
    # ========== BOTTOM SUBPLOT: Individual Peaks Only ==========
    # Plot only the individual Gaussian peaks (no baseline, edge, or total spectrum)
    for peak_name, peak_energy in peak_data:
        # Get peak parameters
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]
        
        # Calculate individual peak using high-density axis (no baseline or edge)
        peak_spectrum = gaussian(energy_sim, height, peak_energy, width)
        
        # Get color based on peak energy (synchronized - same as top plot)
        color = _get_color_for_peak_energy(
            peak_energy, 
            energy_tolerance=energy_tolerance,
            low_energy_threshold=low_energy_threshold,
            low_energy_tolerance=low_energy_tolerance,
            high_energy_tolerance=high_energy_tolerance
        )
        peak_label = f'{peak_energy:.1f} eV' if show_components_in_legend else ''
        
        # Plot peak line
        ax2.plot(energy_sim, peak_spectrum, '-', 
                 color=color, linewidth=1.5, label=peak_label, zorder=4)
        
        # Fill area under peak (from zero)
        ax2.fill_between(energy_sim, 0, peak_spectrum,
                        color=color, alpha=peak_alpha, zorder=2)
    
    # Formatting for bottom subplot
    ax2.set_xlabel('Energy (eV)', fontsize=18)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=18)
    ax2.grid(True, alpha=0.3)
    if show_components_in_legend:
        ax2.legend(loc='upper left', fontsize=10)
    
    # Set x-axis limits if provided (apply to both subplots)
    if xlim is not None:
        ax2.set_xlim(xlim)
    
    # Share x-axis (hide x-axis labels on top subplot)
    ax1.set_xticklabels([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def simulate_nexafs_spectrum_for_fitting(x: np.ndarray, 
                                       peak_params: Dict,
                                       edge_params: Optional[Dict] = None,
                                       baseline: float = 0.0) -> np.ndarray:
    """
    Simulate NEXAFS spectrum for fitting purposes - always uses the provided energy axis.
    This function is used internally by the fitting algorithm.
    
    Parameters:
    x : numpy array
        Energy values (uses exactly this axis, no high-density modification)
    peak_params : dict
        Peak parameters dictionary
    edge_params : dict, optional
        Edge parameters dictionary
    baseline : float
        Baseline value
        
    Returns:
    numpy array
        Simulated spectrum values at the provided energy points
    """
    # Initialize spectrum with baseline - use provided energy axis exactly
    spectrum = np.full_like(x, baseline, dtype=float)
    
    # Add step edge if provided
    if edge_params is not None:
        edge_contribution = edge_bl_func(x, 
                                       edge_params['location'],
                                       edge_params['height'], 
                                       edge_params['width'],
                                       edge_params['decay'])
        spectrum += edge_contribution
    
    # Add Gaussian peaks
    peak_names = get_peak_names(peak_params)
    for peak_name in peak_names:
        width = peak_params[f"{peak_name}_width"]
        height = peak_params[f"{peak_name}_height"]
        energy = peak_params[f"{peak_name}_energy"]
        
        # Add Gaussian peak to spectrum
        peak_contribution = gaussian(x, height, energy, width)
        spectrum += peak_contribution
    
    return spectrum

def simulate_nexafs_spectrum_for_plotting(x: np.ndarray, 
                                        peak_params: Dict,
                                        edge_params: Optional[Dict] = None,
                                        baseline: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate NEXAFS spectrum for plotting purposes - uses high-density energy axis if needed.
    
    Parameters:
    x : numpy array
        Energy values (experimental energy axis)
    peak_params : dict
        Peak parameters dictionary
    edge_params : dict, optional
        Edge parameters dictionary
    baseline : float
        Baseline value
        
    Returns:
    tuple
        (energy_sim, spectrum_sim) where energy_sim is high-density if needed
    """
    # Check energy spacing
    energy_spacing = np.median(np.diff(x))
    
    if energy_spacing > 0.1:
        # Create high-density energy axis with 0.1 eV steps for smooth plotting
        energy_sim = np.arange(x.min(), x.max() + 0.1, 0.1)
        print(f"Using high-density energy axis for plotting: {energy_spacing:.3f} eV -> 0.1 eV spacing")
        print(f"Points: {len(x)} -> {len(energy_sim)}")
    else:
        # Use original energy axis
        energy_sim = x.copy()
    
    # Calculate spectrum on the appropriate axis
    spectrum = simulate_nexafs_spectrum_for_fitting(energy_sim, peak_params, edge_params, baseline)
    
    return energy_sim, spectrum


def generate_fitted_params_from_results(results, 
                                      energy_tolerance=0.1, 
                                      width_tolerance=0.2, 
                                      height_tolerance=0.3,
                                      edge_location_tolerance=0.1,
                                      edge_height_tolerance=0.2,
                                      edge_width_tolerance=0.2,
                                      edge_decay_tolerance=0.01):
    """
    Generate new fitted parameter sets from fitting results with updated bounds.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from fit_nexafs_spectrum containing fitted parameters
    energy_tolerance : float, default 0.1
        +/- range for energy bounds (eV)
    width_tolerance : float, default 0.2
        +/- range for width bounds
    height_tolerance : float, default 0.3
        +/- range for height bounds
    edge_location_tolerance : float, default 0.1
        +/- range for edge location bounds (eV)
    edge_height_tolerance : float, default 0.2
        +/- range for edge height bounds
    edge_width_tolerance : float, default 0.2
        +/- range for edge width bounds
    edge_decay_tolerance : float, default 0.01
        +/- range for edge decay bounds
    
    Returns:
    --------
    tuple
        (fitted_peak_params_new, fitted_edge_params_new)
    """
    
    # Extract fitted parameters from results
    fitted_peak_params = results['fitted_peak_params']
    fitted_edge_params = results['fitted_edge_params']
    
    # Generate new peak parameters with updated bounds
    fitted_peak_params_new = {}
    
    # Extract and sort peaks by energy to maintain order
    peak_data = []
    for key in fitted_peak_params.keys():
        if key.endswith('_energy'):
            peak_num = key.split('_')[1]
            energy_val = fitted_peak_params[f"peak_{peak_num}_energy"]
            peak_data.append((peak_num, energy_val))
    
    # Sort by energy value
    peak_data.sort(key=lambda x: x[1])
    
    # Generate new parameters for each peak
    for peak_num, energy_val in peak_data:
        # Get fitted values
        energy_val = fitted_peak_params[f"peak_{peak_num}_energy"]
        width_val = fitted_peak_params[f"peak_{peak_num}_width"]
        height_val = fitted_peak_params[f"peak_{peak_num}_height"]
        
        # Get original bounds to extract fit flags
        energy_bounds = fitted_peak_params[f"peak_{peak_num}_energy_bounds"]
        width_bounds = fitted_peak_params[f"peak_{peak_num}_width_bounds"]
        height_bounds = fitted_peak_params[f"peak_{peak_num}_height_bounds"]
        
        # Extract fit flags (third element of bounds tuple)
        energy_fit_flag = energy_bounds[2]
        width_fit_flag = width_bounds[2]
        height_fit_flag = height_bounds[2]
        
        # Generate new bounds with specified tolerances
        fitted_peak_params_new[f"peak_{peak_num}_energy"] = energy_val
        fitted_peak_params_new[f"peak_{peak_num}_energy_bounds"] = (
            energy_val - energy_tolerance, 
            energy_val + energy_tolerance, 
            energy_fit_flag
        )
        
        fitted_peak_params_new[f"peak_{peak_num}_width"] = width_val
        fitted_peak_params_new[f"peak_{peak_num}_width_bounds"] = (
            width_val - width_tolerance, 
            width_val + width_tolerance, 
            width_fit_flag
        )
        
        fitted_peak_params_new[f"peak_{peak_num}_height"] = height_val
        fitted_peak_params_new[f"peak_{peak_num}_height_bounds"] = (
            height_val - height_tolerance, 
            height_val + height_tolerance, 
            height_fit_flag
        )
    
    # Generate new edge parameters with updated bounds
    fitted_edge_params_new = {}
    
    if fitted_edge_params is not None:
        # Get fitted edge values
        edge_location = fitted_edge_params["location"]
        edge_height = fitted_edge_params["height"]
        edge_width = fitted_edge_params["width"]
        edge_decay = fitted_edge_params["decay"]
        
        # Generate new edge parameters (edge parameters don't have bounds in the same format)
        fitted_edge_params_new = {
            "location": edge_location,
            "height": edge_height,
            "width": edge_width,
            "decay": edge_decay
        }
    
    return fitted_peak_params_new, fitted_edge_params_new