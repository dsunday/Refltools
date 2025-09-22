
import kkcalc #this is the only library that you might not already have.
from kkcalc import data
from kkcalc import kk

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

from scipy.optimize import curve_fit, least_squares, minimize, differential_evolution, dual_annealing

def calculate_refractive_index(input_file, chemical_formula, density, x_min=None, x_max=None):
    """
    Calculate refractive index (Delta, Beta) from NEXAFS spectrum.
    
    Parameters:
    -----------
    input_file : str
        Path to the input spectrum file
    chemical_formula : str
        Chemical formula of the material (e.g. 'C8H8')
    density : float
        Density in g/cc
    x_min : float, optional
        Minimum energy value for merging points
    x_max : float, optional
        Maximum energy value for merging points
    
    Returns:
    --------
    numpy.ndarray
        Array with columns: Energy, Delta, Beta
    """
    import numpy as np
    import kkcalc as kk
    from kkcalc import data
    
    # Parse chemical formula and calculate formula mass
    stoichiometry = kk.data.ParseChemicalFormula(chemical_formula)
    formula_mass = data.calculate_FormulaMass(stoichiometry)
    
    # Define merge points if provided
    merge_points = None
    if x_min is not None and x_max is not None:
        merge_points = [x_min, x_max]
    
    # Calculate the real part using Kramers-Kronig transform
    output = kk.kk_calculate_real(
        input_file,
        chemical_formula,
        load_options=None,
        input_data_type='Beta',
        merge_points=merge_points,
        add_background=False,
        fix_distortions=False,
        curve_tolerance=0.05,
        curve_recursion=100
    )
    
    # Convert ASF to refractive index components
    Delta = data.convert_data(
        output[:,[0,1]],
        'ASF',
        'refractive_index', 
        Density=density, 
        Formula_Mass=formula_mass
    )[:,1]
    
    Beta = data.convert_data(
        output[:,[0,2]],
        'ASF',
        'refractive_index', 
        Density=density, 
        Formula_Mass=formula_mass
    )[:,1]
    
    E = data.convert_data(
        output[:,[0,2]],
        'ASF',
        'refractive_index', 
        Density=density, 
        Formula_Mass=formula_mass
    )[:,0]
    
    # Combine into a single array
    result = np.column_stack((E, Delta, Beta))
    
    return result

def EnergytoWavelength(Energy):
    """
    Convert energy in eV to wavelength in angstroms.
    
    Parameters:
    -----------
    Energy : float or array
        Energy values in eV
    
    Returns:
    --------
    float or array
        Wavelength values in angstroms (Å)
    """
    import numpy as np
    # Constants
    h = 4.135667696e-15  # Planck's constant in eV⋅s
    c = 299792458        # Speed of light in m/s
    
    # Convert eV to wavelength in angstroms (1 nm = 10 Å)
    wavelength = (h * c / Energy) * 1e10
    
    return wavelength

def DeltaBetatoSLD(DeltaBeta):
    """
    Convert Delta and Beta values to Scattering Length Density (SLD).
    
    Parameters:
    -----------
    DeltaBeta : numpy.ndarray
        Array with columns: Energy, Delta, Beta
    
    Returns:
    --------
    numpy.ndarray
        Array with columns: Energy, SLD_real, SLD_imag, Wavelength
        SLD units are inverse angstroms squared (Å^-2)
    """
    import numpy as np
    
    # Convert energy to wavelength in angstroms
    Wavelength = EnergytoWavelength(DeltaBeta[:,0])
    
    # Initialize SLD array
    SLD = np.zeros([len(DeltaBeta[:,0]), 4])
    
    # Fill SLD array
    SLD[:,0] = DeltaBeta[:,0]                                           # Energy (eV)
    SLD[:,3] = Wavelength                                               # Wavelength (Å)
    SLD[:,1] = 2 * np.pi * DeltaBeta[:,1] / (np.power(Wavelength, 2))*1000000   # SLD real (Å^-2)
    SLD[:,2] = 2 * np.pi * DeltaBeta[:,2] / (np.power(Wavelength, 2))*1000000   # SLD imag (Å^-2)
    
    return SLD

def process_nexafs_to_SLD(input_file, chemical_formula, density, x_min=None, x_max=None):
    """
    Process NEXAFS spectrum to calculate refractive index and then convert to SLD.
    
    Parameters:
    -----------
    input_file : str
        Path to the input spectrum file
    chemical_formula : str
        Chemical formula of the material (e.g. 'C8H8')
    density : float
        Density in g/cc
    x_min : float, optional
        Minimum energy value for merging points
    x_max : float, optional
        Maximum energy value for merging points
    
    Returns:
    --------
    tuple
        (DeltaBeta, SLD) where:
        - DeltaBeta is an array with columns: Energy, Delta, Beta
        - SLD is an array with columns: Energy, SLD_real, SLD_imag, Wavelength
    """
    # Calculate refractive index
    DeltaBeta = calculate_refractive_index(input_file, chemical_formula, density, x_min, x_max)
    
    # Convert to SLD
    SLD = DeltaBetatoSLD(DeltaBeta)
    
    return DeltaBeta, SLD

def binary_contrast_numpy(n1_array, n2_array=None, plot=False, title=None, save_path=None, xlim=None):
    """
    Calculate binary contrast between two refractive indices represented as numpy arrays.
    
    Parameters:
    -----------
    n1_array : numpy.ndarray
        First component array with shape (n, 3) where:
        - First column is energy values
        - Second column is delta (δ) values
        - Third column is beta (β) values
    
    n2_array : numpy.ndarray, optional
        Second component array with the same structure as n1_array.
        If None, vacuum is assumed (all zeros).
    
    plot : bool, optional
        Whether to plot the binary contrast (default: False)
    
    title : str, optional
        Title for the plot. If None, a default title will be used.
        
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    
    xlim : tuple, optional
        Energy range (min, max) to display on the plot, e.g. (280, 290).
        If None, the full energy range is displayed.
    
    Returns:
    --------
    numpy.ndarray
        Array with shape (n, 2) where:
        - First column is energy values
        - Second column is the binary contrast values
    """
    import numpy as np
    
    # Extract energy, delta and beta from the first array
    energy = n1_array[:, 0]
    delta1 = n1_array[:, 1]
    beta1 = n1_array[:, 2]
    
    if n2_array is None:
        # Second component is vacuum (zeros)
        delta2 = np.zeros_like(delta1)
        beta2 = np.zeros_like(beta1)
        component2_name = "vacuum"
    else:
        # Interpolate second component to match the energy array of the first
        delta2 = np.interp(energy, n2_array[:, 0], n2_array[:, 1])
        beta2 = np.interp(energy, n2_array[:, 0], n2_array[:, 2])
        component2_name = "material 2"
    
    # Calculate binary contrast: energy^4 * ((delta1-delta2)^2 + (beta1-beta2)^2)
    contrast = energy**4 * ((delta1 - delta2)**2 + (beta1 - beta2)**2)
    
    # Create result array
    result = np.column_stack((energy, contrast))
    
    # Plot if requested
    if plot:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(energy, contrast, 'b-', linewidth=2)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.xlabel('Energy (eV)', fontsize=12)
        plt.ylabel('Binary Contrast (log scale)', fontsize=12)
        
        if title is None:
            title = f"Binary Contrast: Material 1 vs {component2_name}"
        plt.title(title, fontsize=14)
        
        # Add minor grid lines for log scale
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        # Set xlim if provided and adjust y-axis based on visible data
        if xlim is not None:
            plt.xlim(xlim)
            
            # Find values within the xlim range to optimize y-axis scale
            mask = (energy >= xlim[0]) & (energy <= xlim[1])
            if np.any(mask):  # Make sure there are values in the range
                visible_contrast = contrast[mask]
                min_contrast = np.min(visible_contrast)
                max_contrast = np.max(visible_contrast)
                
                # Add a bit of padding to the y-axis limits
                plt.ylim(min_contrast * 0.8, max_contrast * 1.2)
                
                # Find max contrast within the visible range
                max_idx_visible = np.argmax(visible_contrast)
                max_energy_visible = energy[mask][max_idx_visible]
                max_contrast_visible = visible_contrast[max_idx_visible]
                
                plt.annotate(f'Max: {max_contrast_visible:.2e} @ {max_energy_visible:.1f} eV',
                             xy=(max_energy_visible, max_contrast_visible),
                             xytext=(max_energy_visible + (xlim[1]-xlim[0])*0.05, max_contrast_visible*1.2),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                             fontsize=10)
        else:
            # If no xlim provided, annotate global maximum
            max_idx = np.argmax(contrast)
            max_energy = energy[max_idx]
            max_contrast = contrast[max_idx]
            
            plt.annotate(f'Max: {max_contrast:.2e} @ {max_energy:.1f} eV',
                         xy=(max_energy, max_contrast),
                         xytext=(max_energy + (energy[-1]-energy[0])*0.05, max_contrast*1.2),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                         fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    return result

import numpy as np
from scipy import interpolate, special
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt

def erf(x):
    """
    Error function wrapper for numpy's error function.
    """
    return special.erf(x)

def edge_bl_func(x, location, height, width, decay):
    """
    Python equivalent of the Igor Pro Edge_BLFunc function with explicit parameters.
    
    Parameters:
    x : float or numpy array
        The x values at which to evaluate the function
    location : float
        The location parameter (equivalent to s.cwave[0] in Igor)
    height : float
        The height parameter (equivalent to s.cwave[1] in Igor)
    width : float
        The width parameter (equivalent to s.cwave[2] in Igor)
    decay : float
        The decay parameter (equivalent to s.cwave[3] in Igor)
    
    Returns:
    float or numpy array
        The function value(s) at x
    """
    # Determine step (1 if x >= location + width, 0 otherwise)
    step = np.where(x >= location + width, 1, 0)
    
    # Calculate the error function component
    erf_component = (1 + erf((x - location) * 2 * math.log(2) / width))
    
    # Calculate the decay component
    decay_component = (1 + step * (np.exp(-decay * (x - location - width)) - 1))
    
    # Calculate the final result
    result = (height / 2) * erf_component * decay_component
    
    return result

def gaussian(x, height, center, width):
    """
    Calculate a Gaussian peak.
    
    Parameters:
    x : numpy array
        Energy values
    height : float
        Peak height
    center : float
        Peak center energy
    width : float
        Peak width (FWHM)
    
    Returns:
    numpy array
        Gaussian peak values
    """
    sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)

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

def simulate_nexafs_spectrum(x: np.ndarray, 
                           peak_params: Dict,
                           edge_params: Optional[Dict] = None,
                           baseline: float = 0.0,
                           experimental_intensity: Optional[np.ndarray] = None,
                           plot: bool = False,
                           plot_kwargs: Optional[Dict] = None) -> Union[Tuple[np.ndarray, np.ndarray], plt.Figure]:
    """
    Simulate a NEXAFS spectrum with Gaussian peaks and optional step edge.
    Automatically uses high-density energy axis if input has sparse spacing.
    
    Parameters:
    x : numpy array
        Energy values (experimental energy axis)
    peak_params : dict
        Peak parameters dictionary
    edge_params : dict, optional
        Edge parameters dictionary
    baseline : float
        Baseline value
    experimental_intensity : numpy array, optional
        Experimental intensity data for plotting comparison
    plot : bool
        If True, automatically create plot and return figure instead of data
    plot_kwargs : dict, optional
        Additional keyword arguments for plotting function
        
    Returns:
    tuple or matplotlib.figure.Figure
        If plot=False: (energy_sim, spectrum_sim) tuple
        If plot=True: matplotlib Figure object
    """
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

def load_spectrum_data(data: Union[str, np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum data from various formats.
    
    Parameters:
    data : str, numpy array, or pandas DataFrame
        - str: filepath to load data from
        - numpy array: 2D array with columns [energy, intensity]
        - pandas DataFrame: DataFrame with energy and intensity columns
    
    Returns:
    tuple
        (energy_array, intensity_array)
    """
    if isinstance(data, str):
        # Try to load from file
        try:
            # Try pandas first (handles various formats)
            df = pd.read_csv(data, sep=None, engine='python')
            if df.shape[1] >= 2:
                energy = df.iloc[:, 0].values
                intensity = df.iloc[:, 1].values
            else:
                raise ValueError("Data file must have at least 2 columns")
        except:
            # Fallback to numpy
            arr = np.loadtxt(data)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("Data must have shape (n_points, 2) with energy and intensity")
            energy = arr[:, 0]
            intensity = arr[:, 1]
    
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] >= 2:
            energy = data.iloc[:, 0].values
            intensity = data.iloc[:, 1].values
        else:
            raise ValueError("DataFrame must have at least 2 columns")
    
    elif isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Data must have shape (n_points, 2) with energy and intensity")
        energy = data[:, 0]
        intensity = data[:, 1]
    
    else:
        raise ValueError("Data must be filepath, numpy array, or pandas DataFrame")
    
    return energy, intensity

def fit_nexafs_spectrum(data: Union[str, np.ndarray, pd.DataFrame],
                       peak_params: Dict,
                       edge_params: Optional[Dict] = None,
                       baseline: float = 0.0,
                       method: str = 'lm',
                       max_iterations: int = 1000,
                       tolerance: float = 1e-8,
                       algorithm: str = 'curve_fit',
                       global_opt_params: Optional[Dict] = None) -> Dict:
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
    
    Returns:
    dict
        Dictionary containing fitting results
    """
    
    # Load experimental data
    energy_exp, intensity_exp = load_spectrum_data(data)
    
    print(f"Experimental data: {len(energy_exp)} points, energy spacing: {np.median(np.diff(energy_exp)):.3f} eV")
    
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
            'experimental_intensity': intensity_exp
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
            'experimental_intensity': intensity_exp
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
    energy_exp = results['experimental_energy']
    intensity_exp = results['experimental_intensity']
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
    
    # Plot experimental data
    ax1.plot(energy_exp, intensity_exp, 'ko', markersize=3, alpha=0.7, 
             label='Experimental Data', zorder=1)
    
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

def print_fit_results(results: Dict):
    """
    Print a summary of fitting results.
    
    Parameters:
    results : dict
        Results dictionary from fit_nexafs_spectrum
    """
    fit_result = results['fit_result']
    
    if fit_result['success']:
        print("=== NEXAFS Fitting Results ===")
        print(f"R-squared: {results['r_squared']:.6f}")
        print(f"RMSE: {results['rmse']:.6f}")
        print(f"Degrees of freedom: {fit_result['degrees_of_freedom']}")
        print("\nFitted Parameters:")
        print("-" * 50)
        
        for param_name, value in fit_result['fitted_values'].items():
            uncertainty = fit_result['parameter_uncertainties'].get(param_name, 0)
            print(f"{param_name:25s}: {value:10.4f} ± {uncertainty:8.4f}")
            
    else:
        print("=== Fitting Failed ===")
        print(f"Error: {fit_result['error_message']}")

def fit_spectra_with_edge(target_spectrum, spectrum1, spectrum2, 
                         initial_guess=None, include_edge=True,
                         bounds=None):
    """
    Fits a linear combination of two spectra plus an optional step edge 
    to match a target spectrum.
    
    Parameters:
    -----------
    target_spectrum : ndarray, shape (n, 2)
        Target spectrum with columns [Energy, Intensity]
    spectrum1 : ndarray, shape (m, 2)
        First input spectrum with columns [Energy, Intensity]
    spectrum2 : ndarray, shape (k, 2)
        Second input spectrum with columns [Energy, Intensity]
    initial_guess : list or None, optional
        Initial guess for parameters:
        [fraction, edge_location, edge_height, edge_width, edge_decay]
        If None, defaults to [0.5, (energy_range_mid), 0.1, 1.0, 0.5]
    include_edge : bool, optional
        Whether to include the step edge in the fit
    bounds : list of tuples or None, optional
        Bounds for parameters as [(min_frac, max_frac), 
        (min_loc, max_loc), (min_height, max_height), 
        (min_width, max_width), (min_decay, max_decay)]
        If None, uses default bounds
        
    Returns:
    --------
    params : list
        The optimal parameters [fraction, edge_location, edge_height, edge_width, edge_decay]
        (if include_edge=False, the edge parameters will be zeros)
    combined_spectrum : ndarray, shape (n, 2)
        The optimal combination of spectrum1, spectrum2, and edge
    components : dict
        Dictionary with separate components: 
        {'spec1_contrib', 'spec2_contrib', 'edge_contrib'}
    """
    # Get common energy grid (use target's energy grid)
    target_energies = target_spectrum[:, 0]
    
    # Set default initial guess if not provided
    if initial_guess is None:
        energy_mid = (target_energies.min() + target_energies.max()) / 2
        initial_guess = [0.5, energy_mid, 0.1, 1.0, 0.5]
    
    # Create interpolation functions for both input spectra
    interp1 = interpolate.interp1d(spectrum1[:, 0], spectrum1[:, 1], 
                                  bounds_error=False, fill_value=0)
    interp2 = interpolate.interp1d(spectrum2[:, 0], spectrum2[:, 1], 
                                  bounds_error=False, fill_value=0)
    
    # Interpolate input spectra onto target energy grid
    intensity1 = interp1(target_energies)
    intensity2 = interp2(target_energies)
    
    # Define objective function to minimize (sum of squared differences)
    if include_edge:
        def objective(params):
            fraction, edge_loc, edge_height, edge_width, edge_decay = params
            # Calculate linear combination of spectra
            spec_combined = fraction * intensity1 + (1 - fraction) * intensity2
            # Add edge function
            edge_contribution = edge_bl_func(target_energies, edge_loc, edge_height, edge_width, edge_decay)
            combined_intensity = spec_combined + edge_contribution
            # Return sum of squared differences
            return np.sum((combined_intensity - target_spectrum[:, 1])**2)
        
        # Set default bounds if not provided
        if bounds is None:
            energy_min, energy_max = target_energies.min(), target_energies.max()
            bounds = [
                (0, 1),                     # fraction between 0 and 1
                (energy_min, energy_max),   # edge location within energy range
                (0, 1),                     # edge height between 0 and 1
                (0.1, 10),                  # edge width (reasonable range)
                (0, 10)                     # edge decay (reasonable range)
            ]
    else:
        # Simplified objective function without edge
        def objective(params):
            fraction = params[0]
            combined_intensity = fraction * intensity1 + (1 - fraction) * intensity2
            return np.sum((combined_intensity - target_spectrum[:, 1])**2)
        
        # Use only the first parameter if no edge
        initial_guess = [initial_guess[0]]
        if bounds is None:
            bounds = [(0, 1)]  # Just the fraction bound
    
    # Perform optimization
    result = minimize(objective, initial_guess, bounds=bounds)
    optimal_params = result.x
    
    # Ensure we return a consistent parameter list format
    if not include_edge:
        optimal_params = np.append(optimal_params, [0, 0, 0, 0])
    
    # Calculate final combined spectrum and components
    fraction = optimal_params[0]
    spec1_contrib = fraction * intensity1
    spec2_contrib = (1 - fraction) * intensity2
    
    if include_edge:
        edge_loc, edge_height, edge_width, edge_decay = optimal_params[1:5]
        edge_contrib = edge_bl_func(target_energies, edge_loc, edge_height, edge_width, edge_decay)
    else:
        edge_contrib = np.zeros_like(target_energies)
    
    combined_intensity = spec1_contrib + spec2_contrib + edge_contrib
    combined_spectrum = np.column_stack((target_energies, combined_intensity))
    
    # Create components dictionary
    components = {
        'spec1_contrib': np.column_stack((target_energies, spec1_contrib)),
        'spec2_contrib': np.column_stack((target_energies, spec2_contrib)),
        'edge_contrib': np.column_stack((target_energies, edge_contrib))
    }
    
    return optimal_params, combined_spectrum, components



def plot_spectral_fit_with_edge(target_spectrum, spectrum1, spectrum2, 
                              optimal_params, combined_spectrum, components,
                              labels=None, title=None, figsize=(12, 8)):
    """
    Creates a comprehensive plot showing the fit results with the edge function component.
    
    Parameters:
    -----------
    target_spectrum : ndarray, shape (n, 2)
        Target spectrum with columns [Energy, Intensity]
    spectrum1 : ndarray, shape (m, 2)
        First input spectrum with columns [Energy, Intensity]
    spectrum2 : ndarray, shape (k, 2)
        Second input spectrum with columns [Energy, Intensity]
    optimal_params : list
        The optimal parameters [fraction, edge_location, edge_height, edge_width, edge_decay]
    combined_spectrum : ndarray, shape (n, 2)
        The optimal combination of spectrum1, spectrum2, and edge
    components : dict
        Dictionary with separate components: 
        {'spec1_contrib', 'spec2_contrib', 'edge_contrib'}
    labels : list, optional
        List of labels for the spectra in the order [target, spectrum1, spectrum2, edge]
        Default: ['Target', 'Spectrum 1', 'Spectrum 2', 'Edge']
    title : str, optional
        Title for the figure
    figsize : tuple, optional
        Figure size as (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axes : list of matplotlib.axes.Axes
        The axes objects
    """
    if labels is None:
        labels = ['Target', 'Spectrum 1', 'Spectrum 2', 'Edge']
    
    # Extract parameters
    fraction = optimal_params[0]
    has_edge = np.any(components['edge_contrib'][:, 1] != 0)
    
    # Calculate the fit quality
    residuals = target_spectrum[:, 1] - combined_spectrum[:, 1]
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Set up the figure
    if has_edge:
        fig, axes = plt.subplots(2, 2, figsize=figsize, 
                                 gridspec_kw={'height_ratios': [3, 1.5]})
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = [axes[0], axes[1], None, None]
    
    # Left top plot: Reference spectra
    axes[0].plot(spectrum1[:, 0], spectrum1[:, 1], 'b-', label=labels[1])
    axes[0].plot(spectrum2[:, 0], spectrum2[:, 1], 'g-', label=labels[2])
    axes[0].set_xlabel('Energy')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Reference Spectra')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right top plot: Target vs. Fitted
    axes[1].plot(target_spectrum[:, 0], target_spectrum[:, 1], 'k-', label=labels[0])
    axes[1].plot(combined_spectrum[:, 0], combined_spectrum[:, 1], 'r--', 
                label=f'Fit (RMSE: {rmse:.5f})')
    
    # Add fit parameters to the legend
    edge_text = ''
    if has_edge:
        edge_loc, edge_height, edge_width, edge_decay = optimal_params[1:5]
        edge_text = f'\nEdge: loc={edge_loc:.2f}, h={edge_height:.2f}, w={edge_width:.2f}, d={edge_decay:.2f}'
    
    axes[1].text(0.02, 0.02, 
                f'Fraction: {fraction:.3f} × {labels[1]} + {1-fraction:.3f} × {labels[2]}{edge_text}',
                transform=axes[1].transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))
    
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('Target vs. Fitted Spectrum')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if has_edge:
        # Left bottom plot: Component contributions
        axes[2].plot(components['spec1_contrib'][:, 0], components['spec1_contrib'][:, 1], 'b-', 
                    label=f'{fraction:.3f} × {labels[1]}')
        axes[2].plot(components['spec2_contrib'][:, 0], components['spec2_contrib'][:, 1], 'g-', 
                    label=f'{1-fraction:.3f} × {labels[2]}')
        axes[2].plot(components['edge_contrib'][:, 0], components['edge_contrib'][:, 1], 'm-', 
                    label=f'{labels[3]}')
        axes[2].set_xlabel('Energy')
        axes[2].set_ylabel('Intensity')
        axes[2].set_title('Component Contributions')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Right bottom plot: Residuals
        axes[3].plot(target_spectrum[:, 0], residuals, 'k-')
        axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[3].set_xlabel('Energy')
        axes[3].set_ylabel('Residuals')
        axes[3].set_title('Fit Residuals')
        axes[3].grid(True, alpha=0.3)
    
    # Adjust layout and add overall title if provided
    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=0.9)
    
    return fig, axes

def plot_spectra_with_fixed_params(target_spectrum, spectrum1, spectrum2, 
                                  params, include_edge=True,
                                  labels=None, title=None, figsize=(10, 6)):
    """
    Plots the target spectrum against a linear combination with user-specified parameters.
    
    Parameters:
    -----------
    target_spectrum : ndarray, shape (n, 2)
        Target spectrum with columns [Energy, Intensity]
    spectrum1 : ndarray, shape (m, 2)
        First input spectrum with columns [Energy, Intensity]
    spectrum2 : ndarray, shape (k, 2)
        Second input spectrum with columns [Energy, Intensity]
    params : list
        The parameters [fraction, edge_location, edge_height, edge_width, edge_decay]
    include_edge : bool, optional
        Whether to include the step edge in the plot
    labels : list, optional
        List of labels for the spectra in the order [target, spectrum1, spectrum2, edge]
        Default: ['Target', 'Spectrum 1', 'Spectrum 2', 'Edge']
    title : str, optional
        Title for the figure
    figsize : tuple, optional
        Figure size as (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The axes object
    rmse : float
        Root mean square error between target and calculated intensities
    """
    if labels is None:
        labels = ['Target', 'Spectrum 1', 'Spectrum 2', 'Edge']
        
    # Get target energy grid
    target_energies = target_spectrum[:, 0]
    
    # Create interpolation functions for both input spectra
    interp1 = interpolate.interp1d(spectrum1[:, 0], spectrum1[:, 1], 
                                   bounds_error=False, fill_value=0)
    interp2 = interpolate.interp1d(spectrum2[:, 0], spectrum2[:, 1], 
                                   bounds_error=False, fill_value=0)
    
    # Interpolate input spectra onto target energy grid
    intensity1 = interp1(target_energies)
    intensity2 = interp2(target_energies)
    
    # Extract the parameters
    fraction = params[0]
    spec1_contrib = fraction * intensity1
    spec2_contrib = (1 - fraction) * intensity2
    
    # Calculate edge contribution if included
    if include_edge and len(params) >= 5:
        edge_loc, edge_height, edge_width, edge_decay = params[1:5]
        edge_contrib = edge_bl_func(target_energies, edge_loc, edge_height, edge_width, edge_decay)
    else:
        edge_contrib = np.zeros_like(target_energies)
        
    # Calculate combined intensity
    combined_intensity = spec1_contrib + spec2_contrib + edge_contrib
    
    # Calculate residuals and RMSE
    residuals = target_spectrum[:, 1] - combined_intensity
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the spectra
    ax.plot(target_spectrum[:, 0], target_spectrum[:, 1], 'k-', label=labels[0])
    ax.plot(target_energies, combined_intensity, 'r--', 
            label=f'Calculated (RMSE: {rmse:.5f})')
    
    # Add component spectra
    ax.plot(target_energies, spec1_contrib, 'b-', alpha=0.4, 
            label=f'{fraction:.3f} × {labels[1]}')
    ax.plot(target_energies, spec2_contrib, 'g-', alpha=0.4, 
            label=f'{1-fraction:.3f} × {labels[2]}')
    
    if include_edge and np.any(edge_contrib != 0):
        ax.plot(target_energies, edge_contrib, 'm-', alpha=0.4, 
                label=f'{labels[3]} (loc={params[1]:.2f}, h={params[2]:.2f})')
    
    # Add parameter text box
    param_text = f'Fraction: {fraction:.3f}'
    if include_edge and len(params) >= 5:
        param_text += f'\nEdge: loc={params[1]:.2f}, h={params[2]:.2f}, w={params[3]:.2f}, d={params[4]:.2f}'
    
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Energy')
    ax.set_ylabel('Intensity')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Spectra Comparison with Fixed Parameters')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax, rmse

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
                                 use_experimental_data: bool = True) -> plt.Figure:
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
    
    # Formatting
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10) #bbox_to_anchor=(1.05, 1),
    
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


class NEXAFSDatabase:
    """
    A database for storing NEXAFS spectrum data and associated SLD calculations.
    """
    
    def __init__(self, database_file='nexafs_database.json'):
        """
        Initialize the database.
        
        Parameters:
        -----------
        database_file : str
            Path to the JSON file that stores the database
        """
        self.database_file = database_file
        self.data = self._load_database()
    
    def _load_database(self):
        """Load the database from the JSON file or create a new one if it doesn't exist."""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # File exists but has invalid JSON or is empty
                print(f"Warning: {self.database_file} exists but contains invalid JSON. Creating new database.")
                return {"samples": {}}
        else:
            # File doesn't exist
            return {"samples": {}}
    
    def _save_database(self):
        """Save the database to the JSON file."""
        with open(self.database_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_sample(self, sample_id, chemical_formula, density, metadata=None):
        """
        Add a new sample to the database.
        
        Parameters:
        -----------
        sample_id : str
            Unique identifier for the sample
        chemical_formula : str
            Chemical formula of the sample
        density : float
            Density of the sample in g/cc
        metadata : dict, optional
            Additional metadata for the sample
            
        Returns:
        --------
        bool
            True if the sample was added successfully, False if it already exists
        """
        if sample_id in self.data["samples"]:
            return False
        
        self.data["samples"][sample_id] = {
            "chemical_formula": chemical_formula,
            "density": density,
            "metadata": metadata or {},
            "measurements": {}
        }
        
        self._save_database()
        return True
    
    def add_measurement(self, sample_id, measurement_id, input_file, x_min=None, x_max=None, 
                   collection_mode=None, collection_date=None, beamline=None,
                   plot_results=True, overwrite=False):
        """
        Add a new measurement to a sample.
        
        Parameters:
        -----------
        sample_id : str
            Identifier for the sample
        measurement_id : str
            Identifier for the measurement
        input_file : str
            Path to the input spectrum file
        x_min : float, optional
            Minimum energy value for merging points
        x_max : float, optional
            Maximum energy value for merging points
        collection_mode : str, optional
            Collection mode ('TEY', 'PEY', 'Transmission')
        collection_date : str, optional
            Collection date in format 'MonthYear' (e.g., 'May2023')
        beamline : str, optional
            Beamline used for the measurement
        plot_results : bool, optional
            Whether to plot the SLD values (default: True)
        overwrite : bool, optional
            Whether to overwrite an existing measurement with the same ID (default: False)
            
        Returns:
        --------
        tuple, None, or False
            (DeltaBeta, SLD) if successful, 
            None if the sample doesn't exist,
            False if a duplicate measurement exists and overwrite=False
        """
        if sample_id not in self.data["samples"]:
            return None
        
        sample = self.data["samples"][sample_id]
        
        # Check for duplicate measurement ID
        if measurement_id in sample["measurements"]:
            if not overwrite:
                print(f"Warning: Measurement '{measurement_id}' already exists for sample '{sample_id}'.")
                print("To overwrite, set overwrite=True when calling add_measurement.")
                return False
            else:
                print(f"Overwriting existing measurement '{measurement_id}' for sample '{sample_id}'.")
        
        # Calculate refractive index and SLD
        DeltaBeta, SLD = process_nexafs_to_SLD(
            input_file=input_file,
            chemical_formula=sample["chemical_formula"],
            density=sample["density"],
            x_min=x_min,
            x_max=x_max
        )
        
        # Save measurement metadata
        metadata = {
            "input_file": input_file,
            "x_min": x_min,
            "x_max": x_max,
            "collection_mode": collection_mode,
            "collection_date": collection_date,
            "beamline": beamline
        }
        
        # Save the data
        # We need to convert numpy arrays to lists for JSON serialization
        DeltaBeta_list = DeltaBeta.tolist()
        SLD_list = SLD.tolist()
        
        sample["measurements"][measurement_id] = {
            "metadata": metadata,
            "DeltaBeta": DeltaBeta_list,
            "SLD": SLD_list
        }
        
        self._save_database()
        
        # Plot the results if requested
        if plot_results:
            self.plot_sld(sample_id, measurement_id, x_min, x_max)
        
        return DeltaBeta, SLD
    
    def plot_sld(self, sample_id, measurement_id, x_min=None, x_max=None):
        """
        Plot the SLD values for a specific measurement.
        
        Parameters:
        -----------
        sample_id : str
            Identifier for the sample
        measurement_id : str
            Identifier for the measurement
        x_min : float, optional
            Minimum energy value to plot
        x_max : float, optional
            Maximum energy value to plot
        """
        # Get the data
        result = self.get_measurement(sample_id, measurement_id)
        if result is None:
            print(f"Measurement {measurement_id} for sample {sample_id} not found.")
            return
        
        DeltaBeta, SLD, metadata = result
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Filter data by energy range if specified
        mask = np.ones(len(SLD), dtype=bool)
        if x_min is not None:
            mask = mask & (SLD[:,0] >= x_min)
        if x_max is not None:
            mask = mask & (SLD[:,0] <= x_max)
        
        filtered_SLD = SLD[mask]
        
        # Scale SLD values to 10^-6 Å^-2
        sld_real_scaled = filtered_SLD[:,1] 
        sld_imag_scaled = filtered_SLD[:,2] 
        
        # Plot real part of SLD
        ax1.plot(filtered_SLD[:,0], sld_real_scaled, 'b-')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('SLD Real (10$^{-6}$ Å$^{-2}$)')
        ax1.set_title(f'Real SLD - {sample_id} - {measurement_id}')
        ax1.grid(True)
        
        # Plot imaginary part of SLD
        ax2.plot(filtered_SLD[:,0], sld_imag_scaled, 'r-')
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('SLD Imaginary (10$^{-6}$ Å$^{-2}$)')
        ax2.set_title(f'Imaginary SLD - {sample_id} - {measurement_id}')
        ax2.grid(True)
        
        # Add metadata as text
        metadata_str = []
        if metadata["collection_mode"]:
            metadata_str.append(f"Mode: {metadata['collection_mode']}")
        if metadata["collection_date"]:
            metadata_str.append(f"Date: {metadata['collection_date']}")
        if metadata["beamline"]:
            metadata_str.append(f"Beamline: {metadata['beamline']}")
        
        if metadata_str:
            fig.text(0.01, 0.01, '\n'.join(metadata_str), fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Also create a version that shows Delta and Beta
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        filtered_DeltaBeta = DeltaBeta[mask]
        
        # Plot Delta
        ax1.plot(filtered_DeltaBeta[:,0], filtered_DeltaBeta[:,1], 'b-')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Delta')
        ax1.set_title(f'Delta - {sample_id} - {measurement_id}')
        ax1.grid(True)
        
        # Plot Beta
        ax2.plot(filtered_DeltaBeta[:,0], filtered_DeltaBeta[:,2], 'r-')
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Beta')
        ax2.set_title(f'Beta - {sample_id} - {measurement_id}')
        ax2.grid(True)
        
        if metadata_str:
            fig.text(0.01, 0.01, '\n'.join(metadata_str), fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def search_samples(self, query):
        """
        Search for samples with IDs containing the query string.
        
        Parameters:
        -----------
        query : str
            Search string to look for in sample IDs
            
        Returns:
        --------
        list
            List of sample IDs that match the query
        """
        pattern = re.compile(query, re.IGNORECASE)
        matches = []
        
        for sample_id in self.data["samples"]:
            if pattern.search(sample_id):
                matches.append(sample_id)
        
        return matches
    
    def compare_samples(self, sample_ids, measurement_ids=None, x_min=None, x_max=None,
                        collection_mode=None, collection_date=None, beamline=None,
                        combined_plot=False):
        """
        Compare SLD values for multiple samples or measurements.
        
        Parameters:
        -----------
        sample_ids : list or str
            Sample IDs to compare. If a string, will be treated as a search query
        measurement_ids : list or None, optional
            Specific measurement IDs to use for each sample. If None, uses the first measurement for each sample
        x_min : float, optional
            Minimum energy value to plot
        x_max : float, optional
            Maximum energy value to plot
        collection_mode : str, optional
            Filter by collection mode
        collection_date : str, optional
            Filter by collection date
        beamline : str, optional
            Filter by beamline
        combined_plot : bool, optional
            If True, plots real and imaginary on the same graph (default: False)
            
        Returns:
        --------
        dict
            Dictionary of sample data that was plotted
        """
        # Convert string query to list of sample IDs
        if isinstance(sample_ids, str):
            sample_ids = self.search_samples(sample_ids)
        
        if not sample_ids:
            print("No samples found matching the criteria.")
            return {}
        
        # Prepare measurement data for each sample
        plot_data = {}
        
        for i, sample_id in enumerate(sample_ids):
            sample = self.get_sample(sample_id)
            if not sample:
                print(f"Sample {sample_id} not found.")
                continue
                
            # Get measurements for this sample
            if measurement_ids is not None and i < len(measurement_ids):
                meas_id = measurement_ids[i]
                measurement_list = [meas_id] if meas_id in sample["measurements"] else []
            else:
                # Filter measurements based on metadata criteria
                measurement_list = []
                for meas_id, meas in sample["measurements"].items():
                    meta = meas["metadata"]
                    include = True
                    
                    if collection_mode and meta.get("collection_mode") != collection_mode:
                        include = False
                    if collection_date and meta.get("collection_date") != collection_date:
                        include = False
                    if beamline and meta.get("beamline") != beamline:
                        include = False
                    
                    if include:
                        measurement_list.append(meas_id)
            
            if not measurement_list:
                print(f"No measurements found for sample {sample_id} matching the criteria.")
                continue
                
            # Get data for each measurement
            for meas_id in measurement_list:
                result = self.get_measurement(sample_id, meas_id)
                if result:
                    DeltaBeta, SLD, metadata = result
                    label = f"{sample_id} - {meas_id}"
                    plot_data[label] = {
                        "SLD": SLD,
                        "DeltaBeta": DeltaBeta,
                        "metadata": metadata
                    }
        
        if not plot_data:
            print("No data found matching the criteria.")
            return {}
            
        # Plot the data
        self._plot_comparison(plot_data, x_min, x_max, combined_plot)
        
        return plot_data
    
    def _plot_comparison(self, plot_data, x_min=None, x_max=None, combined_plot=False):
        """
        Plot comparison of multiple samples.
        
        Parameters:
        -----------
        plot_data : dict
            Dictionary of sample data to plot
        x_min : float, optional
            Minimum energy value to plot
        x_max : float, optional
            Maximum energy value to plot
        combined_plot : bool, optional
            If True, plots real and imaginary on the same graph
        """
        if combined_plot:
            # Single plot with real and imaginary values
            plt.figure(figsize=(12, 6))
            
            for label, data in plot_data.items():
                SLD = data["SLD"]
                
                # Filter data by energy range if specified
                mask = np.ones(len(SLD), dtype=bool)
                if x_min is not None:
                    mask = mask & (SLD[:,0] >= x_min)
                if x_max is not None:
                    mask = mask & (SLD[:,0] <= x_max)
                
                filtered_SLD = SLD[mask]
                
                # Scale SLD values to 10^-6 Å^-2
                sld_real_scaled = filtered_SLD[:,1] 
                sld_imag_scaled = filtered_SLD[:,2] 
                
                plt.plot(filtered_SLD[:,0], sld_real_scaled, '-', label=f"{label} (Real)")
                plt.plot(filtered_SLD[:,0], sld_imag_scaled, '--', label=f"{label} (Imag)")
            
            plt.xlabel('Energy (eV)')
            plt.ylabel('SLD (10$^{-6}$ Å$^{-2}$)')
            plt.title('SLD Comparison')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            # Separate plots for real and imaginary parts
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            for label, data in plot_data.items():
                SLD = data["SLD"]
                
                # Filter data by energy range if specified
                mask = np.ones(len(SLD), dtype=bool)
                if x_min is not None:
                    mask = mask & (SLD[:,0] >= x_min)
                if x_max is not None:
                    mask = mask & (SLD[:,0] <= x_max)
                
                filtered_SLD = SLD[mask]
                
                # Scale SLD values to 10^-6 Å^-2
                sld_real_scaled = filtered_SLD[:,1] 
                sld_imag_scaled = filtered_SLD[:,2] 
                
                ax1.plot(filtered_SLD[:,0], sld_real_scaled, '-', label=label)
                ax2.plot(filtered_SLD[:,0], sld_imag_scaled, '-', label=label)
            
            ax1.set_xlabel('Energy (eV)')
            ax1.set_ylabel('SLD Real (10$^{-6}$ Å$^{-2}$)')
            ax1.set_title('Real SLD Comparison')
            ax1.grid(True)
            ax1.legend()
            
            ax2.set_xlabel('Energy (eV)')
            ax2.set_ylabel('SLD Imaginary (10$^{-6}$ Å$^{-2}$)')
            ax2.set_title('Imaginary SLD Comparison')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
    
    def get_sample(self, sample_id):
        """Get a sample by its ID."""
        return self.data["samples"].get(sample_id)
    
    def get_measurement(self, sample_id, measurement_id):
        """Get a measurement by its ID."""
        sample = self.get_sample(sample_id)
        if sample and measurement_id in sample["measurements"]:
            measurement = sample["measurements"][measurement_id]
            
            # Convert lists back to numpy arrays
            DeltaBeta = np.array(measurement["DeltaBeta"])
            SLD = np.array(measurement["SLD"])
            
            return DeltaBeta, SLD, measurement["metadata"]
        return None
    
    def list_samples(self):
        """List all samples in the database."""
        return list(self.data["samples"].keys())
    
    def list_measurements(self, sample_id):
        """List all measurements for a sample."""
        sample = self.get_sample(sample_id)
        if sample:
            return list(sample["measurements"].keys())
        return []
    
    
