from refnx.reflect import SLD, MaterialSLD
from refnx.analysis import Parameter
import numpy as np
import pandas as pd
import re
from datetime import datetime
import os
import pickle
import copy
from refnx.analysis import CurveFitter

def create_reflectometry_model_density(materials_list, layer_params, layer_order=None, ignore_layers=None, 
                          sample_name=None, energy=None, wavelength=None, probe="x-ray"):
    """
    Create a complete reflectometry model with consolidated parameters using density instead of SLD.
    
    Args:
        materials_list: List of dictionaries with 'name' and either 'density' or direct SLD values:
            - name: Name of the material
            - density: Density of the material in g/cm³
            - formula: Chemical formula of the material
            - OR you can provide direct SLD values with 'real' and 'imag' keys
        layer_params: Dictionary mapping material names to parameter dictionaries
        layer_order: List of layer names in desired order (top to bottom)
        ignore_layers: List of layer names to ignore in the model name generation
        sample_name: Name of the sample being analyzed
        energy: Energy value in keV (for X-ray experiments)
        wavelength: Wavelength in Angstroms (can be used for either X-ray or neutron)
        probe: Type of radiation, 'x-ray' (default) or 'neutron'
        
    Returns:
        Tuple of (materials, layers, structure, model_name) containing all created objects and a model name
    """
   
    
    # Set default for ignore_layers if not provided
    if ignore_layers is None:
        ignore_layers = ["Si", "SiO2"]  # Default to ignoring Si and SiO2
    
    # Validate probe type
    if probe not in ["x-ray", "neutron"]:
        raise ValueError(f"Invalid probe type: {probe}. Must be 'x-ray' or 'neutron'")
    
    # For X-rays, convert energy to wavelength if needed
    if probe == "x-ray":
        if energy is not None and wavelength is None:
            # Convert energy (keV) to wavelength (Å)
            # E = hc/λ, where h is Planck's constant, c is speed of light
            # Constants for conversion: 12.398 is hc in eV·Å
            wavelength = 12.398 / energy
        elif energy is None and wavelength is None:
            raise ValueError("For X-ray measurements, either energy or wavelength must be provided")
    
    # For neutrons, wavelength is required
    if probe == "neutron" and wavelength is None:
        raise ValueError("For neutron measurements, wavelength must be provided")
    
    # Store material objects for later use with density constraints
    density_params = {}
    
    # Step 1: Create materials using refnx's MaterialSLD class for density-to-SLD conversion
    materials = {}
    for material_info in materials_list:
        name = material_info['name']
        
        # Check if we're dealing with direct SLD values (old format) or density-based (new format)
        if 'real' in material_info and 'imag' in material_info:
            # Old format with direct SLD values
            real = material_info['real']
            imag = material_info['imag']
            
            # Handle complex numbers if imag is already complex
            if isinstance(imag, complex):
                materials[name] = SLD(real + imag, name=name)
            else:
                materials[name] = SLD(real + imag*1j, name=name)
                
        elif 'density' in material_info:
            # New format with density
            density = material_info['density']
            
            # Create a Parameter for density
            density_param = Parameter(density, name=f"{name}_density")
            density_params[name] = density_param
            
            if 'formula' in material_info:
                formula = material_info['formula']
                
                # Create MaterialSLD with proper probe type and wavelength
                material_obj = MaterialSLD(formula, density=density, probe=probe, 
                                           wavelength=wavelength, name=name)
                
                materials[name] = material_obj
            else:
                # If no formula provided, use direct SLD values if available
                sld_real = material_info.get('sld_real', 0)
                sld_imag = material_info.get('sld_imag', 0)
                materials[name] = SLD(sld_real + sld_imag*1j, name=name)
        else:
            # Missing both direct SLD and density
            raise ValueError(f"Material {name} must have either 'real'/'imag' or 'density' values")
    
    # Step 2: Create layers with consolidated parameters
    Layer = {}
    
    # Track which parameters are being varied for model naming
    varying_params = {
        "D": set(),  # Density
        "T": set(),  # Thickness
        "Rg": set()  # Roughness
    }
    
    # Track which materials have varying parameters
    materials_varying = set()
    
    for name, params in layer_params.items():
        if name in materials:
            thickness = params.get("thickness", 0)
            roughness = params.get("roughness", 0)
            Layer[name] = materials[name](thickness, roughness)
            
            has_varying_param = False
            
            # Apply density bounds and track variations
            if "density_bounds" in params:
                lower, upper, vary = params["density_bounds"]
                
                # Get the density parameter if it exists
                if name in density_params and isinstance(materials[name], MaterialSLD):
                    density_param = density_params[name]
                    density_param.setp(bounds=(lower, upper), vary=vary)
                    
                    # For MaterialSLD, just link the density parameter directly
                    material_obj = materials[name]
                    material_obj.density = density_param
                    
                    if vary and name not in ignore_layers:
                        varying_params["D"].add(name)
                        has_varying_param = True
                
            # Apply thickness bounds and track variations
            if "thickness_bounds" in params:
                lower, upper, vary = params["thickness_bounds"]
                Layer[name].thick.setp(bounds=(lower, upper), vary=vary)
                if vary and name not in ignore_layers:
                    varying_params["T"].add(name)
                    has_varying_param = True
                
            # Apply roughness bounds and track variations
            if "roughness_bounds" in params:
                lower, upper, vary = params["roughness_bounds"]
                Layer[name].rough.setp(bounds=(lower, upper), vary=vary)
                if vary and name not in ignore_layers:
                    varying_params["Rg"].add(name)
                    has_varying_param = True
            
            # Add material to the set of materials with varying parameters
            if has_varying_param:
                materials_varying.add(name)
    
    # Step 3: Create structure if layer order is provided
    structure = None
    if layer_order:
        # Start with the first layer
        structure = Layer[layer_order[0]]
        
        # Add the remaining layers using the | operator
        for layer_name in layer_order[1:]:
            structure = structure | Layer[layer_name]
            
        # Ensure structure has wavelength information
        structure.wavelength = wavelength
    
    # Generate simplified model name
    # Count layers except those in ignore_layers
    active_layers = [layer for layer in layer_order if layer not in ignore_layers and layer != "air"]
    num_layers = len(active_layers)
    
    # Construct the model name with the new format: Sample_Layers
    if sample_name is None:
        sample_name = "Unknown"  # Default sample name if not provided
    
    # # Add probe type to model name
    # probe_info = ""
    # if probe == "x-ray":
    #     if energy is not None:
    #         probe_info = f"XR{energy}keV"
    #     else:
    #         probe_info = f"XR{wavelength}A"
    # elif probe == "neutron":
    #     probe_info = f"N{wavelength}A"
    
    model_name = f"{sample_name}{num_layers}Layers"
    
    # Return additional objects for fitting
    return materials, Layer, structure, model_name


def get_param_type(param_name):
    """
    Determine parameter type based on parameter name.
    
    Args:
        param_name: Name of the parameter
        
    Returns:
        String indicating parameter type ('density', 'thickness', etc.)
    """
    if '_density' in param_name:
        return 'density'
    elif 'thick' in param_name:
        return 'thickness'
    elif 'rough' in param_name:
        return 'roughness'
    elif 'scale' in param_name:
        return 'scale'
    elif 'bkg' in param_name:
        return 'background'
    elif 'sld' in param_name:
        if 'real' in param_name:
            return 'sld_real'
        elif 'imag' in param_name:
            return 'sld_imag'
        else:
            return 'sld'
    elif 'dq' in param_name:
        return 'resolution'
    else:
        return 'other'


def log_fitting_results(objective, model_name, results_df=None):
    """
    Log the results of model fitting to a pandas DataFrame, adding a numbered suffix
    to the model name if a model with the same name has been logged before.
    
    This version is adapted to properly handle density parameters.
    
    Args:
        objective: The objective function after fitting
        model_name: Name of the model used for fitting
        results_df: Existing DataFrame to append to (if None, creates new DataFrame)
        
    Returns:
        Updated pandas DataFrame with fitting results and potentially modified model name
    """
    # Initialize a new DataFrame if none is provided
    if results_df is None:
        results_df = pd.DataFrame(columns=[
            'timestamp', 'model_name', 'goodness_of_fit', 
            'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary',
            'param_type'  # Added to track if parameter is density, thickness, etc.
        ])
    
    # Check if the model name already exists and add a suffix if needed
    base_model_name = model_name
    
    # Extract existing model names from the DataFrame
    existing_names = results_df['model_name'].unique() if not results_df.empty else []
    
    # Look for models with the same base name
    matching_names = [name for name in existing_names if name == base_model_name or 
                     re.match(f"{re.escape(base_model_name)}_R[0-9]+$", name)]
    
    if matching_names:
        # Find the highest run number
        max_run = 0
        for name in matching_names:
            if name == base_model_name:
                # Base name exists without suffix
                max_run = max(max_run, 1)
            else:
                # Extract run number from suffix
                match = re.search(r"_R([0-9]+)$", name)
                if match:
                    run_num = int(match.group(1))
                    max_run = max(max_run, run_num)
        
        # Create new model name with incremented run number
        model_name = f"{base_model_name}_R{max_run + 1}"
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract the goodness of fit
    try:
        gof = objective.chisqr()
    except Exception as e:
        print(f"Error getting chisqr: {str(e)}")
        gof = None
    
    # Extract all parameters from the model
    model = objective.model
    rows = []
    
    # Try direct approach using flattened parameters
    try:
        for param in model.parameters.flattened():
            try:
                # Get parameter name and value
                param_name = param.name
                value = param.value
                stderr = getattr(param, 'stderr', None)
                vary = getattr(param, 'vary', False)
                
                # Determine parameter type
                param_type = get_param_type(param_name)
                
                # Handle bounds - refnx uses Interval object with lb and ub attributes
                bound_low = None
                bound_high = None
                
                try:
                    bounds = getattr(param, 'bounds', None)
                    if bounds is not None:
                        # Try to access as Interval object with lb and ub attributes
                        if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                            bound_low = bounds.lb
                            bound_high = bounds.ub
                        # Fallback to tuple unpacking
                        elif isinstance(bounds, tuple) and len(bounds) == 2:
                            bound_low, bound_high = bounds
                except Exception as e:
                    print(f"Error extracting bounds for {param_name}: {str(e)}")
                
                # Create a row for this parameter
                row = {
                    'timestamp': timestamp,
                    'model_name': model_name,
                    'goodness_of_fit': gof,
                    'parameter': param_name,
                    'value': value,
                    'stderr': stderr,
                    'bound_low': bound_low,
                    'bound_high': bound_high,
                    'vary': vary,
                    'param_type': param_type
                }
                rows.append(row)
            except Exception as e:
                print(f"Error processing parameter '{param_name}': {str(e)}")
                continue
    except Exception as e:
        print(f"Error accessing flattened parameters: {str(e)}")
    
    # If we couldn't extract any parameters, add a minimal entry
    if not rows:
        rows.append({
            'timestamp': timestamp,
            'model_name': model_name,
            'goodness_of_fit': gof,
            'parameter': 'unknown',
            'value': None,
            'stderr': None,
            'bound_low': None,
            'bound_high': None,
            'vary': None,
            'param_type': None
        })
    
    # Add new rows to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
    
    # Print info about the modified model name
    if model_name != base_model_name:
        print(f"Model name modified to '{model_name}' to avoid duplication")
    
    return results_df, model_name


def print_fit_results(results_df, model_spec='recent', filter_parameters=None, show_substrate=False, show_other=False):
    """
    Print a summary of fitting results with color highlighting:
    - Red: Parameters within 1% of bounds
    - Green: Varying parameters not near bounds
    - Normal: Fixed parameters
    
    Args:
        results_df: DataFrame containing fitting results
        model_spec: Specification of which model to show:
                    'recent' - the most recently added model
                    'best' - the model with the lowest goodness of fit
                    str - specific model name to display
        filter_parameters: Optional filter to show only certain parameters (string or list of strings)
                          e.g., 'thick' to show only thickness parameters
        show_substrate: Whether to show substrate layers (air, Si, SiO2)
        show_other: Whether to show parameters with type 'other' (default False)
    """
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    if results_df is None or results_df.empty:
        print("No fitting results available.")
        return
    
    # Check if param_type column exists, add it if it doesn't
    if 'param_type' not in results_df.columns:
        results_df['param_type'] = results_df['parameter'].apply(get_param_type)
    
    # Determine which model to show
    if model_spec == 'recent':
        # Get the most recent timestamp
        latest_time = results_df['timestamp'].max()
        model_results = results_df[results_df['timestamp'] == latest_time]
        if not model_results.empty:
            selected_model = model_results['model_name'].iloc[0]
        else:
            print("No models found in results.")
            return
    
    elif model_spec == 'best':
        # Group by model_name and find the one with lowest goodness of fit
        grouped = results_df.groupby('model_name')['goodness_of_fit'].first()
        valid_fits = grouped.dropna()
        
        if not valid_fits.empty:
            selected_model = valid_fits.idxmin()
        else:
            print("No valid goodness of fit values found.")
            return
    
    else:
        # Use the provided model name
        selected_model = model_spec
        if selected_model not in results_df['model_name'].unique():
            print(f"Model '{selected_model}' not found in results.")
            return
    
    # Filter for the selected model
    model_results = results_df[results_df['model_name'] == selected_model]
    
    # First filter out substrate layers unless specifically requested
    if not show_substrate:
        # Create a mask to exclude substrate layers
        substrate_mask = (
            model_results['parameter'].str.contains('air', case=False) |
            model_results['parameter'].str.contains('Si -', case=False) |
            model_results['parameter'].str.contains('SiO2 -', case=False)
        )
        # Keep only non-substrate parameters
        model_results = model_results[~substrate_mask]
    
    # Filter out 'other' parameters unless specifically requested
    if not show_other:
        model_results = model_results[model_results['param_type'] != 'other']
    
    # Then apply any additional parameter filters
    if filter_parameters:
        if isinstance(filter_parameters, str):
            # Single string filter
            model_results = model_results[model_results['parameter'].str.contains(filter_parameters, case=False)]
        elif isinstance(filter_parameters, list):
            # List of filters (any match)
            filter_mask = False
            for filter_str in filter_parameters:
                filter_mask |= model_results['parameter'].str.contains(filter_str, case=False)
            model_results = model_results[filter_mask]
    
    # Check if we have any results after filtering
    if model_results.empty:
        print(f"No matching parameters found for model '{selected_model}'.")
        return
    
    # Print header with model info
    print(f"Results for model: {selected_model}")
    
    # Get goodness of fit value (using first occurrence)
    unique_gof = model_results['goodness_of_fit'].iloc[0]
    if pd.notnull(unique_gof):
        print(f"Goodness of fit: {unique_gof:.6g}")
    else:
        print("Goodness of fit: Not available")
    
    # Print parameter values
    print("\nParameter values:")
    
    # Sort parameters for better readability
    # First by parameter type, then by whether they vary, then by name
    sorted_results = model_results.sort_values(by=['param_type', 'vary', 'parameter'], ascending=[True, False, True])
    
    current_type = None
    
    for _, row in sorted_results.iterrows():
        # Print section headers for different parameter types
        if current_type != row['param_type'] and row['param_type'] is not None:
            current_type = row['param_type']
            print(f"\n{BLUE}--- {current_type.upper()} PARAMETERS ---{RESET}")
        
        # Check if parameter is near bounds (within 1%)
        near_bounds = False
        
        if (pd.notnull(row['value']) and 
            pd.notnull(row['bound_low']) and 
            pd.notnull(row['bound_high'])):
            
            value = row['value']
            low = row['bound_low']
            high = row['bound_high']
            
            # Calculate bound range and threshold
            bound_range = high - low
            threshold = 0.01 * bound_range  # 1% of range
            
            # Check if parameter is within threshold of either bound
            if abs(value - low) < threshold or abs(high - value) < threshold:
                near_bounds = True
        
        # Construct the parameter string
        param_str = f"  {row['parameter']}: "
        
        # Add value with units based on parameter type
        if pd.notnull(row['value']):
            if row['param_type'] == 'density':
                param_str += f"{row['value']:.4f} g/cm³"
            elif row['param_type'] == 'thickness':
                param_str += f"{row['value']:.2f} Å"
            elif row['param_type'] == 'roughness':
                param_str += f"{row['value']:.2f} Å"
            elif row['param_type'] in ['sld_real', 'sld_imag', 'sld']:
                param_str += f"{row['value']:.6g} × 10⁻⁶ Å⁻²"
            else:
                param_str += f"{row['value']:.6g}"
        else:
            param_str += "N/A"
        
        # Add stderr if varying and available
        if row['vary'] and pd.notnull(row['stderr']) and row['stderr'] is not None:
            if row['param_type'] == 'density':
                param_str += f" ± {row['stderr']:.4f} g/cm³"
            elif row['param_type'] in ['thickness', 'roughness']:
                param_str += f" ± {row['stderr']:.2f} Å"
            else:
                param_str += f" ± {row['stderr']:.6g}"
        
        # Add bounds if available
        if pd.notnull(row['bound_low']) and pd.notnull(row['bound_high']):
            if row['param_type'] == 'density':
                param_str += f" (bounds: {row['bound_low']:.4f} to {row['bound_high']:.4f} g/cm³)"
            elif row['param_type'] in ['thickness', 'roughness']:
                param_str += f" (bounds: {row['bound_low']:.2f} to {row['bound_high']:.2f} Å)"
            else:
                param_str += f" (bounds: {row['bound_low']:.6g} to {row['bound_high']:.6g})"
        
        # Add vary status
        if pd.notnull(row['vary']):
            param_str += " (varying)" if row['vary'] else " (fixed)"
        
        # Apply color to the entire line based on status
        if near_bounds:
            # Red for parameters near bounds
            print(f"{RED}{param_str}{RESET}")
        elif row['vary']:
            # Green for varying parameters not near bounds
            print(f"{GREEN}{param_str}{RESET}")
        else:
            # Normal color for fixed parameters
            print(param_str)



def density_summary(results_df, model_spec='recent'):
    """
    Generate a summary table of just the density parameters for a model,
    useful for comparing density values across different materials.
    
    Args:
        results_df: DataFrame containing fitting results
        model_spec: Specification of which model to show ('recent', 'best', or a specific model name)
        
    Returns:
        DataFrame with just the density values and their uncertainties
    """
    # Check if param_type column exists, add it if it doesn't
    if 'param_type' not in results_df.columns:
        results_df['param_type'] = results_df['parameter'].apply(get_param_type)
    
    # Determine which model to show
    if model_spec == 'recent':
        # Get the most recent timestamp
        latest_time = results_df['timestamp'].max()
        model_results = results_df[results_df['timestamp'] == latest_time]
        if not model_results.empty:
            selected_model = model_results['model_name'].iloc[0]
        else:
            print("No models found in results.")
            return None
    
    elif model_spec == 'best':
        # Group by model_name and find the one with lowest goodness of fit
        grouped = results_df.groupby('model_name')['goodness_of_fit'].first()
        valid_fits = grouped.dropna()
        
        if not valid_fits.empty:
            selected_model = valid_fits.idxmin()
        else:
            print("No valid goodness of fit values found.")
            return None
    
    else:
        # Use the provided model name
        selected_model = model_spec
        if selected_model not in results_df['model_name'].unique():
            print(f"Model '{selected_model}' not found in results.")
            return None
    
    # Filter for density parameters in the selected model
    density_results = results_df[(results_df['model_name'] == selected_model) & 
                                (results_df['param_type'] == 'density')]
    
    if density_results.empty:
        print(f"No density parameters found for model '{selected_model}'.")
        return None
    
    # Create a summary DataFrame
    summary = []
    for _, row in density_results.iterrows():
        # Extract material name from parameter name
        material_name = row['parameter'].split('_density')[0]
        
        # Format value and uncertainty
        if pd.notnull(row['stderr']) and row['stderr'] is not None:
            uncertainty = row['stderr']
        else:
            uncertainty = None
        
        summary.append({
            'Material': material_name,
            'Density (g/cm³)': row['value'],
            'Uncertainty': uncertainty,
            'Varying': row['vary']
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Sort by density value
    summary_df = summary_df.sort_values('Density (g/cm³)', ascending=False)
    
    return summary_df


def run_fitting(objective, optimization_method='differential_evolution', 
                opt_workers=8, opt_popsize=20, burn_samples=5, 
                production_samples=5, prod_steps=1, pool=16,
                results_log=None, log_mcmc_stats=True,
                save_dir=None, save_objective=False, save_results=False,
                results_log_file=None, save_log_in_save_dir=False, 
                structure=None):
    """
    Run fitting procedure on a reflectometry model with optimization and MCMC sampling,
    and automatically log results. Updated to support density-based modeling.
    
    Args:
        objective: The objective function to fit
        optimization_method: Optimization method to use ('differential_evolution', 'least_squares', etc.)
        opt_workers: Number of workers for parallel optimization
        opt_popsize: Population size for genetic algorithms
        burn_samples: Number of burn-in samples to discard (in thousands)
        production_samples: Number of production samples to keep (in thousands)
        prod_steps: Number of steps between stored samples
        pool: Number of parallel processes for MCMC sampling
        results_log: Existing results log DataFrame to append to (None to create new)
        log_mcmc_stats: Whether to add MCMC statistics to the log
        save_dir: Directory to save objective and results (None to skip saving)
        save_objective: Whether to save the objective function
        save_results: Whether to save the results dictionary
        results_log_file: Filename to load/save the results log DataFrame (None to skip saving)
        save_log_in_save_dir: If True and save_dir is specified, save the log file in save_dir 
                             regardless of results_log_file path
        structure: The structure object associated with the objective, to be saved along with results
        
    Returns:
        Tuple of (results_dict, updated_results_log, model_name)
    """
    # Extract model name if available
    model_name = getattr(objective.model, 'name', 'unnamed_model')
    print(f"Fitting model: {model_name}")
    
    # Generate timestamp for logs and internal tracking (but not filenames)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine the actual log file path
    actual_log_file = results_log_file
    if save_dir is not None and save_log_in_save_dir:
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        
        # If results_log_file is specified, use just the filename part in save_dir
        if results_log_file:
            log_filename = os.path.basename(results_log_file)
        else:
            # Default log filename if not specified
            log_filename = f"fitting_results_log.csv"
        
        actual_log_file = os.path.join(save_dir, log_filename)
    
    # Try to load existing results_log if filename is provided
    if results_log is None and actual_log_file is not None:
        if os.path.exists(actual_log_file):
            try:
                print(f"Loading existing results log from {actual_log_file}")
                results_log = pd.read_csv(actual_log_file)
                
                # Check if the loaded DataFrame has the new 'param_type' column
                # If not, add it and populate it based on parameter names
                if 'param_type' not in results_log.columns:
                    print("Upgrading results log to include parameter types")
                    results_log['param_type'] = results_log['parameter'].apply(get_param_type)
            except Exception as e:
                print(f"Error loading results log: {str(e)}")
                print("Initializing new results log")
                results_log = pd.DataFrame(columns=[
                    'timestamp', 'model_name', 'goodness_of_fit', 
                    'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary', 'param_type'
                ])
    
    # Initialize results dictionary
    results = {
        'objective': objective,
        'initial_chi_squared': objective.chisqr(),
        'optimized_parameters': None,
        'optimized_chi_squared': None,
        'mcmc_samples': None,
        'mcmc_stats': None,
        'timestamp': timestamp,  # Keep timestamp in results for logs
        'structure': structure  # Include the structure if provided
    }
    
    # Create fitter
    print(f"Initializing CurveFitter with {model_name} model")
    fitter = CurveFitter(objective)
    
    # Run optimization
    print(f"Starting optimization using {optimization_method}...")
    if optimization_method == 'differential_evolution':
        fitter.fit(optimization_method, workers=opt_workers, popsize=opt_popsize)
    else:
        fitter.fit(optimization_method)
    
    # Store optimization results
    results['optimized_parameters'] = objective.parameters.pvals.copy()
    results['optimized_chi_squared'] = objective.chisqr()
    
    print(f"Optimization complete. Chi-squared improved from {results['initial_chi_squared']:.4f} to {results['optimized_chi_squared']:.4f}")
    
    # Run burn-in MCMC samples
    if burn_samples > 0:
        print(f"Running {burn_samples}k burn-in MCMC samples...")
        fitter.sample(burn_samples, pool=pool)
        print("Burn-in complete. Resetting chain...")
        fitter.reset()
    
    # Run production MCMC samples
    if production_samples > 0:
        print(f"Running {production_samples}k production MCMC samples with {prod_steps} steps between stored samples...")
        results['mcmc_samples'] = fitter.sample(production_samples, prod_steps, pool=pool)
        
        # Calculate statistics from MCMC chain
        try:
            print("Calculating parameter statistics from MCMC chain...")
            results['mcmc_stats'] = {}
            
            # Process parameter statistics
            for param in objective.parameters.flattened():
                name = param.name
                if param.vary:
                    param_stats = {
                        'name': name,
                        'value': param.value,
                        'stderr': param.stderr,
                        'median': None,
                        'mean': None,
                        'std': None,
                        'percentiles': {}
                    }
                    
                    # Extract chain for this parameter
                    chain_index = fitter.var_pars.index(param)
                    if chain_index >= 0 and results['mcmc_samples'] is not None:
                        # Calculate statistics
                        chain_values = results['mcmc_samples'][:, chain_index]
                        param_stats['median'] = np.median(chain_values)
                        param_stats['mean'] = np.mean(chain_values)
                        param_stats['std'] = np.std(chain_values)
                        
                        # Calculate percentiles
                        for percentile in [2.5, 16, 50, 84, 97.5]:
                            param_stats['percentiles'][percentile] = np.percentile(chain_values, percentile)
                    
                    results['mcmc_stats'][name] = param_stats
            
            # Print a summary of key parameters
            print("\nParameter summary from MCMC:")
            for name, stats in results['mcmc_stats'].items():
                if 'median' in stats and stats['median'] is not None:
                    param_type = get_param_type(name)
                    if param_type == 'density':
                        print(f"  {name}: {stats['median']:.4f} +{stats['percentiles'][84] - stats['median']:.4f} -{stats['median'] - stats['percentiles'][16]:.4f} g/cm³")
                    elif param_type in ['thickness', 'roughness']:
                        print(f"  {name}: {stats['median']:.2f} +{stats['percentiles'][84] - stats['median']:.2f} -{stats['median'] - stats['percentiles'][16]:.2f} Å")
                    else:
                        print(f"  {name}: {stats['median']:.6g} +{stats['percentiles'][84] - stats['median']:.6g} -{stats['median'] - stats['percentiles'][16]:.6g}")
        
        except Exception as e:
            print(f"Error calculating MCMC statistics: {str(e)}")
    
    # Initialize results_log if not provided and not loaded from file
    if results_log is None:
        print("Initializing new results log")
        results_log = pd.DataFrame(columns=[
            'timestamp', 'model_name', 'goodness_of_fit', 
            'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary', 'param_type'
        ])
    
    # Log the results
    print(f"Logging results for model {model_name}")
    
    # Log the optimized values first
    results_log, model_name = log_fitting_results(objective, model_name, results_log)
    
    # If MCMC was performed and we want to log those stats, create a second entry
    if log_mcmc_stats and results['mcmc_stats'] is not None:
        print("Adding MCMC statistics to the log...")
        
        # Create a temporary copy of the objective to store MCMC medians
        mcmc_objective = copy.deepcopy(objective)
        
        # Update parameter values to MCMC medians
        for name, stats in results['mcmc_stats'].items():
            if 'median' in stats and stats['median'] is not None:
                # Find the parameter and update its value and error
                for param in mcmc_objective.parameters.flattened():
                    if param.name == name:
                        param.value = stats['median']
                        # Use percentiles for errors
                        upper_err = stats['percentiles'][84] - stats['median']
                        lower_err = stats['median'] - stats['percentiles'][16]
                        # Use the larger of the two for stderr
                        param.stderr = max(upper_err, lower_err)
                        break
        
        # Log the MCMC results with a modified name
        mcmc_model_name = f"{model_name}_MCMC"
        results_log, _ = log_fitting_results(mcmc_objective, mcmc_model_name, results_log)
    
    # Save the objective and/or results if requested
    if save_dir is not None and (save_objective or save_results):
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        
        # Save the objective if requested - without timestamp in filename
        if save_objective:
            objective_filename = os.path.join(save_dir, f"{model_name}_objective.pkl")
            try:
                with open(objective_filename, 'wb') as f:
                    pickle.dump(objective, f)
                print(f"Saved objective to {objective_filename}")
            except Exception as e:
                print(f"Error saving objective: {str(e)}")
        
        # Save the results if requested - without timestamp in filename
        if save_results:
            # Create a copy of results without the objective (to avoid duplication if saving both)
            save_results_copy = results.copy()
            if 'objective' in save_results_copy and save_objective:
                save_results_copy['objective'] = None  # Remove the objective to avoid duplication
            
            results_filename = os.path.join(save_dir, f"{model_name}_results.pkl")
            try:
                with open(results_filename, 'wb') as f:
                    pickle.dump(save_results_copy, f)
                print(f"Saved results to {results_filename}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
            
            # Save a combined file with results and structure - without timestamp in filename
            combined_filename = os.path.join(save_dir, f"{model_name}_combined.pkl")
            try:
                combined_data = {
                    'results': save_results_copy,
                    'structure': structure,
                    'objective': objective if save_objective else None,
                    'model_name': model_name,
                    'timestamp': timestamp  # Keep timestamp in the data
                }
                with open(combined_filename, 'wb') as f:
                    pickle.dump(combined_data, f)
                print(f"Saved combined results and structure to {combined_filename}")
            except Exception as e:
                print(f"Error saving combined data: {str(e)}")
            
            # Additionally, save MCMC samples as numpy array if they exist - without timestamp in filename
            if results['mcmc_samples'] is not None:
                mcmc_filename = os.path.join(save_dir, f"{model_name}_mcmc_samples.npy")
                try:
                    # Check if the MCMC samples array is homogeneous before saving
                    # This addresses the "setting an array element with a sequence" error
                    mcmc_samples = results['mcmc_samples']
                    
                    # Test if array is homogeneous by trying to convert to float
                    try:
                        mcmc_samples_float = np.array(mcmc_samples, dtype=float)
                        np.save(mcmc_filename, mcmc_samples_float)
                    except (ValueError, TypeError):
                        # If conversion fails, it might have an inhomogeneous shape
                        print("Warning: MCMC samples may have inhomogeneous shape, saving as object array")
                        # Save as object array instead
                        np.save(mcmc_filename, np.array(mcmc_samples, dtype=object))
                        
                    print(f"Saved MCMC samples to {mcmc_filename}")
                except Exception as e:
                    print(f"Error saving MCMC samples: {str(e)}")
                    # Try to print more information about the array
                    try:
                        print(f"MCMC samples shape: {np.array(results['mcmc_samples']).shape}")
                        print(f"MCMC samples type: {type(results['mcmc_samples'])}")
                    except:
                        pass
    
    # Save the results log if a filename was provided
    if actual_log_file is not None:
        # Create directory for results_log_file if it doesn't exist
        log_dir = os.path.dirname(actual_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created directory for results log: {log_dir}")
        
        try:
            # Save to CSV
            results_log.to_csv(actual_log_file, index=False)
            print(f"Saved results log to {actual_log_file}")
        except Exception as e:
            print(f"Error saving results log: {str(e)}")
    
    return results, results_log, model_name


def modelcomparisonplot(obj_list, structure_list, shade_start=None, 
                        fig_size_w=16, colors=None, profile_shift=-10, xlim=None,
                        zoom_xlim=None, zoom_ylim=None):
    """
    Create a flexible, comprehensive comparison plot for multiple reflectometry models
    with a zoomed-in view of the reflectivity.
    
    Args:
        obj_list: List of objective functions for each model (data will be extracted from these)
        structure_list: List of structure objects for each model
        shade_start: List of starting positions for layer shading (None for auto-detection)
        fig_size_w: Width of the figure (height will be adjusted for three rows)
        colors: List of colors for layer shading (None for default colors)
        profile_shift: Shift applied to depth profiles (default: -10)
        xlim: Custom x-axis limits for SLD plots as [min, max] (None for auto)
        zoom_xlim: X-axis limits for zoomed reflectivity plots as (min, max).
                  Default is (0, 0.05) to focus on low-Q region.
        zoom_ylim: Y-axis limits for zoomed reflectivity plots as (min, max).
                  Default is based on scale parameter (scale*0.5, scale*1.5).
        
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
    
    # Set default zoom_xlim if not provided
    if zoom_xlim is None:
        zoom_xlim = (0, 0.05)  # Focus on low-Q region
    
    # Create figure and axes - now with 3 rows
    if plot_number == 1:
        fig, axes = plt.subplots(3, 1, figsize=(fig_size_w, 12))  # Increased height for 3 rows
        # For single plot, axes will be a 1D array with 3 elements
        ax_refl = axes[0]
        ax_refl_zoom = axes[1]  # New zoomed reflectivity axis
        ax_sld = axes[2]
    else:
        fig, axes = plt.subplots(3, plot_number, figsize=(fig_size_w, 12))  # Increased height for 3 rows
    
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
            ax_refl_zoom = axes[1, i]  # New zoomed reflectivity axis
            ax_sld = axes[2, i]
        
        # Extract data from objective
        data = obj_list[i].data
        
        # Try to extract scale parameter for y-axis zoom limits
        try:
            # Attempt to get the scale parameter from the model
            scale = obj_list[i].model.scale.value
            
            # Set default zoom_ylim based on scale if not provided
            current_zoom_ylim = zoom_ylim
            if current_zoom_ylim is None:
                current_zoom_ylim = (scale * 0.5, scale * 1.5)
        except (AttributeError, IndexError):
            # Fallback if scale parameter can't be accessed
            current_zoom_ylim = zoom_ylim if zoom_ylim is not None else (0.5, 1.5)
        
        # Plot reflectivity data on first row (full view)
        ax_refl.plot(data.data[0], data.data[1], 'o', markersize=3, label='Data')
        ax_refl.set_yscale('log')
        ax_refl.plot(data.data[0], obj_list[i].model(data.data[0]), '-', label='Simulation')
        ax_refl.legend(loc='upper right')
        ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_refl.set_ylabel('Reflectivity (a.u)')
        ax_refl.text(0.125, 1, f'RelGF {chi_values[i, 1]}', size=12, 
                   horizontalalignment='center', verticalalignment='top')
        
        # Plot reflectivity data again on second row (zoomed view)
        ax_refl_zoom.plot(data.data[0], data.data[1], 'o', markersize=3, label='Data')
        model_y = obj_list[i].model(data.data[0])
        ax_refl_zoom.plot(data.data[0], model_y, '-', label='Simulation')
        
        # Set custom x limits for zoom
        ax_refl_zoom.set_xlim(zoom_xlim)
        
        # Set linear scale for zoomed view (instead of log)
        ax_refl_zoom.set_yscale('linear')
        
        # Set y limits based on data within the zoom region
        # Filter data points within the zoom x-range
        mask = (data.data[0] >= zoom_xlim[0]) & (data.data[0] <= zoom_xlim[1])
        if np.any(mask):
            # Extract corresponding y values and model predictions
            y_in_range = data.data[1][mask]
            model_in_range = model_y[mask]
            
            # Combine data and model values to find range
            all_y = np.concatenate([y_in_range, model_in_range])
            
            # Set y limits to include all visible data plus some margin
            y_min = np.min(all_y) * 0.9
            y_max = np.max(all_y) * 1.1
            
            # Use these limits unless custom zoom_ylim is provided
            if current_zoom_ylim is not None:
                ax_refl_zoom.set_ylim(current_zoom_ylim)
            else:
                ax_refl_zoom.set_ylim(y_min, y_max)
        else:
            # If no data in range, use provided zoom_ylim or defaults
            if current_zoom_ylim is not None:
                ax_refl_zoom.set_ylim(current_zoom_ylim)
        
        ax_refl_zoom.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_refl_zoom.set_ylabel('Reflectivity (linear)')
        
        # Add a rectangle to the full view showing the zoom region
        import matplotlib.patches as patches
        
        # Get the actual y limits after they've been set
        y_limits = ax_refl_zoom.get_ylim()
        
        # Create a rectangular patch in log space
        rect = patches.Rectangle((zoom_xlim[0], 10**(np.log10(ax_refl.get_ylim()[0]) * 0.9)), 
                                 zoom_xlim[1]-zoom_xlim[0], 
                                 10**(np.log10(ax_refl.get_ylim()[1]) * 0.9) - 10**(np.log10(ax_refl.get_ylim()[0]) * 0.9), 
                                 linewidth=1, edgecolor='r', facecolor='none',
                                 alpha=0.5)
        ax_refl.add_patch(rect)
        
        # Annotate the zoom region
        ax_refl.annotate('Zoom\nRegion', 
                        xy=(zoom_xlim[0] + (zoom_xlim[1]-zoom_xlim[0])/2, 
                            10**(np.log10(ax_refl.get_ylim()[0]) * 0.95)),
                        xytext=(zoom_xlim[0] + (zoom_xlim[1]-zoom_xlim[0])/2, 
                                10**(np.log10(ax_refl.get_ylim()[0]) * 0.95)),
                        ha='center', va='bottom',
                        color='red', fontsize=8)
        
        # Plot SLD profiles on third row
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
        
        # Add legend and axis labels
        ax_sld.legend(loc='upper right')
        ax_sld.set_xlabel(r'Distance from Si ($\AA$)')
        ax_sld.set_ylabel(r'SLD $(10^{-6})$ $\AA^{-2}')
    
    plt.tight_layout()
    
    # Return the appropriate axes structure
    return fig, axes


def profileflip(structure):
    """Helper function to extract SLD profiles from a structure."""
    # This function needs to be implemented based on your existing code
    # Typically, it should return depth and SLD arrays for both real and imaginary components
    try:
        z, sld = structure.sld_profile(z=None)
        return z, sld.real, z, sld.imag
    except:
        # If the above doesn't work, provide a fallback implementation
        import numpy as np
        try:
            # Try alternate method for refnx structures
            z = np.linspace(-10, 300, 1000)
            sld_profile = structure.sld_profile(z)
            return z, np.real(sld_profile), z, np.imag(sld_profile)
        except:
            # Last resort fallback
            z = np.linspace(0, 300, 1000)
            return z, np.zeros_like(z), z, np.zeros_like(z)
        
import numpy as np
import pandas as pd
import copy
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from refnx.analysis import CurveFitter
from refnx.reflect import Structure

def setup_parameter_sweep(objective, param_name, sweep_values, 
                          existing_results_log=None, model_name=None):
    """
    Setup a parameter sweep for a reflectometry model.
    
    Args:
        objective: The objective function to fit
        param_name: Name of the parameter to sweep (must match exactly a parameter in the model)
        sweep_values: List or array of values to sweep the parameter over
        existing_results_log: Existing results log DataFrame to append to (None to create new)
        model_name: Base name for the model (if None, will use objective.model.name)
        
    Returns:
        Dictionary containing the sweep setup information
    """
    # Validate inputs
    if not param_name:
        raise ValueError("Parameter name must be provided")
    
    if not isinstance(sweep_values, (list, np.ndarray)):
        raise ValueError("sweep_values must be a list or numpy array")
    
    if len(sweep_values) == 0:
        raise ValueError("sweep_values must contain at least one value")
    
    # Extract model name if not provided
    if model_name is None:
        model_name = getattr(objective.model, 'name', 'unnamed_model')
    
    # Find the parameter in the objective
    param_found = False
    for param in objective.parameters.flattened():
        if param.name == param_name:
            param_found = True
            original_value = param.value
            original_vary = param.vary
            break
    
    if not param_found:
        raise ValueError(f"Parameter '{param_name}' not found in the model")
    
    # Initialize results_log if not provided
    if existing_results_log is None:
        results_log = pd.DataFrame(columns=[
            'timestamp', 'model_name', 'goodness_of_fit', 
            'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary', 'param_type',
            'swept_param', 'swept_value'  # Add columns for the swept parameter
        ])
    else:
        # Check if the existing log has the swept parameter columns
        if 'swept_param' not in existing_results_log.columns:
            existing_results_log['swept_param'] = None
        if 'swept_value' not in existing_results_log.columns:
            existing_results_log['swept_value'] = None
        results_log = existing_results_log
    
    # Create a dictionary to hold sweep information
    sweep_info = {
        'param_name': param_name,
        'sweep_values': sweep_values,
        'original_value': original_value,
        'original_vary': original_vary,
        'base_model_name': model_name,
        'results_log': results_log,
        'objective': objective,
        'models': [],  # Will store model names for each sweep value
        'goodness_of_fit': [],  # Will store goodness of fit for each sweep value
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    print(f"Parameter sweep setup for '{param_name}' with {len(sweep_values)} values")
    print(f"Base model name: {model_name}")
    
    return sweep_info


def run_parameter_sweep(sweep_info, optimization_method='differential_evolution', 
                       opt_workers=8, opt_popsize=20, burn_samples=0, 
                       production_samples=0, prod_steps=1, pool=16,
                       save_dir=None, save_intermediate=True, 
                       save_combined=True, save_log_file=None):
    """
    Run a parameter sweep for a reflectometry model.
    
    Args:
        sweep_info: Dictionary from setup_parameter_sweep containing sweep parameters
        optimization_method: Optimization method to use
        opt_workers: Number of workers for parallel optimization
        opt_popsize: Population size for genetic algorithms
        burn_samples: Number of burn-in samples for MCMC (in thousands, 0 to skip)
        production_samples: Number of production samples for MCMC (in thousands, 0 to skip)
        prod_steps: Number of steps between stored MCMC samples
        pool: Number of parallel processes for MCMC sampling
        save_dir: Directory to save results (None to skip saving)
        save_intermediate: Whether to save intermediate results for each sweep value
        save_combined: Whether to save combined results at the end
        save_log_file: Filename to save the results log (None to skip saving)
        
    Returns:
        Updated sweep_info dictionary with results
    """
    # Extract parameters from sweep_info
    param_name = sweep_info['param_name']
    sweep_values = sweep_info['sweep_values']
    original_value = sweep_info['original_value']
    original_vary = sweep_info['original_vary']
    base_model_name = sweep_info['base_model_name']
    results_log = sweep_info['results_log']
    objective = sweep_info['objective']
    
    # Create save directory if specified
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Find the parameter in the objective
    param = None
    for p in objective.parameters.flattened():
        if p.name == param_name:
            param = p
            break
    
    if param is None:
        raise ValueError(f"Parameter '{param_name}' not found in the model")
    
    # Store original state to restore later
    original_obj = copy.deepcopy(objective)
    
    # Initialize arrays to store results
    sweep_info['models'] = []
    sweep_info['goodness_of_fit'] = []
    sweep_info['parameter_values'] = []
    sweep_info['all_results'] = []
    
    # Run the sweep
    print(f"Starting parameter sweep for '{param_name}' over {len(sweep_values)} values")
    
    for i, sweep_value in enumerate(sweep_values):
        print(f"\n--- Sweep {i+1}/{len(sweep_values)}: {param_name} = {sweep_value} ---")
        
        # Create a fresh copy of the objective for this sweep
        current_obj = copy.deepcopy(original_obj)
        
        # Find the parameter in the new objective
        current_param = None
        for p in current_obj.parameters.flattened():
            if p.name == param_name:
                current_param = p
                break
        
        # Set the parameter value and fix it
        current_param.value = sweep_value
        current_param.vary = False
        
        # Create a unique model name for this sweep
        model_name = f"{base_model_name}_{param_name}_{sweep_value}"
        current_obj.model.name = model_name
        
        # Create fitter
        fitter = CurveFitter(current_obj)
        
        # Run optimization
        print(f"Starting optimization using {optimization_method}...")
        if optimization_method == 'differential_evolution':
            fitter.fit(optimization_method, workers=opt_workers, popsize=opt_popsize)
        else:
            fitter.fit(optimization_method)
        
        # Store optimization results
        optimized_chi_squared = current_obj.chisqr()
        print(f"Optimization complete. Chi-squared: {optimized_chi_squared:.4f}")
        
        # Run MCMC if requested
        mcmc_samples = None
        mcmc_stats = None
        
        if burn_samples > 0 and production_samples > 0:
            # Run burn-in MCMC samples
            print(f"Running {burn_samples}k burn-in MCMC samples...")
            fitter.sample(burn_samples, pool=pool)
            print("Burn-in complete. Resetting chain...")
            fitter.reset()
            
            # Run production MCMC samples
            print(f"Running {production_samples}k production MCMC samples...")
            mcmc_samples = fitter.sample(production_samples, prod_steps, pool=pool)
            
            # Calculate statistics from MCMC chain
            try:
                print("Calculating parameter statistics from MCMC chain...")
                mcmc_stats = {}
                
                for param in current_obj.parameters.flattened():
                    if param.vary:
                        param_stats = {
                            'name': param.name,
                            'value': param.value,
                            'stderr': param.stderr,
                            'median': None,
                            'mean': None,
                            'std': None,
                            'percentiles': {}
                        }
                        
                        # Extract chain for this parameter
                        chain_index = fitter.var_pars.index(param)
                        if chain_index >= 0 and mcmc_samples is not None:
                            # Calculate statistics
                            chain_values = mcmc_samples[:, chain_index]
                            param_stats['median'] = np.median(chain_values)
                            param_stats['mean'] = np.mean(chain_values)
                            param_stats['std'] = np.std(chain_values)
                            
                            # Calculate percentiles
                            for percentile in [2.5, 16, 50, 84, 97.5]:
                                param_stats['percentiles'][percentile] = np.percentile(chain_values, percentile)
                        
                        mcmc_stats[param.name] = param_stats
            except Exception as e:
                print(f"Error calculating MCMC statistics: {str(e)}")
        
        # Store results for this sweep value
        current_results = {
            'objective': current_obj,
            'model_name': model_name,
            'sweep_value': sweep_value,
            'optimized_chi_squared': optimized_chi_squared,
            'optimized_parameters': current_obj.parameters.pvals.copy(),
            'mcmc_samples': mcmc_samples,
            'mcmc_stats': mcmc_stats
        }
        
        sweep_info['all_results'].append(current_results)
        sweep_info['models'].append(model_name)
        sweep_info['goodness_of_fit'].append(optimized_chi_squared)
        sweep_info['parameter_values'].append(sweep_value)
        
        # Log the results with extra columns for swept parameter
        from copy import deepcopy
        log_obj = deepcopy(current_obj)
        structure = getattr(log_obj.model, 'structure', None)
        
        # Update results_log with this sweep value
        results_log, model_name = log_fitting_results_with_sweep(
            log_obj, model_name, results_log, param_name, sweep_value
        )
        
        # Save intermediate results if requested
        if save_dir and save_intermediate:
            save_sweep_results(current_results, save_dir)
    
    # Restore original parameter value and vary status
    param.value = original_value
    param.vary = original_vary
    
    # Update the results log in the sweep_info
    sweep_info['results_log'] = results_log
    
    # Find the best fit
    best_idx = np.argmin(sweep_info['goodness_of_fit'])
    best_value = sweep_info['parameter_values'][best_idx]
    best_gof = sweep_info['goodness_of_fit'][best_idx]
    
    print(f"\nBest fit at {param_name} = {best_value} with goodness of fit = {best_gof:.6g}")
    
    # Add best_fit info to sweep_info
    sweep_info['best_fit'] = {
        'index': best_idx,
        'value': best_value,
        'gof': best_gof,
        'model_name': sweep_info['models'][best_idx]
    }
    
    
    # Save combined results if requested
    if save_dir and save_combined:
        sweep_filename = os.path.join(save_dir, f"{base_model_name}_sweep_{param_name}.pkl")
        try:
            with open(sweep_filename, 'wb') as f:
                pickle.dump(sweep_info, f)
            print(f"Saved combined sweep results to {sweep_filename}")
        except Exception as e:
            print(f"Error saving combined sweep results: {str(e)}")
    
    # Save the results log if a filename was provided
    if save_log_file:
        log_path = save_log_file
        if save_dir:
            log_path = os.path.join(save_dir, os.path.basename(save_log_file))
        
        try:
            # Save to CSV
            results_log.to_csv(log_path, index=False)
            print(f"Saved results log to {log_path}")
        except Exception as e:
            print(f"Error saving results log: {str(e)}")
    
    # Print summary of sweep results
    print("\n--- Parameter Sweep Summary ---")
    print(f"Parameter: {param_name}")
    print(f"Number of values: {len(sweep_values)}")
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Sweep Value': sweep_info['parameter_values'],
        'Model Name': sweep_info['models'],
        'Goodness of Fit': sweep_info['goodness_of_fit']
    })
    
    
    
    return sweep_info, summary_df


def log_fitting_results_with_sweep(objective, model_name, results_df, swept_param, swept_value):
    """
    Log the results of model fitting to a pandas DataFrame, with additional columns
    for the parameter being swept and its current value.
    
    Args:
        objective: The objective function after fitting
        model_name: Name of the model used for fitting
        results_df: Existing DataFrame to append to
        swept_param: Name of the parameter being swept
        swept_value: Current value of the swept parameter
        
    Returns:
        Updated pandas DataFrame with fitting results and potentially modified model name
    """
    # Check if the results_df has the swept parameter columns
    if 'swept_param' not in results_df.columns:
        results_df['swept_param'] = None
    if 'swept_value' not in results_df.columns:
        results_df['swept_value'] = None
    
    # Check if the model name already exists and add a suffix if needed
    base_model_name = model_name
    
    # Extract existing model names from the DataFrame
    existing_names = results_df['model_name'].unique() if not results_df.empty else []
    
    # Look for models with the same base name
    import re
    matching_names = [name for name in existing_names if name == base_model_name or 
                    re.match(f"{re.escape(base_model_name)}_R[0-9]+$", name)]
    
    if matching_names:
        # Find the highest run number
        max_run = 0
        for name in matching_names:
            if name == base_model_name:
                # Base name exists without suffix
                max_run = max(max_run, 1)
            else:
                # Extract run number from suffix
                match = re.search(r"_R([0-9]+)$", name)
                if match:
                    run_num = int(match.group(1))
                    max_run = max(max_run, run_num)
        
        # Create new model name with incremented run number
        model_name = f"{base_model_name}_R{max_run + 1}"
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract the goodness of fit
    try:
        gof = objective.chisqr()
    except Exception as e:
        print(f"Error getting chisqr: {str(e)}")
        gof = None
    
    # Extract all parameters from the model
    model = objective.model
    rows = []
    
    # Try direct approach using flattened parameters
    try:
        for param in model.parameters.flattened():
            try:
                # Get parameter name and value
                param_name = param.name
                value = param.value
                stderr = getattr(param, 'stderr', None)
                vary = getattr(param, 'vary', False)
                
                # Determine parameter type
                param_type = get_param_type(param_name)
                
                # Handle bounds - refnx uses Interval object with lb and ub attributes
                bound_low = None
                bound_high = None
                
                try:
                    bounds = getattr(param, 'bounds', None)
                    if bounds is not None:
                        # Try to access as Interval object with lb and ub attributes
                        if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                            bound_low = bounds.lb
                            bound_high = bounds.ub
                        # Fallback to tuple unpacking
                        elif isinstance(bounds, tuple) and len(bounds) == 2:
                            bound_low, bound_high = bounds
                except Exception as e:
                    print(f"Error extracting bounds for {param_name}: {str(e)}")
                
                # Create a row for this parameter
                row = {
                    'timestamp': timestamp,
                    'model_name': model_name,
                    'goodness_of_fit': gof,
                    'parameter': param_name,
                    'value': value,
                    'stderr': stderr,
                    'bound_low': bound_low,
                    'bound_high': bound_high,
                    'vary': vary,
                    'param_type': param_type,
                    'swept_param': swept_param,
                    'swept_value': swept_value
                }
                rows.append(row)
            except Exception as e:
                print(f"Error processing parameter '{param_name}': {str(e)}")
                continue
    except Exception as e:
        print(f"Error accessing flattened parameters: {str(e)}")
    
    # If we couldn't extract any parameters, add a minimal entry
    if not rows:
        rows.append({
            'timestamp': timestamp,
            'model_name': model_name,
            'goodness_of_fit': gof,
            'parameter': 'unknown',
            'value': None,
            'stderr': None,
            'bound_low': None,
            'bound_high': None,
            'vary': None,
            'param_type': None,
            'swept_param': swept_param,
            'swept_value': swept_value
        })
    
    # Add new rows to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
    
    # Print info about the modified model name
    if model_name != base_model_name:
        print(f"Model name modified to '{model_name}' to avoid duplication")
    
    return results_df, model_name


def save_sweep_results(results, save_dir, prefix=None):
    """
    Save the results of a single point in a parameter sweep.
    
    Args:
        results: Dictionary containing results for a single sweep point
        save_dir: Directory to save results
        prefix: Optional prefix for filenames
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_name = results['model_name']
    if prefix:
        model_name = f"{prefix}_{model_name}"
    
    # Save the objective
    objective_filename = os.path.join(save_dir, f"{model_name}_objective.pkl")
    try:
        with open(objective_filename, 'wb') as f:
            pickle.dump(results['objective'], f)
        print(f"Saved objective to {objective_filename}")
    except Exception as e:
        print(f"Error saving objective: {str(e)}")
    
    # Save MCMC samples if they exist
    if results['mcmc_samples'] is not None:
        mcmc_filename = os.path.join(save_dir, f"{model_name}_mcmc_samples.npy")
        try:
            np.save(mcmc_filename, np.array(results['mcmc_samples'], dtype=float))
            print(f"Saved MCMC samples to {mcmc_filename}")
        except Exception as e:
            print(f"Error saving MCMC samples: {str(e)}")


def plot_parameter_sweep(sweep_info, param_limits=None, log_y=True, 
                        normalize_gof=False, highlight_best=True,
                        uncertainty_percent=10, shade_uncertainty=True,
                        figure_size=(10, 6), save_path=None):
    """
    Plot the results of a parameter sweep, showing goodness of fit vs parameter value.
    
    Args:
        sweep_info: Dictionary from run_parameter_sweep containing sweep results
        param_limits: Optional tuple of (min, max) to limit x-axis
        log_y: Whether to use log scale for y-axis (goodness of fit)
        normalize_gof: Whether to normalize goodness of fit relative to the best fit
        highlight_best: Whether to highlight the best fit point
        uncertainty_percent: Percent change in goodness of fit to use for uncertainty (default: 10)
        shade_uncertainty: Whether to add shading for uncertainty range
        figure_size: Size of the figure as (width, height)
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract data from sweep_info
    param_name = sweep_info['param_name']
    param_values = sweep_info['parameter_values']
    gof_values = sweep_info['goodness_of_fit']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Normalize GOF if requested
    y_values = gof_values.copy() if isinstance(gof_values, np.ndarray) else gof_values.copy()
    y_label = 'Goodness of Fit'
    
    if normalize_gof:
        min_gof = min(gof_values)
        y_values = [gof / min_gof for gof in gof_values]
        y_label = 'Relative Goodness of Fit'
    
    # Plot the data
    ax.plot(param_values, y_values, 'o-', linewidth=2, markersize=8)
    
    # Set logarithmic scale for y-axis if requested
    if log_y:
        ax.set_yscale('log')
    
    # Set x-axis limits if provided
    if param_limits:
        ax.set_xlim(param_limits)
    
    # Determine best fit index and value
    best_idx = None
    best_x = None
    best_y = None
    uncertainty_threshold = None
    
    if 'best_fit' in sweep_info:
        best_idx = sweep_info['best_fit']['index']
        best_x = param_values[best_idx]
        best_y = y_values[best_idx]
        
        # Calculate the uncertainty threshold based on percentage change
        uncertainty_threshold = best_y * (1 + uncertainty_percent/100)
    
    # Highlight the best fit if requested
    if highlight_best and best_idx is not None:
        ax.plot(best_x, best_y, 'o', markersize=12, color='red')
        ax.annotate(f'Best fit: {best_x:.4g}',
                   xy=(best_x, best_y),
                   xytext=(0, 20),
                   textcoords='offset points',
                   ha='center',
                   arrowprops=dict(arrowstyle='->'))
    
    # Add shading for uncertainty range if requested
    if shade_uncertainty and best_idx is not None and uncertainty_threshold is not None:
        # Find values within the uncertainty threshold
        within_threshold = [i for i, gof in enumerate(y_values) if gof <= uncertainty_threshold]
        
        if within_threshold:
            # Create mask for values within threshold
            x_array = np.array(param_values)
            y_array = np.array(y_values)
            mask = y_array <= uncertainty_threshold
            
            # Get min and max parameter values within threshold
            min_param = min(x_array[mask])
            max_param = max(x_array[mask])
            
            # Add vertical shading for uncertainty range
            ax.axvspan(min_param, max_param, alpha=0.2, color='blue', 
                     label=f'{uncertainty_percent}% GOF uncertainty')
            
            # Add horizontal line at uncertainty threshold
            ax.axhline(y=uncertainty_threshold, linestyle='--', color='blue', alpha=0.7,
                      label=f'{uncertainty_percent}% worse than best fit')
            
            # Add text annotation
            if log_y:
                # Position the text in log space to avoid crowding
                text_y = best_y * (uncertainty_threshold / best_y) ** 0.5
            else:
                text_y = best_y + (uncertainty_threshold - best_y) / 2
            
            # Determine position for annotation
            text_x = best_x
            
            uncertainty_text = f"Parameter range for {uncertainty_percent}% GOF uncertainty:\n"
            uncertainty_text += f"{param_name} ∈ [{min_param:.4g}, {max_param:.4g}]"
            
            # Add text box with uncertainty information on the plot
            ax.text(0.5, 0.95, uncertainty_text, 
                   transform=ax.transAxes, ha='center', va='top', 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add labels and title
    ax.set_xlabel(f'Parameter Value: {param_name}')
    ax.set_ylabel(y_label)
    ax.set_title(f'Parameter Sweep Results for {param_name}')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if we have the uncertainty items
    if shade_uncertainty and best_idx is not None and uncertainty_threshold is not None:
        ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def compare_sweep_reflectivity(sweep_info, indices=None, values=None,
                             include_best=True, figure_size=(16, 12),
                             zoom_xlim=None, zoom_ylim=None, save_path=None):
    """
    Compare reflectivity curves from different points in a parameter sweep,
    using a similar format to the modelcomparisonplot function.
    
    Args:
        sweep_info: Dictionary from run_parameter_sweep containing sweep results
        indices: List of indices to include in the comparison (None for all)
        values: List of parameter values to include (alternative to indices)
        include_best: Whether to include the best fit curve
        figure_size: Size of the figure as (width, height)
        zoom_xlim: X-axis limits for zoomed reflectivity plot as (min, max)
        zoom_ylim: Y-axis limits for zoomed reflectivity plot as (min, max)
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches
    
    # Validate inputs - either indices or values should be provided, not both
    if indices is not None and values is not None:
        raise ValueError("Provide either indices or values, not both")
    
    # Get the parameter name and all values
    param_name = sweep_info['param_name']
    all_values = sweep_info['parameter_values']
    all_results = sweep_info['all_results']
    
    # Determine which indices to include
    if indices is None and values is None:
        # Use all indices if neither is provided
        indices = list(range(len(all_values)))
    elif values is not None:
        # Convert values to indices
        indices = []
        for value in values:
            try:
                idx = all_values.index(value)
                indices.append(idx)
            except ValueError:
                print(f"Warning: Value {value} not found in sweep values")
    
    # Add the best fit index if requested
    if include_best and 'best_fit' in sweep_info:
        best_idx = sweep_info['best_fit']['index']
        if best_idx not in indices:
            indices.append(best_idx)
    
    # Create a 3-row figure similar to modelcomparisonplot
    fig, axes = plt.subplots(3, 1, figsize=figure_size)
    ax_refl = axes[0]      # Full reflectivity plot
    ax_refl_zoom = axes[1] # Zoomed reflectivity plot
    ax_sld = axes[2]       # SLD profiles
    
    # Color cycle for multiple profiles
    colors = plt.cm.tab10.colors
    
    # Get reference data from the first valid objective
    reference_obj = None
    for idx in indices:
        if idx < 0 or idx >= len(all_results):
            continue
        reference_obj = all_results[idx]['objective']
        break
    
    if reference_obj is None:
        print("Error: No valid objective found")
        return fig, axes

def plot_best_fit_with_uncertainty(sweep_info, uncertainty_percent=10, 
                                 figure_size=(16, 12), zoom_xlim=None, zoom_ylim=None,
                                 sld_xlim=None, profile_shift=0, save_path=None):
    """
    Plot the best fit reflectivity curve and SLD profile with shading to indicate
    uncertainty corresponding to a specified percent change in goodness of fit.
    
    Args:
        sweep_info: Dictionary from run_parameter_sweep containing sweep results
        uncertainty_percent: Percent change in goodness of fit to use for uncertainty (default: 10)
        figure_size: Size of the figure as (width, height)
        zoom_xlim: X-axis limits for zoomed reflectivity plot as (min, max)
        zoom_ylim: Y-axis limits for zoomed reflectivity plot as (min, max)
        sld_xlim: X-axis limits for SLD profile plot as (min, max)
        profile_shift: Shift applied to depth profiles
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches
    
    # Check if best fit exists
    if 'best_fit' not in sweep_info:
        raise ValueError("No best fit found in sweep_info")
    
    # Get the parameter name and best fit value
    param_name = sweep_info['param_name']
    best_idx = sweep_info['best_fit']['index']
    best_value = sweep_info['parameter_values'][best_idx]
    best_gof = sweep_info['goodness_of_fit'][best_idx]
    
    # Calculate the threshold for uncertainty
    gof_threshold = best_gof * (1 + uncertainty_percent/100)
    
    # Find parameter values within the threshold
    values_within_threshold = []
    indices_within_threshold = []
    
    for i, (value, gof) in enumerate(zip(sweep_info['parameter_values'], sweep_info['goodness_of_fit'])):
        if gof <= gof_threshold:
            values_within_threshold.append(value)
            indices_within_threshold.append(i)
    
    # Get the min and max values within threshold
    if values_within_threshold:
        min_value = min(values_within_threshold)
        max_value = max(values_within_threshold)
    else:
        # If no values within threshold, just use the best value
        min_value = best_value
        max_value = best_value
    
    # Create a 3-row figure similar to modelcomparisonplot
    fig, axes = plt.subplots(3, 1, figsize=figure_size)
    ax_refl = axes[0]      # Full reflectivity plot
    ax_refl_zoom = axes[1] # Zoomed reflectivity plot
    ax_sld = axes[2]       # SLD profiles
    
    # Get reference objective (best fit)
    best_obj = sweep_info['all_results'][best_idx]['objective']
    data = best_obj.data
    
    # Get best fit model and data
    best_model_y = best_obj.model(data.data[0])
    
    # ---- Plot full reflectivity ----
    # Plot data points
    ax_refl.plot(data.data[0], data.data[1], 'o', markersize=3, color='black', label='Data')
    
    # Plot best fit
    ax_refl.plot(data.data[0], best_model_y, '-', linewidth=2, color='blue', 
               label=f'Best fit ({param_name}={best_value})')
    
    # Calculate uncertainty band for reflectivity
    min_model_y = None
    max_model_y = None
    
    if len(indices_within_threshold) > 1:
        # Initialize with the first model within threshold
        first_idx = indices_within_threshold[0]
        first_obj = sweep_info['all_results'][first_idx]['objective']
        first_model_y = first_obj.model(data.data[0])
        
        min_model_y = first_model_y.copy()
        max_model_y = first_model_y.copy()
        
        # Find min and max for each Q point across all models within threshold
        for idx in indices_within_threshold[1:]:
            obj = sweep_info['all_results'][idx]['objective']
            model_y = obj.model(data.data[0])
            
            min_model_y = np.minimum(min_model_y, model_y)
            max_model_y = np.maximum(max_model_y, model_y)
    
    # Add uncertainty band if calculated
    if min_model_y is not None and max_model_y is not None:
        ax_refl.fill_between(data.data[0], min_model_y, max_model_y, 
                           color='blue', alpha=0.2, 
                           label=f'Uncertainty ({uncertainty_percent}% GOF)')
    
    # Set up full reflectivity plot
    ax_refl.set_yscale('log')
    ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
    ax_refl.set_ylabel('Reflectivity (a.u)')
    ax_refl.legend(loc='upper right')
    
    # ---- Plot zoomed reflectivity ----
    # Plot data points
    ax_refl_zoom.plot(data.data[0], data.data[1], 'o', markersize=3, color='black', label='Data')
    
    # Plot best fit
    ax_refl_zoom.plot(data.data[0], best_model_y, '-', linewidth=2, color='blue', 
                    label=f'Best fit ({param_name}={best_value})')
    
    # Add uncertainty band if calculated
    if min_model_y is not None and max_model_y is not None:
        ax_refl_zoom.fill_between(data.data[0], min_model_y, max_model_y, 
                                color='blue', alpha=0.2, 
                                label=f'Uncertainty ({uncertainty_percent}% GOF)')
    
    # Set custom x limits for zoom
    if zoom_xlim is None:
        zoom_xlim = (0, 0.05)  # Default zoom to low-Q region
    
    ax_refl_zoom.set_xlim(zoom_xlim)
    
    # Set linear scale for zoomed view
    ax_refl_zoom.set_yscale('linear')
    
    # Set custom y limits for zoom if provided
    if zoom_ylim is not None:
        ax_refl_zoom.set_ylim(zoom_ylim)
    else:
        # Auto-determine y limits based on data in the zoom region
        mask = (data.data[0] >= zoom_xlim[0]) & (data.data[0] <= zoom_xlim[1])
        if np.any(mask):
            y_in_range = data.data[1][mask]
            best_y_in_range = best_model_y[mask]
            
            # Also include uncertainty ranges if available
            all_y = []
            all_y.extend(y_in_range)
            all_y.extend(best_y_in_range)
            
            if min_model_y is not None and max_model_y is not None:
                all_y.extend(min_model_y[mask])
                all_y.extend(max_model_y[mask])
            
            # Set y limits with margin
            y_min = np.min(all_y) * 0.9
            y_max = np.max(all_y) * 1.1
            ax_refl_zoom.set_ylim(y_min, y_max)
    
    ax_refl_zoom.set_xlabel(r'Q ($\AA^{-1}$)')
    ax_refl_zoom.set_ylabel('Reflectivity (linear)')
    ax_refl_zoom.legend(loc='best')
    
    # Add a rectangle to the full view showing the zoom region
    rect = patches.Rectangle((zoom_xlim[0], 10**(np.log10(ax_refl.get_ylim()[0]) * 0.9)), 
                           zoom_xlim[1]-zoom_xlim[0], 
                           10**(np.log10(ax_refl.get_ylim()[1]) * 0.9) - 10**(np.log10(ax_refl.get_ylim()[0]) * 0.9), 
                           linewidth=1, edgecolor='r', facecolor='none',
                           alpha=0.5)
    ax_refl.add_patch(rect)
    
    # Annotate the zoom region
    ax_refl.annotate('Zoom\nRegion', 
                    xy=(zoom_xlim[0] + (zoom_xlim[1]-zoom_xlim[0])/2, 
                        10**(np.log10(ax_refl.get_ylim()[0]) * 0.95)),
                    xytext=(zoom_xlim[0] + (zoom_xlim[1]-zoom_xlim[0])/2, 
                            10**(np.log10(ax_refl.get_ylim()[0]) * 0.95)),
                    ha='center', va='bottom',
                    color='red', fontsize=8)
    
    # ---- Plot SLD profiles ----
    # Get best fit SLD profile
    best_structure = getattr(best_obj.model, 'structure', None)
    
    if best_structure is not None:
        # Get best SLD profile
        z, best_sld = best_structure.sld_profile(z=None)
        
        # MODIFIED: Flip the SLD profile by:
        # 1. Find the maximum depth value (will be the new origin for Silicon)
        max_depth = np.max(z)
        # 2. Create flipped z-values so Silicon is at 0 and air is at max
        flipped_z = max_depth - z
        
        # Plot best fit SLD profile with flipped z-axis
        ax_sld.plot(flipped_z + profile_shift, best_sld.real, '-', linewidth=2, color='blue',
                  label=f'Best fit ({param_name}={best_value})')
        ax_sld.plot(flipped_z + profile_shift, best_sld.imag, ':', linewidth=1.5, color='blue',
                  label='Imaginary SLD')
        
        # Calculate uncertainty bands for SLD profile
        min_sld_real = None
        max_sld_real = None
        min_sld_imag = None
        max_sld_imag = None
        
        if len(indices_within_threshold) > 1:
            # Initialize arrays for min/max SLD values
            min_sld_real = np.full_like(best_sld.real, np.inf)
            max_sld_real = np.full_like(best_sld.real, -np.inf)
            min_sld_imag = np.full_like(best_sld.imag, np.inf)
            max_sld_imag = np.full_like(best_sld.imag, -np.inf)
            
            # Find min and max SLD values across all models within threshold
            for idx in indices_within_threshold:
                obj = sweep_info['all_results'][idx]['objective']
                structure = getattr(obj.model, 'structure', None)
                
                if structure is not None:
                    try:
                        # Get current SLD profile
                        current_z, current_sld = structure.sld_profile(z)
                        
                        # Update min/max values
                        min_sld_real = np.minimum(min_sld_real, current_sld.real)
                        max_sld_real = np.maximum(max_sld_real, current_sld.real)
                        min_sld_imag = np.minimum(min_sld_imag, current_sld.imag)
                        max_sld_imag = np.maximum(max_sld_imag, current_sld.imag)
                    except Exception as e:
                        print(f"Error calculating SLD profile for index {idx}: {str(e)}")
        
        # Add uncertainty bands if calculated - with flipped z-axis
        if min_sld_real is not None and max_sld_real is not None:
            # Check if we have valid values (not inf)
            if not (np.isinf(min_sld_real).any() or np.isinf(max_sld_real).any()):
                ax_sld.fill_between(flipped_z + profile_shift, min_sld_real, max_sld_real, 
                                  color='blue', alpha=0.2, 
                                  label=f'Uncertainty ({uncertainty_percent}% GOF)')
            
            # Also show imaginary SLD uncertainty if requested
            if not (np.isinf(min_sld_imag).any() or np.isinf(max_sld_imag).any()):
                ax_sld.fill_between(flipped_z + profile_shift, min_sld_imag, max_sld_imag, 
                                  color='blue', alpha=0.1)
        
        # Set up SLD plot
        if sld_xlim is not None:
            # Adjust sld_xlim if provided
            ax_sld.set_xlim(sld_xlim)
        else:
            # If no specific limits are provided, ensure we show the full flipped range
            ax_sld.set_xlim(0, max_depth + profile_shift)
        
        # Modify x-axis label to reflect the new orientation
        ax_sld.set_xlabel(r'Distance from Si ($\AA$)')
        ax_sld.set_ylabel(r'SLD ($10^{-6}$ $\AA^{-2}$)')
        ax_sld.legend(loc='best')
        
        # Add annotation about uncertainty
        min_val = min_value if min_value != best_value else None
        max_val = max_value if max_value != best_value else None
        
        uncertainty_text = f"Parameter range for {uncertainty_percent}% GOF uncertainty:\n"
        uncertainty_text += f"{param_name} = {best_value}"
        
        if min_val is not None or max_val is not None:
            uncertainty_text += " ["
            if min_val is not None:
                uncertainty_text += f"{min_val}"
            else:
                uncertainty_text += f"{best_value}"
            
            uncertainty_text += " to "
            
            if max_val is not None:
                uncertainty_text += f"{max_val}"
            else:
                uncertainty_text += f"{best_value}"
            
            uncertainty_text += "]"
        
        # Add text box with uncertainty information
        ax_sld.text(0.02, 0.98, uncertainty_text, 
                  transform=ax_sld.transAxes, ha='left', va='top', 
                  fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add a note about the orientation
        ax_sld.text(0.98, 0.02, "SLD profile flipped: Si at 0, air at max depth", 
                  transform=ax_sld.transAxes, ha='right', va='bottom', 
                  fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add title with sweep parameter information
    fig.suptitle(f"Best Fit with {uncertainty_percent}% GOF Uncertainty for {param_name}", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, axes

def get_param_type(param_name):
    """
    Determine parameter type based on parameter name.
    
    Args:
        param_name: Name of the parameter
        
    Returns:
        String indicating parameter type ('density', 'thickness', etc.)
    """
    if '_density' in param_name:
        return 'density'
    elif 'thick' in param_name:
        return 'thickness'
    elif 'rough' in param_name:
        return 'roughness'
    elif 'scale' in param_name:
        return 'scale'
    elif 'bkg' in param_name:
        return 'background'
    elif 'sld' in param_name:
        if 'real' in param_name:
            return 'sld_real'
        elif 'imag' in param_name:
            return 'sld_imag'
        else:
            return 'sld'
    elif 'dq' in param_name:
        return 'resolution'
    else:
        return 'other'


def compare_sweep_profiles(sweep_info, indices=None, values=None,
                          include_best=True, figure_size=(12, 8),
                          sld_xlim=None, profile_shift=0, save_path=None):
    """
    Compare SLD profiles from different points in a parameter sweep.
    
    Args:
        sweep_info: Dictionary from run_parameter_sweep containing sweep results
        indices: List of indices to include in the comparison (None for all)
        values: List of parameter values to include (alternative to indices)
        include_best: Whether to include the best fit profile
        figure_size: Size of the figure as (width, height)
        sld_xlim: x-axis limits for SLD profile as (min, max)
        profile_shift: Shift applied to depth profiles
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Validate inputs - either indices or values should be provided, not both
    if indices is not None and values is not None:
        raise ValueError("Provide either indices or values, not both")
    
    # Get the parameter name and all values
    param_name = sweep_info['param_name']
    all_values = sweep_info['parameter_values']
    all_results = sweep_info['all_results']
    
    # Determine which indices to include
    if indices is None and values is None:
        # Use all indices if neither is provided
        indices = list(range(len(all_values)))
    elif values is not None:
        # Convert values to indices
        indices = []
        for value in values:
            try:
                idx = all_values.index(value)
                indices.append(idx)
            except ValueError:
                print(f"Warning: Value {value} not found in sweep values")
    
    # Add the best fit index if requested
    if include_best and 'best_fit' in sweep_info:
        best_idx = sweep_info['best_fit']['index']
        if best_idx not in indices:
            indices.append(best_idx)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Color cycle for multiple profiles
    colors = plt.cm.tab10.colors
    
    # Plot each selected profile
    for i, idx in enumerate(indices):
        if idx < 0 or idx >= len(all_results):
            print(f"Warning: Index {idx} out of range")
            continue
        
        # Get the objective and extract structure
        obj = all_results[idx]['objective']
        model = obj.model
        structure = getattr(model, 'structure', None)
        
        if structure is None:
            print(f"Warning: No structure found for index {idx}")
            continue
        
        # Generate the SLD profile
        try:
            # Try to get the SLD profile
            z, sld = structure.sld_profile(z=None)
            
            # Plot the profile
            value = all_values[idx]
            color = colors[i % len(colors)]
            
            # Add marker for the best fit
            is_best = 'best_fit' in sweep_info and idx == sweep_info['best_fit']['index']
            marker = 'o' if is_best else None
            markersize = 4 if is_best else 0
            linestyle = '-' if is_best else '--'
            linewidth = 2 if is_best else 1.5
            
            # Create label based on parameter value
            label = f"{param_name} = {value}"
            if is_best:
                label += " (Best Fit)"
            
            # Plot the real part of the SLD
            ax.plot(z + profile_shift, sld.real, color=color, linestyle=linestyle, 
                   linewidth=linewidth, marker=marker, markersize=markersize,
                   markevery=20, label=label)
            
        except Exception as e:
            print(f"Error generating SLD profile for index {idx}: {str(e)}")
    
    # Set axis labels and title
    ax.set_xlabel(r'Distance from interface ($\AA$)')
    ax.set_ylabel(r'SLD ($10^{-6}$ $\AA^{-2}$)')
    ax.set_title(f'SLD Profiles for {param_name} Sweep')
    
    # Set x-axis limits if provided
    if sld_xlim:
        ax.set_xlim(sld_xlim)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def extract_parameter_trends(sweep_info, param_pattern=None, include_fixed=False):
    """
    Extract trends of how other parameters change during a parameter sweep.
    
    Args:
        sweep_info: Dictionary from run_parameter_sweep containing sweep results
        param_pattern: String pattern to filter parameters (None for all)
        include_fixed: Whether to include fixed parameters
        
    Returns:
        DataFrame with parameter values across the sweep
    """
    # Extract data from sweep_info
    swept_param = sweep_info['param_name']
    sweep_values = sweep_info['parameter_values']
    all_results = sweep_info['all_results']
    
    # Create a list to store all parameters
    all_params = {}
    
    # First, collect all parameter names across all results
    for result in all_results:
        obj = result['objective']
        for param in obj.parameters.flattened():
            if param.name != swept_param and (include_fixed or param.vary):
                all_params[param.name] = []
    
    # Fill in parameter values for each sweep point
    for result in all_results:
        obj = result['objective']
        
        # Keep track of which parameters we've found in this objective
        found_params = set()
        
        for param in obj.parameters.flattened():
            if param.name in all_params:
                all_params[param.name].append(param.value)
                found_params.add(param.name)
        
        # Fill in None for any parameters not found in this objective
        for param_name in all_params:
            if param_name not in found_params:
                all_params[param_name].append(None)
    
    # Create a DataFrame with the swept parameter values as index
    df = pd.DataFrame(index=sweep_values)
    df.index.name = swept_param
    
    # Add each parameter as a column
    for param_name, values in all_params.items():
        # Skip if parameter_pattern is provided and doesn't match
        if param_pattern is not None and param_pattern not in param_name:
            continue
        
        df[param_name] = values
    
    return df


def plot_parameter_trends(trend_df, param_names=None, figure_size=(12, 8), 
                         log_y=False, save_path=None):
    """
    Plot trends of how other parameters change during a parameter sweep.
    
    Args:
        trend_df: DataFrame from extract_parameter_trends
        param_names: List of parameter names to plot (None for all)
        figure_size: Size of the figure as (width, height)
        log_y: Whether to use log scale for y-axis
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    # Determine which parameters to plot
    if param_names is None:
        param_names = trend_df.columns
    
    # Create subplots
    n_params = len(param_names)
    if n_params == 0:
        print("No parameters to plot")
        return None, None
    
    # Determine optimal subplot grid based on number of parameters
    import math
    n_cols = min(3, n_params)
    n_rows = math.ceil(n_params / n_cols)
    
    # Adjust figure size based on grid
    adj_figure_size = (figure_size[0], figure_size[1] * (n_rows / 2))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=adj_figure_size)
    
    # Handle case where axes is a single axis instead of array
    if n_params == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy indexing
    axes = np.array(axes).flatten()
    
    # Get the index name (swept parameter)
    swept_param = trend_df.index.name
    
    # Plot each parameter
    for i, param_name in enumerate(param_names):
        if i < len(axes) and param_name in trend_df.columns:
            ax = axes[i]
            
            # Plot the parameter values
            ax.plot(trend_df.index, trend_df[param_name], 'o-', linewidth=2)
            
            # Set y-axis scale if requested
            if log_y:
                ax.set_yscale('log')
            
            # Add labels
            ax.set_xlabel(swept_param)
            ax.set_ylabel(param_name)
            ax.set_title(f'Trend of {param_name}')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.5)
    
    # Hide any unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, axes


def run_multi_parameter_sweep(objective, param_configs, optimization_method='differential_evolution',
                             opt_workers=8, opt_popsize=20, save_dir=None, include_mcmc=False):
    """
    Run a sweep over multiple parameters, one at a time.
    
    Args:
        objective: The objective function to fit
        param_configs: List of dictionaries, each with:
                      - 'name': Parameter name
                      - 'values': List of values to sweep
                      - 'model_name_prefix': Optional prefix for model naming
        optimization_method: Optimization method to use
        opt_workers: Number of workers for parallel optimization
        opt_popsize: Population size for genetic algorithms
        save_dir: Directory to save results
        include_mcmc: Whether to include MCMC sampling
        
    Returns:
        Dictionary with results for each parameter sweep
    """
    # Initialize results dictionary
    multi_sweep_results = {}
    
    # Extract base model name
    base_model_name = getattr(objective.model, 'name', 'base_model')
    
    # Create save directory if specified
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Initialize results log DataFrame
    results_log = pd.DataFrame(columns=[
        'timestamp', 'model_name', 'goodness_of_fit', 
        'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary', 'param_type',
        'swept_param', 'swept_value'
    ])
    
    # Run sweep for each parameter
    for i, param_config in enumerate(param_configs):
        param_name = param_config['name']
        param_values = param_config['values']
        model_name_prefix = param_config.get('model_name_prefix', f"Sweep{i+1}")
        
        print(f"\n=== Starting sweep {i+1}/{len(param_configs)}: {param_name} ===")
        
        # Create a custom model name for this sweep
        model_name = f"{model_name_prefix}_{base_model_name}"
        
        # Setup the parameter sweep
        sweep_info = setup_parameter_sweep(
            objective=objective,
            param_name=param_name,
            sweep_values=param_values,
            existing_results_log=results_log,
            model_name=model_name
        )
        
        # Set MCMC parameters based on include_mcmc flag
        burn_samples = 5 if include_mcmc else 0
        production_samples = 5 if include_mcmc else 0
        
        # Run the parameter sweep
        sweep_info, summary_df = run_parameter_sweep(
            sweep_info=sweep_info,
            optimization_method=optimization_method,
            opt_workers=opt_workers,
            opt_popsize=opt_popsize,
            burn_samples=burn_samples,
            production_samples=production_samples,
            save_dir=save_dir,
            save_intermediate=True,
            save_combined=True,
            save_log_file=os.path.join(save_dir, f"multi_sweep_results.csv") if save_dir else None
        )
        
        # Update the shared results log
        results_log = sweep_info['results_log']
        
        # Store sweep results
        multi_sweep_results[param_name] = {
            'sweep_info': sweep_info,
            'summary': summary_df
        }
        
        # Plot and save the parameter sweep results
        if save_dir:
            fig, ax = plot_parameter_sweep(sweep_info)
            plt.savefig(os.path.join(save_dir, f"{model_name_prefix}_{param_name}_sweep.png"), dpi=300)
            plt.close(fig)
    
    # Create a comprehensive summary DataFrame
    summary_data = []
    for param_name, result in multi_sweep_results.items():
        sweep_info = result['sweep_info']
        best_fit = sweep_info['best_fit']
        
        summary_data.append({
            'Parameter': param_name,
            'Number of Values': len(sweep_info['parameter_values']),
            'Best Value': best_fit['value'],
            'Best Goodness of Fit': best_fit['gof'],
            'Model Name': best_fit['model_name']
        })
    
    multi_summary_df = pd.DataFrame(summary_data)
    
    # Save the comprehensive summary
    if save_dir:
        multi_summary_df.to_csv(os.path.join(save_dir, "multi_sweep_summary.csv"), index=False)
    
    # Return the results
    return multi_sweep_results, multi_summary_df, results_log


def global_parameter_scan(objective, param_configs, n_repeats=1, optimization_method='differential_evolution',
                         opt_workers=8, opt_popsize=20, save_dir=None):
    """
    Perform a comprehensive scan by testing all combinations of parameter values.
    
    Args:
        objective: The objective function to fit
        param_configs: List of dictionaries, each with:
                      - 'name': Parameter name
                      - 'values': List of values to sweep
        n_repeats: Number of times to repeat each combination (for statistical analysis)
        optimization_method: Optimization method to use
        opt_workers: Number of workers for parallel optimization
        opt_popsize: Population size for genetic algorithms
        save_dir: Directory to save results
        
    Returns:
        DataFrame with results for all parameter combinations
    """
    # Extract parameter names and values
    param_names = [config['name'] for config in param_configs]
    param_values = [config['values'] for config in param_configs]
    
    # Generate all combinations of parameter values
    import itertools
    combinations = list(itertools.product(*param_values))
    
    # Create save directory if specified
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Initialize results DataFrame
    results = []
    
    # Extract base model name
    base_model_name = getattr(objective.model, 'name', 'scan_model')
    
    # Process each combination
    total_combinations = len(combinations) * n_repeats
    counter = 0
    
    for combo in combinations:
        # Create parameter dictionary for this combination
        combo_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
        
        # Generate a model name for this combination
        model_name = f"{base_model_name}_scan"
        for name, value in combo_dict.items():
            model_name += f"_{name}_{value}"
        
        # Repeat the fitting process n_repeats times
        for repeat in range(n_repeats):
            counter += 1
            print(f"\n--- Processing combination {counter}/{total_combinations} ---")
            print(f"Parameters: {combo_dict}")
            if n_repeats > 1:
                print(f"Repeat: {repeat+1}/{n_repeats}")
            
            # Create a fresh copy of the objective
            current_obj = copy.deepcopy(objective)
            
            # Set parameter values and fix them
            fixed_params = []
            for name, value in combo_dict.items():
                # Find the parameter in the model
                for param in current_obj.parameters.flattened():
                    if param.name == name:
                        param.value = value
                        param.vary = False
                        fixed_params.append(name)
                        break
            
            # Update model name with repeat number if needed
            if n_repeats > 1:
                current_model_name = f"{model_name}_R{repeat+1}"
            else:
                current_model_name = model_name
            
            current_obj.model.name = current_model_name
            
            # Create fitter
            fitter = CurveFitter(current_obj)
            
            # Run optimization
            print(f"Starting optimization using {optimization_method}...")
            if optimization_method == 'differential_evolution':
                fitter.fit(optimization_method, workers=opt_workers, popsize=opt_popsize)
            else:
                fitter.fit(optimization_method)
            
            # Record goodness of fit
            gof = current_obj.chisqr()
            print(f"Optimization complete. Chi-squared: {gof:.4f}")
            
            # Save objective if requested
            if save_dir:
                objective_filename = os.path.join(save_dir, f"{current_model_name}_objective.pkl")
                try:
                    with open(objective_filename, 'wb') as f:
                        pickle.dump(current_obj, f)
                except Exception as e:
                    print(f"Error saving objective: {str(e)}")
            
            # Create result entry
            result = {
                'model_name': current_model_name,
                'goodness_of_fit': gof,
                'repeat': repeat + 1 if n_repeats > 1 else None
            }
            
            # Add parameter values to result
            for name, value in combo_dict.items():
                result[name] = value
            
            # Add other varying parameters to result
            for param in current_obj.parameters.flattened():
                if param.vary:
                    result[f"{param.name}_value"] = param.value
                    if hasattr(param, 'stderr') and param.stderr is not None:
                        result[f"{param.name}_error"] = param.stderr
            
            # Add to results list
            results.append(result)
            
            # Save intermediate results
            if save_dir:
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(save_dir, "scan_results.csv"), index=False)
    
    # Create final DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    if save_dir:
        results_df.to_csv(os.path.join(save_dir, "scan_results.csv"), index=False)
    
    return results_df


def plot_scan_heatmap(scan_results, x_param, y_param, value_param='goodness_of_fit',
                    log_scale=True, cmap='viridis_r', figure_size=(10, 8),
                    save_path=None):
    """
    Plot a heatmap of scan results for two parameters.
    
    Args:
        scan_results: DataFrame from global_parameter_scan
        x_param: Parameter name for x-axis
        y_param: Parameter name for y-axis
        value_param: Parameter to use for color values (default: goodness_of_fit)
        log_scale: Whether to use log scale for color values
        cmap: Colormap to use
        figure_size: Size of the figure as (width, height)
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract unique parameter values
    x_values = sorted(scan_results[x_param].unique())
    y_values = sorted(scan_results[y_param].unique())
    
    # Create a grid of values
    grid = np.zeros((len(y_values), len(x_values)))
    
    # If there are repeats, calculate mean values
    if 'repeat' in scan_results.columns and scan_results['repeat'].nunique() > 1:
        # Group by parameter values and calculate mean
        grouped = scan_results.groupby([x_param, y_param])[value_param].mean().reset_index()
        
        # Fill the grid
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                # Find the corresponding row in grouped
                idx = (grouped[x_param] == x_val) & (grouped[y_param] == y_val)
                if idx.any():
                    grid[i, j] = grouped[value_param][idx.idxmax()]
    else:
        # Fill the grid directly
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                # Find the corresponding row
                idx = (scan_results[x_param] == x_val) & (scan_results[y_param] == y_val)
                if idx.any():
                    grid[i, j] = scan_results[value_param][idx.idxmax()]
    
    # Apply log scale if requested
    if log_scale and np.all(grid > 0):
        grid = np.log10(grid)
        value_label = f'log10({value_param})'
    else:
        value_label = value_param
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Plot heatmap
    im = ax.imshow(grid, cmap=cmap, aspect='auto', origin='lower',
                 extent=[min(x_values), max(x_values), min(y_values), max(y_values)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_label)
    
    # Add labels and title
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f'Parameter Scan Results: {value_param}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax