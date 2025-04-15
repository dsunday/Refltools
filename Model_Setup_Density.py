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