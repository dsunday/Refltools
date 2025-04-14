import numpy as np
from copy import deepcopy
import pandas as pd
from datetime import datetime
import re


from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel, MaterialSLD
from refnx.reflect.structure import isld_profile

import matplotlib.pyplot as plt
import pickle



## Model Generation Functions

def create_materials(materials_list):
    """
    Create multiple SLD objects from a list of material definitions
    using the existing SLD class.
    
    Args:
        materials_list: List of dictionaries containing name, real, and imag values
        
    Returns:
        Dictionary of material names to SLD objects
    """
    materials = {}
    
    for material in materials_list:
        name = material['name']
        real = material['real']
        imag = material['imag']
        
        # Create SLD object using the existing class
        materials[name] = SLD(real +imag, name=name)
    
    return materials

def create_layers(materials, thicknesses, roughnesses):
    """
    Create Layer objects from materials with specified thicknesses and roughnesses
    
    Args:
        materials: Dictionary of material names to SLD objects
        thicknesses: Dictionary mapping material names to thickness values
        roughnesses: Dictionary mapping material names to roughness values
        
    Returns:
        Dictionary of Layers keyed by material name
    """
    layers = {}
    
    for name, material in materials.items():
        thickness = thicknesses.get(name, 0)  # Default to 0 if not specified
        roughness = roughnesses.get(name, 0)  # Default to 0 if not specified
        
        # Create the Layer with the material and parameters using the correct format
        # Layer[name] = material[name](thickness, roughness)
        Layer[name] = materials[name](thickness, roughness)
    
    return Layer





def apply_bounds(layers, sld_params=None, thickness_params=None, roughness_params=None):
    """
    Apply bounds and vary settings to the SLD, thickness, and roughness parameters of layers
    
    Args:
        layers: Dictionary of layers keyed by material name
        sld_params: Dictionary mapping material names to (lower, upper, vary) tuples for SLD
        thickness_params: Dictionary mapping material names to (lower, upper, vary) tuples for thickness
        roughness_params: Dictionary mapping material names to (lower, upper, vary) tuples for roughness
    """
    # Initialize empty dictionaries if not provided
    sld_params = sld_params or {}
    thickness_params = thickness_params or {}
    roughness_params = roughness_params or {}
    
    for name, layer in layers.items():
        # Apply SLD real bounds and vary setting if specified for this material
        if name in sld_params:
            lower, upper, vary = sld_params[name]
            layer.sld.real.setp(vary=vary, bounds=(lower, upper))
        
        # Apply thickness bounds and vary setting if specified for this material
        if name in thickness_params:
            lower, upper, vary = thickness_params[name]
            layer.thick.setp(bounds=(lower, upper), vary=vary)
        
        # Apply roughness bounds and vary setting if specified for this material
        if name in roughness_params:
            lower, upper, vary = roughness_params[name]
            layer.rough.setp(bounds=(lower, upper), vary=vary)

def create_structure(layers, order):
    """
    Create a structure by combining layers in the specified order
    
    Args:
        layers: Dictionary of layers keyed by material name
        order: List of layer names in desired order (from top to bottom)
        
    Returns:
        Structure object created by combining the layers
    """
    # Start with the first layer
    structure = layers[order[0]]
    
    # Add the remaining layers using the | operator
    for layer_name in order[1:]:
        structure = structure | layers[layer_name]
    
    return structure


def create_reflectometry_model(materials_list, layer_params, layer_order=None, ignore_layers=None, 
                          sample_name=None, energy=None):
    """
    Create a complete reflectometry model with consolidated parameters.
    
    Args:
        materials_list: List of dictionaries with 'name', 'real', and 'imag' values
        layer_params: Dictionary mapping material names to parameter dictionaries
        layer_order: List of layer names in desired order (top to bottom)
        ignore_layers: List of layer names to ignore in the model name generation
        sample_name: Name of the sample being analyzed
        energy: Energy value used in the measurement
        
    Returns:
        Tuple of (materials, layers, structure, model_name) containing all created objects and a model name
    """
    # Set default for ignore_layers if not provided
    if ignore_layers is None:
        ignore_layers = ["Si", "SiO2"]  # Default to ignoring Si and SiO2
    
    # Step 1: Create materials
    materials = {}
    for material in materials_list:
        name = material['name']
        real = material['real']
        imag = material['imag']
        materials[name] = SLD(real + imag, name=name)
    
    # Step 2: Create layers with consolidated parameters
    Layer = {}
    
    # Track which parameters are being varied for model naming
    varying_params = {
        "R": set(),  # Real SLD
        "I": set(),  # Imaginary SLD
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
            
            # Apply real SLD bounds and track variations
            if "sld_real_bounds" in params:
                lower, upper, vary = params["sld_real_bounds"]
                Layer[name].sld.real.setp(vary=vary, bounds=(lower, upper))
                if vary and name not in ignore_layers:
                    varying_params["R"].add(name)
                    has_varying_param = True
                
            # Apply imaginary SLD bounds and track variations
            if "sld_imag_bounds" in params:
                lower, upper, vary = params["sld_imag_bounds"]
                Layer[name].sld.imag.setp(vary=vary, bounds=(lower, upper))
                if vary and name not in ignore_layers:
                    varying_params["I"].add(name)
                    has_varying_param = True
                
            if "thickness_bounds" in params:
                lower, upper, vary = params["thickness_bounds"]
                Layer[name].thick.setp(bounds=(lower, upper), vary=vary)
                if vary and name not in ignore_layers:
                    varying_params["T"].add(name)
                    has_varying_param = True
                
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
    
    # Generate simplified model name
    # Count layers except those in ignore_layers
    active_layers = [layer for layer in layer_order if layer not in ignore_layers and layer != "air"]
    num_layers = len(active_layers)
    
    # Construct the model name with the new format: Sample_Energy_Layers
    if sample_name is None:
        sample_name = "Unknown"  # Default sample name if not provided
    
    if energy is None:
        energy = "Unknown"  # Default energy if not provided
    else:
        # Convert energy to string if it's a number
        energy = str(energy)
    
    model_name = f"{sample_name}_{energy}_{num_layers}Layers"
    
    return materials, Layer, structure, model_name




def create_model_and_objective(structure, data, model_name=None, scale=1.0, bkg=None, dq=1.6, 
                             vary_scale=True, vary_bkg=True, vary_dq=False,
                             scale_bounds=(0.1, 10), bkg_bounds=(0.01, 10), dq_bounds=(0.5, 2.0),
                             transform='logY'):
    """
    Create a reflectometry model and objective with specified parameters and bounds.
    
    Args:
        structure: The layer structure created by the previous function
        data: The experimental data to fit
        model_name: Optional name for the model (if None, will be auto-generated)
        scale: Initial scale factor
        bkg: Initial background value (if None, will use minimum value in data)
        dq: Initial resolution parameter
        vary_scale: Whether to vary the scale parameter
        vary_bkg: Whether to vary the background parameter
        vary_dq: Whether to vary the resolution parameter
        scale_bounds: Tuple of (lower_factor, upper_factor) to multiply scale by for bounds
        bkg_bounds: Tuple of (lower_factor, upper_factor) to multiply bkg by for bounds
        dq_bounds: Tuple of (lower, upper) absolute values for dq bounds
        transform: Transform to apply to the data ('logY', 'YX4', etc.)
        
    Returns:
        Tuple of (model, objective)
    """
    # Set background to minimum value in data if not specified
    if bkg is None:
        # Assuming data has a .y attribute containing the reflectivity values
        # This might need to be adjusted based on your data structure
        try:
            bkg = min(data.y)
        except (AttributeError, TypeError):
            # If data.y doesn't exist or isn't iterable, try alternatives
            try:
                if hasattr(data, 'data'):
                    bkg = min(data.data[:, 1])  # Assuming column 1 contains y values
                else:
                    # Default fallback
                    bkg = 0.000001
            except:
                bkg = 0.000001
        
        print(f"Auto-setting background to {bkg:.3e} (minimum value in data)")
    
    # Create the reflectometry model
    model = ReflectModel(structure, scale=scale, bkg=bkg, dq=dq, name=model_name)
    
    # Set parameter bounds and vary flags
    if vary_scale:
        lower_scale = scale * scale_bounds[0]
        upper_scale = scale * scale_bounds[1]
        model.scale.setp(bounds=(lower_scale, upper_scale), vary=True)
    
    if vary_bkg:
        lower_bkg = bkg * bkg_bounds[0]
        upper_bkg = bkg * bkg_bounds[1]
        model.bkg.setp(bounds=(lower_bkg, upper_bkg), vary=True)
    
    if vary_dq:
        model.dq.setp(bounds=dq_bounds, vary=True)
    
    # Create the objective function
    objective = Objective(model, data, transform=Transform(transform))
    
    return model, objective















########################################
#Logging Functions




def log_fitting_results(objective, model_name, results_df=None):
    """
    Log the results of model fitting to a pandas DataFrame, adding a numbered suffix
    to the model name if a model with the same name has been logged before.
    
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
            'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary'
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
                    'vary': vary
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
            'vary': None
        })
    
    # Add new rows to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
    
    # Print info about the modified model name
    if model_name != base_model_name:
        print(f"Model name modified to '{model_name}' to avoid duplication")
    
    return results_df, model_name



def print_fit_results(results_df, model_spec='recent', filter_parameters=None, show_substrate=False):
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
    """
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    if results_df is None or results_df.empty:
        print("No fitting results available.")
        return
    
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
    # First by whether they vary, then by name
    sorted_results = model_results.sort_values(by=['vary', 'parameter'], ascending=[False, True])
    
    for _, row in sorted_results.iterrows():
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
        
        # Add value if available
        if pd.notnull(row['value']):
            param_str += f"{row['value']:.6g}"
        else:
            param_str += "N/A"
        
        # Add stderr if varying and available
        if row['vary'] and pd.notnull(row['stderr']) and row['stderr'] is not None:
            param_str += f" ± {row['stderr']:.6g}"
        
        # Add bounds if available
        if pd.notnull(row['bound_low']) and pd.notnull(row['bound_high']):
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
            
            
            
######################################################################
# Fitting and Optimization

def run_fitting(objective, optimization_method='differential_evolution', 
                opt_workers=8, opt_popsize=20, burn_samples=5, 
                production_samples=5, prod_steps=1, pool=16,
                results_log=None, log_mcmc_stats=True,
                save_dir=None, save_objective=False, save_results=False,
                results_log_file=None, save_log_in_save_dir=False, 
                structure=None):
    """
    Run fitting procedure on a reflectometry model with optimization and MCMC sampling,
    and automatically log results.
    
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
    import os
    import pickle
    import datetime
    import pandas as pd
    
    # Extract model name if available
    model_name = getattr(objective.model, 'name', 'unnamed_model')
    print(f"Fitting model: {model_name}")
    
    # Generate timestamp for logs and internal tracking (but not filenames)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
            except Exception as e:
                print(f"Error loading results log: {str(e)}")
                print("Initializing new results log")
                results_log = pd.DataFrame(columns=[
                    'timestamp', 'model_name', 'goodness_of_fit', 
                    'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary'
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
            for name, param in objective.parameters.flattened():
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
                    print(f"  {name}: {stats['median']:.6g} +{stats['percentiles'][84] - stats['median']:.6g} -{stats['median'] - stats['percentiles'][16]:.6g}")
        
        except Exception as e:
            print(f"Error calculating MCMC statistics: {str(e)}")
    
    # Initialize results_log if not provided and not loaded from file
    if results_log is None:
        print("Initializing new results log")
        results_log = pd.DataFrame(columns=[
            'timestamp', 'model_name', 'goodness_of_fit', 
            'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary'
        ])
    
    # Log the results
    print(f"Logging results for model {model_name}")
    
    # Log the optimized values first
    results_log, model_name = log_fitting_results(objective, model_name, results_log)
    
    # If MCMC was performed and we want to log those stats, create a second entry
    if log_mcmc_stats and results['mcmc_stats'] is not None:
        print("Adding MCMC statistics to the log...")
        
        # Create a temporary copy of the objective to store MCMC medians
        import copy
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
                import numpy as np
                mcmc_filename = os.path.join(save_dir, f"{model_name}_mcmc_samples.npy")
                try:
                    np.save(mcmc_filename, results['mcmc_samples'])
                    print(f"Saved MCMC samples to {mcmc_filename}")
                except Exception as e:
                    print(f"Error saving MCMC samples: {str(e)}")
    
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




def load_fitting_file(filename):
    """
    Load a fitting results file saved by run_fitting, with robust handling for different file types.
    This function can handle both combined files and individual component files (objective, results).
    
    Args:
        filename: Path to the saved file
        
    Returns:
        dict: A dictionary containing the extracted components with consistent keys
    """
    import pickle
    import os
    import numpy as np
    
    try:
        print(f"Loading file from {filename}")
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Initialize the result dictionary with default values
        result = {
            'objective': None,
            'structure': None,
            'results': None,
            'mcmc_samples': None,
            'mcmc_stats': None,
            'model_name': 'unknown',
            'timestamp': None,
            'has_objective': False,
            'has_structure': False,
            'has_mcmc_samples': False,
            'has_mcmc_stats': False,
            'file_type': 'unknown'
        }
        
        # Check the type of the loaded data to determine what we have
        if isinstance(loaded_data, dict) and 'results' in loaded_data and 'structure' in loaded_data:
            # This is likely a combined file
            print("Detected combined results file format")
            result['file_type'] = 'combined'
            
            # Transfer data from the combined file
            result['structure'] = loaded_data.get('structure')
            result['objective'] = loaded_data.get('objective')
            result['model_name'] = loaded_data.get('model_name', 'unknown')
            result['timestamp'] = loaded_data.get('timestamp')
            
            # Process results if available
            if loaded_data.get('results') is not None:
                results_data = loaded_data['results']
                result['results'] = results_data
                
                # Extract MCMC data if available
                if isinstance(results_data, dict):
                    result['mcmc_samples'] = results_data.get('mcmc_samples')
                    result['mcmc_stats'] = results_data.get('mcmc_stats')
        
        elif hasattr(loaded_data, 'model') and hasattr(loaded_data, 'chisqr'):
            # This appears to be an objective file
            print("Detected objective file format")
            result['file_type'] = 'objective'
            result['objective'] = loaded_data
            
            # Try to extract model name if available
            try:
                result['model_name'] = getattr(loaded_data.model, 'name', 'unnamed_model')
            except:
                pass
            
            # Extract the filename parts to try to find the timestamp
            base_filename = os.path.basename(filename)
            parts = base_filename.split('_')
            if len(parts) > 1:
                # Try to extract timestamp if it's in the filename
                for part in parts:
                    if len(part) == 8 and part.isdigit():  # Date format YYYYMMDD
                        result['timestamp'] = part
                        break
        
        elif isinstance(loaded_data, dict) and 'mcmc_samples' in loaded_data:
            # This appears to be a results file
            print("Detected results file format")
            result['file_type'] = 'results'
            result['results'] = loaded_data
            result['mcmc_samples'] = loaded_data.get('mcmc_samples')
            result['mcmc_stats'] = loaded_data.get('mcmc_stats')
            result['timestamp'] = loaded_data.get('timestamp')
            
            # Try to extract objective if present
            if 'objective' in loaded_data and loaded_data['objective'] is not None:
                result['objective'] = loaded_data['objective']
                try:
                    result['model_name'] = getattr(loaded_data['objective'].model, 'name', 'unnamed_model')
                except:
                    pass
            
            # Try to extract structure if present
            if 'structure' in loaded_data:
                result['structure'] = loaded_data['structure']
                
        elif isinstance(loaded_data, np.ndarray):
            # This appears to be an MCMC samples file
            print("Detected MCMC samples file format")
            result['file_type'] = 'mcmc_samples'
            result['mcmc_samples'] = loaded_data
            
        else:
            # Unknown format
            print("Unknown file format - returning raw loaded data")
            return {'raw_data': loaded_data, 'file_type': 'unknown'}
        
        # Update flags based on what we found
        result['has_objective'] = result['objective'] is not None
        result['has_structure'] = result['structure'] is not None
        result['has_mcmc_samples'] = result['mcmc_samples'] is not None
        result['has_mcmc_stats'] = result['mcmc_stats'] is not None
        
        # Print a summary of what was loaded
        print(f"Successfully loaded file of type: {result['file_type']}")
        print(f"  Model name: {result['model_name']}")
        if result['timestamp']:
            print(f"  Timestamp: {result['timestamp']}")
        print(f"  Contains objective: {result['has_objective']}")
        print(f"  Contains structure: {result['has_structure']}")
        print(f"  Contains MCMC samples: {result['has_mcmc_samples']}")
        print(f"  Contains MCMC stats: {result['has_mcmc_stats']}")
        
        return result
        
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def extract_structure(loaded_data):
    """
    Extract the structure object from the loaded data.
    
    Args:
        loaded_data: The dictionary returned by load_fitting_file
        
    Returns:
        The structure object if available, None otherwise
    """
    if loaded_data is None or not loaded_data.get('has_structure', False):
        print("No structure available in the loaded data")
        return None
    
    return loaded_data.get('structure')


def extract_objective(loaded_data):
    """
    Extract the objective function from the loaded data.
    
    Args:
        loaded_data: The dictionary returned by load_fitting_file
        
    Returns:
        The objective function if available, None otherwise
    """
    if loaded_data is None or not loaded_data.get('has_objective', False):
        print("No objective available in the loaded data")
        return None
    
    return loaded_data.get('objective')


def extract_results(loaded_data):
    """
    Extract the results dictionary from the loaded data.
    
    Args:
        loaded_data: The dictionary returned by load_fitting_file
        
    Returns:
        The results dictionary if available, None otherwise
    """
    if loaded_data is None:
        print("Loaded data is None")
        return None
    
    return loaded_data.get('results')


def extract_mcmc_samples(loaded_data):
    """
    Extract the MCMC samples from the loaded data.
    
    Args:
        loaded_data: The dictionary returned by load_fitting_file
        
    Returns:
        The MCMC samples if available, None otherwise
    """
    if loaded_data is None or not loaded_data.get('has_mcmc_samples', False):
        print("No MCMC samples available in the loaded data")
        return None
    
    return loaded_data.get('mcmc_samples')


def get_parameter_summary(loaded_data):
    """
    Get a summary of parameter values and uncertainties from the MCMC statistics.
    
    Args:
        loaded_data: The dictionary returned by load_fitting_file
        
    Returns:
        A DataFrame containing parameter names, values, and uncertainties
    """
    import pandas as pd
    
    if loaded_data is None or not loaded_data.get('has_mcmc_stats', False):
        print("No MCMC statistics available in the loaded data")
        return None
    
    mcmc_stats = loaded_data.get('mcmc_stats')
    if mcmc_stats is None:
        return None
    
    # Extract parameter statistics
    stats = []
    for name, param_stats in mcmc_stats.items():
        if 'median' in param_stats and param_stats['median'] is not None:
            upper_err = param_stats['percentiles'][84] - param_stats['median']
            lower_err = param_stats['median'] - param_stats['percentiles'][16]
            
            stats.append({
                'parameter': name,
                'median': param_stats['median'],
                'mean': param_stats.get('mean'),
                'std': param_stats.get('std'),
                'upper_error': upper_err,
                'lower_error': lower_err,
                'value': param_stats.get('value'),  # Optimized value
                'stderr': param_stats.get('stderr')  # Error from optimization
            })
    
    return pd.DataFrame(stats)


def extract_structure(combined_data):
    """
    Extract the structure object from the combined data.
    
    Args:
        combined_data: The dictionary returned by load_combined_results
        
    Returns:
        The structure object if available, None otherwise
    """
    if combined_data is None or not combined_data.get('has_structure', False):
        print("No structure available in the combined data")
        return None
    
    return combined_data.get('structure')


def extract_objective(combined_data):
    """
    Extract the objective function from the combined data.
    
    Args:
        combined_data: The dictionary returned by load_combined_results
        
    Returns:
        The objective function if available, None otherwise
    """
    if combined_data is None or not combined_data.get('has_objective', False):
        print("No objective available in the combined data")
        return None
    
    return combined_data.get('objective')


def extract_results(combined_data):
    """
    Extract the results dictionary from the combined data.
    
    Args:
        combined_data: The dictionary returned by load_combined_results
        
    Returns:
        The results dictionary if available, None otherwise
    """
    if combined_data is None:
        print("Combined data is None")
        return None
    
    return combined_data.get('results')


def extract_mcmc_samples(combined_data):
    """
    Extract the MCMC samples from the combined data.
    
    Args:
        combined_data: The dictionary returned by load_combined_results
        
    Returns:
        The MCMC samples if available, None otherwise
    """
    if (combined_data is None or 
        not combined_data.get('has_mcmc_samples', False) or 
        combined_data.get('results') is None):
        print("No MCMC samples available in the combined data")
        return None
    
    return combined_data['results'].get('mcmc_samples')


def get_parameter_summary(combined_data):
    """
    Get a summary of parameter values and uncertainties from the MCMC statistics.
    
    Args:
        combined_data: The dictionary returned by load_combined_results
        
    Returns:
        A DataFrame containing parameter names, values, and uncertainties
    """
    import pandas as pd
    
    if (combined_data is None or 
        combined_data.get('results') is None or 
        combined_data['results'].get('mcmc_stats') is None):
        print("No MCMC statistics available in the combined data")
        return None
    
    # Extract parameter statistics
    stats = []
    for name, param_stats in combined_data['results']['mcmc_stats'].items():
        if 'median' in param_stats and param_stats['median'] is not None:
            upper_err = param_stats['percentiles'][84] - param_stats['median']
            lower_err = param_stats['median'] - param_stats['percentiles'][16]
            
            stats.append({
                'parameter': name,
                'median': param_stats['median'],
                'mean': param_stats['mean'],
                'std': param_stats['std'],
                'upper_error': upper_err,
                'lower_error': lower_err,
                'value': param_stats['value'],  # Optimized value
                'stderr': param_stats['stderr']  # Error from optimization
            })
    
    return pd.DataFrame(stats)


##########################################
# Optical Constant Handling

def DeltaBetatoSLD(DeltaBeta):
    #DeltaBeta needs to be 3 columns Energy,Delta,Beta
    Wavelength=EnergytoWavelength(DeltaBeta[:,0])
    SLD=np.zeros([len(DeltaBeta[:,0]),4])
    SLD[:,0]=DeltaBeta[:,0]
    SLD[:,3]=Wavelength
    SLD[:,1]=2*np.pi*DeltaBeta[:,1]/(np.power(Wavelength,2))*1000000
    SLD[:,2]=2*np.pi*DeltaBeta[:,2]/(np.power(Wavelength,2))*1000000
    return SLD

def SLDinterp(Energy,SLDarray):
    Real= np.interp(Energy, SLDarray[:,0],SLDarray[:,1])
    Imag=np.interp(Energy, SLDarray[:,0],SLDarray[:,2])*1j
    
    return(Real,Imag)


from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Physical constants
CLASSICAL_ELECTRON_RADIUS = 2.8179403227e-15  # in meters
AVOGADRO_NUMBER = 6.022140857e23  # mol^-1
HC_EV_NM = 1239.8  # hc in eV·nm

def energy_to_wavelength(energy_ev):
    """Convert energy in eV to wavelength in nm"""
    return HC_EV_NM / energy_ev


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def interpolate_sld_from_file(file_path, target_energy):
    """
    Interpolate SLD values from a data file at a specified energy.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file with columns: [energy(eV), real, imaginary]
        The real and imaginary values are already energy-corrected SLD components
    target_energy : float
        Energy in eV at which to interpolate the SLD
        
    Returns:
    --------
    tuple
        (sld_real, sld_imag) in the same units as the input file
    """
    # Load the data file
    try:
        data = pd.read_csv(file_path, sep='\s+', header=None, 
                          names=['energy_eV', 'real', 'imag'])
    except Exception as e:
        print(f"Error reading file: {e}")
        # Try alternative format (comma-separated)
        try:
            data = pd.read_csv(file_path, header=None, 
                              names=['energy_eV', 'real', 'imag'])
        except Exception as e2:
            print(f"Failed to read file as space or comma separated: {e2}")
            return None
    
    # Ensure data is sorted by energy
    data = data.sort_values('energy_eV')
    
    # Check if target energy is within the data range
    min_energy = data['energy_eV'].min()
    max_energy = data['energy_eV'].max()
    
    if target_energy < min_energy or target_energy > max_energy:
        print(f"Warning: Target energy {target_energy} eV is outside the data range [{min_energy}, {max_energy}] eV")
        print("Extrapolating values, but results may be unreliable.")
    
    # Create interpolation functions for real and imaginary components
    real_interp = interp1d(data['energy_eV'], data['real'], kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
    imag_interp = interp1d(data['energy_eV'], data['imag'], kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
    
    # Interpolate values at target energy
    sld_real = real_interp(target_energy)
    sld_imag = imag_interp(target_energy)
    
    return sld_real, sld_imag