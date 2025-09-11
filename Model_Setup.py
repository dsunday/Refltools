import numpy as np
from copy import deepcopy
import pandas as pd
from datetime import datetime
import re
import os

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
                opt_workers=8, opt_popsize=50, burn_samples=5, 
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








####Parameter Bound update Functions######



def update_parameter_bounds(objective, parameter_name, new_bounds, verbose=True):
    """
    Update the bounds of a specific parameter in a refnx objective.
    
    Args:
        objective: The refnx objective function to update
        parameter_name (str): Name of the parameter to update
        new_bounds (tuple): New bounds as (lower, upper)
        verbose (bool): Whether to print information about the update
        
    Returns:
        bool: True if parameter was found and updated, False otherwise
    """
    # Check inputs
    if not hasattr(objective, 'parameters'):
        if verbose:
            print("Error: Objective does not have a parameters attribute")
        return False
    
    if not isinstance(new_bounds, tuple) or len(new_bounds) != 2:
        if verbose:
            print("Error: new_bounds must be a tuple of (lower, upper)")
        return False
    
    lower_bound, upper_bound = new_bounds
    
    if lower_bound >= upper_bound:
        if verbose:
            print("Error: Lower bound must be less than upper bound")
        return False
    
    # Find the parameter by name
    found = False
    for param in objective.parameters.flattened():
        if param.name == parameter_name:
            # Store old bounds for reporting
            old_bounds = None
            try:
                bounds = getattr(param, 'bounds', None)
                if bounds is not None:
                    # Extract bound values
                    if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                        old_bounds = (bounds.lb, bounds.ub)
                    elif isinstance(bounds, tuple) and len(bounds) == 2:
                        old_bounds = bounds
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not get current bounds for {parameter_name}: {str(e)}")
            
            # Set the new bounds
            try:
                param.bounds = new_bounds
                found = True
                
                if verbose:
                    if old_bounds:
                        print(f"Updated bounds for {parameter_name}: {old_bounds} -> {new_bounds}")
                    else:
                        print(f"Set bounds for {parameter_name} to {new_bounds}")
            except Exception as e:
                if verbose:
                    print(f"Error updating bounds for {parameter_name}: {str(e)}")
                return False
            
            break
    
    if not found and verbose:
        print(f"Parameter '{parameter_name}' not found in the objective")
    
    return found


def update_multiple_parameter_bounds(objective, parameters_dict, verbose=True):
    """
    Update bounds for multiple parameters in a refnx objective.
    
    Args:
        objective: The refnx objective function to update
        parameters_dict (dict): Dictionary mapping parameter names to new bounds tuples
        verbose (bool): Whether to print information about the updates
        
    Returns:
        dict: Dictionary of results for each parameter (True if updated, False if not)
    """
    results = {}
    
    for param_name, new_bounds in parameters_dict.items():
        results[param_name] = update_parameter_bounds(
            objective, param_name, new_bounds, verbose=verbose
        )
    
    if verbose:
        num_updated = sum(1 for success in results.values() if success)
        print(f"Updated {num_updated} of {len(parameters_dict)} parameters")
    
    return results


def expand_sld_parameter_bounds(objective, parameter_names=None, expansion_factor=1.5, min_bound=None, 
                               filter_pattern='sld', verbose=True):
    """
    Expand bounds for SLD parameters in a refnx objective.
    
    Args:
        objective: The refnx objective function to update
        parameter_names (list, optional): List of specific parameter names to update.
                                       If None, update all parameters matching filter_pattern.
        expansion_factor (float): Factor to expand bounds by (around the current value or midpoint)
        min_bound (float, optional): Minimum value for lower bounds (useful for imaginary SLDs)
        filter_pattern (str): Pattern to match parameter names if parameter_names is None
        verbose (bool): Whether to print information about the updates
        
    Returns:
        dict: Dictionary of results for each parameter (True if updated, False if not)
    """
    import re
    
    # Check if we need to find parameters by pattern
    if parameter_names is None:
        parameter_names = []
        for param in objective.parameters.flattened():
            if re.search(filter_pattern, param.name, re.IGNORECASE):
                parameter_names.append(param.name)
    
    if verbose:
        print(f"Found {len(parameter_names)} parameters to update")
    
    # Collect bounds to update
    bounds_to_update = {}
    
    for param_name in parameter_names:
        # Find the parameter
        param = None
        for p in objective.parameters.flattened():
            if p.name == param_name:
                param = p
                break
        
        if param is None:
            if verbose:
                print(f"Parameter '{param_name}' not found")
            continue
        
        # Get current bounds
        current_bounds = None
        try:
            bounds = getattr(param, 'bounds', None)
            if bounds is not None:
                # Extract bound values
                if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                    current_bounds = (bounds.lb, bounds.ub)
                elif isinstance(bounds, tuple) and len(bounds) == 2:
                    current_bounds = bounds
        except Exception as e:
            if verbose:
                print(f"Warning: Could not get current bounds for {param_name}: {str(e)}")
            continue
        
        if current_bounds is None:
            if verbose:
                print(f"Parameter '{param_name}' has no bounds to expand")
            continue
        
        # Calculate new bounds
        current_value = param.value
        low, high = current_bounds
        
        # Calculate midpoint (either current value or middle of bounds)
        if low <= current_value <= high:
            midpoint = current_value
        else:
            midpoint = (low + high) / 2
        
        # Calculate expanded range
        current_range = high - low
        expanded_range = current_range * expansion_factor
        
        # Calculate new bounds centered on midpoint
        new_low = midpoint - expanded_range / 2
        new_high = midpoint + expanded_range / 2
        
        # Enforce minimum bound if specified
        if min_bound is not None and 'isld' in param_name.lower():
            new_low = max(new_low, min_bound)
        
        bounds_to_update[param_name] = (new_low, new_high)
    
    # Update all bounds
    return update_multiple_parameter_bounds(objective, bounds_to_update, verbose=verbose)


def get_parameter_info(objective, parameter_pattern=None):
    """
    Get information about parameters in a refnx objective.
    
    Args:
        objective: The refnx objective function to query
        parameter_pattern (str, optional): Pattern to filter parameter names
        
    Returns:
        pandas.DataFrame: DataFrame containing parameter information
    """
    import pandas as pd
    import re
    
    # Initialize lists to store parameter information
    param_info = []
    
    # Iterate through all parameters
    for param in objective.parameters.flattened():
        # Skip if pattern doesn't match
        if parameter_pattern and not re.search(parameter_pattern, param.name, re.IGNORECASE):
            continue
        
        # Get basic parameter info
        info = {
            'name': param.name,
            'value': param.value,
            'stderr': getattr(param, 'stderr', None),
            'vary': getattr(param, 'vary', None),
            'bound_low': None,
            'bound_high': None,
            'near_bounds': False
        }
        
        # Get bounds if available
        try:
            bounds = getattr(param, 'bounds', None)
            if bounds is not None:
                # Extract bound values
                if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                    info['bound_low'] = bounds.lb
                    info['bound_high'] = bounds.ub
                elif isinstance(bounds, tuple) and len(bounds) == 2:
                    info['bound_low'], info['bound_high'] = bounds
        except Exception:
            pass
        
        # Check if parameter is near bounds
        if info['bound_low'] is not None and info['bound_high'] is not None:
            bound_range = info['bound_high'] - info['bound_low']
            threshold = 0.02 * bound_range  # 2% threshold
            
            if bound_range > 0:
                near_low = abs(info['value'] - info['bound_low']) < threshold
                near_high = abs(info['bound_high'] - info['value']) < threshold
                info['near_bounds'] = near_low or near_high
        
        param_info.append(info)
    
    # Create DataFrame
    if param_info:
        df = pd.DataFrame(param_info)
        
        # Add bound percentage column
        df['bound_pct'] = None
        mask = (df['bound_high'].notnull() & df['bound_low'].notnull() & 
               (df['bound_high'] > df['bound_low']))
        
        df.loc[mask, 'bound_pct'] = ((df.loc[mask, 'value'] - df.loc[mask, 'bound_low']) / 
                                    (df.loc[mask, 'bound_high'] - df.loc[mask, 'bound_low']) * 100)
        
        return df
    else:
        return pd.DataFrame(columns=['name', 'value', 'stderr', 'vary', 
                                   'bound_low', 'bound_high', 'near_bounds', 'bound_pct'])






def run_fitting_v2(objective, method='differential_evolution', 
                   workers=-1, popsize=15, steps=1000, burn=500,
                   nthin=1, nwalkers=100,
                   results_database=None, log_mcmc_stats=True,
                   save_dir=None, save_objective=False, save_results=False,
                   structure=None, model_name=None, add_to_database=True,
                   verbose=False, energy=None):
    """
    Run fitting procedure on a reflectometry model with optimization and MCMC sampling,
    with option to add results to a pandas database. If the same model is run again,
    automatically creates a comparison plot and replaces the old result if the new one is better.
    
    Args:
        objective: The objective function to fit
        method: Optimization method to use ('differential_evolution', 'least_squares', etc.)
        workers: Number of workers for parallel optimization (-1 for all cores)
        popsize: Population size for differential evolution
        steps: Number of steps for MCMC sampling (total steps, not in thousands)
        burn: Number of burn-in steps to discard (handled post-sampling)
        nthin: Thinning factor for MCMC sampling
        nwalkers: Number of walkers for MCMC sampling
        results_database: Existing pandas DataFrame to append results to (None to create new)
        log_mcmc_stats: Whether to add MCMC statistics to the database
        save_dir: Directory to save objective and results (None to skip saving)
        save_objective: Whether to save the objective function
        save_results: Whether to save the results dictionary
        structure: The structure object associated with the objective
        model_name: Name for the model (if None, uses objective.model.name)
        add_to_database: Whether to add results to the database (True by default)
        verbose: Whether to print detailed output (False by default)
        energy: Energy value in eV for this fit (None if not energy-dependent)
        
    Returns:
        Tuple of (results_dict, updated_results_database, final_model_name)
    """
    from refnx.analysis import CurveFitter
    
    # Extract model name if not provided
    if model_name is None:
        model_name = getattr(objective.model, 'name', 'unnamed_model')
    
    if verbose:
        print(f"Fitting model: {model_name}")
    
    # Initialize results database if none provided and we want to add to it
    if results_database is None and add_to_database:
        results_database = create_empty_results_database()
    
    # Check if this model already exists in the database
    existing_fit = None
    if add_to_database and results_database is not None and not results_database.empty:
        # Check for existing fit with same model name AND energy
        if energy is not None:
            existing_models = results_database[
                (results_database['model_name'] == model_name) & 
                (results_database['energy'] == energy)
            ]
        else:
            # If no energy specified, check model name only (for non-energy-dependent fits)
            existing_models = results_database[results_database['model_name'] == model_name]
            
        if not existing_models.empty:
            existing_fit = existing_models.iloc[0]['goodness_of_fit']
            if verbose:
                energy_str = f" at {energy} eV" if energy is not None else ""
                print(f"Found existing fit for {model_name}{energy_str} with goodness of fit: {existing_fit:.6g}")
    
    # Initialize results dictionary
    results = {
        'objective': objective,
        'initial_chi_squared': objective.chisqr(),
        'optimized_parameters': None,
        'optimized_chi_squared': None,
        'mcmc_samples': None,
        'mcmc_stats': None,
        'structure': structure
    }
    
    # Create fitter
    if verbose:
        print(f"Initializing CurveFitter with {model_name} model")
    fitter = CurveFitter(objective)
    
    # Run optimization
    if verbose:
        print(f"Starting optimization using {method}...")
    if method == 'differential_evolution':
        fitter.fit(method, workers=workers, popsize=popsize)
    else:
        fitter.fit(method)
    
    # Store optimization results
    results['optimized_parameters'] = objective.parameters.pvals.copy()
    results['optimized_chi_squared'] = objective.chisqr()
    
    if verbose:
        print(f"Optimization complete. Chi-squared: {results['optimized_chi_squared']:.6g}")
    
    # Run MCMC if requested
    if steps > 0:
        if verbose:
            print(f"Starting MCMC sampling with {steps} steps...")
        
        try:
            # Run MCMC sampling
            fitter.sample(steps, nthin=nthin)
            
            # Store the chain
            results['mcmc_samples'] = fitter.chain
            
            # Check what we got and handle burn-in
            if results['mcmc_samples'] is not None and burn > 0:
                if verbose:
                    print(f"Chain shape: {results['mcmc_samples'].shape}")
                
                # Handle burn-in removal based on chain dimensions
                if results['mcmc_samples'].ndim == 3:  # (nwalkers, nsteps, nparams)
                    burn_steps = min(burn, results['mcmc_samples'].shape[1])
                    results['mcmc_samples'] = results['mcmc_samples'][:, burn_steps:, :]
                elif results['mcmc_samples'].ndim == 2:  # (nsteps, nparams)
                    burn_steps = min(burn, results['mcmc_samples'].shape[0])
                    results['mcmc_samples'] = results['mcmc_samples'][burn_steps:, :]
                
                if verbose:
                    print(f"Removed {burn_steps} burn-in steps")
                    print(f"Final chain shape: {results['mcmc_samples'].shape}")
            
            # Calculate MCMC statistics if requested
            if log_mcmc_stats and results['mcmc_samples'] is not None:
                if verbose:
                    print("Calculating MCMC statistics...")
                results['mcmc_stats'] = calculate_mcmc_statistics(results['mcmc_samples'], 
                                                                objective.parameters)
                        
        except Exception as e:
            if verbose:
                print(f"MCMC sampling failed: {str(e)}")
            results['mcmc_samples'] = None
            results['mcmc_stats'] = None
    
    # Handle database updates and comparisons
    final_model_name = model_name
    if add_to_database and results_database is not None:
        new_gof = results['optimized_chi_squared']
        
        if existing_fit is not None:
            # Compare with existing fit
            if verbose:
                print(f"\nComparing fits:")
                print(f"  Previous fit: {existing_fit:.6g}")
                print(f"  New fit: {new_gof:.6g}")
                
                # Create parameter comparison table only
                print("\nParameter Comparison Table:")
                create_parameter_comparison_table(results_database, model_name, objective, results['mcmc_stats'], energy)
            
            # Replace if new fit is better
            if new_gof < existing_fit:
                if verbose:
                    print(f"\nNew fit is better! Replacing old result in database.")
                results_database = replace_fit_in_database(results_database, model_name, 
                                                         objective, results['mcmc_stats'], energy)
            else:
                if verbose:
                    print(f"\nPrevious fit was better. Not updating database.")
                
        else:
            # First time running this model
            if verbose:
                print("Adding new fit to database...")
            results_database, final_model_name = add_fit_to_database(
                objective, model_name, results_database, results['mcmc_stats'], energy)
    
    # Save files if requested
    if save_dir is not None:
        save_fitting_files(results, save_dir, final_model_name, save_objective, 
                          save_results, structure)
    
    if verbose:
        print(f"\nFitting complete for model: {final_model_name}")
    
    return results, results_database, final_model_name


def calculate_mcmc_statistics(chain, parameters):
    """
    Calculate statistics from MCMC chain.
    """
    import numpy as np
    
    stats = {}
    
    try:
        # Flatten the chain if needed
        if chain.ndim == 3:
            flat_chain = chain.reshape(-1, chain.shape[-1])
        else:
            flat_chain = chain
        
        # Get parameter names for varying parameters only
        param_names = [p.name for p in parameters.flattened() if p.vary]
        
        for i, param_name in enumerate(param_names):
            if i < flat_chain.shape[1]:
                samples = flat_chain[:, i]
                
                if len(samples) >= 3:
                    percentiles = np.percentile(samples, [16, 50, 84])
                else:
                    percentiles = [samples[0], samples[0], samples[0]] if len(samples) > 0 else [0, 0, 0]
                
                stats[param_name] = {
                    'mean': np.mean(samples) if len(samples) > 0 else 0,
                    'median': np.median(samples) if len(samples) > 0 else 0,
                    'std': np.std(samples) if len(samples) > 1 else 0,
                    'percentiles': percentiles
                }
    
    except Exception as e:
        print(f"Error calculating MCMC statistics: {str(e)}")
        
    return stats


def create_parameter_comparison_table(results_database, model_name, new_objective, new_mcmc_stats=None, energy=None):
    """
    Create a table comparing parameters between old and new fits.
    Only includes parameters that are being fit (vary=True).
    Compares fits with the same model name and energy.
    """
    # Define color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    # Get old parameters from database (only varying parameters)
    # Filter by both model name and energy
    if energy is not None:
        old_params = results_database[
            (results_database['model_name'] == model_name) & 
            (results_database['energy'] == energy) &
            (results_database['vary'] == True)
        ].copy()
    else:
        old_params = results_database[
            (results_database['model_name'] == model_name) & 
            (results_database['vary'] == True)
        ].copy()
    
    # Get new parameters (only varying parameters)
    new_params = []
    for param in new_objective.parameters.flattened():
        if not getattr(param, 'vary', False):
            continue  # Skip fixed parameters
            
        stderr = getattr(param, 'stderr', None)
        # Use MCMC stderr if available
        if new_mcmc_stats and param.name in new_mcmc_stats:
            mcmc_stderr = new_mcmc_stats[param.name].get('std', None)
            if mcmc_stderr is not None:
                stderr = mcmc_stderr
        
        # Get bounds
        try:
            bounds = getattr(param, 'bounds', None)
            if bounds is not None and len(bounds) == 2:
                bound_low, bound_high = bounds
            else:
                bound_low, bound_high = None, None
        except:
            bound_low, bound_high = None, None
        
        new_params.append({
            'parameter': param.name,
            'value': param.value,
            'stderr': stderr,
            'vary': True,
            'bound_low': bound_low,
            'bound_high': bound_high
        })
    
    # Create comparison DataFrame
    comparison_data = []
    for new_param in new_params:
        param_name = new_param['parameter']
        old_match = old_params[old_params['parameter'] == param_name]
        
        if not old_match.empty:
            old_value = old_match.iloc[0]['value']
            old_stderr = old_match.iloc[0]['stderr']
            
            # Calculate relative change
            if old_value != 0:
                rel_change = ((new_param['value'] - old_value) / old_value) * 100
            else:
                rel_change = None
            
            # Check if new parameter is near bounds (within 1%)
            near_bounds = False
            if (new_param['bound_low'] is not None and 
                new_param['bound_high'] is not None):
                bound_range = new_param['bound_high'] - new_param['bound_low']
                threshold = 0.01 * bound_range  # 1% of range
                
                if (abs(new_param['value'] - new_param['bound_low']) < threshold or 
                    abs(new_param['bound_high'] - new_param['value']) < threshold):
                    near_bounds = True
            
            # Format the new value with color coding
            if near_bounds:
                new_value_str = f"{RED}{new_param['value']:.6g}{RESET}"
                new_error_str = f"{RED}{new_param['stderr']:.6g}{RESET}" if new_param['stderr'] is not None else f"{RED}N/A{RESET}"
            else:
                new_value_str = f"{GREEN}{new_param['value']:.6g}{RESET}"
                new_error_str = f"{GREEN}{new_param['stderr']:.6g}{RESET}" if new_param['stderr'] is not None else f"{GREEN}N/A{RESET}"
            
            comparison_data.append({
                'Parameter': param_name,
                'Old Value': f"{old_value:.6g}" if old_value is not None else "N/A",
                'New Value': new_value_str,
                'Change (%)': f"{rel_change:.2f}" if rel_change is not None else "N/A"
            })
    
    # Display table
    if comparison_data:
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        print("Parameter Comparison (Fitted Parameters Only):")
        print("Legend: " + GREEN + "Green = Normal" + RESET + " | " + RED + "Red = Near bounds (±1%)" + RESET)
        print(df.to_string(index=False))
    else:
        print("No varying parameters to compare.")


def get_sld_profile(structure):
    """
    Get SLD profile from a structure object.
    Returns depth and SLD arrays for both real and imaginary components.
    """
    try:
        z, sld = structure.sld_profile()
        if hasattr(sld, 'real') and hasattr(sld, 'imag'):
            return z, sld.real, z, sld.imag
        else:
            # If sld is just real values
            return z, sld, z, np.zeros_like(sld)
    except:
        # Fallback implementation
        try:
            z = np.linspace(-10, 300, 1000)
            sld_profile = structure.sld_profile(z)
            if hasattr(sld_profile, 'real'):
                return z, sld_profile.real, z, sld_profile.imag
            else:
                return z, sld_profile, z, np.zeros_like(sld_profile)
        except:
            # Last resort fallback
            z = np.linspace(0, 300, 1000)
            return z, np.zeros_like(z), z, np.zeros_like(z)


def replace_fit_in_database(results_database, model_name, new_objective, new_mcmc_stats=None, energy=None):
    """
    Replace an existing fit in the database with a new better fit.
    
    Args:
        results_database: DataFrame containing existing results
        model_name: Name of the model to replace
        new_objective: New objective function
        new_mcmc_stats: New MCMC statistics
        energy: Energy value for this fit
        
    Returns:
        Updated database with the old fit removed and new fit added
    """
    # Remove old entries with matching model name and energy
    if energy is not None:
        results_database = results_database[
            ~((results_database['model_name'] == model_name) & 
              (results_database['energy'] == energy))
        ]
    else:
        # If no energy specified, remove by model name only
        results_database = results_database[results_database['model_name'] != model_name]
    
    # Add new entries
    results_database, _ = add_fit_to_database(new_objective, model_name, results_database, new_mcmc_stats, energy)
    
    return results_database


def create_empty_results_database():
    """
    Create an empty pandas DataFrame with the standard columns for storing fitting results.
    
    Returns:
        Empty pandas DataFrame with predefined columns
    """
    columns = [
        'model_name', 'goodness_of_fit', 'energy',
        'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary'
    ]
    
    return pd.DataFrame(columns=columns)


def add_fit_to_database(objective, model_name, results_database, mcmc_stats=None, energy=None):
    """
    Add the results of a single fit to the pandas database.
    
    Args:
        objective: The objective function after fitting
        model_name: Name of the model used for fitting
        results_database: Existing DataFrame to append to
        mcmc_stats: Optional MCMC statistics dictionary
        energy: Energy value in eV for this fit
        
    Returns:
        Tuple of (updated_database, final_model_name)
    """
    
    # Extract the goodness of fit (energy from chi-squared)
    try:
        gof = objective.chisqr()
    except Exception as e:
        print(f"Error getting chisqr: {str(e)}")
        gof = None
    
    # Extract all parameters from the model
    model = objective.model
    rows = []
    
    try:
        for param in model.parameters.flattened():
            try:
                # Get parameter name and value
                param_name = param.name
                value = param.value
                stderr = getattr(param, 'stderr', None)
                vary = getattr(param, 'vary', False)
                
                # Get bounds if available
                try:
                    bounds = getattr(param, 'bounds', None)
                    if bounds is not None:
                        # Handle different bounds formats
                        if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                            # Bounds object with lb/ub attributes
                            bound_low, bound_high = bounds.lb, bounds.ub
                        elif isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                            # Tuple/list format
                            bound_low, bound_high = bounds
                        elif hasattr(bounds, '__iter__') and len(list(bounds)) == 2:
                            # Other iterable format
                            bound_vals = list(bounds)
                            bound_low, bound_high = bound_vals[0], bound_vals[1]
                        else:
                            bound_low, bound_high = None, None
                    else:
                        bound_low, bound_high = None, None
                except Exception as e:
                    print(f"Error extracting bounds for {param_name}: {str(e)}")
                    bound_low, bound_high = None, None
                
                # Use MCMC stderr if available and better than optimization stderr
                if mcmc_stats and param_name in mcmc_stats:
                    mcmc_stderr = mcmc_stats[param_name].get('std', None)
                    if mcmc_stderr is not None and (stderr is None or mcmc_stderr > 0):
                        stderr = mcmc_stderr
                
                # Create row for this parameter
                row = {
                    'model_name': model_name,
                    'goodness_of_fit': gof,
                    'energy': energy,
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
            'model_name': model_name,
            'goodness_of_fit': gof,
            'energy': energy,
            'parameter': 'unknown',
            'value': None,
            'stderr': None,
            'bound_low': None,
            'bound_high': None,
            'vary': None
        })
    
    # Add new rows to the DataFrame
    new_rows_df = pd.DataFrame(rows)
    results_database = pd.concat([results_database, new_rows_df], ignore_index=True)
    
    return results_database, model_name


def calculate_mcmc_statistics(chain, parameters):
    """
    Calculate statistics from MCMC chain.
    
    Args:
        chain: MCMC chain array
        parameters: Parameters object from the fit
        
    Returns:
        Dictionary of statistics for each parameter
    """
    import numpy as np
    
    stats = {}
    
    try:
        # Flatten the chain if needed (remove burn-in and walker dimensions)
        if chain.ndim == 3:
            # Shape is (nwalkers, nsteps, nparams)
            flat_chain = chain.reshape(-1, chain.shape[-1])
        else:
            flat_chain = chain
        
        # Get parameter names
        param_names = [p.name for p in parameters.flattened() if p.vary]
        
        for i, param_name in enumerate(param_names):
            if i < flat_chain.shape[1]:
                samples = flat_chain[:, i]
                
                stats[param_name] = {
                    'mean': np.mean(samples),
                    'median': np.median(samples),
                    'std': np.std(samples),
                    'percentiles': np.percentile(samples, [16, 50, 84])
                }
    
    except Exception as e:
        print(f"Error calculating MCMC statistics: {str(e)}")
        
    return stats


def save_fitting_files(results, save_dir, model_name, save_objective, save_results, 
                      structure):
    """
    Save fitting results to files.
    
    Args:
        results: Results dictionary
        save_dir: Directory to save files
        model_name: Name of the model
        save_objective: Whether to save objective
        save_results: Whether to save results
        structure: Structure object
    """
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Save objective if requested
    if save_objective:
        objective_filename = os.path.join(save_dir, f"{model_name}_objective.pkl")
        try:
            with open(objective_filename, 'wb') as f:
                pickle.dump(results['objective'], f)
            print(f"Saved objective to {objective_filename}")
        except Exception as e:
            print(f"Error saving objective: {str(e)}")
    
    # Save results if requested
    if save_results:
        if structure is not None:
            # Save combined results and structure
            combined_filename = os.path.join(save_dir, f"{model_name}_results_structure.pkl")
            try:
                # Create a copy of results without the objective to avoid circular references
                save_results_copy = results.copy()
                save_results_copy['objective'] = None
                
                combined_data = {
                    'results': save_results_copy,
                    'structure': structure,
                    'objective': results['objective'] if save_objective else None,
                    'model_name': model_name
                }
                with open(combined_filename, 'wb') as f:
                    pickle.dump(combined_data, f)
                print(f"Saved combined results and structure to {combined_filename}")
            except Exception as e:
                print(f"Error saving combined data: {str(e)}")
        
        # Additionally, save MCMC samples as numpy array if they exist
        if results['mcmc_samples'] is not None:
            import numpy as np
            mcmc_filename = os.path.join(save_dir, f"{model_name}_mcmc_samples.npy")
            try:
                np.save(mcmc_filename, results['mcmc_samples'])
                print(f"Saved MCMC samples to {mcmc_filename}")
            except Exception as e:
                print(f"Error saving MCMC samples: {str(e)}")


def save_database_to_file(results_database, filename):
    """
    Save the results database to a CSV file.
    
    Args:
        results_database: pandas DataFrame containing the results
        filename: Path to save the CSV file
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        results_database.to_csv(filename, index=False)
        print(f"Saved results database to {filename}")
    except Exception as e:
        print(f"Error saving database to file: {str(e)}")


def load_database_from_file(filename):
    """
    Load a results database from a CSV file.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        pandas DataFrame containing the results, or empty DataFrame if file doesn't exist
    """
    try:
        if os.path.exists(filename):
            database = pd.read_csv(filename)
            print(f"Loaded results database from {filename}")
            return database
        else:
            print(f"File {filename} not found. Creating new database.")
            return create_empty_results_database()
    except Exception as e:
        print(f"Error loading database from file: {str(e)}")
        return create_empty_results_database()


def query_database(results_database, model_name=None, parameter=None, best_fit=False):
    """
    Query the results database for specific information.
    
    Args:
        results_database: pandas DataFrame containing the results
        model_name: Specific model name to filter by (None for all)
        parameter: Specific parameter name to filter by (None for all)
        best_fit: If True, return only the best fit (lowest goodness_of_fit)
        
    Returns:
        Filtered pandas DataFrame
    """
    filtered_df = results_database.copy()
    
    # Filter by model name
    if model_name is not None:
        filtered_df = filtered_df[filtered_df['model_name'] == model_name]
    
    # Filter by parameter
    if parameter is not None:
        filtered_df = filtered_df[filtered_df['parameter'] == parameter]
    
    # Get best fit if requested
    if best_fit and not filtered_df.empty:
        # Find the model with the lowest goodness of fit
        best_gof = filtered_df['goodness_of_fit'].min()
        best_models = filtered_df[filtered_df['goodness_of_fit'] == best_gof]['model_name'].unique()
        if len(best_models) > 0:
            filtered_df = filtered_df[filtered_df['model_name'] == best_models[0]]
    
    return filtered_df


def plot_energy_vs_gof(results_database, model_names=None, figsize=(10, 6), 
                       xlim=None, ylim=None, save_path=None):
    """
    Plot Energy vs Goodness of Fit for models.
    
    Args:
        results_database: pandas DataFrame containing the results
        model_names: List of specific model names to plot (None for all models)
        figsize: Figure size as (width, height)
        xlim: X-axis limits as (min, max) tuple (None for auto)
        ylim: Y-axis limits as (min, max) tuple (None for auto)
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Filter data
    if model_names is not None:
        filtered_db = results_database[results_database['model_name'].isin(model_names)]
    else:
        filtered_db = results_database.copy()
    
    # Remove rows with no energy values
    filtered_db = filtered_db.dropna(subset=['energy'])
    
    if filtered_db.empty:
        print("No data with energy values found.")
        return None, None
    
    # Get unique model names
    unique_models = filtered_db['model_name'].unique()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extended color and marker cycles for better differentiation
    colors = plt.cm.tab20.colors  # 20 colors instead of 10
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', '+', 'd', '8', 'P', 'H']
    
    # Plot each model with unique color/marker combination
    for i, model in enumerate(unique_models):
        model_data = filtered_db[filtered_db['model_name'] == model]
        # Get unique energy-GOF pairs for this model
        gof_data = model_data.groupby('energy')['goodness_of_fit'].first()
        
        ax.scatter(gof_data.index, gof_data.values, 
                  color=colors[i % len(colors)], 
                  marker=markers[i % len(markers)],
                  s=80, alpha=0.7, label=model, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Goodness of Fit (χ²)')
    ax.set_title('Energy vs Goodness of Fit')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if len(unique_models) > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax



def plot_energy_vs_sld(results_database, materials=None, model_names=None, 
                       exclude_materials=['Si', 'SiO2', 'air'], plot_type='both',
                       figsize=(12, 8), xlim=None, ylim=None, materials_bounds=None, save_path=None):
    """
    Plot Energy vs SLD (real and/or imaginary) for materials.
    
    Args:
        results_database: pandas DataFrame containing the results
        materials: List of materials to plot (None for all except excluded)
        model_names: List of model names to include (None for all)
        exclude_materials: List of materials to exclude by default
        plot_type: 'real', 'imag', or 'both' to plot real SLD, imaginary SLD, or both
        figsize: Figure size as (width, height)
        xlim: X-axis limits as (min, max) tuple (None for auto)
        ylim: Y-axis limits as (min, max) tuple (None for auto)
        materials_bounds: List of materials to show bounds for (None for no bounds)
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Filter by model names if specified
    if model_names is not None:
        filtered_db = results_database[results_database['model_name'].isin(model_names)]
    else:
        filtered_db = results_database.copy()
    
    # Remove rows with no energy values
    filtered_db = filtered_db.dropna(subset=['energy'])
    
    # Filter SLD parameters
    sld_params = filtered_db[
        (filtered_db['parameter'].str.contains('sld', case=False)) &
        (~filtered_db['parameter'].str.contains('isld', case=False))
    ]
    isld_params = filtered_db[filtered_db['parameter'].str.contains('isld', case=False)]
    
    # Get material names from parameters
    def extract_material_name(param_name):
        # Handle different parameter naming conventions
        if ' - ' in param_name:
            return param_name.split(' - ')[0]
        elif '_' in param_name:
            return param_name.split('_')[0]
        else:
            return param_name.split()[0] if ' ' in param_name else param_name
    
    # Filter materials
    if materials is None:
        all_materials = set()
        for param in sld_params['parameter']:
            mat = extract_material_name(param)
            if mat not in exclude_materials:
                all_materials.add(mat)
        for param in isld_params['parameter']:
            mat = extract_material_name(param)
            if mat not in exclude_materials:
                all_materials.add(mat)
        materials = sorted(list(all_materials))
    
    if not materials:
        print("No materials found after filtering.")
        return None, None
    
    # Create subplots
    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        axes = [ax1, ax2]
        titles = ['Real SLD vs Energy', 'Imaginary SLD vs Energy']
        param_types = [sld_params, isld_params]
    elif plot_type == 'real':
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        titles = ['Real SLD vs Energy']
        param_types = [sld_params]
    else:  # imag
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        titles = ['Imaginary SLD vs Energy']
        param_types = [isld_params]
    
    # Extended color and marker cycles for better differentiation
    colors = plt.cm.tab20.colors  # 20 colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', '+', 'd', '8', 'P', 'H']
    
    unique_models = filtered_db['model_name'].unique()
    
    # Plot each parameter type
    for ax_idx, (ax, title, params) in enumerate(zip(axes, titles, param_types)):
        
        # Plot data points and error bars for bounds
        combo_idx = 0
        for material in materials:
            # Filter parameters for this material
            mat_params = params[params['parameter'].str.contains(material, case=False)]
            
            for model in unique_models:
                model_mat_params = mat_params[mat_params['model_name'] == model]
                
                if not model_mat_params.empty:
                    energies = model_mat_params['energy'].values
                    values = model_mat_params['value'].values
                    
                    # Check if we should show bounds for this material
                    show_bounds = materials_bounds is not None and material in materials_bounds
                    
                    if show_bounds:
                        # Calculate error bars from bounds
                        lower_errors = []
                        upper_errors = []
                        
                        for i, (energy, value) in enumerate(zip(energies, values)):
                            energy_row = model_mat_params[model_mat_params['energy'] == energy].iloc[0]
                            bound_low = energy_row['bound_low']
                            bound_high = energy_row['bound_high']
                            
                            if pd.notnull(bound_low) and pd.notnull(bound_high):
                                # Calculate asymmetric error bars
                                lower_error = value - bound_low  # Distance from value to lower bound
                                upper_error = bound_high - value  # Distance from value to upper bound
                                lower_errors.append(max(0, lower_error))  # Ensure non-negative
                                upper_errors.append(max(0, upper_error))
                            else:
                                lower_errors.append(0)
                                upper_errors.append(0)
                        
                        # Plot with error bars showing bounds
                        ax.errorbar(energies, values,
                                   yerr=[lower_errors, upper_errors],
                                   fmt='none',  # No marker for error bars
                                   ecolor=colors[combo_idx % len(colors)],
                                   elinewidth=2,
                                   capsize=8,
                                   capthick=2,
                                   alpha=0.6,
                                   label=f"{material} bounds" if combo_idx == 0 else "",
                                   zorder=2)
                    
                    # Plot the actual data points on top
                    ax.scatter(energies, values,
                              color=colors[combo_idx % len(colors)],
                              marker=markers[combo_idx % len(markers)],
                              s=80, alpha=0.9, edgecolors='black', linewidth=0.5,
                              label=f"{material} ({model})" if len(unique_models) > 1 else material,
                              zorder=3)  # Ensure points appear on top
                    combo_idx += 1
        
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('SLD (Å⁻²)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Set axis limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, axes if plot_type == 'both' else ax


def plot_energy_vs_thickness(results_database, materials=None, model_names=None,
                            exclude_materials=['Si', 'SiO2', 'air'],
                            figsize=(10, 6), xlim=None, ylim=None, materials_bounds=None, save_path=None):
    """
    Plot Energy vs Thickness for materials.
    
    Args:
        results_database: pandas DataFrame containing the results
        materials: List of materials to plot (None for all except excluded)
        model_names: List of model names to include (None for all)
        exclude_materials: List of materials to exclude by default
        figsize: Figure size as (width, height)
        xlim: X-axis limits as (min, max) tuple (None for auto)
        ylim: Y-axis limits as (min, max) tuple (None for auto)
        materials_bounds: List of materials to show bounds for (None for no bounds)
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Filter by model names if specified
    if model_names is not None:
        filtered_db = results_database[results_database['model_name'].isin(model_names)]
    else:
        filtered_db = results_database.copy()
    
    # Remove rows with no energy values
    filtered_db = filtered_db.dropna(subset=['energy'])
    
    # Filter thickness parameters
    thickness_params = filtered_db[filtered_db['parameter'].str.contains('thick', case=False)]
    
    # Get material names from parameters
    def extract_material_name(param_name):
        if ' - ' in param_name:
            return param_name.split(' - ')[0]
        elif '_' in param_name:
            return param_name.split('_')[0]
        else:
            return param_name.split()[0] if ' ' in param_name else param_name
    
    # Filter materials
    if materials is None:
        all_materials = set()
        for param in thickness_params['parameter']:
            mat = extract_material_name(param)
            if mat not in exclude_materials:
                all_materials.add(mat)
        materials = sorted(list(all_materials))
    
    if not materials:
        print("No materials found after filtering.")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extended color and marker cycles for better differentiation
    colors = plt.cm.tab20.colors  # 20 colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', '+', 'd', '8', 'P', 'H']
    
    unique_models = filtered_db['model_name'].unique()
    
    # Plot data points and error bars for bounds
    combo_idx = 0
    for material in materials:
        # Filter parameters for this material
        mat_params = thickness_params[thickness_params['parameter'].str.contains(material, case=False)]
        
        for model in unique_models:
            model_mat_params = mat_params[mat_params['model_name'] == model]
            
            if not model_mat_params.empty:
                energies = model_mat_params['energy'].values
                values = model_mat_params['value'].values
                
                # Check if we should show bounds for this material
                show_bounds = materials_bounds is not None and material in materials_bounds
                
                if show_bounds:
                    # Calculate error bars from bounds
                    lower_errors = []
                    upper_errors = []
                    
                    for i, (energy, value) in enumerate(zip(energies, values)):
                        energy_row = model_mat_params[model_mat_params['energy'] == energy].iloc[0]
                        bound_low = energy_row['bound_low']
                        bound_high = energy_row['bound_high']
                        
                        if pd.notnull(bound_low) and pd.notnull(bound_high):
                            # Calculate asymmetric error bars
                            lower_error = value - bound_low  # Distance from value to lower bound
                            upper_error = bound_high - value  # Distance from value to upper bound
                            lower_errors.append(max(0, lower_error))  # Ensure non-negative
                            upper_errors.append(max(0, upper_error))
                        else:
                            lower_errors.append(0)
                            upper_errors.append(0)
                    
                    # Plot with error bars showing bounds
                    ax.errorbar(energies, values,
                               yerr=[lower_errors, upper_errors],
                               fmt='none',  # No marker for error bars
                               ecolor=colors[combo_idx % len(colors)],
                               elinewidth=2,
                               capsize=8,
                               capthick=2,
                               alpha=0.6,
                               label=f"{material} bounds" if combo_idx == 0 else "",
                               zorder=2)
                
                # Plot the actual data points on top
                ax.scatter(energies, values,
                          color=colors[combo_idx % len(colors)],
                          marker=markers[combo_idx % len(markers)],
                          s=80, alpha=0.9, edgecolors='black', linewidth=0.5,
                          label=f"{material} ({model})" if len(unique_models) > 1 else material,
                          zorder=3)  # Ensure points appear on top
                combo_idx += 1
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Thickness (Å)')
    ax.set_title('Energy vs Thickness')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def plot_energy_vs_roughness(results_database, materials=None, model_names=None,
                            exclude_materials=['Si', 'SiO2', 'air'],
                            figsize=(10, 6), xlim=None, ylim=None, materials_bounds=None, save_path=None):
    """
    Plot Energy vs Roughness for materials.
    
    Args:
        results_database: pandas DataFrame containing the results
        materials: List of materials to plot (None for all except excluded)
        model_names: List of model names to include (None for all)
        exclude_materials: List of materials to exclude by default
        figsize: Figure size as (width, height)
        xlim: X-axis limits as (min, max) tuple (None for auto)
        ylim: Y-axis limits as (min, max) tuple (None for auto)
        materials_bounds: List of materials to show bounds for (None for no bounds)
        save_path: Path to save the figure (None to skip saving)
        
    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Filter by model names if specified
    if model_names is not None:
        filtered_db = results_database[results_database['model_name'].isin(model_names)]
    else:
        filtered_db = results_database.copy()
    
    # Remove rows with no energy values
    filtered_db = filtered_db.dropna(subset=['energy'])
    
    # Filter roughness parameters
    roughness_params = filtered_db[filtered_db['parameter'].str.contains('rough', case=False)]
    
    # Get material names from parameters
    def extract_material_name(param_name):
        if ' - ' in param_name:
            return param_name.split(' - ')[0]
        elif '_' in param_name:
            return param_name.split('_')[0]
        else:
            return param_name.split()[0] if ' ' in param_name else param_name
    
    # Filter materials
    if materials is None:
        all_materials = set()
        for param in roughness_params['parameter']:
            mat = extract_material_name(param)
            if mat not in exclude_materials:
                all_materials.add(mat)
        materials = sorted(list(all_materials))
    
    if not materials:
        print("No materials found after filtering.")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extended color and marker cycles for better differentiation
    colors = plt.cm.tab20.colors  # 20 colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', '+', 'd', '8', 'P', 'H']
    
    unique_models = filtered_db['model_name'].unique()
    
    # Plot data points and error bars for bounds
    combo_idx = 0
    for material in materials:
        # Filter parameters for this material
        mat_params = roughness_params[roughness_params['parameter'].str.contains(material, case=False)]
        
        for model in unique_models:
            model_mat_params = mat_params[mat_params['model_name'] == model]
            
            if not model_mat_params.empty:
                energies = model_mat_params['energy'].values
                values = model_mat_params['value'].values
                
                # Check if we should show bounds for this material
                show_bounds = materials_bounds is not None and material in materials_bounds
                
                if show_bounds:
                    # Calculate error bars from bounds
                    lower_errors = []
                    upper_errors = []
                    
                    for i, (energy, value) in enumerate(zip(energies, values)):
                        energy_row = model_mat_params[model_mat_params['energy'] == energy].iloc[0]
                        bound_low = energy_row['bound_low']
                        bound_high = energy_row['bound_high']
                        
                        if pd.notnull(bound_low) and pd.notnull(bound_high):
                            # Calculate asymmetric error bars
                            lower_error = value - bound_low  # Distance from value to lower bound
                            upper_error = bound_high - value  # Distance from value to upper bound
                            lower_errors.append(max(0, lower_error))  # Ensure non-negative
                            upper_errors.append(max(0, upper_error))
                        else:
                            lower_errors.append(0)
                            upper_errors.append(0)
                    
                    # Plot with error bars showing bounds
                    ax.errorbar(energies, values,
                               yerr=[lower_errors, upper_errors],
                               fmt='none',  # No marker for error bars
                               ecolor=colors[combo_idx % len(colors)],
                               elinewidth=2,
                               capsize=8,
                               capthick=2,
                               alpha=0.6,
                               label=f"{material} bounds" if combo_idx == 0 else "",
                               zorder=2)
                
                # Plot the actual data points on top
                ax.scatter(energies, values,
                          color=colors[combo_idx % len(colors)],
                          marker=markers[combo_idx % len(markers)],
                          s=80, alpha=0.9, edgecolors='black', linewidth=0.5,
                          label=f"{material} ({model})" if len(unique_models) > 1 else material,
                          zorder=3)  # Ensure points appear on top
                combo_idx += 1
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Roughness (Å)')
    ax.set_title('Energy vs Roughness')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def generate_sld_array_from_material(formula, density, energy_list, probe="x-ray"):
    """
    Generate an SLD array from MaterialSLD for a given energy range.
    
    Args:
        formula: Chemical formula string
        density: Material density in g/cm³
        energy_list: List of energy values in eV
        probe: "x-ray" or "neutron"
    
    Returns:
        numpy array with columns [Energy, Real SLD, Imag SLD]
    """
    import numpy as np
    from refnx.reflect import MaterialSLD
    
    sld_data = []
    
    for energy_ev in energy_list:
        # Convert energy to wavelength (Å)
        # For X-rays: E(eV) * wavelength(Å) = 12398 (hc constant in eV·Å)
        wavelength = 12398.0 / energy_ev
        
        # Create MaterialSLD object for this energy
        material = MaterialSLD(formula, density=density, probe=probe, wavelength=wavelength)
        
        # Extract SLD values using complex() method
        sld_complex = complex(material)
        
        real_sld = sld_complex.real
        imag_sld = sld_complex.imag
        
        sld_data.append([energy_ev, real_sld, imag_sld])
    
    return np.array(sld_data)

def update_objective_with_plotting(objectives_dict, energy, material, updates, plot=False, 
                                  figsize=(12, 8), profile_shift=-20, save_plot=False, 
                                  plot_filename=None, structures_dict=None):
    """
    Update parameter values and bounds for a specific energy and material in objectives dictionary,
    with optional plotting of simulated vs experimental data.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        energy (float): Energy value to update
        material (str): Material name (e.g., "Resist", "PS")
        updates (dict): Dictionary of parameter updates with the following possible keys:
            - "thickness": new thickness value
            - "roughness": new roughness value  
            - "sld_real": new real SLD value
            - "sld_imag": new imaginary SLD value
            - "thickness_bounds": (lower, upper, vary) tuple for thickness bounds
            - "roughness_bounds": (lower, upper, vary) tuple for roughness bounds
            - "sld_real_bounds": (lower, upper, vary) tuple for real SLD bounds
            - "sld_imag_bounds": (lower, upper, vary) tuple for imaginary SLD bounds
        plot (bool): Whether to create plots showing experimental vs simulated data
        figsize (tuple): Figure size for plots as (width, height)
        profile_shift (float): Depth shift for SLD profile plots
        save_plot (bool): Whether to save the plot to file
        plot_filename (str, optional): Filename for saved plot. If None, auto-generates name.
        structures_dict (dict, optional): Dictionary mapping energy values to Structure objects.
                                        Required if plot=True and SLD profiles are desired.
            
    Returns:
        dict: Updated objectives dictionary
        
    Example:
        updated_objectives = update_objective_with_plotting(
            objectives_dict=my_objectives,
            energy=270.0,
            material="Resist",
            updates={
                "thickness": 180.0,
                "roughness": 5.0,
                "sld_real_bounds": (2.0, 8.0, True),
                "sld_imag_bounds": (0.0, 6.0, True),
                "thickness_bounds": (175.0, 185.0, True),
                "roughness_bounds": (3.0, 7.0, True)
            },
            plot=True,
            structures_dict=my_structures
        )
    """
    import copy
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a working copy
    updated_objectives = copy.deepcopy(objectives_dict)
    
    # Check if energy exists
    if energy not in updated_objectives:
        print(f"Warning: Energy {energy} not found in objectives")
        return updated_objectives
    
    objective = updated_objectives[energy]
    
    # Track updates made
    updates_made = []
    
    # Iterate through all parameters to find matches
    for param in objective.parameters.flattened():
        param_name = param.name.lower()
        
        # Check if this parameter belongs to the specified material
        if f"{material.lower()} - " not in param_name:
            continue
            
        # Determine parameter type and update accordingly
        param_updated = False
        
        # Handle thickness parameters
        if "thick" in param_name:
            if "thickness" in updates:
                old_value = param.value
                param.value = updates["thickness"]
                updates_made.append(f"thickness value: {old_value} -> {param.value}")
                param_updated = True
                
            if "thickness_bounds" in updates:
                lower, upper, vary = updates["thickness_bounds"]
                old_bounds = getattr(param, 'bounds', None)
                old_vary = getattr(param, 'vary', None)
                param.bounds = (lower, upper)
                param.vary = vary
                updates_made.append(f"thickness bounds: {old_bounds} -> {param.bounds}, vary: {old_vary} -> {param.vary}")
                param_updated = True
        
        # Handle roughness parameters
        elif "rough" in param_name:
            if "roughness" in updates:
                old_value = param.value
                param.value = updates["roughness"]
                updates_made.append(f"roughness value: {old_value} -> {param.value}")
                param_updated = True
                
            if "roughness_bounds" in updates:
                lower, upper, vary = updates["roughness_bounds"]
                old_bounds = getattr(param, 'bounds', None)
                old_vary = getattr(param, 'vary', None)
                param.bounds = (lower, upper)
                param.vary = vary
                updates_made.append(f"roughness bounds: {old_bounds} -> {param.bounds}, vary: {old_vary} -> {param.vary}")
                param_updated = True
        
        # Handle real SLD parameters
        elif "sld" in param_name and "isld" not in param_name:
            if "sld_real" in updates:
                old_value = param.value
                param.value = updates["sld_real"]
                updates_made.append(f"real SLD value: {old_value} -> {param.value}")
                param_updated = True
                
            if "sld_real_bounds" in updates:
                lower, upper, vary = updates["sld_real_bounds"]
                old_bounds = getattr(param, 'bounds', None)
                old_vary = getattr(param, 'vary', None)
                param.bounds = (lower, upper)
                param.vary = vary
                updates_made.append(f"real SLD bounds: {old_bounds} -> {param.bounds}, vary: {old_vary} -> {param.vary}")
                param_updated = True
        
        # Handle imaginary SLD parameters
        elif "isld" in param_name:
            if "sld_imag" in updates:
                old_value = param.value
                param.value = updates["sld_imag"]
                updates_made.append(f"imaginary SLD value: {old_value} -> {param.value}")
                param_updated = True
                
            if "sld_imag_bounds" in updates:
                lower, upper, vary = updates["sld_imag_bounds"]
                old_bounds = getattr(param, 'bounds', None)
                old_vary = getattr(param, 'vary', None)
                param.bounds = (lower, upper)
                param.vary = vary
                updates_made.append(f"imaginary SLD bounds: {old_bounds} -> {param.bounds}, vary: {old_vary} -> {param.vary}")
                param_updated = True
    
    # Print summary of updates
    if updates_made:
        print(f"Updated {material} parameters at {energy} eV:")
        for update in updates_made:
            print(f"  - {update}")
    else:
        print(f"No matching parameters found for {material} at {energy} eV")
    
    # Create plots if requested
    if plot:
        _create_comparison_plot(updated_objectives, energy, material, figsize, 
                              profile_shift, save_plot, plot_filename, structures_dict)
    
    return updated_objectives


def _create_comparison_plot(objectives_dict, energy, material, figsize, profile_shift, 
                          save_plot, plot_filename, structures_dict):
    """
    Helper function to create comparison plots of experimental vs simulated data.
    
    Args:
        objectives_dict: Dictionary of objective functions
        energy: Energy value
        material: Material name
        figsize: Figure size
        profile_shift: Depth shift for SLD profiles
        save_plot: Whether to save the plot
        plot_filename: Filename for saved plot
        structures_dict: Dictionary of structure objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get the objective for this energy
    objective = objectives_dict[energy]
    
    # Calculate chi-squared
    chi_squared = objective.chisqr()
    
    # Determine subplot layout
    if structures_dict and energy in structures_dict:
        # Create 2x2 subplot layout: reflectivity, SLD profile, and parameter summary
        fig, ((ax_refl, ax_sld), (ax_params, ax_empty)) = plt.subplots(2, 2, figsize=figsize)
        ax_empty.axis('off')  # Hide the empty subplot
    else:
        # Create 1x2 subplot layout: reflectivity and parameter summary
        fig, (ax_refl, ax_params) = plt.subplots(1, 2, figsize=figsize)
        ax_sld = None
    
    # Plot reflectivity data vs model
    data = objective.data
    ax_refl.plot(data.data[0], data.data[1], 'o', label='Experimental Data', 
                markersize=4, color='black', alpha=0.7)
    ax_refl.plot(data.data[0], objective.model(data.data[0]), '-', 
                label='Simulated Data', linewidth=2, color='red')
    
    ax_refl.set_yscale('log')
    ax_refl.set_xlabel(r'Q ($\AA^{-1}$)', fontsize=12)
    ax_refl.set_ylabel('Reflectivity (a.u.)', fontsize=12)
    ax_refl.set_title(f'Reflectivity at {energy} eV\nχ² = {chi_squared:.4f}', fontsize=14)
    ax_refl.legend(loc='best', fontsize=10)
    ax_refl.grid(True, alpha=0.3)
    ax_refl.tick_params(labelsize=10)
    
    # Plot SLD profile if structure is available
    if ax_sld is not None and structures_dict and energy in structures_dict:
        structure = structures_dict[energy]
        
        try:
            # Try to get SLD profile using profileflip function if available
            try:
                from Plotting_Refl import profileflip
                Real_depth, Real_SLD, Imag_Depth, Imag_SLD = profileflip(structure, depth_shift=0)
            except ImportError:
                # Fallback method using refnx structure methods
                z = np.linspace(-10, 300, 1000)
                sld_profile = structure.sld_profile(z)
                Real_depth = Imag_Depth = z
                Real_SLD = np.real(sld_profile)
                Imag_SLD = np.imag(sld_profile)
            
            # Apply profile shift
            Real_depth_shifted = Real_depth + profile_shift
            Imag_Depth_shifted = Imag_Depth + profile_shift
            
            # Plot SLD profiles
            ax_sld.plot(Real_depth_shifted, Real_SLD, 'b-', label='Real SLD', linewidth=2)
            ax_sld.plot(Imag_Depth_shifted, Imag_SLD, 'r--', label='Imaginary SLD', linewidth=2)
            
            ax_sld.set_xlabel('Depth (Å)', fontsize=12)
            ax_sld.set_ylabel('SLD (×10⁻⁶ Å⁻²)', fontsize=12)
            ax_sld.set_title(f'SLD Profile at {energy} eV', fontsize=14)
            ax_sld.legend(loc='best', fontsize=10)
            ax_sld.grid(True, alpha=0.3)
            ax_sld.tick_params(labelsize=10)
            
        except Exception as e:
            ax_sld.text(0.5, 0.5, f'SLD Profile unavailable\n{str(e)}', 
                       transform=ax_sld.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax_sld.set_title(f'SLD Profile at {energy} eV (Error)', fontsize=14)
    
    # Create parameter summary table
    param_data = []
    for param in objective.parameters.flattened():
        if f"{material.lower()} - " in param.name.lower():
            param_type = param.name.split(' - ')[-1] if ' - ' in param.name else param.name
            bounds = getattr(param, 'bounds', None)
            vary = getattr(param, 'vary', False)
            
            if bounds is not None:
                if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                    bounds_str = f"({bounds.lb:.3f}, {bounds.ub:.3f})"
                elif isinstance(bounds, tuple) and len(bounds) == 2:
                    bounds_str = f"({bounds[0]:.3f}, {bounds[1]:.3f})"
                else:
                    bounds_str = str(bounds)
            else:
                bounds_str = "None"
            
            param_data.append([param_type, f"{param.value:.4f}", bounds_str, str(vary)])
    
    # Display parameter table
    if param_data:
        ax_params.axis('tight')
        ax_params.axis('off')
        table = ax_params.table(cellText=param_data,
                               colLabels=['Parameter', 'Value', 'Bounds', 'Vary'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax_params.set_title(f'{material} Parameters at {energy} eV', fontsize=14, pad=20)
    else:
        ax_params.text(0.5, 0.5, f'No {material} parameters found', 
                      transform=ax_params.transAxes, ha='center', va='center',
                      fontsize=12)
        ax_params.set_title(f'{material} Parameters at {energy} eV', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if plot_filename is None:
            plot_filename = f"{material}_energy_{energy}eV_comparison.png"
        
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    plt.show()




def batch_fit_selected_models_v2(objectives_dict, structures_dict, energy_list=None,
                                          method='differential_evolution', workers=-1, popsize=15,
                                          steps=1000, burn=500, nthin=1, nwalkers=100,
                                          results_database=None, save_dir=None, 
                                          save_objectives=False, save_results=False,
                                          preserve_originals=True, verbose=True, 
                                          database_file=None, model_name=None):
    """
    Run batch fitting procedure on selected reflectometry models using run_fitting_V2.
    Uses database approach with standardized refnx terminology.
    
    Enhanced version that includes ALL objectives (both fitted and non-fitted) in the 
    'fitted_objectives' output dictionary for convenient access to complete datasets.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to fit. If None, fit all available models.
        method (str): Optimization method ('differential_evolution', 'least_squares', etc.)
        workers (int): Number of workers for parallel optimization (-1 for all cores)
        popsize (int): Population size for differential evolution
        steps (int): Number of MCMC steps for sampling
        burn (int): Number of burn-in steps to discard
        nthin (int): Thinning factor for MCMC sampling
        nwalkers (int): Number of walkers for MCMC sampling
        results_database (DataFrame, optional): Existing results database to append to
        save_dir (str, optional): Directory to save objectives and results
        save_objectives (bool): Whether to save objective functions
        save_results (bool): Whether to save results dictionaries
        preserve_originals (bool): Whether to preserve original objectives in a separate dict
        verbose (bool): Whether to print detailed output
        database_file (str, optional): Path to save/load database CSV file
        model_name (str, optional): Name for all models in this batch. If None, uses 'Model1'.
        
    Returns:
        dict: Dictionary containing:
            - 'fitted_objectives': Dictionary of ALL objective functions (fitted AND non-fitted)
                                  Non-fitted objectives are included as-is from the input
            - 'original_objectives': Dictionary of original objectives (if preserve_originals=True)
            - 'results_database': Updated pandas DataFrame with fitting results (fitted energies only)
            - 'individual_results': Dictionary of individual fitting results (fitted energies only)
            - 'summary_stats': Summary statistics of the batch fitting
            - 'fitted_energies': List of energies that were actually fitted
            - 'non_fitted_energies': List of energies that were included but not fitted
            
    Note:
        The 'fitted_objectives' dictionary now contains ALL energies from the input objectives_dict.
        For energies that were fitted, it contains the fitted objective function.
        For energies that were NOT fitted (not in energy_list or missing structures), 
        it contains the original objective function unchanged.
        This allows for convenient access to complete datasets while preserving the distinction
        between fitted and non-fitted data through the 'fitted_energies' and 'non_fitted_energies' lists.
    """
    import copy
    import pandas as pd
    import numpy as np
    import os
    
    # Import the required functions (assuming they're available in the current scope)
    try:
        from Model_Setup import run_fitting_v2, create_empty_results_database, save_database_to_file, load_database_from_file
    except ImportError as e:
        print(f"Warning: Could not import required functions: {e}")
        print("Make sure Model_Setup.py is available and contains the required functions")
        return None
    
    print("="*60)
    print("BATCH FITTING WITH DATABASE APPROACH - V2 ENHANCED")
    print("="*60)
    
    # Load existing database if file is provided
    if database_file is not None:
        if os.path.exists(database_file):
            results_database = load_database_from_file(database_file)
            print(f"Loaded existing database from {database_file}")
        else:
            print(f"Database file {database_file} not found. Creating new database.")
    
    # Initialize results database if not provided
    if results_database is None:
        results_database = create_empty_results_database()
        print("Created new results database")
    
    # Start with ALL objectives from input (both fitted and non-fitted will be included)
    fitted_objectives = copy.deepcopy(objectives_dict)
    print(f"Initialized fitted_objectives with {len(fitted_objectives)} total objectives")
    
    # Determine which energies to fit
    if energy_list is None:
        available_energies = sorted(list(set(objectives_dict.keys()) & set(structures_dict.keys())))
        print(f"No energy list provided. Using all available energies: {len(available_energies)} total")
    else:
        # Filter to only include energies that have both objectives and structures
        available_energies = []
        for energy in energy_list:
            if energy in objectives_dict and energy in structures_dict:
                available_energies.append(energy)
            else:
                print(f"Warning: Energy {energy} eV not found in both objectives and structures dictionaries")
        print(f"Filtered energy list: {len(available_energies)} energies to fit")
    
    if not available_energies:
        print("ERROR: No valid energies found for fitting!")
        return None
    
    # Track fitted vs non-fitted energies
    all_energies = set(objectives_dict.keys())
    fitted_energies = set(available_energies)
    non_fitted_energies = all_energies - fitted_energies
    
    print(f"Energy breakdown:")
    print(f"  Total energies in input: {len(all_energies)}")
    print(f"  Energies to be fitted: {len(fitted_energies)}")
    print(f"  Energies kept as-is: {len(non_fitted_energies)}")
    if non_fitted_energies:
        print(f"  Non-fitted energies: {sorted(list(non_fitted_energies))}")
    
    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Using save directory: {save_dir}")
    else:
        save_dir = None
    
    # Initialize return dictionaries
    original_objectives = {}
    original_structures = {}  # Add original structures preservation
    individual_results = {}
    
    # Statistics tracking
    successful_fits = 0
    failed_fits = 0
    initial_chi_squared_total = 0
    final_chi_squared_total = 0
    
    print(f"\\nStarting batch fitting of {len(available_energies)} models...")
    print(f"Using method: {method}")
    print(f"MCMC settings: {steps} steps, {burn} burn-in, {nwalkers} walkers")
    
    # Main fitting loop - only process available_energies
    for i, energy in enumerate(available_energies):
        print(f"\\n--- Processing Energy {i+1}/{len(available_energies)}: {energy} eV ---")
        
        try:
            # Get the objective and structure for this energy
            original_objective = objectives_dict[energy]
            structure = structures_dict[energy]
            
            # Preserve original if requested
            if preserve_originals:
                original_objectives[energy] = copy.deepcopy(original_objective)
                original_structures[energy] = copy.deepcopy(structure)
            
            # Create a working copy of the objective
            working_objective = copy.deepcopy(original_objective)
            
            # Use the provided model name or default to 'Model1'
            if model_name is not None:
                energy_model_name = model_name
            else:
                energy_model_name = "Model1"
            
            # Record initial chi-squared
            initial_chi2 = working_objective.chisqr()
            initial_chi_squared_total += initial_chi2
            
            if verbose:
                print(f"  Initial χ²: {initial_chi2:.6g}")
            
            # Run fitting using run_fitting_V2
            fit_result = run_fitting_v2(
                objective=working_objective,
                method=method,
                workers=workers,
                popsize=popsize,
                steps=steps,
                burn=burn,
                nthin=nthin,
                nwalkers=nwalkers,
                results_database=results_database,
                log_mcmc_stats=True,
                save_dir=save_dir,
                save_objective=save_objectives,
                save_results=save_results,
                structure=structure,
                model_name=energy_model_name,
                add_to_database=True,
                verbose=verbose,
                energy=energy
            )
            
            # Handle different return formats from run_fitting_v2
            if isinstance(fit_result, tuple):
                if len(fit_result) == 2:
                    results, updated_database = fit_result
                elif len(fit_result) == 3:
                    results, updated_database, _ = fit_result  # Ignore third value
                else:
                    # Unexpected number of return values
                    print(f"Warning: run_fitting_v2 returned {len(fit_result)} values, expected 2 or 3")
                    results = fit_result[0]
                    updated_database = fit_result[1]
            else:
                # Single return value
                results = fit_result
                updated_database = results_database
            
            # Update the main database
            results_database = updated_database
            
            # Replace the original objective with the fitted one in fitted_objectives
            fitted_objectives[energy] = working_objective
            individual_results[energy] = results
            
            # Record final chi-squared
            final_chi2 = working_objective.chisqr()
            final_chi_squared_total += final_chi2
            
            # Calculate improvement
            if initial_chi2 > 0:
                improvement = (initial_chi2 - final_chi2) / initial_chi2 * 100
                print(f"  Final χ²: {final_chi2:.6g} (improvement: {improvement:.1f}%)")
            else:
                print(f"  Final χ²: {final_chi2:.6g}")
            
            successful_fits += 1
            
        except Exception as e:
            print(f"  ERROR: Failed to fit energy {energy} eV: {str(e)}")
            failed_fits += 1
            # Keep the original objective in fitted_objectives for failed fits
            continue
    
    # Calculate summary statistics
    print(f"\\n{'='*60}")
    print("BATCH FITTING SUMMARY")
    print(f"{'='*60}")
    print(f"Total models processed: {len(available_energies)}")
    print(f"Successful fits: {successful_fits}")
    print(f"Failed fits: {failed_fits}")
    print(f"Non-fitted objectives included: {len(non_fitted_energies)}")
    
    if successful_fits > 0:
        print(f"\\nOverall Performance:")
        print(f"  Total initial χ²: {initial_chi_squared_total:.6g}")
        print(f"  Total final χ²: {final_chi_squared_total:.6g}")
        
        if initial_chi_squared_total > 0:
            overall_improvement = (initial_chi_squared_total - final_chi_squared_total) / initial_chi_squared_total * 100
            print(f"  Overall improvement: {overall_improvement:.1f}%")
        
        # Database statistics
        print(f"\\nDatabase Statistics:")
        print(f"  Total entries in database: {len(results_database)}")
        print(f"  Unique models: {results_database['model_name'].nunique()}")
        print(f"  Energy range: {results_database['energy'].min():.1f} - {results_database['energy'].max():.1f} eV")
    
    # Save database to file if specified
    if database_file is not None:
        save_database_to_file(results_database, database_file)
        print(f"\\nSaved updated database to: {database_file}")
    
    # Prepare summary statistics
    summary_stats = {
        'total_models': len(available_energies),
        'total_objectives_in_output': len(fitted_objectives),
        'successful_fits': successful_fits,
        'failed_fits': failed_fits,
        'non_fitted_count': len(non_fitted_energies),
        'initial_chi_squared_total': initial_chi_squared_total,
        'final_chi_squared_total': final_chi_squared_total,
        'overall_improvement_percent': (initial_chi_squared_total - final_chi_squared_total) / initial_chi_squared_total * 100 if initial_chi_squared_total > 0 else 0,
        'save_directory': save_dir
    }
    
    # Prepare return dictionary
    return_dict = {
        'fitted_objectives': fitted_objectives,  # Contains ALL objectives (fitted + non-fitted)
        'results_database': results_database,    # Contains only fitted results
        'individual_results': individual_results, # Contains only fitted results
        'summary_stats': summary_stats,
        'fitted_energies': sorted(list(fitted_energies)),
        'non_fitted_energies': sorted(list(non_fitted_energies))
    }
    
    # Add originals if preserved
    if preserve_originals:
        return_dict['original_objectives'] = original_objectives
        return_dict['original_structures'] = original_structures
    
    print(f"\\nBatch fitting completed successfully!")
    print(f"fitted_objectives dictionary contains {len(fitted_objectives)} total objectives")
    print(f"  - {len(fitted_energies)} fitted objectives")
    print(f"  - {len(non_fitted_energies)} non-fitted objectives (included as-is)")
    
    return return_dict

def update_objective(objectives_dict, energy, material, updates):
    """
    Update parameter values and bounds for a specific energy and material in objectives dictionary.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        energy (float): Energy value to update
        material (str): Material name (e.g., "Resist", "PS")
        updates (dict): Dictionary of parameter updates with the following possible keys:
            - "thickness": new thickness value
            - "roughness": new roughness value  
            - "sld_real": new real SLD value
            - "sld_imag": new imaginary SLD value
            - "thickness_bounds": (lower, upper, vary) tuple for thickness bounds
            - "roughness_bounds": (lower, upper, vary) tuple for roughness bounds
            - "sld_real_bounds": (lower, upper, vary) tuple for real SLD bounds
            - "sld_imag_bounds": (lower, upper, vary) tuple for imaginary SLD bounds
            
    Returns:
        dict: Updated objectives dictionary
        
    Example:
        updated_objectives = update_objective(
            objectives_dict=my_objectives,
            energy=270.0,
            material="Resist",
            updates={
                "thickness": 180.0,
                "roughness": 5.0,
                "sld_real_bounds": (2.0, 8.0, True),
                "sld_imag_bounds": (0.0, 6.0, True),
                "thickness_bounds": (175.0, 185.0, True),
                "roughness_bounds": (3.0, 7.0, True)
            }
        )
    """
    import copy
    
    # Create a working copy
    updated_objectives = copy.deepcopy(objectives_dict)
    
    # Check if energy exists
    if energy not in updated_objectives:
        print(f"Warning: Energy {energy} not found in objectives")
        return updated_objectives
    
    objective = updated_objectives[energy]
    
    # Track updates made
    updates_made = []
    
    # Iterate through all parameters to find matches
    for param in objective.parameters.flattened():
        param_name = param.name.lower()
        
        # Check if this parameter belongs to the specified material
        if f"{material.lower()} - " not in param_name:
            continue
            
        # Determine parameter type and update accordingly
        param_updated = False
        
        # Handle thickness parameters
        if "thick" in param_name:
            if "thickness" in updates:
                old_value = param.value
                param.value = updates["thickness"]
                updates_made.append(f"thickness value: {old_value} -> {param.value}")
                param_updated = True
                
            if "thickness_bounds" in updates:
                lower, upper, vary = updates["thickness_bounds"]
                old_bounds = getattr(param, 'bounds', None)
                old_vary = getattr(param, 'vary', None)
                param.bounds = (lower, upper)
                param.vary = vary
                updates_made.append(f"thickness bounds: {old_bounds} -> {param.bounds}, vary: {old_vary} -> {param.vary}")
                param_updated = True
        
        # Handle roughness parameters
        elif "rough" in param_name:
            if "roughness" in updates:
                old_value = param.value
                param.value = updates["roughness"]
                updates_made.append(f"roughness value: {old_value} -> {param.value}")
                param_updated = True
                
            if "roughness_bounds" in updates:
                lower, upper, vary = updates["roughness_bounds"]
                old_bounds = getattr(param, 'bounds', None)
                old_vary = getattr(param, 'vary', None)
                param.bounds = (lower, upper)
                param.vary = vary
                updates_made.append(f"roughness bounds: {old_bounds} -> {param.bounds}, vary: {old_vary} -> {param.vary}")
                param_updated = True
        
        # Handle real SLD parameters
        elif "sld" in param_name and "isld" not in param_name:
            if "sld_real" in updates:
                old_value = param.value
                param.value = updates["sld_real"]
                updates_made.append(f"sld_real value: {old_value} -> {param.value}")
                param_updated = True
                
            if "sld_real_bounds" in updates:
                lower, upper, vary = updates["sld_real_bounds"]
                old_bounds = getattr(param, 'bounds', None)
                old_vary = getattr(param, 'vary', None)
                param.bounds = (lower, upper)
                param.vary = vary
                updates_made.append(f"sld_real bounds: {old_bounds} -> {param.bounds}, vary: {old_vary} -> {param.vary}")
                param_updated = True
        
        # Handle imaginary SLD parameters
        elif "isld" in param_name:
            if "sld_imag" in updates:
                old_value = param.value
                param.value = updates["sld_imag"]
                updates_made.append(f"sld_imag value: {old_value} -> {param.value}")
                param_updated = True
                
            if "sld_imag_bounds" in updates:
                lower, upper, vary = updates["sld_imag_bounds"]
                old_bounds = getattr(param, 'bounds', None)
                old_vary = getattr(param, 'vary', None)
                param.bounds = (lower, upper)
                param.vary = vary
                updates_made.append(f"sld_imag bounds: {old_bounds} -> {param.bounds}, vary: {old_vary} -> {param.vary}")
                param_updated = True
        
        if param_updated:
            print(f"Updated parameter: {param.name}")
    
    # Print summary
    if updates_made:
        print(f"\nSuccessfully updated {len(updates_made)} parameter(s) for {material} at {energy} eV:")
        for update in updates_made:
            print(f"  - {update}")
    else:
        print(f"No matching parameters found for {material} at {energy} eV")
    
    return updated_objectives


def get_parameter_info(objectives_dict, energy, material=None):
    """
    Get current parameter values and bounds for a specific energy and optionally material.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        energy (float): Energy value to inspect
        material (str, optional): Material name to filter by
        
    Returns:
        dict: Parameter information
    """
    if energy not in objectives_dict:
        print(f"Energy {energy} not found in objectives")
        return {}
    
    objective = objectives_dict[energy]
    param_info = {}
    
    for param in objective.parameters.flattened():
        param_name = param.name
        
        # Filter by material if specified
        if material and f"{material} - " not in param_name:
            continue
        
        param_info[param_name] = {
            'value': param.value,
            'bounds': getattr(param, 'bounds', None),
            'vary': getattr(param, 'vary', None)
        }
    
    return param_info

def plot_material_sld(sld_df, material_name, plot_type='both', figsize=(10, 6),
                      color='blue', marker='o', save_path=None, show_values=False,
                      xlim=None, ylim=None, title=None):
    """
    Plot SLD values for a single material with enhanced formatting.
    
    Args:
        sld_df (pandas.DataFrame): SLD DataFrame from extract_sld_from_objectives
        material_name (str): Name of the material to plot
        plot_type (str): 'real', 'imag', or 'both'
        figsize (tuple): Figure size as (width, height)
        color (str): Color for the plot lines
        marker (str): Marker style
        save_path (str, optional): Path to save the figure
        show_values (bool): Whether to annotate data points with values
        xlim (tuple, optional): X-axis limits
        ylim (tuple, optional): Y-axis limits
        title (str, optional): Custom title
        
    Returns:
        matplotlib figure and axes objects
    """
    
    # Filter data for the specified material
    material_data = sld_df[sld_df['Material'] == material_name].sort_values('Energy')
    
    if material_data.empty:
        print(f"Error: No data found for material '{material_name}'")
        available_materials = sorted(sld_df['Material'].unique())
        print(f"Available materials: {available_materials}")
        return None, None
    
    # Create subplots
    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    
    # Plot real component
    if plot_type in ['real', 'both']:
        ax_real = axes[0] if plot_type == 'both' else axes[0]
        
        ax_real.plot(material_data['Energy'], material_data['SLD_Real'],
                    color=color, marker=marker, markersize=8, linewidth=2.5,
                    markeredgecolor='white', markeredgewidth=1, alpha=0.8)
        
        ax_real.set_ylabel('Real SLD ($10^{-6}$ $Å^{-2}$)')
        ax_real.set_title(f'{material_name} - Real SLD vs Energy')
        ax_real.grid(True, alpha=0.3)
        
        # Add value annotations if requested
        if show_values:
            for _, row in material_data.iterrows():
                ax_real.annotate(f'{row["SLD_Real"]:.3f}', 
                                (row['Energy'], row['SLD_Real']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.7)
    
    # Plot imaginary component
    if plot_type in ['imag', 'both']:
        ax_imag = axes[1] if plot_type == 'both' else axes[0]
        
        ax_imag.plot(material_data['Energy'], material_data['SLD_Imag'],
                    color='red' if plot_type == 'both' else color, 
                    marker=marker, markersize=8, linewidth=2.5,
                    markeredgecolor='white', markeredgewidth=1, alpha=0.8)
        
        ax_imag.set_ylabel('Imaginary SLD ($10^{-6}$ $Å^{-2}$)')
        ax_imag.set_title(f'{material_name} - Imaginary SLD vs Energy')
        ax_imag.grid(True, alpha=0.3)
        
        # Add value annotations if requested
        if show_values:
            for _, row in material_data.iterrows():
                ax_imag.annotate(f'{row["SLD_Imag"]:.3f}', 
                                (row['Energy'], row['SLD_Imag']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.7)
    
    # Set x-axis properties
    axes[-1].set_xlabel('Energy (eV)')
    
    # Apply limits if specified
    if xlim is not None:
        axes[-1].set_xlim(xlim)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)
    
    # Set overall title
    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.98)
    elif plot_type == 'both':
        fig.suptitle(f'{material_name} - SLD vs Energy', fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    if title is not None or plot_type == 'both':
        plt.subplots_adjust(top=0.92)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {material_name} SLD plot to: {save_path}")
    
    return fig, axes


def plot_sld_comparison(sld_df, materials_comparison, figsize=(14, 10), save_path=None):
    """
    Create a comprehensive comparison plot of multiple materials showing both real and imaginary SLD.
    
    Args:
        sld_df (pandas.DataFrame): SLD DataFrame from extract_sld_from_objectives
        materials_comparison (list): List of materials to compare
        figsize (tuple): Figure size as (width, height)
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib figure and axes objects
    """
    
    # Filter available materials
    available_materials = sld_df['Material'].unique()
    valid_materials = [m for m in materials_comparison if m in available_materials]
    
    if not valid_materials:
        print("Error: No valid materials found for comparison")
        return None, None
    
    if len(valid_materials) != len(materials_comparison):
        missing = [m for m in materials_comparison if m not in available_materials]
        print(f"Warning: Materials {missing} not found. Plotting: {valid_materials}")
    
    # Create a 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Colors for materials
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # Plot 1: Real SLD comparison
    for i, material in enumerate(valid_materials):
        material_data = sld_df[sld_df['Material'] == material].sort_values('Energy')
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax1.plot(material_data['Energy'], material_data['SLD_Real'],
                color=color, marker=marker, label=material, markersize=6, linewidth=2)
    
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Real SLD ($10^{-6}$ $Å^{-2}$)')
    ax1.set_title('Real SLD Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Imaginary SLD comparison
    for i, material in enumerate(valid_materials):
        material_data = sld_df[sld_df['Material'] == material].sort_values('Energy')
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax2.plot(material_data['Energy'], material_data['SLD_Imag'],
                color=color, marker=marker, label=material, markersize=6, linewidth=2)
    
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Imaginary SLD ($10^{-6}$ $Å^{-2}$)')
    ax2.set_title('Imaginary SLD Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Complex plane plot (Real vs Imaginary)
    for i, material in enumerate(valid_materials):
        material_data = sld_df[sld_df['Material'] == material].sort_values('Energy')
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax3.scatter(material_data['SLD_Real'], material_data['SLD_Imag'],
                   c=material_data['Energy'], cmap='viridis', 
                   s=60, alpha=0.7, marker=marker, label=material)
    
    ax3.set_xlabel('Real SLD ($10^{-6}$ $Å^{-2}$)')
    ax3.set_ylabel('Imaginary SLD ($10^{-6}$ $Å^{-2}$)')
    ax3.set_title('Complex SLD Plane (colored by energy)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: SLD magnitude vs energy
    for i, material in enumerate(valid_materials):
        material_data = sld_df[sld_df['Material'] == material].sort_values('Energy')
        sld_magnitude = np.sqrt(material_data['SLD_Real']**2 + material_data['SLD_Imag']**2)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax4.plot(material_data['Energy'], sld_magnitude,
                color=color, marker=marker, label=material, markersize=6, linewidth=2)
    
    ax4.set_xlabel('Energy (eV)')
    ax4.set_ylabel('SLD Magnitude ($10^{-6}$ $Å^{-2}$)')
    ax4.set_title('SLD Magnitude vs Energy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('SLD Analysis Comparison', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SLD comparison plot to: {save_path}")
    
    return fig, (ax1, ax2, ax3, ax4)

def extract_sld_from_objectives(objectives_dict, materials_filter=None, verbose=True):
    """
    Extract SLD values from all materials in an objectives dictionary and convert to a pandas DataFrame.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        materials_filter (list, optional): List of material names to include. If None, extracts all materials.
        verbose (bool): Whether to print detailed extraction information
        
    Returns:
        pandas.DataFrame: DataFrame with columns ['Energy', 'Material', 'SLD_Real', 'SLD_Imag']
    """
    
    if not objectives_dict:
        raise ValueError("objectives_dict cannot be empty")
    
    sld_data = []
    energies = sorted(list(objectives_dict.keys()))
    
    if verbose:
        print(f"Extracting SLD values from {len(energies)} energies: {energies}")
    
    for energy in energies:
        if verbose:
            print(f"\nProcessing energy: {energy} eV")
        
        objective = objectives_dict[energy]
        
        # Access the model and structure
        model = objective.model
        structure = getattr(model, 'structure', None)
        
        if structure is None:
            if verbose:
                print(f"  Warning: No structure found for energy {energy} eV")
            continue
        
        # Extract materials from the structure
        materials_found = []
        
        # Get all layers from the structure
        # The structure should have layers or slabs that contain materials
        if hasattr(structure, '__iter__'):
            # Structure is iterable (list of layers)
            layers = structure
        elif hasattr(structure, 'layers'):
            # Structure has a layers attribute
            layers = structure.layers
        elif hasattr(structure, 'slabs'):
            # Structure has slabs method
            layers = structure.slabs()
        else:
            # Try to access as a list-like object
            try:
                layers = list(structure)
            except:
                if verbose:
                    print(f"  Warning: Could not access layers from structure for energy {energy} eV")
                continue
        
        # Extract SLD from each layer/slab
        for layer in layers:
            try:
                # Get the material/SLD object from the layer
                material_obj = None
                material_name = "unknown"
                
                # Try different ways to access the material
                if hasattr(layer, 'sld'):
                    material_obj = layer.sld
                    material_name = getattr(material_obj, 'name', 'unknown')
                elif hasattr(layer, 'material'):
                    material_obj = layer.material
                    material_name = getattr(material_obj, 'name', 'unknown')
                elif hasattr(layer, '__getitem__') and len(layer) > 0:
                    # For slab-like objects, the SLD might be the first element
                    material_obj = layer[0] if hasattr(layer[0], 'real') or hasattr(layer[0], 'complex') else None
                
                if material_obj is None:
                    continue
                
                # Apply materials filter if specified
                if materials_filter is not None and material_name not in materials_filter:
                    continue
                
                # Extract SLD values
                sld_real = None
                sld_imag = None
                
                # Method 1: Try complex() method (for MaterialSLD objects)
                if hasattr(material_obj, 'complex'):
                    try:
                        sld_complex = complex(material_obj)
                        sld_real = sld_complex.real
                        sld_imag = sld_complex.imag
                    except:
                        pass
                
                # Method 2: Try direct complex access (for SLD objects)
                if sld_real is None and hasattr(material_obj, 'real'):
                    try:
                        if hasattr(material_obj, 'imag'):
                            sld_real = float(material_obj.real)
                            sld_imag = float(material_obj.imag)
                        else:
                            # material_obj might be a complex number directly
                            complex_val = complex(material_obj)
                            sld_real = complex_val.real
                            sld_imag = complex_val.imag
                    except:
                        pass
                
                # Method 3: Try value attribute
                if sld_real is None and hasattr(material_obj, 'value'):
                    try:
                        value = material_obj.value
                        if isinstance(value, complex):
                            sld_real = value.real
                            sld_imag = value.imag
                        else:
                            sld_real = float(value)
                            sld_imag = 0.0
                    except:
                        pass
                
                # Method 4: Direct conversion to complex
                if sld_real is None:
                    try:
                        complex_val = complex(material_obj)
                        sld_real = complex_val.real
                        sld_imag = complex_val.imag
                    except:
                        if verbose:
                            print(f"    Warning: Could not extract SLD from material {material_name}")
                        continue
                
                # Store the extracted values
                if sld_real is not None and sld_imag is not None:
                    sld_data.append({
                        'Energy': energy,
                        'Material': material_name,
                        'SLD_Real': sld_real,
                        'SLD_Imag': sld_imag
                    })
                    materials_found.append(material_name)
                    
                    if verbose:
                        print(f"    {material_name}: SLD = {sld_real:.6f} + {sld_imag:.6f}j")
                
            except Exception as e:
                if verbose:
                    print(f"    Error processing layer: {str(e)}")
                continue
        
        if verbose and materials_found:
            print(f"  Found {len(materials_found)} materials: {materials_found}")
        elif verbose:
            print(f"  No materials extracted for energy {energy} eV")
    
    # Convert to DataFrame
    if not sld_data:
        if verbose:
            print("\nWarning: No SLD data was extracted from any objectives")
        return pd.DataFrame(columns=['Energy', 'Material', 'SLD_Real', 'SLD_Imag'])
    
    df = pd.DataFrame(sld_data)
    
    # Sort by energy and material for better organization
    df = df.sort_values(['Energy', 'Material']).reset_index(drop=True)
    
    if verbose:
        print(f"\n=== SLD Extraction Summary ===")
        print(f"Total data points: {len(df)}")
        print(f"Energy range: {df['Energy'].min():.1f} - {df['Energy'].max():.1f} eV")
        print(f"Materials found: {sorted(df['Material'].unique())}")
        print(f"Data points per material:")
        for material in sorted(df['Material'].unique()):
            count = len(df[df['Material'] == material])
            print(f"  {material}: {count} energies")
    
    return df






def save_batch_fit_results(batch_results, filename, save_dir=None, include_objectives=True, include_mcmc_samples=True):
    """
    Save the complete results from batch_fit_selected_models_v2 to a pickle file.
    
    Args:
        batch_results (dict): The complete results dictionary from batch_fit_selected_models_v2
        filename (str): Name of the file to save (without extension)
        save_dir (str, optional): Directory to save the file. If None, saves in current directory
        include_objectives (bool): Whether to include objective functions (default: True)
        include_mcmc_samples (bool): Whether to include MCMC samples (default: True)
        
    Returns:
        str: Full path to the saved file
    """
    
    # Validate inputs
    if not batch_results:
        raise ValueError("batch_results cannot be empty")
    
    required_keys = ['fitted_objectives', 'results_database', 'individual_results', 'summary_stats']
    missing_keys = [key for key in required_keys if key not in batch_results]
    if missing_keys:
        raise ValueError(f"batch_results is missing required keys: {missing_keys}")
    
    # Prepare the filename with .pkl extension
    if not filename.endswith('.pkl'):
        filename = f"{filename}.pkl"
    
    # Determine full file path
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)
    else:
        file_path = filename
    
    # Prepare data for saving
    save_data = {
        'file_type': 'batch_fit_results_v2',
        'results_database': batch_results['results_database'].copy(),
        'summary_stats': batch_results['summary_stats'].copy(),
        'metadata': {
            'include_objectives': include_objectives,
            'include_mcmc_samples': include_mcmc_samples,
            'version': 'v2'
        }
    }
    
    # Handle fitted_objectives
    if include_objectives and 'fitted_objectives' in batch_results:
        save_data['fitted_objectives'] = batch_results['fitted_objectives']
        print(f"Including {len(batch_results['fitted_objectives'])} fitted objectives")
    else:
        save_data['fitted_objectives'] = None
        print("Excluding fitted objectives from save")
    
    # Handle original_objectives if present
    if 'original_objectives' in batch_results:
        if include_objectives:
            save_data['original_objectives'] = batch_results['original_objectives']
            print(f"Including {len(batch_results['original_objectives'])} original objectives")
        else:
            save_data['original_objectives'] = None
            print("Excluding original objectives from save")
    
    # Handle original_structures if present
    if 'original_structures' in batch_results:
        save_data['original_structures'] = batch_results['original_structures']
        print(f"Including {len(batch_results['original_structures'])} original structures")
    
    # Handle individual_results and MCMC samples
    if include_mcmc_samples:
        save_data['individual_results'] = batch_results['individual_results']
        print(f"Including {len(batch_results['individual_results'])} individual results with MCMC samples")
    else:
        # Create a copy of individual_results without MCMC samples to save space
        individual_results_no_mcmc = {}
        for energy, results in batch_results['individual_results'].items():
            if isinstance(results, dict):
                results_copy = results.copy()
                results_copy['mcmc_samples'] = None
                individual_results_no_mcmc[energy] = results_copy
            else:
                individual_results_no_mcmc[energy] = results
        save_data['individual_results'] = individual_results_no_mcmc
        print(f"Including {len(individual_results_no_mcmc)} individual results without MCMC samples")
    
    # Save the data
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size for reporting
        file_size = os.path.getsize(file_path) / (1024*1024)  # Size in MB
        
        print(f"Successfully saved batch fit results to: {file_path}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Contains data for {len(batch_results['individual_results'])} energies")
        
        return file_path
        
    except Exception as e:
        print(f"Error saving batch results: {str(e)}")
        raise


def load_batch_fit_results(file_path, verbose=True):
    """
    Load batch fitting results saved by save_batch_fit_results.
    
    Args:
        file_path (str): Path to the saved batch results file
        verbose (bool): Whether to print detailed loading information
        
    Returns:
        dict: Dictionary containing the batch fitting results with the same structure as
              returned by batch_fit_selected_models_v2
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        if verbose:
            print(f"Successfully loaded batch results from: {file_path}")
        
        # Validate file type
        if isinstance(loaded_data, dict) and loaded_data.get('file_type') == 'batch_fit_results_v2':
            if verbose:
                print("Detected batch_fit_results_v2 format")
        else:
            print("Warning: File may not be in expected batch_fit_results_v2 format")
        
        # Reconstruct the batch_results dictionary
        batch_results = {
            'results_database': loaded_data.get('results_database', pd.DataFrame()),
            'summary_stats': loaded_data.get('summary_stats', {}),
            'individual_results': loaded_data.get('individual_results', {}),
        }
        
        # Add fitted_objectives if present
        if loaded_data.get('fitted_objectives') is not None:
            batch_results['fitted_objectives'] = loaded_data['fitted_objectives']
            if verbose:
                print(f"Loaded {len(batch_results['fitted_objectives'])} fitted objectives")
        else:
            if verbose:
                print("No fitted objectives in saved file")
        
        # Add original_objectives if present
        if loaded_data.get('original_objectives') is not None:
            batch_results['original_objectives'] = loaded_data['original_objectives']
            if verbose:
                print(f"Loaded {len(batch_results['original_objectives'])} original objectives")
        
        # Add original_structures if present
        if loaded_data.get('original_structures') is not None:
            batch_results['original_structures'] = loaded_data['original_structures']
            if verbose:
                print(f"Loaded {len(batch_results['original_structures'])} original structures")
        
        # Print summary information
        if verbose:
            metadata = loaded_data.get('metadata', {})

            
            print(f"\n--- Batch Results Summary ---")
            print(f"Number of energies: {len(batch_results['individual_results'])}")
            print(f"Database entries: {len(batch_results['results_database'])}")
            
            if 'summary_stats' in batch_results:
                stats = batch_results['summary_stats']
                print(f"Total models: {stats.get('total_models', 'Unknown')}")
                print(f"Successful fits: {stats.get('successful_fits', 'Unknown')}")
                print(f"Failed fits: {stats.get('failed_fits', 'Unknown')}")
                
                if 'overall_improvement_percent' in stats:
                    print(f"Overall improvement: {stats['overall_improvement_percent']:.1f}%")
            
            # Check for MCMC samples
            mcmc_count = 0
            for energy, results in batch_results['individual_results'].items():
                if isinstance(results, dict) and results.get('mcmc_samples') is not None:
                    mcmc_count += 1
            
            print(f"MCMC samples available for {mcmc_count}/{len(batch_results['individual_results'])} energies")
            
            # List available energies
            energies = sorted(list(batch_results['individual_results'].keys()))
            print(f"Available energies: {energies}")
        
        return batch_results
        
    except Exception as e:
        print(f"Error loading batch results: {str(e)}")
        raise


def get_batch_results_info(file_path):
    """
    Get summary information about a saved batch results file without fully loading it.
    
    Args:
        file_path (str): Path to the saved batch results file
        
    Returns:
        dict: Summary information about the file contents
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Extract summary information
        info = {
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / (1024*1024),
            'file_type': loaded_data.get('file_type', 'unknown'),
            'metadata': loaded_data.get('metadata', {}),
        }
        
        # Count energies
        individual_results = loaded_data.get('individual_results', {})
        info['num_energies'] = len(individual_results)
        info['energy_list'] = sorted(list(individual_results.keys())) if individual_results else []
        
        # Database info
        results_database = loaded_data.get('results_database', pd.DataFrame())
        info['database_entries'] = len(results_database)
        
        # Check what's included
        info['has_fitted_objectives'] = loaded_data.get('fitted_objectives') is not None
        info['has_original_objectives'] = loaded_data.get('original_objectives') is not None
        info['has_original_structures'] = loaded_data.get('original_structures') is not None
        
        # Check for MCMC samples
        mcmc_count = 0
        if individual_results:
            for energy, results in individual_results.items():
                if isinstance(results, dict) and results.get('mcmc_samples') is not None:
                    mcmc_count += 1
        info['mcmc_samples_count'] = mcmc_count
        
        # Summary stats
        summary_stats = loaded_data.get('summary_stats', {})
        info['summary_stats'] = summary_stats
        
        return info
        
    except Exception as e:
        print(f"Error reading batch results info: {str(e)}")
        raise



def create_interactive_fit_explorer(batch_results, figsize=(16, 12), default_material=None, fitted_only=False):
    """
    Create an interactive widget for exploring fit results with the exact layout specified.
    
    Args:
        batch_results (dict): Output from batch_fit_selected_models_v2
        figsize (tuple): Figure size as (width, height)
        default_material (str): Default material to select (if None, uses first available)
        fitted_only (bool): If True, only plot fitted data (no original data). Default is False.
        
    Returns:
        ipywidgets.VBox: Interactive widget
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import pandas as pd
    import numpy as np
    
    # Validate batch_results
    if not batch_results or 'results_database' not in batch_results:
        return widgets.HTML("<b>Error: Invalid batch_results provided</b>")
    
    # Extract data
    results_database = batch_results['results_database']
    fitted_objectives = batch_results['fitted_objectives']
    original_objectives = batch_results.get('original_objectives', {})
    individual_results = batch_results['individual_results']
    
    # Get original structures from batch_results if available
    original_structures = batch_results.get('original_structures', {})
    
    if results_database.empty:
        return widgets.HTML("<b>Error: No data found in results database</b>")
    
    # Extract available materials and energies
    all_parameters = results_database['parameter'].dropna().unique()
    
    # Extract material names (everything before the first ' - ' in parameter names)
    materials = set()
    for param in all_parameters:
        if ' - ' in param:
            material = param.split(' - ')[0].strip()
            materials.add(material)
    
    materials = sorted(list(materials))
    available_energies = sorted(results_database['energy'].dropna().unique())
    
    if not materials:
        return widgets.HTML("<b>Error: No materials found in database</b>")
    
    # Set default material
    if default_material and default_material in materials:
        selected_material = default_material
    else:
        selected_material = materials[0]
    
    print(f"Available materials: {materials}")
    print(f"Available energies: {available_energies}")
    if fitted_only:
        print("Mode: Fitted data only")
    else:
        if original_structures:
            print(f"Original structures available for energies: {list(original_structures.keys())}")
        else:
            print("Warning: No original structures found - SLD comparison will show fitted only")
    
    # Create widgets
    material_dropdown = widgets.Dropdown(
        options=materials,
        value=selected_material,
        description='Material:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    parameter_radio = widgets.RadioButtons(
        options=[
            ('SLD/iSLD', 'sld'),
            ('Thickness', 'thick'),
            ('Roughness', 'rough')
        ],
        value='sld',
        description='Parameter:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )
    
    energy_range_widget = widgets.FloatRangeSlider(
        value=[min(available_energies), max(available_energies)],
        min=min(available_energies),
        max=max(available_energies),
        step=1.0,
        description='Energy range:',
        continuous_update=False,
        readout_format='.1f',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    
    # Output widget for the plots
    plot_output = widgets.Output()
    
    # Status display
    status_label = widgets.HTML(value="Click on a point in the parameter plot to update comparisons")
    
    def get_parameter_data(material, param_type, energy_range):
        """Extract parameter data for plotting"""
        # Filter database
        material_data = results_database[
            results_database['parameter'].str.contains(f"{material} - ", case=False, na=False)
        ]
        
        # Filter by parameter type
        if param_type == 'sld':
            # Include both sld and isld
            param_data = material_data[
                material_data['parameter'].str.contains('sld', case=False, na=False)
            ]
        else:
            param_data = material_data[
                material_data['parameter'].str.contains(param_type, case=False, na=False)
            ]
        
        # Filter by energy range
        param_data = param_data[
            (param_data['energy'] >= energy_range[0]) & 
            (param_data['energy'] <= energy_range[1])
        ]
        
        return param_data
    
    def get_original_parameter_data(material, param_type, energy_range):
        """Extract original parameter data if available"""
        if fitted_only:
            return pd.DataFrame()  # Return empty DataFrame if fitted_only is True
            
        original_data = []
        
        for energy in available_energies:
            if energy < energy_range[0] or energy > energy_range[1]:
                continue
                
            if energy in original_objectives:
                try:
                    orig_obj = original_objectives[energy]
                    for param in orig_obj.parameters.flattened():
                        param_name = param.name
                        
                        # Check if parameter matches material and type
                        if f"{material} - " in param_name:
                            if param_type == 'sld' and 'sld' in param_name.lower():
                                original_data.append({
                                    'energy': energy,
                                    'parameter': param_name,
                                    'value': param.value,
                                    'type': 'original'
                                })
                            elif param_type in param_name.lower():
                                original_data.append({
                                    'energy': energy,
                                    'parameter': param_name,
                                    'value': param.value,
                                    'type': 'original'
                                })
                except Exception as e:
                    continue
        
        return pd.DataFrame(original_data)
    
    def update_plots():
        """Update all plots based on current widget values"""
        
        with plot_output:
            clear_output(wait=True)
            
            # Get current values
            material = material_dropdown.value
            param_type = parameter_radio.value
            energy_range = energy_range_widget.value
            
            # Get parameter data
            fitted_data = get_parameter_data(material, param_type, energy_range)
            original_data = get_original_parameter_data(material, param_type, energy_range)
            
            if fitted_data.empty:
                print(f"No data found for {material} - {param_type}")
                return
            
            # Create the figure with custom layout - tighter spacing
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1],
                         hspace=0.25, wspace=0.25, top=0.95, bottom=0.08, left=0.08, right=0.95)
            
            # Top plot spans both columns (parameter vs energy)
            ax_param = fig.add_subplot(gs[0, :])
            
            # Bottom plots
            ax_refl = fig.add_subplot(gs[1, 0])
            ax_sld = fig.add_subplot(gs[1, 1])
            
            # Plot parameter data - separate SLD and iSLD by marker shape
            clickable_artists = []
            
            # Define markers for different parameter types
            markers = {'sld': 'o', 'isld': 's'}  # circle for sld, square for isld
            
            if param_type == 'sld':
                # For SLD type, plot both sld and isld with different markers
                param_types_to_plot = ['sld', 'isld']
            else:
                # For other types, just plot the selected type
                param_types_to_plot = [param_type]
            
            # First pass: Plot bounds (behind the points)
            for sub_param_type in param_types_to_plot:
                # Process fitted data for bounds
                fitted_sub_data = fitted_data[
                    fitted_data['parameter'].str.contains(sub_param_type, case=False, na=False)
                ]
                
                if not fitted_sub_data.empty:
                    # Group by energy to get one point per energy
                    fitted_grouped = fitted_sub_data.groupby('energy').agg({
                        'value': 'first',
                        'parameter': 'first',
                        'bound_low': 'first',
                        'bound_high': 'first'
                    }).reset_index()
                    
                    # Add bounds as error bars for fitted data only
                    valid_bounds = fitted_grouped.dropna(subset=['bound_low', 'bound_high'])
                    if not valid_bounds.empty:
                        # Only plot error bars where bounds are meaningful (not infinite)
                        finite_bounds = valid_bounds[
                            (valid_bounds['bound_low'] > -1e10) & 
                            (valid_bounds['bound_high'] < 1e10)
                        ]
                        
                        if not finite_bounds.empty:
                            lower_errs = finite_bounds['value'] - finite_bounds['bound_low']
                            upper_errs = finite_bounds['bound_high'] - finite_bounds['value']
                            
                            # Choose color based on parameter type
                            if sub_param_type == 'sld':
                                bounds_color = 'black'
                            elif sub_param_type == 'isld':
                                bounds_color = 'grey'
                            else:
                                bounds_color = 'black'  # default for other parameters
                            
                            ax_param.errorbar(
                                finite_bounds['energy'], finite_bounds['value'],
                                yerr=[lower_errs, upper_errs],
                                fmt='none',  # No markers, just error bars
                                color=bounds_color,
                                alpha=0.6,
                                capsize=3,
                                capthick=1,
                                elinewidth=1,
                                zorder=1  # Behind the points
                            )
            
            # Second pass: Plot data points (on top of bounds)
            for sub_param_type in param_types_to_plot:
                # Process original data (red) - only if not fitted_only
                if not fitted_only and not original_data.empty:
                    # Filter for specific parameter type (sld or isld)
                    orig_sub_data = original_data[
                        original_data['parameter'].str.contains(sub_param_type, case=False, na=False)
                    ]
                    
                    if not orig_sub_data.empty:
                        # Group by energy to get one point per energy
                        orig_grouped = orig_sub_data.groupby('energy').agg({
                            'value': 'first',
                            'parameter': 'first'
                        }).reset_index()
                        
                        marker = markers.get(sub_param_type, 'o')
                        label = f'Original {sub_param_type.upper()}'
                        
                        scatter_orig = ax_param.scatter(
                            orig_grouped['energy'], orig_grouped['value'],
                            c='red', s=60, alpha=0.7, picker=True, 
                            edgecolors='black', linewidth=0.5,
                            marker=marker, label=label,
                            zorder=3  # On top of bounds
                        )
                        clickable_artists.append(('original', scatter_orig, orig_grouped))
                
                # Process fitted data (blue)
                fitted_sub_data = fitted_data[
                    fitted_data['parameter'].str.contains(sub_param_type, case=False, na=False)
                ]
                
                if not fitted_sub_data.empty:
                    # Group by energy to get one point per energy
                    fitted_grouped = fitted_sub_data.groupby('energy').agg({
                        'value': 'first',
                        'parameter': 'first',
                        'bound_low': 'first',
                        'bound_high': 'first'
                    }).reset_index()
                    
                    marker = markers.get(sub_param_type, 'o')
                    label = f'Fitted {sub_param_type.upper()}'
                    
                    scatter_fit = ax_param.scatter(
                        fitted_grouped['energy'], fitted_grouped['value'],
                        c='blue', s=60, alpha=0.7, picker=True,
                        edgecolors='black', linewidth=0.5,
                        marker=marker, label=label,
                        zorder=3  # On top of bounds
                    )
                    clickable_artists.append(('fitted', scatter_fit, fitted_grouped))
            
            # Style parameter plot
            ax_param.set_xlabel('Energy (eV)', fontsize=12)
            ax_param.set_ylabel(f'{param_type.upper()} Value', fontsize=12)
            
            # Update title based on mode
            if fitted_only:
                plot_title = f'Plot of parameters vs energy (fitted only)\\nMaterial: {material}'
            else:
                plot_title = f'Plot of parameters vs energy (with bounds)\\nMaterial: {material}'
            
            ax_param.set_title(plot_title, 
                              fontsize=14, fontweight='bold', color='white', 
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='teal', alpha=0.8))
            ax_param.grid(True, alpha=0.3)
            
            # Place legend inside the plot area to avoid cutoff
            if len(clickable_artists) > 0:
                ax_param.legend(loc='upper right', fontsize='small', framealpha=0.9)
            
            # Initialize bottom plots with placeholder text
            ax_refl.text(0.5, 0.5, 'Reflectivity plot\\ncomparing original and fitted result\\n\\nClick on a point above to update', 
                        ha='center', va='center', transform=ax_refl.transAxes,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='teal', alpha=0.8, edgecolor='white'),
                        color='white', fontsize=12, fontweight='bold')
            ax_refl.set_xticks([])
            ax_refl.set_yticks([])
            
            ax_sld.text(0.5, 0.5, 'SLD profile Plot\\nComparing original\\nand fitted result\\n\\nClick on a point above to update', 
                       ha='center', va='center', transform=ax_sld.transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='teal', alpha=0.8, edgecolor='white'),
                       color='white', fontsize=12, fontweight='bold')
            ax_sld.set_xticks([])
            ax_sld.set_yticks([])
            
            def update_comparison_plots(energy):
                """Update reflectivity and SLD comparison plots for selected energy"""
                
                # Clear the plots
                ax_refl.clear()
                ax_sld.clear()
                
                try:
                    # Get objectives for this energy
                    fitted_obj = fitted_objectives.get(energy)
                    
                    # Only get original objective if not in fitted_only mode
                    if fitted_only:
                        original_obj = fitted_obj  # Use fitted as "original" for comparison
                    else:
                        original_obj = original_objectives.get(energy, fitted_obj)
                    
                    if fitted_obj is None:
                        ax_refl.text(0.5, 0.5, f'No data for {energy} eV', ha='center', va='center',
                                   transform=ax_refl.transAxes)
                        ax_sld.text(0.5, 0.5, f'No data for {energy} eV', ha='center', va='center',
                                  transform=ax_sld.transAxes)
                        return
                    
                    # Plot reflectivity comparison
                    dataset = fitted_obj.data
                    q = dataset.x
                    r_obs = dataset.y
                    
                    if fitted_only:
                        # Only plot fitted data
                        r_fit = fitted_obj.model(q)
                        
                        ax_refl.semilogy(q, r_obs, 'o', markersize=3, alpha=0.6, color='gray', label='Observed')
                        ax_refl.semilogy(q, r_fit, '-', linewidth=2, color='blue', label='Fitted')
                    else:
                        # Plot both original and fitted
                        r_orig = original_obj.model(q)
                        r_fit = fitted_obj.model(q)
                        
                        ax_refl.semilogy(q, r_obs, 'o', markersize=3, alpha=0.6, color='gray', label='Observed')
                        ax_refl.semilogy(q, r_orig, '-', linewidth=2, color='red', label='Original')
                        ax_refl.semilogy(q, r_fit, '-', linewidth=2, color='blue', label='Fitted')
                    
                    ax_refl.set_xlabel('Q (Å⁻¹)')
                    ax_refl.set_ylabel('Reflectivity')
                    ax_refl.set_title(f'Reflectivity Comparison\\n{energy} eV', fontweight='bold')
                    ax_refl.legend(fontsize='small')
                    ax_refl.grid(True, alpha=0.3)
                    
                    # Plot SLD profile comparison - handle structures parallel to fitted structures
                    fitted_structure = None
                    original_structure = None
                    
                    # Get fitted structure (same way as before)
                    if energy in individual_results and 'structure' in individual_results[energy]:
                        fitted_structure = individual_results[energy]['structure']
                    
                    # Get original structure (only if not fitted_only mode)
                    if not fitted_only and energy in original_structures:
                        original_structure = original_structures[energy]
                    
                    if fitted_structure is not None or original_structure is not None:
                        try:
                            # Plot original SLD profile if available and not in fitted_only mode
                            if not fitted_only and original_structure is not None:
                                z_orig, sld_orig = original_structure.sld_profile()
                                
                                if hasattr(sld_orig, 'real'):
                                    ax_sld.plot(z_orig, sld_orig.real, 'r-', linewidth=2, label='Real SLD (Original)')
                                    if hasattr(sld_orig, 'imag'):
                                        ax_sld.plot(z_orig, sld_orig.imag, 'r--', linewidth=2, label='Imag SLD (Original)')
                                else:
                                    ax_sld.plot(z_orig, sld_orig, 'r-', linewidth=2, label='SLD (Original)')
                            
                            # Plot fitted SLD profile if available
                            if fitted_structure is not None:
                                z_fit, sld_fit = fitted_structure.sld_profile()
                                
                                if hasattr(sld_fit, 'real'):
                                    ax_sld.plot(z_fit, sld_fit.real, 'b-', linewidth=2, label='Real SLD (Fitted)')
                                    if hasattr(sld_fit, 'imag'):
                                        ax_sld.plot(z_fit, sld_fit.imag, 'b--', linewidth=2, label='Imag SLD (Fitted)')
                                else:
                                    ax_sld.plot(z_fit, sld_fit, 'b-', linewidth=2, label='SLD (Fitted)')
                            
                            ax_sld.set_xlabel('Distance (Å)')
                            ax_sld.set_ylabel('SLD (×10⁻⁶ Å⁻²)')
                            ax_sld.set_title(f'SLD Profile Comparison\\n{energy} eV', fontweight='bold')
                            ax_sld.legend(fontsize='small')
                            ax_sld.grid(True, alpha=0.3)
                            
                        except Exception as e:
                            ax_sld.text(0.5, 0.5, f'SLD Profile Error:\\n{str(e)}', 
                                       ha='center', va='center', transform=ax_sld.transAxes)
                    else:
                        ax_sld.text(0.5, 0.5, 'No structures available\\nfor SLD profile comparison', 
                                   ha='center', va='center', transform=ax_sld.transAxes)
                    
                    # Update status
                    if fitted_only:
                        fitted_chi2 = fitted_obj.chisqr()
                        status_label.value = (f"<b>Energy:</b> {energy} eV | "
                                            f"<b>Fitted χ²:</b> {fitted_chi2:.4g}")
                    else:
                        orig_chi2 = original_obj.chisqr()
                        fitted_chi2 = fitted_obj.chisqr()
                        improvement = (orig_chi2 - fitted_chi2) / orig_chi2 * 100 if orig_chi2 > 0 else 0
                        
                        status_label.value = (f"<b>Energy:</b> {energy} eV | "
                                            f"<b>Original χ²:</b> {orig_chi2:.4g} | "
                                            f"<b>Fitted χ²:</b> {fitted_chi2:.4g} | "
                                            f"<b>Improvement:</b> {improvement:.1f}%")
                    
                except Exception as e:
                    ax_refl.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                               transform=ax_refl.transAxes)
                    ax_sld.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                              transform=ax_sld.transAxes)
                    status_label.value = f"<b>Error:</b> {str(e)}"
                
                fig.canvas.draw_idle()
            
            def on_pick(event):
                """Handle click events on parameter plot"""
                try:
                    if hasattr(event, 'ind') and len(event.ind) > 0:
                        ind = event.ind[0]
                        
                        # Find which artist was clicked
                        for plot_type, artist, data in clickable_artists:
                            if event.artist == artist:
                                energy = data.iloc[ind]['energy']
                                print(f"Clicked on {plot_type} point at {energy} eV")
                                update_comparison_plots(energy)
                                break
                                
                except Exception as e:
                    print(f"Error in click handler: {e}")
                    status_label.value = f"<b>Error:</b> {str(e)}"
            
            # Connect the click event
            fig.canvas.mpl_connect('pick_event', on_pick)
            
            # Tight layout with minimal padding
            plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.25, wspace=0.25)
            plt.show()
    
    # Connect widget events
    def on_widget_change(change):
        update_plots()
    
    material_dropdown.observe(on_widget_change, names='value')
    parameter_radio.observe(on_widget_change, names='value')
    energy_range_widget.observe(on_widget_change, names='value')
    
    # Create the layout with minimal spacing and no header
    controls_box = widgets.HBox([
        widgets.VBox([
            widgets.HTML("<b>Box to select materials</b>", 
                        layout=widgets.Layout(background_color='teal', color='white', 
                                            padding='3px', margin='1px')),
            material_dropdown
        ], layout=widgets.Layout(width='250px')),
        
        widgets.VBox([
            parameter_radio
        ], layout=widgets.Layout(width='200px')),
        
        widgets.VBox([
            energy_range_widget
        ], layout=widgets.Layout(width='450px'))
    ], layout=widgets.Layout(margin='0px 0px 5px 0px'))  # Minimal margin
    
    # Main widget layout with minimal spacing
    main_widget = widgets.VBox([
        controls_box,
        plot_output,
        status_label
    ], layout=widgets.Layout(margin='0px', padding='0px'))
    
    # Initial plot
    update_plots()
    
    return main_widget