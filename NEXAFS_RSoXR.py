### This code will be used to fit batches of RSoXR data based on NEXAFS Spectra


import os
import re
import numpy as np
from refnx.dataset import ReflectDataset
from pathlib import Path

from Model_Setup import create_reflectometry_model, create_model_and_objective, SLDinterp, run_fitting, print_fit_results, log_fitting_results
from Plotting_Refl import modelcomparisonplot, profileflip

def import_batch_reflectivity(folder_path, file_type='smoothed'):
    """
    Import a batch of reflectivity data files from a folder.
    
    Args:
        folder_path (str): Path to the folder containing reflectivity data files
        file_type (str): Type of files to import - 'raw', 'smoothed', or 'all'
                       - 'raw' will import *raw.dat files
                       - 'smoothed' will import *smoothed.dat files
                       - 'all' will import all .dat files
    
    Returns:
        tuple: (data_dict, energy_list)
            - data_dict: Dictionary mapping energy values to ReflectDataset objects
            - energy_list: Sorted list of all energy values found
    """
    # Dictionary to store ReflectDataset objects keyed by energy
    data_dict = {}
    
    # List to store the energy values
    energy_list = []
    
    # Determine file pattern based on file_type
    if file_type == 'raw':
        pattern = r'.*_([0-9.]+).*raw\.dat$'
    elif file_type == 'smoothed':
        pattern = r'.*_([0-9.]+).*smoothed\.dat$'
    else:  # 'all'
        pattern = r'.*_([0-9.]+).*\.dat$'
    
    # Get list of all files in the directory
    folder_path = Path(folder_path)
    all_files = list(folder_path.glob('*.dat'))
    
    # Track files that match the pattern but failed to load
    failed_files = []
    
    # Process each file
    for file_path in all_files:
        file_name = file_path.name
        # Try to extract energy from filename
        match = re.match(pattern, file_name)
        
        if match:
            try:
                # Extract energy value from filename
                energy = float(match.group(1))
                
                # Load the reflectivity data
                data = ReflectDataset(str(file_path))
                
                # Store in the dictionary
                data_dict[energy] = data
                energy_list.append(energy)
                
                print(f"Loaded {file_name} with energy {energy} eV")
            except Exception as e:
                failed_files.append((file_name, str(e)))
    
    # Report any failures
    if failed_files:
        print("\nThe following files matched the pattern but failed to load:")
        for file_name, error in failed_files:
            print(f"  - {file_name}: {error}")
    
    # Sort the energy list
    energy_list = sorted(energy_list)
    
    # Report summary
    print(f"\nSuccessfully loaded {len(data_dict)} reflectivity datasets")
    print(f"Energy range: {min(energy_list) if energy_list else 'N/A'} to {max(energy_list) if energy_list else 'N/A'} eV")
    
    return data_dict, energy_list



def generate_materials_from_sld_arrays(energy_list, material_sld_arrays, constant_materials=None):
    """
    Generate materials lists for each energy by interpolating SLD values from NumPy arrays.
    
    Args:
        energy_list (list): List of energies to generate materials for
        material_sld_arrays (dict): Dictionary mapping material names to numpy arrays
                                  Arrays should have columns [Energy, Real SLD, Imag SLD]
        constant_materials (dict, optional): Dictionary of materials with constant SLD values
                                          Format: {"name": {"real": value, "imag": value}}
    
    Returns:
        dict: Dictionary mapping energies to material lists
    """
    from scipy.interpolate import interp1d
    
    # Dictionary to store materials for each energy
    energy_materials = {}
    
    # Dictionary to store interpolation functions for each material
    interp_funcs = {}
    
    # Create interpolation functions from NumPy arrays
    print("Creating interpolation functions from SLD arrays...")
    for material, data in material_sld_arrays.items():
        try:
            # Check data format
            if data.shape[1] < 3:
                print(f"Warning: Array for {material} has incorrect format. Expected columns: [Energy, Real SLD, Imag SLD]")
                continue
            
            # Create interpolation functions
            real_interp = interp1d(
                data[:, 0],  # Energy values
                data[:, 1],  # Real SLD values
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            imag_interp = interp1d(
                data[:, 0],  # Energy values
                data[:, 2],  # Imaginary SLD values
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            interp_funcs[material] = (real_interp, imag_interp, data[:, 0].min(), data[:, 0].max())
            
            print(f"Created interpolation functions for {material}: {len(data)} points, energy range {data[:, 0].min():.1f}-{data[:, 0].max():.1f} eV")
            
        except Exception as e:
            print(f"Error processing SLD array for {material}: {str(e)}")
    
    # Set default for constant materials if not provided
    if constant_materials is None:
        constant_materials = {
            "air": {"real": 0.0, "imag": 0.0}
        }
    
    # Generate materials for each energy
    print("\nGenerating materials for each energy...")
    for energy in energy_list:
        # Create a materials list for this energy
        materials_list = []
        
        # Add interpolated materials
        for material, (real_interp, imag_interp, min_e, max_e) in interp_funcs.items():
            # Get interpolated SLD values
            real_sld = real_interp(energy)
            imag_sld = imag_interp(energy)
            
            # Create the material entry
            materials_list.append({
                "name": material,
                "real": real_sld,
                "imag": imag_sld*1j ##### added 1j here
            })
            
            # Warn if energy is outside the range of the original data
            if energy < min_e or energy > max_e:
                print(f"Warning: Energy {energy} eV is outside the data range for {material} ({min_e:.1f}-{max_e:.1f} eV)")
        
        # Add constant materials
        for name, values in constant_materials.items():
            materials_list.append({
                "name": name,
                "real": values["real"],
                "imag": values["imag"]
            })
        
        # Store the materials list for this energy
        energy_materials[energy] = materials_list
        
    print(f"Generated materials lists for {len(energy_list)} energies")
    
    return energy_materials

def generate_layer_params_with_flexible_bounds(energy_materials, base_layer_params, sld_offset_bounds=None):
    """
    Generate layer parameters with SLD bounds, prioritizing explicit bounds in base_layer_params.
    If no bounds are provided in either base_layer_params or sld_offset_bounds, the parameter will not be varied.
    Ensures that the lower bound for imaginary SLD components is never less than zero.
    
    Args:
        energy_materials (dict): Dictionary mapping energies to materials lists
        base_layer_params (dict): Base layer parameters that may include explicit SLD bounds
        sld_offset_bounds (dict, optional): Dictionary of absolute offset bounds for materials
                                        without explicit bounds in base_layer_params
    
    Returns:
        dict: Dictionary mapping energies to layer parameter dictionaries
    """
    # Initialize output dictionary
    energy_layer_params = {}
    
    # Set default offset bounds if not provided
    if sld_offset_bounds is None:
        sld_offset_bounds = {}
    
    # Process each energy
    for energy, materials_list in energy_materials.items():
        # Create a deep copy of the base layer parameters
        import copy
        layer_params = copy.deepcopy(base_layer_params)
        
        # Create a dictionary to quickly look up material SLD values by name
        material_sld_map = {material["name"]: material for material in materials_list}
        
        # Update SLD bounds for each material
        for material_name, params in layer_params.items():
            # Skip if the material isn't in the materials list
            if material_name not in material_sld_map:
                continue
            
            # Get the SLD values for this material at this energy
            material_sld = material_sld_map[material_name]
            real_sld = material_sld["real"]
            imag_sld = material_sld["imag"]
            
            # Check if explicit SLD bounds are provided in base_layer_params
            has_explicit_real_bounds = "sld_real_bounds" in params
            has_explicit_imag_bounds = "sld_imag_bounds" in params
            
            # Only apply offset bounds if explicit bounds are not provided
            if not has_explicit_real_bounds:
                if material_name in sld_offset_bounds and "real" in sld_offset_bounds[material_name]:
                    # Apply offset bounds
                    real_offsets = sld_offset_bounds[material_name]["real"]
                    min_offset, max_offset, vary = real_offsets
                    params["sld_real_bounds"] = (
                        real_sld + min_offset,  # Lower bound
                        real_sld + max_offset,  # Upper bound
                        vary                    # Whether to vary
                    )
                else:
                    # No bounds provided, set to fixed value
                    params["sld_real_bounds"] = (
                        real_sld * 0.999,  # Tiny range for numerical stability
                        real_sld * 1.001,
                        False              # Not varied
                    )
            
            if not has_explicit_imag_bounds:
                if material_name in sld_offset_bounds and "imag" in sld_offset_bounds[material_name]:
                    # Apply offset bounds
                    imag_offsets = sld_offset_bounds[material_name]["imag"]
                    min_offset, max_offset, vary = imag_offsets
                    
                    # Calculate lower bound and ensure it's not less than zero
                    lower_bound = max(0, imag_sld + min_offset)
                    
                    params["sld_imag_bounds"] = (
                        lower_bound,          # Lower bound (never less than zero)
                        imag_sld + max_offset,  # Upper bound
                        vary                    # Whether to vary
                    )
                else:
                    # No bounds provided, set to fixed value
                    params["sld_imag_bounds"] = (
                        max(0, imag_sld * 0.999),  # Tiny range for numerical stability, never less than zero
                        imag_sld * 1.001,
                        False              # Not varied
                    )
        
        # Store the updated layer parameters for this energy
        energy_layer_params[energy] = layer_params
    
    return energy_layer_params

def generate_batch_models(data_dict, energy_list, material_sld_arrays, constant_materials,
                         base_layer_params, layer_order, sld_offset_bounds=None, 
                         sample_name="Sample", scale =1, scale_bounds=(0.1, 10), bkg_bounds=(0.01, 10),
                         dq_bounds=(1.0, 2.0), vary_scale=True, vary_bkg=True, vary_dq=False, 
                         dq=1.6, verbose=True):
    """
    Generate a batch of reflectometry models without fitting.
    
    Args:
        data_dict (dict): Dictionary mapping energy values to ReflectDataset objects
        energy_list (list): List of energies to generate models for
        material_sld_arrays (dict): Dictionary mapping material names to numpy arrays
                                  Arrays should have columns [Energy, Real SLD, Imag SLD]
        constant_materials (dict): Dictionary of materials with constant SLD values
        base_layer_params (dict): Base layer parameters that may include explicit SLD bounds
        layer_order (list): Order of layers from top to bottom
        sld_offset_bounds (dict, optional): Offset bounds for materials without explicit bounds
        sample_name (str): Name prefix for the sample
        scale_bounds (tuple): Bounds for scale parameter (lower, upper)
        bkg_bounds (tuple): Bounds for background parameter (lower, upper)
        dq_bounds (tuple): Bounds for resolution parameter (lower, upper)
        vary_scale (bool): Whether to vary the scale parameter
        vary_bkg (bool): Whether to vary the background parameter
        vary_dq (bool): Whether to vary the resolution parameter
        dq (float): Initial resolution parameter value
        verbose (bool): Whether to print detailed information during generation
        
    Returns:
        tuple: (models_dict, structures_dict, objectives_dict)
            - models_dict: Dictionary mapping energy values to ReflectModel objects
            - structures_dict: Dictionary mapping energy values to Structure objects
            - objectives_dict: Dictionary mapping energy values to Objective objects
    """
    if verbose:
        print("Generating materials for each energy...")
    
    # Step 1: Generate materials lists for each energy
    energy_materials = generate_materials_from_sld_arrays(
        energy_list,
        material_sld_arrays,
        constant_materials
    )
    
    # Step 2: Generate layer parameters with flexible SLD bounds
    if verbose:
        print("\nGenerating layer parameters with flexible SLD bounds...")
    
    energy_layer_params = generate_layer_params_with_flexible_bounds(
        energy_materials,
        base_layer_params,
        sld_offset_bounds
    )
    
    # Step 3: Set up models for all energies
    if verbose:
        print("\nSetting up reflectometry models...")
    
    # Initialize dictionaries
    models_dict = {}
    structures_dict = {}
    objectives_dict = {}
    
    # Process each energy
    for energy in energy_list:
        # Skip if there's no data or parameters for this energy
        if energy not in data_dict or energy not in energy_materials or energy not in energy_layer_params:
            if verbose:
                print(f"Warning: Missing data, materials, or parameters for energy {energy} eV. Skipping.")
            continue
        
        try:
            # Get materials and parameters for this energy
            materials_list = energy_materials[energy]
            layer_params = energy_layer_params[energy]
            
            # Create the model
            materials, layers, structure, model_name = create_reflectometry_model(
                materials_list=materials_list,
                layer_params=layer_params,
                layer_order=layer_order,
                sample_name=sample_name,
                energy=energy
            )
            
            # Create the model and objective
            model, objective = create_model_and_objective(
                structure=structure,
                data=data_dict[energy],
                model_name=model_name,
                scale=scale,
                bkg=None,  # Auto-set to minimum value in data
                dq=dq,
                vary_scale=vary_scale,
                vary_bkg=vary_bkg,
                vary_dq=vary_dq,
                scale_bounds=scale_bounds,
                bkg_bounds=bkg_bounds,
                dq_bounds=dq_bounds
            )
            
            # Store in dictionaries
            models_dict[energy] = model
            structures_dict[energy] = structure
            objectives_dict[energy] = objective
            
            if verbose:
                print(f"Created model for energy {energy} eV: {model_name}")
            
        except Exception as e:
            if verbose:
                print(f"Error creating model for energy {energy} eV: {str(e)}")
    
    if verbose:
        print(f"\nCreated {len(models_dict)} models for {len(energy_list)} energies")
    
    return models_dict, structures_dict, objectives_dict


def visualize_batch_models(objectives_dict, structures_dict, energies=None, 
                          shade_start=None, profile_shift=-20, xlim=None, 
                          fig_size_w=16, colors=None, save_path=None):
    """
    Visualize reflectometry models for specific energies using the modelcomparisonplot function.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energies (list or float, optional): Energy or list of energies to visualize.
                                         If None, visualizes all available energies.
        shade_start (list, optional): List of starting positions for layer shading
                                   If None, uses the same value for all plots
        profile_shift (float): Shift applied to depth profiles
        xlim (list, optional): Custom x-axis limits for SLD plots as [min, max]
        fig_size_w (float): Width of the figure
        colors (list, optional): List of colors for layer shading
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, axes) - matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Process the energies parameter
    if energies is None:
        # Use all available energies
        available_energies = sorted(list(set(objectives_dict.keys()) & set(structures_dict.keys())))
    elif isinstance(energies, (int, float)):
        # Single energy provided
        available_energies = [energies]
    else:
        # List of energies provided
        available_energies = sorted(energies)
    
    # Filter by available data
    energies_to_plot = []
    for energy in available_energies:
        if energy in objectives_dict and energy in structures_dict:
            energies_to_plot.append(energy)
        else:
            print(f"Warning: No data available for energy {energy} eV. Skipping.")
    
    if not energies_to_plot:
        print("No valid energies to plot.")
        return None, None
    
    print(f"Generating plots for energies: {energies_to_plot}")
    
    # Prepare lists for the modelcomparisonplot function
    obj_list = [objectives_dict[energy] for energy in energies_to_plot]
    structure_list = [structures_dict[energy] for energy in energies_to_plot]
    
    # Prepare shade_start
    if shade_start is None:
        shade_start = [0] * len(energies_to_plot)
    elif isinstance(shade_start, (int, float)):
        shade_start = [shade_start] * len(energies_to_plot)
    
    # Call the modelcomparisonplot function
    try:
        fig, axes = modelcomparisonplot(
            obj_list=obj_list, 
            structure_list=structure_list,
            shade_start=shade_start,
            profile_shift=profile_shift,
            xlim=xlim,
            fig_size_w=fig_size_w,
            colors=colors
        )
        
        # Add energy labels to each plot
        num_plots = len(energies_to_plot)
        if num_plots == 1:
            # Single plot case
            axes[0].set_title(f"Reflectivity - {energies_to_plot[0]} eV")
            axes[1].set_title(f"SLD Profile - {energies_to_plot[0]} eV")
        else:
            # Multiple plots case
            for i, energy in enumerate(energies_to_plot):
                axes[0, i].set_title(f"Reflectivity - {energy} eV")
                axes[1, i].set_title(f"SLD Profile - {energy} eV")
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, axes
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return None, None
    
    
def batch_fit_selected_models(objectives_dict, structures_dict, energy_list=None, 
                             optimization_method='differential_evolution', 
                             opt_workers=8, opt_popsize=20, burn_samples=5, 
                             production_samples=5, prod_steps=1, pool=16,
                             results_log=None, log_mcmc_stats=True,
                             save_dir=None, save_objective=False, save_results=False,
                             results_log_file=None, save_log_in_save_dir=False):
    """
    Run fitting procedure on selected reflectometry models.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to fit. If None, fit all available models.
        optimization_method (str): Optimization method to use
        opt_workers (int): Number of workers for parallel optimization
        opt_popsize (int): Population size for genetic algorithms
        burn_samples (int): Number of burn-in samples to discard (in thousands)
        production_samples (int): Number of production samples to keep (in thousands)
        prod_steps (int): Number of steps between stored samples
        pool (int): Number of parallel processes for MCMC sampling
        results_log (DataFrame, optional): Existing results log DataFrame to append to
        log_mcmc_stats (bool): Whether to add MCMC statistics to the log
        save_dir (str, optional): Directory to save objective and results
        save_objective (bool): Whether to save the objective function
        save_results (bool): Whether to save the results dictionary
        results_log_file (str, optional): Filename to load/save the results log DataFrame
        save_log_in_save_dir (bool): If True, save the log file in save_dir
        
    Returns:
        tuple: (results_dict, updated_results_df)
            - results_dict: Dictionary mapping energy values to fitting results
            - updated_results_df: Combined DataFrame of all fitting results
    """
    import os
    import pandas as pd
    from datetime import datetime
    import pickle
    
    # Dictionary to store fitting results
    results_dict = {}
    
    # If no energy list is provided, use all available energies
    if energy_list is None:
        energy_list = sorted(list(objectives_dict.keys()))
    
    # Filter to ensure we only process energies that have both objectives and structures
    valid_energies = []
    for energy in energy_list:
        if energy in objectives_dict and energy in structures_dict:
            valid_energies.append(energy)
        else:
            print(f"Warning: Missing objective or structure for energy {energy} eV. Skipping.")
    
    if not valid_energies:
        print("No valid energies to fit.")
        return {}, results_log
    
    print(f"Fitting {len(valid_energies)} models: {valid_energies}")
    
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
    
    # Initialize results_log if not provided and not loaded from file
    if results_log is None:
        print("Initializing new results log")
        results_log = pd.DataFrame(columns=[
            'timestamp', 'model_name', 'goodness_of_fit', 
            'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary'
        ])
    
    # Process each energy
    for energy in valid_energies:
        objective = objectives_dict[energy]
        structure = structures_dict[energy]
        
        # Extract model name if available
        model_name = getattr(objective.model, 'name', f"Model_{energy}eV")
        print(f"Fitting model: {model_name}")
        
        # Generate timestamp for logs and internal tracking (but not filenames)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize results dictionary
        results = {
            'objective': objective,
            'initial_chi_squared': objective.chisqr(),
            'optimized_parameters': None,
            'optimized_chi_squared': None,
            'mcmc_samples': None,
            'mcmc_stats': None,
            'timestamp': timestamp,
            'structure': structure
        }
        
        try:
            # Create fitter
            from refnx.analysis import CurveFitter
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
                            if chain_index >= 0 and results['mcmc_samples'] is not None:
                                # Calculate statistics
                                chain_values = results['mcmc_samples'][:, chain_index]
                                param_stats['median'] = np.median(chain_values)
                                param_stats['mean'] = np.mean(chain_values)
                                param_stats['std'] = np.std(chain_values)
                                
                                # Calculate percentiles
                                for percentile in [2.5, 16, 50, 84, 97.5]:
                                    param_stats['percentiles'][percentile] = np.percentile(chain_values, percentile)
                            
                            results['mcmc_stats'][param.name] = param_stats
                    
                    # Print a summary of key parameters
                    print("\nParameter summary from MCMC:")
                    for name, stats in results['mcmc_stats'].items():
                        if 'median' in stats and stats['median'] is not None:
                            print(f"  {name}: {stats['median']:.6g} +{stats['percentiles'][84] - stats['median']:.6g} -{stats['median'] - stats['percentiles'][16]:.6g}")
                
                except Exception as e:
                    print(f"Error calculating MCMC statistics: {str(e)}")
            
            # Log the results
            print(f"Logging results for model {model_name}")
            
            # Log the optimized values first
            results_log, updated_model_name = log_fitting_results(objective, model_name, results_log)
            
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
                mcmc_model_name = f"{updated_model_name}_MCMC"
                results_log, _ = log_fitting_results(mcmc_objective, mcmc_model_name, results_log)
            
            # Save the objective and/or results if requested
            if save_dir is not None and (save_objective or save_results):
                # Create the directory if it doesn't exist
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    print(f"Created directory: {save_dir}")
                
                # Save the objective if requested
                if save_objective:
                    objective_filename = os.path.join(save_dir, f"{updated_model_name}_objective.pkl")
                    try:
                        with open(objective_filename, 'wb') as f:
                            pickle.dump(objective, f)
                        print(f"Saved objective to {objective_filename}")
                    except Exception as e:
                        print(f"Error saving objective: {str(e)}")
                
                # Save the results if requested
                if save_results:
                    # Create a copy of results without the objective (to avoid duplication if saving both)
                    save_results_copy = results.copy()
                    if 'objective' in save_results_copy and save_objective:
                        save_results_copy['objective'] = None  # Remove the objective to avoid duplication
                    
                    results_filename = os.path.join(save_dir, f"{updated_model_name}_results.pkl")
                    try:
                        with open(results_filename, 'wb') as f:
                            pickle.dump(save_results_copy, f)
                        print(f"Saved results to {results_filename}")
                    except Exception as e:
                        print(f"Error saving results: {str(e)}")
                    
                    # Save a combined file with results and structure
                    combined_filename = os.path.join(save_dir, f"{updated_model_name}_combined.pkl")
                    try:
                        combined_data = {
                            'results': save_results_copy,
                            'structure': structure,
                            'objective': objective if save_objective else None,
                            'model_name': updated_model_name,
                            'timestamp': timestamp,
                            'energy': energy
                        }
                        with open(combined_filename, 'wb') as f:
                            pickle.dump(combined_data, f)
                        print(f"Saved combined results and structure to {combined_filename}")
                    except Exception as e:
                        print(f"Error saving combined data: {str(e)}")
                    
                    # Additionally, save MCMC samples as numpy array if they exist
                    if results['mcmc_samples'] is not None:
                        mcmc_filename = os.path.join(save_dir, f"{updated_model_name}_mcmc_samples.npy")
                        try:
                            np.save(mcmc_filename, results['mcmc_samples'])
                            print(f"Saved MCMC samples to {mcmc_filename}")
                        except Exception as e:
                            print(f"Error saving MCMC samples: {str(e)}")
            
            # Store the results in the dictionary
            results_dict[energy] = results
            
            # Update the model name in the objective
            objective.model.name = updated_model_name
            
        except Exception as e:
            print(f"Error during fitting for energy {energy} eV: {str(e)}")
            # Store minimal information in the results dictionary
            results_dict[energy] = {
                'error': str(e),
                'timestamp': timestamp,
                'structure': structure,
                'objective': objective,
                'initial_chi_squared': results['initial_chi_squared']
            }
    
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
    
    print(f"Completed fitting for {len(results_dict)} models.")
    
    return results_dict, results_log


def extract_results_from_objectives(results_dict, energy_list=None):
    """
    Extract results directly from objectives in the results dictionary.
    
    Args:
        results_dict (dict): Dictionary mapping energy values to fitting results
        energy_list (list, optional): List of energies to process. If None, process all energies.
        
    Returns:
        DataFrame: DataFrame with fitting results
    """
    import pandas as pd
    from datetime import datetime
    
    # Initialize an empty list to store rows
    rows = []
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Use all energies if not specified
    if energy_list is None:
        energy_list = list(results_dict.keys())
    
    # Process each energy
    for energy in energy_list:
        if energy not in results_dict:
            print(f"No results for energy {energy}")
            continue
            
        result = results_dict[energy]
        
        if 'objective' not in result:
            print(f"No objective for energy {energy}")
            continue
            
        objective = result['objective']
        model_name = getattr(objective.model, 'name', f"Model_{energy}eV")
        
        # Try to get chi-squared directly from the objective
        try:
            gof = objective.chisqr()
            print(f"Energy {energy}: Chi-squared = {gof:.4f}")
        except Exception as e:
            print(f"Could not get chi-squared for energy {energy}: {str(e)}")
            gof = None
        
        # Process parameters directly from the objective
        for param in objective.parameters.flattened():
            # Get parameter name and value
            param_name = param.name
            value = param.value
            stderr = getattr(param, 'stderr', None)
            vary = getattr(param, 'vary', False)
            
            # Handle bounds
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
    
    # Create DataFrame
    if rows:
        return pd.DataFrame(rows)
    else:
        print("No rows created - check if objectives have valid parameters")
        return pd.DataFrame()
    

def plot_energy_dependent_sld(results_df, energy_list, material_name, 
                            bound_threshold=0.02, figsize=(12, 10), 
                            save_path=None):
    """
    Plot real and imaginary SLD components as a function of energy.
    
    Args:
        results_df (DataFrame): DataFrame containing fitting results
        energy_list (list): List of energies to include in the plot
        material_name (str): Name of the material to plot
        bound_threshold (float): Threshold for highlighting values near bounds (as a fraction)
        figsize (tuple): Figure size as (width, height)
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, axes) - matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Rectangle
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Colors
    normal_color = 'blue'
    near_bound_color = 'red'
    bound_color = 'gray'
    
    # Extract data for this material
    real_sld_data = []
    imag_sld_data = []
    
    for energy in energy_list:
        # Find models for this energy
        energy_models = results_df[results_df['model_name'].str.contains(f"_{energy}_")]
        
        if energy_models.empty:
            continue
        
        # Extract real SLD component
        real_sld_params = energy_models[energy_models['parameter'].str.contains(f"{material_name} - sld")]
        if not real_sld_params.empty:
            value = real_sld_params['value'].iloc[0]
            stderr = real_sld_params['stderr'].iloc[0] if pd.notnull(real_sld_params['stderr'].iloc[0]) else 0
            bound_low = real_sld_params['bound_low'].iloc[0] if pd.notnull(real_sld_params['bound_low'].iloc[0]) else None
            bound_high = real_sld_params['bound_high'].iloc[0] if pd.notnull(real_sld_params['bound_high'].iloc[0]) else None
            
            # Check if value is near bounds
            near_bound = False
            if bound_low is not None and bound_high is not None:
                bound_range = bound_high - bound_low
                near_bound = (abs(value - bound_low) < bound_threshold * bound_range or 
                             abs(bound_high - value) < bound_threshold * bound_range)
            
            real_sld_data.append({
                'energy': energy,
                'value': value,
                'stderr': stderr,
                'bound_low': bound_low,
                'bound_high': bound_high,
                'near_bound': near_bound
            })
        
        # Extract imaginary SLD component
        imag_sld_params = energy_models[energy_models['parameter'].str.contains(f"{material_name} - isld")]
        if not imag_sld_params.empty:
            value = imag_sld_params['value'].iloc[0]
            stderr = imag_sld_params['stderr'].iloc[0] if pd.notnull(imag_sld_params['stderr'].iloc[0]) else 0
            bound_low = imag_sld_params['bound_low'].iloc[0] if pd.notnull(imag_sld_params['bound_low'].iloc[0]) else None
            bound_high = imag_sld_params['bound_high'].iloc[0] if pd.notnull(imag_sld_params['bound_high'].iloc[0]) else None
            
            # Check if value is near bounds
            near_bound = False
            if bound_low is not None and bound_high is not None:
                bound_range = bound_high - bound_low
                near_bound = (abs(value - bound_low) < bound_threshold * bound_range or 
                             abs(bound_high - value) < bound_threshold * bound_range)
            
            imag_sld_data.append({
                'energy': energy,
                'value': value,
                'stderr': stderr,
                'bound_low': bound_low,
                'bound_high': bound_high,
                'near_bound': near_bound
            })
    
    # Sort by energy
    real_sld_data.sort(key=lambda x: x['energy'])
    imag_sld_data.sort(key=lambda x: x['energy'])
    
    # Plot real SLD components
    ax = axes[0]
    
    # Connect data points with lines
    if real_sld_data:
        x_values = [data['energy'] for data in real_sld_data]
        y_values = [data['value'] for data in real_sld_data]
        ax.plot(x_values, y_values, '-', color=normal_color, alpha=0.7)
    
    # Plot each point with error bars and bounds separately
    for data in real_sld_data:
        # Plot bounds as vertical error bars if available
        if data['bound_low'] is not None and data['bound_high'] is not None:
            # Plot vertical range with slightly wider horizontal lines
            bound_width = 0.03 * (max(energy_list) - min(energy_list))  # 3% of x-axis range
            
            # Draw bounds as vertical line
            ax.plot([data['energy'], data['energy']], 
                   [data['bound_low'], data['bound_high']], 
                   '-', color=bound_color, alpha=0.5, linewidth=3)
            
            # Draw horizontal caps
            ax.plot([data['energy'] - bound_width, data['energy'] + bound_width], 
                   [data['bound_low'], data['bound_low']], 
                   '-', color=bound_color, alpha=0.5, linewidth=2)
            ax.plot([data['energy'] - bound_width, data['energy'] + bound_width], 
                   [data['bound_high'], data['bound_high']], 
                   '-', color=bound_color, alpha=0.5, linewidth=2)
        
        # Plot the data point with error bar
        color = near_bound_color if data['near_bound'] else normal_color
        ax.errorbar(
            data['energy'], 
            data['value'], 
            yerr=data['stderr'], 
            fmt='o', 
            color=color, 
            markersize=8, 
            capsize=5
        )
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Real SLD (10⁻⁶ Å⁻²)')
    ax.set_title(f'Real SLD vs Energy for {material_name}')
    ax.grid(True, alpha=0.3)
    
    # Plot imaginary SLD components
    ax = axes[1]
    
    # Connect data points with lines
    if imag_sld_data:
        x_values = [data['energy'] for data in imag_sld_data]
        y_values = [data['value'] for data in imag_sld_data]
        ax.plot(x_values, y_values, '-', color=normal_color, alpha=0.7)
    
    # Plot each point with error bars and bounds separately
    for data in imag_sld_data:
        # Plot bounds as vertical error bars if available
        if data['bound_low'] is not None and data['bound_high'] is not None:
            # Plot vertical range with slightly wider horizontal lines
            bound_width = 0.03 * (max(energy_list) - min(energy_list))  # 3% of x-axis range
            
            # Draw bounds as vertical line
            ax.plot([data['energy'], data['energy']], 
                   [data['bound_low'], data['bound_high']], 
                   '-', color=bound_color, alpha=0.5, linewidth=3)
            
            # Draw horizontal caps
            ax.plot([data['energy'] - bound_width, data['energy'] + bound_width], 
                   [data['bound_low'], data['bound_low']], 
                   '-', color=bound_color, alpha=0.5, linewidth=2)
            ax.plot([data['energy'] - bound_width, data['energy'] + bound_width], 
                   [data['bound_high'], data['bound_high']], 
                   '-', color=bound_color, alpha=0.5, linewidth=2)
        
        # Plot the data point with error bar
        color = near_bound_color if data['near_bound'] else normal_color
        ax.errorbar(
            data['energy'], 
            data['value'], 
            yerr=data['stderr'], 
            fmt='o', 
            color=color, 
            markersize=8, 
            capsize=5
        )
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Imaginary SLD (10⁻⁶ Å⁻²)')
    ax.set_title(f'Imaginary SLD vs Energy for {material_name}')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    normal_patch = Rectangle((0, 0), 1, 1, color=normal_color)
    near_bound_patch = Rectangle((0, 0), 1, 1, color=near_bound_color)
    bound_patch = Rectangle((0, 0), 1, 1, color=bound_color, alpha=0.5)
    
    for ax in axes:
        ax.legend(
            [normal_patch, near_bound_patch, bound_patch],
            ['Normal', f'Near Bounds (within {bound_threshold*100}%)', 'Bound Range'],
            loc='best'
        )
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def visualize_fit_results(results_dict, structures_dict, energies=None, 
                         shade_start=None, profile_shift=-20, xlim=None, 
                         fig_size_w=16, colors=None, save_path=None):
    """
    Visualize fitted reflectometry models for specific energies.
    
    Args:
        results_dict (dict): Dictionary mapping energy values to fitting results
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energies (list or float, optional): Energy or list of energies to visualize.
                                         If None, visualizes all available energies.
        shade_start (list, optional): List of starting positions for layer shading
        profile_shift (float): Shift applied to depth profiles
        xlim (list, optional): Custom x-axis limits for SLD plots as [min, max]
        fig_size_w (float): Width of the figure
        colors (list, optional): List of colors for layer shading
        save_path (str, optional): Path to save the figure
        
    Returns:
        tuple: (fig, axes) - matplotlib figure and axes objects
    """
    # Extract updated objectives from the results
    objectives_dict = {energy: results['objective'] 
                      for energy, results in results_dict.items() 
                      if 'objective' in results}
    
    # Get chi-squared values for each energy
    chi_squared_values = {energy: results['optimized_chi_squared'] 
                         for energy, results in results_dict.items() 
                         if 'optimized_chi_squared' in results}
    
    # Call the existing visualization function
    fig, axes = visualize_batch_models(
        objectives_dict=objectives_dict,
        structures_dict=structures_dict,
        energies=energies,
        shade_start=shade_start,
        profile_shift=profile_shift,
        xlim=xlim,
        fig_size_w=fig_size_w,
        colors=colors
    )
    
    # Add chi-squared values to the plot titles
    if fig is not None:
        num_plots = len(objectives_dict) if energies is None else len(energies)
        
        if num_plots == 1:
            # Single plot case
            energy = list(objectives_dict.keys())[0] if energies is None else energies[0]
            if energy in chi_squared_values:
                axes[0].set_title(f"Reflectivity - {energy} eV (χ²: {chi_squared_values[energy]:.4f})")
        else:
            # Multiple plots case
            plot_energies = list(objectives_dict.keys()) if energies is None else energies
            for i, energy in enumerate(plot_energies):
                if i < axes.shape[1] and energy in chi_squared_values:
                    axes[0, i].set_title(f"Reflectivity - {energy} eV (χ²: {chi_squared_values[energy]:.4f})")
    
    # Save the figure if a path is provided
    if save_path and fig is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_energy_dependent_parameters(results_df, energy_list, material_name, 
                            bound_threshold=0.02, figsize=(14, 16), 
                            save_path=None, parameter_types=None, debug=False, xlim=None):
    """
    Plot SLD components, thickness, and roughness as a function of energy.
    
    Args:
        results_df (DataFrame): DataFrame containing fitting results
        energy_list (list): List of energies to include in the plot
        material_name (str): Name of the material to plot
        bound_threshold (float): Threshold for highlighting values near bounds (as a fraction)
        figsize (tuple): Figure size as (width, height)
        save_path (str, optional): Path to save the figure
        parameter_types (list, optional): List of parameter types to plot
                                        Options: 'sld', 'isld', 'thick', 'rough'
                                        Default: All parameter types
        debug (bool, optional): If True, print debug information about parameter matching
        
    Returns:
        tuple: (fig, axes) - matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Rectangle
    
    # Determine which parameters to plot
    if parameter_types is None:
        parameter_types = ['sld', 'isld', 'thick', 'rough', 'gof']
        
    # Calculate number of subplots needed
    n_plots = len(parameter_types)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    # If only one parameter type, make axes a list
    if n_plots == 1:
        axes = [axes]
    
    # Colors
    normal_color = 'blue'
    near_bound_color = 'red'
    bound_color = 'gray'
    
    # Parameter type to search string mapping - using exact patterns from analysis
    param_mapping = {
        'sld': f"{material_name} - sld",
        'isld': f"{material_name} - isld",
        'thick': f"{material_name} - thick",
        'rough': f"{material_name} - rough",
        'gof': None  # Special case, handled differently
    }
    
    # Parameter type to title mapping
    title_mapping = {
        'sld': f'Real SLD vs Energy for {material_name}',
        'isld': f'Imaginary SLD vs Energy for {material_name}',
        'thick': f'Thickness vs Energy for {material_name}',
        'rough': f'Roughness vs Energy for {material_name}',
        'gof': 'Goodness of Fit vs Energy'
    }
    
    # Parameter type to y-axis label mapping
    ylabel_mapping = {
        'sld': 'Real SLD (10⁻⁶ Å⁻²)',
        'isld': 'Imaginary SLD (10⁻⁶ Å⁻²)',
        'thick': 'Thickness (Å)',
        'rough': 'Roughness (Å)',
        'gof': 'Chi-squared'
    }
    
    # Process each parameter type
    for i, param_type in enumerate(parameter_types):
        # Get current axis
        ax = axes[i]
        
        # Extract data for this parameter
        param_data = []
        
        for energy in energy_list:
            # Find models for this energy
            energy_models = results_df[results_df['model_name'].str.contains(f"_{energy}_")]
            
            if energy_models.empty:
                continue
            
            # Special handling for goodness of fit
            if param_type == 'gof':
                # Get goodness of fit values
                gof_values = energy_models['goodness_of_fit'].dropna()
                
                if not gof_values.empty:
                    # Use the first (or only) value
                    value = gof_values.iloc[0]
                    
                    param_data.append({
                        'energy': energy,
                        'value': value,
                        'stderr': 0,  # No error for GOF
                        'bound_low': None,
                        'bound_high': None,
                        'near_bound': False
                    })
                    
                    if debug:
                        print(f"Found GOF for energy {energy}: {value}")
                
                continue  # Skip the rest of the loop for GOF
            
            # Extract parameter values
            param_string = param_mapping[param_type]
            
            # Handle both string and list patterns for parameter matching
            if isinstance(param_string, list):
                # Try each possible pattern until we find a match
                param_values = pd.DataFrame()
                for pattern in param_string:
                    temp_values = energy_models[energy_models['parameter'].str.contains(pattern, regex=True)]
                    if not temp_values.empty:
                        param_values = temp_values
                        if debug:
                            print(f"Found match for {param_type} using pattern '{pattern}'")
                            print(f"Matching parameters: {param_values['parameter'].tolist()}")
                        break
                    elif debug:
                        print(f"No match found for pattern '{pattern}' for {param_type}")
            else:
                # Single pattern case
                param_values = energy_models[energy_models['parameter'].str.contains(param_string)]
                if debug:
                    if param_values.empty:
                        print(f"No match found for pattern '{param_string}' for {param_type}")
                    else:
                        print(f"Found match for {param_type} using pattern '{param_string}'")
                        print(f"Matching parameters: {param_values['parameter'].tolist()}")
            
            # Debug: Show all parameters for this energy
            if debug and param_values.empty:
                print(f"\nAll available parameters for energy {energy}:")
                for param in energy_models['parameter'].unique():
                    print(f"  - {param}")
            
            if not param_values.empty:
                value = param_values['value'].iloc[0]
                stderr = param_values['stderr'].iloc[0] if pd.notnull(param_values['stderr'].iloc[0]) else 0
                bound_low = param_values['bound_low'].iloc[0] if pd.notnull(param_values['bound_low'].iloc[0]) else None
                bound_high = param_values['bound_high'].iloc[0] if pd.notnull(param_values['bound_high'].iloc[0]) else None
                
                # Check if value is near bounds
                near_bound = False
                if bound_low is not None and bound_high is not None:
                    bound_range = bound_high - bound_low
                    near_bound = (abs(value - bound_low) < bound_threshold * bound_range or 
                                 abs(bound_high - value) < bound_threshold * bound_range)
                
                param_data.append({
                    'energy': energy,
                    'value': value,
                    'stderr': stderr,
                    'bound_low': bound_low,
                    'bound_high': bound_high,
                    'near_bound': near_bound
                })
        
        # Sort by energy
        param_data.sort(key=lambda x: x['energy'])
        
        # Skip plotting if no data found
        if not param_data:
            ax.text(0.5, 0.5, f'No data found for {param_type}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(title_mapping[param_type])
            continue
        
        # Connect data points with lines
        x_values = [data['energy'] for data in param_data]
        y_values = [data['value'] for data in param_data]
        
        # Use logarithmic scale for GOF
        if param_type == 'gof':
            ax.semilogy(x_values, y_values, '-', color=normal_color, alpha=0.7)
            # Add horizontal grid lines for log scale
            ax.grid(True, which='both', linestyle='--', alpha=0.3)
        else:
            ax.plot(x_values, y_values, '-', color=normal_color, alpha=0.7)
        
        # Plot each point with error bars and bounds separately
        for data in param_data:
            # Skip bounds visualization for GOF
            if param_type == 'gof':
                ax.plot(data['energy'], data['value'], 'o', color=normal_color, markersize=8)
                continue
            # Plot bounds as vertical error bars if available
            if data['bound_low'] is not None and data['bound_high'] is not None:
                # Plot vertical range with slightly wider horizontal lines
                bound_width = 0.03 * (max(energy_list) - min(energy_list))  # 3% of x-axis range
                
                # Draw bounds as vertical line
                ax.plot([data['energy'], data['energy']], 
                       [data['bound_low'], data['bound_high']], 
                       '-', color=bound_color, alpha=0.5, linewidth=3)
                
                # Draw horizontal caps
                ax.plot([data['energy'] - bound_width, data['energy'] + bound_width], 
                       [data['bound_low'], data['bound_low']], 
                       '-', color=bound_color, alpha=0.5, linewidth=2)
                ax.plot([data['energy'] - bound_width, data['energy'] + bound_width], 
                       [data['bound_high'], data['bound_high']], 
                       '-', color=bound_color, alpha=0.5, linewidth=2)
            
            # Plot the data point with error bar
            color = near_bound_color if data['near_bound'] else normal_color
            ax.errorbar(
                data['energy'], 
                data['value'], 
                yerr=data['stderr'], 
                fmt='o', 
                color=color, 
                markersize=8, 
                capsize=5
            )
        
        # Set labels and title
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel(ylabel_mapping[param_type])
        ax.set_title(title_mapping[param_type])
        
        # Set xlim if provided
        if xlim is not None:
            ax.set_xlim(xlim)
            
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend only for parameter types with bounds
        if param_type != 'gof':
            normal_patch = Rectangle((0, 0), 1, 1, color=normal_color)
            near_bound_patch = Rectangle((0, 0), 1, 1, color=near_bound_color)
            bound_patch = Rectangle((0, 0), 1, 1, color=bound_color, alpha=0.5)
            
            ax.legend(
                [normal_patch, near_bound_patch, bound_patch],
                ['Normal', f'Near Bounds (within {bound_threshold*100}%)', 'Bound Range'],
                loc='best'
            )
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes



def compare_sld_with_nexafs(results_df, energy_list, material_name, 
                         nexafs_data_list=None, nexafs_labels=None,
                         bound_threshold=0.02, figsize=(12, 10), 
                         save_path=None, debug=False, xlim=None):
    """
    Compare fitted SLD components with NEXAFS data.
    
    Args:
        results_df (DataFrame): DataFrame containing fitting results
        energy_list (list): List of energies to include in the plot
        material_name (str): Name of the material to plot
        nexafs_data_list (list, optional): List of NEXAFS data arrays, each with columns [Energy, Real, Imaginary]
        nexafs_labels (list, optional): List of labels for each NEXAFS dataset
        bound_threshold (float): Threshold for highlighting values near bounds (as a fraction)
        figsize (tuple): Figure size as (width, height)
        save_path (str, optional): Path to save the figure
        debug (bool): Whether to print debug information
        xlim (tuple, optional): Custom x-axis limits as (min, max) energy values.
                               If None, will use the range of energy_list.
        
    Returns:
        tuple: (fig, axes) - matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Rectangle
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Colors for fitted data
    fit_color = 'blue'
    near_bound_color = 'red'
    bound_color = 'gray'
    
    # Colors for NEXAFS data - use a different colormap for distinction
    nexafs_colors = plt.cm.Set2.colors
    
    # Parameter type to search string mapping
    param_mapping = {
        'sld': f"{material_name} - sld",
        'isld': f"{material_name} - isld"
    }
    
    # Title mapping
    title_mapping = {
        'sld': f'Real SLD vs Energy for {material_name}',
        'isld': f'Imaginary SLD vs Energy for {material_name}'
    }
    
    # Y-axis label mapping
    ylabel_mapping = {
        'sld': 'Real SLD (10⁻⁶ Å⁻²)',
        'isld': 'Imaginary SLD (10⁻⁶ Å⁻²)'
    }
    
    # Determine xlim (energy range) for the plots
    if xlim is None and energy_list:
        # Use the range of energy_list with a small margin (2%)
        e_min, e_max = min(energy_list), max(energy_list)
        margin = 0.02 * (e_max - e_min)
        auto_xlim = (e_min - margin, e_max + margin)
    else:
        auto_xlim = xlim  # Use provided xlim or None
    
    # Process each SLD component (real and imaginary)
    for i, param_type in enumerate(['sld', 'isld']):
        # Get current axis
        ax = axes[i]
        
        # Extract data for this parameter from fitting results
        param_data = []
        
        for energy in energy_list:
            # Find models for this energy
            energy_models = results_df[results_df['model_name'].str.contains(f"_{energy}_")]
            
            if energy_models.empty:
                if debug:
                    print(f"No model found for energy {energy} eV")
                continue
            
            # Extract parameter values
            param_string = param_mapping[param_type]
            param_values = energy_models[energy_models['parameter'].str.contains(param_string)]
            
            if param_values.empty:
                if debug:
                    print(f"No {param_type} parameter found for energy {energy} eV")
                    print(f"Available parameters: {energy_models['parameter'].unique()}")
                continue
            
            value = param_values['value'].iloc[0]
            stderr = param_values['stderr'].iloc[0] if pd.notnull(param_values['stderr'].iloc[0]) else 0
            bound_low = param_values['bound_low'].iloc[0] if pd.notnull(param_values['bound_low'].iloc[0]) else None
            bound_high = param_values['bound_high'].iloc[0] if pd.notnull(param_values['bound_high'].iloc[0]) else None
            
            # Check if value is near bounds
            near_bound = False
            if bound_low is not None and bound_high is not None:
                bound_range = bound_high - bound_low
                near_bound = (abs(value - bound_low) < bound_threshold * bound_range or 
                             abs(bound_high - value) < bound_threshold * bound_range)
            
            param_data.append({
                'energy': energy,
                'value': value,
                'stderr': stderr,
                'bound_low': bound_low,
                'bound_high': bound_high,
                'near_bound': near_bound
            })
        
        # Sort by energy
        param_data.sort(key=lambda x: x['energy'])
        
        # Plot fitted SLD data if available
        if param_data:
            # Connect data points with lines
            x_values = [data['energy'] for data in param_data]
            y_values = [data['value'] for data in param_data]
            ax.plot(x_values, y_values, '-', color=fit_color, alpha=0.7, 
                   label=f'Fitted {param_type.upper()}')
            
            # Plot each point with error bars and bounds
            for data in param_data:
                # Plot bounds as vertical error bars if available
                if data['bound_low'] is not None and data['bound_high'] is not None:
                    # Plot vertical range with slightly wider horizontal lines
                    bound_width = 0.03 * (max(energy_list) - min(energy_list))  # 3% of x-axis range
                    
                    # Draw bounds as vertical line
                    ax.plot([data['energy'], data['energy']], 
                           [data['bound_low'], data['bound_high']], 
                           '-', color=bound_color, alpha=0.5, linewidth=3)
                    
                    # Draw horizontal caps
                    ax.plot([data['energy'] - bound_width, data['energy'] + bound_width], 
                           [data['bound_low'], data['bound_low']], 
                           '-', color=bound_color, alpha=0.5, linewidth=2)
                    ax.plot([data['energy'] - bound_width, data['energy'] + bound_width], 
                           [data['bound_high'], data['bound_high']], 
                           '-', color=bound_color, alpha=0.5, linewidth=2)
                
                # Plot the data point with error bar
                color = near_bound_color if data['near_bound'] else fit_color
                ax.errorbar(
                    data['energy'], 
                    data['value'], 
                    yerr=data['stderr'], 
                    fmt='o', 
                    color=color, 
                    markersize=8, 
                    capsize=5
                )
        else:
            ax.text(0.5, 0.5, f'No fitted {param_type} data found', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        
        # Plot NEXAFS data if provided
        if nexafs_data_list:
            for j, nexafs_data in enumerate(nexafs_data_list):
                if nexafs_data is None:
                    continue
                    
                # Determine which column to use (1=real, 2=imaginary)
                col_index = 1 if param_type == 'sld' else 2
                
                # Get label for this NEXAFS dataset
                if nexafs_labels and j < len(nexafs_labels):
                    label = nexafs_labels[j]
                else:
                    label = f'NEXAFS {j+1}'
                
                # Plot NEXAFS data
                ax.plot(
                    nexafs_data[:, 0],  # Energy
                    nexafs_data[:, col_index],  # Real or Imaginary SLD
                    '--',  # Dashed line
                    color=nexafs_colors[j % len(nexafs_colors)],
                    linewidth=2,
                    label=label
                )
        
        # Set labels and title
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel(ylabel_mapping[param_type])
        ax.set_title(title_mapping[param_type])
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='best')
        
        # Set x-axis limits if specified
        if auto_xlim:
            ax.set_xlim(auto_xlim)
    
    # Add a common legend for fitted data points
    if param_data:
        normal_patch = Rectangle((0, 0), 1, 1, color=fit_color)
        near_bound_patch = Rectangle((0, 0), 1, 1, color=near_bound_color)
        bound_patch = Rectangle((0, 0), 1, 1, color=bound_color, alpha=0.5)
        
        fig.legend(
            [normal_patch, near_bound_patch, bound_patch],
            ['Normal Fit', f'Near Bounds (within {bound_threshold*100}%)', 'Bound Range'],
            loc='lower center',
            ncol=3,
            bbox_to_anchor=(0.5, 0.02)
        )
        
    # Add energy range information
    if auto_xlim:
        energy_range_text = f"Energy range: {auto_xlim[0]:.1f} - {auto_xlim[1]:.1f} eV"
        fig.text(0.99, 0.01, energy_range_text, fontsize=8, 
                horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for the common legend
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def batch_fit_with_bounds_iteration(objectives_dict, structures_dict, energy_list=None, 
                             optimization_method='differential_evolution', 
                             opt_workers=8, opt_popsize=20, burn_samples=5, 
                             production_samples=5, prod_steps=1, pool=16,
                             results_log=None, log_mcmc_stats=True,
                             save_dir=None, save_objective=False, save_results=False,
                             results_log_file=None, save_log_in_save_dir=False,
                             bound_threshold=0.02, max_iterations=3, bound_expansion_factor=1.5,
                             verbose=True):
    """
    Run fitting procedure on selected reflectometry models with automatic iteration of models
    where SLD parameters end up near bounds.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to fit. If None, fit all available models.
        optimization_method (str): Optimization method to use
        opt_workers (int): Number of workers for parallel optimization
        opt_popsize (int): Population size for genetic algorithms
        burn_samples (int): Number of burn-in samples to discard (in thousands)
        production_samples (int): Number of production samples to keep (in thousands)
        prod_steps (int): Number of steps between stored samples
        pool (int): Number of parallel processes for MCMC sampling
        results_log (DataFrame, optional): Existing results log DataFrame to append to
        log_mcmc_stats (bool): Whether to add MCMC statistics to the log
        save_dir (str, optional): Directory to save objective and results
        save_objective (bool): Whether to save the objective function
        save_results (bool): Whether to save the results dictionary
        results_log_file (str, optional): Filename to load/save the results log DataFrame
        save_log_in_save_dir (bool): If True, save the log file in save_dir
        bound_threshold (float): Threshold for detecting parameters near bounds (as fraction of bound range)
        max_iterations (int): Maximum number of iterations for each model
        bound_expansion_factor (float): Factor to expand bounds by when a parameter is near bounds
        verbose (bool): Whether to print detailed progress information
        
    Returns:
        tuple: (results_dict, updated_results_df, iterations_dict)
            - results_dict: Dictionary mapping energy values to fitting results
            - updated_results_df: Combined DataFrame of all fitting results
            - iterations_dict: Dictionary tracking iterations performed for each energy
    """
    import os
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import copy
    import pickle
    import re
    
    # Dictionary to store fitting results
    results_dict = {}
    
    # Dictionary to track iterations for each energy
    iterations_dict = {}
    
    # If no energy list is provided, use all available energies
    if energy_list is None:
        energy_list = sorted(list(objectives_dict.keys()))
    
    # Filter to ensure we only process energies that have both objectives and structures
    valid_energies = []
    for energy in energy_list:
        if energy in objectives_dict and energy in structures_dict:
            valid_energies.append(energy)
        else:
            if verbose:
                print(f"Warning: Missing objective or structure for energy {energy} eV. Skipping.")
    
    if not valid_energies:
        if verbose:
            print("No valid energies to fit.")
        return {}, results_log, {}
    
    if verbose:
        print(f"Fitting {len(valid_energies)} models: {valid_energies}")
    
    # Determine the actual log file path
    actual_log_file = results_log_file
    if save_dir is not None and save_log_in_save_dir:
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            if verbose:
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
                if verbose:
                    print(f"Loading existing results log from {actual_log_file}")
                results_log = pd.read_csv(actual_log_file)
            except Exception as e:
                if verbose:
                    print(f"Error loading results log: {str(e)}")
                    print("Initializing new results log")
                results_log = pd.DataFrame(columns=[
                    'timestamp', 'model_name', 'goodness_of_fit', 
                    'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary'
                ])
    
    # Initialize results_log if not provided and not loaded from file
    if results_log is None:
        if verbose:
            print("Initializing new results log")
        results_log = pd.DataFrame(columns=[
            'timestamp', 'model_name', 'goodness_of_fit', 
            'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary'
        ])
    
    # Process each energy
    for energy in valid_energies:
        # Get the original objective and structure
        original_objective = objectives_dict[energy]
        structure = structures_dict[energy]
        
        # Extract initial model name
        model_name = getattr(original_objective.model, 'name', f"Model_{energy}eV")
        if verbose:
            print(f"\nFitting model: {model_name}")
        
        # Initialize iteration tracking
        iterations_dict[energy] = {
            'iterations': 0,
            'expanded_parameters': [],
            'final_iteration': 0,
            'iteration_gof': {}
        }
        
        # Create a working copy of the objective
        current_objective = copy.deepcopy(original_objective)
        
        # Iterate up to max_iterations
        for iteration in range(max_iterations):
            # Check if we're in a subsequent iteration
            is_iteration = iteration > 0
            
            # Update the model name to include iteration number if needed
            if is_iteration:
                # Extract base model name without previous iteration suffix
                base_model_name = re.sub(r'_iter\d+$', '', model_name)
                current_model_name = f"{base_model_name}_iter{iteration}"
            else:
                current_model_name = model_name
            
            # Update the model name in the objective
            current_objective.model.name = current_model_name
            
            if verbose:
                print(f"\nIteration {iteration} for energy {energy} eV")
                if is_iteration:
                    print(f"Using expanded bounds for parameters: {iterations_dict[energy]['expanded_parameters']}")
            
            # Generate timestamp for logs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Initialize results dictionary
            results = {
                'objective': current_objective,
                'initial_chi_squared': current_objective.chisqr(),
                'optimized_parameters': None,
                'optimized_chi_squared': None,
                'mcmc_samples': None,
                'mcmc_stats': None,
                'timestamp': timestamp,
                'structure': structure,
                'iteration': iteration
            }
            
            try:
                # Create fitter
                from refnx.analysis import CurveFitter
                fitter = CurveFitter(current_objective)
                
                # Run optimization
                if verbose:
                    print(f"Starting optimization using {optimization_method}...")
                if optimization_method == 'differential_evolution':
                    fitter.fit(optimization_method, workers=opt_workers, popsize=opt_popsize)
                else:
                    fitter.fit(optimization_method)
                
                # Store optimization results
                results['optimized_parameters'] = current_objective.parameters.pvals.copy()
                results['optimized_chi_squared'] = current_objective.chisqr()
                
                if verbose:
                    print(f"Optimization complete. Chi-squared improved from {results['initial_chi_squared']:.4f} to {results['optimized_chi_squared']:.4f}")
                
                # Run burn-in MCMC samples
                if burn_samples > 0:
                    if verbose:
                        print(f"Running {burn_samples}k burn-in MCMC samples...")
                    fitter.sample(burn_samples, pool=pool)
                    if verbose:
                        print("Burn-in complete. Resetting chain...")
                    fitter.reset()
                
                # Run production MCMC samples
                if production_samples > 0:
                    if verbose:
                        print(f"Running {production_samples}k production MCMC samples with {prod_steps} steps between stored samples...")
                    results['mcmc_samples'] = fitter.sample(production_samples, prod_steps, pool=pool)
                    
                    # Calculate statistics from MCMC chain
                    try:
                        if verbose:
                            print("Calculating parameter statistics from MCMC chain...")
                        results['mcmc_stats'] = {}
                        
                        # Process parameter statistics
                        for param in current_objective.parameters.flattened():
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
                                if chain_index >= 0 and results['mcmc_samples'] is not None:
                                    # Calculate statistics
                                    chain_values = results['mcmc_samples'][:, chain_index]
                                    param_stats['median'] = np.median(chain_values)
                                    param_stats['mean'] = np.mean(chain_values)
                                    param_stats['std'] = np.std(chain_values)
                                    
                                    # Calculate percentiles
                                    for percentile in [2.5, 16, 50, 84, 97.5]:
                                        param_stats['percentiles'][percentile] = np.percentile(chain_values, percentile)
                                
                                results['mcmc_stats'][param.name] = param_stats
                        
                        # Print a summary of key parameters
                        if verbose:
                            print("\nParameter summary from MCMC:")
                            for name, stats in results['mcmc_stats'].items():
                                if 'median' in stats and stats['median'] is not None:
                                    print(f"  {name}: {stats['median']:.6g} +{stats['percentiles'][84] - stats['median']:.6g} -{stats['median'] - stats['percentiles'][16]:.6g}")
                    
                    except Exception as e:
                        if verbose:
                            print(f"Error calculating MCMC statistics: {str(e)}")
                
                # Log the results
                if verbose:
                    print(f"Logging results for model {current_model_name}")
                
                # Log the optimized values
                from Model_Setup import log_fitting_results
                results_log, updated_model_name = log_fitting_results(current_objective, current_model_name, results_log)
                
                # If MCMC was performed and we want to log those stats, create a second entry
                if log_mcmc_stats and results['mcmc_stats'] is not None:
                    if verbose:
                        print("Adding MCMC statistics to the log...")
                    
                    # Create a temporary copy of the objective to store MCMC medians
                    mcmc_objective = copy.deepcopy(current_objective)
                    
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
                    mcmc_model_name = f"{updated_model_name}_MCMC"
                    results_log, _ = log_fitting_results(mcmc_objective, mcmc_model_name, results_log)
                
                # Save the objective and/or results if requested
                if save_dir is not None and (save_objective or save_results):
                    # Create the directory if it doesn't exist
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                        if verbose:
                            print(f"Created directory: {save_dir}")
                    
                    # Save the objective if requested
                    if save_objective:
                        objective_filename = os.path.join(save_dir, f"{updated_model_name}_objective.pkl")
                        try:
                            with open(objective_filename, 'wb') as f:
                                pickle.dump(current_objective, f)
                            if verbose:
                                print(f"Saved objective to {objective_filename}")
                        except Exception as e:
                            if verbose:
                                print(f"Error saving objective: {str(e)}")
                    
                    # Save the results if requested
                    if save_results:
                        # Create a copy of results without the objective (to avoid duplication if saving both)
                        save_results_copy = results.copy()
                        if 'objective' in save_results_copy and save_objective:
                            save_results_copy['objective'] = None  # Remove the objective to avoid duplication
                        
                        results_filename = os.path.join(save_dir, f"{updated_model_name}_results.pkl")
                        try:
                            with open(results_filename, 'wb') as f:
                                pickle.dump(save_results_copy, f)
                            if verbose:
                                print(f"Saved results to {results_filename}")
                        except Exception as e:
                            if verbose:
                                print(f"Error saving results: {str(e)}")
                        
                        # Save a combined file with results and structure
                        combined_filename = os.path.join(save_dir, f"{updated_model_name}_combined.pkl")
                        try:
                            combined_data = {
                                'results': save_results_copy,
                                'structure': structure,
                                'objective': current_objective if save_objective else None,
                                'model_name': updated_model_name,
                                'timestamp': timestamp,
                                'energy': energy,
                                'iteration': iteration
                            }
                            with open(combined_filename, 'wb') as f:
                                pickle.dump(combined_data, f)
                            if verbose:
                                print(f"Saved combined results and structure to {combined_filename}")
                        except Exception as e:
                            if verbose:
                                print(f"Error saving combined data: {str(e)}")
                        
                        # Additionally, save MCMC samples as numpy array if they exist
                        if results['mcmc_samples'] is not None:
                            mcmc_filename = os.path.join(save_dir, f"{updated_model_name}_mcmc_samples.npy")
                            try:
                                np.save(mcmc_filename, results['mcmc_samples'])
                                if verbose:
                                    print(f"Saved MCMC samples to {mcmc_filename}")
                            except Exception as e:
                                if verbose:
                                    print(f"Error saving MCMC samples: {str(e)}")
                
                # Store the results in the dictionary (overwrite any previous iteration)
                results_dict[energy] = results
                
                # Track goodness of fit for this iteration
                iterations_dict[energy]['iteration_gof'][iteration] = results['optimized_chi_squared']
                iterations_dict[energy]['iterations'] = iteration + 1
                iterations_dict[energy]['final_iteration'] = iteration
                
                # Check if any SLD parameters are near bounds
                near_bounds_params = []
                for param in current_objective.parameters.flattened():
                    # Only check SLD parameters that are varying
                    if param.vary and ('sld' in param.name.lower() or 'isld' in param.name.lower()):
                        # Get bounds
                        try:
                            bounds = getattr(param, 'bounds', None)
                            if bounds is not None:
                                # Extract bound values
                                if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                                    low = bounds.lb
                                    high = bounds.ub
                                elif isinstance(bounds, tuple) and len(bounds) == 2:
                                    low, high = bounds
                                else:
                                    continue  # Skip if bounds format is unknown
                                
                                # Calculate bound range and threshold
                                bound_range = high - low
                                threshold = bound_threshold * bound_range
                                
                                # Check if parameter is within threshold of either bound
                                if abs(param.value - low) < threshold or abs(high - param.value) < threshold:
                                    near_bounds_params.append(param.name)
                                    if verbose:
                                        print(f"Parameter {param.name} is near bounds: value={param.value:.6g}, bounds=[{low:.6g}, {high:.6g}]")
                        except Exception as e:
                            if verbose:
                                print(f"Error checking bounds for {param.name}: {str(e)}")
                
                # If no parameters are near bounds or we've reached max iterations, stop iterating
                if not near_bounds_params or iteration == max_iterations - 1:
                    if verbose:
                        if not near_bounds_params:
                            print(f"No SLD parameters near bounds. Stopping iteration.")
                        elif iteration == max_iterations - 1:
                            print(f"Reached maximum iterations ({max_iterations}). Stopping iteration.")
                    break
                
                # Prepare for next iteration by expanding bounds for parameters near bounds
                if verbose:
                    print(f"Expanding bounds for parameters: {near_bounds_params}")
                
                # Create a fresh copy of the objective for the next iteration
                next_objective = copy.deepcopy(original_objective)
                
                # Expand bounds for SLD parameters that are near bounds
                for param_name in near_bounds_params:
                    # Find the parameter in the next objective
                    for param in next_objective.parameters.flattened():
                        if param.name == param_name:
                            # Get current bounds
                            try:
                                bounds = getattr(param, 'bounds', None)
                                if bounds is not None:
                                    # Extract bound values
                                    if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                                        low = bounds.lb
                                        high = bounds.ub
                                    elif isinstance(bounds, tuple) and len(bounds) == 2:
                                        low, high = bounds
                                    else:
                                        continue  # Skip if bounds format is unknown
                                    
                                    # Calculate current bound range and midpoint
                                    bound_range = high - low
                                    midpoint = (high + low) / 2
                                    
                                    # Expand bounds around the midpoint
                                    expanded_range = bound_range * bound_expansion_factor
                                    new_low = midpoint - expanded_range / 2
                                    new_high = midpoint + expanded_range / 2
                                    
                                    # Check if imaginary SLD to ensure lower bound is not negative
                                    if 'isld' in param.name.lower() and new_low < 0:
                                        new_low = 0
                                    
                                    # Set the new bounds
                                    param.bounds = (new_low, new_high)
                                    
                                    if verbose:
                                        print(f"Expanded bounds for {param.name}: [{low:.6g}, {high:.6g}] -> [{new_low:.6g}, {new_high:.6g}]")
                            except Exception as e:
                                if verbose:
                                    print(f"Error expanding bounds for {param.name}: {str(e)}")
                
                # Update the current objective for the next iteration
                current_objective = next_objective
                
                # Track expanded parameters
                iterations_dict[energy]['expanded_parameters'].extend(near_bounds_params)
                
            except Exception as e:
                if verbose:
                    print(f"Error during fitting for energy {energy} eV, iteration {iteration}: {str(e)}")
                # Store minimal information in the results dictionary
                results_dict[energy] = {
                    'error': str(e),
                    'timestamp': timestamp,
                    'structure': structure,
                    'objective': current_objective,
                    'initial_chi_squared': results['initial_chi_squared'],
                    'iteration': iteration
                }
                
                # Stop iterating for this energy
                break
    
    # Save the results log if a filename was provided
    if actual_log_file is not None:
        # Create directory for results_log_file if it doesn't exist
        log_dir = os.path.dirname(actual_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            if verbose:
                print(f"Created directory for results log: {log_dir}")
        
        try:
            # Save to CSV
            results_log.to_csv(actual_log_file, index=False)
            if verbose:
                print(f"Saved results log to {actual_log_file}")
        except Exception as e:
            if verbose:
                print(f"Error saving results log: {str(e)}")
    
    if verbose:
        print(f"\nCompleted fitting for {len(results_dict)} models.")
        print("\nIteration summary:")
        for energy, info in iterations_dict.items():
            print(f"Energy {energy} eV: {info['iterations']} iterations performed")
            if info['expanded_parameters']:
                print(f"  Expanded parameters: {set(info['expanded_parameters'])}")
            print(f"  Goodness of fit by iteration: {', '.join([f'{i}: {gof:.4f}' for i, gof in info['iteration_gof'].items()])}")
    
    return results_dict, results_log, iterations_dict


def analyze_iteration_results(iterations_dict, results_log=None, energy_list=None):
    """
    Analyze the results of the iterative fitting process.
    
    Args:
        iterations_dict (dict): Dictionary of iteration information from batch_fit_with_bounds_iteration
        results_log (DataFrame, optional): DataFrame containing fitting results
        energy_list (list, optional): List of energies to analyze. If None, analyze all energies.
        
    Returns:
        dict: Dictionary with analysis results
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initialize results
    analysis = {
        'energies': {},
        'summary': {
            'total_energies': 0,
            'total_iterations': 0,
            'energies_requiring_iteration': 0,
            'avg_iterations_per_energy': 0,
            'most_expanded_parameters': []
        }
    }
    
    # Use all energies if not specified
    if energy_list is None:
        energy_list = list(iterations_dict.keys())
    
    # Filter to valid energies
    valid_energies = [e for e in energy_list if e in iterations_dict]
    
    # Initialize counters
    total_iterations = 0
    energies_with_iterations = 0
    parameter_expansion_counts = {}
    
    # Process each energy
    for energy in valid_energies:
        info = iterations_dict[energy]
        iterations = info['iterations']
        expanded_params = info['expanded_parameters']
        final_iteration = info['final_iteration']
        iteration_gof = info.get('iteration_gof', {})
        
        # Count total iterations
        total_iterations += iterations
        
        # Count energies requiring iteration
        if iterations > 1:
            energies_with_iterations += 1
        
        # Count expanded parameters
        for param in expanded_params:
            if param in parameter_expansion_counts:
                parameter_expansion_counts[param] += 1
            else:
                parameter_expansion_counts[param] = 1
        
        # Store energy-specific info
        analysis['energies'][energy] = {
            'iterations': iterations,
            'expanded_parameters': expanded_params,
            'unique_expanded_parameters': list(set(expanded_params)),
            'final_iteration': final_iteration,
            'iteration_gof': iteration_gof,
            'gof_improvement': None
        }
        
        # Calculate GOF improvement if available
        if 0 in iteration_gof and final_iteration in iteration_gof:
            initial_gof = iteration_gof[0]
            final_gof = iteration_gof[final_iteration]
            improvement = (initial_gof - final_gof) / initial_gof * 100  # As percentage
            analysis['energies'][energy]['gof_improvement'] = improvement
    
    # Get most commonly expanded parameters
    sorted_params = sorted(parameter_expansion_counts.items(), key=lambda x: x[1], reverse=True)
    most_expanded = [p[0] for p in sorted_params[:5]]  # Top 5
    
    # Set summary metrics
    analysis['summary']['total_energies'] = len(valid_energies)
    analysis['summary']['total_iterations'] = total_iterations
    analysis['summary']['energies_requiring_iteration'] = energies_with_iterations
    if len(valid_energies) > 0:
        analysis['summary']['avg_iterations_per_energy'] = total_iterations / len(valid_energies)
    analysis['summary']['most_expanded_parameters'] = most_expanded
    analysis['summary']['parameter_expansion_counts'] = parameter_expansion_counts
    
    # Create detailed analysis if results_log is provided
    if results_log is not None:
        # Add parameter statistics from the results log
        analysis['parameter_stats'] = {}
        
        # Analyze parameters for the final iteration of each energy
        for energy in valid_energies:
            final_iteration = iterations_dict[energy]['final_iteration']
            
            # Look for models for this energy and iteration
            energy_models = results_log[results_log['model_name'].str.contains(f"_{energy}_")]
            
            if final_iteration > 0:
                # Filter for the final iteration
                final_models = energy_models[energy_models['model_name'].str.contains(f"_iter{final_iteration}")]
            else:
                # Use models without iteration suffix
                final_models = energy_models[~energy_models['model_name'].str.contains(f"_iter")]
            
            # Proceed only if we have models for this energy/iteration
            if not final_models.empty:
                # Compare with initial models
                initial_models = energy_models[~energy_models['model_name'].str.contains(f"_iter")]
                
                if not initial_models.empty:
                    # Analyze parameter changes
                    param_changes = {}
                    
                    # Get initial parameter values
                    initial_params = {}
                    for _, row in initial_models.iterrows():
                        param_name = row['parameter']
                        if 'sld' in param_name.lower() or 'isld' in param_name.lower():
                            initial_params[param_name] = {
                                'value': row['value'],
                                'bound_low': row['bound_low'],
                                'bound_high': row['bound_high']
                            }
                    
                    # Get final parameter values and calculate changes
                    for _, row in final_models.iterrows():
                        param_name = row['parameter']
                        if param_name in initial_params:
                            initial_value = initial_params[param_name]['value']
                            final_value = row['value']
                            
                            # Calculate percent change
                            if initial_value != 0:
                                percent_change = (final_value - initial_value) / abs(initial_value) * 100
                            else:
                                percent_change = np.nan
                            
                            # Calculate bound expansion
                            initial_low = initial_params[param_name]['bound_low']
                            initial_high = initial_params[param_name]['bound_high']
                            final_low = row['bound_low']
                            final_high = row['bound_high']
                            
                            if initial_low is not None and initial_high is not None and final_low is not None and final_high is not None:
                                initial_range = initial_high - initial_low
                                final_range = final_high - final_low
                                
                                if initial_range > 0:
                                    bound_expansion = (final_range / initial_range - 1) * 100  # As percentage
                                else:
                                    bound_expansion = np.nan
                            else:
                                bound_expansion = np.nan
                            
                            # Store the changes
                            param_changes[param_name] = {
                                'initial_value': initial_value,
                                'final_value': final_value,
                                'percent_change': percent_change,
                                'initial_bounds': (initial_low, initial_high),
                                'final_bounds': (final_low, final_high),
                                'bound_expansion': bound_expansion
                            }
                    
                    # Store the parameter changes
                    analysis['energies'][energy]['parameter_changes'] = param_changes
    
    return analysis


def plot_iteration_analysis(analysis, figsize=(14, 14), save_path=None):
    """
    Plot the results of the iteration analysis.
    
    Args:
        analysis (dict): Analysis dictionary from analyze_iteration_results
        figsize (tuple): Figure size as (width, height)
        save_path (str, optional): Path to save the figure
    
    Returns:
        tuple: (fig, axes) - matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with subplots - increased figure size and spacing
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    
    # Extract energy list and sort
    energy_list = sorted(analysis['energies'].keys())
    
    # Plot 1: Number of iterations by energy
    ax = axes[0, 0]
    iterations = [analysis['energies'][e]['iterations'] for e in energy_list]
    ax.bar(energy_list, iterations)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Number of Iterations')
    ax.set_title('Iterations Required per Energy')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(iterations):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    # Plot 2: GOF improvement by energy
    ax = axes[0, 1]
    improvements = []
    improved_energies = []
    
    for e in energy_list:
        if 'gof_improvement' in analysis['energies'][e] and analysis['energies'][e]['gof_improvement'] is not None:
            improvements.append(analysis['energies'][e]['gof_improvement'])
            improved_energies.append(e)
    
    if improvements:
        ax.bar(improved_energies, improvements, color='green')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('GOF Improvement (%)')
        ax.set_title('Goodness of Fit Improvement')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(improvements):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    else:
        ax.text(0.5, 0.5, 'No GOF improvement data available', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes)
    
    # Plot 3: Parameter expansion counts
    ax = axes[1, 0]
    param_counts = analysis['summary']['parameter_expansion_counts']
    
    if param_counts:
        # Sort by count and get top N
        max_params = 10
        sorted_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)
        top_params = sorted_params[:max_params]
        
        # Extract names and counts
        names = [p[0] for p in top_params]
        counts = [p[1] for p in top_params]
        
        # Shorten parameter names if needed
        short_names = []
        for name in names:
            if len(name) > 20:
                short_name = name[:17] + "..."
            else:
                short_name = name
            short_names.append(short_name)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(short_names))
        ax.barh(y_pos, counts, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Number of Expansions')
        ax.set_title('Most Frequently Expanded Parameters')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels at end of bars
        for i, v in enumerate(counts):
            ax.text(v + 0.1, i, str(v), va='center')
    else:
        ax.text(0.5, 0.5, 'No parameter expansion data available', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes)
    
    # Plot 4: GOF by iteration for energies requiring multiple iterations
    ax = axes[1, 1]
    multi_iter_energies = [e for e in energy_list if analysis['energies'][e]['iterations'] > 1]
    
    if multi_iter_energies:
        max_iterations = max([analysis['energies'][e]['iterations'] for e in multi_iter_energies])
        
        # Plot GOF for each energy across iterations
        for e in multi_iter_energies:
            iteration_gof = analysis['energies'][e]['iteration_gof']
            iterations = sorted(iteration_gof.keys())
            gof_values = [iteration_gof[i] for i in iterations]
            
            # Calculate relative GOF (normalized to initial value)
            if gof_values[0] > 0:
                rel_gof = [g / gof_values[0] for g in gof_values]
                ax.plot(iterations, rel_gof, 'o-', label=f"{e} eV")
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Goodness of Fit')
        ax.set_title('GOF Improvement by Iteration')
        ax.grid(True, alpha=0.3)
        
        # Set nice x-axis ticks (0, 1, 2, ...)
        ax.set_xticks(range(max_iterations + 1))
        
        # Start y-axis at 0 if there's significant improvement
        min_rel_gof = min([min(analysis['energies'][e]['iteration_gof'].values()) / 
                         analysis['energies'][e]['iteration_gof'][0] 
                         for e in multi_iter_energies])
        if min_rel_gof < 0.5:  # If there's at least 50% improvement
            ax.set_ylim(bottom=0)
        
        if len(multi_iter_energies) > 1:
            ax.legend(loc='best')
    else:
        ax.text(0.5, 0.5, 'No energies required multiple iterations', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes)
    
    # Add summary statistics as text - use figtext with smaller font and more compact text
    summary_text = (f"Summary: {analysis['summary']['total_energies']} energies analyzed, "
                   f"{analysis['summary']['energies_requiring_iteration']} required iteration. "
                   f"Average iterations: {analysis['summary']['avg_iterations_per_energy']:.1f}")
    
    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes

