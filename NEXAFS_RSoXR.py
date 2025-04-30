### This code will be used to fit batches of RSoXR data based on NEXAFS Spectra


import os
import re
import numpy as np
from refnx.dataset import ReflectDataset
from pathlib import Path
import matplotlib.pyplot as plt

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
            
            # For the imaginary part, we need the magnitude of the complex number
            imag_sld_complex = material_sld["imag"]
            imag_sld = abs(imag_sld_complex)  # Get magnitude for bounds
            
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
                bound_width = 0.003 * (max(energy_list) - min(energy_list))  # 0.3% of x-axis range
                
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
                capsize=1
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
                    bound_width = 0.003 * (max(energy_list) - min(energy_list))  # 0.3% of x-axis range
                    
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




# Visualization Widgets

def create_interactive_model_visualizer(objectives_dict, structures_dict, energy_list=None, 
                                 shade_start=None, profile_shift=-20, xlim=None, 
                                 fig_size_w=14, colors=None):
    """
    Create an interactive widget to visualize reflectometry models for specific energies.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to include. If None, uses all available energies.
        shade_start (float, optional): Starting position for layer shading
        profile_shift (float): Shift applied to depth profiles
        xlim (list, optional): Custom x-axis limits for SLD plots as [min, max]
        fig_size_w (float): Width of the figure
        colors (list, optional): List of colors for layer shading
        
    Returns:
        ipywidgets.VBox: Interactive widget for model visualization
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    # Determine the available energies
    if energy_list is None:
        available_energies = sorted(list(set(objectives_dict.keys()) & set(structures_dict.keys())))
    else:
        # Filter by available data
        available_energies = []
        for energy in energy_list:
            if energy in objectives_dict and energy in structures_dict:
                available_energies.append(energy)
    
    if not available_energies:
        return widgets.HTML("<b>No valid energies to visualize.</b>")
    
    # Create widgets
    energy_dropdown = widgets.Dropdown(
        options=[(f"{energy} eV", energy) for energy in available_energies],
        description='Energy:',
        style={'description_width': 'initial'}
    )
    
    # Add button to navigate through energies
    prev_button = widgets.Button(description='Previous')
    next_button = widgets.Button(description='Next')
    
    # Chi-squared label
    chi_label = widgets.Label(value="Chi-squared: N/A")
    
    # Create the output widget for the plots
    output = widgets.Output()
    
    # Create the subplot function
    def create_plots(energy):
        # Get the objective and structure for this energy
        obj = objectives_dict[energy]
        structure = structures_dict[energy]
        
        # Calculate chi-squared
        try:
            chi_squared = obj.chisqr()
            chi_label.value = f"Chi-squared: {chi_squared:.4f}"
        except:
            chi_label.value = "Chi-squared: N/A"
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(fig_size_w, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig)
        
        # Create axes
        ax_refl = fig.add_subplot(gs[0])
        ax_sld = fig.add_subplot(gs[1])
        
        # Extract data from objective
        data = obj.data
        
        # Plot reflectivity data
        ax_refl.plot(data.data[0], data.data[1], label='Data')
        ax_refl.set_yscale('log')
        ax_refl.plot(data.data[0], obj.model(data.data[0]), label='Simulation')
        ax_refl.legend(loc='upper right')
        ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_refl.set_ylabel('Reflectivity (a.u)')
        ax_refl.set_title(f'Reflectivity - {energy} eV (χ²: {chi_squared:.4f})')
        
        # Plot SLD profiles
        from Plotting_Refl import profileflip
        Real_depth, Real_SLD, Imag_Depth, Imag_SLD = profileflip(structure, depth_shift=0)
        
        # Apply manual shift
        Real_depth = Real_depth + profile_shift
        Imag_Depth = Imag_Depth + profile_shift
        
        # Set initial plot
        ax_sld.plot(Real_depth, Real_SLD, color='blue', label='Real SLD', zorder=2)
        ax_sld.plot(Imag_Depth, Imag_SLD, linestyle='dashed', color='blue', label='Im SLD', zorder=2)
        
        # Set custom xlim if provided
        if xlim is not None:
            ax_sld.set_xlim(xlim)
        
        # Shade layers
        slabs = structure.slabs()
        num_layers = len(slabs)
        
        # Auto-calculate shade_start if not provided
        if shade_start is None:
            current_shade_start = 0
        else:
            current_shade_start = shade_start
        
        # Set default colors if not provided
        if colors is None:
            colors = ['silver', 'grey', 'blue', 'violet', 'orange', 'purple', 'red', 'green', 'yellow']
        
        # Calculate layer positions
        thicknesses = [current_shade_start]
        
        # Extract parameter values
        pvals = obj.parameters.pvals
        
        # Extract thicknesses from slabs or parameter values
        for j in range(1, num_layers):
            thickness_index = (num_layers - j - 1) * 5 + 9  # Based on your pattern
            
            if thickness_index < len(pvals):
                thicknesses.append(thicknesses[-1] + pvals[thickness_index])
            else:
                # Fallback to using slab thickness
                thicknesses.append(thicknesses[-1] + slabs[j]['thickness'])
        
        # Add silver shading between 0 and the first layer
        if len(thicknesses) > 0:
            ax_sld.axvspan(0, thicknesses[0], color='silver', alpha=0.3, zorder=0)
        
        # Shade each layer
        for j in range(len(thicknesses) - 1):
            color_index = min(j, len(colors) - 1)
            ax_sld.axvspan(thicknesses[j], thicknesses[j + 1], 
                          color=colors[color_index], alpha=0.2, zorder=1)
        
        # Add legend and axis labels
        ax_sld.legend(loc='upper right')
        ax_sld.set_xlabel(r'Distance from Si ($\AA$)')
        ax_sld.set_ylabel(r'SLD $(10^{-6})$ $\AA^{-2}$')
        ax_sld.set_title(f'SLD Profile - {energy} eV')
        
        plt.tight_layout()
        return fig
    
    # Update function for the widget
    def update_plot(change=None):
        with output:
            clear_output(wait=True)
            energy = energy_dropdown.value
            fig = create_plots(energy)
            plt.show()
    
    # Define button click handlers
    def on_prev_button_clicked(b):
        current_index = available_energies.index(energy_dropdown.value)
        if current_index > 0:
            energy_dropdown.value = available_energies[current_index - 1]
    
    def on_next_button_clicked(b):
        current_index = available_energies.index(energy_dropdown.value)
        if current_index < len(available_energies) - 1:
            energy_dropdown.value = available_energies[current_index + 1]
    
    # Connect handlers
    energy_dropdown.observe(update_plot, names='value')
    prev_button.on_click(on_prev_button_clicked)
    next_button.on_click(on_next_button_clicked)
    
    # Create button row
    button_row = widgets.HBox([prev_button, next_button, chi_label])
    
    # Create widget layout
    widget = widgets.VBox([energy_dropdown, button_row, output])
    
    # Initialize the plot
    update_plot()
    
    return widget


def create_multi_energy_comparison(objectives_dict, structures_dict, energy_list=None, 
                                  shade_start=None, profile_shift=-20, xlim=None, 
                                  fig_size_w=16, colors=None, max_energies=4):
    """
    Create an interactive widget to compare multiple energy models side by side.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to include. If None, uses all available energies.
        shade_start (float, optional): Starting position for layer shading
        profile_shift (float): Shift applied to depth profiles
        xlim (list, optional): Custom x-axis limits for SLD plots as [min, max]
        fig_size_w (float): Width of the figure
        colors (list, optional): List of colors for layer shading
        max_energies (int): Maximum number of energies to display at once
        
    Returns:
        ipywidgets.VBox: Interactive widget for model comparison
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    # Determine the available energies
    if energy_list is None:
        available_energies = sorted(list(set(objectives_dict.keys()) & set(structures_dict.keys())))
    else:
        # Filter by available data
        available_energies = []
        for energy in energy_list:
            if energy in objectives_dict and energy in structures_dict:
                available_energies.append(energy)
    
    if not available_energies:
        return widgets.HTML("<b>No valid energies to visualize.</b>")
    
    # Create widgets
    energy_select = widgets.SelectMultiple(
        options=[(f"{energy} eV", energy) for energy in available_energies],
        description='Energies:',
        rows=min(10, len(available_energies)),
        style={'description_width': 'initial'}
    )
    
    # Add a button to update the plot
    update_button = widgets.Button(description='Update Plot')
    
    # Add info label
    info_label = widgets.Label(value=f"Select up to {max_energies} energies to compare")
    
    # Create the output widget for the plots
    output = widgets.Output()
    
    # Create the subplot function
    def create_comparison_plots(selected_energies):
        # Limit the number of energies
        if len(selected_energies) > max_energies:
            selected_energies = selected_energies[:max_energies]
            info_label.value = f"Showing {max_energies} of {len(selected_energies)} selected energies"
        else:
            info_label.value = f"Comparing {len(selected_energies)} energies"
        
        n_energies = len(selected_energies)
        
        # Prepare lists for the modelcomparisonplot function
        obj_list = [objectives_dict[energy] for energy in selected_energies]
        structure_list = [structures_dict[energy] for energy in selected_energies]
        
        # Prepare shade_start
        if shade_start is None:
            shade_start_list = [0] * n_energies
        elif isinstance(shade_start, (int, float)):
            shade_start_list = [shade_start] * n_energies
        else:
            shade_start_list = shade_start
        
        # Call your modelcomparisonplot function
        from Plotting_Refl import modelcomparisonplot
        fig, axes = modelcomparisonplot(
            obj_list=obj_list, 
            structure_list=structure_list,
            shade_start=shade_start_list,
            profile_shift=profile_shift,
            xlim=xlim,
            fig_size_w=fig_size_w,
            colors=colors
        )
        
        # Add energy labels to each plot
        if n_energies == 1:
            # Single plot case
            axes[0].set_title(f"Reflectivity - {selected_energies[0]} eV")
            axes[1].set_title(f"SLD Profile - {selected_energies[0]} eV")
        else:
            # Multiple plots case
            for i, energy in enumerate(selected_energies):
                chi_squared = obj_list[i].chisqr()
                axes[0, i].set_title(f"Reflectivity - {energy} eV (χ²: {chi_squared:.4f})")
                axes[1, i].set_title(f"SLD Profile - {energy} eV")
        
        plt.tight_layout()
        return fig
    
    # Update function for the widget
    def update_plot(b):
        with output:
            clear_output(wait=True)
            selected_energies = list(energy_select.value)
            
            if not selected_energies:
                print("Please select at least one energy to visualize.")
                return
            
            fig = create_comparison_plots(selected_energies)
            plt.show()
    
    # Connect handler
    update_button.on_click(update_plot)
    
    # Create widget layout
    controls = widgets.VBox([energy_select, update_button, info_label])
    widget = widgets.HBox([controls, output])
    
    return widget


def analyze_model_parameters_by_energy(objectives_dict, energy_list=None, parameter_patterns=None):
    """
    Analyze how parameters change across different energies.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        energy_list (list, optional): List of energies to include. If None, uses all available energies.
        parameter_patterns (list, optional): List of parameter name patterns to include.
                                         If None, includes all parameters.
        
    Returns:
        pandas.DataFrame: DataFrame containing parameter values by energy
    """
    import pandas as pd
    import re
    import numpy as np
    
    # Determine the available energies
    if energy_list is None:
        energy_list = sorted(objectives_dict.keys())
    else:
        # Filter to valid energies
        energy_list = [e for e in energy_list if e in objectives_dict]
    
    if not energy_list:
        print("No valid energies to analyze.")
        return pd.DataFrame()
    
    # Collect parameter values across energies
    param_values = {}
    
    for energy in energy_list:
        objective = objectives_dict[energy]
        
        for param in objective.parameters.flattened():
            # Skip if parameter doesn't match patterns
            if parameter_patterns:
                if not any(re.search(pattern, param.name, re.IGNORECASE) for pattern in parameter_patterns):
                    continue
            
            # Initialize parameter entry if not already present
            if param.name not in param_values:
                param_values[param.name] = {
                    'values': {},
                    'bounds': {},
                    'stderr': {},
                    'vary': {}
                }
            
            # Store parameter value for this energy
            param_values[param.name]['values'][energy] = param.value
            
            # Store error if available
            stderr = getattr(param, 'stderr', None)
            param_values[param.name]['stderr'][energy] = stderr
            
            # Store vary flag
            vary = getattr(param, 'vary', None)
            param_values[param.name]['vary'][energy] = vary
            
            # Store bounds if available
            bounds = None
            try:
                bounds_obj = getattr(param, 'bounds', None)
                if bounds_obj is not None:
                    if hasattr(bounds_obj, 'lb') and hasattr(bounds_obj, 'ub'):
                        bounds = (bounds_obj.lb, bounds_obj.ub)
                    elif isinstance(bounds_obj, tuple) and len(bounds_obj) == 2:
                        bounds = bounds_obj
            except:
                pass
            
            param_values[param.name]['bounds'][energy] = bounds
    
    # Create DataFrame
    rows = []
    
    for param_name, param_data in param_values.items():
        row = {'parameter': param_name}
        
        # Add values by energy
        for energy in energy_list:
            if energy in param_data['values']:
                row[f"{energy}"] = param_data['values'][energy]
                row[f"{energy}_stderr"] = param_data['stderr'].get(energy)
                
                # Calculate percentage of bound range
                bounds = param_data['bounds'].get(energy)
                if bounds and bounds[0] is not None and bounds[1] is not None:
                    lower, upper = bounds
                    value = param_data['values'][energy]
                    bound_range = upper - lower
                    
                    if bound_range > 0:
                        percentage = (value - lower) / bound_range * 100
                        row[f"{energy}_bound_pct"] = percentage
        
        rows.append(row)
    
    # Create DataFrame and sort by parameter name
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values('parameter')
    
    return df


def plot_parameter_energy_trends(param_df, parameter_names=None, figsize=(12, 8)):
    """
    Plot trends in parameter values across energies.
    
    Args:
        param_df (DataFrame): DataFrame from analyze_model_parameters_by_energy
        parameter_names (list, optional): List of parameter names to plot.
                                      If None, include all parameters.
        figsize (tuple): Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    
    # Check if DataFrame is valid
    if param_df.empty:
        print("Empty DataFrame provided. Nothing to plot.")
        return None
    
    # Extract energy columns (those that don't have '_stderr' or '_bound_pct' suffix)
    energy_cols = [col for col in param_df.columns 
                  if col != 'parameter' and '_stderr' not in col and '_bound_pct' not in col]
    
    # Convert energy columns to numeric values
    energies = sorted([float(col) for col in energy_cols])
    
    # Filter parameters if needed
    if parameter_names:
        df = param_df[param_df['parameter'].isin(parameter_names)].copy()
    else:
        df = param_df.copy()
    
    if df.empty:
        print("No parameters match the specified names.")
        return None
    
    # Group parameters by type
    param_groups = {}
    
    for param in df['parameter']:
        # Try to categorize parameters
        if 'sld' in param.lower() and 'isld' not in param.lower():
            group = 'Real SLD'
        elif 'isld' in param.lower():
            group = 'Imaginary SLD'
        elif 'thick' in param.lower():
            group = 'Thickness'
        elif 'rough' in param.lower():
            group = 'Roughness'
        else:
            group = 'Other'
        
        if group not in param_groups:
            param_groups[group] = []
        param_groups[group].append(param)
    
    # Determine number of subplots
    n_groups = len(param_groups)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_groups, 1, figsize=figsize, sharex=True)
    
    # Use a single subplot if there's only one group
    if n_groups == 1:
        axes = [axes]
    
    # Plot each parameter group
    for i, (group, params) in enumerate(param_groups.items()):
        ax = axes[i]
        
        for param in params:
            # Extract values for this parameter
            values = []
            errors = []
            
            for energy in energies:
                if f"{energy}" in df.columns and param in df['parameter'].values:
                    param_row = df[df['parameter'] == param]
                    value = param_row[f"{energy}"].values[0]
                    error = param_row[f"{energy}_stderr"].values[0] if f"{energy}_stderr" in param_row else None
                    
                    values.append(value)
                    errors.append(error)
            
            # Plot values with error bars if available
            if None not in values:
                if all(err is not None for err in errors):
                    ax.errorbar(energies, values, yerr=errors, marker='o', label=param)
                else:
                    ax.plot(energies, values, marker='o', label=param)
        
        # Set labels and title
        ax.set_ylabel(group)
        ax.set_title(f"{group} Parameters vs Energy")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # Set x-axis label on the bottom subplot
    axes[-1].set_xlabel("Energy (eV)")
    
    plt.tight_layout()
    return fig


def interactive_parameter_explorer(objectives_dict, structures_dict, energy_list=None):
    """
    Create an interactive widget to explore parameter values and model visualizations.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to include. If None, uses all available energies.
        
    Returns:
        ipywidgets.VBox: Interactive widget for parameter exploration
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import pandas as pd
    
    # Determine the available energies
    if energy_list is None:
        available_energies = sorted(list(set(objectives_dict.keys()) & set(structures_dict.keys())))
    else:
        # Filter by available data
        available_energies = []
        for energy in energy_list:
            if energy in objectives_dict and energy in structures_dict:
                available_energies.append(energy)
    
    if not available_energies:
        return widgets.HTML("<b>No valid energies to explore.</b>")
    
    # Analyze parameters across energies
    param_df = analyze_model_parameters_by_energy(objectives_dict, available_energies)
    
    # Get unique parameter names
    if param_df.empty:
        return widgets.HTML("<b>No parameters found to analyze.</b>")
    
    param_names = param_df['parameter'].unique().tolist()
    
    # Group parameters by type for the dropdown
    param_groups = {
        'SLD Parameters': [p for p in param_names if 'sld' in p.lower()],
        'Thickness Parameters': [p for p in param_names if 'thick' in p.lower()],
        'Roughness Parameters': [p for p in param_names if 'rough' in p.lower()],
        'Other Parameters': [p for p in param_names if not any(x in p.lower() for x in ['sld', 'thick', 'rough'])]
    }
    
    # Create a flattened list of options with group headers
    dropdown_options = []
    
    for group, params in param_groups.items():
        if params:
            dropdown_options.append((group, None))  # Add group header
            dropdown_options.extend([(f"  {p}", p) for p in sorted(params)])  # Add indented parameters
    
    # Create widgets
    param_dropdown = widgets.Dropdown(
        options=dropdown_options,
        description='Parameter:',
        style={'description_width': 'initial'}
    )
    
    energy_select = widgets.SelectMultiple(
        options=[(f"{energy} eV", energy) for energy in available_energies],
        description='Energies:',
        rows=min(6, len(available_energies)),
        style={'description_width': 'initial'}
    )
    
    # Tabs for different views
    tab_output = widgets.Output()
    param_output = widgets.Output()
    model_output = widgets.Output()
    trend_output = widgets.Output()
    
    tabs = widgets.Tab(children=[tab_output, param_output, model_output, trend_output])
    tabs.set_title(0, 'Parameter Table')
    tabs.set_title(1, 'Parameter Details')
    tabs.set_title(2, 'Model Visualization')
    tabs.set_title(3, 'Parameter Trends')
    
    # Update buttons
    table_button = widgets.Button(description='Update Table')
    detail_button = widgets.Button(description='Show Parameter')
    model_button = widgets.Button(description='Show Models')
    trend_button = widgets.Button(description='Show Trends')
    
    # Filter widgets for table
    filter_text = widgets.Text(
        value='',
        placeholder='Filter parameters...',
        description='Filter:',
        style={'description_width': 'initial'}
    )
    
    # Update functions
    def update_table(b):
        with tab_output:
            clear_output(wait=True)
            # Get selected energies
            selected_energies = list(energy_select.value)
            if not selected_energies:
                selected_energies = available_energies[:5]  # Default to first 5
            
            # Filter parameters
            filter_val = filter_text.value.strip().lower()
            if filter_val:
                filtered_df = param_df[param_df['parameter'].str.lower().str.contains(filter_val)]
            else:
                filtered_df = param_df
            
            # Select columns
            display_cols = ['parameter']
            for energy in selected_energies:
                if f"{energy}" in filtered_df.columns:
                    display_cols.extend([f"{energy}", f"{energy}_stderr"])
            
            # Display table
            if filtered_df.empty:
                print("No parameters match the filter criteria.")
            else:
                display(filtered_df[display_cols])
    
    def show_parameter_details(b):
        with param_output:
            clear_output(wait=True)
            # Get selected parameter
            param_name = param_dropdown.value
            
            if param_name is None:
                print("Please select a parameter to view.")
                return
            
            # Get parameter data
            param_row = param_df[param_df['parameter'] == param_name]
            
            if param_row.empty:
                print(f"Parameter '{param_name}' not found in the data.")
                return
            
            # Create a table of values by energy
            data = []
            
            for energy in available_energies:
                if f"{energy}" in param_row.columns:
                    value = param_row[f"{energy}"].values[0]
                    stderr = param_row[f"{energy}_stderr"].values[0] if f"{energy}_stderr" in param_row else None
                    bound_pct = param_row[f"{energy}_bound_pct"].values[0] if f"{energy}_bound_pct" in param_row else None
                    
                    data.append({
                        'Energy': energy,
                        'Value': value,
                        'Std Error': stderr,
                        'Bound %': bound_pct
                    })
            
            detail_df = pd.DataFrame(data)
            display(detail_df)
    
    def show_models(b):
        with model_output:
            clear_output(wait=True)
            # Get selected energies
            selected_energies = list(energy_select.value)
            
            if not selected_energies:
                print("Please select at least one energy to visualize.")
                return
            
            if len(selected_energies) > 4:
                print("Warning: Showing only the first 4 selected energies.")
                selected_energies = selected_energies[:4]
            
            # Prepare lists for the modelcomparisonplot function
            obj_list = [objectives_dict[energy] for energy in selected_energies]
            structure_list = [structures_dict[energy] for energy in selected_energies]
            
            # Get the modelcomparisonplot function
            from Plotting_Refl import modelcomparisonplot
            
            # Create the plots
            fig, axes = modelcomparisonplot(
                obj_list=obj_list, 
                structure_list=structure_list,
                shade_start=[0] * len(selected_energies),
                profile_shift=-20,
                xlim=None,
                fig_size_w=14,
                colors=None
            )
            
            # Add energy labels to each plot
            if len(selected_energies) == 1:
                # Single plot case
                axes[0].set_title(f"Reflectivity - {selected_energies[0]} eV")
                axes[1].set_title(f"SLD Profile - {selected_energies[0]} eV")
            else:
                # Multiple plots case
                for i, energy in enumerate(selected_energies):
                    chi_squared = obj_list[i].chisqr()
                    axes[0, i].set_title(f"Reflectivity - {energy} eV (χ²: {chi_squared:.4f})")
                    axes[1, i].set_title(f"SLD Profile - {energy} eV")
            
            plt.tight_layout()
            plt.show()
    
    def show_parameter_trends(b):
        with trend_output:
            clear_output(wait=True)
            # Get selected parameter
            param_name = param_dropdown.value
            
            if param_name is None:
                print("Please select a parameter to view trends.")
                return
            
            # Create trend plot
            fig = plot_parameter_energy_trends(param_df, parameter_names=[param_name], figsize=(10, 6))
            
            if fig:
                plt.show()
    
    # Connect handlers
    table_button.on_click(update_table)
    detail_button.on_click(show_parameter_details)
    model_button.on_click(show_models)
    trend_button.on_click(show_parameter_trends)
    
    # Create button rows
    table_controls = widgets.HBox([filter_text, table_button])
    detail_controls = widgets.HBox([param_dropdown, detail_button])
    model_controls = widgets.HBox([widgets.Label("Use energy selection above"), model_button])
    trend_controls = widgets.HBox([widgets.Label("Use parameter selection above"), trend_button])
    
    # Create layout for controls
    controls = widgets.VBox([
        widgets.HTML("<b>Select Energies:</b>"),
        energy_select,
        widgets.HTML("<hr style='margin:10px 0'>"),
        widgets.HTML("<b>Table Controls:</b>"),
        table_controls,
        widgets.HTML("<hr style='margin:10px 0'>"),
        widgets.HTML("<b>Parameter Controls:</b>"),
        detail_controls,
        widgets.HTML("<hr style='margin:10px 0'>"),
        widgets.HTML("<b>Model Visualization:</b>"),
        model_controls,
        widgets.HTML("<hr style='margin:10px 0'>"),
        widgets.HTML("<b>Trend Visualization:</b>"),
        trend_controls
    ])
    
    # Initialize with the table view
    update_table(None)
    
    # Create the main widget
    main_widget = widgets.VBox([
        widgets.HTML("<h3>Interactive Parameter Explorer</h3>"),
        widgets.HBox([controls, tabs])
    ])
    
    return main_widget


# First, install the ipympl package if you haven't already
# Uncomment and run this cell if needed
# !pip install ipympl

# Set the matplotlib backend - CRITICAL FOR VS CODE INTERACTIVITY


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

def create_vscode_sld_explorer(objectives_dict, structures_dict, energy_list=None, 
                             material_name="PS", parameter_type='sld', 
                             figsize=(12, 10), min_bound=None):
    """
    Create an interactive plot for VS Code where clicking on a point in the SLD vs energy plot
    shows the corresponding reflectivity curve.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to include. If None, uses all available energies.
        material_name (str): Name of the material to analyze
        parameter_type (str): Type of parameter to plot ('sld' or 'isld')
        figsize (tuple): Figure size as (width, height)
        min_bound (float, optional): Minimum y-axis value for SLD plot
        
    Returns:
        matplotlib.figure.Figure: Interactive figure
    """
    # Determine the available energies
    if energy_list is None:
        available_energies = sorted(list(set(objectives_dict.keys()) & set(structures_dict.keys())))
    else:
        # Filter by available data
        available_energies = []
        for energy in energy_list:
            if energy in objectives_dict and energy in structures_dict:
                available_energies.append(energy)
    
    if not available_energies:
        print("No valid energies found.")
        return None
    
    # Extract SLD values and parameter info for each energy
    param_data = []
    param_string = f"{material_name} - {parameter_type}"
    
    for energy in available_energies:
        # Get objective for this energy
        objective = objectives_dict[energy]
        
        # Find the specified parameter
        found = False
        for param in objective.parameters.flattened():
            if param.name == param_string:
                value = param.value
                stderr = getattr(param, 'stderr', None)
                bound_low = None
                bound_high = None
                
                try:
                    bounds = getattr(param, 'bounds', None)
                    if bounds is not None:
                        # Extract bound values
                        if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                            bound_low = bounds.lb
                            bound_high = bounds.ub
                        elif isinstance(bounds, tuple) and len(bounds) == 2:
                            bound_low, bound_high = bounds
                except:
                    pass
                
                param_data.append({
                    'energy': energy,
                    'value': value,
                    'stderr': stderr if stderr is not None else 0,
                    'bound_low': bound_low,
                    'bound_high': bound_high,
                    'chi_squared': objective.chisqr()
                })
                found = True
                break
        
        if not found:
            print(f"Parameter {param_string} not found for energy {energy}")
    
    if not param_data:
        print(f"No data found for parameter {param_string}")
        return None
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(param_data)
    df = df.sort_values('energy')
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 2])
    
    # SLD vs Energy plot (top left)
    ax_sld = fig.add_subplot(gs[0, :2])
    # Reflectivity plot (bottom)
    ax_refl = fig.add_subplot(gs[2, :])
    # SLD profile plot (top right)
    ax_profile = fig.add_subplot(gs[0:2, 2])
    # Chi-squared plot (middle left)
    ax_chi = fig.add_subplot(gs[1, :2])
    
    # Create a colormap based on chi-squared values
    if len(df) > 1:
        norm = plt.Normalize(df['chi_squared'].min(), df['chi_squared'].max())
    else:
        norm = plt.Normalize(0, 1)
    
    # Color points by chi-squared
    sld_points = []
    for i, (_, row) in enumerate(df.iterrows()):
        point = ax_sld.scatter(
            row['energy'], 
            row['value'],
            s=64,  # Larger size for better clickability
            c=[cm.viridis(norm(row['chi_squared']))],
            picker=5  # Enable picking with a 5 pixel tolerance
        )
        sld_points.append(point)
    
    # Add error bars separately (without picker to avoid confusion)
    ax_sld.errorbar(
        df['energy'], 
        df['value'],
        yerr=df['stderr'],
        fmt='none',  # No markers
        ecolor='gray',
        alpha=0.5
    )
    
    # Set up the colorbar for chi-squared
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_sld, label='Chi-squared')
    
    # Add bounds as a shaded region
    has_bounds = df['bound_low'].notna().any() and df['bound_high'].notna().any()
    if has_bounds:
        # Get min and max energies
        min_energy = df['energy'].min()
        max_energy = df['energy'].max()
        
        # Create arrays for bounds
        x_bounds = np.linspace(min_energy, max_energy, 100)
        y_low = np.interp(x_bounds, df['energy'][df['bound_low'].notna()], 
                         df['bound_low'][df['bound_low'].notna()])
        y_high = np.interp(x_bounds, df['energy'][df['bound_high'].notna()], 
                          df['bound_high'][df['bound_high'].notna()])
        
        # Plot bounds
        ax_sld.fill_between(x_bounds, y_low, y_high, alpha=0.2, color='gray')
    
    # Set labels and title for SLD plot
    y_label = 'Real SLD (10⁻⁶ Å⁻²)' if parameter_type == 'sld' else 'Imaginary SLD (10⁻⁶ Å⁻²)'
    ax_sld.set_xlabel('Energy (eV)')
    ax_sld.set_ylabel(y_label)
    ax_sld.set_title(f'{y_label} vs Energy for {material_name}')
    ax_sld.grid(True, alpha=0.3)
    
    # Set minimum y value if specified
    if min_bound is not None:
        ylim = ax_sld.get_ylim()
        ax_sld.set_ylim(min_bound, ylim[1])
    
    # Plot chi-squared vs energy
    chi_points = ax_chi.scatter(
        df['energy'], 
        df['chi_squared'], 
        c='red', 
        s=64,
        picker=5
    )
    ax_chi.plot(df['energy'], df['chi_squared'], 'r-', alpha=0.5)
    ax_chi.set_xlabel('Energy (eV)')
    ax_chi.set_ylabel('Chi-squared')
    ax_chi.set_title('Goodness of Fit vs Energy')
    ax_chi.grid(True, alpha=0.3)
    
    # Use log scale for chi-squared if values vary by more than a factor of 10
    if df['chi_squared'].max() / df['chi_squared'].min() > 10:
        ax_chi.set_yscale('log')
    
    # Text annotations for info display
    energy_text = fig.text(0.02, 0.02, "", fontsize=10, transform=fig.transFigure)
    value_text = fig.text(0.3, 0.02, "", fontsize=10, transform=fig.transFigure)
    gof_text = fig.text(0.6, 0.02, "", fontsize=10, transform=fig.transFigure)
    
    # Create a vertical line to show the current energy on the SLD plot
    energy_line = ax_sld.axvline(x=df['energy'].iloc[0], color='red', linestyle='--', alpha=0.7)
    
    # Create a vertical line for the chi-squared plot
    chi_line = ax_chi.axvline(x=df['energy'].iloc[0], color='red', linestyle='--', alpha=0.7)
    
    # Create highlight point to mark the selected energy in SLD plot
    highlight_point = ax_sld.plot(df['energy'].iloc[0], df['value'].iloc[0], 'o', 
                                 color='red', markersize=12, alpha=0.7)[0]
    
    # Function to update the plots when a point is clicked
    def update_plots(energy):
        # Find the index for this energy
        idx = df[df['energy'] == energy].index
        if len(idx) == 0:
            return
        idx = idx[0]
        
        # Get the value and chi-squared for this energy
        value = df.loc[idx, 'value']
        chi_squared = df.loc[idx, 'chi_squared']
        
        # Update vertical lines
        energy_line.set_xdata([energy, energy])
        chi_line.set_xdata([energy, energy])
        
        # Update highlight point
        highlight_point.set_data([energy], [value])
        
        # Get the objective and structure for this energy
        obj = objectives_dict[energy]
        structure = structures_dict[energy]
        
        # Update reflectivity plot
        ax_refl.clear()
        data = obj.data
        ax_refl.plot(data.data[0], data.data[1], 'o', label='Data', markersize=3)
        ax_refl.plot(data.data[0], obj.model(data.data[0]), '-', label='Simulation')
        ax_refl.set_yscale('log')
        ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_refl.set_ylabel('Reflectivity (a.u)')
        ax_refl.set_title(f'Reflectivity - {energy} eV (χ²: {chi_squared:.4f})')
        ax_refl.grid(True, alpha=0.3)
        ax_refl.legend(loc='best')
        
        # Update SLD profile plot
        ax_profile.clear()
        
        # Get SLD profiles
        from Plotting_Refl import profileflip
        Real_depth, Real_SLD, Imag_Depth, Imag_SLD = profileflip(structure, depth_shift=0)
        
        # Apply profile shift
        profile_shift = -20
        Real_depth = Real_depth + profile_shift
        Imag_Depth = Imag_Depth + profile_shift
        
        # Plot SLD profiles
        ax_profile.plot(Real_depth, Real_SLD, 'b-', label='Real SLD')
        ax_profile.plot(Imag_Depth, Imag_SLD, 'b--', label='Imag SLD')
        ax_profile.set_xlabel(r'Distance from Si ($\AA$)')
        ax_profile.set_ylabel(r'SLD $(10^{-6})$ $\AA^{-2}$')
        ax_profile.set_title(f'SLD Profile - {energy} eV')
        ax_profile.grid(True, alpha=0.3)
        ax_profile.legend(loc='best')
        
        # Update text annotations
        energy_text.set_text(f"Energy: {energy} eV")
        value_text.set_text(f"{parameter_type.upper()}: {value:.6g}")
        gof_text.set_text(f"χ²: {chi_squared:.4g}")
        
        # Force the figure to update
        fig.canvas.draw_idle()
    
    # Connect the pick event
    def on_pick(event):
        # Check if we have a valid pick event with data points
        if hasattr(event, 'ind') and len(event.ind) > 0:
            # Get the artist that was picked
            artist = event.artist
            
            # Determine which energy was clicked
            ind = event.ind[0]  # Use the first point if multiple were picked
            
            # Extract the energy value depending on which plot was clicked
            if artist in sld_points or artist == chi_points:
                # Get x data from the artist
                xdata = artist.get_offsets()[:, 0]
                # Get the energy at the picked index
                energy = xdata[ind]
                # Update the plots
                update_plots(energy)
    
    # Connect the pick event to the figure
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Initial update with the first energy
    update_plots(df['energy'].iloc[0])
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for text
    
    print("The plot is now interactive. Click on any point to update the reflectivity and SLD profile.")
    print("If clicking doesn't work, make sure you have installed ipympl and used %matplotlib widget.")
    
    return fig

# Example usage:
# fig = create_vscode_sld_explorer(objectives_dict, structures_dict, material_name="PS", parameter_type='sld')


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

def create_vscode_parameter_explorer(objectives_dict, structures_dict, energy_list=None, 
                                    material_name="PS", figsize=(14, 12), xlim=None, 
                                    bound_threshold=0.02):
    """
    Create an interactive plot for VS Code where clicking on a parameter's energy trend
    shows the corresponding reflectivity curve and SLD profile.
    
    Args:
        objectives_dict (dict): Dictionary mapping energy values to Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to include. If None, uses all available energies.
        material_name (str): Name of the material to analyze
        figsize (tuple): Figure size as (width, height)
        xlim (tuple, optional): Custom x-axis limits for energy plots as (min, max)
        bound_threshold (float): Threshold for highlighting values near bounds (as fraction of bound range)
        
    Returns:
        matplotlib.figure.Figure: Interactive figure
    """
    # Determine the available energies
    if energy_list is None:
        available_energies = sorted(list(set(objectives_dict.keys()) & set(structures_dict.keys())))
    else:
        # Filter by available data
        available_energies = []
        for energy in energy_list:
            if energy in objectives_dict and energy in structures_dict:
                available_energies.append(energy)
    
    if not available_energies:
        print("No valid energies found.")
        return None
    
    # Extract parameter values and info for each energy
    param_strings = [
        f"{material_name} - sld",
        f"{material_name} - isld",
        f"{material_name} - thick",
        f"{material_name} - rough"
    ]
    
    # Initialize data structure
    param_data = {param: [] for param in param_strings}
    chi_squared_data = []
    
    for energy in available_energies:
        # Get objective for this energy
        objective = objectives_dict[energy]
        chi_squared = objective.chisqr()
        chi_squared_data.append({'energy': energy, 'chi_squared': chi_squared})
        
        # Find parameter values
        for param_string in param_strings:
            found = False
            for param in objective.parameters.flattened():
                if param.name == param_string:
                    value = param.value
                    stderr = getattr(param, 'stderr', None)
                    bound_low = None
                    bound_high = None
                    
                    try:
                        bounds = getattr(param, 'bounds', None)
                        if bounds is not None:
                            # Extract bound values
                            if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                                bound_low = bounds.lb
                                bound_high = bounds.ub
                            elif isinstance(bounds, tuple) and len(bounds) == 2:
                                bound_low, bound_high = bounds
                    except:
                        pass
                    
                    # Calculate if the value is near bounds
                    near_bound = False
                    if bound_low is not None and bound_high is not None:
                        bound_range = bound_high - bound_low
                        threshold = bound_threshold * bound_range
                        near_bound = (abs(value - bound_low) < threshold or 
                                     abs(bound_high - value) < threshold)
                    
                    param_data[param_string].append({
                        'energy': energy,
                        'value': value,
                        'stderr': stderr if stderr is not None else 0,
                        'bound_low': bound_low,
                        'bound_high': bound_high,
                        'near_bound': near_bound,
                        'chi_squared': chi_squared
                    })
                    found = True
                    break
            
            if not found:
                print(f"Parameter {param_string} not found for energy {energy}")
    
    # Convert to DataFrames
    param_dfs = {}
    for param_string, data in param_data.items():
        if data:
            df = pd.DataFrame(data)
            param_dfs[param_string] = df.sort_values('energy')
    
    chi_squared_df = pd.DataFrame(chi_squared_data).sort_values('energy')
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 1.5])
    
    # Create axis for each parameter (5 rows, including roughness)
    ax_sld = fig.add_subplot(gs[0, :2])
    ax_isld = fig.add_subplot(gs[1, :2])
    ax_thick = fig.add_subplot(gs[2, :2])
    ax_rough = fig.add_subplot(gs[3, :2])
    ax_chi = fig.add_subplot(gs[4, :2])
    
    # Create axis for reflectivity and SLD profile (right column)
    ax_refl = fig.add_subplot(gs[3:5, 2])
    ax_profile = fig.add_subplot(gs[0:3, 2])
    
    # Dictionary to store scatter plots for interaction
    scatter_plots = {}
    
    # Parameter type to color mapping
    param_colors = {
        'sld': 'blue',
        'isld': 'orange',
        'thick': 'green',
        'rough': 'purple',
        'chi': 'red'
    }
    
    # Function to plot parameter with bounds, errors, and near-bound highlighting
    def plot_parameter(ax, param_string, param_type):
        if param_string not in param_dfs:
            return None, None
        
        df = param_dfs[param_string]
        
        # Create masked arrays for normal and near-bound points
        mask_normal = ~df['near_bound']
        mask_near_bound = df['near_bound']
        
        # Plot normal points
        if any(mask_normal):
            normal_scatter = ax.scatter(
                df.loc[mask_normal, 'energy'], 
                df.loc[mask_normal, 'value'], 
                s=64, color=param_colors[param_type], picker=5,
                label=f"{param_string} (normal)"
            )
        else:
            normal_scatter = None
            
        # Plot near-bound points
        if any(mask_near_bound):
            near_bound_scatter = ax.scatter(
                df.loc[mask_near_bound, 'energy'], 
                df.loc[mask_near_bound, 'value'], 
                s=64, color='red', picker=5,
                label=f"{param_string} (near bound)"
            )
        else:
            near_bound_scatter = None
        
        # Add bounds as shaded region
        has_bounds = df['bound_low'].notna().any() and df['bound_high'].notna().any()
        if has_bounds:
            # Create arrays for bounds (use only rows with valid bounds)
            valid_bounds = df['bound_low'].notna() & df['bound_high'].notna()
            if any(valid_bounds):
                bound_df = df[valid_bounds].sort_values('energy')
                
                # Get energies and bounds
                energies = bound_df['energy']
                lower_bounds = bound_df['bound_low']
                upper_bounds = bound_df['bound_high']
                
                # Plot bounds as shaded region
                ax.fill_between(
                    energies, lower_bounds, upper_bounds, 
                    color=param_colors[param_type], alpha=0.1
                )
                
                # Add thin lines for the bounds
                ax.plot(
                    energies, lower_bounds, '--', 
                    color=param_colors[param_type], alpha=0.5, linewidth=1
                )
                ax.plot(
                    energies, upper_bounds, '--', 
                    color=param_colors[param_type], alpha=0.5, linewidth=1
                )
        
        # Add error bars
        ax.errorbar(
            df['energy'], df['value'], 
            yerr=df['stderr'], 
            fmt='none', ecolor='gray', alpha=0.5
        )
        
        # Connect points with a line
        ax.plot(
            df['energy'], df['value'], '-', 
            color=param_colors[param_type], alpha=0.5
        )
        
        # Return combined scatter plot for picker functionality
        if normal_scatter is not None and near_bound_scatter is not None:
            # Both types of points exist
            return (normal_scatter, near_bound_scatter)
        elif normal_scatter is not None:
            return normal_scatter, None
        elif near_bound_scatter is not None:
            return near_bound_scatter, None
        else:
            return None, None
    
    # Plot each parameter
    scatter_plots['sld'], near_bound_sld = plot_parameter(ax_sld, f"{material_name} - sld", 'sld')
    scatter_plots['isld'], near_bound_isld = plot_parameter(ax_isld, f"{material_name} - isld", 'isld')
    scatter_plots['thick'], near_bound_thick = plot_parameter(ax_thick, f"{material_name} - thick", 'thick')
    scatter_plots['rough'], near_bound_rough = plot_parameter(ax_rough, f"{material_name} - rough", 'rough')
    
    # Set labels and titles
    ax_sld.set_ylabel('Real SLD\n(10⁻⁶ Å⁻²)')
    ax_sld.set_title(f'Real SLD vs Energy for {material_name}')
    ax_sld.grid(True, alpha=0.3)
    
    ax_isld.set_ylabel('Imag SLD\n(10⁻⁶ Å⁻²)')
    ax_isld.set_title(f'Imaginary SLD vs Energy for {material_name}')
    ax_isld.grid(True, alpha=0.3)
    
    ax_thick.set_ylabel('Thickness (Å)')
    ax_thick.set_title(f'Thickness vs Energy for {material_name}')
    ax_thick.grid(True, alpha=0.3)
    
    ax_rough.set_ylabel('Roughness (Å)')
    ax_rough.set_title(f'Roughness vs Energy for {material_name}')
    ax_rough.grid(True, alpha=0.3)
    
    # Plot chi-squared vs energy
    scatter_plots['chi'] = ax_chi.scatter(
        chi_squared_df['energy'], chi_squared_df['chi_squared'], 
        s=64, color='red', picker=5
    )
    ax_chi.plot(chi_squared_df['energy'], chi_squared_df['chi_squared'], 'r-', alpha=0.5)
    ax_chi.set_xlabel('Energy (eV)')
    ax_chi.set_ylabel('Chi-squared')
    ax_chi.set_title('Goodness of Fit vs Energy')
    ax_chi.grid(True, alpha=0.3)
    
    # Use log scale for chi-squared if values vary by more than a factor of 10
    if chi_squared_df['chi_squared'].max() / chi_squared_df['chi_squared'].min() > 10:
        ax_chi.set_yscale('log')
    
    # Text annotations for info display
    energy_text = fig.text(0.02, 0.02, "", fontsize=10, transform=fig.transFigure)
    sld_text = fig.text(0.20, 0.02, "", fontsize=10, transform=fig.transFigure)
    isld_text = fig.text(0.36, 0.02, "", fontsize=10, transform=fig.transFigure)
    thick_text = fig.text(0.52, 0.02, "", fontsize=10, transform=fig.transFigure)
    rough_text = fig.text(0.68, 0.02, "", fontsize=10, transform=fig.transFigure)
    gof_text = fig.text(0.84, 0.02, "", fontsize=10, transform=fig.transFigure)
    
    # Apply custom xlim if provided
    if xlim is not None:
        for ax in [ax_sld, ax_isld, ax_thick, ax_rough, ax_chi]:
            ax.set_xlim(xlim)
    
    # Create vertical lines for each plot
    energy_lines = {}
    if f"{material_name} - sld" in param_dfs:
        energy_lines['sld'] = ax_sld.axvline(
            x=available_energies[0], color='red', linestyle='--', alpha=0.7
        )
    if f"{material_name} - isld" in param_dfs:
        energy_lines['isld'] = ax_isld.axvline(
            x=available_energies[0], color='red', linestyle='--', alpha=0.7
        )
    if f"{material_name} - thick" in param_dfs:
        energy_lines['thick'] = ax_thick.axvline(
            x=available_energies[0], color='red', linestyle='--', alpha=0.7
        )
    if f"{material_name} - rough" in param_dfs:
        energy_lines['rough'] = ax_rough.axvline(
            x=available_energies[0], color='red', linestyle='--', alpha=0.7
        )
    energy_lines['chi'] = ax_chi.axvline(
        x=available_energies[0], color='red', linestyle='--', alpha=0.7
    )
    
    # Create highlight points for each parameter
    highlight_points = {}
    for param_type, param_string in zip(
        ['sld', 'isld', 'thick', 'rough'], 
        [f"{material_name} - sld", f"{material_name} - isld", 
         f"{material_name} - thick", f"{material_name} - rough"]
    ):
        if param_string in param_dfs:
            df = param_dfs[param_string]
            if not df.empty:
                ax = locals()[f"ax_{param_type}"]  # Get the right axis
                highlight_points[param_type] = ax.plot(
                    df['energy'].iloc[0], df['value'].iloc[0], 
                    'o', color='red', markersize=12, alpha=0.7
                )[0]
    
    # Function to update the plots when a point is clicked
    def update_plots(energy):
        # Update vertical lines
        for line in energy_lines.values():
            line.set_xdata([energy, energy])
        
        # Get the objective and structure for this energy
        obj = objectives_dict[energy]
        structure = structures_dict[energy]
        
        # Find values for this energy
        sld_value = None
        isld_value = None
        thick_value = None
        rough_value = None
        
        # Flags for near bounds
        sld_near_bound = False
        isld_near_bound = False
        thick_near_bound = False
        rough_near_bound = False
        
        # Helper function to get parameter value
        def get_param_value(param_string):
            if param_string in param_dfs:
                df = param_dfs[param_string]
                row = df[df['energy'] == energy]
                if not row.empty:
                    return row['value'].iloc[0], row['near_bound'].iloc[0]
            return None, False
        
        # Get values
        sld_value, sld_near_bound = get_param_value(f"{material_name} - sld")
        isld_value, isld_near_bound = get_param_value(f"{material_name} - isld")
        thick_value, thick_near_bound = get_param_value(f"{material_name} - thick")
        rough_value, rough_near_bound = get_param_value(f"{material_name} - rough")
        
        # Update highlight points
        for param_type, value in zip(
            ['sld', 'isld', 'thick', 'rough'],
            [sld_value, isld_value, thick_value, rough_value]
        ):
            if param_type in highlight_points and value is not None:
                highlight_points[param_type].set_data([energy], [value])
                
        # Get chi-squared
        chi_row = chi_squared_df[chi_squared_df['energy'] == energy]
        chi_squared = chi_row['chi_squared'].iloc[0] if not chi_row.empty else None
        
        # Update reflectivity plot
        ax_refl.clear()
        data = obj.data
        ax_refl.plot(data.data[0], data.data[1], 'o', label='Data', markersize=3)
        ax_refl.plot(data.data[0], obj.model(data.data[0]), '-', label='Simulation')
        ax_refl.set_yscale('log')
        ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_refl.set_ylabel('Reflectivity (a.u)')
        ax_refl.set_title(f'Reflectivity - {energy} eV' + 
                         (f' (χ²: {chi_squared:.4f})' if chi_squared is not None else ''))
        ax_refl.grid(True, alpha=0.3)
        ax_refl.legend(loc='best')
        
        # Update SLD profile plot
        ax_profile.clear()
        
        # Get SLD profiles
        from Plotting_Refl import profileflip
        Real_depth, Real_SLD, Imag_Depth, Imag_SLD = profileflip(structure, depth_shift=0)
        
        # Apply profile shift
        profile_shift = -20
        Real_depth = Real_depth + profile_shift
        Imag_Depth = Imag_Depth + profile_shift
        
        # Plot SLD profiles
        ax_profile.plot(Real_depth, Real_SLD, 'b-', label='Real SLD')
        ax_profile.plot(Imag_Depth, Imag_SLD, 'b--', label='Imag SLD')
        ax_profile.set_xlabel(r'Distance from Si ($\AA$)')
        ax_profile.set_ylabel(r'SLD $(10^{-6})$ $\AA^{-2}$')
        ax_profile.set_title(f'SLD Profile - {energy} eV')
        ax_profile.grid(True, alpha=0.3)
        ax_profile.legend(loc='best')
        
        # Function to format parameter text
        def format_param_text(label, value, near_bound):
            if value is None:
                return ""
            
            color = "red" if near_bound else "black"
            return f"<span style='color:{color}'>{label}: {value:.4g}</span>"
        
        # Update text annotations
        energy_text.set_text(f"Energy: {energy} eV")
        
        # Use HTML formatting to show near-bound parameters in red
        sld_text.set_text(format_param_text("SLD", sld_value, sld_near_bound))
        isld_text.set_text(format_param_text("iSLD", isld_value, isld_near_bound))
        thick_text.set_text(format_param_text("Thick", thick_value, thick_near_bound) + " Å" if thick_value is not None else "")
        rough_text.set_text(format_param_text("Rough", rough_value, rough_near_bound) + " Å" if rough_value is not None else "")
        gof_text.set_text(f"χ²: {chi_squared:.4g}" if chi_squared is not None else "")
        
        # Force the figure to update
        fig.canvas.draw_idle()
    
    # Add click event handler
    def on_pick(event):
        # Check if we have a valid pick event with data points
        if hasattr(event, 'ind') and len(event.ind) > 0:
            # Get the artist that was picked
            artist = event.artist
            
            # Determine which energy was clicked
            ind = event.ind[0]  # Use the first point if multiple were picked
            
            # Extract the energy value from the offsets
            xdata = artist.get_offsets()[:, 0]
            energy = xdata[ind]
            
            # Update the plots
            update_plots(energy)
    
    # Connect the pick event to the figure
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Initial update with the first energy
    update_plots(available_energies[0])
    
    # Add legend to explain coloring
    legend_text = (
        f"Red points are within {bound_threshold*100:.0f}% of parameter bounds. "
        "Shaded regions show parameter bounds."
    )
    fig.text(0.5, 0.01, legend_text, ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Adjust layout to make room for text
    
    print("The plot is now interactive. Click on any point to update the reflectivity and SLD profile.")
    
    return fig



import os
import pandas as pd
import numpy as np
from datetime import datetime
import copy
import pickle

def batch_fit_selected_models_enhanced(objectives_dict, structures_dict, energy_list=None, 
                                     optimization_method='differential_evolution', 
                                     opt_workers=8, opt_popsize=20, burn_samples=5, 
                                     production_samples=5, prod_steps=1, pool=16,
                                     results_log=None, log_mcmc_stats=True,
                                     save_dir=None, save_objective=False, save_results=False,
                                     results_log_file=None, save_log_in_save_dir=False,
                                     save_originals=True, verbose=True):
    """
    Run fitting procedure on selected reflectometry models with option to save original objectives.
    
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
        save_originals (bool): Whether to save original objectives for comparison
        verbose (bool): Whether to print detailed progress information
        
    Returns:
        tuple: (results_dict, updated_results_df, original_objectives_dict)
            - results_dict: Dictionary mapping energy values to fitting results
            - updated_results_df: Combined DataFrame of all fitting results
            - original_objectives_dict: Dictionary of original (pre-fitting) objectives
    """
    # Dictionary to store fitting results
    results_dict = {}
    
    # Save original objectives if requested
    original_objectives_dict = {}
    if save_originals:
        if verbose:
            print("Making deep copies of original objectives for comparison...")
        for energy, obj in objectives_dict.items():
            original_objectives_dict[energy] = copy.deepcopy(obj)
    
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
        return {}, results_log, original_objectives_dict
    
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
        objective = objectives_dict[energy]
        structure = structures_dict[energy]
        
        # Extract model name if available
        model_name = getattr(objective.model, 'name', f"Model_{energy}eV")
        if verbose:
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
            if verbose:
                print(f"Starting optimization using {optimization_method}...")
            if optimization_method == 'differential_evolution':
                fitter.fit(optimization_method, workers=opt_workers, popsize=opt_popsize)
            else:
                fitter.fit(optimization_method)
            
            # Store optimization results
            results['optimized_parameters'] = objective.parameters.pvals.copy()
            results['optimized_chi_squared'] = objective.chisqr()
            
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
                print(f"Logging results for model {model_name}")
            
            # Import log_fitting_results if not already available in the namespace
            try:
                from Model_Setup import log_fitting_results
            except ImportError:
                # Define a simplified version if not available
                def log_fitting_results(objective, model_name, results_df=None):
                    """Simplified function to log fitting results"""
                    # Initialize a new DataFrame if none is provided
                    if results_df is None:
                        results_df = pd.DataFrame(columns=[
                            'timestamp', 'model_name', 'goodness_of_fit', 
                            'parameter', 'value', 'stderr', 'bound_low', 'bound_high', 'vary'
                        ])
                    
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
                    
                    # Process each parameter
                    for param in model.parameters.flattened():
                        try:
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
                        except Exception as e:
                            print(f"Error processing parameter: {str(e)}")
                            continue
                    
                    # Add new rows to the DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
                    
                    return results_df, model_name
            
            # Log the optimized values first
            results_log, updated_model_name = log_fitting_results(objective, model_name, results_log)
            
            # If MCMC was performed and we want to log those stats, create a second entry
            if log_mcmc_stats and results['mcmc_stats'] is not None:
                if verbose:
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
                            pickle.dump(objective, f)
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
                            'objective': objective if save_objective else None,
                            'model_name': updated_model_name,
                            'timestamp': timestamp,
                            'energy': energy
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
            
            # Store the results in the dictionary
            results_dict[energy] = results
            
            # Update the model name in the objective
            objective.model.name = updated_model_name
            
        except Exception as e:
            if verbose:
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
        print(f"Completed fitting for {len(results_dict)} models.")
    
    return results_dict, results_log, original_objectives_dict


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import copy
from IPython.display import display, clear_output


def visualize_before_after_fitting(results_dict, original_objectives, structures_dict, 
                                  energy_list=None, figsize=(16, 12), 
                                  max_cols=3, title_prefix=""):
    """
    Visualize before and after fitting results for multiple energies in a grid layout.
    
    Args:
        results_dict (dict): Dictionary returned by batch_fit_selected_models_enhanced
        original_objectives (dict): Dictionary mapping energy values to original Objective objects
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to include. If None, uses all available.
        figsize (tuple): Figure size as (width, height)
        max_cols (int): Maximum number of columns in the grid
        title_prefix (str): Optional prefix for plot titles
        
    Returns:
        matplotlib.figure.Figure: Figure with before/after comparison plots
    """
    # Extract fitted objectives from results
    fitted_objectives = {}
    
    for energy, result in results_dict.items():
        if 'objective' in result:
            fitted_objectives[energy] = result['objective']
    
    # Determine the available energies
    if energy_list is None:
        available_energies = sorted(list(
            set(original_objectives.keys()) & 
            set(fitted_objectives.keys()) & 
            set(structures_dict.keys())
        ))
    else:
        # Filter by available data
        available_energies = []
        for energy in energy_list:
            if (energy in original_objectives and 
                energy in fitted_objectives and 
                energy in structures_dict):
                available_energies.append(energy)
    
    if not available_energies:
        print("No valid energies found for comparison.")
        return None
    
    # Determine grid layout
    n_energies = len(available_energies)
    n_cols = min(n_energies, max_cols)
    n_rows = (n_energies + n_cols - 1) // n_cols  # Ceiling division
    
    # Increase height slightly for extra row of profile plots
    adjusted_figsize = (figsize[0], figsize[1] + n_rows * 2)
    
    # Create figure
    fig = plt.figure(figsize=adjusted_figsize)
    
    # Create a more complex GridSpec to accommodate both reflectivity and SLD profiles
    gs = GridSpec(n_rows * 2, n_cols, figure=fig, height_ratios=np.ones(n_rows * 2).tolist())
    
    # Track total chi-squared values for overall statistics
    total_original_chi = 0
    total_fitted_chi = 0
    
    # Plot reflectivity and SLD profiles for each energy
    for i, energy in enumerate(available_energies):
        # Calculate position in grid (each energy takes two rows)
        row = (i // n_cols) * 2  # Multiply by 2 for reflectivity and profile rows
        col = i % n_cols
        
        # Create axes
        ax_refl = fig.add_subplot(gs[row, col])
        ax_profile = fig.add_subplot(gs[row+1, col])
        
        # Get objectives and structure
        original_obj = original_objectives[energy]
        fitted_obj = fitted_objectives[energy]
        structure = structures_dict[energy]
        
        # Calculate chi-squared values
        original_chi = original_obj.chisqr()
        fitted_chi = fitted_obj.chisqr()
        improvement = (original_chi - fitted_chi) / original_chi * 100
        
        # Add to totals for overall statistics
        total_original_chi += original_chi
        total_fitted_chi += fitted_chi
        
        # Plot reflectivity data
        data = original_obj.data
        
        # Original model
        ax_refl.plot(data.data[0], data.data[1], 'o', label='Data', markersize=2, color='black')
        ax_refl.plot(data.data[0], original_obj.model(data.data[0]), '-', 
                   label=f'Original (χ²: {original_chi:.2f})', linewidth=2, color='blue', alpha=0.7)
        
        # Fitted model
        ax_refl.plot(data.data[0], fitted_obj.model(data.data[0]), '--', 
                   label=f'Fitted (χ²: {fitted_chi:.2f})', linewidth=2, color='red')
        
        ax_refl.set_yscale('log')
        ax_refl.set_title(f"{title_prefix}{energy} eV ({improvement:.1f}% better)")
        
        # Only add x-axis label for bottom row
        if row == (n_rows-1) * 2:
            ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
        
        # Add y-axis label only for leftmost column
        if col == 0:
            ax_refl.set_ylabel('Reflectivity (a.u)')
            
        ax_refl.legend(loc='lower left', fontsize='x-small')
        ax_refl.grid(True, alpha=0.3)
        
        # Plot SLD profiles
        from Plotting_Refl import profileflip
        
        # Apply profile shift
        profile_shift = -20
        
        try:
            # Original profile
            # Make a separate copy of the structure with original parameters
            orig_structure = copy.deepcopy(structure)
            
            # Copy original parameters to the structure
            for param in original_obj.parameters.flattened():
                # Find matching parameter in structure
                for struct_param in orig_structure.parameters.flattened():
                    if param.name == struct_param.name:
                        struct_param.value = param.value
                        break
            
            # Calculate SLD profiles
            Real_depth, Real_SLD, Imag_Depth, Imag_SLD = profileflip(orig_structure, depth_shift=0)
            Real_depth = Real_depth + profile_shift
            Imag_Depth = Imag_Depth + profile_shift
            
            # Plot original profiles
            ax_profile.plot(Real_depth, Real_SLD, '-', color='blue', 
                          label='Original Real', linewidth=2, alpha=0.7)
            ax_profile.plot(Imag_Depth, Imag_SLD, '--', color='blue', 
                          label='Original Imag', linewidth=2, alpha=0.7)
            
            # Fitted profile
            # Make a copy to avoid changing the original
            fitted_structure = copy.deepcopy(structure)
            
            # Copy fitted parameters to the structure
            for param in fitted_obj.parameters.flattened():
                # Find matching parameter in structure
                for struct_param in fitted_structure.parameters.flattened():
                    if param.name == struct_param.name:
                        struct_param.value = param.value
                        break
            
            # Calculate SLD profiles for fitted structure
            Real_depth_fit, Real_SLD_fit, Imag_Depth_fit, Imag_SLD_fit = profileflip(fitted_structure, depth_shift=0)
            Real_depth_fit = Real_depth_fit + profile_shift
            Imag_Depth_fit = Imag_Depth_fit + profile_shift
            
            # Plot fitted profiles
            ax_profile.plot(Real_depth_fit, Real_SLD_fit, '-', color='red', 
                          label='Fitted Real', linewidth=2)
            ax_profile.plot(Imag_Depth_fit, Imag_SLD_fit, '--', color='red', 
                          label='Fitted Imag', linewidth=2)
        except Exception as e:
            print(f"Error calculating SLD profiles for energy {energy}: {e}")
            ax_profile.text(0.5, 0.5, "SLD profile calculation error", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax_profile.transAxes)
        
        # Only add x-axis label for bottom row
        if row == (n_rows-1) * 2:
            ax_profile.set_xlabel(r'Distance from Si ($\AA$)')
            
        # Add y-axis label only for leftmost column
        if col == 0:
            ax_profile.set_ylabel(r'SLD $(10^{-6})$ $\AA^{-2}$')
            
        ax_profile.legend(loc='best', fontsize='x-small')
        ax_profile.grid(True, alpha=0.3)
    
    # Calculate overall improvement
    if total_original_chi > 0:
        overall_improvement = (total_original_chi - total_fitted_chi) / total_original_chi * 100
        plt.suptitle(f"Batch Fitting Results - Overall improvement: {overall_improvement:.1f}%", 
                    fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add space for title
    return fig


def create_vscode_before_after_explorer(results_dict, original_objectives_dict, structures_dict, 
                                       energy_list=None, material_name=None):
    """
    Create an interactive VSCode-friendly visualization to explore before/after fitting results.
    
    Args:
        results_dict (dict): Dictionary returned by batch_fit_selected_models_enhanced
        original_objectives_dict (dict): Dictionary of original objectives
        structures_dict (dict): Dictionary mapping energy values to Structure objects
        energy_list (list, optional): List of energies to include. If None, uses all available.
        material_name (str, optional): Optional material name to display in title
        
    Returns:
        matplotlib.figure.Figure: Interactive figure for exploring fitting results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # Extract fitted objectives from results
    fitted_objectives = {}
    for energy, result in results_dict.items():
        if 'objective' in result:
            fitted_objectives[energy] = result['objective']
    
    # Determine the available energies
    if energy_list is None:
        available_energies = sorted(list(
            set(original_objectives_dict.keys()) & 
            set(fitted_objectives.keys()) & 
            set(structures_dict.keys())
        ))
    else:
        # Filter by available data
        available_energies = []
        for energy in energy_list:
            if (energy in original_objectives_dict and 
                energy in fitted_objectives and 
                energy in structures_dict):
                available_energies.append(energy)
    
    if not available_energies:
        print("No valid energies found for comparison.")
        return None
    
    # Create figure with complex layout
    fig = plt.figure(figsize=(14, 12))
    
    # Row 1: Energy selection
    # Row 2: Reflectivity
    # Row 3: SLD profiles
    # Row 4: Parameter comparison
    gs = GridSpec(4, 1, figure=fig, height_ratios=[0.5, 1, 1, 1.5])
    
    # Create axes
    ax_energy = fig.add_subplot(gs[0])
    ax_refl = fig.add_subplot(gs[1])
    ax_profile = fig.add_subplot(gs[2])
    ax_params = fig.add_subplot(gs[3])
    
    # Create vertical line for energy selection
    current_energy = available_energies[0]
    energy_line = ax_energy.axvline(x=current_energy, color='red', linestyle='-', linewidth=2)
    
    # Create scatter plot for energies with improvement as color
    improvements = []
    for energy in available_energies:
        original_chi = original_objectives_dict[energy].chisqr()
        fitted_chi = fitted_objectives[energy].chisqr()
        improvement = (original_chi - fitted_chi) / original_chi * 100
        improvements.append(improvement)
    
    # Create a colormap from red to green based on improvement
    if improvements:
        # Create a normalization for the color scale
        norm = plt.Normalize(min(0, min(improvements)), max(50, max(improvements)))
        
        # Create the scatter plot
        scatter = ax_energy.scatter(
            available_energies, 
            np.ones_like(available_energies),  # All at same y-position
            c=improvements, 
            cmap='RdYlGn',
            norm=norm,
            s=100,
            picker=5,  # Make points pickable
            zorder=10
        )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax_energy, orientation='vertical', pad=0.01)
        cbar.set_label('Improvement (%)')
        
        # Set axis limits and labels
        ax_energy.set_xlim(min(available_energies) - 2, max(available_energies) + 2)
        ax_energy.set_ylim(0.5, 1.5)
        ax_energy.set_yticks([])  # Hide y-axis
        ax_energy.set_xlabel('Energy (eV)')
        ax_energy.set_title('Select Energy (Color indicates fitting improvement)')
        
        # Add energy labels
        for energy, improvement in zip(available_energies, improvements):
            # Position text above or below based on position
            y_pos = 1.2 if energy % 5 == 0 else 0.8
            ax_energy.annotate(
                f"{energy}",
                xy=(energy, 1),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center',
                fontsize=8
            )
    
    # Create title with material name if provided
    if material_name:
        fig.suptitle(f"{material_name} - Before/After Fitting Comparison", fontsize=16)
    else:
        fig.suptitle("Before/After Fitting Comparison", fontsize=16)
    
    # Function to update the plots for a given energy
    def update_plots(energy):
        nonlocal current_energy
        current_energy = energy
        
        # Update energy selection line
        energy_line.set_xdata([energy, energy])
        
        # Get the objectives and structure
        original_obj = original_objectives_dict[energy]
        fitted_obj = fitted_objectives[energy]
        structure = structures_dict[energy]
        
        # Calculate chi-squared values
        original_chi = original_obj.chisqr()
        fitted_chi = fitted_obj.chisqr()
        improvement = (original_chi - fitted_chi) / original_chi * 100
        
        # Update reflectivity plot
        ax_refl.clear()
        data = original_obj.data
        
        # Original model
        ax_refl.plot(data.data[0], data.data[1], 'o', label='Data', markersize=2, color='black')
        ax_refl.plot(data.data[0], original_obj.model(data.data[0]), '-', 
                   label=f'Original (χ²: {original_chi:.4f})', linewidth=2, color='blue', alpha=0.7)
        
        # Fitted model
        ax_refl.plot(data.data[0], fitted_obj.model(data.data[0]), '--', 
                   label=f'Fitted (χ²: {fitted_chi:.4f})', linewidth=2, color='red')
        
        ax_refl.set_yscale('log')
        ax_refl.set_title(f"Reflectivity - {energy} eV ({improvement:.1f}% better)")
        ax_refl.set_xlabel(r'Q ($\AA^{-1}$)')
        ax_refl.set_ylabel('Reflectivity (a.u)')
        ax_refl.legend(loc='best')
        ax_refl.grid(True, alpha=0.3)
        
        # Update SLD profile plot
        ax_profile.clear()
        
        # Apply profile shift
        profile_shift = -20
        
        try:
            # Original profile
            from Plotting_Refl import profileflip
            orig_structure = copy.deepcopy(structure)
            
            # Copy original parameters to the structure
            for param in original_obj.parameters.flattened():
                # Find matching parameter in structure
                for struct_param in orig_structure.parameters.flattened():
                    if param.name == struct_param.name:
                        struct_param.value = param.value
                        break
            
            # Calculate SLD profiles
            Real_depth, Real_SLD, Imag_Depth, Imag_SLD = profileflip(orig_structure, depth_shift=0)
            Real_depth = Real_depth + profile_shift
            Imag_Depth = Imag_Depth + profile_shift
            
            # Plot original profiles
            ax_profile.plot(Real_depth, Real_SLD, '-', color='blue', 
                          label='Original Real', linewidth=2, alpha=0.7)
            ax_profile.plot(Imag_Depth, Imag_SLD, '--', color='blue', 
                          label='Original Imag', linewidth=2, alpha=0.7)
            
            # Fitted profile
            fitted_structure = copy.deepcopy(structure)
            
            # Copy fitted parameters to the structure
            for param in fitted_obj.parameters.flattened():
                # Find matching parameter in structure
                for struct_param in fitted_structure.parameters.flattened():
                    if param.name == struct_param.name:
                        struct_param.value = param.value
                        break
            
            # Calculate SLD profiles for fitted structure
            Real_depth_fit, Real_SLD_fit, Imag_Depth_fit, Imag_SLD_fit = profileflip(fitted_structure, depth_shift=0)
            Real_depth_fit = Real_depth_fit + profile_shift
            Imag_Depth_fit = Imag_Depth_fit + profile_shift
            
            # Plot fitted profiles
            ax_profile.plot(Real_depth_fit, Real_SLD_fit, '-', color='red', 
                          label='Fitted Real', linewidth=2)
            ax_profile.plot(Imag_Depth_fit, Imag_SLD_fit, '--', color='red', 
                          label='Fitted Imag', linewidth=2)
            
            ax_profile.set_xlabel(r'Distance from Si ($\AA$)')
            ax_profile.set_ylabel(r'SLD $(10^{-6})$ $\AA^{-2}$')
            ax_profile.set_title(f'SLD Profile - {energy} eV')
            ax_profile.legend(loc='best')
            ax_profile.grid(True, alpha=0.3)
            
        except Exception as e:
            ax_profile.text(0.5, 0.5, f"Error calculating SLD profiles: {str(e)}",
                         ha='center', va='center', transform=ax_profile.transAxes)
            ax_profile.set_title('SLD Profile - Error')
        
        # Update parameters plot
        ax_params.clear()
        
        # Get all parameters that were varied
        varied_params = []
        param_names = []
        orig_values = []
        fitted_values = []
        percent_changes = []
        
        for param_fitted in fitted_obj.parameters.flattened():
            # Only look at parameters that were varied or are important SLD parameters
            include_param = False
            
            if hasattr(param_fitted, 'vary') and param_fitted.vary:
                include_param = True
            elif any(x in param_fitted.name for x in ['sld', 'isld', 'thick']):
                include_param = True
                
            if not include_param:
                continue
            
            # Find the same parameter in the original objective
            for param_orig in original_obj.parameters.flattened():
                if param_orig.name == param_fitted.name:
                    # Calculate percent change
                    if param_orig.value != 0:
                        percent_change = (param_fitted.value - param_orig.value) / abs(param_orig.value) * 100
                    else:
                        percent_change = 0 if param_fitted.value == 0 else 100  # Arbitrary 100% if from 0 to non-zero
                    
                    varied_params.append(param_fitted)
                    param_names.append(param_fitted.name)
                    orig_values.append(param_orig.value)
                    fitted_values.append(param_fitted.value)
                    percent_changes.append(percent_change)
                    
                    break
        
        # If we have parameters to show
        if param_names:
            # Sort by percent change
            indices = np.argsort(np.abs(percent_changes))[::-1]  # Descending order
            param_names = [param_names[i] for i in indices]
            orig_values = [orig_values[i] for i in indices]
            fitted_values = [fitted_values[i] for i in indices]
            percent_changes = [percent_changes[i] for i in indices]
            
            # Limit to top 15 parameters for readability
            if len(param_names) > 15:
                param_names = param_names[:15]
                orig_values = orig_values[:15]
                fitted_values = fitted_values[:15]
                percent_changes = percent_changes[:15]
            
            # Create barplot of percent changes
            x_pos = np.arange(len(param_names))
            bars = ax_params.bar(x_pos, percent_changes, 
                              color=['green' if x >= 0 else 'red' for x in percent_changes])
            
            # Add value labels
            for i, (bar, orig, fitted) in enumerate(zip(bars, orig_values, fitted_values)):
                if abs(percent_changes[i]) > 2:  # Only label significant changes
                    # Position text at the top/bottom of the bar
                    height = bar.get_height()
                    text_pos = height + 1 if height >= 0 else height - 3
                    ax_params.text(bar.get_x() + bar.get_width()/2., text_pos,
                                f'{fitted:.2g}\n({percent_changes[i]:+.1f}%)',
                                ha='center', va='bottom' if height >= 0 else 'top', 
                                fontsize=8, rotation=45)
            
            # Customize the plot
            ax_params.set_xticks(x_pos)
            ax_params.set_xticklabels(param_names, rotation=45, ha='right')
            ax_params.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_params.set_ylabel('Parameter Change (%)')
            ax_params.set_title('Parameter Changes')
            
            # Add grid
            ax_params.grid(True, axis='y', alpha=0.3)
            
            # Adjust y-limits to better fit the data
            max_change = max(abs(min(percent_changes)), abs(max(percent_changes)))
            y_limit = min(100, max(20, max_change * 1.2))  # Cap at 100%, but at least 20%
            ax_params.set_ylim(-y_limit, y_limit)
        else:
            ax_params.text(0.5, 0.5, "No varied parameters found",
                        ha='center', va='center', transform=ax_params.transAxes)
            ax_params.set_title('Parameter Changes')
        
        # Update the figure
        fig.canvas.draw_idle()
    
    # Function to handle pick events (clicking on energy points)
    def on_pick(event):
        # Check if we have a valid pick event with data points
        if hasattr(event, 'ind') and len(event.ind) > 0:
            # Get the index of the clicked point in the scatter plot
            ind = event.ind[0]
            
            # Get the energy value
            energy = available_energies[ind]
            
            # Update the plots
            update_plots(energy)
    
    # Connect the pick event to the figure
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Key press handler for navigating with arrow keys
    def on_key(event):
        if event.key == 'right':
            # Go to next energy
            current_index = available_energies.index(current_energy)
            if current_index < len(available_energies) - 1:
                update_plots(available_energies[current_index + 1])
        elif event.key == 'left':
            # Go to previous energy
            current_index = available_energies.index(current_energy)
            if current_index > 0:
                update_plots(available_energies[current_index - 1])
    
    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial update
    update_plots(available_energies[0])
    
    plt.tight_layout()
    
    print("Interactive plot ready. Click on energy points to view details. Use left/right arrow keys to navigate.")
    
    return fig


def export_best_parameters(results_dict, original_objectives_dict, output_file=None):
    """
    Export the best-fit parameters from a batch fitting session.
    
    Args:
        results_dict (dict): Dictionary returned by batch_fit_selected_models_enhanced
        original_objectives_dict (dict): Dictionary of original objectives
        output_file (str, optional): Path to save CSV file. If None, returns DataFrame only.
        
    Returns:
        pandas.DataFrame: DataFrame with best-fit parameters and improvement statistics
    """
    import pandas as pd
    
    # Extract fitted objectives from results
    fitted_objectives = {}
    for energy, result in results_dict.items():
        if 'objective' in result:
            fitted_objectives[energy] = result['objective']
    
    # Initialize lists for DataFrame rows
    rows = []
    
    # Process each energy
    for energy in sorted(fitted_objectives.keys()):
        if energy not in original_objectives_dict:
            continue
            
        fitted_obj = fitted_objectives[energy]
        original_obj = original_objectives_dict[energy]
        
        # Calculate improvement
        original_chi = original_obj.chisqr()
        fitted_chi = fitted_obj.chisqr()
        improvement = (original_chi - fitted_chi) / original_chi * 100
        
        # Extract all parameter values
        for param in fitted_obj.parameters.flattened():
            # Get parameter name and value
            param_name = param.name
            value = param.value
            stderr = getattr(param, 'stderr', None)
            vary = getattr(param, 'vary', False)
            
            # Get original value
            original_value = None
            for orig_param in original_obj.parameters.flattened():
                if orig_param.name == param_name:
                    original_value = orig_param.value
                    break
            
            # Calculate percent change
            if original_value is not None and original_value != 0:
                percent_change = (value - original_value) / abs(original_value) * 100
            else:
                percent_change = None
            
            # Create a row for this parameter
            row = {
                'energy': energy,
                'parameter': param_name,
                'original_value': original_value,
                'fitted_value': value,
                'stderr': stderr,
                'percent_change': percent_change,
                'varied': vary,
                'original_chi2': original_chi,
                'fitted_chi2': fitted_chi,
                'improvement_percent': improvement
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV if output file is specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Exported parameters to {output_file}")
    
    return df