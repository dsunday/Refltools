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


def create_reflectometry_model_unified(materials_list, layer_params, layer_order=None, ignore_layers=None, 
                                     sample_name=None, energy=None, wavelength=None, probe="x-ray"):
    """
    Create a complete reflectometry model with unified support for both SLD and density inputs.
    Automatically detects whether materials use SLD (real/imag) or density approach.
    Uses pandas database approach - optimization logger is not used.
    
    Args:
        materials_list: List of dictionaries with 'name' and either:
            - SLD approach: 'real' and 'imag' values
            - Density approach: 'density' and 'formula' values
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
    
    # Step 1: Create materials - automatically detect SLD vs density approach
    materials = {}
    approach_type = None  # Track which approach is being used
    
    for material_info in materials_list:
        name = material_info['name']
        
        # Detect approach type from first material and validate consistency
        if 'real' in material_info and 'imag' in material_info:
            # SLD approach
            if approach_type is None:
                approach_type = "SLD"
            elif approach_type != "SLD":
                raise ValueError("Mixed approaches detected. All materials must use either SLD (real/imag) or density approach, not both.")
            
            real = material_info['real']
            imag = material_info['imag']
            
            # Handle complex numbers if imag is already complex
            if isinstance(imag, complex):
                materials[name] = SLD(real + imag, name=name)
            else:
                materials[name] = SLD(real + imag*1j, name=name)
                
        elif 'density' in material_info:
            # Density approach
            if approach_type is None:
                approach_type = "density"
            elif approach_type != "density":
                raise ValueError("Mixed approaches detected. All materials must use either SLD (real/imag) or density approach, not both.")
            
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
            # Missing required parameters
            raise ValueError(f"Material {name} must have either 'real'/'imag' (SLD approach) or 'density' (density approach) values")
    
    print(f"Detected approach: {approach_type}")
    
    # Step 2: Create layers with consolidated parameters
    Layer = {}
    
    # Track which parameters are being varied for model naming
    if approach_type == "SLD":
        varying_params = {
            "R": set(),  # Real SLD
            "I": set(),  # Imaginary SLD
            "T": set(),  # Thickness
            "Rg": set() # Roughness
        }
    else:  # density approach
        varying_params = {
            "D": set(),  # Density
            "T": set(),  # Thickness
            "Rg": set() # Roughness
        }
    
    # Track which materials have varying parameters
    materials_varying = set()
    
    for name, params in layer_params.items():
        if name in materials:
            thickness = params.get("thickness", 0)
            roughness = params.get("roughness", 0)
            
            # Create the layer using the material and parameters
            Layer[name] = materials[name](thickness, roughness)
            
            has_varying_param = False
            
            if approach_type == "SLD":
                # Handle SLD-based parameters
                
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
                        
            else:  # density approach
                # Handle density-based parameters
                
                # Apply density bounds and track variations
                if "density_bounds" in params:
                    lower, upper, vary = params["density_bounds"]
                    
                    # Get the density parameter if it exists
                    if name in density_params and isinstance(materials[name], MaterialSLD):
                        density_param = density_params[name]
                        density_param.setp(bounds=(lower, upper), vary=vary)
                        
                        # For MaterialSLD, link the density parameter directly
                        material_obj = materials[name]
                        material_obj.density = density_param
                        
                        if vary and name not in ignore_layers:
                            varying_params["D"].add(name)
                            has_varying_param = True
                        
            # Handle common parameters (thickness and roughness) for both approaches
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
            
        # Ensure structure has wavelength information
        structure.wavelength = wavelength
    
    # Generate simplified model name
    # Count layers except those in ignore_layers
    active_layers = [layer for layer in layer_order if layer not in ignore_layers and layer != "air"]
    num_layers = len(active_layers)
    
    # Construct the model name with the new format: Sample_Layers
    if sample_name is None:
        sample_name = "Unknown"  # Default sample name if not provided
    
    model_name = f"{sample_name}{num_layers}Layers"
    
    # Return additional objects for fitting
    return materials, Layer, structure, model_name


def create_materials(materials_list):
    """
    Legacy function for backward compatibility with SLD approach.
    Create multiple SLD objects from a list of material definitions.
    
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
        materials[name] = SLD(real + imag, name=name)
    
    return materials


def create_layers(materials, thicknesses, roughnesses):
    """
    Legacy function for backward compatibility.
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
        
        # Create the Layer with the material and parameters
        layers[name] = material(thickness, roughness)
    
    return layers


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
    
    This version supports both density and SLD parameters.
    
    Args:
        objective: The objective function that was fitted
        model_name: Base name for the model
        results_df: Existing pandas DataFrame to append to (None creates new one)
        
    Returns:
        Updated pandas DataFrame with fitting results
    """
    
    if results_df is None:
        results_df = pd.DataFrame(columns=['timestamp', 'model_name', 'parameter', 'value', 'stderr', 
                                         'vary', 'bound_low', 'bound_high', 'goodness_of_fit', 'param_type'])
    
    # Check if this model name already exists and add suffix if needed
    existing_names = results_df['model_name'].unique()
    original_name = model_name
    counter = 1
    while model_name in existing_names:
        model_name = f"{original_name}_{counter}"
        counter += 1
    
    # Get current timestamp
    timestamp = datetime.now()
    
    # Calculate goodness of fit
    try:
        gof = objective.chisqr()
    except:
        gof = None
    
    # Extract parameter information
    new_rows = []
    for param in objective.parameters.varying_parameters():
        param_type = get_param_type(param.name)
        
        new_row = {
            'timestamp': timestamp,
            'model_name': model_name,
            'parameter': param.name,
            'value': param.value,
            'stderr': param.stderr if hasattr(param, 'stderr') else None,
            'vary': param.vary,
            'bound_low': param.bounds.lb if param.bounds is not None else None,
            'bound_high': param.bounds.ub if param.bounds is not None else None,
            'goodness_of_fit': gof,
            'param_type': param_type
        }
        new_rows.append(new_row)
    
    # Add new rows to the DataFrame
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        results_df = pd.concat([results_df, new_df], ignore_index=True)
    
    return results_df


def create_empty_results_database():
    """
    Create an empty pandas DataFrame with the standard schema for storing fitting results.
    Supports both SLD and density approaches.
    
    Returns:
        Empty pandas DataFrame with appropriate columns
    """
    return pd.DataFrame(columns=[
        'timestamp', 'model_name', 'parameter', 'value', 'stderr', 
        'vary', 'bound_low', 'bound_high', 'goodness_of_fit', 'param_type',
        'energy', 'wavelength', 'probe'  # Additional columns for energy-dependent fits
    ])


def save_database_to_file(results_df, filename):
    """
    Save the results database to a CSV file.
    
    Args:
        results_df: pandas DataFrame containing results
        filename: Path to save the CSV file
    """
    try:
        results_df.to_csv(filename, index=False)
        print(f"Database saved to {filename}")
    except Exception as e:
        print(f"Error saving database: {str(e)}")


def load_database_from_file(filename):
    """
    Load the results database from a CSV file.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        pandas DataFrame with results
    """
    try:
        results_df = pd.read_csv(filename)
        # Convert timestamp column to datetime if it exists
        if 'timestamp' in results_df.columns:
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        print(f"Database loaded from {filename}")
        return results_df
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return create_empty_results_database()


def run_fitting_v2(objective, method='differential_evolution', 
                   workers=-1, popsize=15, steps=1000, burn=500,
                   nthin=1, nwalkers=100,
                   results_database=None, log_mcmc_stats=True,
                   save_dir=None, save_objective=False, save_results=False,
                   structure=None, model_name=None, add_to_database=True,
                   verbose=False, energy=None):
    """
    Run fitting procedure on a reflectometry model with optimization and MCMC sampling,
    with option to add results to a pandas database. Supports both SLD and density approaches.
    
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
        Tuple of (results_dict, updated_results_database)
    """
    
    print(f"Starting fitting procedure for model: {model_name}")
    
    # Initialize results database if not provided
    if results_database is None:
        results_database = create_empty_results_database()
    
    # Set up the fitter
    fitter = CurveFitter(objective)
    
    # Run optimization
    if verbose:
        print(f"Running optimization using {method}...")
    
    if method == 'differential_evolution':
        optimization_result = fitter.fit(method='differential_evolution', 
                                       workers=workers, popsize=popsize)
    else:
        optimization_result = fitter.fit(method=method)
    
    if verbose:
        print(f"Optimization completed. Success: {optimization_result.success}")
        print(f"Chi-squared: {objective.chisqr():.6g}")
    
    # Run MCMC sampling
    if steps > 0:
        if verbose:
            print(f"Running MCMC sampling with {steps} steps...")
        
        # Set up and run MCMC
        fitter.sample(steps=steps, nthin=nthin, nwalkers=nwalkers, verbose=verbose)
        
        # Process MCMC results
        if burn > 0:
            fitter.sampler.reset()  # Remove burn-in samples
        
        if verbose:
            print("MCMC sampling completed.")
    
    # Create results dictionary
    results_dict = {
        'optimization_result': optimization_result,
        'objective': objective,
        'structure': structure,
        'model_name': model_name,
        'energy': energy,
        'chisqr': objective.chisqr(),
        'timestamp': datetime.now()
    }
    
    if steps > 0:
        results_dict['sampler'] = fitter.sampler
        results_dict['mcmc_steps'] = steps
        results_dict['burn_steps'] = burn
    
    # Add results to database
    if add_to_database:
        results_database = log_fitting_results(objective, model_name, results_database)
    
    # Save files if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        if save_objective:
            obj_filename = os.path.join(save_dir, f"{model_name}_objective.pkl")
            try:
                with open(obj_filename, 'wb') as f:
                    pickle.dump(objective, f)
                if verbose:
                    print(f"Saved objective to {obj_filename}")
            except Exception as e:
                print(f"Error saving objective: {str(e)}")
        
        if save_results:
            results_filename = os.path.join(save_dir, f"{model_name}_results.pkl")
            try:
                with open(results_filename, 'wb') as f:
                    pickle.dump(results_dict, f)
                if verbose:
                    print(f"Saved results to {results_filename}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
    
    return results_dict, results_database


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