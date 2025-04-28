
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
    
    
