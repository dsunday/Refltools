import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py
import json
from scipy.stats import linregress
from scipy.interpolate import interp1d
import os
import glob

class LariatDataProcessor:
    """
    A class for processing Lariat datacube files and extracting/analyzing spectra.
    """
    
    def __init__(self, filepath=None):
        """
        Initialize the LariatDataProcessor.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the HDF5 file to load immediately
        """
        self.data = None
        self.metadata = None
        self.filepath = filepath
        
        if filepath is not None:
            self.load_data(filepath)
    
    def load_data(self, filepath):
        """
        Load lariat datacube from HDF5 file.
        
        Parameters:
        -----------
        filepath : str
            Path to the HDF5 file
        """
        self.filepath = filepath
        self.data, self.metadata = self._read_lariat_datacube(filepath)
    
    def _read_lariat_datacube(self, filepath):
        """
        Read lariat datacube from HDF5 file.
        
        Parameters:
        -----------
        filepath : str
            Path to the HDF5 file
            
        Returns:
        --------
        data : xarray.DataArray
            3D xarray data with dimensions of energy, pix_x, pix_y
        metadata : dict
            Metadata from the file
        """
        f = h5py.File(filepath)
        metadata = json.loads(f['File Version'].attrs['Meta Data'])
        energies = np.arange(*metadata['BeamEnergy'])
        image_list = []
        energy_list = []
        for i, energy in enumerate(energies):
            image = f['Images'][f'Image{i}']['ImagePlane0']
            image_list.append(image)
            energy_list.append(energy)
        return xr.DataArray(image_list, dims=['energy','pix_x','pix_y'], coords={'energy':energy_list}), metadata
    
    def load_hdf5_data(self, filename, path):
        """
        Load data from a specific path in an HDF5 file.
        
        Parameters:
        -----------
        filename : str
            Path to the HDF5 file
        path : str
            Path to the dataset within the HDF5 file
            
        Returns:
        --------
        data : numpy.ndarray or h5py.Group
            The data from the specified path or a group object
        """
        with h5py.File(filename, 'r') as f:
            # Check if the path exists
            if path not in f:
                raise ValueError(f"Path '{path}' not found in the HDF5 file")
            
            # Get the item
            item = f[path]
            
            # Check if it's a dataset or a group
            if isinstance(item, h5py.Dataset):
                # If it's a dataset, load the data
                data = item[()]
                return data
            else:
                # If it's a group, print its keys and return None
                print(f"\n'{path}' is a group with the following keys:")
                for key in item.keys():
                    print(f"  - {key}")
                return None
    
    def print_structure(self, name, obj):
        """
        Print the structure of an HDF5 file.
        
        Parameters:
        -----------
        name : str
            Name of the object
        obj : h5py object
            HDF5 object to print structure for
        """
        print(name)
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                self.print_structure(name + "/" + key, obj[key])
    
    def plot_energy_slice(self, energy_value):
        """
        Plot a 2D image slice of the data at a specific energy value.
        
        Parameters:
        -----------
        energy_value : float
            The energy value to select for plotting
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Select the energy slice
        slice_data = self.data.sel(energy=energy_value, method='nearest')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = slice_data.plot(ax=ax, cmap='viridis')
        ax.set_title(f'Energy = {slice_data.energy.values:.2f}')
        
        return fig, ax
    
    def plot_with_roi(self, energy_value, roi=None):
        """
        Plot a 2D image slice with an optional ROI box.
        
        Parameters:
        -----------
        energy_value : float
            The energy value to select for plotting
        roi : tuple, optional
            (x_min, y_min, width, height) defining the ROI
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        fig, ax = self.plot_energy_slice(energy_value)
        
        # Add ROI box if provided
        if roi is not None:
            x_min, y_min, width, height = roi
            rect = Rectangle((x_min, y_min), width, height, 
                             edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            
            # Annotate the ROI
            ax.text(x_min + width/2, y_min + height + 2, 'ROI', 
                    color='red', fontweight='bold', ha='center')
        
        return fig, ax
    
    def extract_roi_spectrum(self, roi, plot=False, energy_slice=None):
        """
        Extract the average spectrum from a region of interest and optionally plot it.
        
        Parameters:
        -----------
        roi : tuple
            (x_min, y_min, width, height) defining the ROI
        plot : bool, optional
            If True, plot the extracted spectrum
        energy_slice : float, optional
            If provided, also shows the 2D image at this energy with the ROI
            
        Returns:
        --------
        spectrum_xr : xarray.DataArray
            1D xarray with the average spectrum over the ROI
        spectrum_np : numpy.ndarray
            2D numpy array with columns [Energy, Intensity]
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        x_min, y_min, width, height = roi
        x_max = x_min + width
        y_max = y_min + height
        
        # Extract the ROI for all energies
        roi_data = self.data.sel(
            pix_x=slice(x_min, x_max - 1),  # -1 because slice is inclusive on both ends
            pix_y=slice(y_min, y_max - 1)
        )
        
        # Average over the spatial dimensions
        spectrum_xr = roi_data.mean(dim=['pix_x', 'pix_y'])
        
        # Create numpy array with [Energy, Intensity] columns
        energy_values = spectrum_xr.energy.values
        intensity_values = spectrum_xr.values
        spectrum_np = np.column_stack((energy_values, intensity_values))
        
        # Plot if requested
        if plot:
            # Create a figure with one or two subplots depending on whether to show the image
            if energy_slice is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot the 2D image with ROI
                slice_data = self.data.sel(energy=energy_slice, method='nearest')
                im = slice_data.plot(ax=ax1, cmap='viridis')
                ax1.set_title(f'Energy = {slice_data.energy.values:.2f}')
                plt.colorbar(im, ax=ax1, label='Intensity')
                
                # Add ROI box
                rect = Rectangle((x_min, y_min), width, height, 
                                edgecolor='red', facecolor='none', linewidth=2)
                ax1.add_patch(rect)
                ax1.text(x_min + width/2, y_min + height + 2, 'ROI', 
                        color='red', fontweight='bold', ha='center')
                
                # Plot spectrum in second subplot
                ax2.plot(energy_values, intensity_values, 'o-')
                ax2.set_title('Average Spectrum from ROI')
                ax2.set_xlabel('Energy')
                ax2.set_ylabel('Intensity')
                ax2.grid(True)
                
                # Add a vertical line at the selected energy
                ax2.axvline(x=energy_slice, color='red', linestyle='--', alpha=0.7)
                ax2.text(energy_slice, ax2.get_ylim()[1]*0.95, 'Current Slice', 
                        color='red', ha='center', va='top', rotation=90, alpha=0.7)
                
            else:
                # Just plot the spectrum
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(energy_values, intensity_values, 'o-')
                ax.set_title('Average Spectrum from ROI')
                ax.set_xlabel('Energy')
                ax.set_ylabel('Intensity')
                ax.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return spectrum_xr, spectrum_np
    
    def subtract_pre_edge(self, spectrum_np, pre_edge_range=None):
        """
        Subtract a pre-edge value from the spectrum.
        The pre-edge is calculated as the average intensity within a specified energy range.
        
        Parameters:
        -----------
        spectrum_np : numpy.ndarray
            2D numpy array with columns [Energy, Intensity]
        pre_edge_range : tuple, optional
            (min_energy, max_energy) defining the range to calculate pre-edge from.
            If None, uses the first 10% of the energy range.
            
        Returns:
        --------
        spectrum_corrected : numpy.ndarray
            2D numpy array with columns [Energy, Pre-edge-subtracted Intensity]
        pre_edge_value : float
            The average intensity in the pre-edge range that was subtracted
        """
        # Extract energy and intensity columns
        energy = spectrum_np[:, 0]
        intensity = spectrum_np[:, 1]
        
        # Determine pre-edge range if not provided
        if pre_edge_range is None:
            e_min = energy.min()
            e_max = energy.max()
            e_range = e_max - e_min
            pre_edge_range = (e_min, e_min + 0.1 * e_range)  # Default to first 10% of range
        
        # Find indices within the pre-edge range
        pre_edge_indices = np.where((energy >= pre_edge_range[0]) & (energy <= pre_edge_range[1]))
        
        # Calculate average intensity in the pre-edge range
        pre_edge_value = np.mean(intensity[pre_edge_indices])
        
        # Subtract pre-edge value from all intensities
        intensity_corrected = intensity - pre_edge_value
        
        # Create a new array with the corrected intensity
        spectrum_corrected = np.column_stack((energy, intensity_corrected))
        
        return spectrum_corrected, pre_edge_value
    
    def normalize_by_pre_edge_slope(self, spectrum_np, fit_range=None):
        """
        Normalize a spectrum by dividing it by a line fitted to a specified pre-edge portion.
        
        Parameters:
        -----------
        spectrum_np : numpy.ndarray
            2D numpy array with columns [Energy, Intensity]
        fit_range : tuple, optional
            (min_energy, max_energy) defining the pre-edge range to fit the line to.
            If None, uses the first 20% of the energy range.
            
        Returns:
        --------
        spectrum_normalized : numpy.ndarray
            2D numpy array with columns [Energy, Normalized Intensity]
        fit_params : tuple
            (slope, intercept) of the fitted pre-edge line
        """
        # Extract energy and intensity columns
        energy = spectrum_np[:, 0]
        intensity = spectrum_np[:, 1]
        
        # Determine fit range if not provided
        if fit_range is None:
            e_min = energy.min()
            e_max = energy.max()
            e_range = e_max - e_min
            fit_range = (e_min, e_min + 0.2 * e_range)  # Default to first 20% of range
        
        # Find indices within the fit range
        fit_indices = np.where((energy >= fit_range[0]) & (energy <= fit_range[1]))
        
        # Fit a line to the selected pre-edge range
        slope, intercept, r_value, p_value, std_err = linregress(
            energy[fit_indices], intensity[fit_indices])
        
        # Calculate the fitted line values for all energy points
        line_values = slope * energy + intercept
        
        # Divide the spectrum by the fitted line
        # Add a small value to avoid division by zero
        epsilon = 1e-10
        intensity_normalized = intensity / (line_values + epsilon)
        
        # Create a new array with the normalized intensity
        spectrum_normalized = np.column_stack((energy, intensity_normalized))
        
        return spectrum_normalized, (slope, intercept)
    
    def normalize_by_post_edge(self, spectrum_np, post_edge_range=None):
        """
        Normalize a spectrum by dividing it by the average intensity in a post-edge region.
        
        Parameters:
        -----------
        spectrum_np : numpy.ndarray
            2D numpy array with columns [Energy, Intensity]
        post_edge_range : tuple, optional
            (min_energy, max_energy) defining the post-edge range to average.
            If None, uses the last 20% of the energy range.
            
        Returns:
        --------
        spectrum_normalized : numpy.ndarray
            2D numpy array with columns [Energy, Normalized Intensity]
        post_edge_value : float
            The average intensity in the post-edge range used for normalization
        """
        # Extract energy and intensity columns
        energy = spectrum_np[:, 0]
        intensity = spectrum_np[:, 1]
        
        # Determine post-edge range if not provided
        if post_edge_range is None:
            e_min = energy.min()
            e_max = energy.max()
            e_range = e_max - e_min
            post_edge_range = (e_max - 0.2 * e_range, e_max)  # Default to last 20% of range
        
        # Find indices within the post-edge range
        post_edge_indices = np.where((energy >= post_edge_range[0]) & (energy <= post_edge_range[1]))
        
        # Calculate average intensity in the post-edge range
        post_edge_value = np.mean(intensity[post_edge_indices])
        
        # Divide the spectrum by the post-edge average value
        # Add a small value to avoid division by zero
        epsilon = 1e-10
        intensity_normalized = intensity / (post_edge_value + epsilon)
        
        # Create a new array with the normalized intensity
        spectrum_normalized = np.column_stack((energy, intensity_normalized))
        
        return spectrum_normalized, post_edge_value
    
    def extract_izero(self, file=None):
        """
        Extract I0 data from the HDF5 file.
        
        Parameters:
        -----------
        file : str, optional
            Path to the HDF5 file. If None, uses self.filepath
            
        Returns:
        --------
        izero_array : numpy.ndarray
            2D numpy array with columns [Energy, Normalized I0 Intensity]
        """
        if file is None:
            if self.filepath is None:
                raise ValueError("No file path provided and no data loaded. Please specify a file path.")
            file = self.filepath
        
        if self.metadata is None:
            raise ValueError("No metadata loaded. Please load data first using load_data().")
        
        # Extract I0 data
        izero_raw = self.load_hdf5_data(file, 'Vectors/Vector2/Dimension0')
        
        # Get energies from metadata
        energies = np.arange(*self.metadata['BeamEnergy'])
        
        # Debug: print shapes to understand the mismatch
        print(f"Energies shape: {energies.shape}")
        print(f"I0 raw shape: {izero_raw.shape}")
        
        # Handle shape mismatch - take the minimum length
        min_length = min(len(energies), len(izero_raw))
        energies_trimmed = energies[:min_length]
        izero_trimmed = izero_raw[:min_length]
        
        print(f"Using length: {min_length}")
        
        # Stack energies and I0 data
        izero_array = np.stack([energies_trimmed, izero_trimmed]).T
        
        # Delete spurious first data point
        izero_array = np.delete(izero_array, (0), axis=0)
        
        # Normalize I0 by its minimum value
        izero_array[:, 1] = izero_array[:, 1] / izero_array[:, 1].min()
        
        return izero_array
    
    def normalize_by_izero(self, izero_array=None, interpolate=True, extract_from_file=True):
        """
        Normalize xarray data by dividing each energy slice by the corresponding I0 value.
        
        Parameters:
        -----------
        izero_array : numpy.ndarray, optional
            2D numpy array with columns [Energy, Intensity] where Intensity represents I0 values.
            If None and extract_from_file is True, will extract I0 from the loaded file.
        interpolate : bool, optional
            If True, interpolate the I0 values to match the energy coordinates of the xarray data
        extract_from_file : bool, optional
            If True and izero_array is None, extract I0 data from the loaded file
            
        Returns:
        --------
        None (modifies self.data in place)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Extract I0 data if not provided
        if izero_array is None:
            if extract_from_file:
                izero_array = self.extract_izero()
            else:
                raise ValueError("No I0 data provided and extract_from_file is False. Please provide izero_array or set extract_from_file=True.")
        
        # Extract energy coordinates from xarray
        xarray_energies = self.data.energy.values
        
        # Extract energies and I0 values from izero_array
        izero_energies = izero_array[:, 0]
        izero_values = izero_array[:, 1]
        
        # Check if interpolation is needed
        if interpolate or not np.array_equal(xarray_energies, izero_energies):
            # Create interpolation function for I0 values
            if len(izero_energies) < 2:
                raise ValueError("Need at least 2 points for interpolation")
            
            # Check if xarray energies are within the range of I0 energies
            if xarray_energies.min() < izero_energies.min() or xarray_energies.max() > izero_energies.max():
                print(f"Warning: Some xarray energies are outside the range of I0 energies "
                      f"({izero_energies.min():.4f} - {izero_energies.max():.4f}). "
                      f"Extrapolation will be used.")
            
            # Create interpolation function (use 'cubic' if enough points, otherwise 'linear')
            kind = 'cubic' if len(izero_energies) >= 4 else 'linear'
            f_interp = interp1d(izero_energies, izero_values, kind=kind, 
                               bounds_error=False, fill_value='extrapolate')
            
            # Interpolate I0 values to match xarray energies
            interpolated_izero = f_interp(xarray_energies)
            
            # Add a small value to avoid division by zero
            epsilon = 1e-10
            interpolated_izero = np.maximum(interpolated_izero, epsilon)
            
            # Normalize each energy slice of the xarray data
            data_normalized = self.data.copy()
            
            # Apply normalization (divide each energy slice by the corresponding I0 value)
            for i, energy in enumerate(xarray_energies):
                # Division using broadcasting - each pixel at this energy gets divided by the same I0 value
                data_normalized.loc[dict(energy=energy)] = self.data.loc[dict(energy=energy)] / interpolated_izero[i]
        
        else:
            # If energies match exactly and no interpolation needed
            # Add a small value to avoid division by zero
            epsilon = 1e-10
            safe_izero = np.maximum(izero_values, epsilon)
            
            # Normalize each energy slice of the xarray data
            data_normalized = self.data.copy()
            
            # Apply normalization using exact matching
            for i, energy in enumerate(xarray_energies):
                data_normalized.loc[dict(energy=energy)] = self.data.loc[dict(energy=energy)] / safe_izero[i]
        
        # Copy attributes from original data
        data_normalized.attrs.update(self.data.attrs)
        data_normalized.attrs['normalized_by_izero'] = 'True'
        
        # Update the data in the class
        self.data = data_normalized
    
    def extract_and_process_spectrum(self, roi, pre_edge_norm_range=None, pre_edge_sub_range=None, 
                                    post_edge_range=None, do_pre_edge_norm=True, do_pre_edge_sub=True,
                                    do_post_edge_norm=True, plot=False, energy_slice=None):
        """
        Extract a spectrum from an ROI and apply processing steps in this order:
        1. Normalize by pre-edge slope
        2. Subtract pre-edge background
        3. Normalize by post-edge average
        
        Parameters:
        -----------
        roi : tuple
            (x_min, y_min, width, height) defining the ROI
        pre_edge_norm_range : tuple, optional
            (min_energy, max_energy) defining the range for pre-edge slope normalization.
            If None, uses the first 20% of the energy range.
        pre_edge_sub_range : tuple, optional
            (min_energy, max_energy) defining the range for pre-edge background subtraction.
            If None, uses the first 10% of the energy range.
        post_edge_range : tuple, optional
            (min_energy, max_energy) defining the range for post-edge normalization.
            If None, uses the last 20% of the energy range.
        do_pre_edge_norm : bool, optional
            If True, perform pre-edge slope normalization
        do_pre_edge_sub : bool, optional
            If True, perform pre-edge background subtraction
        do_post_edge_norm : bool, optional
            If True, perform post-edge normalization
        plot : bool, optional
            If True, plot the extracted and processed spectra
        energy_slice : float, optional
            If provided, also shows the 2D image at this energy with the ROI
            
        Returns:
        --------
        spectrum_xr : xarray.DataArray
            1D xarray with the average spectrum over the ROI
        spectrum_np : numpy.ndarray
            2D numpy array with columns [Energy, Intensity]
        spectrum_final : numpy.ndarray
            2D numpy array with columns [Energy, Processed Intensity]
        processing_info : dict
            Dictionary containing processing parameters and values for each step performed
        """
        # Extract the ROI spectrum
        spectrum_xr, spectrum_np = self.extract_roi_spectrum(roi, plot=False)
        
        # Initialize processing info dictionary
        processing_info = {}
        
        # Make a copy of the original spectrum for processing
        spectrum_processed = spectrum_np.copy()
        
        # Store intermediate results for plotting
        spectrum_after_step1 = None
        spectrum_after_step2 = None
        
        # Step 1: Apply pre-edge slope normalization if requested
        if do_pre_edge_norm:
            spectrum_processed, pre_edge_slope_params = self.normalize_by_pre_edge_slope(spectrum_processed, pre_edge_norm_range)
            processing_info['pre_edge_slope_params'] = pre_edge_slope_params
            spectrum_after_step1 = spectrum_processed.copy()
        
        # Step 2: Apply pre-edge background subtraction if requested
        if do_pre_edge_sub:
            spectrum_processed, pre_edge_value = self.subtract_pre_edge(spectrum_processed, pre_edge_sub_range)
            processing_info['pre_edge_value'] = pre_edge_value
            spectrum_after_step2 = spectrum_processed.copy()
        
        # Step 3: Apply post-edge normalization if requested
        if do_post_edge_norm:
            spectrum_processed, post_edge_value = self.normalize_by_post_edge(spectrum_processed, post_edge_range)
            processing_info['post_edge_value'] = post_edge_value
        
        # Store the final processed spectrum
        spectrum_final = spectrum_processed
        
        # Plot if requested
        if plot:
            energy = spectrum_np[:, 0]
            intensity_orig = spectrum_np[:, 1]
            
            # Count the number of processing steps applied
            steps_applied = sum([do_pre_edge_norm, do_pre_edge_sub, do_post_edge_norm])
            
            # Create plots based on how many processing steps were applied
            if energy_slice is not None:
                # Always show the image and the original spectrum
                n_plots = steps_applied + 2  # Image + Original + each processing step
                fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
                
                # Plot the 2D image with ROI
                ax_img = axes[0]
                x_min, y_min, width, height = roi
                slice_data = self.data.sel(energy=energy_slice, method='nearest')
                im = slice_data.plot(ax=ax_img, cmap='viridis')
                ax_img.set_title(f'Energy = {slice_data.energy.values:.2f}')
                plt.colorbar(im, ax=ax_img, label='Intensity')
                
                # Add ROI box
                rect = Rectangle((x_min, y_min), width, height, 
                                edgecolor='red', facecolor='none', linewidth=2)
                ax_img.add_patch(rect)
                ax_img.text(x_min + width/2, y_min + height + 2, 'ROI', 
                           color='red', fontweight='bold', ha='center')
                
                # Spectrum plots start at index 1
                plot_offset = 1
            else:
                # No image, just the original spectrum and processing steps
                n_plots = steps_applied + 1  # Original + each processing step
                fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
                if n_plots == 1:
                    axes = [axes]  # Make it indexable
                plot_offset = 0
            
            # Plot original spectrum with all the processing regions highlighted
            ax_orig = axes[plot_offset]
            ax_orig.plot(energy, intensity_orig, 'o-', label='Original')
            
            # Visualize the pre-edge normalization fit range and line
            if do_pre_edge_norm and pre_edge_norm_range is not None:
                norm_mask = (energy >= pre_edge_norm_range[0]) & (energy <= pre_edge_norm_range[1])
                if 'pre_edge_slope_params' in processing_info:
                    slope, intercept = processing_info['pre_edge_slope_params']
                    line_values = slope * energy + intercept
                    ax_orig.plot(energy, line_values, 'g--', 
                                label=f'Pre-edge Slope: y = {slope:.4f}x + {intercept:.4f}')
                    ax_orig.plot(energy[norm_mask], intensity_orig[norm_mask], 'o-', 
                                color='green', label='Pre-edge Norm Region')
            
            # Visualize the pre-edge subtraction region
            if do_pre_edge_sub and pre_edge_sub_range is not None:
                sub_mask = (energy >= pre_edge_sub_range[0]) & (energy <= pre_edge_sub_range[1])
                ax_orig.plot(energy[sub_mask], intensity_orig[sub_mask], 'o-', 
                            color='red', label='Pre-edge Sub Region')
            
            # Visualize the post-edge normalization region
            if do_post_edge_norm and post_edge_range is not None:
                post_mask = (energy >= post_edge_range[0]) & (energy <= post_edge_range[1])
                ax_orig.plot(energy[post_mask], intensity_orig[post_mask], 'o-', 
                            color='purple', label='Post-edge Norm Region')
            
            ax_orig.set_title('Original Spectrum from ROI')
            ax_orig.set_xlabel('Energy')
            ax_orig.set_ylabel('Intensity')
            ax_orig.legend()
            ax_orig.grid(True)
            
            # Plot the intermediate and final results based on which steps were applied
            plot_idx = plot_offset + 1
            
            # After Step 1: Pre-edge slope normalization
            if do_pre_edge_norm:
                ax = axes[plot_idx]
                intensity_step1 = spectrum_after_step1[:, 1]
                ax.plot(energy, intensity_step1, 'o-', color='green')
                
                # Highlight regions for the next steps if they'll be applied
                if do_pre_edge_sub and pre_edge_sub_range is not None:
                    sub_mask = (energy >= pre_edge_sub_range[0]) & (energy <= pre_edge_sub_range[1])
                    ax.plot(energy[sub_mask], intensity_step1[sub_mask], 'o-', 
                           color='red', label='Pre-edge Sub Region')
                    
                    if 'pre_edge_value' in processing_info:
                        pre_edge_val = processing_info['pre_edge_value']
                        # This value is from the next step, so we need to approximate where it would be
                        # in the current plot
                        ax.axhline(y=pre_edge_val, color='r', linestyle='--', 
                                  label=f'Pre-edge Level: {pre_edge_val:.4f}')
                
                ax.set_title('After Pre-edge Slope Normalization')
                ax.set_xlabel('Energy')
                ax.set_ylabel('Normalized Intensity')
                if do_pre_edge_sub or do_post_edge_norm:
                    ax.legend()
                ax.grid(True)
                plot_idx += 1
            
            # After Step 2: Pre-edge background subtraction
            if do_pre_edge_sub:
                ax = axes[plot_idx]
                
                if spectrum_after_step2 is not None:
                    intensity_step2 = spectrum_after_step2[:, 1]
                    ax.plot(energy, intensity_step2, 'o-', color='blue')
                    
                    # Highlight regions for the next step if it'll be applied
                    if do_post_edge_norm and post_edge_range is not None:
                        post_mask = (energy >= post_edge_range[0]) & (energy <= post_edge_range[1])
                        ax.plot(energy[post_mask], intensity_step2[post_mask], 'o-', 
                               color='purple', label='Post-edge Norm Region')
                        
                        if 'post_edge_value' in processing_info:
                            post_edge_val = processing_info['post_edge_value']
                            # This value is from the next step, so we need to approximate where it would be
                            ax.axhline(y=post_edge_val, color='purple', linestyle='--', 
                                      label=f'Post-edge Level: {post_edge_val:.4f}')
                    
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    ax.set_title('After Pre-edge Background Subtraction')
                    ax.set_xlabel('Energy')
                    if do_pre_edge_norm:
                        ax.set_ylabel('Normalized & Subtracted Intensity')
                    else:
                        ax.set_ylabel('Subtracted Intensity')
                    if do_post_edge_norm:
                        ax.legend()
                    ax.grid(True)
                    plot_idx += 1
            
            # After Step 3: Post-edge normalization (Final result)
            if do_post_edge_norm:
                ax = axes[plot_idx]
                intensity_final = spectrum_final[:, 1]
                ax.plot(energy, intensity_final, 'o-', color='purple')
                ax.axhline(y=1, color='purple', linestyle='--', 
                           label=f'Post-edge Level (1.0)')
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                title_parts = []
                if do_pre_edge_norm:
                    title_parts.append("Pre-edge Slope Normalized")
                if do_pre_edge_sub:
                    title_parts.append("Pre-edge Subtracted")
                if do_post_edge_norm:
                    title_parts.append("Post-edge Normalized")
                
                ax.set_title(f'Final: {" & ".join(title_parts)} Spectrum')
                ax.set_xlabel('Energy')
                ax.set_ylabel('Fully Processed Intensity')
                ax.legend()
                ax.grid(True)
            
            # If we're displaying an energy slice, add vertical line to all spectra
            if energy_slice is not None:
                for i in range(plot_offset, len(axes)):
                    ax = axes[i]
                    ax.axvline(x=energy_slice, color='red', linestyle='--', alpha=0.7)
                    ax.text(energy_slice, ax.get_ylim()[1]*0.95, 'Current Slice', 
                           color='red', ha='center', va='top', rotation=90, alpha=0.7)
            
            plt.tight_layout()
            plt.show()
        
        return spectrum_xr, spectrum_np, spectrum_final, processing_info



    def extract_multiple_roi_spectra(self, roi_list, roi_labels=None, plot=False, energy_slice=None):
        """
        Extract the average spectrum from multiple regions of interest and optionally plot them.
        
        Parameters:
        -----------
        roi_list : list of tuples
            List of ROIs, each as (x_min, y_min, width, height)
        roi_labels : list of str, optional
            Labels for each ROI. If None, will use "ROI 1", "ROI 2", etc.
        plot : bool, optional
            If True, plot the extracted spectra
        energy_slice : float, optional
            If provided, also shows the 2D image at this energy with all ROIs
            
        Returns:
        --------
        spectra_dict : dict
            Dictionary with ROI labels as keys and tuples of (spectrum_xr, spectrum_np) as values
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if roi_labels is None:
            roi_labels = [f"ROI {i+1}" for i in range(len(roi_list))]
        
        if len(roi_labels) != len(roi_list):
            raise ValueError("Number of ROI labels must match number of ROIs")
        
        # Extract spectra for each ROI
        spectra_dict = {}
        for i, (roi, label) in enumerate(zip(roi_list, roi_labels)):
            spectrum_xr, spectrum_np = self.extract_roi_spectrum(roi, plot=False)
            spectra_dict[label] = (spectrum_xr, spectrum_np)
        
        # Plot if requested
        if plot:
            if energy_slice is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot the 2D image with all ROIs
                slice_data = self.data.sel(energy=energy_slice, method='nearest')
                im = slice_data.plot(ax=ax1, cmap='viridis')
                ax1.set_title(f'Energy = {slice_data.energy.values:.2f}')
                plt.colorbar(im, ax=ax1, label='Intensity')
                
                # Add all ROI boxes with different colors
                colors = plt.cm.tab10(np.linspace(0, 1, len(roi_list)))
                for i, (roi, label) in enumerate(zip(roi_list, roi_labels)):
                    x_min, y_min, width, height = roi
                    rect = Rectangle((x_min, y_min), width, height, 
                                edgecolor=colors[i], facecolor='none', linewidth=2)
                    ax1.add_patch(rect)
                    ax1.text(x_min + width/2, y_min + height + 2, label, 
                        color=colors[i], fontweight='bold', ha='center')
                
                # Plot all spectra in second subplot
                for i, (label, (spectrum_xr, spectrum_np)) in enumerate(spectra_dict.items()):
                    energy_values = spectrum_np[:, 0]
                    intensity_values = spectrum_np[:, 1]
                    ax2.plot(energy_values, intensity_values, 'o-', color=colors[i], label=label)
                
                ax2.set_title('Average Spectra from Multiple ROIs')
                ax2.set_xlabel('Energy')
                ax2.set_ylabel('Intensity')
                ax2.legend()
                ax2.grid(True)
                
                # Add a vertical line at the selected energy
                ax2.axvline(x=energy_slice, color='red', linestyle='--', alpha=0.7)
                ax2.text(energy_slice, ax2.get_ylim()[1]*0.95, 'Current Slice', 
                    color='red', ha='center', va='top', rotation=90, alpha=0.7)
                
            else:
                # Just plot the spectra
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.tab10(np.linspace(0, 1, len(roi_list)))
                
                for i, (label, (spectrum_xr, spectrum_np)) in enumerate(spectra_dict.items()):
                    energy_values = spectrum_np[:, 0]
                    intensity_values = spectrum_np[:, 1]
                    ax.plot(energy_values, intensity_values, 'o-', color=colors[i], label=label)
                
                ax.set_title('Average Spectra from Multiple ROIs')
                ax.set_xlabel('Energy')
                ax.set_ylabel('Intensity')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return spectra_dict


    def extract_and_process_multiple_spectra(self, roi_list, roi_labels=None, pre_edge_norm_range=None, 
                                       pre_edge_sub_range=None, post_edge_range=None, 
                                       do_pre_edge_norm=True, do_pre_edge_sub=True, do_post_edge_norm=True, 
                                       plot=False, energy_slice=None, save_data=False, save_path=None, save_format='csv'):
        """
        Extract and process spectra from multiple ROIs with the same processing parameters.
        
        Parameters:
        -----------
        roi_list : list of tuples
            List of ROIs, each as (x_min, y_min, width, height)
        roi_labels : list of str, optional
            Labels for each ROI. If None, will use "ROI 1", "ROI 2", etc.
        pre_edge_norm_range : tuple, optional
            (min_energy, max_energy) defining the range for pre-edge slope normalization.
        pre_edge_sub_range : tuple, optional
            (min_energy, max_energy) defining the range for pre-edge background subtraction.
        post_edge_range : tuple, optional
            (min_energy, max_energy) defining the range for post-edge normalization.
        do_pre_edge_norm : bool, optional
            If True, perform pre-edge slope normalization
        do_pre_edge_sub : bool, optional
            If True, perform pre-edge background subtraction
        do_post_edge_norm : bool, optional
            If True, perform post-edge normalization
        plot : bool, optional
            If True, plot the processed spectra
        energy_slice : float, optional
            If provided, also shows the 2D image at this energy with all ROIs
        save_data : bool, optional
            If True, save the processed spectra to files
        save_path : str, optional
            Directory path to save files. If None, saves to current directory
        save_format : str, optional
            Format to save data ('csv', 'txt', or 'npz'). Default is 'csv'
            
        Returns:
        --------
        results_dict : dict
            Dictionary with ROI labels as keys and tuples of 
            (spectrum_xr, spectrum_np, spectrum_final, processing_info) as values
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if roi_labels is None:
            roi_labels = [f"ROI {i+1}" for i in range(len(roi_list))]
        
        if len(roi_labels) != len(roi_list):
            raise ValueError("Number of ROI labels must match number of ROIs")
        
        # Process spectra for each ROI
        results_dict = {}
        for roi, label in zip(roi_list, roi_labels):
            spectrum_xr, spectrum_np, spectrum_final, processing_info = self.extract_and_process_spectrum(
                roi, pre_edge_norm_range=pre_edge_norm_range, pre_edge_sub_range=pre_edge_sub_range,
                post_edge_range=post_edge_range, do_pre_edge_norm=do_pre_edge_norm,
                do_pre_edge_sub=do_pre_edge_sub, do_post_edge_norm=do_post_edge_norm,
                plot=False  # Don't plot individual spectra
            )
            results_dict[label] = (spectrum_xr, spectrum_np, spectrum_final, processing_info)
        
        # Plot if requested
        if plot:
            if energy_slice is not None:
                # Create subplots: image + original spectra + processed spectra
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
                
                # Plot the 2D image with all ROIs
                slice_data = self.data.sel(energy=energy_slice, method='nearest')
                im = slice_data.plot(ax=ax1, cmap='viridis')
                ax1.set_title(f'Energy = {slice_data.energy.values:.2f}')
                plt.colorbar(im, ax=ax1, label='Intensity')
                
                # Add all ROI boxes with different colors
                colors = plt.cm.tab10(np.linspace(0, 1, len(roi_list)))
                for i, (roi, label) in enumerate(zip(roi_list, roi_labels)):
                    x_min, y_min, width, height = roi
                    rect = Rectangle((x_min, y_min), width, height, 
                                edgecolor=colors[i], facecolor='none', linewidth=2)
                    ax1.add_patch(rect)
                    ax1.text(x_min + width/2, y_min + height + 2, label, 
                        color=colors[i], fontweight='bold', ha='center')
                
                # Plot original spectra
                for i, (label, (spectrum_xr, spectrum_np, spectrum_final, processing_info)) in enumerate(results_dict.items()):
                    energy_values = spectrum_np[:, 0]
                    intensity_values = spectrum_np[:, 1]
                    ax2.plot(energy_values, intensity_values, 'o-', color=colors[i], label=label)
                
                ax2.set_title('Original Spectra from Multiple ROIs')
                ax2.set_xlabel('Energy')
                ax2.set_ylabel('Intensity')
                ax2.legend()
                ax2.grid(True)
                ax2.axvline(x=energy_slice, color='red', linestyle='--', alpha=0.7)
                
                # Plot processed spectra
                for i, (label, (spectrum_xr, spectrum_np, spectrum_final, processing_info)) in enumerate(results_dict.items()):
                    energy_values = spectrum_final[:, 0]
                    intensity_values = spectrum_final[:, 1]
                    ax3.plot(energy_values, intensity_values, 'o-', color=colors[i], label=label)
                
                # Create title based on processing steps
                title_parts = []
                if do_pre_edge_norm:
                    title_parts.append("Pre-edge Slope Normalized")
                if do_pre_edge_sub:
                    title_parts.append("Pre-edge Subtracted")
                if do_post_edge_norm:
                    title_parts.append("Post-edge Normalized")
                
                if title_parts:
                    ax3.set_title(f'Processed Spectra: {" & ".join(title_parts)}')
                else:
                    ax3.set_title('Processed Spectra (No Processing)')
                
                ax3.set_xlabel('Energy')
                ax3.set_ylabel('Processed Intensity')
                ax3.legend()
                ax3.grid(True)
                ax3.axvline(x=energy_slice, color='red', linestyle='--', alpha=0.7)
                
                # Add reference lines for processed spectra
                if do_post_edge_norm:
                    ax3.axhline(y=1, color='purple', linestyle='--', alpha=0.5, label='Post-edge Level')
                if do_pre_edge_sub:
                    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
            else:
                # Just plot original and processed spectra side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                colors = plt.cm.tab10(np.linspace(0, 1, len(roi_list)))
                
                # Plot original spectra
                for i, (label, (spectrum_xr, spectrum_np, spectrum_final, processing_info)) in enumerate(results_dict.items()):
                    energy_values = spectrum_np[:, 0]
                    intensity_values = spectrum_np[:, 1]
                    ax1.plot(energy_values, intensity_values, 'o-', color=colors[i], label=label)
                
                ax1.set_title('Original Spectra from Multiple ROIs')
                ax1.set_xlabel('Energy')
                ax1.set_ylabel('Intensity')
                ax1.legend()
                ax1.grid(True)
                
                # Plot processed spectra
                for i, (label, (spectrum_xr, spectrum_np, spectrum_final, processing_info)) in enumerate(results_dict.items()):
                    energy_values = spectrum_final[:, 0]
                    intensity_values = spectrum_final[:, 1]
                    ax2.plot(energy_values, intensity_values, 'o-', color=colors[i], label=label)
                
                # Create title based on processing steps
                title_parts = []
                if do_pre_edge_norm:
                    title_parts.append("Pre-edge Slope Normalized")
                if do_pre_edge_sub:
                    title_parts.append("Pre-edge Subtracted")
                if do_post_edge_norm:
                    title_parts.append("Post-edge Normalized")
                
                if title_parts:
                    ax2.set_title(f'Processed Spectra: {" & ".join(title_parts)}')
                else:
                    ax2.set_title('Processed Spectra (No Processing)')
                
                ax2.set_xlabel('Energy')
                ax2.set_ylabel('Processed Intensity')
                ax2.legend()
                ax2.grid(True)
                
                # Add reference lines for processed spectra
                if do_post_edge_norm:
                    ax2.axhline(y=1, color='purple', linestyle='--', alpha=0.5, label='Post-edge Level')
                if do_pre_edge_sub:
                    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.show()
        
        # Save data if requested
        if save_data:
            import os
            
            if save_path is None:
                save_path = os.getcwd()
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            for label, (spectrum_xr, spectrum_np, spectrum_final, processing_info) in results_dict.items():
                # Clean label for filename (remove spaces and special characters)
                clean_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_label = clean_label.replace(' ', '_')
                
                if save_format.lower() == 'csv':
                    # Save original spectrum
                    orig_filename = f"{clean_label}_original.csv"
                    orig_filepath = os.path.join(save_path, orig_filename)
                    np.savetxt(orig_filepath, spectrum_np, delimiter=',', 
                            header='Energy,Intensity', comments='')
                    
                    # Save processed spectrum
                    proc_filename = f"{clean_label}_processed.csv"
                    proc_filepath = os.path.join(save_path, proc_filename)
                    np.savetxt(proc_filepath, spectrum_final, delimiter=',', 
                            header='Energy,Processed_Intensity', comments='')
                    
                    # Save processing info
                    info_filename = f"{clean_label}_processing_info.txt"
                    info_filepath = os.path.join(save_path, info_filename)
                    with open(info_filepath, 'w') as f:
                        f.write(f"Processing Information for {label}\n")
                        f.write(f"ROI: {roi_list[roi_labels.index(label)]}\n")
                        f.write(f"Pre-edge normalization: {do_pre_edge_norm}\n")
                        f.write(f"Pre-edge subtraction: {do_pre_edge_sub}\n")
                        f.write(f"Post-edge normalization: {do_post_edge_norm}\n")
                        f.write(f"Pre-edge norm range: {pre_edge_norm_range}\n")
                        f.write(f"Pre-edge sub range: {pre_edge_sub_range}\n")
                        f.write(f"Post-edge range: {post_edge_range}\n")
                        for key, value in processing_info.items():
                            f.write(f"{key}: {value}\n")
                
                elif save_format.lower() == 'txt':
                    # Save as space-delimited text files
                    orig_filename = f"{clean_label}_original.txt"
                    orig_filepath = os.path.join(save_path, orig_filename)
                    np.savetxt(orig_filepath, spectrum_np, delimiter=' ', 
                            header='Energy Intensity')
                    
                    proc_filename = f"{clean_label}_processed.txt"
                    proc_filepath = os.path.join(save_path, proc_filename)
                    np.savetxt(proc_filepath, spectrum_final, delimiter=' ', 
                            header='Energy Processed_Intensity')
                
                elif save_format.lower() == 'npz':
                    # Save as compressed numpy arrays
                    data_filename = f"{clean_label}_data.npz"
                    data_filepath = os.path.join(save_path, data_filename)
                    np.savez_compressed(data_filepath,
                                    original_spectrum=spectrum_np,
                                    processed_spectrum=spectrum_final,
                                    processing_info=processing_info,
                                    roi=roi_list[roi_labels.index(label)])
            
            print(f"Data saved to {save_path} in {save_format.upper()} format")
        
        return results_dict
    
    def load_and_compare_spectra(self, file_paths=None, file_directory=None, file_pattern=None, 
                           spectrum_labels=None, spectrum_type='processed', plot=True, 
                           normalize_for_comparison=False, energy_range=None, save_comparison=False, 
                           save_path=None, comparison_name='spectrum_comparison'):
        """
        Load and compare multiple saved spectra from different files.
        
        Parameters:
        -----------
        file_paths : list of str, optional
            List of specific file paths to load. Can include files from different directories.
        file_directory : str, optional
            Directory to search for spectrum files. Used with file_pattern.
        file_pattern : str, optional
            Pattern to match files (e.g., '*_processed.csv', '*_original.txt').
            Used with file_directory.
        spectrum_labels : list of str, optional
            Custom labels for each spectrum. If None, uses filenames.
        spectrum_type : str, optional
            Type of spectrum to load ('processed', 'original', or 'auto'). 
            'auto' tries to detect from filename. Default is 'processed'.
        plot : bool, optional
            If True, plot the comparison of all spectra.
        normalize_for_comparison : bool, optional
            If True, normalize all spectra to their maximum value for comparison.
        energy_range : tuple, optional
            (min_energy, max_energy) to limit the comparison to a specific range.
        save_comparison : bool, optional
            If True, save the comparison plot and combined data.
        save_path : str, optional
            Directory to save comparison results. If None, uses current directory.
        comparison_name : str, optional
            Base name for saved comparison files.
            
        Returns:
        --------
        spectra_data : dict
            Dictionary with labels as keys and spectrum data arrays as values.
        file_info : dict
            Dictionary with information about each loaded file.
        """
        
        # Determine which files to load
        files_to_load = []
        
        if file_paths is not None:
            # Use specific file paths
            files_to_load = file_paths
        elif file_directory is not None and file_pattern is not None:
            # Search directory for files matching pattern
            search_pattern = os.path.join(file_directory, file_pattern)
            files_to_load = glob.glob(search_pattern)
            if not files_to_load:
                raise ValueError(f"No files found matching pattern '{file_pattern}' in directory '{file_directory}'")
        else:
            raise ValueError("Must provide either file_paths or both file_directory and file_pattern")
        
        print(f"Found {len(files_to_load)} files to load")
        
        # Load spectra data
        spectra_data = {}
        file_info = {}
        
        for i, file_path in enumerate(files_to_load):
            try:
                # Determine file format
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.csv':
                    # Load CSV file
                    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip header
                elif file_ext == '.txt':
                    # Load TXT file
                    data = np.loadtxt(file_path, skiprows=1)  # Skip header
                elif file_ext == '.npz':
                    # Load NPZ file
                    npz_data = np.load(file_path)
                    if spectrum_type == 'processed' and 'processed_spectrum' in npz_data:
                        data = npz_data['processed_spectrum']
                    elif spectrum_type == 'original' and 'original_spectrum' in npz_data:
                        data = npz_data['original_spectrum']
                    elif 'spectrum' in npz_data:
                        data = npz_data['spectrum']
                    else:
                        # Use the first 2D array found
                        for key in npz_data.files:
                            if npz_data[key].ndim == 2 and npz_data[key].shape[1] == 2:
                                data = npz_data[key]
                                break
                        else:
                            raise ValueError(f"Could not find suitable spectrum data in {file_path}")
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
                
                # Generate label
                if spectrum_labels is not None and i < len(spectrum_labels):
                    label = spectrum_labels[i]
                else:
                    # Use filename without extension as label
                    label = os.path.splitext(os.path.basename(file_path))[0]
                    # Clean up common suffixes
                    for suffix in ['_processed', '_original', '_data']:
                        if label.endswith(suffix):
                            label = label[:-len(suffix)]
                            break
                
                # Apply energy range filter if specified
                if energy_range is not None:
                    energy_mask = (data[:, 0] >= energy_range[0]) & (data[:, 0] <= energy_range[1])
                    data = data[energy_mask]
                
                # Store data and info
                spectra_data[label] = data
                file_info[label] = {
                    'file_path': file_path,
                    'file_format': file_ext[1:],  # Remove the dot
                    'data_points': len(data),
                    'energy_range': (data[0, 0], data[-1, 0]),
                    'intensity_range': (np.min(data[:, 1]), np.max(data[:, 1]))
                }
                
                print(f"Loaded {label}: {len(data)} points, energy range {data[0,0]:.2f}-{data[-1,0]:.2f}")
                
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if not spectra_data:
            raise ValueError("No spectra could be loaded successfully")
        
        # Normalize for comparison if requested
        if normalize_for_comparison:
            for label in spectra_data:
                data = spectra_data[label].copy()
                max_intensity = np.max(data[:, 1])
                if max_intensity > 0:
                    data[:, 1] = data[:, 1] / max_intensity
                spectra_data[label] = data
        
        # Plot comparison if requested
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Generate colors for each spectrum
            colors = plt.cm.tab10(np.linspace(0, 1, len(spectra_data)))
            
            # Plot each spectrum
            for i, (label, data) in enumerate(spectra_data.items()):
                energy = data[:, 0]
                intensity = data[:, 1]
                ax.plot(energy, intensity, 'o-', color=colors[i], label=label, markersize=3, linewidth=1.5)
            
            ax.set_xlabel('Energy')
            if normalize_for_comparison:
                ax.set_ylabel('Normalized Intensity')
                ax.set_title('Comparison of Normalized Spectra')
            else:
                ax.set_ylabel('Intensity')
                ax.set_title('Comparison of Spectra')
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add energy range indication if used
            if energy_range is not None:
                ax.set_xlim(energy_range)
                ax.text(0.02, 0.98, f'Energy range: {energy_range[0]:.1f} - {energy_range[1]:.1f}', 
                    transform=ax.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_comparison:
                if save_path is None:
                    save_path = os.getcwd()
                os.makedirs(save_path, exist_ok=True)
                
                plot_filename = f"{comparison_name}_plot.png"
                plot_filepath = os.path.join(save_path, plot_filename)
                plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                print(f"Comparison plot saved to {plot_filepath}")
            
            plt.show()
        
        # Save combined data if requested
        if save_comparison:
            if save_path is None:
                save_path = os.getcwd()
            os.makedirs(save_path, exist_ok=True)
            
            # Save as CSV with all spectra
            csv_filename = f"{comparison_name}_data.csv"
            csv_filepath = os.path.join(save_path, csv_filename)
            
            # Find common energy grid or use the first spectrum's energy grid
            reference_energy = list(spectra_data.values())[0][:, 0]
            
            # Create combined data array
            combined_data = [reference_energy]
            header_parts = ['Energy']
            
            for label, data in spectra_data.items():
                # Interpolate to common energy grid if needed
                if not np.array_equal(data[:, 0], reference_energy):
                    from scipy.interpolate import interp1d
                    f_interp = interp1d(data[:, 0], data[:, 1], kind='linear', 
                                    bounds_error=False, fill_value=np.nan)
                    interpolated_intensity = f_interp(reference_energy)
                    combined_data.append(interpolated_intensity)
                else:
                    combined_data.append(data[:, 1])
                header_parts.append(label)
            
            combined_array = np.column_stack(combined_data)
            np.savetxt(csv_filepath, combined_array, delimiter=',', 
                    header=','.join(header_parts), comments='')
            
            # Save file information
            info_filename = f"{comparison_name}_file_info.txt"
            info_filepath = os.path.join(save_path, info_filename)
            with open(info_filepath, 'w') as f:
                f.write("Spectrum Comparison File Information\n")
                f.write("=" * 50 + "\n")
                for label, info in file_info.items():
                    f.write(f"\nSpectrum: {label}\n")
                    f.write(f"  File: {info['file_path']}\n")
                    f.write(f"  Format: {info['file_format']}\n")
                    f.write(f"  Data points: {info['data_points']}\n")
                    f.write(f"  Energy range: {info['energy_range'][0]:.3f} - {info['energy_range'][1]:.3f}\n")
                    f.write(f"  Intensity range: {info['intensity_range'][0]:.3f} - {info['intensity_range'][1]:.3f}\n")
                
                if normalize_for_comparison:
                    f.write(f"\nNote: All spectra were normalized to their maximum value for comparison.\n")
                if energy_range is not None:
                    f.write(f"Energy range filter applied: {energy_range[0]} - {energy_range[1]}\n")
            
            print(f"Combined data saved to {csv_filepath}")
            print(f"File information saved to {info_filepath}")
        
        return spectra_data, file_info


    def load_and_compare_by_sample_and_roi(self, sample_directories, roi_patterns=None, sample_labels=None,
                                        spectrum_type='processed', plot=True, normalize_for_comparison=False,
                                        energy_range=None, save_comparison=False, save_path=None,
                                        comparison_name='sample_roi_comparison'):
        """
        Load and compare spectra organized by sample directories and ROI patterns.
        Useful for comparing the same ROIs across different samples.
        
        Parameters:
        -----------
        sample_directories : list of str
            List of directories, each containing spectra from one sample.
        roi_patterns : list of str, optional
            List of ROI patterns to look for in each sample directory.
            If None, loads all spectrum files found.
        sample_labels : list of str, optional
            Labels for each sample. If None, uses directory names.
        spectrum_type : str, optional
            Type of spectrum to load ('processed', 'original'). Default is 'processed'.
        plot : bool, optional
            If True, create comparison plots.
        normalize_for_comparison : bool, optional
            If True, normalize all spectra for comparison.
        energy_range : tuple, optional
            (min_energy, max_energy) to limit comparison range.
        save_comparison : bool, optional
            If True, save comparison results.
        save_path : str, optional
            Directory to save results.
        comparison_name : str, optional
            Base name for saved files.
            
        Returns:
        --------
        organized_data : dict
            Nested dictionary: {sample_label: {roi_label: spectrum_data}}
        comparison_summary : dict
            Summary information about the comparison.
        """
        
        # Set default ROI patterns
        if roi_patterns is None:
            roi_patterns = [f'*_{spectrum_type}.csv', f'*_{spectrum_type}.txt', f'*_{spectrum_type}.npz']
        
        # Set default sample labels
        if sample_labels is None:
            sample_labels = [os.path.basename(os.path.normpath(directory)) for directory in sample_directories]
        
        if len(sample_labels) != len(sample_directories):
            raise ValueError("Number of sample labels must match number of sample directories")
        
        organized_data = {}
        all_roi_names = set()
        
        # Load data for each sample
        for sample_dir, sample_label in zip(sample_directories, sample_labels):
            print(f"\nLoading data for sample: {sample_label}")
            organized_data[sample_label] = {}
            
            # Find all matching files in this sample directory
            sample_files = []
            for pattern in roi_patterns:
                search_pattern = os.path.join(sample_dir, pattern)
                sample_files.extend(glob.glob(search_pattern))
            
            # Load each file
            for file_path in sample_files:
                try:
                    # Load the spectrum using the main loading function
                    spectra_data, file_info = self.load_and_compare_spectra(
                        file_paths=[file_path], 
                        spectrum_type=spectrum_type,
                        plot=False,
                        normalize_for_comparison=normalize_for_comparison,
                        energy_range=energy_range
                    )
                    
                    # Extract ROI name from filename
                    filename = os.path.splitext(os.path.basename(file_path))[0]
                    # Remove common suffixes to get ROI name
                    roi_name = filename
                    for suffix in ['_processed', '_original', '_data']:
                        if roi_name.endswith(suffix):
                            roi_name = roi_name[:-len(suffix)]
                            break
                    
                    # Store the data
                    spectrum_data = list(spectra_data.values())[0]  # Get the first (and only) spectrum
                    organized_data[sample_label][roi_name] = spectrum_data
                    all_roi_names.add(roi_name)
                    
                    print(f"  Loaded ROI '{roi_name}': {len(spectrum_data)} points")
                    
                except Exception as e:
                    print(f"  Warning: Could not load {file_path}: {e}")
                    continue
        
        # Create comparison plots if requested
        if plot:
            # Plot 1: All spectra grouped by ROI
            if len(all_roi_names) > 1:
                n_rois = len(all_roi_names)
                fig, axes = plt.subplots(1, n_rois, figsize=(5*n_rois, 6))
                if n_rois == 1:
                    axes = [axes]
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(sample_labels)))
                
                for ax_idx, roi_name in enumerate(sorted(all_roi_names)):
                    ax = axes[ax_idx]
                    
                    for sample_idx, (sample_label, sample_data) in enumerate(organized_data.items()):
                        if roi_name in sample_data:
                            data = sample_data[roi_name]
                            energy = data[:, 0]
                            intensity = data[:, 1]
                            ax.plot(energy, intensity, 'o-', color=colors[sample_idx], 
                                label=sample_label, markersize=3, linewidth=1.5)
                    
                    ax.set_title(f'ROI: {roi_name}')
                    ax.set_xlabel('Energy')
                    if normalize_for_comparison:
                        ax.set_ylabel('Normalized Intensity')
                    else:
                        ax.set_ylabel('Intensity')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            
            # Plot 2: All spectra from each sample
            fig, axes = plt.subplots(1, len(sample_labels), figsize=(5*len(sample_labels), 6))
            if len(sample_labels) == 1:
                axes = [axes]
            
            roi_colors = plt.cm.Set3(np.linspace(0, 1, len(all_roi_names)))
            roi_color_map = {roi: color for roi, color in zip(sorted(all_roi_names), roi_colors)}
            
            for sample_idx, (sample_label, sample_data) in enumerate(organized_data.items()):
                ax = axes[sample_idx]
                
                for roi_name, data in sample_data.items():
                    energy = data[:, 0]
                    intensity = data[:, 1]
                    ax.plot(energy, intensity, 'o-', color=roi_color_map[roi_name], 
                        label=roi_name, markersize=3, linewidth=1.5)
                
                ax.set_title(f'Sample: {sample_label}')
                ax.set_xlabel('Energy')
                if normalize_for_comparison:
                    ax.set_ylabel('Normalized Intensity')
                else:
                    ax.set_ylabel('Intensity')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Prepare comparison summary
        comparison_summary = {
            'n_samples': len(sample_labels),
            'n_rois': len(all_roi_names),
            'sample_labels': sample_labels,
            'roi_names': sorted(all_roi_names),
            'normalized': normalize_for_comparison,
            'energy_range': energy_range
        }
        
        # Save results if requested
        if save_comparison:
            if save_path is None:
                save_path = os.getcwd()
            os.makedirs(save_path, exist_ok=True)
            
            # Save summary information
            summary_filename = f"{comparison_name}_summary.txt"
            summary_filepath = os.path.join(save_path, summary_filename)
            with open(summary_filepath, 'w') as f:
                f.write("Sample and ROI Comparison Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Number of samples: {comparison_summary['n_samples']}\n")
                f.write(f"Number of ROIs: {comparison_summary['n_rois']}\n")
                f.write(f"Samples: {', '.join(comparison_summary['sample_labels'])}\n")
                f.write(f"ROIs: {', '.join(comparison_summary['roi_names'])}\n")
                f.write(f"Normalized for comparison: {comparison_summary['normalized']}\n")
                f.write(f"Energy range filter: {comparison_summary['energy_range']}\n")
                
                f.write("\nData availability matrix:\n")
                f.write("Sample\\ROI")
                for roi in sorted(all_roi_names):
                    f.write(f"\t{roi}")
                f.write("\n")
                
                for sample_label, sample_data in organized_data.items():
                    f.write(f"{sample_label}")
                    for roi in sorted(all_roi_names):
                        f.write(f"\t{'Yes' if roi in sample_data else 'No'}")
                    f.write("\n")
            
            print(f"Comparison summary saved to {summary_filepath}")
        
        return organized_data, comparison_summary

