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
    
    def __init__(self, filepath=None, pixel_size_um=50.0):
        """
        Initialize the LariatDataProcessor.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the HDF5 file to load immediately
        pixel_size_um : float, optional
            Size of each pixel in micrometers. Default is 50.0 um
        """
        self.data = None
        self.metadata = None
        self.filepath = filepath
        self.pixel_size_um = pixel_size_um  # Add pixel size parameter
        
        if filepath is not None:
            self.load_data(filepath)
    
    def set_pixel_size(self, pixel_size_um):
        """
        Set or update the pixel size in micrometers.
        
        Parameters:
        -----------
        pixel_size_um : float
            Size of each pixel in micrometers
        """
        self.pixel_size_um = pixel_size_um
        print(f"Pixel size set to {pixel_size_um} μm")

    def pixels_to_um(self, pixel_coords):
        """
        Convert pixel coordinates to micrometers.
        
        Parameters:
        -----------
        pixel_coords : array-like
            Pixel coordinates to convert
            
        Returns:
        --------
        um_coords : array-like
            Coordinates in micrometers
        """
        return np.array(pixel_coords) * self.pixel_size_um

    def um_to_pixels(self, um_coords):
        """
        Convert micrometer coordinates to pixels.
        
        Parameters:
        -----------
        um_coords : array-like
            Coordinates in micrometers
            
        Returns:
        --------
        pixel_coords : array-like
            Pixel coordinates
        """
        return np.array(um_coords) / self.pixel_size_um

    def get_extent_um(self, image_shape=None):
        """
        Get the extent of the image in micrometers for matplotlib plotting.
        
        Parameters:
        -----------
        image_shape : tuple, optional
            (height, width) of the image. If None, uses self.data shape
            
        Returns:
        --------
        extent : tuple
            (left, right, bottom, top) in micrometers for matplotlib extent
        """
        if image_shape is None:
            if self.data is None:
                raise ValueError("No data loaded and no image_shape provided")
            image_shape = self.data.isel(energy=0).shape
        
        height, width = image_shape
        # extent = [left, right, bottom, top]
        extent = [0, width * self.pixel_size_um, 0, height * self.pixel_size_um]
        return extent
    
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
        return xr.DataArray(image_list, dims=['energy','pix_y','pix_x'], coords={'energy':energy_list}), metadata
    
    def crop_image(self, crop_region=None, interactive=False, preview_energy=None, 
               update_data=True, plot_preview=True, save_crop_info=False, 
               save_path=None, crop_name='cropped_data'):
        """
        Crop the datacube to a specified region.
        
        Parameters:
        -----------
        crop_region : tuple, optional
            (x_min, y_min, x_max, y_max) defining the crop region in pixel coordinates.
            If None and interactive=False, will prompt for coordinates.
        interactive : bool, optional
            If True, display an image and allow interactive selection of crop region.
            Default is False.
        preview_energy : float, optional
            Energy value to use for preview/interactive selection. 
            If None, uses middle energy of dataset.
        update_data : bool, optional
            If True, update self.data with cropped version. If False, return cropped data.
            Default is True.
        plot_preview : bool, optional
            If True, show before/after preview of the crop. Default is True.
        save_crop_info : bool, optional
            If True, save information about the crop operation.
        save_path : str, optional
            Directory to save crop information. If None, uses current directory.
        crop_name : str, optional
            Base name for saved crop files.
            
        Returns:
        --------
        cropped_data : xarray.DataArray
            The cropped datacube
        crop_info : dict
            Information about the crop operation
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector
        import os
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Get original data dimensions
        original_shape = self.data.shape
        n_energies, n_y, n_x = original_shape
        
        print(f"Original image size: {n_x} x {n_y} pixels ({n_energies} energies)")
        
        # Select preview energy
        if preview_energy is None:
            mid_idx = len(self.data.energy) // 2
            preview_energy = float(self.data.energy.isel(energy=mid_idx).values)
        
        preview_slice = self.data.sel(energy=preview_energy, method='nearest')
        preview_image = preview_slice.values
        actual_preview_energy = float(preview_slice.energy.values)
        
        # Interactive crop selection
        if interactive:
            print(f"Interactive crop selection at energy {actual_preview_energy:.2f} eV")
            print("Instructions:")
            print("1. Click and drag to select crop region")
            print("2. Close the plot window when satisfied with selection")
            print("3. The crop region will be applied automatically")
            
            # Enable interactive backend if needed
            import matplotlib
            backend = matplotlib.get_backend()
            if 'inline' in backend.lower():
                print("Note: Interactive selection works best with non-inline backends")
                print("Try running: %matplotlib qt or %matplotlib tk in Jupyter")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(preview_image, cmap='viridis', origin='lower')
            ax.set_title(f'Click and Drag to Select Crop Region\nEnergy: {actual_preview_energy:.2f} eV')
            ax.set_xlabel('Pixel X')
            ax.set_ylabel('Pixel Y')
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Variables to store selection
            selection_coords = {'x_min': 0, 'y_min': 0, 'x_max': n_x, 'y_max': n_y}
            selection_made = {'selected': False}
            
            def onselect(eclick, erelease):
                """Callback function for rectangle selector"""
                if eclick.xdata is None or erelease.xdata is None:
                    return
                if eclick.ydata is None or erelease.ydata is None:
                    return
                    
                x_min = int(round(min(eclick.xdata, erelease.xdata)))
                x_max = int(round(max(eclick.xdata, erelease.xdata)))
                y_min = int(round(min(eclick.ydata, erelease.ydata)))
                y_max = int(round(max(eclick.ydata, erelease.ydata)))
                
                # Ensure coordinates are within bounds
                x_min = max(0, x_min)
                x_max = min(n_x, x_max)
                y_min = max(0, y_min)
                y_max = min(n_y, y_max)
                
                # Ensure minimum size
                if x_max - x_min < 5:
                    x_max = min(n_x, x_min + 5)
                if y_max - y_min < 5:
                    y_max = min(n_y, y_min + 5)
                
                selection_coords.update({
                    'x_min': x_min, 'y_min': y_min, 
                    'x_max': x_max, 'y_max': y_max
                })
                selection_made['selected'] = True
                
                crop_width = x_max - x_min
                crop_height = y_max - y_min
                print(f"Selected region: ({x_min}, {y_min}, {x_max}, {y_max})")
                print(f"Crop size: {crop_width} x {crop_height} pixels")
            
            # Create rectangle selector with simpler configuration
            try:
                selector = RectangleSelector(ax, onselect, 
                                        button=[1],  # Only left mouse button
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True,
                                        useblit=False)  # Disable blitting for better compatibility
                
                # Add instruction text to the plot
                ax.text(0.02, 0.98, 'Click and drag to select region\nClose window when done', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    fontsize=10)
                
                plt.show()
                
                # Check if a selection was made
                if not selection_made['selected']:
                    print("No selection made. Using fallback method...")
                    # Fallback to manual input
                    print(f"Current image size: {n_x} x {n_y} pixels")
                    try:
                        x_min = int(input(f"Enter x_min (0 to {n_x-1}): ") or "0")
                        y_min = int(input(f"Enter y_min (0 to {n_y-1}): ") or "0")
                        x_max = int(input(f"Enter x_max ({x_min+1} to {n_x}): ") or str(n_x))
                        y_max = int(input(f"Enter y_max ({y_min+1} to {n_y}): ") or str(n_y))
                        selection_coords.update({
                            'x_min': x_min, 'y_min': y_min, 
                            'x_max': x_max, 'y_max': y_max
                        })
                    except (ValueError, KeyboardInterrupt):
                        print("Using full image (no crop)")
                        selection_coords = {'x_min': 0, 'y_min': 0, 'x_max': n_x, 'y_max': n_y}
                
            except Exception as e:
                print(f"Interactive selection failed: {e}")
                print("Falling back to manual input...")
                print(f"Current image size: {n_x} x {n_y} pixels")
                try:
                    x_min = int(input(f"Enter x_min (0 to {n_x-1}): ") or "0")
                    y_min = int(input(f"Enter y_min (0 to {n_y-1}): ") or "0")
                    x_max = int(input(f"Enter x_max ({x_min+1} to {n_x}): ") or str(n_x))
                    y_max = int(input(f"Enter y_max ({y_min+1} to {n_y}): ") or str(n_y))
                    selection_coords.update({
                        'x_min': x_min, 'y_min': y_min, 
                        'x_max': x_max, 'y_max': y_max
                    })
                except (ValueError, KeyboardInterrupt):
                    print("Using full image (no crop)")
                    selection_coords = {'x_min': 0, 'y_min': 0, 'x_max': n_x, 'y_max': n_y}
            
            # Extract crop region from interactive selection
            crop_region = (selection_coords['x_min'], selection_coords['y_min'],
                        selection_coords['x_max'], selection_coords['y_max'])
            
            print(f"Final crop region: {crop_region}")
        
        # If no crop region specified and not interactive, prompt user
        elif crop_region is None:
            print(f"Current image size: {n_x} x {n_y} pixels")
            print("Please specify crop region as (x_min, y_min, x_max, y_max)")
            
            try:
                x_min = int(input(f"Enter x_min (0 to {n_x-1}): "))
                y_min = int(input(f"Enter y_min (0 to {n_y-1}): "))
                x_max = int(input(f"Enter x_max ({x_min+1} to {n_x}): "))
                y_max = int(input(f"Enter y_max ({y_min+1} to {n_y}): "))
                crop_region = (x_min, y_min, x_max, y_max)
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or operation cancelled.")
                return None, None
        
        # Validate crop region
        x_min, y_min, x_max, y_max = crop_region
        
        if not (0 <= x_min < x_max <= n_x and 0 <= y_min < y_max <= n_y):
            raise ValueError(f"Invalid crop region {crop_region}. Must be within image bounds "
                            f"(0, 0, {n_x}, {n_y}) and x_min < x_max, y_min < y_max")
        
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        
        print(f"Cropping to region: ({x_min}, {y_min}, {x_max}, {y_max})")
        print(f"New size: {crop_width} x {crop_height} pixels")
        print(f"Reduction: {original_shape[2] * original_shape[1]:,} -> {crop_width * crop_height:,} pixels "
            f"({(1 - (crop_width * crop_height) / (original_shape[2] * original_shape[1])) * 100:.1f}% smaller)")
        
        # Perform the crop
        cropped_data = self.data.isel(pix_x=slice(x_min, x_max), pix_y=slice(y_min, y_max))
        
        # Update coordinates to reflect the crop (optional - keeps original pixel coordinates)
        # If you want to reset coordinates to start from 0:
        # new_x_coords = np.arange(crop_width)
        # new_y_coords = np.arange(crop_height)
        # cropped_data = cropped_data.assign_coords(pix_x=new_x_coords, pix_y=new_y_coords)
        
        # Store crop information
        crop_info = {
            'original_shape': original_shape,
            'crop_region': crop_region,
            'cropped_shape': cropped_data.shape,
            'crop_size': (crop_width, crop_height),
            'pixels_removed': original_shape[1] * original_shape[2] - crop_width * crop_height,
            'size_reduction_percent': (1 - (crop_width * crop_height) / (original_shape[2] * original_shape[1])) * 100,
            'preview_energy': actual_preview_energy
        }
        
        # Update attributes
        cropped_data.attrs = self.data.attrs.copy()
        cropped_data.attrs['cropped'] = True
        cropped_data.attrs['crop_region'] = str(crop_region)
        cropped_data.attrs['original_shape'] = str(original_shape)
        
        # Show preview if requested
        if plot_preview:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original image with crop region highlighted
            im1 = ax1.imshow(preview_image, cmap='viridis', origin='lower')
            ax1.add_patch(plt.Rectangle((x_min, y_min), crop_width, crop_height, 
                                    fill=False, edgecolor='red', linewidth=2))
            ax1.set_title(f'Original Image ({n_x} x {n_y})\nEnergy: {actual_preview_energy:.2f} eV')
            ax1.set_xlabel('Pixel X')
            ax1.set_ylabel('Pixel Y')
            plt.colorbar(im1, ax=ax1, label='Intensity')
            
            # Cropped image
            cropped_preview = cropped_data.sel(energy=actual_preview_energy, method='nearest')
            im2 = ax2.imshow(cropped_preview.values, cmap='viridis', origin='lower')
            ax2.set_title(f'Cropped Image ({crop_width} x {crop_height})\nEnergy: {actual_preview_energy:.2f} eV')
            ax2.set_xlabel('Pixel X (cropped coordinates)')
            ax2.set_ylabel('Pixel Y (cropped coordinates)')
            plt.colorbar(im2, ax=ax2, label='Intensity')
            
            # Add crop information as text
            info_text = f"""Crop Information:
    Original: {original_shape[2]} x {original_shape[1]} pixels
    Cropped: {crop_width} x {crop_height} pixels
    Region: ({x_min}, {y_min}, {x_max}, {y_max})
    Size reduction: {crop_info['size_reduction_percent']:.1f}%
    Pixels removed: {crop_info['pixels_removed']:,}"""
            
            fig.text(0.02, 0.02, info_text, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    verticalalignment='bottom')
            
            plt.tight_layout()
            plt.show()
        
        # Save crop information if requested
        if save_crop_info:
            if save_path is None:
                save_path = os.getcwd()
            os.makedirs(save_path, exist_ok=True)
            
            # Save cropped data
            crop_filename = f"{crop_name}.nc"
            crop_filepath = os.path.join(save_path, crop_filename)
            cropped_data.to_netcdf(crop_filepath)
            
            # Save crop information
            info_filename = f"{crop_name}_info.txt"
            info_filepath = os.path.join(save_path, info_filename)
            with open(info_filepath, 'w') as f:
                f.write("Image Crop Information\n")
                f.write("=" * 30 + "\n")
                f.write(f"Original shape: {original_shape[2]} x {original_shape[1]} x {original_shape[0]} (x, y, energy)\n")
                f.write(f"Crop region: ({x_min}, {y_min}, {x_max}, {y_max})\n")
                f.write(f"Cropped shape: {crop_width} x {crop_height} x {original_shape[0]} (x, y, energy)\n")
                f.write(f"Pixels removed: {crop_info['pixels_removed']:,}\n")
                f.write(f"Size reduction: {crop_info['size_reduction_percent']:.1f}%\n")
                f.write(f"Preview energy: {actual_preview_energy:.3f} eV\n")
                
                if self.metadata:
                    f.write(f"\nOriginal metadata preserved in cropped data\n")
            
            print(f"Cropped data saved to {crop_filepath}")
            print(f"Crop info saved to {info_filepath}")
        
        # Create new processor object with cropped data
        cropped_processor = LariatDataProcessor()
        cropped_processor.data = cropped_data
        cropped_processor.metadata = self.metadata.copy() if self.metadata else None
        cropped_processor.filepath = None  # No longer corresponds to original file
        
        # Update metadata if it exists
        if cropped_processor.metadata:
            cropped_processor.metadata['cropped'] = True
            cropped_processor.metadata['crop_region'] = crop_region
            cropped_processor.metadata['original_shape'] = original_shape
        
        # Update self.data if requested
        if update_data:
            print("Updating self.data with cropped version")
            self.data = cropped_data
            if self.metadata:
                self.metadata['cropped'] = True
                self.metadata['crop_region'] = crop_region
                self.metadata['original_shape'] = original_shape
            print("Note: Original data has been replaced. Reload from file if you need the full image.")
        
        return cropped_processor, crop_info
    
    
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
    
    def plot_energy_slice(self, energy_value, use_um=True):
        """
        Plot a 2D image slice of the data at a specific energy value.
        
        Parameters:
        -----------
        energy_value : float
            The energy value to select for plotting
        use_um : bool, optional
            If True, display coordinates in micrometers. If False, use pixels.
        
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
        
        if use_um:
            extent = self.get_extent_um(slice_data.shape)
            im = ax.imshow(slice_data, cmap='viridis', origin='lower', extent=extent)
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
        else:
            im = ax.imshow(slice_data, cmap='viridis', origin='lower')
            ax.set_xlabel('Pixel X')
            ax.set_ylabel('Pixel Y')
        
        ax.set_title(f'Energy = {slice_data.energy.values:.2f} eV')
        plt.colorbar(im, ax=ax, label='Intensity')
        
        return fig, ax
    
    def plot_with_roi(self, energy_value, roi=None, use_um=True, roi_in_um=False):
        """
        Plot a 2D image slice with an optional ROI box.
        
        Parameters:
        -----------
        energy_value : float
            The energy value to select for plotting
        roi : tuple, optional
            ROI definition. Format depends on roi_in_um parameter.
        use_um : bool, optional
            If True, display coordinates in micrometers
        roi_in_um : bool, optional
            If True, roi is specified in micrometers. If False, in pixels.
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        fig, ax = self.plot_energy_slice(energy_value, use_um=use_um)
        
        # Add ROI box if provided
        if roi is not None:
            x_min, y_min, width, height = roi
            
            # Convert ROI to display coordinates if needed
            if roi_in_um and not use_um:
                # ROI in um, display in pixels
                x_min = self.um_to_pixels(x_min)
                y_min = self.um_to_pixels(y_min)
                width = self.um_to_pixels(width)
                height = self.um_to_pixels(height)
            elif not roi_in_um and use_um:
                # ROI in pixels, display in um
                x_min = self.pixels_to_um(x_min)
                y_min = self.pixels_to_um(y_min)
                width = self.pixels_to_um(width)
                height = self.pixels_to_um(height)
            
            rect = Rectangle((x_min, y_min), width, height, 
                            edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            
            # Annotate the ROI
            ax.text(x_min + width/2, y_min + height + (5 if use_um else 2), 'ROI', 
                    color='red', fontweight='bold', ha='center')
        
        return fig, ax
    
    def extract_roi_spectrum(self, roi, plot=False, energy_slice=None, use_um=True, roi_in_um=False):
        """
        Extract the average spectrum from a region of interest and optionally plot it.
        
        Parameters:
        -----------
        roi : tuple
            ROI definition as (x_min, y_min, width, height).
            Units depend on roi_in_um parameter.
        plot : bool, optional
            If True, plot the extracted spectrum
        energy_slice : float, optional
            If provided, also shows the 2D image at this energy with the ROI
        use_um : bool, optional
            If True, display image coordinates in micrometers
        roi_in_um : bool, optional
            If True, roi is specified in micrometers. If False, in pixels.
            
        Returns:
        --------
        spectrum_xr : xarray.DataArray
            1D xarray with the average spectrum over the ROI
        spectrum_np : numpy.ndarray
            2D numpy array with columns [Energy, Intensity]
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Convert ROI to pixels if specified in micrometers
        if roi_in_um:
            x_min_um, y_min_um, width_um, height_um = roi
            x_min = int(self.um_to_pixels(x_min_um))
            y_min = int(self.um_to_pixels(y_min_um))
            width = int(self.um_to_pixels(width_um))
            height = int(self.um_to_pixels(height_um))
            print(f"ROI in μm: ({x_min_um}, {y_min_um}, {width_um}, {height_um})")
            print(f"ROI in pixels: ({x_min}, {y_min}, {width}, {height})")
            roi_pixels = (x_min, y_min, width, height)
        else:
            roi_pixels = roi
            x_min, y_min, width, height = roi_pixels
        
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
                
                if use_um:
                    extent = self.get_extent_um(slice_data.shape)
                    im = ax1.imshow(slice_data, cmap='viridis', origin='lower', extent=extent)
                    ax1.set_xlabel('X (μm)')
                    ax1.set_ylabel('Y (μm)')
                    
                    # Convert ROI to display coordinates
                    if roi_in_um:
                        roi_display = roi  # Already in micrometers
                    else:
                        roi_display = (self.pixels_to_um(x_min), self.pixels_to_um(y_min),
                                    self.pixels_to_um(width), self.pixels_to_um(height))
                else:
                    im = ax1.imshow(slice_data, cmap='viridis', origin='lower')
                    ax1.set_xlabel('Pixel X')
                    ax1.set_ylabel('Pixel Y')
                    roi_display = roi_pixels  # Use pixel coordinates
                
                ax1.set_title(f'Energy = {slice_data.energy.values:.2f} eV')
                plt.colorbar(im, ax=ax1, label='Intensity')
                
                # Add ROI box
                x_roi, y_roi, w_roi, h_roi = roi_display
                rect = Rectangle((x_roi, y_roi), w_roi, h_roi, 
                                edgecolor='red', facecolor='none', linewidth=2)
                ax1.add_patch(rect)
                ax1.text(x_roi + w_roi/2, y_roi + h_roi + (5 if use_um else 2), 'ROI', 
                        color='red', fontweight='bold', ha='center')
                
                # Plot spectrum in second subplot
                ax2.plot(energy_values, intensity_values, 'o-')
                ax2.set_title('Average Spectrum from ROI')
                ax2.set_xlabel('Energy (eV)')
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
                ax.set_xlabel('Energy (eV)')
                ax.set_ylabel('Intensity')
                ax.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return spectrum_xr, spectrum_np
    
    def define_roi_um(self, x_um, y_um, width_um, height_um):
        """
        Helper function to define an ROI in micrometers.
        
        Parameters:
        -----------
        x_um, y_um : float
            Bottom-left corner coordinates in micrometers
        width_um, height_um : float
            ROI dimensions in micrometers
            
        Returns:
        --------
        roi_um : tuple
            ROI definition in micrometers (x_min, y_min, width, height)
        roi_pixels : tuple
            ROI definition in pixels (x_min, y_min, width, height)
        """
        roi_um = (x_um, y_um, width_um, height_um)
        
        # Convert to pixels for internal use
        x_pix = int(self.um_to_pixels(x_um))
        y_pix = int(self.um_to_pixels(y_um))
        w_pix = int(self.um_to_pixels(width_um))
        h_pix = int(self.um_to_pixels(height_um))
        roi_pixels = (x_pix, y_pix, w_pix, h_pix)
        
        print(f"ROI defined:")
        print(f"  In micrometers: ({x_um}, {y_um}, {width_um}, {height_um})")
        print(f"  In pixels: ({x_pix}, {y_pix}, {w_pix}, {h_pix})")
        
        return roi_um, roi_pixels
    
    
    def get_image_dimensions_um(self):
        """
        Get the total image dimensions in micrometers.
        
        Returns:
        --------
        dimensions : dict
            Dictionary containing image dimensions in both pixels and micrometers
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        shape = self.data.isel(energy=0).shape
        height_pixels, width_pixels = shape
        
        dimensions = {
            'width_pixels': width_pixels,
            'height_pixels': height_pixels,
            'width_um': width_pixels * self.pixel_size_um,
            'height_um': height_pixels * self.pixel_size_um,
            'pixel_size_um': self.pixel_size_um
        }
        
        print(f"Image dimensions:")
        print(f"  Pixels: {width_pixels} × {height_pixels}")
        print(f"  Physical size: {dimensions['width_um']:.1f} × {dimensions['height_um']:.1f} μm")
        print(f"  Pixel size: {self.pixel_size_um} μm")
        
        return dimensions
    
    
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

    def create_spectral_animation(self, spectrum_np=None, roi=None, output_filename='spectrum_animation.gif', 
                                dpi=100, total_time=10.0, energy_range=None, contrast_percentiles=(5, 95)):
        """
        Create an animation showing the xarray data at each energy alongside the spectrum.
        A vertical line moves through the spectrum to indicate the current energy.
        
        Parameters:
        -----------
        spectrum_np : numpy.ndarray, optional
            2D numpy array with columns [Energy, Intensity] for the processed spectrum.
            If None, will extract spectrum from the provided ROI or use the full average.
        roi : tuple, optional
            (x_min, y_min, width, height) defining the ROI to highlight in the image.
            If spectrum_np is None, this ROI will be used to extract the spectrum.
        output_filename : str, optional
            Filename for the output animation (must end with .mp4 or .gif)
        dpi : int, optional
            Resolution of the animation in dots per inch
        total_time : float, optional
            Total duration of the animation in seconds. Default is 10.0 seconds.
        energy_range : tuple, optional
            (min_energy, max_energy) to limit the animation to a specific energy range
        contrast_percentiles : tuple, optional
            (low, high) percentiles for dynamic contrast adjustment. Default is (5, 95)
            
        Returns:
        --------
        animation : matplotlib.animation.Animation
            The animation object (also saves the animation to the specified file if possible)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Rectangle
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Extract spectrum if not provided
        if spectrum_np is None:
            if roi is not None:
                print("Extracting spectrum from provided ROI...")
                spectrum_xr, spectrum_np = self.extract_roi_spectrum(roi, plot=False)
            else:
                print("No spectrum or ROI provided. Using average spectrum from entire image...")
                # Create a spectrum from the entire image
                full_spectrum = self.data.mean(dim=['pix_x', 'pix_y'])
                energy_values = full_spectrum.energy.values
                intensity_values = full_spectrum.values
                spectrum_np = np.column_stack((energy_values, intensity_values))
        
        # Filter data by energy range if specified
        data_to_use = self.data
        if energy_range is not None:
            energy_mask = (data_to_use.energy >= energy_range[0]) & (data_to_use.energy <= energy_range[1])
            data_to_use = data_to_use.isel(energy=energy_mask)
            
            # Also filter the spectrum
            spec_energy_mask = (spectrum_np[:, 0] >= energy_range[0]) & (spectrum_np[:, 0] <= energy_range[1])
            spectrum_np = spectrum_np[spec_energy_mask]
            
            print(f"Animation limited to energy range: {energy_range[0]:.2f} - {energy_range[1]:.2f} eV")
        
        # Extract energy values from filtered data
        energies = data_to_use.energy.values
        n_frames = len(energies)
        
        if n_frames == 0:
            raise ValueError("No energy points found in the specified range.")
        
        # Calculate fps from total time and number of frames
        fps = n_frames / total_time
        interval_ms = 1000 / fps  # Convert to milliseconds for matplotlib
        
        print(f"Creating animation with {n_frames} frames over {total_time:.1f} seconds")
        print(f"Calculated fps: {fps:.2f}, interval: {interval_ms:.1f} ms per frame")
        print(f"Energy range: {energies[0]:.2f} to {energies[-1]:.2f} eV")
        
        # Set up the figure with two subplots
        fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(15, 5),
                                            gridspec_kw={'width_ratios': [1, 2]})
        
        # Initialize the image plot
        img_data = data_to_use.isel(energy=0)
        img = ax_img.imshow(img_data, cmap='viridis', origin='lower')
        img_title = ax_img.set_title(f'Energy: {energies[0]:.2f} eV')
        plt.colorbar(img, ax=ax_img, label='Intensity')
        ax_img.set_xlabel('Pixel X')
        ax_img.set_ylabel('Pixel Y')
        
        # Add ROI rectangle if provided
        roi_rect = None
        if roi is not None:
            x_min, y_min, width, height = roi
            roi_rect = Rectangle((x_min, y_min), width, height, 
                                edgecolor='red', facecolor='none', linewidth=2)
            ax_img.add_patch(roi_rect)
            ax_img.text(x_min + width/2, y_min + height + 2, 'ROI', 
                    color='red', fontweight='bold', ha='center')
        
        # Plot the spectrum
        ax_spec.plot(spectrum_np[:, 0], spectrum_np[:, 1], 'o-', markersize=3, linewidth=1)
        ax_spec.set_xlabel('Energy (eV)')
        ax_spec.set_ylabel('Intensity')
        ax_spec.set_title('Spectrum')
        ax_spec.grid(True, alpha=0.3)
        
        # Set spectrum plot limits
        ax_spec.set_xlim(spectrum_np[0, 0] - 1, spectrum_np[-1, 0] + 1)
        ax_spec.set_ylim(np.min(spectrum_np[:, 1]) * 0.95, np.max(spectrum_np[:, 1]) * 1.05)
        
        # Add a vertical line to indicate the current energy
        energy_line = ax_spec.axvline(x=energies[0], color='red', linestyle='-', linewidth=2, alpha=0.8)
        
        # Add energy text and animation info on the spectrum plot
        energy_text = ax_spec.text(0.02, 0.98, f'Current: {energies[0]:.2f} eV', 
                                transform=ax_spec.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        time_text = ax_spec.text(0.02, 0.90, f'Time: {total_time:.1f}s | FPS: {fps:.1f}', 
                                transform=ax_spec.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Function to update the animation for each frame
        def update_frame(frame):
            # Update the image data
            img_data = data_to_use.isel(energy=frame)
            img.set_array(img_data)
            
            # Adjust contrast dynamically using percentiles
            vmin, vmax = np.nanpercentile(img_data, contrast_percentiles)
            img.set_clim(vmin, vmax)
            
            # Update the image title
            current_energy = energies[frame]
            img_title.set_text(f'Energy: {current_energy:.2f} eV')
            
            # Update the vertical line position
            energy_line.set_xdata(current_energy)
            
            # Update energy text
            energy_text.set_text(f'Current: {current_energy:.2f} eV')
            
            return img, img_title, energy_line, energy_text
        
        # Create the animation
        anim = animation.FuncAnimation(fig, update_frame, frames=n_frames, 
                                    interval=interval_ms, blit=True, repeat=True)
        
        # Try to save the animation with fallbacks
        try:
            # If output file is .mp4, try to save using FFmpeg
            if output_filename.endswith('.mp4'):
                try:
                    from matplotlib.animation import FFMpegWriter
                    writer = FFMpegWriter(fps=fps, bitrate=5000, codec='h264')
                    anim.save(output_filename, writer=writer, dpi=dpi)
                    print(f"Animation saved to {output_filename}")
                except (FileNotFoundError, RuntimeError):
                    print("FFmpeg not found or failed. Falling back to GIF format.")
                    output_filename = output_filename.replace('.mp4', '.gif')
                    from matplotlib.animation import PillowWriter
                    writer = PillowWriter(fps=fps)
                    anim.save(output_filename, writer=writer, dpi=dpi)
                    print(f"Animation saved to {output_filename}")
            
            # If output file is .gif, use PillowWriter
            elif output_filename.endswith('.gif'):
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
                anim.save(output_filename, writer=writer, dpi=dpi)
                print(f"Animation saved to {output_filename}")
            
            else:
                print("Warning: Output filename must end with .mp4 or .gif. Saving as GIF.")
                output_filename = output_filename.split('.')[0] + '.gif'
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
                anim.save(output_filename, writer=writer, dpi=dpi)
                print(f"Animation saved to {output_filename}")
        
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Returning animation object instead. Display it using: plt.show()")
            print("In Jupyter notebooks, you can also use: HTML(anim.to_html5_video())")
        
        plt.tight_layout()
        return anim


    def create_roi_comparison_animation(self, roi_list, roi_labels=None, output_filename='roi_comparison_animation.gif',
                                    dpi=100, total_time=12.0, energy_range=None, normalize_spectra=False,
                                    contrast_percentiles=(5, 95)):
        """
        Create an animation comparing multiple ROIs, showing the image with all ROIs marked
        and their corresponding spectra with moving energy indicators.
        
        Parameters:
        -----------
        roi_list : list of tuples
            List of ROIs, each as (x_min, y_min, width, height)
        roi_labels : list of str, optional
            Labels for each ROI. If None, will use "ROI 1", "ROI 2", etc.
        output_filename : str, optional
            Filename for the output animation (must end with .mp4 or .gif)
        dpi : int, optional
            Resolution of the animation in dots per inch
        total_time : float, optional
            Total duration of the animation in seconds. Default is 12.0 seconds.
        energy_range : tuple, optional
            (min_energy, max_energy) to limit the animation to a specific energy range
        normalize_spectra : bool, optional
            If True, normalize all spectra to their maximum value for comparison
        contrast_percentiles : tuple, optional
            (low, high) percentiles for dynamic contrast adjustment
            
        Returns:
        --------
        animation : matplotlib.animation.Animation
            The animation object
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Rectangle
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if roi_labels is None:
            roi_labels = [f"ROI {i+1}" for i in range(len(roi_list))]
        
        # Extract spectra for all ROIs
        print("Extracting spectra from ROIs...")
        spectra_dict = {}
        for roi, label in zip(roi_list, roi_labels):
            spectrum_xr, spectrum_np = self.extract_roi_spectrum(roi, plot=False)
            if normalize_spectra:
                max_intensity = np.max(spectrum_np[:, 1])
                if max_intensity > 0:
                    spectrum_np[:, 1] = spectrum_np[:, 1] / max_intensity
            spectra_dict[label] = spectrum_np
        
        # Filter data by energy range if specified
        data_to_use = self.data
        if energy_range is not None:
            energy_mask = (data_to_use.energy >= energy_range[0]) & (data_to_use.energy <= energy_range[1])
            data_to_use = data_to_use.isel(energy=energy_mask)
            print(f"Animation limited to energy range: {energy_range[0]:.2f} - {energy_range[1]:.2f} eV")
        
        energies = data_to_use.energy.values
        n_frames = len(energies)
        
        # Calculate fps from total time and number of frames
        fps = n_frames / total_time
        interval_ms = 1000 / fps  # Convert to milliseconds for matplotlib
        
        print(f"Creating comparison animation with {n_frames} frames for {len(roi_list)} ROIs")
        print(f"Total time: {total_time:.1f} seconds, calculated fps: {fps:.2f}")
        
        # Set up the figure
        fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(16, 6),
                                            gridspec_kw={'width_ratios': [1, 2]})
        
        # Initialize the image plot
        img_data = data_to_use.isel(energy=0)
        img = ax_img.imshow(img_data, cmap='viridis', origin='lower')
        img_title = ax_img.set_title(f'Energy: {energies[0]:.2f} eV')
        plt.colorbar(img, ax=ax_img, label='Intensity')
        ax_img.set_xlabel('Pixel X')
        ax_img.set_ylabel('Pixel Y')
        
        # Add all ROI rectangles with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(roi_list)))
        roi_patches = []
        for i, (roi, label) in enumerate(zip(roi_list, roi_labels)):
            x_min, y_min, width, height = roi
            rect = Rectangle((x_min, y_min), width, height, 
                            edgecolor=colors[i], facecolor='none', linewidth=2)
            ax_img.add_patch(rect)
            ax_img.text(x_min + width/2, y_min + height + 2, label, 
                    color=colors[i], fontweight='bold', ha='center')
            roi_patches.append(rect)
        
        # Plot all spectra
        energy_lines = []
        for i, (label, spectrum_data) in enumerate(spectra_dict.items()):
            ax_spec.plot(spectrum_data[:, 0], spectrum_data[:, 1], 'o-', 
                        color=colors[i], label=label, markersize=2, linewidth=1)
            # Add vertical line for each spectrum
            line = ax_spec.axvline(x=energies[0], color=colors[i], linestyle='-', 
                                linewidth=2, alpha=0.8)
            energy_lines.append(line)
        
        ax_spec.set_xlabel('Energy (eV)')
        if normalize_spectra:
            ax_spec.set_ylabel('Normalized Intensity')
            ax_spec.set_title('Normalized Spectra Comparison')
        else:
            ax_spec.set_ylabel('Intensity')
            ax_spec.set_title('Spectra Comparison')
        ax_spec.legend(loc='upper right')
        ax_spec.grid(True, alpha=0.3)
        
        # Add energy text and animation info
        energy_text = ax_spec.text(0.02, 0.98, f'Current: {energies[0]:.2f} eV', 
                                transform=ax_spec.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        time_text = ax_spec.text(0.02, 0.90, f'Time: {total_time:.1f}s | FPS: {fps:.1f}', 
                                transform=ax_spec.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Function to update the animation for each frame
        def update_frame(frame):
            # Update the image data
            img_data = data_to_use.isel(energy=frame)
            img.set_array(img_data)
            
            # Adjust contrast dynamically
            vmin, vmax = np.nanpercentile(img_data, contrast_percentiles)
            img.set_clim(vmin, vmax)
            
            # Update the image title
            current_energy = energies[frame]
            img_title.set_text(f'Energy: {current_energy:.2f} eV')
            
            # Update all vertical lines
            for line in energy_lines:
                line.set_xdata(current_energy)
            
            # Update energy text
            energy_text.set_text(f'Current: {current_energy:.2f} eV')
            
            return [img, img_title, energy_text] + energy_lines
        
        # Create the animation
        anim = animation.FuncAnimation(fig, update_frame, frames=n_frames, 
                                    interval=interval_ms, blit=True, repeat=True)
        
        # Save the animation
        try:
            if output_filename.endswith('.mp4'):
                try:
                    from matplotlib.animation import FFMpegWriter
                    writer = FFMpegWriter(fps=fps, bitrate=5000, codec='h264')
                    anim.save(output_filename, writer=writer, dpi=dpi)
                    print(f"ROI comparison animation saved to {output_filename}")
                except (FileNotFoundError, RuntimeError):
                    print("FFmpeg not found. Saving as GIF instead.")
                    output_filename = output_filename.replace('.mp4', '.gif')
                    from matplotlib.animation import PillowWriter
                    writer = PillowWriter(fps=fps)
                    anim.save(output_filename, writer=writer, dpi=dpi)
                    print(f"ROI comparison animation saved to {output_filename}")
            else:
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
                anim.save(output_filename, writer=writer, dpi=dpi)
                print(f"ROI comparison animation saved to {output_filename}")
        
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Returning animation object instead.")
        
        plt.tight_layout()
        return anim



    def generate_sensitivity_calibration(self, calibration_energy, normalization_method='min_pixel', 
                                        normalization_pixel=None, exclude_zero=True, min_threshold=None,
                                        plot=True, save_calibration=False, save_path=None, 
                                        calibration_name='sensitivity_calibration'):
        """
        Generate an image sensitivity calibration from a specific energy slice.
        
        Parameters:
        -----------
        calibration_energy : float
            Energy value to use for generating the calibration
        normalization_method : str, optional
            Method for normalization:
            - 'min_pixel': Divide by the minimum intensity pixel (default)
            - 'specific_pixel': Divide by a user-specified pixel location
            - 'mean': Divide by the mean intensity of the image
            - 'median': Divide by the median intensity of the image
        normalization_pixel : tuple, optional
            (x, y) coordinates of pixel to use for normalization when method is 'specific_pixel'
        exclude_zero : bool, optional
            If True, exclude zero/near-zero pixels from min calculation. Default is True.
        min_threshold : float, optional
            Minimum threshold for valid pixels. If None, uses 1% of max intensity.
        plot : bool, optional
            If True, plot the calibration image and statistics
        save_calibration : bool, optional
            If True, save the calibration to file
        save_path : str, optional
            Directory to save calibration files. If None, uses current directory.
        calibration_name : str, optional
            Base name for saved calibration files
            
        Returns:
        --------
        calibration_image : numpy.ndarray
            2D array containing the sensitivity calibration factors
        calibration_info : dict
            Dictionary containing calibration metadata and statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Find the closest energy to the requested calibration energy
        energy_slice = self.data.sel(energy=calibration_energy, method='nearest')
        actual_energy = float(energy_slice.energy.values)
        
        print(f"Generating sensitivity calibration from energy {actual_energy:.2f} eV")
        print(f"(Requested: {calibration_energy:.2f} eV)")
        
        # Get the 2D image data
        image_data = energy_slice.values.astype(float)
        
        # Handle zero/low intensity pixels
        if min_threshold is None:
            min_threshold = np.max(image_data) * 0.01  # 1% of max intensity
        
        if exclude_zero:
            valid_mask = image_data > min_threshold
            if not np.any(valid_mask):
                raise ValueError(f"No pixels above threshold {min_threshold:.3f}. Try lowering min_threshold.")
            print(f"Excluding {np.sum(~valid_mask)} pixels below threshold {min_threshold:.3f}")
        else:
            valid_mask = np.ones_like(image_data, dtype=bool)
        
        # Calculate normalization value based on method
        if normalization_method == 'min_pixel':
            if exclude_zero:
                norm_value = np.min(image_data[valid_mask])
            else:
                norm_value = np.min(image_data)
            norm_description = f"minimum pixel value: {norm_value:.3f}"
            
        elif normalization_method == 'specific_pixel':
            if normalization_pixel is None:
                raise ValueError("normalization_pixel must be specified when using 'specific_pixel' method")
            x, y = normalization_pixel
            if x < 0 or x >= image_data.shape[1] or y < 0 or y >= image_data.shape[0]:
                raise ValueError(f"Pixel coordinates ({x}, {y}) are outside image bounds")
            norm_value = image_data[y, x]
            norm_description = f"pixel at ({x}, {y}): {norm_value:.3f}"
            
        elif normalization_method == 'mean':
            if exclude_zero:
                norm_value = np.mean(image_data[valid_mask])
            else:
                norm_value = np.mean(image_data)
            norm_description = f"mean intensity: {norm_value:.3f}"
            
        elif normalization_method == 'median':
            if exclude_zero:
                norm_value = np.median(image_data[valid_mask])
            else:
                norm_value = np.median(image_data)
            norm_description = f"median intensity: {norm_value:.3f}"
            
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")
        
        print(f"Normalizing by {norm_description}")
        
        # Generate calibration image
        # Avoid division by zero
        safe_image = np.where(image_data > min_threshold/10, image_data, min_threshold/10)
        calibration_image = safe_image / norm_value
        
        # Handle invalid pixels
        if exclude_zero:
            calibration_image[~valid_mask] = 1.0  # Set invalid pixels to no correction
        
        # Calculate statistics
        cal_stats = {
            'mean': np.mean(calibration_image[valid_mask]),
            'median': np.median(calibration_image[valid_mask]),
            'std': np.std(calibration_image[valid_mask]),
            'min': np.min(calibration_image[valid_mask]),
            'max': np.max(calibration_image[valid_mask]),
            'range': np.max(calibration_image[valid_mask]) - np.min(calibration_image[valid_mask])
        }
        
        # Store calibration info
        calibration_info = {
            'calibration_energy': actual_energy,
            'requested_energy': calibration_energy,
            'normalization_method': normalization_method,
            'normalization_value': norm_value,
            'normalization_pixel': normalization_pixel,
            'min_threshold': min_threshold,
            'exclude_zero': exclude_zero,
            'image_shape': image_data.shape,
            'valid_pixels': np.sum(valid_mask),
            'total_pixels': image_data.size,
            'statistics': cal_stats
        }
        
        print(f"Calibration statistics:")
        print(f"  Valid pixels: {calibration_info['valid_pixels']:,} / {calibration_info['total_pixels']:,}")
        print(f"  Mean correction factor: {cal_stats['mean']:.3f}")
        print(f"  Correction range: {cal_stats['min']:.3f} - {cal_stats['max']:.3f}")
        print(f"  Standard deviation: {cal_stats['std']:.3f}")
        
        # Plot calibration if requested
        if plot:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Original image
            im1 = ax1.imshow(image_data, cmap='viridis', origin='lower')
            ax1.set_title(f'Original Image\nEnergy: {actual_energy:.2f} eV')
            ax1.set_xlabel('Pixel X')
            ax1.set_ylabel('Pixel Y')
            plt.colorbar(im1, ax=ax1, label='Intensity')
            
            # Mark normalization pixel if using specific pixel method
            if normalization_method == 'specific_pixel':
                ax1.plot(normalization_pixel[0], normalization_pixel[1], 'r+', 
                        markersize=15, markeredgewidth=3, label='Norm pixel')
                ax1.legend()
            
            # Calibration image
            im2 = ax2.imshow(calibration_image, cmap='RdBu_r', origin='lower', vmin=0.5, vmax=1.5)
            ax2.set_title('Sensitivity Calibration\n(Correction Factors)')
            ax2.set_xlabel('Pixel X')
            ax2.set_ylabel('Pixel Y')
            plt.colorbar(im2, ax=ax2, label='Correction Factor')
            
            # Line profiles through the center
            center_y = image_data.shape[0] // 2
            center_x = image_data.shape[1] // 2
            
            # Horizontal line profile
            ax3.plot(image_data[center_y, :], 'b-', linewidth=2, label='Original')
            ax3_twin = ax3.twinx()
            ax3_twin.plot(calibration_image[center_y, :], 'r-', linewidth=2, label='Correction')
            ax3.set_xlabel('Pixel X')
            ax3.set_ylabel('Original Intensity', color='b')
            ax3_twin.set_ylabel('Correction Factor', color='r')
            ax3.set_title(f'Horizontal Profile (Y = {center_y})')
            ax3.grid(True, alpha=0.3)
            
            # Add legends
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Vertical line profile
            ax4.plot(image_data[:, center_x], 'b-', linewidth=2, label='Original')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(calibration_image[:, center_x], 'r-', linewidth=2, label='Correction')
            ax4.set_xlabel('Pixel Y')
            ax4.set_ylabel('Original Intensity', color='b')
            ax4_twin.set_ylabel('Correction Factor', color='r')
            ax4.set_title(f'Vertical Profile (X = {center_x})')
            ax4.grid(True, alpha=0.3)
            
            # Add legends
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Add text box with statistics
            stats_text = f"""Calibration Statistics:
    Mean factor: {cal_stats['mean']:.3f}
    Std dev: {cal_stats['std']:.3f}
    Range: {cal_stats['min']:.3f} - {cal_stats['max']:.3f}
    Valid pixels: {calibration_info['valid_pixels']:,}"""
            
            # Add text box to the calibration image
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8), fontsize=9)
            
            plt.tight_layout()
            plt.show()
        
        # Save calibration if requested
        if save_calibration:
            import os
            if save_path is None:
                save_path = os.getcwd()
            os.makedirs(save_path, exist_ok=True)
            
            # Save calibration image as numpy array
            cal_filename = f"{calibration_name}.npz"
            cal_filepath = os.path.join(save_path, cal_filename)
            np.savez_compressed(cal_filepath,
                            calibration_image=calibration_image,
                            calibration_info=calibration_info,
                            original_image=image_data)
            
            # Save calibration info as text
            info_filename = f"{calibration_name}_info.txt"
            info_filepath = os.path.join(save_path, info_filename)
            with open(info_filepath, 'w') as f:
                f.write("Image Sensitivity Calibration Information\n")
                f.write("=" * 50 + "\n")
                f.write(f"Calibration energy: {calibration_info['calibration_energy']:.3f} eV\n")
                f.write(f"Requested energy: {calibration_info['requested_energy']:.3f} eV\n")
                f.write(f"Normalization method: {calibration_info['normalization_method']}\n")
                f.write(f"Normalization value: {calibration_info['normalization_value']:.6f}\n")
                if calibration_info['normalization_pixel']:
                    f.write(f"Normalization pixel: {calibration_info['normalization_pixel']}\n")
                f.write(f"Minimum threshold: {calibration_info['min_threshold']:.6f}\n")
                f.write(f"Exclude zero pixels: {calibration_info['exclude_zero']}\n")
                f.write(f"Image shape: {calibration_info['image_shape']}\n")
                f.write(f"Valid pixels: {calibration_info['valid_pixels']:,} / {calibration_info['total_pixels']:,}\n")
                f.write("\nStatistics:\n")
                for key, value in calibration_info['statistics'].items():
                    f.write(f"  {key}: {value:.6f}\n")
            
            print(f"Calibration saved to {cal_filepath}")
            print(f"Calibration info saved to {info_filepath}")
        
        return calibration_image, calibration_info


    def apply_sensitivity_calibration(self, calibration_image=None, calibration_file=None, 
                                    method='divide', invert_calibration=False):
        """
        Apply sensitivity calibration to all images in the dataset.
        
        Parameters:
        -----------
        calibration_image : numpy.ndarray, optional
            2D calibration image to apply. Either this or calibration_file must be provided.
        calibration_file : str, optional
            Path to saved calibration file (.npz format). Either this or calibration_image must be provided.
        method : str, optional
            How to apply calibration:
            - 'divide': Divide each image by calibration (default, removes sensitivity variations)
            - 'multiply': Multiply each image by calibration
        invert_calibration : bool, optional
            If True, invert the calibration before applying (1/calibration). Default is False.
            
        Returns:
        --------
        None (modifies self.data in place)
        calibration_info : dict
            Information about the applied calibration
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Load calibration
        if calibration_image is not None:
            cal_image = calibration_image.copy()
            cal_info = {'source': 'provided_array', 'method': method}
            print("Using provided calibration image")
        elif calibration_file is not None:
            print(f"Loading calibration from {calibration_file}")
            cal_data = np.load(calibration_file)
            cal_image = cal_data['calibration_image']
            if 'calibration_info' in cal_data:
                cal_info = cal_data['calibration_info'].item()  # .item() converts numpy array back to dict
                cal_info['source'] = calibration_file
                cal_info['method'] = method
            else:
                cal_info = {'source': calibration_file, 'method': method}
            print(f"Loaded calibration from energy {cal_info.get('calibration_energy', 'unknown'):.3f} eV")
        else:
            raise ValueError("Either calibration_image or calibration_file must be provided")
        
        # Check image dimensions
        data_shape = self.data.isel(energy=0).shape
        if cal_image.shape != data_shape:
            raise ValueError(f"Calibration image shape {cal_image.shape} does not match "
                            f"data image shape {data_shape}")
        
        # Invert calibration if requested
        if invert_calibration:
            # Avoid division by zero
            cal_image = np.where(np.abs(cal_image) > 1e-10, 1.0 / cal_image, 1.0)
            print("Applied calibration inversion")
        
        # Apply calibration to all energy slices
        print(f"Applying sensitivity calibration to {len(self.data.energy)} energy slices...")
        
        if method == 'divide':
            # Avoid division by zero
            safe_cal = np.where(np.abs(cal_image) > 1e-10, cal_image, 1.0)
            corrected_data = self.data / safe_cal
            print("Applied calibration by division (removing sensitivity variations)")
        elif method == 'multiply':
            corrected_data = self.data * cal_image
            print("Applied calibration by multiplication")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'divide' or 'multiply'.")
        
        # Update the data
        self.data = corrected_data
        
        # Update attributes
        if not hasattr(self.data, 'attrs'):
            self.data.attrs = {}
        self.data.attrs['sensitivity_calibration_applied'] = True
        self.data.attrs['calibration_method'] = method
        self.data.attrs['calibration_inverted'] = invert_calibration
        if 'calibration_energy' in cal_info:
            self.data.attrs['calibration_energy'] = cal_info['calibration_energy']
        
        print("Sensitivity calibration applied successfully")
        
        return cal_info


    def load_sensitivity_calibration(self, calibration_file):
        """
        Load a sensitivity calibration from file without applying it.
        
        Parameters:
        -----------
        calibration_file : str
            Path to saved calibration file (.npz format)
            
        Returns:
        --------
        calibration_image : numpy.ndarray
            2D calibration image
        calibration_info : dict
            Calibration metadata and statistics
        """
        print(f"Loading calibration from {calibration_file}")
        cal_data = np.load(calibration_file)
        calibration_image = cal_data['calibration_image']
        
        if 'calibration_info' in cal_data:
            calibration_info = cal_data['calibration_info'].item()
            print(f"Calibration from energy {calibration_info.get('calibration_energy', 'unknown'):.3f} eV")
            print(f"Normalization method: {calibration_info.get('normalization_method', 'unknown')}")
            print(f"Image shape: {calibration_info.get('image_shape', 'unknown')}")
        else:
            calibration_info = {'source': calibration_file}
            print("Limited calibration information available")
        
        return calibration_image, calibration_info


    def preview_sensitivity_calibration(self, calibration_image=None, calibration_file=None, 
                                    preview_energy=None, method='divide', invert_calibration=False):
        """
        Preview the effect of sensitivity calibration on a specific energy slice.
        
        Parameters:
        -----------
        calibration_image : numpy.ndarray, optional
            2D calibration image. Either this or calibration_file must be provided.
        calibration_file : str, optional
            Path to saved calibration file (.npz format)
        preview_energy : float, optional
            Energy to use for preview. If None, uses the middle energy of the dataset.
        method : str, optional
            How calibration would be applied ('divide' or 'multiply')
        invert_calibration : bool, optional
            Whether to invert calibration before applying
            
        Returns:
        --------
        None (creates comparison plot)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Load calibration
        if calibration_image is not None:
            cal_image = calibration_image.copy()
            print("Using provided calibration image")
        elif calibration_file is not None:
            cal_image, cal_info = self.load_sensitivity_calibration(calibration_file)
            print(f"Loaded calibration for preview")
        else:
            raise ValueError("Either calibration_image or calibration_file must be provided")
        
        # Select preview energy
        if preview_energy is None:
            mid_idx = len(self.data.energy) // 2
            preview_energy = float(self.data.energy.isel(energy=mid_idx).values)
        
        # Get original image
        original_slice = self.data.sel(energy=preview_energy, method='nearest')
        original_image = original_slice.values
        actual_energy = float(original_slice.energy.values)
        
        # Apply calibration for preview
        if invert_calibration:
            cal_preview = np.where(np.abs(cal_image) > 1e-10, 1.0 / cal_image, 1.0)
        else:
            cal_preview = cal_image
        
        if method == 'divide':
            safe_cal = np.where(np.abs(cal_preview) > 1e-10, cal_preview, 1.0)
            corrected_image = original_image / safe_cal
        else:  # multiply
            corrected_image = original_image * cal_preview
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        im1 = ax1.imshow(original_image, cmap='viridis', origin='lower')
        ax1.set_title(f'Original Image\nEnergy: {actual_energy:.2f} eV')
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # Calibration image
        im2 = ax2.imshow(cal_preview, cmap='RdBu_r', origin='lower')
        cal_title = 'Calibration'
        if invert_calibration:
            cal_title += ' (Inverted)'
        ax2.set_title(cal_title)
        plt.colorbar(im2, ax=ax2, label='Correction Factor')
        
        # Corrected image
        im3 = ax3.imshow(corrected_image, cmap='viridis', origin='lower')
        ax3.set_title(f'After Calibration ({method})')
        plt.colorbar(im3, ax=ax3, label='Corrected Intensity')
        
        # Difference image
        difference = corrected_image - original_image
        im4 = ax4.imshow(difference, cmap='RdBu_r', origin='lower')
        ax4.set_title('Difference (After - Before)')
        plt.colorbar(im4, ax=ax4, label='Intensity Change')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('Pixel X')
            ax.set_ylabel('Pixel Y')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nPreview Statistics for Energy {actual_energy:.2f} eV:")
        print(f"Original image - Mean: {np.mean(original_image):.3f}, Std: {np.std(original_image):.3f}")
        print(f"Corrected image - Mean: {np.mean(corrected_image):.3f}, Std: {np.std(corrected_image):.3f}")
        print(f"Relative change in std: {(np.std(corrected_image) / np.std(original_image) - 1) * 100:.1f}%")



    def save_2d_images_at_energies(self, save_energies, save_path=None, 
                                image_format='png', image_dpi=300, image_cmap='viridis', 
                                contrast_percentiles=(2, 98), roi_list=None, roi_labels=None,
                                add_roi_overlays=True, add_scalebar=False, 
                                scalebar_length_um=None, pixel_size_um=None,
                                filename_prefix='image', include_energy_in_filename=True,
                                save_colorbar=True, figsize=(8, 6)):
        """
        Save 2D images from the datacube at specified energy values.
        
        Parameters:
        -----------
        save_energies : list of float
            List of energy values where 2D images should be saved
        save_path : str, optional
            Directory path to save images. If None, saves to current directory
        image_format : str, optional
            Format to save images ('png', 'tiff', 'pdf', 'svg', 'jpg'). Default is 'png'
        image_dpi : int, optional
            DPI for saved images. Default is 300 for publication quality
        image_cmap : str, optional
            Colormap for 2D images. Default is 'viridis'
        contrast_percentiles : tuple, optional
            (low, high) percentiles for image contrast adjustment. Default is (2, 98)
        roi_list : list of tuples, optional
            List of ROIs to overlay as (x_min, y_min, width, height). If None, no ROIs shown.
        roi_labels : list of str, optional
            Labels for each ROI. If None, will use "ROI 1", "ROI 2", etc.
        add_roi_overlays : bool, optional
            If True and roi_list provided, save additional versions with ROI overlays
        add_scalebar : bool, optional
            If True, add scale bar to images (requires pixel_size_um and scalebar_length_um)
        scalebar_length_um : float, optional
            Length of scale bar in micrometers
        pixel_size_um : float, optional
            Size of each pixel in micrometers (needed for scale bar)
        filename_prefix : str, optional
            Prefix for saved filenames. Default is 'image'
        include_energy_in_filename : bool, optional
            If True, include energy value in filename. Default is True
        save_colorbar : bool, optional
            If True, save images with colorbar. Default is True
        figsize : tuple, optional
            Figure size as (width, height) in inches. Default is (8, 6)
            
        Returns:
        --------
        saved_files : list
            List of saved file paths
        image_info : dict
            Dictionary with information about each saved image
        """
        import os
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib_scalebar.scalebar import ScaleBar
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Set up save directory
        if save_path is None:
            save_path = os.getcwd()
        os.makedirs(save_path, exist_ok=True)
        
        # Prepare ROI information
        if roi_list is not None:
            if roi_labels is None:
                roi_labels = [f"ROI {i+1}" for i in range(len(roi_list))]
            if len(roi_labels) != len(roi_list):
                raise ValueError("Number of ROI labels must match number of ROIs")
            colors = plt.cm.tab10(np.linspace(0, 1, len(roi_list)))
        
        saved_files = []
        image_info = {}
        
        print(f"Saving 2D images at {len(save_energies)} energy points to {save_path}")
        
        for energy in save_energies:
            # Get the closest energy slice
            energy_slice = self.data.sel(energy=energy, method='nearest')
            actual_energy = float(energy_slice.energy.values)
            image_data = energy_slice.values
            
            print(f"Processing energy {actual_energy:.2f} eV (requested: {energy:.2f} eV)")
            
            # Calculate contrast limits
            vmin, vmax = np.nanpercentile(image_data, contrast_percentiles)
            
            # Create filename
            if include_energy_in_filename:
                clean_energy = f"{actual_energy:.2f}".replace('.', 'p')
                base_filename = f"{filename_prefix}_{clean_energy}eV"
            else:
                base_filename = f"{filename_prefix}_{len(saved_files)+1:03d}"
            
            # Save main image without ROIs
            fig, ax = plt.subplots(figsize=figsize)
            
            im = ax.imshow(image_data, cmap=image_cmap, origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f'Energy: {actual_energy:.2f} eV')
            ax.set_xlabel('Pixel X')
            ax.set_ylabel('Pixel Y')
            
            # Add colorbar if requested
            if save_colorbar:
                cbar = plt.colorbar(im, ax=ax, label='Intensity')
            
            # Add scale bar if requested
            if add_scalebar and pixel_size_um is not None and scalebar_length_um is not None:
                scalebar = ScaleBar(pixel_size_um * 1e-6, units='m', 
                                length_fraction=scalebar_length_um/pixel_size_um/image_data.shape[1],
                                location='lower right', box_alpha=0.8, color='white')
                ax.add_artist(scalebar)
            
            plt.tight_layout()
            
            # Save main image
            main_filename = f"{base_filename}.{image_format}"
            main_filepath = os.path.join(save_path, main_filename)
            plt.savefig(main_filepath, dpi=image_dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            saved_files.append(main_filepath)
            
            plt.close()
            
            # Save image with ROI overlays if requested
            if add_roi_overlays and roi_list is not None:
                fig, ax = plt.subplots(figsize=figsize)
                
                im = ax.imshow(image_data, cmap=image_cmap, origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f'Energy: {actual_energy:.2f} eV (with ROIs)')
                ax.set_xlabel('Pixel X')
                ax.set_ylabel('Pixel Y')
                
                # Add ROI rectangles
                for i, (roi, label) in enumerate(zip(roi_list, roi_labels)):
                    x_min, y_min, width, height = roi
                    rect = Rectangle((x_min, y_min), width, height, 
                                edgecolor=colors[i], facecolor='none', linewidth=2)
                    ax.add_patch(rect)
                    # Add label
                    ax.text(x_min + width/2, y_min + height + 2, label, 
                        color=colors[i], fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                # Add colorbar if requested
                if save_colorbar:
                    cbar = plt.colorbar(im, ax=ax, label='Intensity')
                
                # Add scale bar if requested
                if add_scalebar and pixel_size_um is not None and scalebar_length_um is not None:
                    scalebar = ScaleBar(pixel_size_um * 1e-6, units='m', 
                                    length_fraction=scalebar_length_um/pixel_size_um/image_data.shape[1],
                                    location='lower right', box_alpha=0.8, color='white')
                    ax.add_artist(scalebar)
                
                plt.tight_layout()
                
                # Save ROI overlay image
                roi_filename = f"{base_filename}_with_ROIs.{image_format}"
                roi_filepath = os.path.join(save_path, roi_filename)
                plt.savefig(roi_filepath, dpi=image_dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
                saved_files.append(roi_filepath)
                
                plt.close()
            
            # Store image information
            image_info[actual_energy] = {
                'requested_energy': energy,
                'actual_energy': actual_energy,
                'image_shape': image_data.shape,
                'intensity_range': (np.nanmin(image_data), np.nanmax(image_data)),
                'contrast_range': (vmin, vmax),
                'main_file': main_filename,
                'roi_file': f"{base_filename}_with_ROIs.{image_format}" if add_roi_overlays and roi_list else None
            }
        
        print(f"\nSaved {len(saved_files)} images:")
        for file_path in saved_files:
            print(f"  - {os.path.basename(file_path)}")
        
        # Save summary information
        summary_filename = f"{filename_prefix}_summary.txt"
        summary_filepath = os.path.join(save_path, summary_filename)
        with open(summary_filepath, 'w') as f:
            f.write("2D Image Export Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Total images saved: {len(saved_files)}\n")
            f.write(f"Image format: {image_format.upper()}\n")
            f.write(f"Image DPI: {image_dpi}\n")
            f.write(f"Colormap: {image_cmap}\n")
            f.write(f"Contrast percentiles: {contrast_percentiles}\n")
            f.write(f"Figure size: {figsize}\n")
            if roi_list:
                f.write(f"ROI overlays: {add_roi_overlays}\n")
                f.write(f"Number of ROIs: {len(roi_list)}\n")
            if add_scalebar:
                f.write(f"Scale bar: {scalebar_length_um} μm\n")
                f.write(f"Pixel size: {pixel_size_um} μm\n")
            f.write("\nEnergy Points:\n")
            for energy, info in image_info.items():
                f.write(f"  {energy:.2f} eV (requested: {info['requested_energy']:.2f} eV)\n")
                f.write(f"    Files: {info['main_file']}")
                if info['roi_file']:
                    f.write(f", {info['roi_file']}")
                f.write(f"\n    Intensity range: {info['intensity_range'][0]:.2e} - {info['intensity_range'][1]:.2e}\n")
                f.write(f"    Contrast range: {info['contrast_range'][0]:.2e} - {info['contrast_range'][1]:.2e}\n")
        
        saved_files.append(summary_filepath)
        print(f"  - {os.path.basename(summary_filepath)} (summary)")
        
        return saved_files, image_info


    def save_energy_series_comparison(self, save_energies, save_path=None, 
                                    image_format='png', image_dpi=300, image_cmap='viridis',
                                    contrast_method='individual', contrast_percentiles=(2, 98),
                                    roi_list=None, roi_labels=None, figsize=(15, 10),
                                    filename='energy_series_comparison'):
        """
        Save a single figure showing multiple energy slices in a grid for easy comparison.
        
        Parameters:
        -----------
        save_energies : list of float
            List of energy values to include in the comparison
        save_path : str, optional
            Directory path to save the comparison image
        image_format : str, optional
            Format to save image ('png', 'pdf', 'svg'). Default is 'png'
        image_dpi : int, optional
            DPI for saved image. Default is 300
        image_cmap : str, optional
            Colormap for images. Default is 'viridis'
        contrast_method : str, optional
            How to set contrast: 'individual' (per image) or 'global' (same for all)
        contrast_percentiles : tuple, optional
            Percentiles for contrast adjustment. Default is (2, 98)
        roi_list : list of tuples, optional
            List of ROIs to overlay
        roi_labels : list of str, optional
            Labels for ROIs
        figsize : tuple, optional
            Figure size as (width, height). Default is (15, 10)
        filename : str, optional
            Base filename for saved comparison
            
        Returns:
        --------
        saved_file : str
            Path to saved comparison image
        """
        import os
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        if save_path is None:
            save_path = os.getcwd()
        os.makedirs(save_path, exist_ok=True)
        
        # Calculate grid dimensions
        n_images = len(save_energies)
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))
        
        # Prepare data for all energies
        energy_data = []
        actual_energies = []
        
        for energy in save_energies:
            energy_slice = self.data.sel(energy=energy, method='nearest')
            actual_energy = float(energy_slice.energy.values)
            image_data = energy_slice.values
            energy_data.append(image_data)
            actual_energies.append(actual_energy)
        
        # Calculate contrast limits
        if contrast_method == 'global':
            all_data = np.concatenate([img.flatten() for img in energy_data])
            vmin, vmax = np.nanpercentile(all_data, contrast_percentiles)
            contrast_limits = [(vmin, vmax)] * n_images
        else:  # individual
            contrast_limits = []
            for img in energy_data:
                vmin, vmax = np.nanpercentile(img, contrast_percentiles)
                contrast_limits.append((vmin, vmax))
        
        # Create comparison figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_images == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Prepare ROI colors if needed
        if roi_list is not None:
            if roi_labels is None:
                roi_labels = [f"ROI {i+1}" for i in range(len(roi_list))]
            colors = plt.cm.tab10(np.linspace(0, 1, len(roi_list)))
        
        # Plot each energy slice
        for i, (img_data, actual_energy, (vmin, vmax)) in enumerate(zip(energy_data, actual_energies, contrast_limits)):
            ax = axes[i]
            
            im = ax.imshow(img_data, cmap=image_cmap, origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f'{actual_energy:.2f} eV', fontsize=12)
            ax.set_xlabel('Pixel X')
            ax.set_ylabel('Pixel Y')
            
            # Add ROIs if provided
            if roi_list is not None:
                for j, (roi, label) in enumerate(zip(roi_list, roi_labels)):
                    x_min, y_min, width, height = roi
                    rect = Rectangle((x_min, y_min), width, height, 
                                edgecolor=colors[j], facecolor='none', linewidth=1.5)
                    ax.add_patch(rect)
                    # Add label only to first image to avoid clutter
                    if i == 0:
                        ax.text(x_min + width/2, y_min + height + 2, label, 
                            color=colors[j], fontweight='bold', ha='center', fontsize=10)
            
            # Add individual colorbar for each subplot
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Energy Series Comparison ({len(save_energies)} energies)', fontsize=16)
        plt.tight_layout()
        
        # Save the comparison
        comparison_filename = f"{filename}.{image_format}"
        comparison_filepath = os.path.join(save_path, comparison_filename)
        plt.savefig(comparison_filepath, dpi=image_dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Energy series comparison saved to {comparison_filepath}")
        
        return comparison_filepath
    
    def process_image_pixel_by_pixel(self, pre_edge_norm_range=None, pre_edge_sub_range=None, 
                                post_edge_range=None, do_pre_edge_norm=True, do_pre_edge_sub=True,
                                do_post_edge_norm=True, plot_comparison=False, comparison_energies=None,
                                plot_pixel_spectrum=False, pixel_location=None, 
                                save_processed_data=False, save_path=None, save_name='processed_image'):
        """
        Apply spectral processing (pre-edge normalization, subtraction, post-edge normalization) 
        to each pixel individually across the entire image.
        
        Parameters:
        -----------
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
        plot_comparison : bool, optional
            If True, plot comparison of original vs processed images at specified energies
        comparison_energies : list of float, optional
            Energy values to use for comparison plots. If None, uses 3 evenly spaced energies.
        plot_pixel_spectrum : bool, optional
            If True, plot the spectrum processing for a specific pixel
        pixel_location : tuple, optional
            (x, y) coordinates of pixel to plot spectrum for. If None, uses center pixel.
        save_processed_data : bool, optional
            If True, save the processed datacube
        save_path : str, optional
            Directory to save processed data. If None, uses current directory.
        save_name : str, optional
            Base name for saved files
            
        Returns:
        --------
        processed_data : xarray.DataArray
            The processed datacube with same structure as original but processed spectra
        processing_info : dict
            Information about the processing parameters used
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import linregress
        from scipy.interpolate import interp1d
        import os
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        print("Starting pixel-by-pixel spectral processing...")
        
        # Get energy coordinates and data dimensions
        energies = self.data.energy.values
        n_energies, n_y, n_x = self.data.shape
        
        # Set default ranges if not provided
        e_min, e_max = energies.min(), energies.max()
        e_range = e_max - e_min
        
        if pre_edge_norm_range is None:
            pre_edge_norm_range = (e_min, e_min + 0.2 * e_range)
        if pre_edge_sub_range is None:
            pre_edge_sub_range = (e_min, e_min + 0.1 * e_range)
        if post_edge_range is None:
            post_edge_range = (e_max - 0.2 * e_range, e_max)
        
        print(f"Processing {n_x} x {n_y} pixels across {n_energies} energy points")
        print(f"Pre-edge norm range: {pre_edge_norm_range[0]:.2f} - {pre_edge_norm_range[1]:.2f} eV")
        print(f"Pre-edge sub range: {pre_edge_sub_range[0]:.2f} - {pre_edge_sub_range[1]:.2f} eV")
        print(f"Post-edge range: {post_edge_range[0]:.2f} - {post_edge_range[1]:.2f} eV")
        
        # Create copy of original data for processing
        processed_data = self.data.copy()
        
        # Initialize arrays to store processing statistics
        processing_stats = {
            'pre_edge_slopes': np.full((n_y, n_x), np.nan),
            'pre_edge_intercepts': np.full((n_y, n_x), np.nan),
            'pre_edge_values': np.full((n_y, n_x), np.nan),
            'post_edge_values': np.full((n_y, n_x), np.nan),
            'valid_pixels': np.zeros((n_y, n_x), dtype=bool)
        }
        
        # Process each pixel
        pixels_processed = 0
        pixels_failed = 0
        
        for y in range(n_y):
            if y % 10 == 0:  # Progress update every 10 rows
                progress = (y * n_x) / (n_y * n_x) * 100
                print(f"Progress: {progress:.1f}% (row {y}/{n_y})")
            
            for x in range(n_x):
                try:
                    # Extract spectrum for this pixel
                    pixel_spectrum = self.data[:, y, x].values
                    
                    # Skip pixels with all zeros or very low intensity
                    if np.all(pixel_spectrum <= 0) or np.max(pixel_spectrum) < 1e-10:
                        pixels_failed += 1
                        continue
                    
                    # Create spectrum array [Energy, Intensity]
                    spectrum_np = np.column_stack((energies, pixel_spectrum))
                    
                    # Apply processing steps in order
                    spectrum_processed = spectrum_np.copy()
                    
                    # Step 1: Pre-edge slope normalization
                    if do_pre_edge_norm:
                        try:
                            # Find indices for pre-edge normalization range
                            norm_mask = (energies >= pre_edge_norm_range[0]) & (energies <= pre_edge_norm_range[1])
                            if np.sum(norm_mask) < 2:
                                raise ValueError("Not enough points for pre-edge normalization")
                            
                            # Fit line to pre-edge region
                            slope, intercept, r_value, p_value, std_err = linregress(
                                energies[norm_mask], pixel_spectrum[norm_mask])
                            
                            # Store slope and intercept
                            processing_stats['pre_edge_slopes'][y, x] = slope
                            processing_stats['pre_edge_intercepts'][y, x] = intercept
                            
                            # Calculate fitted line for all energies and normalize
                            line_values = slope * energies + intercept
                            epsilon = 1e-10
                            spectrum_processed[:, 1] = spectrum_processed[:, 1] / (line_values + epsilon)
                            
                        except Exception as e:
                            # If slope normalization fails, skip this pixel
                            pixels_failed += 1
                            continue
                    
                    # Step 2: Pre-edge background subtraction
                    if do_pre_edge_sub:
                        try:
                            # Find indices for pre-edge subtraction range
                            sub_mask = (energies >= pre_edge_sub_range[0]) & (energies <= pre_edge_sub_range[1])
                            if np.sum(sub_mask) < 1:
                                raise ValueError("No points in pre-edge subtraction range")
                            
                            # Calculate pre-edge value and subtract
                            pre_edge_value = np.mean(spectrum_processed[sub_mask, 1])
                            processing_stats['pre_edge_values'][y, x] = pre_edge_value
                            spectrum_processed[:, 1] = spectrum_processed[:, 1] - pre_edge_value
                            
                        except Exception as e:
                            pixels_failed += 1
                            continue
                    
                    # Step 3: Post-edge normalization
                    if do_post_edge_norm:
                        try:
                            # Find indices for post-edge range
                            post_mask = (energies >= post_edge_range[0]) & (energies <= post_edge_range[1])
                            if np.sum(post_mask) < 1:
                                raise ValueError("No points in post-edge range")
                            
                            # Calculate post-edge value and normalize
                            post_edge_value = np.mean(spectrum_processed[post_mask, 1])
                            processing_stats['post_edge_values'][y, x] = post_edge_value
                            
                            epsilon = 1e-10
                            spectrum_processed[:, 1] = spectrum_processed[:, 1] / (post_edge_value + epsilon)
                            
                        except Exception as e:
                            pixels_failed += 1
                            continue
                    
                    # Store processed spectrum back to datacube
                    processed_data[:, y, x] = spectrum_processed[:, 1]
                    processing_stats['valid_pixels'][y, x] = True
                    pixels_processed += 1
                    
                except Exception as e:
                    pixels_failed += 1
                    continue
        
        print(f"Processing complete: {pixels_processed:,} pixels processed, {pixels_failed:,} pixels failed")
        
        # Store processing information
        processing_info = {
            'pre_edge_norm_range': pre_edge_norm_range,
            'pre_edge_sub_range': pre_edge_sub_range,
            'post_edge_range': post_edge_range,
            'do_pre_edge_norm': do_pre_edge_norm,
            'do_pre_edge_sub': do_pre_edge_sub,
            'do_post_edge_norm': do_post_edge_norm,
            'pixels_processed': pixels_processed,
            'pixels_failed': pixels_failed,
            'processing_stats': processing_stats
        }
        
        # Update attributes
        processed_data.attrs = self.data.attrs.copy()
        processed_data.attrs['pixel_by_pixel_processing'] = True
        processed_data.attrs['processing_steps'] = f"norm:{do_pre_edge_norm}, sub:{do_pre_edge_sub}, post:{do_post_edge_norm}"
        
        # Plot comparison if requested
        if plot_comparison:
            if comparison_energies is None:
                # Use 3 evenly spaced energies
                energy_indices = np.linspace(0, len(energies)-1, 3, dtype=int)
                comparison_energies = energies[energy_indices]
            
            n_energies_plot = len(comparison_energies)
            fig, axes = plt.subplots(2, n_energies_plot, figsize=(5*n_energies_plot, 10))
            if n_energies_plot == 1:
                axes = axes.reshape(2, 1)
            
            for i, energy in enumerate(comparison_energies):
                # Original image
                orig_slice = self.data.sel(energy=energy, method='nearest')
                axes[0, i].imshow(orig_slice.values, cmap='viridis', origin='lower')
                axes[0, i].set_title(f'Original\n{float(orig_slice.energy.values):.2f} eV')
                axes[0, i].set_xlabel('Pixel X')
                axes[0, i].set_ylabel('Pixel Y')
                
                # Processed image
                proc_slice = processed_data.sel(energy=energy, method='nearest')
                im = axes[1, i].imshow(proc_slice.values, cmap='viridis', origin='lower')
                axes[1, i].set_title(f'Processed\n{float(proc_slice.energy.values):.2f} eV')
                axes[1, i].set_xlabel('Pixel X')
                axes[1, i].set_ylabel('Pixel Y')
                
                # Add colorbar to each processed image
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.show()
        
        # Plot pixel spectrum if requested
        if plot_pixel_spectrum:
            if pixel_location is None:
                pixel_location = (n_x // 2, n_y // 2)  # Center pixel
            
            x_pixel, y_pixel = pixel_location
            
            if x_pixel >= n_x or y_pixel >= n_y or x_pixel < 0 or y_pixel < 0:
                print(f"Warning: Pixel location ({x_pixel}, {y_pixel}) is outside image bounds. Using center pixel.")
                x_pixel, y_pixel = (n_x // 2, n_y // 2)
            
            # Extract original and processed spectra for this pixel
            orig_spectrum = self.data[:, y_pixel, x_pixel].values
            proc_spectrum = processed_data[:, y_pixel, x_pixel].values
            
            # Create processing steps visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Original spectrum with processing regions highlighted
            ax = axes[0, 0]
            ax.plot(energies, orig_spectrum, 'o-', label='Original', markersize=3)
            
            # Highlight processing regions
            if do_pre_edge_norm:
                norm_mask = (energies >= pre_edge_norm_range[0]) & (energies <= pre_edge_norm_range[1])
                ax.axvspan(pre_edge_norm_range[0], pre_edge_norm_range[1], alpha=0.3, color='green', 
                        label='Pre-edge Norm')
            if do_pre_edge_sub:
                sub_mask = (energies >= pre_edge_sub_range[0]) & (energies <= pre_edge_sub_range[1])
                ax.axvspan(pre_edge_sub_range[0], pre_edge_sub_range[1], alpha=0.3, color='red', 
                        label='Pre-edge Sub')
            if do_post_edge_norm:
                post_mask = (energies >= post_edge_range[0]) & (energies <= post_edge_range[1])
                ax.axvspan(post_edge_range[0], post_edge_range[1], alpha=0.3, color='purple', 
                        label='Post-edge Norm')
            
            ax.set_title(f'Original Spectrum\nPixel ({x_pixel}, {y_pixel})')
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Processed spectrum
            ax = axes[0, 1]
            ax.plot(energies, proc_spectrum, 'o-', color='orange', markersize=3)
            ax.set_title(f'Processed Spectrum\nPixel ({x_pixel}, {y_pixel})')
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Processed Intensity')
            ax.grid(True, alpha=0.3)
            
            # Add reference lines
            if do_post_edge_norm:
                ax.axhline(y=1, color='purple', linestyle='--', alpha=0.7, label='Post-edge Level')
            if do_pre_edge_sub:
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Pre-edge Level')
            ax.legend()
            
            # Comparison plot
            ax = axes[1, 0]
            ax.plot(energies, orig_spectrum, 'o-', label='Original', markersize=3, alpha=0.7)
            ax.plot(energies, proc_spectrum, 'o-', label='Processed', markersize=3)
            ax.set_title('Original vs Processed')
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Processing statistics for this pixel
            ax = axes[1, 1]
            ax.axis('off')
            
            stats_text = f"Processing Statistics for Pixel ({x_pixel}, {y_pixel}):\n\n"
            if processing_stats['valid_pixels'][y_pixel, x_pixel]:
                if do_pre_edge_norm:
                    slope = processing_stats['pre_edge_slopes'][y_pixel, x_pixel]
                    intercept = processing_stats['pre_edge_intercepts'][y_pixel, x_pixel]
                    stats_text += f"Pre-edge slope: {slope:.4f}\n"
                    stats_text += f"Pre-edge intercept: {intercept:.4f}\n"
                if do_pre_edge_sub:
                    pre_val = processing_stats['pre_edge_values'][y_pixel, x_pixel]
                    stats_text += f"Pre-edge value: {pre_val:.4f}\n"
                if do_post_edge_norm:
                    post_val = processing_stats['post_edge_values'][y_pixel, x_pixel]
                    stats_text += f"Post-edge value: {post_val:.4f}\n"
            else:
                stats_text += "Processing failed for this pixel"
            
            ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), fontsize=10)
            
            plt.tight_layout()
            plt.show()
        
        # Save processed data if requested
        if save_processed_data:
            import os
            if save_path is None:
                save_path = os.getcwd()
            os.makedirs(save_path, exist_ok=True)
            
            # Save processed datacube as NetCDF
            processed_filename = f"{save_name}_processed.nc"
            processed_filepath = os.path.join(save_path, processed_filename)
            processed_data.to_netcdf(processed_filepath)
            
            # Save processing statistics
            stats_filename = f"{save_name}_processing_stats.npz"
            stats_filepath = os.path.join(save_path, stats_filename)
            np.savez_compressed(stats_filepath, **processing_stats)
            
            # Save processing info
            info_filename = f"{save_name}_processing_info.txt"
            info_filepath = os.path.join(save_path, info_filename)
            with open(info_filepath, 'w') as f:
                f.write("Pixel-by-Pixel Processing Information\n")
                f.write("=" * 50 + "\n")
                f.write(f"Pre-edge norm range: {pre_edge_norm_range[0]:.3f} - {pre_edge_norm_range[1]:.3f} eV\n")
                f.write(f"Pre-edge sub range: {pre_edge_sub_range[0]:.3f} - {pre_edge_sub_range[1]:.3f} eV\n")
                f.write(f"Post-edge range: {post_edge_range[0]:.3f} - {post_edge_range[1]:.3f} eV\n")
                f.write(f"Pre-edge normalization: {do_pre_edge_norm}\n")
                f.write(f"Pre-edge subtraction: {do_pre_edge_sub}\n")
                f.write(f"Post-edge normalization: {do_post_edge_norm}\n")
                f.write(f"Pixels processed: {pixels_processed:,}\n")
                f.write(f"Pixels failed: {pixels_failed:,}\n")
                f.write(f"Success rate: {pixels_processed/(pixels_processed+pixels_failed)*100:.1f}%\n")
            
            print(f"Processed data saved to {processed_filepath}")
            print(f"Processing statistics saved to {stats_filepath}")
            print(f"Processing info saved to {info_filepath}")
        
        return processed_data, processing_info