import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from copy import deepcopy
import glob
from pathlib import Path
from collections import defaultdict
import itertools
from scipy.interpolate import interp1d
import datetime

class RSoXRProcessor:
    """
    Class for processing Resonant Soft X-ray Reflectivity (RSoXR) data
    """
    
    def __init__(self, log_file=None, calibration_file=None):
        """
        Initialize the processor with an optional log file for metadata
        and calibration file for photon conversion
        
        Parameters:
        -----------
        log_file : str
            Path to the log file containing metadata
        calibration_file : str
            Path to the photodiode calibration file with energy and responsivity
        """
        self.log_data = None
        self.calibration_data = None
        self.log_file = log_file
        
        if log_file:
            self.load_log_file(log_file)
            
        if calibration_file:
            self.load_calibration_file(calibration_file)
        
        self.scan_registry = {}  # Maps scan numbers to file info
        self.next_scan_number = 1
    
    def load_calibration_file(self, calibration_file):
        """
        Load the photodiode calibration file
        
        Parameters:
        -----------
        calibration_file : str
            Path to the calibration file with two columns: energy (eV) and responsivity (A/W)
        """
        try:
            # Try different approaches to load the calibration file
            try:
                # Try to load as CSV/TXT
                self.calibration_data = pd.read_csv(calibration_file, 
                                                   delim_whitespace=True, 
                                                   header=None,
                                                   names=['energy', 'responsivity'])
            except:
                try:
                    # Try to load as Excel
                    self.calibration_data = pd.read_excel(calibration_file)
                except:
                    # If both fail, create a default calibration dataset
                    print(f"Could not load calibration file {calibration_file}, creating default calibration data")
                    # Create a simple default calibration covering the range of 250-600 eV
                    # Typical responsivity values for silicon photodiodes in this range
                    energies = np.linspace(250, 600, 20)
                    # Responsivities generally decrease with increasing energy in this range
                    # These are approximate values for illustration
                    responsivities = 0.2 - 0.1 * (energies - 250) / 350  # A/W
                    self.calibration_data = pd.DataFrame({
                        'energy': energies,
                        'responsivity': responsivities
                    })
                    return self.calibration_data
            
            # Basic validation
            if 'energy' not in self.calibration_data.columns or 'responsivity' not in self.calibration_data.columns:
                # Try to use the first two columns
                if self.calibration_data.shape[1] >= 2:
                    self.calibration_data = self.calibration_data.iloc[:, 0:2]
                    self.calibration_data.columns = ['energy', 'responsivity']
                else:
                    print(f"Calibration file doesn't have required columns (energy, responsivity)")
                    print("Creating default calibration data instead")
                    # Create default calibration data
                    energies = np.linspace(250, 600, 20)
                    responsivities = 0.2 - 0.1 * (energies - 250) / 350  # A/W
                    self.calibration_data = pd.DataFrame({
                        'energy': energies,
                        'responsivity': responsivities
                    })
            
            print(f"Loaded calibration data with {len(self.calibration_data)} energy points")
            return self.calibration_data
            
        except Exception as e:
            print(f"Error loading calibration file: {str(e)}")
            print("Creating default calibration data instead")
            # Create default calibration as a fallback
            energies = np.linspace(250, 600, 20)
            responsivities = 0.2 - 0.1 * (energies - 250) / 350  # A/W
            self.calibration_data = pd.DataFrame({
                'energy': energies,
                'responsivity': responsivities
            })
            return self.calibration_data
    
    def amp_to_photon_flux(self, current, energy):
        """
        Convert photodiode current (in amps) to photon flux (photons/sec)
        
        Parameters:
        -----------
        current : float or array
            Photodiode current in Amps
        energy : float
            Photon energy in eV
        
        Returns:
        --------
        photon_flux : float or array
            Photon flux in photons/sec
        """
        if self.calibration_data is None:
            print("Warning: No calibration data loaded. Cannot convert to photon flux.")
            return current  # Return original current
        
        # Get responsivity (A/W) at this energy by interpolation
        energy_values = self.calibration_data['energy'].values
        responsivity_values = self.calibration_data['responsivity'].values
        
        # Create interpolation function
        if len(energy_values) > 1:
            resp_interp = interp1d(energy_values, responsivity_values, 
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
            responsivity = resp_interp(energy)
        else:
            # If only one calibration point is available
            responsivity = responsivity_values[0]
        
        # Convert current to power (Watts)
        power = current / responsivity
        
        # Calculate energy per photon (Joules)
        # E = hν, where h = 6.626e-34 J·s and ν = E(eV) * 1.602e-19 J/eV / h
        energy_joules = energy * 1.602e-19  # Convert eV to Joules
        
        # Calculate photon flux (photons/sec)
        # Flux = Power (W) / Energy per photon (J)
        photon_flux = power / energy_joules
        
        return photon_flux
    
    def load_log_file(self, log_file):
        """
        Load and parse the instrument log file to extract metadata
        """
        # Read log file
        with open(log_file, 'r') as f:
            log_content = f.readlines()
        
        # Parse log file into a dataframe
        data = []
        columns = None
        
        for line in log_content:
            if '\t' in line:
                parts = line.strip().split('\t')
                if columns is None:
                    columns = parts
                else:
                    # Convert parts to appropriate data types when possible
                    row = []
                    for part in parts:
                        try:
                            # Try to convert to numeric
                            if '.' in part:
                                row.append(float(part))
                            else:
                                row.append(int(part))
                        except (ValueError, TypeError):
                            row.append(part)
                    
                    # Make sure row has the same length as columns
                    while len(row) < len(columns):
                        row.append(None)
                    
                    data.append(row[:len(columns)])
        
        if columns:
            self.log_data = pd.DataFrame(data, columns=columns)
            return self.log_data
        else:
            # Try another approach - space-separated values
            data = []
            columns = None
            
            for line in log_content:
                parts = line.strip().split()
                if len(parts) > 5:  # Assume it's a data line if it has several fields
                    if columns is None:
                        # Try to identify column headers or create generic ones
                        if any(keyword in line.lower() for keyword in ['filename', 'detector', 'energy', 'angle']):
                            columns = parts
                            continue
                    
                    # If we still don't have column headers, create generic ones
                    if columns is None and len(parts) > 5:
                        columns = [f"col_{i}" for i in range(len(parts))]
                    
                    # Convert parts to appropriate data types
                    row = []
                    for part in parts:
                        try:
                            if '.' in part:
                                row.append(float(part))
                            else:
                                row.append(int(part))
                        except (ValueError, TypeError):
                            row.append(part)
                    
                    # Make sure row has the same length as columns
                    if columns and len(row) > len(columns):
                        row = row[:len(columns)]
                    elif columns and len(row) < len(columns):
                        row.extend([None] * (len(columns) - len(row)))
                    
                    data.append(row)
            
            if columns and data:
                self.log_data = pd.DataFrame(data, columns=columns)
                return self.log_data
        
        # If we get here, we couldn't parse the log file
        print(f"Warning: Could not parse log file {log_file}")
        self.log_data = None
        return None
    
    def load_data_file(self, filename, skiprows=1):
        """
        Load a data file from the instrument
        """
        return np.loadtxt(filename, skiprows=skiprows)
    
    def find_nearest(self, array, value):
        """
        Find the index of the value in array closest to the given value
        
        Parameters:
        -----------
        array : array-like
            Array to search in
        value : float
            Value to find
            
        Returns:
        --------
        idx : int
            Index of closest value
        val : float
            Closest value
        """
        # Check if array is empty
        array = np.asarray(array)
        if len(array) == 0:
            return 0, np.nan
            
        # Find closest value
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]
    
    def get_file_metadata(self, filename):
        """
        Extract metadata for a specific file from the log data
        """
        if self.log_data is None:
            print("Warning: No log data loaded. Cannot extract metadata.")
            return None
        
        # Find the matching row in the log data
        base_filename = os.path.basename(filename)
        matching_data = self.log_data[self.log_data['filename'].str.contains(base_filename, regex=False)]
        
        if matching_data.empty:
            print(f"Warning: No metadata found for file {base_filename}")
            return None
        
        return matching_data.iloc[0]
    
    def get_energy_for_file(self, filename):
        """
        Extract the energy value for a specific file from the log data
        """
        metadata = self.get_file_metadata(filename)
        if metadata is None:
            return None
        
        # Find the energy value in the 'stage' column
        energy = None
        if 'stage' in metadata and not pd.isna(metadata['stage']):
            energy = metadata['stage']
        
        # Convert to float if it's a string
        if isinstance(energy, str):
            try:
                energy = float(energy)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert energy value '{energy}' to float for file {filename}")
                return None
        
        # Check if the energy is in eV or nm by looking at 'mono' column
        # But we always want to work in eV, so we'll convert if necessary
        if 'mono' in metadata and not pd.isna(metadata['mono']):
            mono_unit = metadata['mono']
            if isinstance(mono_unit, str):
                # If mono column contains 'nm', the energy might be in wavelength
                if 'nm' in mono_unit.lower() and energy is not None and energy < 100:
                    energy = 1239.8 / energy  # Convert nm to eV
        
        return energy
    
    def get_angle_for_file(self, filename):
        """
        Extract the angle value for a specific file from the log data
        
        Parameters:
        -----------
        filename : str
            Path to the data file
            
        Returns:
        --------
        angle : float
            The angle value in degrees
        """
        metadata = self.get_file_metadata(filename)
        if metadata is None:
            return None
        
        # Look for the angle in 'm unit' column first (actual angle value)
        angle = None
        if 'm unit' in metadata and not pd.isna(metadata['m unit']):
            angle = metadata['m unit']
        # If not found, try other potential columns
        elif 'tilt' in metadata and not pd.isna(metadata['tilt']):
            angle = metadata['tilt']
        elif 'sample' in metadata and not pd.isna(metadata['sample']):
            angle = metadata['sample']
        
        # Convert to float if it's a string
        if isinstance(angle, str):
            try:
                angle = float(angle)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert angle value '{angle}' to float for file {filename}")
                return None
                
        return angle



    def get_angle_range(self, filename):
        """
        Extract the angle range from a data file
        
        Parameters:
        -----------
        filename : str
            Path to the data file
            
        Returns:
        --------
        min_angle : float
            Minimum angle in the scan
        max_angle : float
            Maximum angle in the scan
        """
        try:
            # Load the data file
            data = self.load_data_file(filename)
            
            # Check if it has at least two columns (angle and intensity)
            if data.shape[1] >= 2:
                # Extract angles from the first column
                angles = data[:, 0]
                
                # Return min and max angles
                return np.min(angles), np.max(angles)
            else:
                print(f"Warning: File {filename} doesn't have enough columns for angle data")
                return None, None
                
        except Exception as e:
            print(f"Error loading angle range from {filename}: {str(e)}")
            return None, None
        
    def get_detector_type(self, filename):
        """
        Extract the detector type for a specific file from the log data
        """
        metadata = self.get_file_metadata(filename)
        if metadata is None:
            return None
        
        detector_type = None
        # Check several possible column names for detector information
        possible_columns = ['main detector name', 'main detector', 'detector', 'detector name']
        
        for col in possible_columns:
            if col in metadata and not pd.isna(metadata[col]):
                detector_type = metadata[col]
                break
        
        return detector_type
    def get_count_time(self, filename):
        """
        Extract the count time for CEM digital detectors from log data
        
        Parameters:
        -----------
        filename : str
            Path to the data file
            
        Returns:
        --------
        count_time : float or None
            The count time value if available, None otherwise
        """
        metadata = self.get_file_metadata(filename)
        if metadata is None:
            return None
        
        # Find count time in 'gain/count1' column
        count_time = None
        if 'gain/count1' in metadata and not pd.isna(metadata['gain/count1']):
            # For CEM digital, this will be the count time
            detector_type = self.get_detector_type(filename)
            if detector_type and 'cem' in str(detector_type).lower():
                count_time = metadata['gain/count1']
        
        # Convert to float if it's a string
        if isinstance(count_time, str):
            try:
                count_time = float(count_time)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert count time value '{count_time}' to float for file {filename}")
                return None
                
        return count_time
   
    
    def get_position(self, filename):
        """
        Extract the x, y, z position values for a specific file from the log data
        """
        metadata = self.get_file_metadata(filename)
        if metadata is None:
            return None, None, None
        
        x = y = z = None
        
        # Use the correct column headers for position
        if 'x unit' in metadata and not pd.isna(metadata['x unit']):
            x = metadata['x unit']
        
        if 'y unit' in metadata and not pd.isna(metadata['y unit']):
            y = metadata['y unit']
            
        if 'z unit' in metadata and not pd.isna(metadata['z unit']):
            z = metadata['z unit']
        
        # Convert to float if they're strings
        for var_name, var_value in [('x', x), ('y', y), ('z', z)]:
            if isinstance(var_value, str):
                try:
                    if var_name == 'x':
                        x = float(var_value)
                    elif var_name == 'y':
                        y = float(var_value)
                    elif var_name == 'z':
                        z = float(var_value)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {var_name} value '{var_value}' to float for file {filename}")
        
        return x, y, z
    
    def reduce_data(self, scans, trims, energy, normalize=True, plot=True, convert_to_photons=False, 
               output_dir=None, plot_prefix=None, smooth_data=False, savgol_window=None, 
               savgol_order=2, remove_zeros=True):
        """
        Reduce multiple scans by scaling them to the lowest angle scan
        
        Parameters:
        -----------
        [parameters remain the same as before]
                
        Returns:
        --------
        If smooth_data is True:
            (smoothed_refl_q, raw_refl_q) : tuple of numpy arrays
                Tuple containing the smoothed and raw reflectivity data
        Else:
            refl_q : numpy array
                Processed reflectivity data with columns [Q, R, error]
        """
        # Import required modules
        from scipy.signal import savgol_filter
        
        # Ensure we have trim values for all scans
        if len(trims) < len(scans):
            print(f"Warning: Not enough trim values provided. Adding default trims for {len(scans) - len(trims)} scans.")
            trims.extend([(0, -1)] * (len(scans) - len(trims)))
            
        # Process the first scan
        refl = deepcopy(scans[0][trims[0][0]:(len(scans[0][:,0])+trims[0][1]), 0:2])
        
        # Remove zeros if requested
        if remove_zeros:
            non_zero_mask = refl[:,1] > 0
            if not all(non_zero_mask):
                print(f"Removing {len(refl) - np.sum(non_zero_mask)} zero data points from first scan")
                refl = refl[non_zero_mask]
        
        # Convert to photon flux if requested
        if convert_to_photons and self.calibration_data is not None:
            # Convert current to photon flux
            refl[:,1] = self.amp_to_photon_flux(refl[:,1], energy)
        
        # Store all raw scans for combined plotting
        all_scans = [deepcopy(refl)]
        
        # Process additional scans
        for i in range(1, len(scans)):
            scan = scans[i][trims[i][0]:(len(scans[i][:,0])+trims[i][1]), 0:2]
            
            # Remove zeros if requested
            if remove_zeros:
                non_zero_mask = scan[:,1] > 0
                if not all(non_zero_mask):
                    print(f"Removing {len(scan) - np.sum(non_zero_mask)} zero data points from scan {i+1}")
                    scan = scan[non_zero_mask]
            
            # Convert to photon flux if requested
            if convert_to_photons and self.calibration_data is not None:
                scan[:,1] = self.amp_to_photon_flux(scan[:,1], energy)
            
            # Find overlap between this scan and the combined reflectivity
            idx, val = self.find_nearest(scan[:,0], refl[-1,0])
            
            # Check for valid overlap - need at least one point
            if idx <= 0 or np.isnan(val):
                print(f"Warning: No overlap found for scan {i+1}. Using default scaling factor of 1.0")
                scale = 1.0
            else:
                # Calculate scaling factor based on overlapping region
                scaling = np.zeros(idx)
                for ii in range(idx):
                    idx2, val2 = self.find_nearest(refl[:,0], scan[ii,0])
                    if scan[ii,1] != 0:  # Avoid division by zero
                        scaling[ii] = refl[idx2,1] / scan[ii,1]
                    else:
                        scaling[ii] = 1.0
                
                # Use mean of scaling factors, ignoring zeros and NaNs
                valid_scaling = scaling[~np.isnan(scaling) & (scaling != 0)]
                if len(valid_scaling) > 0:
                    scale = np.mean(valid_scaling)
                else:
                    scale = 1.0
                    
            print(f"Scaling factor for scan {i+1}: {scale:.4f}")
            
            # Apply scaling
            scan[:,1] = scan[:,1] * scale
            
            # Store the scaled scan for plotting
            all_scans.append(deepcopy(scan))
            
            # Concatenate with existing reflectivity
            refl = np.concatenate((refl, scan))
        
        # Create a copy of the unsmoothed data
        raw_refl = deepcopy(refl)
        
        # Apply smoothing if requested
        if smooth_data:
            # Determine the best window size for Savitzky-Golay filter (must be odd)
            if savgol_window is None:
                window_size = max(min(25, len(refl) // 10 * 2 + 1), 5)
                if window_size % 2 == 0:
                    window_size += 1
            else:
                # Use the user-specified window size
                window_size = savgol_window
                # Ensure it's odd and at least 5
                if window_size % 2 == 0:
                    window_size += 1
                window_size = max(window_size, 5)
            
            print(f"Applying Savitzky-Golay smoothing with window size: {window_size}, order: {savgol_order}")
            try:
                # Apply smoothing while preserving the original angles
                smoothed_intensities = savgol_filter(refl[:,1], window_size, savgol_order)
                # Create smoothed data array
                smoothed_refl = deepcopy(refl)
                smoothed_refl[:,1] = smoothed_intensities
            except Exception as e:
                print(f"Warning: Smoothing failed ({str(e)}), using raw data")
                smoothed_refl = refl
        else:
            smoothed_refl = refl
        
        # Convert raw data to Q (momentum transfer)
        wavelength_nm = 1239.9 / energy  # eV to nm
        wavelength_angstrom = wavelength_nm * 10  # nm to Å
        Q_raw = 4 * np.pi * np.sin(np.radians(raw_refl[:,0])) / wavelength_angstrom  # Q in Å^-1
        
        # Convert smoothed data to Q
        Q = 4 * np.pi * np.sin(np.radians(smoothed_refl[:,0])) / wavelength_angstrom  # Q in Å^-1
        
        # Create output array for raw data
        raw_refl_q = np.zeros([Q_raw.size, 3])
        raw_refl_q[:,0] = Q_raw
        
        # Create output array for smoothed data
        refl_q = np.zeros([Q.size, 3])
        refl_q[:,0] = Q
        
        if normalize:
            # Normalize raw data
            max_intensity_raw = raw_refl[:,1].max()
            raw_refl_q[:,1] = raw_refl[:,1] / max_intensity_raw
            raw_refl_q[:,2] = (raw_refl[:,1] / max_intensity_raw) * 0.01  # 1% error
            
            # Normalize smoothed data
            max_intensity = smoothed_refl[:,1].max()
            refl_q[:,1] = smoothed_refl[:,1] / max_intensity
            refl_q[:,2] = (smoothed_refl[:,1] / max_intensity) * 0.01  # 1% error
        else:
            # Raw data without normalization
            raw_refl_q[:,1] = raw_refl[:,1]
            raw_refl_q[:,2] = raw_refl[:,1] * 0.01  # 1% error
            
            # Smoothed data without normalization
            refl_q[:,1] = smoothed_refl[:,1]
            refl_q[:,2] = smoothed_refl[:,1] * 0.01  # 1% error
        
        # Sort by Q
        raw_refl_q = raw_refl_q[raw_refl_q[:, 0].argsort()]
        refl_q = refl_q[refl_q[:, 0].argsort()]
        
        # Create a consolidated plot with both raw and reduced data
        if plot:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot raw data (in angle space) in the first subplot
            for i, scan_data in enumerate(all_scans):
                marker_style = 'o' if i == 0 else ['s', '^', 'D', 'v', '<', '>', 'p', 'h', '*'][i % 9]
                ax1.plot(scan_data[:,0], scan_data[:,1], marker=marker_style, linestyle='-', 
                        markersize=4, alpha=0.7, label=f'Scan {i+1}')
            
            ax1.set_yscale('log')
            ax1.set_xlabel('Angle (degrees)')
            
            if convert_to_photons:
                ax1.set_ylabel('Photon Flux (photons/sec)')
            else:
                ax1.set_ylabel('Intensity')
                
            ax1.set_title(f'Raw RSoXR Data - {energy:.1f} eV')
            ax1.legend()
            ax1.grid(True, which="both", ls="--", alpha=0.3)
            
            # Plot reduced data (in Q space) in the second subplot
            # Plot raw data points
            ax2.errorbar(raw_refl_q[:,0], raw_refl_q[:,1], yerr=raw_refl_q[:,2], fmt='rx', 
                        markersize=4, capsize=3, alpha=0.5, label='Raw Data')
            
            # If smoothing was applied, overlay the smoothed data
            if smooth_data:
                ax2.errorbar(refl_q[:,0], refl_q[:,1], yerr=refl_q[:,2], fmt='b-', 
                            linewidth=2, label='Smoothed Data')
            else:
                # Connect raw data points with a line if not smoothed
                ax2.plot(raw_refl_q[:,0], raw_refl_q[:,1], 'b-', linewidth=1, alpha=0.7)
            
            ax2.set_yscale('log')
            ax2.set_xlabel('Q (Å$^{-1}$)')
            
            if normalize:
                ax2.set_ylabel('Normalized Reflectivity')
            else:
                ax2.set_ylabel('Reflectivity')
                
            ax2.set_title(f'Reduced RSoXR Data - {energy:.1f} eV')
            ax2.legend()
            ax2.grid(True, which="both", ls="--", alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot if output directory is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                if plot_prefix is None:
                    plot_prefix = f"reflectivity_{energy:.1f}eV"
                plot_filename = f"{plot_prefix}_combined.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_path, dpi=150)
                print(f"Saved combined plot to {plot_path}")
                
            plt.show()
        
            # Save both raw and smoothed data if smoothing was applied
        if output_dir:
            if plot_prefix is None:
                plot_prefix = f"reflectivity_{energy:.1f}eV"
            
            # Always save raw data
            raw_data_filename = f"{plot_prefix}_raw.dat"
            raw_data_path = os.path.join(output_dir, raw_data_filename)
            np.savetxt(raw_data_path, raw_refl_q)
            print(f"Saved raw data to {raw_data_path}")
            
            # Save smoothed data if smoothing was applied
            if smooth_data:
                smoothed_data_filename = f"{plot_prefix}_smoothed.dat"
                smoothed_data_path = os.path.join(output_dir, smoothed_data_filename)
                np.savetxt(smoothed_data_path, refl_q)
                print(f"Saved smoothed data to {smoothed_data_path}")
        
        # Return the processed data (both smoothed and raw if smoothing was applied)
        if smooth_data:
            # Return both raw and smoothed data as a tuple
            return refl_q, raw_refl_q
        else:
            return raw_refl_q  # Only raw data if no smoothing

    
    def process_scan_set(self, scan_group, output_filename=None, normalize=True, plot=True, 
            convert_to_photons=False, output_dir=None, plot_prefix=None, 
            save_metadata=True, estimate_thickness=False, min_prominence=0.1,
            min_thickness_nm=20, max_thickness_nm=100, smooth_data=False,
            savgol_window=None, savgol_order=2, remove_zeros=True):
        """
        Process a scan set and combine them
        
        Parameters:
        -----------
        scan_group : dict
            Scan group dictionary from auto_group_scans
            Expected to contain: 'files', 'trims', 'energy', 'sample_name'
            Optional: 'backgrounds' - list of background values to subtract from each file
        output_filename : str, optional
            File to save the processed data
        normalize : bool
            Whether to normalize the final reflectivity
        plot : bool
            Whether to generate plots during processing
        convert_to_photons : bool
            Whether to convert photodiode current to photon flux
        output_dir : str, optional
            Directory to save the output file (will be created if it doesn't exist)
        plot_prefix : str, optional
            Prefix for plot filenames
        save_metadata : bool
            Whether to save metadata about the reduced data
        estimate_thickness : bool
            Whether to estimate film thickness from fringe spacing
        min_prominence : float
            Minimum prominence for peak/valley detection in thickness estimation
        min_thickness_nm : float
            Minimum expected film thickness in nm
        max_thickness_nm : float
            Maximum expected film thickness in nm
        smooth_data : bool
            Whether to apply smoothing to the data 
        savgol_window : int, optional
            Window size for Savitzky-Golay filter. Must be odd and >= 5.
        savgol_order : int
            Polynomial order for the Savitzky-Golay filter. Default is 2.
        remove_zeros : bool
            Whether to remove data points with zero intensity
        
        Returns:
        --------
        If smooth_data is True:
            (smoothed_data, raw_data) : tuple of numpy arrays
                Tuple containing the smoothed and raw reflectivity data
        Else:
            result : numpy array
                Processed reflectivity data
        """
        # Extract information from the scan group
        file_patterns = scan_group['files']
        trims = scan_group['trims']
        energy = scan_group['energy']
        
        # Extract background values if provided (NEW)
        backgrounds = scan_group.get('backgrounds', [0.0] * len(file_patterns))
        
        # Ensure backgrounds list has the same length as files
        if len(backgrounds) < len(file_patterns):
            print(f"Warning: Not enough background values provided. Padding with zeros.")
            backgrounds.extend([0.0] * (len(file_patterns) - len(backgrounds)))
        elif len(backgrounds) > len(file_patterns):
            print(f"Warning: More background values than files. Truncating background list.")
            backgrounds = backgrounds[:len(file_patterns)]
        
        scans = []
        energies = []
        actual_trims = []
        actual_backgrounds = []
        
        # Load all data files
        for i, pattern in enumerate(file_patterns):
            try:
                scan_data = self.load_data_file(pattern)
                scans.append(scan_data)
                
                # Use the trim value for this pattern
                if i < len(trims):
                    actual_trims.append(trims[i])
                else:
                    # Default trim if not provided
                    actual_trims.append((0, -1))
                
                # Use the background value for this pattern (NEW)
                actual_backgrounds.append(backgrounds[i])
                
                energies.append(energy)
                print(f"Loaded file: {pattern}")
                if backgrounds[i] != 0.0:
                    print(f"  Background subtraction: -{backgrounds[i]:.3f}")
            except Exception as e:
                print(f"Error loading file {pattern}: {str(e)}")
        
        if not scans:
            print("No valid scan data found.")
            return None
            
        # Set default plot prefix if not provided
        if output_filename and plot_prefix is None:
            # Use the output filename without extension as plot prefix
            plot_prefix = os.path.splitext(os.path.basename(output_filename))[0]
        elif plot_prefix is None:
            # Default prefix based on energy
            plot_prefix = f"{scan_group['sample_name']}_{energy:.1f}eV"
        
        # Validate smoothing parameters if smoothing is enabled
        if smooth_data:
            # Ensure window size is odd and >= 5
            if savgol_window is None:
                # Auto-determine window size
                window_size = max(min(25, len(scans[0]) // 10 * 2 + 1), 5)
                if window_size % 2 == 0:
                    window_size += 1
            else:
                window_size = savgol_window
                if window_size % 2 == 0:
                    window_size += 1
                window_size = max(window_size, 5)
            
            # Ensure polynomial order is valid
            polynomial_order = min(savgol_order, window_size - 1)
            polynomial_order = max(polynomial_order, 1)
            
            print(f"Using smoothing with window size={window_size}, polynomial order={polynomial_order}")
        else:
            window_size, polynomial_order = None, 2
        
        # Process the scans with background subtraction (MODIFIED)
        result = self.reduce_data_with_background(
            scans, 
            actual_trims, 
            actual_backgrounds,  # NEW: Pass background values
            energy, 
            normalize=normalize, 
            plot=plot,
            convert_to_photons=convert_to_photons,
            output_dir=None,  # Initially set to None to control saving ourselves
            plot_prefix=plot_prefix,
            smooth_data=smooth_data,
            savgol_window=window_size,
            savgol_order=polynomial_order,
            remove_zeros=remove_zeros
        )
        
        if result is None:
            print("Data processing failed.")
            return None
        
        # Save the data if output filename is provided
        if output_filename is not None:
            # Determine which data to save
            data_to_save = result[0] if smooth_data else result
            
            # Create output directory if it doesn't exist
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)
            else:
                output_path = output_filename
            
            # Save the processed data
            header = f"# Processed RSoXR data\n# Sample: {scan_group['sample_name']}\n# Energy: {energy:.1f} eV\n# Columns: Angle(deg), Reflectivity"
            
            # Add background information to header (NEW)
            bg_info = [f"File {i+1}: -{bg:.3f}" for i, bg in enumerate(actual_backgrounds) if bg != 0.0]
            if bg_info:
                header += f"\n# Background subtractions: {', '.join(bg_info)}"
            
            np.savetxt(output_path, data_to_save, delimiter='\t', 
                    header=header, fmt='%.6e')
            
            print(f"Processed data saved to {output_path}")
            
            # Save metadata if requested
            if save_metadata:
                log_file_base = "processed_data"
                if hasattr(self, 'log_file') and self.log_file:
                    log_file_base = os.path.splitext(os.path.basename(self.log_file))[0]
                    
                sample_name = scan_group['sample_name']
                energy = scan_group['energy']
                meta_filename = f"{log_file_base}_{sample_name}_{energy:.1f}eV_metadata.csv"
                
                if output_dir:
                    meta_path = os.path.join(output_dir, meta_filename)
                else:
                    meta_path = meta_filename
                    
                # Create metadata dictionary (ENHANCED with background info)
                metadata = {
                    'Sample': sample_name,
                    'Energy (eV)': energy,
                    'Files Processed': len(scans),
                    'Normalization': normalize,
                    'Photon Conversion': convert_to_photons,
                    'Smoothing': smooth_data,
                    'Zero Removal': remove_zeros,
                    'Processing Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Output File': os.path.basename(output_path),
                    'Background Subtraction Applied': any(bg != 0.0 for bg in actual_backgrounds)
                }
                
                # Add smoothing parameters if used
                if smooth_data:
                    metadata['Smoothing Window'] = window_size
                    metadata['Smoothing Order'] = polynomial_order
                
                # Add background details (NEW)
                for i, bg in enumerate(actual_backgrounds):
                    metadata[f'Background File {i+1}'] = bg
                
                # Save metadata
                try:
                    import pandas as pd
                    meta_df = pd.DataFrame([metadata])
                    meta_df.to_csv(meta_path, index=False)
                    print(f"Metadata saved to {meta_path}")
                except Exception as e:
                    print(f"Warning: Could not save metadata: {str(e)}")
        
        # Estimate thickness if requested
        if estimate_thickness and result is not None:
            try:
                thickness = self.estimate_thickness(result[0] if smooth_data else result,
                                                min_prominence=min_prominence,
                                                min_thickness_nm=min_thickness_nm,
                                                max_thickness_nm=max_thickness_nm)
                if thickness is not None:
                    print(f"Estimated thickness: {thickness:.1f} Å")
                    
                    # Update metadata with thickness if file was saved
                    if save_metadata and output_filename is not None:
                        # Update the metadata file with thickness information
                        log_file_base = "processed_data"
                        if hasattr(self, 'log_file') and self.log_file:
                            log_file_base = os.path.splitext(os.path.basename(self.log_file))[0]
                            
                        sample_name = scan_group['sample_name']
                        energy = scan_group['energy']
                        meta_filename = f"{log_file_base}_{sample_name}_{energy:.1f}eV_metadata.csv"
                        
                        if output_dir:
                            meta_path = os.path.join(output_dir, meta_filename)
                        else:
                            meta_path = meta_filename
                            
                        # Check if metadata file exists and update it
                        if os.path.exists(meta_path):
                            try:
                                import pandas as pd
                                meta_df = pd.read_csv(meta_path)
                                if 'Thickness (Å)' not in meta_df.columns:
                                    meta_df['Thickness (Å)'] = thickness
                                else:
                                    meta_df.loc[0, 'Thickness (Å)'] = thickness
                                meta_df.to_csv(meta_path, index=False)
                                print(f"Updated metadata with thickness information")
                            except Exception as e:
                                print(f"Warning: Could not update metadata file with thickness: {str(e)}")
            except Exception as e:
                print(f"Warning: Thickness estimation failed: {str(e)}")
        
        return result
    
    
    def reduce_data_with_background(self, scans, trims, backgrounds, energy, normalize=True, plot=True, 
                               convert_to_photons=False, output_dir=None, plot_prefix=None,
                               smooth_data=False, savgol_window=None, savgol_order=2, 
                               remove_zeros=True):
        """
        Process and combine multiple scan files with background subtraction
        
        This is an enhanced version of reduce_data that includes background subtraction
        
        Parameters:
        -----------
        scans : list of numpy arrays
            List of loaded scan data (angle, intensity)
        trims : list of tuples
            List of (start, end) trim indices for each scan
        backgrounds : list of floats
            List of background values to subtract from each scan
        energy : float
            Photon energy in eV
        normalize : bool
            Whether to normalize the final reflectivity
        plot : bool
            Whether to generate plots during processing
        convert_to_photons : bool
            Whether to convert photodiode current to photon flux
        output_dir : str, optional
            Directory to save plots
        plot_prefix : str, optional
            Prefix for plot filenames
        smooth_data : bool
            Whether to apply Savitzky-Golay smoothing
        savgol_window : int, optional
            Window size for smoothing filter
        savgol_order : int
            Polynomial order for smoothing filter
        remove_zeros : bool
            Whether to remove zero intensity points
        
        Returns:
        --------
        If smooth_data is True:
            (smoothed_data, raw_data) : tuple of numpy arrays
        Else:
            processed_data : numpy array
        """
        if len(scans) != len(trims) or len(scans) != len(backgrounds):
            print(f"Error: Mismatch in number of scans ({len(scans)}), trims ({len(trims)}), and backgrounds ({len(backgrounds)})")
            return None
        
        # Ensure we have trim values for all scans
        if len(trims) < len(scans):
            print(f"Warning: Not enough trim values provided. Adding default trims for {len(scans) - len(trims)} scans.")
            trims.extend([(0, -1)] * (len(scans) - len(trims)))
        
        # Process the first scan
        refl = deepcopy(scans[0][trims[0][0]:(len(scans[0][:,0])+trims[0][1]), 0:2])
        
        # Apply background subtraction to first scan (NEW)
        if backgrounds[0] != 0.0:
            refl[:, 1] = refl[:, 1] - backgrounds[0]
            print(f"Applied background subtraction to scan 1: -{backgrounds[0]:.3f}")
        
        # Remove zeros if requested
        if remove_zeros:
            non_zero_mask = refl[:,1] > 0
            if not all(non_zero_mask):
                print(f"Removing {len(refl) - np.sum(non_zero_mask)} zero data points from first scan")
                refl = refl[non_zero_mask]
        
        # Convert to photon flux if requested
        if convert_to_photons and self.calibration_data is not None:
            # Convert current to photon flux
            refl[:,1] = self.amp_to_photon_flux(refl[:,1], energy)
        
        # Store all raw scans for combined plotting
        all_scans = [deepcopy(refl)]
        
        # Process additional scans
        for i in range(1, len(scans)):
            scan = scans[i][trims[i][0]:(len(scans[i][:,0])+trims[i][1]), 0:2]
            
            # Apply background subtraction (NEW)
            if backgrounds[i] != 0.0:
                scan[:, 1] = scan[:, 1] - backgrounds[i]
                print(f"Applied background subtraction to scan {i+1}: -{backgrounds[i]:.3f}")
            
            # Remove zeros if requested
            if remove_zeros:
                non_zero_mask = scan[:,1] > 0
                if not all(non_zero_mask):
                    print(f"Removing {len(scan) - np.sum(non_zero_mask)} zero data points from scan {i+1}")
                    scan = scan[non_zero_mask]
            
            # Convert to photon flux if requested
            if convert_to_photons and self.calibration_data is not None:
                scan[:,1] = self.amp_to_photon_flux(scan[:,1], energy)
            
            # Find overlap between this scan and the combined reflectivity
            idx, val = self.find_nearest(scan[:,0], refl[-1,0])
            
            # Check for valid overlap - need at least one point
            if idx <= 0 or np.isnan(val):
                print(f"Warning: No overlap found for scan {i+1}. Appending without scaling.")
                scale = 1.0
            else:
                # Calculate scaling factor using overlap region
                overlap_angles = scan[:idx, 0]
                overlap_intensities = scan[:idx, 1]
                
                # Interpolate the combined data onto the overlap angles
                combined_interp = np.interp(overlap_angles, refl[:, 0], refl[:, 1])
                
                # Calculate scaling factor
                if len(overlap_intensities) > 0 and np.mean(overlap_intensities) > 0:
                    scale = np.mean(combined_interp) / np.mean(overlap_intensities)
                else:
                    scale = 1.0
            
            print(f"Scaling factor for scan {i+1}: {scale:.4f}")
            
            # Apply scaling
            scan[:, 1] = scan[:, 1] * scale
            
            # Store the raw scan for plotting
            all_scans.append(deepcopy(scan))
            
            # Find where to start adding new data (avoid duplication in overlap)
            if len(refl) > 0:
                last_angle = refl[-1, 0]
                start_idx = np.searchsorted(scan[:, 0], last_angle)
                new_data = scan[start_idx:, :]
            else:
                new_data = scan
            
            # Concatenate new data
            if len(new_data) > 0:
                refl = np.concatenate((refl, new_data))
        
        # Sort by angle (just in case)
        sort_indices = np.argsort(refl[:, 0])
        refl = refl[sort_indices]
        
        # Apply smoothing if requested
        if smooth_data and savgol_window is not None:
            from scipy.signal import savgol_filter
            try:
                # Store raw data before smoothing
                raw_refl_q = deepcopy(refl)
                
                # Apply smoothing
                smoothed_intensity = savgol_filter(refl[:, 1], savgol_window, savgol_order)
                refl_q = deepcopy(refl)
                refl_q[:, 1] = smoothed_intensity
                
                print(f"Applied Savitzky-Golay smoothing (window={savgol_window}, order={savgol_order})")
            except Exception as e:
                print(f"Warning: Smoothing failed: {str(e)}. Returning unsmoothed data.")
                raw_refl_q = deepcopy(refl)
                refl_q = deepcopy(refl)
        else:
            raw_refl_q = deepcopy(refl)
            refl_q = deepcopy(refl)
        
        # Normalize if requested
        if normalize:
            if smooth_data:
                refl[:, 1] = self.apply_normalization(refl[:, 1], energy)
                raw_refl_q[:, 1] = self.apply_normalization(raw_refl_q[:, 1], energy)
            else:
                refl[:, 1] = self.apply_normalization(refl[:, 1], energy)
            print("Applied Energy normalization")
        
        # Generate plots if requested
        if plot and plot_prefix:
            self._plot_processed_data_with_background(all_scans, refl_q if smooth_data else refl, 
                                                    backgrounds, energy, plot_prefix, output_dir)
        
        # Return processed data (smoothed and raw if smoothing was applied)
        if smooth_data:
            # Return both raw and smoothed data as a tuple
            return refl_q, raw_refl_q
        else:
            return raw_refl_q  # Only raw data if no smoothing


    def _plot_processed_data_with_background(self, all_scans, final_data, backgrounds, energy, 
                                        plot_prefix, output_dir=None):
        """
        Generate plots showing the effect of background subtraction and data processing
        
        Parameters:
        -----------
        all_scans : list of numpy arrays
            Individual processed scans (after background subtraction and scaling)
        final_data : numpy array
            Final combined and processed data
        backgrounds : list of floats
            Background values that were subtracted
        energy : float
            Photon energy in eV
        plot_prefix : str
            Prefix for saved plot files
        output_dir : str, optional
            Directory to save plots
        """
        import matplotlib.pyplot as plt
        
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Individual scans (after background subtraction)
            for i, (scan, bg) in enumerate(zip(all_scans, backgrounds)):
                label = f'Scan {i+1}'
                if bg != 0.0:
                    label += f' (BG: -{bg:.3f})'
                ax1.plot(scan[:, 0], scan[:, 1], 'o-', alpha=0.7, markersize=3, label=label)
            
            ax1.set_xlabel('Angle (degrees)')
            ax1.set_ylabel('Intensity')
            ax1.set_title(f'Individual Scans (Background Corrected) - {energy:.1f} eV')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Plot 2: Final combined data
            ax2.plot(final_data[:, 0], final_data[:, 1], 'b-', linewidth=2, label='Combined Data')
            ax2.set_xlabel('Angle (degrees)')
            ax2.set_ylabel('Intensity')
            ax2.set_title(f'Final Combined Data - {energy:.1f} eV')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            plt.tight_layout()
            
            # Save plot if output directory is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plot_filename = f"{plot_prefix}_background_corrected.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {str(e)}")
    
    
    
    def batch_process(self, scan_sets, trims_sets, output_template, normalize=True, plot=True, 
                    convert_to_photons=False, output_dir=None):
        """
        Process multiple sets of scans in batch mode
        
        Parameters:
        -----------
        scan_sets : list of lists of str
            List of lists of file patterns for each scan set
        trims_sets : list of lists of tuples
            List of lists of trim settings for each scan set
        output_template : str
            Template for output filenames, e.g. "output_{energy:.1f}eV.dat"
        normalize : bool
            Whether to normalize the reflectivity
        plot : bool
            Whether to generate plots during processing
        convert_to_photons : bool
            Whether to convert photodiode current to photon flux
        output_dir : str, optional
            Directory to save output files
        
        Returns:
        --------
        List of processed reflectivity data arrays
        """
        results = []
        
        for i, (scan_set, trims) in enumerate(zip(scan_sets, trims_sets)):
            print(f"\nProcessing scan set {i+1}/{len(scan_sets)}")
            
            # Load the first file to get the energy
            first_file = None
            for pattern in scan_set:
                matching_files = sorted(glob.glob(pattern))
                if matching_files:
                    first_file = matching_files[0]
                    break
            
            energy = None
            if first_file and self.log_data is not None:
                energy = self.get_energy_for_file(first_file)
            
            if energy is None:
                energy = 0.0
                output_filename = output_template.format(index=i+1, energy="unknown")
            else:
                output_filename = output_template.format(index=i+1, energy=energy)
            
            # Process this scan set
            result = self.process_scan_set(
                scan_set, 
                trims, 
                output_filename=output_filename,
                normalize=normalize,
                plot=plot,
                convert_to_photons=convert_to_photons,
                output_dir=output_dir,
                plot_prefix=os.path.splitext(output_filename)[0]
            )
            
            results.append(result)
        
        return results
    
    def auto_group_scans(self, data_directory=".", position_tolerance=0.4, 
                        energy_tolerance=0.2, auto_trim=False, save_table=True, 
                        output_dir=None, sort_by_energy=True):
        """
        Enhanced auto-grouping that assigns unique scan numbers to each file
        
        Parameters:
        -----------
        data_directory : str
            Directory containing the data files
        position_tolerance : float
            Maximum allowed difference in X and Y positions for scans to be grouped
        energy_tolerance : float
            Maximum allowed difference in energies for scans to be grouped
        auto_trim : bool
            Whether to automatically determine trim values by analyzing the data quality
        save_table : bool
            Whether to save the scan group information to a table file
        output_dir : str, optional
            Directory to save the output table (will be created if it doesn't exist)
        sort_by_energy : bool
            Whether to sort the groups by energy
            
        Returns:
        --------
        groups : list of dicts
            List of file groups with associated metadata and scan numbers
        """
        # Reset scan registry and numbering
        self.scan_registry = {}
        self.next_scan_number = 1
        
        # Call existing auto_group_scans logic (simplified here)
        groups = self._perform_auto_grouping(data_directory, position_tolerance, 
                                           energy_tolerance, auto_trim, sort_by_energy)
        
        # Assign unique scan numbers to each file across all groups
        for group_idx, group in enumerate(groups):
            group['scan_numbers'] = []
            for file_idx, filename in enumerate(group['files']):
                scan_number = self.next_scan_number
                self.next_scan_number += 1
                
                # Store in registry
                self.scan_registry[scan_number] = {
                    'filename': filename,
                    'group_idx': group_idx,
                    'file_idx': file_idx,
                    'metadata': group['metadata'][file_idx] if file_idx < len(group['metadata']) else None,
                    'trim': group['trims'][file_idx] if file_idx < len(group['trims']) else (0, -1)
                }
                
                group['scan_numbers'].append(scan_number)
        
        if save_table and groups:
            self.save_scan_groups_to_table(groups, output_dir=output_dir)
            
        return groups
    
    def edit_groups(self, scan_groups=None, group=None, add_scan_number=None, 
                   remove_scan_number=None, remove_group=None):
        """
        Edit scan groups by moving individual scans between groups
        
        Parameters:
        -----------
        scan_groups : list of dicts, optional
            Current scan groups. If None, uses self.scan_groups
        group : int, optional
            Target group number (1-based) for adding scans
        add_scan_number : int or list of int, optional
            Scan number(s) to add to the target group
        remove_scan_number : int or list of int, optional
            Scan number(s) to remove from the specified group
        remove_group : int, optional
            Group number to remove entirely
            
        Returns:
        --------
        updated_groups : list of dicts
            Updated scan groups
            
        Examples:
        --------
        # Add scan 4 to group 1 (removes it from its current group)
        processor.edit_groups(group=1, add_scan_number=4)
        
        # Add multiple scans to group 1
        processor.edit_groups(group=1, add_scan_number=[4, 5])
        
        # Remove scan 6 from group 1
        processor.edit_groups(group=1, remove_scan_number=6)
        
        # Remove entire group 4
        processor.edit_groups(remove_group=4)
        """
        if scan_groups is None:
            if not hasattr(self, 'scan_groups') or self.scan_groups is None:
                raise ValueError("No scan groups available. Run auto_group_scans first.")
            scan_groups = deepcopy(self.scan_groups)
        else:
            scan_groups = deepcopy(scan_groups)
        
        # Handle removing entire group
        if remove_group is not None:
            return self._remove_group(scan_groups, remove_group)
        
        # Handle removing scans from a specific group
        if remove_scan_number is not None and group is not None:
            return self._remove_scans_from_group(scan_groups, group, remove_scan_number)
        
        # Handle adding scans to a group
        if add_scan_number is not None and group is not None:
            return self._add_scans_to_group(scan_groups, group, add_scan_number)
        
        raise ValueError("Invalid combination of parameters. See docstring for examples.")
    
    def _add_scans_to_group(self, scan_groups, target_group, scan_numbers):
        """Add one or more scans to a target group"""
        if not isinstance(scan_numbers, list):
            scan_numbers = [scan_numbers]
        
        target_group_idx = target_group - 1
        if not (0 <= target_group_idx < len(scan_groups)):
            raise ValueError(f"Invalid group number {target_group}. Valid range is 1-{len(scan_groups)}")
        
        moved_scans = []
        
        for scan_number in scan_numbers:
            if scan_number not in self.scan_registry:
                print(f"Warning: Scan number {scan_number} not found in registry")
                continue
            
            scan_info = self.scan_registry[scan_number]
            source_group_idx = scan_info['group_idx']
            source_file_idx = scan_info['file_idx']
            
            if source_group_idx == target_group_idx:
                print(f"Scan {scan_number} is already in group {target_group}")
                continue
            
            # Remove from source group
            source_group = scan_groups[source_group_idx]
            
            # Find the current position of this scan in the source group
            current_idx = None
            for idx, snum in enumerate(source_group['scan_numbers']):
                if snum == scan_number:
                    current_idx = idx
                    break
            
            if current_idx is None:
                print(f"Warning: Scan {scan_number} not found in expected source group")
                continue
            
            # Remove from source group
            filename = source_group['files'].pop(current_idx)
            metadata = source_group['metadata'].pop(current_idx) if current_idx < len(source_group['metadata']) else None
            trim = source_group['trims'].pop(current_idx) if current_idx < len(source_group['trims']) else (0, -1)
            source_group['scan_numbers'].pop(current_idx)
            
            # Add to target group
            target_group_dict = scan_groups[target_group_idx]
            target_group_dict['files'].append(filename)
            target_group_dict['metadata'].append(metadata)
            target_group_dict['trims'].append(trim)
            target_group_dict['scan_numbers'].append(scan_number)
            
            # Update registry
            self.scan_registry[scan_number]['group_idx'] = target_group_idx
            self.scan_registry[scan_number]['file_idx'] = len(target_group_dict['files']) - 1
            
            moved_scans.append(scan_number)
            
        # Update file indices for remaining scans in source groups
        self._update_scan_indices(scan_groups)
        
        # Remove empty groups
        scan_groups = self._remove_empty_groups(scan_groups)
        
        # Sort files within groups
        for group in scan_groups:
            if hasattr(self, '_sort_files_in_group'):
                group = self._sort_files_in_group(group)
        
        if moved_scans:
            print(f"Moved scans {moved_scans} to group {target_group}")
        
        return scan_groups
    
    def _remove_scans_from_group(self, scan_groups, group_number, scan_numbers):
        """Remove one or more scans from a specific group"""
        if not isinstance(scan_numbers, list):
            scan_numbers = [scan_numbers]
        
        group_idx = group_number - 1
        if not (0 <= group_idx < len(scan_groups)):
            raise ValueError(f"Invalid group number {group_number}. Valid range is 1-{len(scan_groups)}")
        
        removed_scans = []
        group = scan_groups[group_idx]
        
        # Sort scan numbers in reverse order to maintain indices during removal
        scan_numbers_in_group = [(scan_num, idx) for idx, scan_num in enumerate(group['scan_numbers']) 
                                if scan_num in scan_numbers]
        scan_numbers_in_group.sort(key=lambda x: x[1], reverse=True)
        
        for scan_number, file_idx in scan_numbers_in_group:
            # Remove from group
            filename = group['files'].pop(file_idx)
            group['metadata'].pop(file_idx) if file_idx < len(group['metadata']) else None
            group['trims'].pop(file_idx) if file_idx < len(group['trims']) else None
            group['scan_numbers'].pop(file_idx)
            
            # Remove from registry
            if scan_number in self.scan_registry:
                del self.scan_registry[scan_number]
            
            removed_scans.append(scan_number)
            print(f"Removed scan {scan_number} ({os.path.basename(filename)}) from group {group_number}")
        
        # Update indices
        self._update_scan_indices(scan_groups)
        
        # Remove empty groups
        scan_groups = self._remove_empty_groups(scan_groups)
        
        return scan_groups
    
    def _remove_group(self, scan_groups, group_number):
        """Remove an entire group"""
        group_idx = group_number - 1
        if not (0 <= group_idx < len(scan_groups)):
            raise ValueError(f"Invalid group number {group_number}. Valid range is 1-{len(scan_groups)}")
        
        # Remove scans from registry
        group = scan_groups[group_idx]
        for scan_number in group['scan_numbers']:
            if scan_number in self.scan_registry:
                del self.scan_registry[scan_number]
        
        # Remove group
        removed_group = scan_groups.pop(group_idx)
        print(f"Removed group {group_number} ({removed_group['sample_name']}) with {len(removed_group['files'])} scans")
        
        # Update indices for remaining groups
        self._update_scan_indices(scan_groups)
        
        return scan_groups
    
    def _update_scan_indices(self, scan_groups):
        """Update the scan registry with current group and file indices"""
        for group_idx, group in enumerate(scan_groups):
            for file_idx, scan_number in enumerate(group['scan_numbers']):
                if scan_number in self.scan_registry:
                    self.scan_registry[scan_number]['group_idx'] = group_idx
                    self.scan_registry[scan_number]['file_idx'] = file_idx
    
    def _remove_empty_groups(self, scan_groups):
        """Remove groups that have no files"""
        return [group for group in scan_groups if len(group['files']) > 0]
    
    def print_scan_registry(self):
        """Print the current scan registry for debugging"""
        print("\nScan Registry:")
        print("=" * 80)
        print(f"{'Scan#':6} | {'Group':6} | {'File#':6} | {'Filename':30} | {'Detector':12}")
        print("-" * 80)
        
        for scan_num in sorted(self.scan_registry.keys()):
            info = self.scan_registry[scan_num]
            filename = os.path.basename(info['filename'])
            detector = info['metadata']['detector'] if info['metadata'] else 'Unknown'
            print(f"{scan_num:6} | {info['group_idx']+1:6} | {info['file_idx']+1:6} | {filename:30} | {detector:12}")
    
    def print_groups_with_scan_numbers(self, scan_groups, show_details=False):
        """Print group summary showing scan numbers for each file"""
        print(f"\nScan Groups Summary ({len(scan_groups)} groups):")
        print("=" * 100)
        
        for i, group in enumerate(scan_groups):
            print(f"\nGroup {i+1}: {group['sample_name']} ({group['energy']:.1f} eV)")
            print(f"Position: ({group['x']:.2f}, {group['y']:.2f})")
            print(f"Files: {len(group['files'])}")
            
            if show_details:
                print(f"{'Scan#':6} | {'Filename':25} | {'Detector':12} | {'Angle Range':15}")
                print("-" * 65)
                
                for j, (filename, scan_num) in enumerate(zip(group['files'], group['scan_numbers'])):
                    base_filename = os.path.basename(filename)
                    file_meta = group['metadata'][j] if j < len(group['metadata']) else None
                    
                    if file_meta:
                        detector = file_meta['detector'] if file_meta['detector'] else "Unknown"
                        angle_range_str = "N/A"
                        if file_meta.get('min_angle') is not None and file_meta.get('max_angle') is not None:
                            angle_range_str = f"{file_meta['min_angle']:.2f}-{file_meta['max_angle']:.2f}°"
                    else:
                        detector = "Unknown"
                        angle_range_str = "N/A"
                    
                    print(f"{scan_num:6} | {base_filename:25} | {detector:12} | {angle_range_str:15}")
            else:
                scan_nums_str = ", ".join(map(str, group['scan_numbers']))
                print(f"Scan numbers: {scan_nums_str}")
            
            print()
    
    def get_scan_info(self, scan_number):
        """Get detailed information about a specific scan"""
        if scan_number not in self.scan_registry:
            print(f"Scan number {scan_number} not found")
            return None
        
        return self.scan_registry[scan_number]
    
    def find_scans_by_detector(self, detector_type):
        """Find all scans with a specific detector type"""
        matching_scans = []
        for scan_num, info in self.scan_registry.items():
            if info['metadata'] and info['metadata'].get('detector', '').lower() == detector_type.lower():
                matching_scans.append(scan_num)
        return matching_scans
    
    def find_scans_by_filename_pattern(self, pattern):
        """Find all scans whose filenames match a pattern"""
        import re
        matching_scans = []
        for scan_num, info in self.scan_registry.items():
            filename = os.path.basename(info['filename'])
            if re.search(pattern, filename, re.IGNORECASE):
                matching_scans.append(scan_num)
        return matching_scans

    def _perform_auto_grouping(self, data_directory, position_tolerance, 
                              energy_tolerance, auto_trim, sort_by_energy):
        """
        Perform the actual auto-grouping logic based on position, energy, and detector type
        This contains the core implementation from the original auto_group_scans method
        """
        import os
        import glob
        import re
        import pandas as pd
        from collections import defaultdict
        from copy import deepcopy
        
        if self.log_data is None:
            print("Error: Log data is required for automatic scan grouping.")
            return []
            
        # Find all data files with 6 digits before the extension
        data_files = []
        for file in glob.glob(os.path.join(data_directory, "*.dat")) + glob.glob(os.path.join(data_directory, "*.txt")):
            base_name = os.path.basename(file)
            # Check if the filename has 6 digits right before the extension
            if len(base_name) > 10:  # At least needs room for 6 digits + .dat/.txt
                file_ext = os.path.splitext(base_name)[1]  # Get extension (.dat or .txt)
                last_six_before_ext = base_name[-10:-4]  # Get 6 chars before extension
                if last_six_before_ext.isdigit():
                    data_files.append(file)
        
        if not data_files:
            print(f"Warning: No raw data files found in {data_directory}")
            return []
            
        print(f"Found {len(data_files)} potential raw data files")
        
        # Extract metadata for each file
        file_metadata = []
        for filename in data_files:
            # Skip any log files
            if filename.endswith('.log'):
                continue
                
            # Only process files that are in the log file
            base_name = os.path.basename(filename)
            if not any(self.log_data['filename'].str.contains(base_name, regex=False)):
                print(f"Skipping {base_name} - not found in log file")
                continue
                
            # Get metadata
            metadata = self.get_file_metadata(filename)
            if metadata is None:
                print(f"No metadata found for {filename}")
                continue
                
            energy = self.get_energy_for_file(filename)
            angle = self.get_angle_for_file(filename)
            detector = self.get_detector_type(filename)
            count_time = self.get_count_time(filename)
            x, y, z = self.get_position(filename)
            
            # Get angle range from the data file
            min_angle, max_angle = self.get_angle_range(filename)
            
            if energy is not None and detector is not None and x is not None and y is not None:
                file_metadata.append({
                    'filename': filename,
                    'energy': energy,
                    'angle': angle,
                    'detector': detector,
                    'count_time': count_time,
                    'min_angle': min_angle,
                    'max_angle': max_angle,
                    'x': x,
                    'y': y,
                    'z': z
                })
            else:
                print(f"Missing required metadata for {filename}")
        
        if not file_metadata:
            print("Warning: Could not extract metadata for any files.")
            return []
            
        # Group by position and energy
        position_groups = defaultdict(list)
        
        for meta in file_metadata:
            # Create a position key with rounded values
            x_rounded = round(meta['x'] / position_tolerance) * position_tolerance
            y_rounded = round(meta['y'] / position_tolerance) * position_tolerance
            e_rounded = round(meta['energy'] / energy_tolerance) * energy_tolerance
            
            position_key = f"{x_rounded:.2f}_{y_rounded:.2f}_{e_rounded:.1f}"
            position_groups[position_key].append(meta)
        
        # Prepare groups of files with detector information
        scan_groups = []
        
        for pos_key, files in position_groups.items():
            # Sort files by detector type and angle
            photodiode_files = []
            cem_files = []
            
            for f in files:
                detector_str = str(f['detector']).lower() if f['detector'] is not None else ""
                if "photodiode" in detector_str:
                    photodiode_files.append(f)
                elif "cem" in detector_str:
                    cem_files.append(f)
                else:
                    print(f"Unknown detector type: {f['detector']} for file {os.path.basename(f['filename'])}")
            
            # Sort by angle
            if photodiode_files:
                photodiode_files.sort(key=lambda x: x['angle'] if x['angle'] is not None else 0)
            
            if cem_files:
                cem_files.sort(key=lambda x: x['angle'] if x['angle'] is not None else 0)
            
            # Only include groups with at least one detector type
            if not photodiode_files and not cem_files:
                continue
                
            # Get the energy for this group
            group_energy = files[0]['energy']
            
            # Combine the files in the correct order
            group_files = []
            group_metadata = []
            
            # Start with photodiode (lowest angle) files if available
            if photodiode_files:
                group_files.extend([f['filename'] for f in photodiode_files])
                group_metadata.extend(photodiode_files)
            
            # Add cem files (higher angles) if available
            if cem_files:
                group_files.extend([f['filename'] for f in cem_files])
                group_metadata.extend(cem_files)
            
            # Default trim to use all data points
            if auto_trim:
                # Automatically determine trim values by analyzing data
                trims = self._determine_trims(group_files)
            else:
                trims = [(0, -1)] * len(group_files)
            
            # Extract the sample name - most basic approach
            base_name = os.path.basename(group_files[0])
            sample_match = re.match(r'([^_\d]+)', base_name)
            if sample_match:
                sample_name = sample_match.group(1)
            else:
                sample_name = f"sample_{len(scan_groups)+1}"
                
            # Calculate overall angle range for the group
            overall_min_angle = float('inf')
            overall_max_angle = float('-inf')
            
            for meta in group_metadata:
                if meta['min_angle'] is not None and meta['min_angle'] < overall_min_angle:
                    overall_min_angle = meta['min_angle']
                if meta['max_angle'] is not None and meta['max_angle'] > overall_max_angle:
                    overall_max_angle = meta['max_angle']
                    
            # Handle case where no valid angles were found
            if overall_min_angle == float('inf'):
                overall_min_angle = None
            if overall_max_angle == float('-inf'):
                overall_max_angle = None
                
            scan_groups.append({
                'sample_name': sample_name,
                'energy': group_energy,
                'files': group_files,
                'metadata': group_metadata,
                'trims': trims,
                'x': files[0]['x'],
                'y': files[0]['y'],
                'min_angle': overall_min_angle,
                'max_angle': overall_max_angle
            })
        
        # Sort scan groups by energy if requested
        if sort_by_energy:
            scan_groups.sort(key=lambda x: x['energy'])
        
        # Print summary of found groups
        print(f"\nFound {len(scan_groups)} scan groups:")
        print("=" * 115)
        print(f"{'#':3} | {'Sample':12} | {'Energy (eV)':10} | {'Position (X,Y)':20} | {'Angle Range':15} | {'Files':8} | {'Detector Types'}")
        print("-" * 115)
        
        for i, group in enumerate(scan_groups):
            # Count detector types
            detector_types = {}
            for filename in group['files']:
                base_filename = os.path.basename(filename)
                for meta in file_metadata:
                    if os.path.basename(meta['filename']) == base_filename:
                        detector_type = meta['detector']
                        if detector_type in detector_types:
                            detector_types[detector_type] += 1
                        else:
                            detector_types[detector_type] = 1
                        break
            
            detector_str = ", ".join([f"{dtype} ({count})" for dtype, count in detector_types.items()])
            position_str = f"({group['x']:.2f}, {group['y']:.2f})"
            
            # Format angle range
            angle_range_str = "N/A"
            if group['min_angle'] is not None and group['max_angle'] is not None:
                angle_range_str = f"{group['min_angle']:.2f} - {group['max_angle']:.2f}°"
            
            print(f"{i+1:3} | {group['sample_name']:12} | {group['energy']:10.1f} | {position_str:20} | {angle_range_str:15} | {len(group['files']):8} | {detector_str}")
        
        print("=" * 115)
        
        return scan_groups
        
    def _determine_trims(self, filenames):
        """
        Automatically determine trim values for a list of files
        by analyzing the data quality
        
        Parameters:
        -----------
        filenames : list of str
            List of data files to analyze
            
        Returns:
        --------
        trims : list of tuples
            List of (start_idx, end_idx) for each file
        """
        trims = []
        
        for filename in filenames:
            try:
                # Load the data
                data = self.load_data_file(filename)
                
                # Default trim (use all data points)
                start_trim = 0
                end_trim = -1
                
                # Simple signal-to-noise detection
                if data.shape[1] >= 2:
                    # Extract angles and intensities
                    angles = data[:, 0]
                    intensities = data[:, 1]
                    
                    # Find the maximum intensity
                    max_intensity = np.max(intensities)
                    
                    # Determine noise level (simple estimate)
                    # Use the standard deviation of the last 10% of the data
                    noise_level = np.std(intensities[-int(len(intensities)*0.1):])
                    
                    # Calculate signal-to-noise ratio
                    if noise_level > 0:
                        snr = max_intensity / noise_level
                    else:
                        snr = 100  # Default high value if noise is zero
                    
                    # Find regions with intensity below threshold (e.g., 5x noise level)
                    threshold = 5 * noise_level
                    
                    # Find where intensity drops below threshold at the beginning
                    start_indices = np.where(intensities < threshold)[0]
                    if len(start_indices) > 0 and start_indices[0] < len(intensities) * 0.2:
                        # Only apply trim if it's in the first 20% of the data
                        # Find the first point where intensity exceeds threshold
                        valid_starts = np.where(intensities > threshold)[0]
                        if len(valid_starts) > 0:
                            start_trim = valid_starts[0]
                    
                    # Find where intensity drops below threshold at the end
                    end_indices = np.where(intensities < threshold)[0]
                    if len(end_indices) > 0 and end_indices[-1] > len(intensities) * 0.8:
                        # Only apply trim if it's in the last 20% of the data
                        # Find the last point where intensity exceeds threshold
                        valid_ends = np.where(intensities > threshold)[0]
                        if len(valid_ends) > 0:
                            # Convert to negative index for end trim
                            end_trim = valid_ends[-1] - len(intensities)
                
                trims.append((start_trim, end_trim))
                
            except Exception as e:
                print(f"Error determining trim for {filename}: {str(e)}")
                # Use default trim if analysis fails
                trims.append((0, -1))
                
        return trims
    
    def process_auto_groups(self, scan_groups, output_template="{sample_name}_{energy:.1f}eV.dat", 
                      normalize=True, plot=True, convert_to_photons=False, trims=None, 
                      output_dir=None, plot_prefix_template=None, save_metadata=True,
                      estimate_thickness=True, min_prominence=0.1, 
                      min_thickness_nm=20, max_thickness_nm=100, smooth_data=False,
                      savgol_window=None, savgol_order=2, remove_zeros=True,
                      hybrid_trim_first=None):
        """
        Process automatically detected scan groups
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups as returned by auto_group_scans
        output_template : str
            Template for output filenames
        normalize : bool
            Whether to normalize the reflectivity
        plot : bool
            Whether to generate plots during processing
        convert_to_photons : bool
            Whether to convert photodiode current to photon flux
        trims : dict or list or None
            Trim settings to apply to each group. Can be:
            - None: use the trims from each scan_group (default)
            - dict: mapping group indices to trim values, e.g., {0: [(0, -10), (5, -1)]}
            - list: a list of trim lists for each group
        output_dir : str, optional
            Directory to save output files (will be created if it doesn't exist)
        plot_prefix_template : str, optional
            Template for plot filename prefixes, can include {sample_name}, {energy}, {index}
        save_metadata : bool
            Whether to save metadata about the reduced data
        estimate_thickness : bool
            Whether to estimate film thickness from fringe spacing
        min_prominence : float
            Minimum prominence for peak/valley detection in thickness estimation
        min_thickness_nm : float
            Minimum expected film thickness in nm
        max_thickness_nm : float
            Maximum expected film thickness in nm
        smooth_data : bool
            Whether to apply smoothing to the data 
        savgol_window : int, optional
            Window size for Savitzky-Golay filter. Must be odd and >= 5.
        savgol_order : int
            Polynomial order for the Savitzky-Golay filter. Default is 2.
        remove_zeros : bool
            Whether to remove data points with zero intensity
        hybrid_trim_first : int or None
            If not None, manually set the start trim value for the first file in each group
            while auto-determining trims for remaining files. This value will be used as the
            first trim's start value.
            
        Returns:
        --------
        results : list
            List of processed reflectivity data
        """
        results = []
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output will be saved to directory: {output_dir}")
        
        for i, group in enumerate(scan_groups):
            print(f"\nProcessing group {i+1}/{len(scan_groups)}: {group['sample_name']} at {group['energy']:.1f} eV")
            
            # Generate output filename from template
            output_filename = output_template.format(
                sample_name=group['sample_name'],
                energy=group['energy'],
                index=i+1
            )
            
            # Generate plot prefix if template provided
            if plot_prefix_template:
                plot_prefix = plot_prefix_template.format(
                    sample_name=group['sample_name'],
                    energy=group['energy'],
                    index=i+1
                )
            else:
                # Default to using output filename without extension
                plot_prefix = os.path.splitext(output_filename)[0]
            
            # Determine which trims to use
            group_trims = group['trims']  # Default to using group's trims
            
            if trims is not None:
                if isinstance(trims, dict) and i in trims:
                    # Use the trim values for this group index
                    group_trims = trims[i]
                    print(f"  Using custom trims for group {i+1}")
                elif isinstance(trims, list) and i < len(trims):
                    # Use the trim values at this index
                    group_trims = trims[i]
                    print(f"  Using custom trims for group {i+1}")
                else:
                    print(f"  No custom trims specified for group {i+1}, using defaults")
            
            # Apply hybrid trim if specified - autodetect all trims except the first file's start trim
            if hybrid_trim_first is not None:
                if len(group_trims) > 0:
                    # Auto-determine trims for all files
                    auto_trims = self._determine_trims(group['files'])
                    
                    # But replace just the first file's start value with the manual one
                    auto_trims[0] = (hybrid_trim_first, auto_trims[0][1])
                    
                    # Update group_trims with the hybrid approach
                    group_trims = auto_trims
                    print(f"  Using hybrid trimming: manual start trim ({hybrid_trim_first}) for first file, auto-detected trims for others")
            
            # Create a copy of the group with updated trims
            current_group = group.copy()
            current_group['trims'] = group_trims
            
            # Process this scan set
            result = self.process_scan_set(
                scan_group=current_group, 
                output_filename=output_filename,
                normalize=normalize,
                plot=plot,
                convert_to_photons=convert_to_photons,
                output_dir=output_dir,
                plot_prefix=plot_prefix,
                save_metadata=save_metadata,
                estimate_thickness=estimate_thickness,
                min_prominence=min_prominence,
                min_thickness_nm=min_thickness_nm,
                max_thickness_nm=max_thickness_nm,
                smooth_data=smooth_data,
                savgol_window=savgol_window,
                savgol_order=savgol_order,
                remove_zeros=remove_zeros
            )
            
            results.append(result)
            
        return results


    def save_scan_groups_to_table(self, scan_groups, output_format='csv', output_dir=None):
        """
        Save scan group information to a table file
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups from auto_group_scans
        output_format : str
            Output format: 'csv' or 'xlsx'
        output_dir : str, optional
            Directory to save the output file (will be created if it doesn't exist)
            
        Returns:
        --------
        output_path : str
            Path to the saved file
        """
        import pandas as pd
        import os
        from datetime import datetime
        
        # Create a list to hold the table rows
        table_data = []
        
        # Add a row for each file in each group
        for group_idx, group in enumerate(scan_groups):
            for file_idx, filename in enumerate(group['files']):
                # Find metadata for this file
                file_meta = None
                for meta in group['metadata']:
                    if os.path.basename(meta['filename']) == os.path.basename(filename):
                        file_meta = meta
                        break
                
                if file_meta:
                    # Format angle range
                    angle_range_str = "N/A"
                    if file_meta.get('min_angle') is not None and file_meta.get('max_angle') is not None:
                        angle_range_str = f"{file_meta['min_angle']:.2f} - {file_meta['max_angle']:.2f}"
                    
                    # Format count time
                    count_time = file_meta.get('count_time', None)
                    
                    # Get trim information
                    if group_idx < len(scan_groups) and file_idx < len(group['trims']):
                        trim_start, trim_end = group['trims'][file_idx]
                    else:
                        trim_start, trim_end = 0, -1
                    
                    # Add row to table data
                    row = {
                        'Group': group_idx + 1,
                        'Sample': group['sample_name'],
                        'Energy (eV)': group['energy'],
                        'Filename': os.path.basename(filename),
                        'Detector': file_meta.get('detector', 'Unknown'),
                        'Angle Range': angle_range_str,
                        'Count Time': count_time,
                        'Trim Start': trim_start,
                        'Trim End': trim_end,
                        'Position X': group['x'],
                        'Position Y': group['y'],
                        'Min Angle': file_meta.get('min_angle', None),
                        'Max Angle': file_meta.get('max_angle', None),
                    }
                    table_data.append(row)
        
        # Create a DataFrame from the table data
        df = pd.DataFrame(table_data)
        
        # Generate output filename based on the log file name
        log_file = "scan_groups"
        if hasattr(self, 'log_file') and self.log_file:
            log_file = os.path.splitext(os.path.basename(self.log_file))[0]
        
        # Add timestamp to avoid overwriting files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{log_file}_scan_groups_{timestamp}"
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path = output_filename
        
        # Save the DataFrame to the specified format
        if output_format.lower() == 'csv':
            csv_path = f"{output_path}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved scan group table to {csv_path}")
            return csv_path
        elif output_format.lower() == 'xlsx':
            xlsx_path = f"{output_path}.xlsx"
            df.to_excel(xlsx_path, index=False)
            print(f"Saved scan group table to {xlsx_path}")
            return xlsx_path
        else:
            print(f"Unsupported output format: {output_format}. Using CSV instead.")
            csv_path = f"{output_path}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved scan group table to {csv_path}")
            return csv_path


    def save_reduced_data_metadata(self, scan_group, reduced_data, output_dir=None, output_format='csv'):
        """
        Save metadata about the reduced data to a file with the same name as the original log file
        
        Parameters:
        -----------
        scan_group : dict
            Scan group information
        reduced_data : numpy array
            Reduced reflectivity data
        output_dir : str, optional
            Directory to save the output file (will be created if it doesn't exist)
        output_format : str
            Output format: 'csv' or 'xlsx'
            
        Returns:
        --------
        output_path : str
            Path to the saved file
        """
        import pandas as pd
        import os
        import numpy as np
        
        # Extract basic information
        sample_name = scan_group['sample_name']
        energy = scan_group['energy']
        position_x = scan_group.get('x', None)
        position_y = scan_group.get('y', None)
        
        # Calculate Q range
        q_min = np.min(reduced_data[:, 0])
        q_max = np.max(reduced_data[:, 0])
        
        # Get overall angle range
        angle_min = scan_group.get('min_angle', None)
        angle_max = scan_group.get('max_angle', None)
        
        # Count detector types
        detector_counts = {}
        for meta in scan_group.get('metadata', []):
            detector = meta.get('detector', 'Unknown')
            if detector in detector_counts:
                detector_counts[detector] += 1
            else:
                detector_counts[detector] = 1
        
        # If metadata is not available, try to get detector info from filenames
        if not detector_counts:
            for filename in scan_group.get('files', []):
                base_name = os.path.basename(filename)
                if 'PD' in base_name or 'photodiode' in base_name.lower():
                    detector_counts['photodiode'] = detector_counts.get('photodiode', 0) + 1
                elif 'CT' in base_name or 'cem' in base_name.lower():
                    detector_counts['cem'] = detector_counts.get('cem', 0) + 1
                else:
                    detector_counts['Unknown'] = detector_counts.get('Unknown', 0) + 1
        
        detector_str = ", ".join([f"{dtype} ({count})" for dtype, count in detector_counts.items()])
        
        # Create metadata dictionary
        metadata = {
            'Sample': sample_name,
            'Energy (eV)': energy,
            'Position X': position_x,
            'Position Y': position_y,
            'Q Min (Å⁻¹)': q_min,
            'Q Max (Å⁻¹)': q_max,
            'Angle Min (deg)': angle_min,
            'Angle Max (deg)': angle_max,
            'Detector Types': detector_str,
            'File Count': len(scan_group.get('files', [])),
        }
        
        # Add the individual file information
        files_info = []
        for i, filename in enumerate(scan_group.get('files', [])):
            # Find metadata for this file
            file_meta = None
            for meta in scan_group.get('metadata', []):
                if os.path.basename(meta['filename']) == os.path.basename(filename):
                    file_meta = meta
                    break
            
            if file_meta:
                file_info = {
                    'Filename': os.path.basename(filename),
                    'Detector': file_meta.get('detector', 'Unknown'),
                    'Min Angle (deg)': file_meta.get('min_angle', None),
                    'Max Angle (deg)': file_meta.get('max_angle', None),
                    'Count Time': file_meta.get('count_time', None),
                    'Trim Start': scan_group['trims'][i][0] if i < len(scan_group['trims']) else 0,
                    'Trim End': scan_group['trims'][i][1] if i < len(scan_group['trims']) else -1,
                }
                files_info.append(file_info)
            else:
                # If detailed metadata isn't available, create a basic entry
                files_info.append({
                    'Filename': os.path.basename(filename),
                    'Detector': 'Unknown',
                    'Min Angle (deg)': None,
                    'Max Angle (deg)': None,
                    'Count Time': None,
                    'Trim Start': scan_group['trims'][i][0] if i < len(scan_group['trims']) else 0,
                    'Trim End': scan_group['trims'][i][1] if i < len(scan_group['trims']) else -1,
                })
        
        # Generate output filename based on the log file name
        log_file_base = "reduced_data"
        if hasattr(self, 'log_file') and self.log_file:
            log_file_base = os.path.splitext(os.path.basename(self.log_file))[0]
        
        # Create metadata output name
        output_filename = f"{log_file_base}_{sample_name}_{energy:.1f}eV_metadata"
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path = output_filename
        
        # Save the metadata
        if output_format.lower() == 'csv':
            # For CSV, we'll save two files: one for general metadata and one for file details
            meta_path = f"{output_path}.csv"
            files_path = f"{output_path}_files.csv"
            
            # Convert metadata to DataFrame and save
            meta_df = pd.DataFrame([metadata])
            meta_df.to_csv(meta_path, index=False)
            
            # Save file information if available
            if files_info:
                files_df = pd.DataFrame(files_info)
                files_df.to_csv(files_path, index=False)
                
            print(f"Saved metadata to {meta_path}")
            print(f"Saved file details to {files_path}")
            return meta_path
        
        elif output_format.lower() == 'xlsx':
            # For Excel, we'll save both metadata and file info in different sheets
            xlsx_path = f"{output_path}.xlsx"
            
            # Create Excel writer with metadata sheet
            with pd.ExcelWriter(xlsx_path) as writer:
                # Save general metadata
                meta_df = pd.DataFrame([metadata])
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Save file information if available
                if files_info:
                    files_df = pd.DataFrame(files_info)
                    files_df.to_excel(writer, sheet_name='Files', index=False)
            
            print(f"Saved metadata to {xlsx_path}")
            return xlsx_path
        
        else:
            print(f"Unsupported output format: {output_format}. Using CSV instead.")
            meta_path = f"{output_path}.csv"
            meta_df = pd.DataFrame([metadata])
            meta_df.to_csv(meta_path, index=False)
            print(f"Saved metadata to {meta_path}")
            return meta_path

    
    
    def _sort_files_in_group(self, group):
        """
        Sort files within a group by detector type and angle (same as initial grouping)
        
        Parameters:
        -----------
        group : dict
            Scan group dictionary
            
        Returns:
        --------
        group : dict
            Group with sorted files and metadata
        """
        if not group.get('metadata'):
            return group
        
        # Separate files by detector type
        photodiode_items = []
        cem_items = []
        unknown_items = []
        
        for i, meta in enumerate(group['metadata']):
            detector_str = str(meta['detector']).lower() if meta['detector'] is not None else ""
            item = {
                'filename': group['files'][i],
                'metadata': meta,
                'trim': group['trims'][i] if i < len(group['trims']) else (0, -1)
            }
            
            if "photodiode" in detector_str:
                photodiode_items.append(item)
            elif "cem" in detector_str:
                cem_items.append(item)
            else:
                unknown_items.append(item)
        
        # Sort each detector type by angle
        for items in [photodiode_items, cem_items, unknown_items]:
            items.sort(key=lambda x: x['metadata']['angle'] if x['metadata']['angle'] is not None else 0)
        
        # Combine in order: photodiode first, then cem, then unknown
        sorted_items = photodiode_items + cem_items + unknown_items
        
        # Update the group with sorted information
        group['files'] = [item['filename'] for item in sorted_items]
        group['metadata'] = [item['metadata'] for item in sorted_items]
        group['trims'] = [item['trim'] for item in sorted_items]
        
        # Recalculate overall angle range
        overall_min_angle = float('inf')
        overall_max_angle = float('-inf')
        
        for meta in group['metadata']:
            if meta['min_angle'] is not None and meta['min_angle'] < overall_min_angle:
                overall_min_angle = meta['min_angle']
            if meta['max_angle'] is not None and meta['max_angle'] > overall_max_angle:
                overall_max_angle = meta['max_angle']
        
        # Handle case where no valid angles were found
        if overall_min_angle == float('inf'):
            overall_min_angle = None
        if overall_max_angle == float('-inf'):
            overall_max_angle = None
        
        group['min_angle'] = overall_min_angle
        group['max_angle'] = overall_max_angle
        
        return group

    def combine_groups(self, scan_groups, group_indices, new_sample_name=None):
        """
        Combine multiple groups into a single group
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups
        group_indices : list of int
            Indices of groups to combine (1-based, as displayed to user)
        new_sample_name : str, optional
            Name for the combined group. If None, uses the first group's name
            
        Returns:
        --------
        scan_groups : list of dicts
            Updated list of scan groups with combined group
        """
        if len(group_indices) < 2:
            print("Error: Need at least 2 groups to combine")
            return scan_groups
        
        # Convert to 0-based indices
        zero_based_indices = [i - 1 for i in group_indices]
        
        # Validate indices
        valid_indices = [i for i in zero_based_indices if 0 <= i < len(scan_groups)]
        if len(valid_indices) != len(zero_based_indices):
            invalid_user_indices = [group_indices[i] for i, idx in enumerate(zero_based_indices) if idx not in valid_indices]
            print(f"Warning: Invalid group numbers: {invalid_user_indices}. Valid range is 1-{len(scan_groups)}")
            zero_based_indices = valid_indices
        
        if len(zero_based_indices) < 2:
            print("Error: Not enough valid groups to combine")
            return scan_groups
        
        # Sort indices in descending order for safe removal
        zero_based_indices = sorted(zero_based_indices, reverse=True)
        
        # Get the groups to combine (use the first index as the base)
        base_group = deepcopy(scan_groups[zero_based_indices[-1]])  # Lowest index becomes base
        
        # Check if energies are compatible (within tolerance)
        energy_tolerance = 0.5
        base_energy = base_group['energy']
        
        for idx in zero_based_indices[:-1]:  # Skip the base group
            group = scan_groups[idx]
            if abs(group['energy'] - base_energy) > energy_tolerance:
                print(f"Warning: Energy mismatch between groups. Base: {base_energy:.1f} eV, "
                    f"Group {idx+1}: {group['energy']:.1f} eV")
        
        # Combine all files, metadata, and trims
        combined_files = list(base_group['files'])
        combined_metadata = list(base_group['metadata'])
        combined_trims = list(base_group['trims'])
        
        for idx in zero_based_indices[:-1]:  # Skip the base group
            group = scan_groups[idx]
            combined_files.extend(group['files'])
            combined_metadata.extend(group['metadata'])
            combined_trims.extend(group['trims'])
        
        # Update the base group
        base_group['files'] = combined_files
        base_group['metadata'] = combined_metadata
        base_group['trims'] = combined_trims
        
        # Update sample name if provided
        if new_sample_name:
            base_group['sample_name'] = new_sample_name
        
        # Sort the combined group
        base_group = self._sort_files_in_group(base_group)
        
        # Remove the other groups (in reverse order to maintain indices)
        new_scan_groups = list(scan_groups)
        for idx in zero_based_indices[:-1]:  # Skip the base group
            del new_scan_groups[idx]
        
        # Update the base group in the list
        base_index = zero_based_indices[-1]
        # Adjust base index for any groups removed before it
        adjusted_base_index = base_index
        for idx in zero_based_indices[:-1]:
            if idx < base_index:
                adjusted_base_index -= 1
        
        new_scan_groups[adjusted_base_index] = base_group
        
        print(f"Combined {len(zero_based_indices)} groups into group {adjusted_base_index + 1}")
        print(f"Combined group now has {len(base_group['files'])} files")
        
        return new_scan_groups

    def remove_scan_from_group(self, scan_groups, group_index, file_index):
        """
        Remove a specific scan from a group
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups
        group_index : int
            Index of the group to modify (1-based, as displayed to user)
        file_index : int
            Index of the file within the group to remove (1-based, as displayed to user)
            
        Returns:
        --------
        scan_groups : list of dicts
            Updated list of scan groups
        removed_info : dict
            Information about the removed scan (filename, metadata, trim)
        """
        # Convert to 0-based indices
        zero_based_group_index = group_index - 1
        zero_based_file_index = file_index - 1
        
        if not (0 <= zero_based_group_index < len(scan_groups)):
            print(f"Error: Invalid group number {group_index}. Valid range is 1-{len(scan_groups)}")
            return scan_groups, None
        
        group = scan_groups[zero_based_group_index]
        
        if not (0 <= zero_based_file_index < len(group['files'])):
            print(f"Error: Invalid file number {file_index}. Valid range is 1-{len(group['files'])}")
            return scan_groups, None
        
        # Store information about the removed scan
        removed_info = {
            'filename': group['files'][zero_based_file_index],
            'metadata': group['metadata'][zero_based_file_index] if zero_based_file_index < len(group['metadata']) else None,
            'trim': group['trims'][zero_based_file_index] if zero_based_file_index < len(group['trims']) else (0, -1)
        }
        
        # Remove from all lists
        group['files'].pop(zero_based_file_index)
        if zero_based_file_index < len(group['metadata']):
            group['metadata'].pop(zero_based_file_index)
        if zero_based_file_index < len(group['trims']):
            group['trims'].pop(zero_based_file_index)
        
        # Check if group is now empty
        if not group['files']:
            print(f"Warning: Group {group_index} is now empty after removing scan")
            # Optionally remove the empty group
            scan_groups.pop(zero_based_group_index)
            print(f"Removed empty group {group_index}")
        else:
            # Re-sort the group and update angle ranges
            scan_groups[zero_based_group_index] = self._sort_files_in_group(group)
        
        print(f"Removed {os.path.basename(removed_info['filename'])} from group {group_index}")
        
        return scan_groups, removed_info

    def move_scan_to_new_group(self, scan_groups, group_index, file_index, new_sample_name=None):
        """
        Remove a scan from one group and create a new group with just that scan
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups
        group_index : int
            Index of the source group (1-based, as displayed to user)
        file_index : int
            Index of the file within the group to move (1-based, as displayed to user)
        new_sample_name : str, optional
            Name for the new group. If None, derives from the original group
            
        Returns:
        --------
        scan_groups : list of dicts
            Updated list of scan groups with new group added
        """
        # Convert to 0-based for internal operations
        zero_based_group_index = group_index - 1
        
        if not (0 <= zero_based_group_index < len(scan_groups)):
            print(f"Error: Invalid group number {group_index}. Valid range is 1-{len(scan_groups)}")
            return scan_groups
        
        original_group = scan_groups[zero_based_group_index]
        
        # Remove the scan from the original group
        updated_groups, removed_info = self.remove_scan_from_group(scan_groups, group_index, file_index)
        
        if removed_info is None:
            return scan_groups
        
        # Create a new group with the removed scan
        new_group = {
            'sample_name': new_sample_name or f"{original_group['sample_name']}_split",
            'energy': removed_info['metadata']['energy'] if removed_info['metadata'] else original_group['energy'],
            'files': [removed_info['filename']],
            'metadata': [removed_info['metadata']] if removed_info['metadata'] else [],
            'trims': [removed_info['trim']],
            'x': removed_info['metadata']['x'] if removed_info['metadata'] else original_group['x'],
            'y': removed_info['metadata']['y'] if removed_info['metadata'] else original_group['y'],
            'min_angle': removed_info['metadata']['min_angle'] if removed_info['metadata'] else None,
            'max_angle': removed_info['metadata']['max_angle'] if removed_info['metadata'] else None
        }
        
        # Add the new group to the list
        updated_groups.append(new_group)
        
        print(f"Created new group '{new_group['sample_name']}' (group {len(updated_groups)}) with {os.path.basename(removed_info['filename'])}")
        
        return updated_groups

    def move_scan_to_existing_group(self, scan_groups, source_group_index, file_index, target_group_index):
        """
        Move a scan from one group to another existing group
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups
        source_group_index : int
            Index of the source group (1-based, as displayed to user)
        file_index : int
            Index of the file within the source group to move (1-based, as displayed to user)
        target_group_index : int
            Index of the target group (1-based, as displayed to user)
            
        Returns:
        --------
        scan_groups : list of dicts
            Updated list of scan groups
        """
        # Convert to 0-based indices
        zero_based_target_index = target_group_index - 1
        
        if not (0 <= zero_based_target_index < len(scan_groups)):
            print(f"Error: Invalid target group number {target_group_index}. Valid range is 1-{len(scan_groups)}")
            return scan_groups
        
        if source_group_index == target_group_index:
            print("Error: Source and target groups cannot be the same")
            return scan_groups
        
        # Remove the scan from the source group
        updated_groups, removed_info = self.remove_scan_from_group(scan_groups, source_group_index, file_index)
        
        if removed_info is None:
            return scan_groups
        
        # Adjust target index if source group was removed due to being empty
        adjusted_target_index = zero_based_target_index
        if len(updated_groups) < len(scan_groups) and (source_group_index - 1) < zero_based_target_index:
            adjusted_target_index -= 1
        
        if not (0 <= adjusted_target_index < len(updated_groups)):
            print(f"Error: Target group is no longer valid after source group removal")
            return scan_groups
        
        # Add the scan to the target group
        target_group = updated_groups[adjusted_target_index]
        target_group['files'].append(removed_info['filename'])
        if removed_info['metadata']:
            target_group['metadata'].append(removed_info['metadata'])
        target_group['trims'].append(removed_info['trim'])
        
        # Sort the target group
        updated_groups[adjusted_target_index] = self._sort_files_in_group(target_group)
        
        print(f"Moved {os.path.basename(removed_info['filename'])} to group {adjusted_target_index + 1}")
        
        return updated_groups

    def remove_groups(self, scan_groups, group_indices):
        """
        Remove entire groups from the scan groups list
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups
        group_indices : list of int
            Indices of groups to remove (1-based, as displayed to user)
            
        Returns:
        --------
        scan_groups : list of dicts
            Updated list of scan groups with specified groups removed
        removed_groups : list of dicts
            List of removed groups for reference
        """
        if not group_indices:
            print("Error: No group indices provided")
            return scan_groups, []
        
        # Convert to 0-based indices
        zero_based_indices = [i - 1 for i in group_indices]
        
        # Validate indices
        valid_indices = [i for i in zero_based_indices if 0 <= i < len(scan_groups)]
        if len(valid_indices) != len(zero_based_indices):
            invalid_user_indices = [group_indices[i] for i, idx in enumerate(zero_based_indices) if idx not in valid_indices]
            print(f"Warning: Invalid group numbers: {invalid_user_indices}. Valid range is 1-{len(scan_groups)}")
            zero_based_indices = valid_indices
        
        if not zero_based_indices:
            print("Error: No valid groups to remove")
            return scan_groups, []
        
        # Sort indices in descending order for safe removal
        zero_based_indices = sorted(zero_based_indices, reverse=True)
        
        # Store removed groups for reference
        removed_groups = []
        new_scan_groups = list(scan_groups)
        
        # Remove groups in reverse order to maintain index validity
        for idx in zero_based_indices:
            removed_group = new_scan_groups.pop(idx)
            removed_groups.append(removed_group)
            print(f"Removed group {idx + 1}: {removed_group['sample_name']} ({removed_group['energy']:.1f} eV)")
        
        # Reverse the removed_groups list to match original order
        removed_groups.reverse()
        
        print(f"Successfully removed {len(removed_groups)} groups")
        print(f"Remaining groups: {len(new_scan_groups)}")
        
        return new_scan_groups, removed_groups

    def print_group_summary(self, scan_groups, show_details=False):
        """
        Print a summary of the current scan groups
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups
        show_details : bool
            Whether to show detailed information about each file
        """
        print(f"\nCurrent scan groups ({len(scan_groups)} total):")
        print("=" * 115)
        print(f"{'#':3} | {'Sample':12} | {'Energy (eV)':10} | {'Position (X,Y)':20} | {'Angle Range':15} | {'Files':8} | {'Detector Types'}")
        print("-" * 115)
        
        for i, group in enumerate(scan_groups):
            # Count detector types
            detector_types = {}
            for meta in group.get('metadata', []):
                detector_type = meta.get('detector', 'Unknown')
                if detector_type in detector_types:
                    detector_types[detector_type] += 1
                else:
                    detector_types[detector_type] = 1
            
            detector_str = ", ".join([f"{dtype} ({count})" for dtype, count in detector_types.items()])
            position_str = f"({group['x']:.2f}, {group['y']:.2f})"
            
            # Format angle range
            angle_range_str = "N/A"
            if group.get('min_angle') is not None and group.get('max_angle') is not None:
                angle_range_str = f"{group['min_angle']:.2f} - {group['max_angle']:.2f}°"
            
            print(f"{i+1:3} | {group['sample_name']:12} | {group['energy']:10.1f} | {position_str:20} | {angle_range_str:15} | {len(group['files']):8} | {detector_str}")
            
            # Show detailed file information if requested
            if show_details:
                print(f"    Files in group {i+1}:")
                for j, filename in enumerate(group['files']):
                    base_filename = os.path.basename(filename)
                    if j < len(group.get('metadata', [])):
                        meta = group['metadata'][j]
                        detector = meta.get('detector', 'Unknown')
                        angle_range = f"{meta.get('min_angle', 'N/A'):.2f} - {meta.get('max_angle', 'N/A'):.2f}°" if meta.get('min_angle') is not None else "N/A"
                    else:
                        detector = 'Unknown'
                        angle_range = 'N/A'
                    
                    trim_str = f"({group['trims'][j][0]}, {group['trims'][j][1]})" if j < len(group['trims']) else "(0, -1)"
                    print(f"      {j+1:2}: {base_filename:25} | {detector:12} | {angle_range:15} | {trim_str}")
                print()
        
        print("=" * 115)

    def interactive_group_editor(self, scan_groups):
        """
        Interactive command-line interface for editing scan groups
        
        Parameters:
        -----------
        scan_groups : list of dicts
            List of scan groups to edit
            
        Returns:
        --------
        scan_groups : list of dicts
            Modified list of scan groups
        """
        print("\n" + "="*60)
        print("           INTERACTIVE SCAN GROUP EDITOR")
        print("="*60)
        print("Commands:")
        print("  'show' or 's' - Show current groups")
        print("  'details' or 'd' - Show groups with file details")
        print("  'combine X Y Z' - Combine groups X, Y, Z into one group")
        print("  'remove X Y' - Remove file Y from group X")
        print("  'remove_group X Y Z' - Remove entire groups X, Y, Z")
        print("  'move X Y new' - Move file Y from group X to a new group")
        print("  'move X Y Z' - Move file Y from group X to existing group Z")
        print("  'rename X NAME' - Rename group X to NAME")
        print("  'quit' or 'q' - Finish editing")
        print("  'help' or 'h' - Show this help")
        print("="*60)
        
        current_groups = deepcopy(scan_groups)
        
        while True:
            print(f"\nCurrent state: {len(current_groups)} groups")
            command = input("Enter command: ").strip().lower()
            
            if command in ['quit', 'q']:
                break
            elif command in ['help', 'h']:
                print("\nCommands:")
                print("  'show' or 's' - Show current groups")
                print("  'details' or 'd' - Show groups with file details")
                print("  'combine X Y Z' - Combine groups X, Y, Z into one group")
                print("  'remove X Y' - Remove file Y from group X")
                print("  'remove_group X Y Z' - Remove entire groups X, Y, Z")
                print("  'move X Y new' - Move file Y from group X to a new group")
                print("  'move X Y Z' - Move file Y from group X to existing group Z")
                print("  'rename X NAME' - Rename group X to NAME")
                print("  'quit' or 'q' - Finish editing")
            elif command in ['show', 's']:
                self.print_group_summary(current_groups, show_details=False)
            elif command in ['details', 'd']:
                self.print_group_summary(current_groups, show_details=True)
            elif command.startswith('combine'):
                try:
                    parts = command.split()
                    if len(parts) < 3:
                        print("Error: combine command needs at least 2 group indices")
                        continue
                    indices = [int(x) for x in parts[1:]]
                    
                    # Ask for new name
                    new_name = input("Enter new name for combined group (or press Enter to use default): ").strip()
                    if not new_name:
                        new_name = None
                    
                    current_groups = self.combine_groups(current_groups, indices, new_name)
                    print("Groups combined successfully!")
                    
                except ValueError:
                    print("Error: Please provide valid group indices")
                except Exception as e:
                    print(f"Error combining groups: {str(e)}")
                    
            elif command.startswith('remove'):
                try:
                    parts = command.split()
                    if len(parts) != 3:
                        print("Error: remove command format: 'remove GROUP_INDEX FILE_INDEX'")
                        continue
                    group_idx = int(parts[1])
                    file_idx = int(parts[2])
                    
                    current_groups, removed = self.remove_scan_from_group(current_groups, group_idx, file_idx)
                    if removed:
                        print(f"Removed scan successfully!")
                        
                except ValueError:
                    print("Error: Please provide valid indices")
                except Exception as e:
                    print(f"Error removing scan: {str(e)}")
                    
            elif command.startswith('remove_group'):
                try:
                    parts = command.split()
                    if len(parts) < 2:
                        print("Error: remove_group command needs at least one group index")
                        continue
                    indices = [int(x) for x in parts[1:]]
                    
                    # Confirm removal
                    print(f"Are you sure you want to remove groups {indices}? This cannot be undone.")
                    confirm = input("Type 'yes' to confirm: ").strip().lower()
                    if confirm == 'yes':
                        current_groups, removed = self.remove_groups(current_groups, indices)
                        if removed:
                            print(f"Removed {len(removed)} groups successfully!")
                    else:
                        print("Group removal cancelled.")
                        
                except ValueError:
                    print("Error: Please provide valid group indices")
                except Exception as e:
                    print(f"Error removing groups: {str(e)}")
                    
            elif command.startswith('move'):
                try:
                    parts = command.split()
                    if len(parts) != 4:
                        print("Error: move command format: 'move SOURCE_GROUP FILE_INDEX TARGET' where TARGET is group index or 'new'")
                        continue
                        
                    source_idx = int(parts[1])
                    file_idx = int(parts[2])
                    target = parts[3]
                    
                    if target == 'new':
                        new_name = input("Enter name for new group (or press Enter for default): ").strip()
                        if not new_name:
                            new_name = None
                        current_groups = self.move_scan_to_new_group(current_groups, source_idx, file_idx, new_name)
                        print("Moved scan to new group successfully!")
                    else:
                        target_idx = int(target)
                        current_groups = self.move_scan_to_existing_group(current_groups, source_idx, file_idx, target_idx)
                        print("Moved scan successfully!")
                        
                except ValueError:
                    print("Error: Please provide valid indices")
                except Exception as e:
                    print(f"Error moving scan: {str(e)}")
                    
            elif command.startswith('rename'):
                try:
                    parts = command.split(None, 2)  # Split into at most 3 parts
                    if len(parts) < 3:
                        print("Error: rename command format: 'rename GROUP_INDEX NEW_NAME'")
                        continue
                        
                    group_idx = int(parts[1])
                    new_name = parts[2]
                    
                    if 0 <= group_idx < len(current_groups):
                        old_name = current_groups[group_idx]['sample_name']
                        current_groups[group_idx]['sample_name'] = new_name
                        print(f"Renamed group {group_idx} from '{old_name}' to '{new_name}'")
                    else:
                        print(f"Error: Invalid group index {group_idx}")
                        
                except ValueError:
                    print("Error: Please provide a valid group index")
                except Exception as e:
                    print(f"Error renaming group: {str(e)}")
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
        
        print("\nEditing complete!")
        self.print_group_summary(current_groups, show_details=False)
        
        return current_groups
    
    def load_open_beam_file(self, file_path):
        """
        Load open beam intensity data from file
        
        Parameters:
        -----------
        file_path : str
            Path to the open beam data file
            
        Returns:
        --------
        bool
            True if loaded successfully, False otherwise
        """
        try:
            # Based on the example file, the format appears to be:
            # Column 1: Energy (eV) 
            # Column 2: Intensity
            # Column 3: Error (optional)
            # Column 4: Another parameter (optional)
            
            print(f"Loading open beam file: {file_path}")
            
            # Try loading with numpy first (handles whitespace/tab separation automatically)
            try:
                data = np.loadtxt(file_path, skiprows=1)
                print(f"Loaded data shape: {data.shape}")
                
                if len(data.shape) == 1:
                    print("Error: File appears to contain only one row or one column")
                    return False
                    
                if data.shape[1] >= 2:
                    # Use only the first two columns (energy, intensity)
                    self.open_beam_data = pd.DataFrame({
                        'energy': data[:, 0],
                        'intensity': data[:, 1]
                    })
                    print(f"Using columns: Energy (col 1), Intensity (col 2)")
                    
                    # Optionally store additional columns for reference
                    if data.shape[1] >= 3:
                        self.open_beam_data['error'] = data[:, 2]
                        print(f"Also loaded error column (col 3)")
                    if data.shape[1] >= 4:
                        self.open_beam_data['monitor'] = data[:, 3]
                        print(f"Also loaded monitor/additional data (col 4)")
                        
                else:
                    print(f"Error: Open beam file must have at least 2 columns (energy, intensity)")
                    print(f"Found {data.shape[1]} columns")
                    return False
                    
            except Exception as e:
                print(f"numpy.loadtxt failed: {e}")
                print("Trying pandas read_csv...")
                
                # If numpy fails, try pandas with various separators
                for sep in ['\t', ' ', ',', None]:  # None means any whitespace
                    try:
                        print(f"Trying separator: {repr(sep)}")
                        self.open_beam_data = pd.read_csv(file_path, sep=sep, header=None, comment='#')
                        
                        if self.open_beam_data.shape[1] >= 2:
                            # Use only first two columns
                            self.open_beam_data = self.open_beam_data.iloc[:, :2]
                            self.open_beam_data.columns = ['energy', 'intensity']
                            print(f"Successfully loaded with pandas using separator {repr(sep)}")
                            break
                        else:
                            print(f"Not enough columns with separator {repr(sep)}")
                            continue
                            
                    except Exception as e2:
                        print(f"Pandas with separator {repr(sep)} failed: {e2}")
                        continue
                else:
                    # If all methods fail
                    print("All loading methods failed")
                    return False
            
            self.open_beam_file = file_path
            
            # Basic validation
            if len(self.open_beam_data) == 0:
                print(f"Error: Open beam file is empty")
                return False
                
            # Check for valid data
            if self.open_beam_data['energy'].isna().any():
                print("Warning: Some energy values are NaN, removing those rows")
                self.open_beam_data = self.open_beam_data.dropna(subset=['energy'])
                
            if self.open_beam_data['intensity'].isna().any():
                print("Warning: Some intensity values are NaN, removing those rows")
                self.open_beam_data = self.open_beam_data.dropna(subset=['intensity'])
            
            if len(self.open_beam_data) == 0:
                print("Error: No valid data remaining after removing NaN values")
                return False
                
            # Sort by energy for interpolation
            self.open_beam_data = self.open_beam_data.sort_values('energy').reset_index(drop=True)
            
            # Initialize open beam normalization flag
            if not hasattr(self, 'use_open_beam_normalization'):
                self.use_open_beam_normalization = False
            
            print(f"Successfully loaded open beam data from {os.path.basename(file_path)}")
            print(f"Energy range: {self.open_beam_data['energy'].min():.1f} - {self.open_beam_data['energy'].max():.1f} eV")
            print(f"Data points: {len(self.open_beam_data)}")
            print(f"Sample data:")
            print(f"  First few points: Energy={self.open_beam_data['energy'].iloc[0]:.1f} eV, Intensity={self.open_beam_data['intensity'].iloc[0]:.3e}")
            print(f"  Last few points: Energy={self.open_beam_data['energy'].iloc[-1]:.1f} eV, Intensity={self.open_beam_data['intensity'].iloc[-1]:.3e}")
            
            return True
            
        except Exception as e:
            print(f"Error loading open beam file {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


    def get_open_beam_intensity(self, energy):
        """
        Get open beam intensity at a specific energy using interpolation
        
        Parameters:
        -----------
        energy : float
            Energy in eV
            
        Returns:
        --------
        float
            Open beam intensity at the specified energy, or None if error
        """
        if not hasattr(self, 'open_beam_data') or self.open_beam_data is None:
            print("Error: No open beam data loaded")
            return None
            
        try:
            # Create interpolation function
            energy_values = self.open_beam_data['energy'].values
            intensity_values = self.open_beam_data['intensity'].values
            
            if len(energy_values) == 1:
                # If only one data point, use that value
                return intensity_values[0]
            
            # Use linear interpolation with extrapolation for energies outside range
            interp_func = interp1d(energy_values, intensity_values, 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
            
            intensity = float(interp_func(energy))
            
            # Check if extrapolation was used (warn user)
            if energy < energy_values.min() or energy > energy_values.max():
                print(f"Warning: Energy {energy:.1f} eV is outside open beam data range "
                    f"({energy_values.min():.1f} - {energy_values.max():.1f} eV). Using extrapolation.")
            
            return intensity
            
        except Exception as e:
            print(f"Error interpolating open beam intensity at {energy:.1f} eV: {str(e)}")
            return None


    def set_open_beam_normalization(self, use_open_beam=True):
        """
        Enable or disable open beam normalization
        
        Parameters:
        -----------
        use_open_beam : bool
            True to use open beam normalization, False for standard normalization
            
        Returns:
        --------
        bool
            True if setting was successful
        """
        if use_open_beam and (not hasattr(self, 'open_beam_data') or self.open_beam_data is None):
            print("Warning: No open beam data loaded. Please load open beam file first.")
            return False
            
        self.use_open_beam_normalization = use_open_beam
        norm_type = "open beam" if use_open_beam else "standard"
        print(f"Normalization set to: {norm_type}")
        return True


    def plot_open_beam_data(self, save_path=None):
        """
        Plot the loaded open beam data
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not hasattr(self, 'open_beam_data') or self.open_beam_data is None:
            print("Error: No open beam data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.open_beam_data['energy'], self.open_beam_data['intensity'], 'bo-', markersize=4)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Open Beam Intensity')
        plt.title(f'Open Beam Data\n{os.path.basename(self.open_beam_file)}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Open beam plot saved to {save_path}")
        
        plt.show()


    def apply_normalization(self, intensity_data, energy, normalize_type="auto"):
        """
        Apply normalization to intensity data (standard or open beam)
        
        Parameters:
        -----------
        intensity_data : numpy.ndarray
            Intensity data to normalize
        energy : float
            Photon energy in eV
        normalize_type : str
            "auto" (use current setting), "standard", "open_beam", or "none"
            
        Returns:
        --------
        numpy.ndarray
            Normalized intensity data
        """
        if normalize_type == "none":
            return intensity_data
        
        # Determine which normalization to use
        if normalize_type == "auto":
            use_open_beam = getattr(self, 'use_open_beam_normalization', False)
        elif normalize_type == "open_beam":
            use_open_beam = True
        elif normalize_type == "standard":
            use_open_beam = False
        else:
            print(f"Warning: Unknown normalization type '{normalize_type}'. Using standard.")
            use_open_beam = False
        
        if use_open_beam and hasattr(self, 'open_beam_data') and self.open_beam_data is not None:
            # Use open beam normalization
            open_beam_intensity = self.get_open_beam_intensity(energy)
            
            if open_beam_intensity is None or open_beam_intensity <= 0:
                print(f"Warning: Invalid open beam intensity at {energy:.1f} eV. Using standard normalization.")
                normalized_data = intensity_data / np.max(intensity_data)
                print("Applied standard normalization (fallback)")
            else:
                normalized_data = intensity_data / open_beam_intensity
                print(f"Applied open beam normalization at {energy:.1f} eV (I0 = {open_beam_intensity:.3e})")
        else:
            # Use standard normalization
            normalized_data = intensity_data / np.max(intensity_data)
            print("Applied standard normalization")
        
        return normalized_data


    def reduce_data_with_open_beam_option(self, scans, trims, backgrounds, energy, 
                                        normalize=True, plot=False, convert_to_photons=False,
                                        output_dir=None, plot_prefix=None,
                                        smooth_data=False, savgol_window=None, savgol_order=2,
                                        remove_zeros=True, use_open_beam=None):
        """
        Enhanced version of reduce_data_with_backgrounds that supports open beam normalization
        
        Parameters:
        -----------
        scans : list of numpy arrays
            List of loaded scan data (angle, intensity)
        trims : list of tuples
            List of (start, end) trim indices for each scan
        backgrounds : list of floats
            List of background values to subtract from each scan
        energy : float
            Photon energy in eV
        normalize : bool
            Whether to normalize the final reflectivity
        plot : bool
            Whether to generate plots during processing
        convert_to_photons : bool
            Whether to convert photodiode current to photon flux
        output_dir : str, optional
            Directory to save plots
        plot_prefix : str, optional
            Prefix for plot filenames
        smooth_data : bool
            Whether to apply Savitzky-Golay smoothing
        savgol_window : int, optional
            Window size for smoothing filter
        savgol_order : int
            Polynomial order for smoothing filter
        remove_zeros : bool
            Whether to remove zero intensity points
        use_open_beam : bool, optional
            Override for open beam normalization (None = use current setting)
            
        Returns:
        --------
        If smooth_data is True:
            (smoothed_data, raw_data) : tuple of numpy arrays
        Else:
            processed_data : numpy array
        """
        # Temporarily set open beam normalization if override provided
        original_setting = getattr(self, 'use_open_beam_normalization', False)
        if use_open_beam is not None:
            self.use_open_beam_normalization = use_open_beam
        
        try:
            # Use the existing reduce_data_with_backgrounds method structure
            # but replace the normalization part
            
            if len(scans) != len(trims) or len(scans) != len(backgrounds):
                print(f"Error: Mismatch in number of scans ({len(scans)}), trims ({len(trims)}), and backgrounds ({len(backgrounds)})")
                return None
            
            # Ensure we have trim values for all scans
            if len(trims) < len(scans):
                print(f"Warning: Not enough trim values provided. Adding default trims for {len(scans) - len(trims)} scans.")
                trims.extend([(0, -1)] * (len(scans) - len(trims)))
            
            # Process the first scan
            refl = deepcopy(scans[0][trims[0][0]:(len(scans[0][:,0])+trims[0][1]), 0:2])
            
            # Apply background subtraction to first scan
            if backgrounds[0] != 0.0:
                refl[:, 1] = refl[:, 1] - backgrounds[0]
                print(f"Applied background subtraction to scan 1: -{backgrounds[0]:.3f}")
            
            # Remove zeros if requested
            if remove_zeros:
                non_zero_mask = refl[:,1] > 0
                if not all(non_zero_mask):
                    print(f"Removing {len(refl) - np.sum(non_zero_mask)} zero data points from first scan")
                    refl = refl[non_zero_mask]
            
            # Convert to photon flux if requested
            if convert_to_photons and hasattr(self, 'calibration_data') and self.calibration_data is not None:
                refl[:,1] = self.amp_to_photon_flux(refl[:,1], energy)
            
            # Store all raw scans for combined plotting
            all_scans = [deepcopy(refl)]
            
            # Process additional scans (using existing logic from reduce_data_with_backgrounds)
            for i in range(1, len(scans)):
                scan = scans[i][trims[i][0]:(len(scans[i][:,0])+trims[i][1]), 0:2]
                
                # Apply background subtraction
                if backgrounds[i] != 0.0:
                    scan[:, 1] = scan[:, 1] - backgrounds[i]
                    print(f"Applied background subtraction to scan {i+1}: -{backgrounds[i]:.3f}")
                
                # Remove zeros if requested
                if remove_zeros:
                    non_zero_mask = scan[:,1] > 0
                    if not all(non_zero_mask):
                        print(f"Removing {len(scan) - np.sum(non_zero_mask)} zero data points from scan {i+1}")
                        scan = scan[non_zero_mask]
                
                # Convert to photon flux if requested
                if convert_to_photons and hasattr(self, 'calibration_data') and self.calibration_data is not None:
                    scan[:,1] = self.amp_to_photon_flux(scan[:,1], energy)
                
                # Find overlap and scale using existing find_nearest method
                idx, val = self.find_nearest(scan[:,0], refl[-1,0])
                
                # Check for valid overlap
                if idx <= 0 or np.isnan(val):
                    print(f"Warning: No overlap found for scan {i+1}. Appending without scaling.")
                    refl = np.vstack((refl, scan))
                else:
                    # Calculate scaling factor
                    overlap_angle = refl[-1, 0]
                    current_intensity = refl[-1, 1]
                    new_intensity = scan[idx, 1]
                    
                    if new_intensity > 0:
                        scale_factor = current_intensity / new_intensity
                        scan[:, 1] *= scale_factor
                        print(f"Scaled scan {i+1} by factor {scale_factor:.4f}")
                    
                    # Append the scaled scan (excluding the overlap point)
                    if idx + 1 < len(scan):
                        refl = np.vstack((refl, scan[idx+1:]))
                
                all_scans.append(deepcopy(scan))
            
            # Store raw data before smoothing
            raw_refl_q = deepcopy(refl)
            
            # Apply smoothing if requested
            if smooth_data and savgol_window and len(refl) > savgol_window:
                try:
                    from scipy.signal import savgol_filter
                    
                    if savgol_window % 2 == 0:
                        savgol_window += 1  # Must be odd
                    
                    smoothed_intensities = savgol_filter(refl[:,1], savgol_window, savgol_order)
                    refl_smooth = deepcopy(refl)
                    refl_smooth[:,1] = smoothed_intensities
                    
                    print(f"Applied Savitzky-Golay smoothing: window={savgol_window}, order={savgol_order}")
                    
                except Exception as e:
                    print(f"Warning: Smoothing failed ({str(e)}), using raw data")
                    refl_smooth = deepcopy(refl)
            else:
                refl_smooth = deepcopy(refl)
            
            # Apply normalization using the new method
            if normalize:
                # Apply normalization (open beam or standard based on current setting)
                refl_smooth[:, 1] = self.apply_normalization(refl_smooth[:, 1], energy)
                raw_refl_q[:, 1] = self.apply_normalization(raw_refl_q[:, 1], energy)
                
                # Add error column (1% error assumption)
                refl_smooth = np.column_stack((refl_smooth, refl_smooth[:, 1] * 0.01))
                raw_refl_q = np.column_stack((raw_refl_q, raw_refl_q[:, 1] * 0.01))
            else:
                # Add error column without normalization
                refl_smooth = np.column_stack((refl_smooth, refl_smooth[:, 1] * 0.01))
                raw_refl_q = np.column_stack((raw_refl_q, raw_refl_q[:, 1] * 0.01))
            
            # Generate plots if requested
            if plot and plot_prefix:
                self._plot_processed_data_with_open_beam_info(all_scans, refl_smooth if smooth_data else raw_refl_q, 
                                                            backgrounds, energy, plot_prefix, output_dir)
            
            # Return processed data
            if smooth_data:
                return refl_smooth, raw_refl_q
            else:
                return raw_refl_q
        
        finally:
            # Restore original setting if override was used
            if use_open_beam is not None:
                self.use_open_beam_normalization = original_setting


    def _plot_processed_data_with_open_beam_info(self, all_scans, final_data, backgrounds, energy, 
                                            plot_prefix, output_dir=None):
        """
        Generate plots showing the effect of background subtraction and normalization type
        
        Parameters:
        -----------
        all_scans : list of numpy arrays
            Individual processed scans
        final_data : numpy array
            Final combined and processed data
        backgrounds : list of floats
            Background values that were subtracted
        energy : float
            Photon energy in eV
        plot_prefix : str
            Prefix for saved plot files
        output_dir : str, optional
            Directory to save plots
        """
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Individual scans (after background subtraction)
            for i, (scan, bg) in enumerate(zip(all_scans, backgrounds)):
                label = f'Scan {i+1}'
                if bg != 0.0:
                    label += f' (BG: -{bg:.3f})'
                ax1.plot(scan[:, 0], scan[:, 1], 'o-', alpha=0.7, markersize=3, label=label)
            
            ax1.set_xlabel('Angle (degrees)')
            ax1.set_ylabel('Intensity')
            ax1.set_title(f'Individual Scans (Background Corrected) - {energy:.1f} eV')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Plot 2: Final combined data
            use_open_beam = getattr(self, 'use_open_beam_normalization', False)
            norm_type = "Open Beam Normalized" if use_open_beam else "Standard Normalized"
            
            ax2.plot(final_data[:, 0], final_data[:, 1], 'b-', linewidth=2, label=f'Combined Data')
            ax2.set_xlabel('Angle (degrees)')
            ax2.set_ylabel('Normalized Intensity')
            ax2.set_title(f'Final Combined Data ({norm_type}) - {energy:.1f} eV')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Add normalization info to plot
            if use_open_beam and hasattr(self, 'open_beam_data') and self.open_beam_data is not None:
                open_beam_intensity = self.get_open_beam_intensity(energy)
                if open_beam_intensity:
                    ax2.text(0.05, 0.95, f'I₀({energy:.1f} eV) = {open_beam_intensity:.3e}', 
                            transform=ax2.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot if output directory specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plot_path = os.path.join(output_dir, f"{plot_prefix}_processing_comparison.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Processing comparison plot saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error generating plots: {str(e)}")


    def process_scan_set_with_open_beam_option(self, scan_group, output_filename="output.dat",
                                            normalize=True, plot=False, convert_to_photons=False,
                                            smooth_data=False, savgol_window=None, savgol_order=2,
                                            remove_zeros=True, estimate_thickness=False,
                                            min_prominence=0.1, min_thickness_nm=1.0, 
                                            max_thickness_nm=100.0, output_dir=None,
                                            plot_prefix=None, use_open_beam=None):
        """
        Enhanced version of process_scan_set that supports open beam normalization
        
        This method extends the existing process_scan_set functionality with open beam normalization.
        
        Parameters:
        -----------
        scan_group : dict
            Dictionary containing scan information with keys:
            - 'files': list of file paths
            - 'trims': list of (start, end) trim indices  
            - 'energy': photon energy in eV
            - 'backgrounds': list of background values (optional)
        output_filename : str
            Output filename for processed data
        normalize : bool
            Whether to normalize the reflectivity
        plot : bool
            Whether to generate plots
        convert_to_photons : bool
            Whether to convert to photon flux
        smooth_data : bool
            Whether to apply smoothing
        savgol_window : int, optional
            Smoothing window size
        savgol_order : int
            Smoothing polynomial order
        remove_zeros : bool
            Whether to remove zero data points
        estimate_thickness : bool
            Whether to estimate thickness
        min_prominence : float
            Minimum peak prominence for thickness estimation
        min_thickness_nm : float
            Minimum thickness for estimation
        max_thickness_nm : float
            Maximum thickness for estimation
        output_dir : str, optional
            Output directory
        plot_prefix : str, optional
            Plot filename prefix
        use_open_beam : bool, optional
            Override for open beam normalization
            
        Returns:
        --------
        Processed reflectivity data (numpy array or tuple if smoothed)
        """
        # Extract information from the scan group
        file_patterns = scan_group['files']
        trims = scan_group['trims']
        energy = scan_group['energy']
        
        # Extract background values if provided
        backgrounds = scan_group.get('backgrounds', [0.0] * len(file_patterns))
        
        # Ensure backgrounds list has the same length as files
        if len(backgrounds) < len(file_patterns):
            print(f"Warning: Not enough background values provided. Padding with zeros.")
            backgrounds.extend([0.0] * (len(file_patterns) - len(backgrounds)))
        elif len(backgrounds) > len(file_patterns):
            print(f"Warning: More background values than files. Truncating background list.")
            backgrounds = backgrounds[:len(file_patterns)]
        
        scans = []
        energies = []
        actual_trims = []
        actual_backgrounds = []
        
        # Load all data files
        for i, pattern in enumerate(file_patterns):
            try:
                scan_data = self.load_data_file(pattern)
                scans.append(scan_data)
                
                # Use the trim value for this pattern
                if i < len(trims):
                    actual_trims.append(trims[i])
                else:
                    # Default trim if not provided
                    actual_trims.append((0, -1))
                
                # Use the background value for this pattern
                actual_backgrounds.append(backgrounds[i])
                
                energies.append(energy)
                print(f"Loaded file: {pattern}")
                if backgrounds[i] != 0.0:
                    print(f"  Background subtraction: -{backgrounds[i]:.3f}")
            except Exception as e:
                print(f"Error loading file {pattern}: {str(e)}")
        
        if not scans:
            print("No valid scan data found.")
            return None
        
        # Use the enhanced reduce_data method with open beam option
        result = self.reduce_data_with_open_beam_option(
            scans=scans,
            trims=actual_trims,
            backgrounds=actual_backgrounds,
            energy=energy,
            normalize=normalize,
            plot=plot,
            convert_to_photons=convert_to_photons,
            output_dir=output_dir,
            plot_prefix=plot_prefix,
            smooth_data=smooth_data,
            savgol_window=savgol_window,
            savgol_order=savgol_order,
            remove_zeros=remove_zeros,
            use_open_beam=use_open_beam
        )
        
        if result is None:
            return None
        
        # Handle saving and metadata (similar to existing process_scan_set)
        if output_dir or output_filename != "output.dat":
            output_path = os.path.join(output_dir or '.', output_filename)
            
            try:
                if smooth_data and isinstance(result, tuple):
                    # Save smoothed data
                    np.savetxt(output_path, result[0], delimiter='\t',
                            header='Angle(deg)\tIntensity\tError', comments='')
                    print(f"Saved smoothed data to {output_path}")
                else:
                    # Save regular data  
                    data_to_save = result[0] if isinstance(result, tuple) else result
                    np.savetxt(output_path, data_to_save, delimiter='\t',
                            header='Angle(deg)\tIntensity\tError', comments='')
                    print(f"Saved processed data to {output_path}")
                    
                # Save metadata including open beam info
                self._save_metadata_with_open_beam_info(scan_group, output_path, normalize, 
                                                    convert_to_photons, smooth_data, remove_zeros,
                                                    savgol_window, savgol_order, actual_backgrounds)
                                                    
            except Exception as e:
                print(f"Error saving data: {str(e)}")
        
        # Estimate thickness if requested (using existing method)
        if estimate_thickness and hasattr(self, 'estimate_thickness'):
            try:
                data_for_thickness = result[0] if isinstance(result, tuple) else result
                thickness = self.estimate_thickness(data_for_thickness,
                                                min_prominence=min_prominence,
                                                min_thickness_nm=min_thickness_nm,
                                                max_thickness_nm=max_thickness_nm)
                if thickness is not None:
                    print(f"Estimated thickness: {thickness:.1f} Å")
            except Exception as e:
                print(f"Error estimating thickness: {str(e)}")
        
        return result


    def _save_metadata_with_open_beam_info(self, scan_group, output_path, normalize, 
                                        convert_to_photons, smooth_data, remove_zeros,
                                        savgol_window, savgol_order, backgrounds):
        """
        Save metadata including open beam normalization information
        """
        try:
            sample_name = scan_group.get('sample_name', 'Unknown')
            energy = scan_group['energy']
            
            # Create metadata with open beam info
            metadata = {
                'Sample Name': sample_name,
                'Energy (eV)': energy,
                'Files Processed': len(scan_group['files']),
                'Normalization': normalize,
                'Normalization Type': 'Open Beam' if getattr(self, 'use_open_beam_normalization', False) else 'Standard',
                'Photon Conversion': convert_to_photons,
                'Smoothing': smooth_data,
                'Zero Removal': remove_zeros,
                'Processing Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Output File': os.path.basename(output_path),
                'Background Subtraction Applied': any(bg != 0.0 for bg in backgrounds)
            }
            
            # Add open beam specific info
            if getattr(self, 'use_open_beam_normalization', False) and hasattr(self, 'open_beam_data'):
                open_beam_intensity = self.get_open_beam_intensity(energy)
                metadata['Open Beam File'] = getattr(self, 'open_beam_file', 'Unknown')
                metadata['Open Beam Intensity at Energy'] = open_beam_intensity
                energy_range = f"{self.open_beam_data['energy'].min():.1f}-{self.open_beam_data['energy'].max():.1f}"
                metadata['Open Beam Energy Range'] = energy_range
            
            # Add smoothing parameters if used
            if smooth_data:
                metadata['Smoothing Window'] = savgol_window
                metadata['Smoothing Order'] = savgol_order
            
            # Add background details
            for i, bg in enumerate(backgrounds):
                metadata[f'Background File {i+1}'] = bg
            
            # Save metadata
            meta_path = output_path.replace('.dat', '_metadata.csv')
            meta_df = pd.DataFrame([metadata])
            meta_df.to_csv(meta_path, index=False)
            print(f"Metadata saved to {meta_path}")
            
        except Exception as e:
            print(f"Warning: Could not save metadata: {str(e)}")
