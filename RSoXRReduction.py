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
    
    def reduce_data(self, scans, trims, energy, normalize=True, plot=True, convert_to_photons=False, output_dir=None, plot_prefix=None):
        """
        Reduce multiple scans by scaling them to the lowest angle scan
        
        Parameters:
        -----------
        scans : list of numpy arrays
            List of data arrays from each scan
        trims : list of tuples
            List of (start_idx, end_idx) for each scan
        energy : float
            Beam energy in eV
        normalize : bool
            Whether to normalize the final reflectivity
        plot : bool
            Whether to generate plots during processing
        convert_to_photons : bool
            Whether to convert photodiode current to photon flux
        output_dir : str, optional
            Directory to save plots if provided
        plot_prefix : str, optional
            Prefix for plot filenames
            
        Returns:
        --------
        ReflQ : numpy array
            Processed reflectivity data with columns [Q, R, error]
        """
        # Ensure we have trim values for all scans
        if len(trims) < len(scans):
            print(f"Warning: Not enough trim values provided. Adding default trims for {len(scans) - len(trims)} scans.")
            trims.extend([(0, -1)] * (len(scans) - len(trims)))
            
        # Process the first scan
        refl = deepcopy(scans[0][trims[0][0]:(len(scans[0][:,0])+trims[0][1]), 0:2])
        
        # Convert to photon flux if requested
        if convert_to_photons and self.calibration_data is not None:
            # Convert current to photon flux
            refl[:,1] = self.amp_to_photon_flux(refl[:,1], energy)
        
        # Store all raw scans for combined plotting
        all_scans = [deepcopy(refl)]
        
        # Process additional scans
        for i in range(1, len(scans)):
            scan = scans[i][trims[i][0]:(len(scans[i][:,0])+trims[i][1]), 0:2]
            
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
        
        # Convert to Q (momentum transfer)
        wavelength_nm = 1239.9 / energy  # eV to nm
        wavelength_angstrom = wavelength_nm * 10  # nm to Å
        Q = 4 * np.pi * np.sin(np.radians(refl[:,0])) / wavelength_angstrom  # Q in Å^-1
        
        # Create output array
        refl_q = np.zeros([Q.size, 3])
        refl_q[:,0] = Q
        
        if normalize:
            max_intensity = refl[:,1].max()
            refl_q[:,1] = refl[:,1] / max_intensity
            refl_q[:,2] = (refl[:,1] / max_intensity) * 0.01  # 1% error
        else:
            refl_q[:,1] = refl[:,1]
            refl_q[:,2] = refl[:,1] * 0.01  # 1% error
        
        # Sort by Q
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
            ax2.errorbar(refl_q[:,0], refl_q[:,1], yerr=refl_q[:,2], fmt='rx-', 
                        markersize=4, capsize=3, label='Reduced Data')
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
        
        return refl_q
    
    def process_scan_set(self, scan_group, output_filename=None, normalize=True, plot=True, 
                  convert_to_photons=False, output_dir=None, plot_prefix=None, 
                  save_metadata=True, estimate_thickness=True, min_prominence=0.1,
                  min_thickness_nm=20, max_thickness_nm=100, smooth_data=True):
        """
        Process a scan set and combine them
        
        Parameters:
        -----------
        scan_group : dict
            Scan group dictionary from auto_group_scans
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
            Whether to apply smoothing to the data before peak detection
        
        Returns:
        --------
        ReflQ : numpy array
            Processed reflectivity data
        """
        # Extract information from the scan group
        file_patterns = scan_group['files']
        trims = scan_group['trims']
        energy = scan_group['energy']
        
        scans = []
        energies = []
        actual_trims = []
        
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
                
                energies.append(energy)
                print(f"Loaded file: {pattern}")
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
        
        # Process the scans
        refl_q = self.reduce_data(
            scans, 
            actual_trims, 
            energy, 
            normalize=normalize, 
            plot=plot,
            convert_to_photons=convert_to_photons,
            output_dir=output_dir,
            plot_prefix=plot_prefix
        )
        
        # Save to file if requested
        if output_filename and refl_q is not None:
            # Create output directory if it doesn't exist
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)
            else:
                output_path = output_filename
                
            np.savetxt(output_path, refl_q)
            print(f"Saved processed data to {output_path}")
            
            # Save metadata if requested
            if save_metadata:
                self.save_reduced_data_metadata(
                    scan_group, 
                    refl_q, 
                    output_dir=output_dir
                )
        
        # Estimate film thickness if requested
        thickness = None
        if estimate_thickness and refl_q is not None:
            print("\nEstimating film thickness from fringe spacing...")
            thickness, peaks, valleys = self.estimate_film_thickness(
                refl_q, 
                min_prominence=min_prominence,
                min_thickness_nm=min_thickness_nm,
                max_thickness_nm=max_thickness_nm,
                plot=plot,
                output_dir=output_dir,
                plot_prefix=plot_prefix,
                smooth_data=smooth_data
            )
            
            # Update metadata with thickness if available
            if save_metadata and thickness is not None:
                # Update the saved metadata file to include thickness
                log_file_base = "reduced_data"
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
        
        return refl_q

    
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
    
    def auto_group_scans(self, data_directory=".", position_tolerance=0.1, energy_tolerance=0.5, auto_trim=False, save_table=True, output_dir=None):
        """
        Automatically group scans based on position, energy, and detector type
        
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
            
        Returns:
        --------
        groups : list of dicts
            List of file groups with associated metadata
        """
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
        
        # Create and display a consolidated table of scan groups
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
        
        # Print details for each group
        for i, group in enumerate(scan_groups):
            print(f"\nGroup {i+1}: {group['sample_name']} ({group['energy']:.1f} eV)")
            print("-" * 120)
            print(f"{'#':3} | {'Filename':20} | {'Detector':12} | {'Angle Range':15} | {'Count Time':10} | {'Trim'}")
            print("-" * 120)
            
            for j, filename in enumerate(group['files']):
                base_filename = os.path.basename(filename)
                # Find metadata for this file
                file_meta = None
                for meta in group['metadata']:
                    if os.path.basename(meta['filename']) == base_filename:
                        file_meta = meta
                        break
                
                if file_meta:
                    trim_str = f"({group['trims'][j][0]}, {group['trims'][j][1]})"
                    detector = file_meta['detector'] if file_meta['detector'] else "Unknown"
                    
                    # Format angle range
                    angle_range_str = "N/A"
                    if file_meta['min_angle'] is not None and file_meta['max_angle'] is not None:
                        angle_range_str = f"{file_meta['min_angle']:.2f} - {file_meta['max_angle']:.2f}°"
                    
                    # Add count time if available (mainly for CEM detectors)
                    count_time = "N/A"
                    if file_meta.get('count_time') is not None:
                        count_time = f"{file_meta['count_time']:.6f}"
                    
                    print(f"{j+1:3} | {base_filename:20} | {detector:12} | {angle_range_str:15} | {count_time:10} | {trim_str}")
                else:
                    print(f"{j+1:3} | {base_filename:20} | {'Unknown':12} | {'N/A':15} | {'N/A':10} | {group['trims'][j]}")
            
            print()
            
        if save_table and scan_groups:
            self.save_scan_groups_to_table(scan_groups, output_dir=output_dir)
    
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
                      estimate_thickness=True, min_prominence=0.1):
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
                min_prominence=min_prominence
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



    def estimate_film_thickness(self, reflectivity_data, min_prominence=0.1, plot=True, output_dir=None, plot_prefix=None):
        """
        Estimate the film thickness based on the fringe spacing in Q-space
        
        Parameters:
        -----------
        reflectivity_data : numpy array
            Reduced reflectivity data with columns [Q, R, error]
        min_prominence : float
            Minimum prominence for peak/valley detection
        plot : bool
            Whether to generate plots during processing
        output_dir : str, optional
            Directory to save plots if provided
        plot_prefix : str, optional
            Prefix for plot filenames
            
        Returns:
        --------
        thickness : float
            Estimated film thickness in Ångstroms
        peak_positions : array
            Positions of detected peaks in Q-space
        valley_positions : array
            Positions of detected valleys in Q-space
        """
        import numpy as np
        from scipy.signal import find_peaks, peak_prominences
        import matplotlib.pyplot as plt
        import os
        
        # Extract Q and reflectivity
        Q = reflectivity_data[:, 0]
        R = reflectivity_data[:, 1]
        
        # Take log of reflectivity
        log_R = np.log10(R)
        
        # Find the valleys (minima) in the reflectivity
        # For valleys, we'll find peaks in the negative of log_R
        valley_indices, _ = find_peaks(-log_R, prominence=min_prominence)
        valley_positions = Q[valley_indices]
        valley_values = log_R[valley_indices]
        
        # Find the peaks (maxima) in the reflectivity
        peak_indices, _ = find_peaks(log_R, prominence=min_prominence)
        peak_positions = Q[peak_indices]
        peak_values = log_R[peak_indices]
        
        # Calculate the average spacing between adjacent minima
        if len(valley_positions) >= 2:
            valley_spacings = np.diff(valley_positions)
            avg_valley_spacing = np.mean(valley_spacings)
            valley_thickness = 2 * np.pi / avg_valley_spacing
        else:
            avg_valley_spacing = None
            valley_thickness = None
        
        # Calculate the average spacing between adjacent maxima
        if len(peak_positions) >= 2:
            peak_spacings = np.diff(peak_positions)
            avg_peak_spacing = np.mean(peak_spacings)
            peak_thickness = 2 * np.pi / avg_peak_spacing
        else:
            avg_peak_spacing = None
            peak_thickness = None
        
        # Combine the results
        thicknesses = []
        if valley_thickness is not None:
            thicknesses.append(valley_thickness)
        if peak_thickness is not None:
            thicknesses.append(peak_thickness)
        
        # Calculate final thickness estimate
        if thicknesses:
            thickness = np.mean(thicknesses)
        else:
            thickness = None
        
        # Print results
        print("\nFilm Thickness Estimation Results:")
        print("-" * 40)
        
        if valley_thickness is not None:
            print(f"Valley-based estimate: {valley_thickness:.1f} Å")
            print(f"  Average spacing between valleys: {avg_valley_spacing:.4f} Å⁻¹")
            print(f"  Number of valleys detected: {len(valley_positions)}")
        else:
            print("Not enough valleys detected for thickness estimation")
        
        if peak_thickness is not None:
            print(f"Peak-based estimate: {peak_thickness:.1f} Å")
            print(f"  Average spacing between peaks: {avg_peak_spacing:.4f} Å⁻¹")
            print(f"  Number of peaks detected: {len(peak_positions)}")
        else:
            print("Not enough peaks detected for thickness estimation")
        
        if thickness is not None:
            print(f"\nFinal thickness estimate: {thickness:.1f} Å")
        else:
            print("\nCould not estimate thickness - insufficient peaks/valleys detected")
            print("Try adjusting the min_prominence parameter")
        
        # Create plot if requested
        if plot and (valley_positions.size > 0 or peak_positions.size > 0):
            plt.figure(figsize=(10, 6))
            
            # Plot the reflectivity data
            plt.plot(Q, log_R, 'b-', label='log(Reflectivity)')
            
            # Mark the valleys
            if valley_positions.size > 0:
                plt.plot(valley_positions, valley_values, 'rv', markersize=8, label='Valleys')
                
                # Annotate the valley spacing
                for i in range(len(valley_positions)-1):
                    spacing = valley_positions[i+1] - valley_positions[i]
                    midpoint = (valley_positions[i] + valley_positions[i+1]) / 2
                    mid_height = np.interp(midpoint, Q, log_R)
                    plt.annotate(f"{spacing:.4f}", 
                               xy=(midpoint, mid_height),
                               xytext=(0, -20),
                               textcoords='offset points',
                               ha='center',
                               arrowprops=dict(arrowstyle='->'))
            
            # Mark the peaks
            if peak_positions.size > 0:
                plt.plot(peak_positions, peak_values, 'go', markersize=8, label='Peaks')
                
                # Annotate the peak spacing
                for i in range(len(peak_positions)-1):
                    spacing = peak_positions[i+1] - peak_positions[i]
                    midpoint = (peak_positions[i] + peak_positions[i+1]) / 2
                    mid_height = np.interp(midpoint, Q, log_R)
                    plt.annotate(f"{spacing:.4f}", 
                               xy=(midpoint, mid_height),
                               xytext=(0, 20),
                               textcoords='offset points',
                               ha='center',
                               arrowprops=dict(arrowstyle='->'))
            
            # Add thickness annotation
            if thickness is not None:
                title = f"Film Thickness Estimation: {thickness:.1f} Å"
                if valley_thickness is not None and peak_thickness is not None:
                    title += f" (P: {peak_thickness:.1f} Å, V: {valley_thickness:.1f} Å)"
            else:
                title = "Film Thickness Estimation: Insufficient Data"
                
            plt.title(title)
            plt.xlabel('Q (Å⁻¹)')
            plt.ylabel('log(Reflectivity)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save plot if output directory is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                if plot_prefix is None:
                    plot_prefix = "thickness_estimate"
                plot_filename = f"{plot_prefix}_thickness.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_path, dpi=150)
                print(f"\nSaved thickness estimation plot to {plot_path}")
                
            plt.tight_layout()
            plt.show()
        
        # Return the results
        return thickness, peak_positions, valley_positions

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

    def estimate_film_thickness(self, reflectivity_data, min_prominence=0.1, min_spacing=0.01, 
                              max_spacing=0.3, min_thickness_nm=20, max_thickness_nm=100, 
                              plot=True, output_dir=None, plot_prefix=None, smooth_data=True):
        """
        Estimate the film thickness based on the fringe spacing in Q-space
        
        Parameters:
        -----------
        reflectivity_data : numpy array
            Reduced reflectivity data with columns [Q, R, error]
        min_prominence : float
            Minimum prominence for peak/valley detection
        min_spacing : float
            Minimum spacing between adjacent peaks/valleys in Q-space (Å⁻¹)
            For a 100nm film, fringe spacing is ~0.031 Å⁻¹
        max_spacing : float
            Maximum spacing between adjacent peaks/valleys in Q-space (Å⁻¹)
            For a 20nm film, fringe spacing is ~0.157 Å⁻¹
        min_thickness_nm : float
            Minimum expected film thickness in nm
        max_thickness_nm : float
            Maximum expected film thickness in nm
        plot : bool
            Whether to generate plots during processing
        output_dir : str, optional
            Directory to save plots if provided
        plot_prefix : str, optional
            Prefix for plot filenames
        smooth_data : bool
            Whether to apply smoothing to the data before peak detection
            
        Returns:
        --------
        thickness : float
            Estimated film thickness in Ångstroms
        peak_positions : array
            Positions of detected peaks in Q-space
        valley_positions : array
            Positions of detected valleys in Q-space
        """
        import numpy as np
        from scipy.signal import find_peaks, peak_prominences, savgol_filter
        import matplotlib.pyplot as plt
        import os
        
        # Convert thickness range to expected fringe spacing range
        min_fringe_spacing = 2 * np.pi / (max_thickness_nm * 10)  # max thickness -> min spacing
        max_fringe_spacing = 2 * np.pi / (min_thickness_nm * 10)  # min thickness -> max spacing
        
        print(f"Expected fringe spacing range: {min_fringe_spacing:.4f} - {max_fringe_spacing:.4f} Å⁻¹")
        print(f"Based on thickness range: {min_thickness_nm} - {max_thickness_nm} nm")
        
        # Use user-specified spacing if provided
        if min_spacing > 0:
            min_fringe_spacing = min_spacing
        if max_spacing > 0:
            max_fringe_spacing = max_spacing
        
        # Extract Q and reflectivity
        Q = reflectivity_data[:, 0]
        R = reflectivity_data[:, 1]
        
        # Sort data by Q values (just to be safe)
        sort_indices = np.argsort(Q)
        Q = Q[sort_indices]
        R = R[sort_indices]
        
        # Apply smoothing if requested
        if smooth_data:
            # Determine the best window size for Savitzky-Golay filter (must be odd)
            window_size = max(min(25, len(Q) // 10 * 2 + 1), 5)
            if window_size % 2 == 0:
                window_size += 1
                
            # Apply Savitzky-Golay filter to reduce noise
            try:
                log_R_smooth = savgol_filter(np.log10(R), window_size, 2)
            except Exception as e:
                print(f"Warning: Smoothing failed ({str(e)}), using raw data")
                log_R_smooth = np.log10(R)
        else:
            log_R_smooth = np.log10(R)
        
        # Find the valleys (minima) in the reflectivity
        # For valleys, we'll find peaks in the negative of log_R
        valley_indices, valley_props = find_peaks(-log_R_smooth, 
                                                prominence=min_prominence,
                                                distance=int(min_fringe_spacing / (Q[1] - Q[0])))
        
        if len(valley_indices) > 0:
            valley_positions = Q[valley_indices]
            valley_values = log_R_smooth[valley_indices]
            valley_prominences = peak_prominences(-log_R_smooth, valley_indices)[0]
        else:
            valley_positions = np.array([])
            valley_values = np.array([])
            valley_prominences = np.array([])
        
        # Find the peaks (maxima) in the reflectivity
        peak_indices, peak_props = find_peaks(log_R_smooth, 
                                             prominence=min_prominence,
                                             distance=int(min_fringe_spacing / (Q[1] - Q[0])))
        
        if len(peak_indices) > 0:
            peak_positions = Q[peak_indices]
            peak_values = log_R_smooth[peak_indices]
            peak_prominences = peak_prominences(log_R_smooth, peak_indices)[0]
        else:
            peak_positions = np.array([])
            peak_values = np.array([])
            peak_prominences = np.array([])
        
        # Filter out peak/valley pairs that are too close or too far apart
        valid_valley_spacings = []
        valid_valley_positions = []
        
        if len(valley_positions) >= 2:
            valley_spacings = np.diff(valley_positions)
            
            # Filter valid spacings
            valid_indices = np.where((valley_spacings >= min_fringe_spacing) & 
                                    (valley_spacings <= max_fringe_spacing))[0]
            
            # Get valid spacings
            if len(valid_indices) > 0:
                for idx in valid_indices:
                    valid_valley_spacings.append(valley_spacings[idx])
                    valid_valley_positions.append(valley_positions[idx:idx+2])
        
        # Calculate valley-based thickness
        if valid_valley_spacings:
            avg_valley_spacing = np.mean(valid_valley_spacings)
            valley_thickness = 2 * np.pi / avg_valley_spacing
        else:
            avg_valley_spacing = None
            valley_thickness = None
        
        # Similarly for peaks
        valid_peak_spacings = []
        valid_peak_positions = []
        
        if len(peak_positions) >= 2:
            peak_spacings = np.diff(peak_positions)
            
            # Filter valid spacings
            valid_indices = np.where((peak_spacings >= min_fringe_spacing) & 
                                   (peak_spacings <= max_fringe_spacing))[0]
            
            # Get valid spacings
            if len(valid_indices) > 0:
                for idx in valid_indices:
                    valid_peak_spacings.append(peak_spacings[idx])
                    valid_peak_positions.append(peak_positions[idx:idx+2])
        
        # Calculate peak-based thickness
        if valid_peak_spacings:
            avg_peak_spacing = np.mean(valid_peak_spacings)
            peak_thickness = 2 * np.pi / avg_peak_spacing
        else:
            avg_peak_spacing = None
            peak_thickness = None
        
        # Combine the results
        thicknesses = []
        if valley_thickness is not None:
            thicknesses.append(valley_thickness)
        if peak_thickness is not None:
            thicknesses.append(peak_thickness)
        
        # Calculate final thickness estimate
        if thicknesses:
            thickness = np.mean(thicknesses)
            thickness_nm = thickness / 10  # Convert Å to nm
        else:
            thickness = None
            thickness_nm = None
        
        # Print results
        print("\nFilm Thickness Estimation Results:")
        print("-" * 40)
        
        if valley_thickness is not None:
            print(f"Valley-based estimate: {valley_thickness:.1f} Å ({valley_thickness/10:.1f} nm)")
            print(f"  Average spacing between valleys: {avg_valley_spacing:.4f} Å⁻¹")
            print(f"  Number of valid valley spacings: {len(valid_valley_spacings)}")
        else:
            print("No valid valley spacings detected for thickness estimation")
        
        if peak_thickness is not None:
            print(f"Peak-based estimate: {peak_thickness:.1f} Å ({peak_thickness/10:.1f} nm)")
            print(f"  Average spacing between peaks: {avg_peak_spacing:.4f} Å⁻¹")
            print(f"  Number of valid peak spacings: {len(valid_peak_spacings)}")
        else:
            print("No valid peak spacings detected for thickness estimation")
        
        if thickness is not None:
            print(f"\nFinal thickness estimate: {thickness:.1f} Å ({thickness_nm:.1f} nm)")
        else:
            print("\nCould not estimate thickness - insufficient valid peaks/valleys detected")
            print(f"Try adjusting the spacing range ({min_fringe_spacing:.4f} - {max_fringe_spacing:.4f} Å⁻¹)")
            print(f"Or min_prominence parameter (currently {min_prominence})")
        
        # Create plot if requested
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Plot the reflectivity data
            plt.plot(Q, log_R_smooth, 'b-', label='log(Reflectivity) [Smoothed]')
            if smooth_data:
                plt.plot(Q, np.log10(R), 'b-', alpha=0.3, label='log(Reflectivity) [Raw]')
            
            # Mark all detected valleys
            if len(valley_positions) > 0:
                plt.plot(valley_positions, valley_values, 'rv', markersize=8, alpha=0.5, label='All Valleys')
            
            # Mark all detected peaks
            if len(peak_positions) > 0:
                plt.plot(peak_positions, peak_values, 'go', markersize=8, alpha=0.5, label='All Peaks')
            
            # Mark valid valley pairs
            for pos_pair in valid_valley_positions:
                idx1 = np.where(valley_positions == pos_pair[0])[0][0]
                idx2 = np.where(valley_positions == pos_pair[1])[0][0]
                
                spacing = pos_pair[1] - pos_pair[0]
                midpoint = (pos_pair[0] + pos_pair[1]) / 2
                
                plt.plot(pos_pair, [valley_values[idx1], valley_values[idx2]], 'r-', linewidth=2)
                plt.annotate(f"{spacing:.4f}", 
                           xy=(midpoint, np.interp(midpoint, Q, log_R_smooth)),
                           xytext=(0, -20),
                           textcoords='offset points',
                           ha='center',
                           color='red',
                           arrowprops=dict(arrowstyle='->', color='red'))
            
            # Mark valid peak pairs
            for pos_pair in valid_peak_positions:
                idx1 = np.where(peak_positions == pos_pair[0])[0][0]
                idx2 = np.where(peak_positions == pos_pair[1])[0][0]
                
                spacing = pos_pair[1] - pos_pair[0]
                midpoint = (pos_pair[0] + pos_pair[1]) / 2
                
                plt.plot(pos_pair, [peak_values[idx1], peak_values[idx2]], 'g-', linewidth=2)
                plt.annotate(f"{spacing:.4f}", 
                           xy=(midpoint, np.interp(midpoint, Q, log_R_smooth)),
                           xytext=(0, 20),
                           textcoords='offset points',
                           ha='center',
                           color='green',
                           arrowprops=dict(arrowstyle='->', color='green'))
            
            # Add thickness annotation
            if thickness is not None:
                title = f"Film Thickness Estimation: {thickness:.1f} Å ({thickness_nm:.1f} nm)"
                if valley_thickness is not None and peak_thickness is not None:
                    title += f"\nPeak-based: {peak_thickness:.1f} Å, Valley-based: {valley_thickness:.1f} Å"
            else:
                title = "Film Thickness Estimation: Insufficient Valid Data"
                
            plt.title(title)
            plt.xlabel('Q (Å⁻¹)')
            plt.ylabel('log(Reflectivity)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add the expected fringe spacing range as vertical lines
            min_thickness_spacing = 2 * np.pi / (max_thickness_nm * 10)
            max_thickness_spacing = 2 * np.pi / (min_thickness_nm * 10)
            
            # Add a text box with parameter settings
            textbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            info_text = (
                f"Parameters:\n"
                f"Min prominence: {min_prominence}\n"
                f"Q spacing range: {min_fringe_spacing:.4f} - {max_fringe_spacing:.4f} Å⁻¹\n"
                f"Expected thickness: {min_thickness_nm} - {max_thickness_nm} nm"
            )
            plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                    bbox=textbox_props)
            
            # Save plot if output directory is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                if plot_prefix is None:
                    plot_prefix = "thickness_estimate"
                plot_filename = f"{plot_prefix}_thickness.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_path, dpi=150)
                print(f"\nSaved thickness estimation plot to {plot_path}")
                
            plt.tight_layout()
            plt.show()
        
        # Return the results
        return thickness, peak_positions, valley_positions