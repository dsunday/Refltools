
import sys

sys.path.append('/homes/dfs1/nist_cdsaxs/src/cdsaxs/Fitting/')
import CDSAXSFunctions as CD

import numpy as np
import os
from typing import Optional, Union

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Dict

def loadSAXS(filename: str, 
             keep_third_column: bool = False, 
             plot_data: bool = False,
             log_scale: bool = True,
             figsize: Tuple[int, int] = (10, 6),
             title: Optional[str] = None,
             x_label: str = 'q (Å$^{-1}$)',
             y_label: str = 'Intensity (a.u)',
             marker: str = 'o',
             markersize: int = 4,
             linestyle: str = '-',
             color: str = 'b',
             legend_label: Optional[str] = None,
             grid: bool = True) -> np.ndarray:
    """
    Load SAXS data from a CSV file and optionally plot it.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file containing SAXS data
    keep_third_column : bool, optional
        Whether to keep a third column if it exists in the data (default: False)
    plot_data : bool, optional
        Whether to plot the loaded data (default: False)
    log_scale : bool, optional
        Whether to use log scale for the y-axis (default: True)
    figsize : tuple of int, optional
        Figure size as (width, height) in inches (default: (10, 6))
    title : str, optional
        Custom title for the plot (default: filename)
    x_label : str, optional
        Label for the x-axis (default: 'q (Å^-1)')
    y_label : str, optional
        Label for the y-axis (default: 'Intensity (a.u)')
    marker : str, optional
        Marker style for the plot (default: 'o')
    markersize : int, optional
        Size of markers (default: 4)
    linestyle : str, optional
        Line style for the plot (default: '-')
    color : str, optional
        Color for the plot (default: 'b')
    legend_label : str, optional
        Label for the legend (default: derived from filename)
    grid : bool, optional
        Whether to show grid on the plot (default: True)
        
    Returns:
    --------
    data : np.ndarray
        Array containing the loaded data:
        - If keep_third_column=False: Returns [Q, Intensity] array
        - If keep_third_column=True and a third column exists: Returns [Q, Intensity, ThirdColumn] array
        - If keep_third_column=True but no third column exists: Returns [Q, Intensity] array
        
    Raises:
    -------
    FileNotFoundError
        If the specified file doesn't exist
    ValueError
        If the file format is not as expected
    """
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    try:
        # Load the data, skipping the first row (header)
        data_raw = np.loadtxt(filename, skiprows=1, delimiter=',')
        
        # Check if data is empty
        if data_raw.size == 0:
            raise ValueError(f"No data found in file: {filename}")
        
        # Get number of columns in the raw data
        num_cols = data_raw.shape[1]
        
        # Check that we have at least 2 columns
        if num_cols < 2:
            raise ValueError(f"Data must have at least 2 columns, but found {num_cols} columns")
        
        # Handle the third column based on user preference
        if num_cols > 2 and keep_third_column:
            # If there are more than 2 columns and user wants to keep third column
            if num_cols > 3:
                # If more than 3 columns, only keep the first 3
                print(f"Warning: File has {num_cols} columns. Only keeping columns 1-3.")
                data = data_raw[:, :3]
            else:
                # Return all columns (first 3)
                data = data_raw
        else:
            # Only keep first 2 columns
            data = data_raw[:, :2]
        
        # Plot the data if requested
        if plot_data:
            plt.figure(figsize=figsize)
            
            # Set default legend label based on filename if not provided
            if legend_label is None:
                legend_label = os.path.splitext(os.path.basename(filename))[0]
            
            # Plot the data
            plt.plot(data[:, 0], data[:, 1], 
                     marker=marker, markersize=markersize, 
                     linestyle=linestyle, color=color,
                     label=legend_label)
            
            # Set log scale for y-axis if requested
            if log_scale:
                plt.yscale('log')
            
            # Add grid if requested
            if grid:
                plt.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # Set labels
            plt.xlabel(x_label, fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            
            # Set title (use filename if not provided)
            if title is None:
                title = os.path.splitext(os.path.basename(filename))[0]
            plt.title(title, fontsize=16)
            
            # Add legend
            plt.legend(loc='best')
            
            # Show the plot
            plt.tight_layout()
            plt.show()
        
        return data
            
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Error loading data from {filename}: {str(e)}")
        

def apply_power_law_scaling_and_plot(data, q_min, q_max, power, normalize_to_one=True):
    """
    Apply power law scaling to intensity data within a specified Q range,
    apply scalar scaling above q_max for smooth stitching,
    and optionally normalize the final result so the minimum intensity is 1.
    
    Parameters:
    -----------
    data : array
        Array with two columns [Q, Intensity]
    q_min : float
        Minimum Q value for power law scaling
    q_max : float
        Maximum Q value for power law scaling
    power : float
        Power law exponent to use for scaling
    normalize_to_one : bool, optional
        If True, normalize the final result so minimum intensity is 1
        
    Returns:
    --------
    scaled_data : array
        Array with two columns [Q, Scaled_Intensity]
    """
    # Extract Q and Intensity columns
    q_data = data[:, 0]
    i_data = data[:, 1]
    
    # Create masks for different Q regions
    mask_middle = (q_data >= q_min) & (q_data <= q_max)
    mask_high = q_data > q_max
    
    # Initialize scaled intensity as a copy of the original
    i_scaled = i_data.copy()
    
    # Apply power law scaling within the specified range
    scaling_factor_middle = q_data[mask_middle] ** power
    i_scaled[mask_middle] = i_data[mask_middle] * scaling_factor_middle
    
    # Calculate scalar factor for high Q region to ensure smooth stitching
    scaling_factor_high = None
    if np.any(mask_high):
        # Find indices at the boundary
        idx_max = np.where(q_data <= q_max)[0][-1]
        
        # Calculate ratio at the boundary point to ensure continuity
        if idx_max + 1 < len(q_data):  # Make sure we're not at the end of the array
            original_ratio = i_data[idx_max + 1] / i_scaled[idx_max]
            scaling_factor_high = original_ratio
            
            # Apply scalar scaling to high Q region
            i_scaled[mask_high] = i_data[mask_high] / scaling_factor_high
            
            print(f"Applied scalar factor of {scaling_factor_high:.4f} for Q > {q_max}")
    
    # Normalize to minimum value of 1 if requested
    if normalize_to_one:
        min_intensity = np.min(i_scaled)
        if min_intensity > 0:  # Ensure we don't divide by zero
            i_scaled = i_scaled / min_intensity
            print(f"Normalized by dividing by minimum intensity: {min_intensity:.6e}")
        else:
            print("Warning: Minimum intensity is zero or negative, skipping normalization to one.")
    
    # Create output array
    scaled_data = np.column_stack((q_data, i_scaled))
    
    # Plotting
    plt.figure(figsize=(12, 12))
    
    # Plot original data
    plt.subplot(2, 1, 1)
    plt.loglog(q_data, i_data, 'b-', label='Original I(q)')
    plt.axvspan(q_min, q_max, alpha=0.2, color='gray', label=f'Power law region (q^{power:.2f})')
    plt.axvspan(q_max, q_data.max(), alpha=0.1, color='red', label=f'Scalar scaling region')
    plt.xlabel('q')
    plt.ylabel('I(q)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.title('X-ray Scattering Data - Original')
    
    # Plot scaled data with normalization information
    plt.subplot(2, 1, 2)
    plt.loglog(q_data, i_scaled, 'r-', label=f'Scaled I(q)')
    plt.axvspan(q_min, q_max, alpha=0.2, color='gray', label=f'Power law region (q^{power:.2f})')
    plt.axvspan(q_max, q_data.max(), alpha=0.1, color='red', label=f'Scalar scaling region')
    plt.axhline(y=1, color='k', linestyle='--', label='Baseline (I=1)')
    plt.xlabel('q')
    plt.ylabel('Normalized and Scaled I(q)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.title('X-ray Scattering Data - After Scaling and Normalization')
    
    plt.show()
    plt.close()
    
    return scaled_data

import numpy as np
from scipy.signal import find_peaks

import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional, Union, Callable

import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional, Union, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional, Union, Callable

def find_peaks_by_sections(data: np.ndarray, 
                          sections: List[Dict],
                          integration_method: str = 'sum',
                          plot_results: bool = True,
                          figsize: Tuple[int, int] = (12, 10),
                          log_scale: bool = True,
                          show_background: bool = True,
                          show_peak_contributions: bool = True,
                          show_table: bool = True,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> Tuple[Dict, Optional[Tuple]]:
    """
    Find peaks in different sections of data with different parameters and
    calculate integrated intensities for each peak using the full section width.
    The integration is performed between the peak height and a background defined
    by a linear fit between intensities at the section boundaries.
    
    Parameters:
    -----------
    data : ndarray of shape (n, 2)
        Array with [Q, Intensity] values
    sections : list of dicts
        Each dict contains:
        - 'q_range': tuple (q_min, q_max) defining the section boundaries
        - 'height': float or tuple, height parameter for find_peaks
        - 'width': tuple (w_min, w_max), width parameter for find_peaks
        - 'prominence': float or tuple, optional prominence parameter
        - 'distance': float, optional minimum distance between peaks
    integration_method : str, optional
        Method to use for calculating integrated intensities:
        - 'sum': Simple summation of intensity values (default)
        - 'trapz': Trapezoidal rule integration
    plot_results : bool, optional
        Whether to plot the results (default: False)
    figsize : tuple(int, int), optional
        Figure size as (width, height) in inches (default: (12, 10))
    log_scale : bool, optional
        Whether to use log scale for intensity values (default: True)
    show_background : bool, optional
        Whether to show the linear background for each section (default: True)
    show_peak_contributions : bool, optional
        Whether to show the individual peak contributions (default: True)
    show_table : bool, optional
        Whether to show a table of results instead of a bar chart (default: True)
    title : str, optional
        Custom title for the plot (default: None)
    save_path : str, optional
        Path to save the figure (default: None, figure is not saved)
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'peaks_max': ndarray of shape (n, 2) with [q_values, intensities]
          for each peak (peak maximum/height values)
        - 'peaks_raw': ndarray of shape (n, 2) with [q_values, raw_section_integrals]
          for each peak (raw section integrated intensities)
        - 'peaks_net': ndarray of shape (n, 2) with [q_values, net_section_integrals]
          for each peak (background-subtracted section integrated intensities)
        - 'peaks_norm': ndarray of shape (n, 4) with [normalized_q, normalized_intensity,
          normalized_raw_integral, normalized_net_integral] where each column
          is normalized by dividing by its maximum value
        - 'widths': Width values of detected peaks in Q units
        - 'indices': Original indices of peaks in the data array
        - 'section_integrals': List of dictionaries with integration results for each section
    
    plot_output : tuple or None
        If plot_results=True, returns (fig, axes) from the generated plot
        
    Note:
    -----
    - For each peak, the values in 'peaks_raw' and 'peaks_net' are the section
      integrals from the section containing that peak. If multiple peaks are in
      the same section, they all get the same section integral values.
    - The background is calculated as a linear fit between intensities at the section boundaries.
    """
    # Initialize lists to store results
    all_q_peaks = []
    all_intensities = []
    all_section_indices = []  # To track which section each peak belongs to
    all_widths = []
    all_indices = []
    section_integrals = []
    
    # Process each section with its own parameters
    for section_idx, section in enumerate(sections):
        q_min, q_max = section['q_range']
        
        # Find indices corresponding to the Q range
        section_mask = (data[:, 0] >= q_min) & (data[:, 0] <= q_max)
        section_indices = np.where(section_mask)[0]
        
        if len(section_indices) == 0:
            continue
        
        # Extract the section data
        section_data = data[section_mask]
        q_section = section_data[:, 0]
        intensity_section = section_data[:, 1]
        
        # Calculate linear background between section boundaries
        # Get intensities at the section boundaries (or closest points)
        if q_min <= data[0, 0]:
            # If q_min is before the first data point, use the first point
            i_min = data[0, 1]
        else:
            # Find closest point before q_min
            idx_min = np.where(data[:, 0] <= q_min)[0][-1]
            i_min = data[idx_min, 1]
        
        if q_max >= data[-1, 0]:
            # If q_max is after the last data point, use the last point
            i_max = data[-1, 1]
        else:
            # Find closest point after q_max
            idx_max = np.where(data[:, 0] >= q_max)[0][0]
            i_max = data[idx_max, 1]
        
        # Calculate linear background for the entire section
        slope = (i_max - i_min) / (q_max - q_min)
        background_points = i_min + slope * (q_section - q_min)
        background_level = (i_min + i_max) / 2  # Average for reporting
        
        # Calculate background-subtracted intensities
        subtracted_intensity = intensity_section - background_points
        
        # Calculate section integrals based on the selected method
        if integration_method == 'trapz':
            # Use trapezoidal rule for integration
            from scipy.integrate import trapz
            raw_integral = trapz(intensity_section, q_section)
            background_integral = trapz(background_points, q_section)
            background_subtracted_integral = trapz(subtracted_intensity, q_section)
        else:  # 'sum' (default)
            # Simple summation of intensity values
            # For more accurate summation, we scale by the average step size
            if len(q_section) > 1:
                avg_step = (q_section[-1] - q_section[0]) / (len(q_section) - 1)
                raw_integral = np.sum(intensity_section) * avg_step
                background_integral = np.sum(background_points) * avg_step
                background_subtracted_integral = np.sum(subtracted_intensity) * avg_step
            else:
                raw_integral = intensity_section[0]
                background_integral = background_points[0]
                background_subtracted_integral = subtracted_intensity[0]
        
        # Store section integration results
        section_integral_info = {
            'q_range': (q_min, q_max),
            'raw_integral': raw_integral,
            'background_integral': background_integral,
            'background_subtracted_integral': background_subtracted_integral,
            'background_level': background_level,
            'q_points': q_section,
            'intensity_points': intensity_section,
            'background_points': background_points,
            'integration_method': integration_method,
            'boundary_intensities': (i_min, i_max)
        }
        section_integrals.append(section_integral_info)
        
        # Prepare parameters for find_peaks
        params = {}
        if 'height' in section:
            params['height'] = section['height']
        if 'width' in section:
            params['width'] = section['width']
        if 'prominence' in section:
            params['prominence'] = section['prominence']
        if 'distance' in section:
            params['distance'] = section['distance']
            
        # Find peaks in this section
        peak_indices, properties = find_peaks(section_data[:, 1], **params)
        
        if len(peak_indices) == 0:
            continue
        
        # Convert peak indices to global indices
        global_indices = section_indices[peak_indices]
        
        # Get Q values and intensities for these peaks
        q_peaks = section_data[peak_indices, 0]
        intensities = section_data[peak_indices, 1]
        
        # Calculate widths in Q units from peak properties
        if 'width' in section and 'widths' in properties:
            # Get left and right interpolated positions
            left_ips = properties['left_ips']
            right_ips = properties['right_ips']
            
            # Convert to original indices
            left_indices = section_indices[0] + left_ips
            right_indices = section_indices[0] + right_ips
            
            # Interpolate to get Q values at these positions
            widths_q = []
            for i in range(len(left_indices)):
                left_idx = int(np.floor(left_indices[i]))
                left_frac = left_indices[i] - left_idx
                if left_idx >= 0 and left_idx < len(data) - 1:
                    left_q = data[left_idx, 0] * (1 - left_frac) + data[left_idx + 1, 0] * left_frac
                else:
                    left_q = data[max(0, min(left_idx, len(data) - 1)), 0]
                
                right_idx = int(np.floor(right_indices[i]))
                right_frac = right_indices[i] - right_idx
                if right_idx >= 0 and right_idx < len(data) - 1:
                    right_q = data[right_idx, 0] * (1 - right_frac) + data[right_idx + 1, 0] * right_frac
                else:
                    right_q = data[max(0, min(right_idx, len(data) - 1)), 0]
                
                widths_q.append(right_q - left_q)
        else:
            # Default width if not found from peak properties
            widths_q = np.full(len(q_peaks), 0.1)
        
        # Add results to master lists
        all_q_peaks.extend(q_peaks)
        all_intensities.extend(intensities)
        all_section_indices.extend([section_idx] * len(q_peaks))  # Record which section each peak belongs to
        all_widths.extend(widths_q)
        all_indices.extend(global_indices)
    
    # Convert lists to arrays and prepare for sorting
    if all_q_peaks:
        # Create arrays from lists
        q_values = np.array(all_q_peaks)
        intensities = np.array(all_intensities)
        section_indices = np.array(all_section_indices)
        widths = np.array(all_widths)
        indices = np.array(all_indices)
        
        # Sort all arrays by q-value
        sort_idx = np.argsort(q_values)
        
        # Apply sorting to all arrays
        q_values = q_values[sort_idx]
        intensities = intensities[sort_idx]
        section_indices = section_indices[sort_idx]
        widths = widths[sort_idx]
        indices = indices[sort_idx]
        
        # Create the peaks_max array
        peaks_max = np.column_stack((q_values, intensities))
        
        # Create peaks_raw and peaks_net arrays using section integrals
        raw_integrals = np.array([section_integrals[idx]['raw_integral'] for idx in section_indices])
        net_integrals = np.array([section_integrals[idx]['background_subtracted_integral'] for idx in section_indices])
        
        peaks_raw = np.column_stack((q_values, raw_integrals))
        peaks_net = np.column_stack((q_values, net_integrals))
        
        # Create normalized values array
        # Normalize each column by its maximum value
        if len(q_values) > 0:
            q_norm = q_values / np.min(q_values)
            intensity_norm = intensities / np.max(intensities)
            raw_norm = raw_integrals / np.max(raw_integrals)
            net_norm = net_integrals / np.max(net_integrals)
            
            peaks_norm = np.column_stack((q_norm, intensity_norm, raw_norm, net_norm))
        else:
            peaks_norm = np.array([]).reshape(0, 4)
        
        # Prepare the results dictionary
        results = {
            'peaks_max': peaks_max,     # [q_values, intensities]
            'peaks_raw': peaks_raw,     # [q_values, raw_section_integrals]
            'peaks_net': peaks_net,     # [q_values, net_section_integrals]
            'peaks_norm': peaks_norm,   # [normalized_q, normalized_intensity, normalized_raw, normalized_net]
            'widths': widths,
            'indices': indices,
            'section_integrals': section_integrals,
            'section_indices': section_indices  # Store which section each peak belongs to
        }
    else:
        # Return empty arrays if no peaks found
        empty_array = np.array([]).reshape(0, 2)
        empty_norm_array = np.array([]).reshape(0, 4)
        results = {
            'peaks_max': empty_array,   # Empty [q_values, intensities]
            'peaks_raw': empty_array,   # Empty [q_values, raw_section_integrals]
            'peaks_net': empty_array,   # Empty [q_values, net_section_integrals]
            'peaks_norm': empty_norm_array,  # Empty normalized values
            'widths': np.array([]),
            'indices': np.array([]),
            'section_integrals': section_integrals,
            'section_indices': np.array([])
        }
    
    # Plot the results if requested
    if plot_results:
        plot_output = _plot_section_integration(
            data=data, 
            peak_results=results,
            figsize=figsize,
            log_scale=log_scale,
            show_background=show_background,
            show_peak_contributions=show_peak_contributions,
            show_table=show_table,
            title=title,
            save_path=save_path
        )
        return results, plot_output
    else:
        return results, None

def _plot_section_integration(data: np.ndarray, 
                             peak_results: Dict, 
                             figsize: Tuple[int, int] = (12, 10),
                             show_background: bool = True,
                             show_peak_contributions: bool = True,
                             log_scale: bool = True,
                             show_table: bool = True,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> Tuple:
    """
    Internal function to visualize the section-based integration approach.
    
    Note: This is an internal helper function called by find_peaks_by_sections
    when plot_results=True.
    """
    # Determine layout based on whether to show table
    if show_table:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
        ax_main = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
        axes = {'main': ax_main, 'table': ax_table}
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
        axes = {'main': ax_main}
    
    # Extract peak data
    peaks_max = peak_results['peaks_max']
    peaks_net = peak_results['peaks_net']
    peaks_raw = peak_results['peaks_raw']
    peaks_norm = peak_results.get('peaks_norm', None)
    section_indices = peak_results.get('section_indices', np.array([]))
    
    if len(peaks_max) > 0:
        peak_positions = peaks_max[:, 0]
        peak_intensities = peaks_max[:, 1]
        peak_raw_integrals = peaks_raw[:, 1]
        peak_net_integrals = peaks_net[:, 1]
    else:
        peak_positions = np.array([])
        peak_intensities = np.array([])
        peak_raw_integrals = np.array([])
        peak_net_integrals = np.array([])
    
    # Extract section data
    section_integrals = peak_results.get('section_integrals', [])
    
    # Plot the original data
    ax_main.plot(data[:, 0], data[:, 1], 'k-', linewidth=1.5, label='Data')
    
    # Set y-axis scale
    if log_scale:
        ax_main.set_yscale('log')
        
    # Define colors for sections
    section_colors = plt.cm.tab10(np.linspace(0, 1, len(section_integrals)))
    
    # Process each section
    for i, section_info in enumerate(section_integrals):
        q_range = section_info['q_range']
        q_min, q_max = q_range
        
        # Get section data
        q_section = section_info['q_points']
        intensity_section = section_info['intensity_points']
        background_points = section_info['background_points']
        section_color = section_colors[i]
        
        # Highlight section with transparent fill
        ax_main.axvspan(q_min, q_max, alpha=0.1, color=section_color, 
                       label=f'Section {i+1}: [{q_min:.2f}-{q_max:.2f}]')
        
        # Add vertical lines at section boundaries
        ax_main.axvline(q_min, color=section_color, linestyle='--', alpha=0.7, linewidth=1)
        ax_main.axvline(q_max, color=section_color, linestyle='--', alpha=0.7, linewidth=1)
        
        # Plot background if requested
        if show_background:
            ax_main.plot(q_section, background_points, '-', color=section_color, 
                        linewidth=2, alpha=0.7, label=f'Background Section {i+1}')
        
        # Find peaks in this section
        if len(section_indices) > 0:
            section_peaks_mask = (section_indices == i)
            section_peaks = peak_positions[section_peaks_mask]
            section_intensities = peak_intensities[section_peaks_mask]
            section_net_integrals = peak_net_integrals[section_peaks_mask]
        else:
            section_peaks = np.array([])
            section_intensities = np.array([])
            section_net_integrals = np.array([])
        
        # Plot peaks in this section
        if len(section_peaks) > 0:
            ax_main.plot(section_peaks, section_intensities, 'o', color=section_color,
                        markersize=8, label=f'Peaks Section {i+1}')
            
            # Add peak labels with section integral value
            raw_integral = section_info['raw_integral']
            bg_sub_integral = section_info['background_subtracted_integral']
            
            for j, (q, intensity) in enumerate(zip(section_peaks, section_intensities)):
                # Calculate y position for label based on scale
                if log_scale:
                    text_y = intensity * 1.2
                else:
                    text_y = intensity + (np.max(data[:, 1]) - np.min(data[:, 1])) * 0.05
                
                # Add text label
                ax_main.text(q, text_y, f'q={q:.3f}\nSection Integral={bg_sub_integral:.2e}', 
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            
            # Show peak contributions if requested
            if show_peak_contributions:
                for j, (q, intensity) in enumerate(zip(section_peaks, section_intensities)):
                    # Find the background value at this peak position
                    bg_at_peak = np.interp(q, q_section, background_points)
                    
                    # Draw a line from the background to the peak
                    ax_main.plot([q, q], [bg_at_peak, intensity], '-', 
                                color=section_color, linewidth=1.5, alpha=0.6)
                    
                    # Add a marker at the background level
                    ax_main.plot(q, bg_at_peak, 'X', color=section_color, 
                                markersize=6, alpha=0.8)
        
        # Add section integral text
        raw_integral = section_info['raw_integral']
        bg_integral = section_info.get('background_integral', 0)
        bg_sub_integral = section_info['background_subtracted_integral']
        
        # Position the text in the middle of the section
        q_pos = (q_min + q_max) / 2
        if log_scale:
            # In log scale, position at the top of the plot
            y_pos = ax_main.get_ylim()[1] * 0.9
        else:
            # In linear scale, position at the top of the plot
            y_pos = ax_main.get_ylim()[1] * 0.9
        
        # Add text with section integral information
        section_text = f"Section {i+1} Integrals:\nRaw: {raw_integral:.2e}\nBG: {bg_integral:.2e}\nNet: {bg_sub_integral:.2e}"
        ax_main.text(q_pos, y_pos, section_text, 
                    ha='center', va='top', fontsize=9, color=section_color,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Set labels and title
    ax_main.set_xlabel('q (Å$^{-1}$)', fontsize=12)
    ax_main.set_ylabel('Intensity' + (' (log scale)' if log_scale else ''), fontsize=12)
    
    if title:
        ax_main.set_title(title, fontsize=14)
    else:
        ax_main.set_title('Section-Based Peak Integration', fontsize=14)
    
    # Add grid and legend
    ax_main.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_main.legend(loc='upper right', fontsize=9)
    
    # Create a table of results instead of a bar chart
    if show_table and 'table' in axes and len(peak_positions) > 0:
        # Hide the table axes
        ax_table.axis('off')
        
        # Prepare the table data
        # Create data arrays for table - both raw and normalized values
        if peaks_norm is not None and len(peaks_norm) > 0:
            # We have normalized data
            norm_q = peaks_norm[:, 0]
            norm_intensity = peaks_norm[:, 1]
            norm_raw = peaks_norm[:, 2]
            norm_net = peaks_norm[:, 3]
        else:
            # Calculate normalized data on the fly
            norm_q = peak_positions / np.min(peak_positions)
            norm_intensity = peak_intensities / np.max(peak_intensities)
            norm_raw = peak_raw_integrals / np.max(peak_raw_integrals)
            norm_net = peak_net_integrals / np.max(peak_net_integrals)
        
        # Create table data
        table_data = []
        for i in range(len(peak_positions)):
            # Get the section index for this peak
            section_idx = section_indices[i] if len(section_indices) > 0 else -1
            
            # Format the row data
            row = [
                f"{peak_positions[i]:.4f}",  # q-value
                f"{peak_intensities[i]:.2e}",  # Peak intensity
                f"{peak_raw_integrals[i]:.2e}",  # Raw section integral
                f"{peak_net_integrals[i]:.2e}",  # Net section integral
                f"{norm_q[i]:.4f}",  # Normalized q-value
                f"{norm_intensity[i]:.4f}",  # Normalized intensity
                f"{norm_raw[i]:.4f}",  # Normalized raw integral
                f"{norm_net[i]:.4f}",  # Normalized net integral
                f"{section_idx + 1}"  # Section index (1-based)
            ]
            table_data.append(row)
        
        # Create column labels
        col_labels = [
            "q",
            "Max I",
            "Raw Int",
            "Net Int",
            "Norm q",
            "Norm I",
            "Norm Raw",
            "Norm Net",
            "Section"
        ]
        
        # Create the table
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colWidths=[0.09] * len(col_labels),
            bbox=[0.05, 0.0, 0.9, 0.9]  # [left, bottom, width, height]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Highlight the header row
        for i, key in enumerate(col_labels):
            cell = table[0, i]
            cell.set_facecolor('lightblue')
            cell.set_text_props(weight='bold')
        
        # Color cells by section
        for i, row_data in enumerate(table_data):
            section_idx = int(row_data[-1]) - 1  # Convert to 0-based index
            section_color = section_colors[section_idx] if section_idx < len(section_colors) else 'white'
            
            # Set a lighter version of the section color
            light_color = section_color.copy()
            light_color[3] = 0.3  # Reduce alpha for transparency
            
            # Apply color to cells
            for j in range(len(row_data)):
                cell = table[i+1, j]
                cell.set_facecolor(light_color)
        
        # Set the table title
        ax_table.set_title("Peak Data (Raw and Normalized Values)", fontsize=14, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
      
        
        
# def plot_integrated_sections(data, peak_results, figsize=(12, 10)):
#     """
#     Plot detected peaks and integrated sections with background subtraction visualization.
    
#     Parameters:
#     -----------
#     data : ndarray of shape (n, 2)
#         Array with [Q, Intensity] values
#     peak_results : dict
#         Dictionary containing peak and integration information from find_peaks_by_sections
#     figsize : tuple, optional
#         Figure size (width, height) in inches
        
#     Returns:
#     --------
#     fig : matplotlib figure
#         The created figure for further customization if needed
#     """
#     # Check if section_integrals exists in peak_results
#     if 'section_integrals' not in peak_results or not peak_results['section_integrals']:
#         raise ValueError("No section integration results found in peak_results")
    
#     # Create figure with two subplots (stacked vertically)
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
#     # First subplot: Original data with peaks and sections
#     # Plot the original data
#     ax1.plot(data[:, 0], data[:, 1], 'b-', linewidth=1, label='Original Data')
    
#     # Set y-axis to log scale
#     ax1.set_yscale('log')
    
#     # Get min and max for y-axis limits
#     min_positive = np.min(data[data[:, 1] > 0, 1])
#     max_intensity = np.max(data[:, 1])
#     y_min = min_positive / 2
#     y_max = max_intensity * 2
#     ax1.set_ylim(y_min, y_max)
    
#     # Define section colors
#     section_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lavender']
    
#     # Highlight sections and plot background levels
#     for i, section_info in enumerate(peak_results['section_integrals']):
#         # Get section boundaries
#         q_min, q_max = section_info['q_range']
#         q_points = section_info['q_points']
#         intensities = section_info['intensity_points']
#         background = section_info['background_points']
#         raw_integral = section_info['raw_integral']
#         bg_sub_integral = section_info['background_subtracted_integral']
        
#         # Add section shading with transparency
#         color = section_colors[i % len(section_colors)]
#         rect = Rectangle((q_min, y_min), q_max - q_min, y_max - y_min, 
#                          facecolor=color, alpha=0.2, zorder=0)
#         ax1.add_patch(rect)
        
#         # Plot background level
#         ax1.plot(q_points, background, 'r--', linewidth=1.5, alpha=0.7)
        
#         # Add section label with integration results
#         label_y = 0.92 - i * 0.05  # Position labels vertically
#         ax1.text(0.02, label_y, 
#                 f"Section {i+1} ({q_min:.2f}-{q_max:.2f}): Raw={raw_integral:.2e}, BG Sub={bg_sub_integral:.2e}",
#                 transform=ax1.transAxes, fontsize=9, 
#                 bbox=dict(facecolor=color, alpha=0.5, boxstyle='round'))
        
#         # Add vertical lines at section boundaries
#         ax1.axvline(q_min, color='gray', linestyle='--', alpha=0.5, zorder=1)
#         ax1.axvline(q_max, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
#     # Plot the detected peaks
#     if len(peak_results['q_values']) > 0:
#         ax1.plot(peak_results['q_values'], peak_results['intensities'], 'ro', 
#                 markersize=8, label='Detected Peaks')
        
#         # Annotate each peak with its Q value
#         for i, (q, intensity) in enumerate(zip(peak_results['q_values'], peak_results['intensities'])):
#             # In log scale, position the text above the peak
#             log_intensity = np.log10(intensity)
#             text_y = 10**(log_intensity + 0.1)
            
#             ax1.annotate(f'{q:.2f}', (q, text_y), xytext=(0, 5), 
#                         textcoords='offset points', fontsize=9, ha='center')
    
#     # Set axes labels and title for first subplot
#     ax1.set_xlabel('Q (Å$^{-1}$)', fontsize=12)
#     ax1.set_ylabel('Intensity (log scale)', fontsize=12)
#     ax1.set_title('Peak Detection and Section Integration Results', fontsize=14)
#     ax1.legend(loc='upper right')
#     ax1.grid(True, which='major', linestyle='-', alpha=0.5)
#     ax1.grid(True, which='minor', linestyle=':', alpha=0.3)
    
#     # Second subplot: Bar chart of integrated intensities
#     # Extract section data for plotting
#     section_numbers = [f"Sec {i+1}" for i in range(len(peak_results['section_integrals']))]
#     raw_integrals = [info['raw_integral'] for info in peak_results['section_integrals']]
#     bg_sub_integrals = [info['background_subtracted_integral'] for info in peak_results['section_integrals']]
    
#     # Decide whether to use log scale based on data range
#     log_scale_bars = max(raw_integrals) / min(filter(lambda x: x > 0, raw_integrals + bg_sub_integrals)) > 100
    
#     # Set up bar positions
#     x = np.arange(len(section_numbers))
#     width = 0.35
    
#     # Create bar chart
#     if log_scale_bars:
#         ax2.set_yscale('log')
    
#     bar_colors = [section_colors[i % len(section_colors)] for i in range(len(section_numbers))]
#     bars1 = ax2.bar(x - width/2, raw_integrals, width, label='Raw Integral', 
#                    alpha=0.7, color=bar_colors)
#     bars2 = ax2.bar(x + width/2, bg_sub_integrals, width, label='BG Subtracted', 
#                    alpha=0.7, color=[c + '80' for c in bar_colors])  # Lighter shade
    
#     # Add labels and formatting
#     ax2.set_xlabel('Section', fontsize=12)
#     ax2.set_ylabel('Integrated Intensity', fontsize=12)
#     ax2.set_title('Comparison of Raw and Background-Subtracted Integrals', fontsize=14)
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(section_numbers)
#     ax2.legend()
#     ax2.grid(True, which='major', linestyle='--', alpha=0.5)
    
#     # Add value labels on bars
#     def autolabel(bars):
#         for bar in bars:
#             height = bar.get_height()
#             if log_scale_bars:
#                 y_pos = height * 1.1
#             else:
#                 y_pos = height + 0.05 * max(raw_integrals)
            
#             ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
#                     f'{height:.1e}',
#                     ha='center', va='bottom', fontsize=8, rotation=45)
    
#     autolabel(bars1)
#     autolabel(bars2)
    
#     plt.tight_layout()
#     return fig

# Example usage:
# Define different sections with appropriate parameters
def analyze_xrd_data(data):
    sections = [
        {
            'q_range': (1.0, 3.0),  # Strong peaks region
            'height': 200,
            'width': (3, 20),
            'prominence': 50
        },
        {
            'q_range': (3.0, 5.0),  # Medium peaks region
            'height': 100,
            'width': (2, 15),
            'prominence': 20
        },
        {
            'q_range': (5.0, 8.0),  # Weak peaks region
            'height': 50,
            'width': (2, 10),
            'prominence': 10
        }
    ]
    
    return find_peaks_by_sections(data, sections)


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.patches import Rectangle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional

def plot_peaks_over_data(data: np.ndarray, 
                        peak_results: Dict, 
                        sections: Optional[List[Dict]] = None, 
                        figsize: Tuple[int, int] = (12, 8), 
                        plot_widths: bool = True,
                        show_integrals: bool = False):
    """
    Plot detected peaks over the original intensity vs. Q data with different sections highlighted.
    Compatible with both old and new formats of peak_results.
    
    Parameters:
    -----------
    data : ndarray of shape (n, 2)
        Array with [Q, Intensity] values
    peak_results : dict
        Dictionary containing peak information from find_peaks_by_sections
        Can contain either:
        - 'q_values', 'intensities', and 'widths' keys (old format)
        - 'peaks' array with [q, intensity, integral] and 'widths' (new format)
    sections : list of dicts, optional
        Section definitions used for peak finding, to highlight different regions
    figsize : tuple, optional
        Figure size (width, height) in inches
    plot_widths : bool, optional
        Whether to indicate peak widths on the plot
    show_integrals : bool, optional
        Whether to show integrated intensities in annotations (only available with new format)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
        The created figure and axes for further customization if needed
    """
    # Extract peak data based on format
    if 'peaks' in peak_results and isinstance(peak_results['peaks'], np.ndarray) and peak_results['peaks'].shape[1] >= 2:
        # New format - extract from peaks array
        peaks_array = peak_results['peaks']
        q_values = peaks_array[:, 0]
        intensities = peaks_array[:, 1]
        if show_integrals and peaks_array.shape[1] >= 3:
            integrals = peaks_array[:, 2]
        else:
            integrals = None
    elif 'q_values' in peak_results and 'intensities' in peak_results:
        # Old format - use separate arrays
        q_values = peak_results['q_values']
        intensities = peak_results['intensities']
        integrals = None
    else:
        # No valid peaks found
        q_values = np.array([])
        intensities = np.array([])
        integrals = None
    
    # Extract widths
    if 'widths' in peak_results:
        widths = peak_results['widths']
    else:
        widths = np.zeros_like(q_values)
        plot_widths = False  # Disable width plotting if no widths available
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the original data
    ax.plot(data[:, 0], data[:, 1], 'b-', linewidth=1, label='Original Data')
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Ensure all intensity values are positive for log scale
    min_positive = np.min(data[data[:, 1] > 0, 1])
    max_intensity = np.max(data[:, 1])
    
    # Set y-axis limits - we'll use these for the section rectangles too
    y_min = min_positive / 2  # Lower than the minimum for better visibility
    y_max = max_intensity * 2  # Higher than the maximum for better visibility
    ax.set_ylim(y_min, y_max)
    
    # Highlight different sections if provided
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lavender']
    if sections is not None:
        # First pass: add all the background rectangles
        for i, section in enumerate(sections):
            q_min, q_max = section['q_range']
            
            # Add a semi-transparent rectangle covering the entire y-axis
            color = colors[i % len(colors)]
            rect = Rectangle((q_min, y_min), q_max - q_min, y_max - y_min, 
                             facecolor=color, alpha=0.2, zorder=0)  # Low zorder to keep it in background
            ax.add_patch(rect)
        
        # Second pass: add section labels at the top
        # First, determine the height for the text by getting the top of the plot in data coordinates
        fig_height = 0.95  # Place text at 95% of the figure height
        
        for i, section in enumerate(sections):
            q_min, q_max = section['q_range']
            
            # Center position for the label in Q
            q_center = (q_min + q_max) / 2
            
            # Create a text at the top of the figure
            # Use transAxes for y to position relative to figure height
            ax.text(q_center, 0.98, f"Section {i+1}", 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                    transform=ax.transData)  # Only transform x with data coordinates
            
            # Also add thin vertical lines at section boundaries for clarity
            ax.axvline(q_min, color='gray', linestyle='--', alpha=0.5, zorder=1)
            ax.axvline(q_max, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Plot the detected peaks
    if len(q_values) > 0:
        ax.plot(q_values, intensities, 'ro', 
                markersize=8, label='Detected Peaks')
        
        # Annotate each peak with its Q value and optionally its integral
        for i, (q, intensity) in enumerate(zip(q_values, intensities)):
            # In log scale, position the text above the peak
            log_intensity = np.log10(intensity)
            text_y = 10**(log_intensity + 0.1)  # Move further up in log space
            
            # Create annotation text
            if integrals is not None and show_integrals:
                annotation_text = f'q={q:.2f}\nInt={integrals[i]:.2e}'
            else:
                annotation_text = f'{q:.2f}'
            
            ax.annotate(annotation_text, (q, text_y), xytext=(0, 5), 
                        textcoords='offset points', fontsize=9, ha='center')
        
        # Visualize peak widths if requested
        if plot_widths and len(widths) > 0:
            for i, (q, intensity, width) in enumerate(zip(q_values, intensities, widths)):
                if width > 0:
                    # Use a fixed fraction of the peak height that looks good on log scale
                    visual_half_height = intensity / 3
                    
                    # Plot horizontal line at visual half height with width
                    ax.plot([q - width/2, q + width/2], [visual_half_height, visual_half_height], 
                            'g-', linewidth=2)
                    
                    # Add vertical lines to show width boundaries
                    # Since we're in log scale, use multiplicative factors for the vertical lines
                    factor = 1.2
                    ax.plot([q - width/2, q - width/2], [visual_half_height/factor, visual_half_height*factor], 
                            'g-', linewidth=1)
                    ax.plot([q + width/2, q + width/2], [visual_half_height/factor, visual_half_height*factor], 
                            'g-', linewidth=1)
                    
                    # Annotate width value below the half-height line
                    ax.annotate(f'w={width:.3f}', (q, visual_half_height/1.5), xytext=(0, -5), 
                                textcoords='offset points', ha='center', fontsize=8)
    
    # Set axes labels and title
    ax.set_xlabel('Q (Å$^{-1}$)', fontsize=12)
    ax.set_ylabel('Intensity (log scale)', fontsize=12)
    ax.set_title('Peak Detection Results', fontsize=14)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid for better readability - in log scale, use both major and minor grids
    ax.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def BCPSimFit_2DParameterSweep_Modified(data: np.ndarray, 
                                       Pitch_mean: float,
                                       Fraction_start: float,
                                       Fraction_end: float,
                                       Fraction_steps: int,
                                       DW_start: float, 
                                       DW_end: float, 
                                       DW_steps: int, 
                                       I0: float, 
                                       Bk: float, 
                                       additional_data: Optional[np.ndarray] = None,
                                       highlight_percent: float = 5, 
                                       logfit: bool = False, 
                                       plot_results: bool = True,
                                       fit_pitch: bool = False,
                                       intensity_scale: Optional[Tuple[float, float]] = None,
                                       custom_cmap: str = 'viridis',
                                       show_3d_plot: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Fit scattering data while sweeping both the Debye-Waller (DW) and Fraction parameters
    over a 2D grid of values to find the optimal combination, with customizable intensity scaling.
    
    Parameters:
    -----------
    data : np.ndarray of shape (n, 2)
        Array with [Q, Intensity] values to fit
    Pitch_mean : float
        Pitch value for the model (fixed if fit_pitch=False)
    Fraction_start : float
        Lower bound for Fraction parameter sweep
    Fraction_end : float
        Upper bound for Fraction parameter sweep
    Fraction_steps : int
        Number of steps between Fraction_start and Fraction_end
    DW_start : float
        Lower bound for DW parameter sweep
    DW_end : float
        Upper bound for DW parameter sweep
    DW_steps : int
        Number of steps between DW_start and DW_end
    I0 : float
        Initial intensity scaling factor
    Bk : float
        Initial background value
    additional_data : np.ndarray of shape (m, 2), optional
        Additional [Q, Intensity] data to plot alongside the fitted data
    highlight_percent : float, optional
        Percentage difference from best fit to highlight (default: 5%)
    logfit : bool, optional
        Whether to fit in log space (default: False)
    plot_results : bool, optional
        Whether to plot the results (default: True)
    fit_pitch : bool, optional
        Whether to fit the Pitch parameter or keep it fixed (default: False)
    intensity_scale : tuple(float, float), optional
        Custom scale for the intensity (vmin, vmax) in the 2D plot. If None,
        auto-scaling will be used based on data and highlight_percent.
    custom_cmap : str, optional
        Name of the colormap to use for the 2D plot (default: 'viridis')
    show_3d_plot : bool, optional
        Whether to show the 3D plot alongside the 2D plot (default: True)
        
    Returns:
    --------
    best_params : np.ndarray
        Best fit parameters [Pitch, Fraction, I0, DW, Bk]
    best_cov : np.ndarray or None
        Covariance matrix for best fit (may be None if not available)
    sweep_results : dict
        Dictionary containing:
        - 'dw_values': Array of DW values used in sweep
        - 'fraction_values': Array of Fraction values used in sweep
        - 'gf_values': 2D array of goodness of fit values for each parameter combination
        - 'parameters': 2D array of parameter sets for each combination
        
    Raises:
    -------
    ValueError
        If input data is not properly formatted or if all fits fail
    RuntimeError
        If the fitting process fails completely
    """
    # Validate input data
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be a numpy array of shape (n, 2) containing [Q, Intensity] values")
    
    if additional_data is not None and (not isinstance(additional_data, np.ndarray) or 
                                       additional_data.ndim != 2 or 
                                       additional_data.shape[1] != 2):
        raise ValueError("additional_data must be a numpy array of shape (m, 2) containing [Q, Intensity] values")
    
    if DW_steps < 2 or Fraction_steps < 2:
        raise ValueError("DW_steps and Fraction_steps must both be at least 2")
    
    if DW_start >= DW_end:
        raise ValueError("DW_start must be less than DW_end")
        
    if Fraction_start >= Fraction_end:
        raise ValueError("Fraction_start must be less than Fraction_end")

    # Create arrays of parameter values to sweep
    dw_values = np.linspace(DW_start, DW_end, DW_steps)
    fraction_values = np.linspace(Fraction_start, Fraction_end, Fraction_steps)
    
    # Create meshgrid for 2D parameter space
    DW_grid, Fraction_grid = np.meshgrid(dw_values, fraction_values)
    
    # Initialize arrays to store fit results
    gf_values = np.full((Fraction_steps, DW_steps), np.nan)
    all_params = np.full((Fraction_steps, DW_steps), None, dtype=object)
    all_covs = np.full((Fraction_steps, DW_steps), None, dtype=object)
    
    # Define goodness of fit (GF) function - lower is better
    def calc_gf(y_true, y_pred):
        if logfit:
            # For log fit, use mean squared error in log space
            y_true_log = np.log(y_true)
            y_pred_log = np.log(y_pred)
            return np.mean((y_true_log - y_pred_log) ** 2)
        else:
            # For linear fit, use normalized chi-squared
            # (sum of squared relative errors)
            return np.mean(((y_true - y_pred) / y_true) ** 2)
    
    LD = np.log(data[:, 1])
    
    # Track progress
    total_fits = Fraction_steps * DW_steps
    successful_fits = 0
    
    print(f"Starting 2D parameter sweep with {total_fits} combinations...")
    
    # Sweep through parameter combinations
    for i, fraction in enumerate(fraction_values):
        for j, dw in enumerate(dw_values):
            try:
                if fit_pitch:
                    # Set initial parameters and bounds for fitting Pitch, I0, and Bk
                    # (Fraction and DW are fixed)
                    param_init = [Pitch_mean, I0, Bk]
                    bounds = ([Pitch_mean * 0.99, I0 * 0.01, Bk * 0.95], 
                              [Pitch_mean * 1.01, I0 * 100, Bk * 1.05])
                    
                    if not logfit:
                        # Create a wrapper function with fixed Fraction and DW
                        def wrapper(q, Pitch, I0, Bk):
                            return SimInt_BCP(q, Pitch, fraction, I0, dw, Bk)
                        
                        params, params_cov = scipy.optimize.curve_fit(
                            wrapper, data[:, 0], data[:, 1], p0=param_init, bounds=bounds
                        )
                        
                        # Insert the fixed parameters into the parameter list
                        full_params = np.array([params[0], fraction, params[1], dw, params[2]])
                        
                        # Calculate goodness of fit
                        y_pred = SimInt_BCP(data[:, 0], *full_params)
                        gf = calc_gf(data[:, 1], y_pred)
                    else:
                        # Create a wrapper function with fixed Fraction and DW for log fitting
                        def wrapper(q, Pitch, I0, Bk):
                            return SimIntLog_BCP(q, Pitch, fraction, I0, dw, Bk)
                        
                        params, params_cov = scipy.optimize.curve_fit(
                            wrapper, data[:, 0], LD, p0=param_init, bounds=bounds
                        )
                        
                        # Insert the fixed parameters into the parameter list
                        full_params = np.array([params[0], fraction, params[1], dw, params[2]])
                        
                        # Calculate goodness of fit
                        y_pred = SimInt_BCP(data[:, 0], *full_params)
                        gf = calc_gf(data[:, 1], y_pred)
                else:
                    # Fixed Pitch: set initial parameters and bounds for fitting only I0 and Bk
                    # (Pitch, Fraction, and DW are all fixed)
                    param_init = [I0, Bk]
                    bounds = ([I0 * 0.01, Bk * 0.95], 
                              [I0 * 100, Bk * 1.05])
                    
                    if not logfit:
                        # Create a wrapper function with fixed Pitch, Fraction, and DW
                        def wrapper(q, I0, Bk):
                            return SimInt_BCP(q, Pitch_mean, fraction, I0, dw, Bk)
                        
                        params, params_cov = scipy.optimize.curve_fit(
                            wrapper, data[:, 0], data[:, 1], p0=param_init, bounds=bounds
                        )
                        
                        # Insert the fixed parameters into the parameter list
                        full_params = np.array([Pitch_mean, fraction, params[0], dw, params[1]])
                        
                        # Calculate goodness of fit
                        y_pred = SimInt_BCP(data[:, 0], *full_params)
                        gf = calc_gf(data[:, 1], y_pred)
                    else:
                        # Create a wrapper function with fixed Pitch, Fraction, and DW for log fitting
                        def wrapper(q, I0, Bk):
                            return SimIntLog_BCP(q, Pitch_mean, fraction, I0, dw, Bk)
                        
                        params, params_cov = scipy.optimize.curve_fit(
                            wrapper, data[:, 0], LD, p0=param_init, bounds=bounds
                        )
                        
                        # Insert the fixed parameters into the parameter list
                        full_params = np.array([Pitch_mean, fraction, params[0], dw, params[1]])
                        
                        # Calculate goodness of fit
                        y_pred = SimInt_BCP(data[:, 0], *full_params)
                        gf = calc_gf(data[:, 1], y_pred)
                
                # Store results
                gf_values[i, j] = gf
                all_params[i, j] = full_params
                all_covs[i, j] = params_cov
                successful_fits += 1
                
                # Print progress every 10% completion
                progress = (i * DW_steps + j + 1) / total_fits * 100
                if (i * DW_steps + j + 1) % max(1, total_fits // 10) == 0:
                    print(f"Progress: {progress:.1f}% ({successful_fits} successful fits)")
                
            except Exception as e:
                # Quietly continue on failure - we'll handle the NaN values later
                pass
    
    # Find best fit
    valid_mask = ~np.isnan(gf_values)
    if not np.any(valid_mask):
        raise ValueError("All fits failed. Try adjusting the parameter ranges or the data.")
    
    best_idx = np.nanargmin(gf_values.flatten())
    best_i, best_j = np.unravel_index(best_idx, gf_values.shape)
    best_params = all_params[best_i, best_j]
    best_cov = all_covs[best_i, best_j]
    best_fraction = fraction_values[best_i]
    best_dw = dw_values[best_j]
    best_gf = gf_values[best_i, best_j]
    
    print(f"\nCompleted {successful_fits} of {total_fits} fits successfully.")
    
    # Print best fit info
    pitch_status = "variable" if fit_pitch else "fixed"
    print(f"Best fit found at Fraction = {best_fraction:.4f}, DW = {best_dw:.4f} with goodness of fit = {best_gf:.4e}")
    print(f"Best parameters: Pitch = {best_params[0]:.4f}" + (" (fixed)" if not fit_pitch else "") + 
          f", Fraction = {best_params[1]:.4f}, I0 = {best_params[2]:.4e}, " +
          f"DW = {best_params[3]:.4f}, Bk = {best_params[4]:.4e}")
    
    if plot_results:
        # Create a figure with the appropriate number of subplots
        if show_3d_plot:
            fig = plt.figure(figsize=(18, 6))
            gs = fig.add_gridspec(1, 3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        else:
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
        
        # First subplot: Best fit
        # Plot additional data first if provided
        if additional_data is not None:
            ax1.plot(additional_data[:, 0], additional_data[:, 1], 'b-', 
                    label='Additional Data', alpha=0.7, linewidth=1)
        
        # Plot original data and best fit
        ax1.plot(data[:, 0], data[:, 1], 'k-', label='Exp Data', linewidth=1)
        ax1.plot(data[:, 0], SimInt_BCP(data[:, 0], *best_params), 
                color='r', label=f'Best Fit', linewidth=2)
        ax1.scatter(data[:, 0], data[:, 1], color='r', alpha=0.4, s=20)
        ax1.set_yscale('log')
        ax1.set_xlabel('q (Å$^{-1}$)', fontsize=14)
        ax1.set_ylabel('Intensity (a.u)', fontsize=14)
        ax1.set_title(f'Best Fit Result', fontsize=16)
        ax1.legend(loc='upper right')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Show the best parameters
        param_names = ['Pitch', 'Fraction', 'I0', 'DW', 'Bk']
        param_text = "Best Parameters:\n"
        for i, (name, val) in enumerate(zip(param_names, best_params)):
            if name == 'Pitch' and not fit_pitch:
                param_text += f"{name}: {val:.4e} (fixed)\n"
            else:
                param_text += f"{name}: {val:.4e}\n"
        
        # Add text box with best parameters
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.05, param_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # Second subplot: 2D heatmap of goodness of fit
        # Create a mask for invalid values
        masked_gf = np.ma.masked_invalid(gf_values)
        
        # Calculate threshold for highlighting
        threshold = best_gf * (1 + highlight_percent / 100)
        
        # Create a custom colormap that highlights the best values
        cmap = plt.cm.get_cmap(custom_cmap).copy()
        cmap.set_bad('gray', 0.5)  # Gray color for masked values
        
        # Determine intensity scale (vmin, vmax) for the heatmap
        if intensity_scale is None:
            vmin = np.nanmin(gf_values)
            vmax = threshold * 2  # Default scale based on threshold
        else:
            vmin, vmax = intensity_scale
        
        # Plot heatmap with the specified intensity scale
        pcm = ax2.pcolormesh(DW_grid, Fraction_grid, masked_gf, 
                           norm=Normalize(vmin=vmin, vmax=vmax),
                           cmap=cmap, shading='auto')
        
        # Add a colorbar
        cbar = fig.colorbar(pcm, ax=ax2)
        cbar.set_label('Goodness of Fit (lower is better)', fontsize=12)
        
        # Add contour lines for the threshold
        if np.any(masked_gf < threshold):
            ax2.contour(DW_grid, Fraction_grid, masked_gf, 
                       levels=[threshold], colors='red', linestyles='dashed', 
                       linewidths=2, alpha=0.7)
        
        # Mark the best fit point
        ax2.plot(best_dw, best_fraction, 'ro', markersize=10, markeredgecolor='white')
        
        ax2.set_xlabel('Debye-Waller Factor', fontsize=14)
        ax2.set_ylabel('Fraction', fontsize=14)
        ax2.set_title('2D Parameter Space Goodness of Fit', fontsize=16)
        
        # Third subplot: 3D surface plot (if requested)
        if show_3d_plot:
            # Create X, Y for plotting
            X, Y = np.meshgrid(dw_values, fraction_values)
            
            # Plot the 3D surface
            surf = ax3.plot_surface(X, Y, masked_gf, cmap=custom_cmap, 
                                   linewidth=0, antialiased=True, alpha=0.8)
            
            # Add a contour plot at the bottom
            cset = ax3.contourf(X, Y, masked_gf, zdir='z', offset=np.nanmin(gf_values),
                               cmap=custom_cmap, alpha=0.5)
            
            # Mark the best fit point
            ax3.scatter([best_dw], [best_fraction], [best_gf], color='red', s=100, 
                       edgecolor='white', linewidth=1.5)
            
            ax3.set_xlabel('Debye-Waller Factor', fontsize=12)
            ax3.set_ylabel('Fraction', fontsize=12)
            ax3.set_zlabel('Goodness of Fit', fontsize=12)
            ax3.set_title('3D Parameter Space Visualization', fontsize=16)
        
        fig.tight_layout()
        plt.show()
        
        # Also create a focused view around the best parameter combination
        # Calculate the region of interest (within threshold)
        try:
            within_threshold = (gf_values <= threshold) & (~np.isnan(gf_values))
            if np.any(within_threshold):
                plt.figure(figsize=(10, 8))
                
                # Find indices within threshold
                i_indices, j_indices = np.where(within_threshold)
                
                # Get min/max values with padding
                min_i, max_i = max(0, np.min(i_indices) - 1), min(Fraction_steps - 1, np.max(i_indices) + 1)
                min_j, max_j = max(0, np.min(j_indices) - 1), min(DW_steps - 1, np.max(j_indices) + 1)
                
                # Extract the focused region of the parameter grid
                focused_gf = gf_values[min_i:max_i+1, min_j:max_j+1]
                focused_fraction = fraction_values[min_i:max_i+1]
                focused_dw = dw_values[min_j:max_j+1]
                
                # Create new meshgrid for the focused region
                focused_X, focused_Y = np.meshgrid(focused_dw, focused_fraction)
                
                # Create a mask for invalid values
                masked_focused_gf = np.ma.masked_invalid(focused_gf)
                
                # Determine focused intensity scale
                if intensity_scale is None:
                    focused_vmin = np.nanmin(gf_values)
                    focused_vmax = threshold * 1.2
                else:
                    focused_vmin, focused_vmax = intensity_scale
                
                # Plot heatmap for the focused region with the intensity scale
                plt.pcolormesh(focused_X, focused_Y, masked_focused_gf, 
                              norm=Normalize(vmin=focused_vmin, vmax=focused_vmax),
                              cmap=cmap, shading='auto')
                
                # Add contour for the threshold
                plt.contour(focused_X, focused_Y, masked_focused_gf, 
                           levels=[threshold], colors='red', linestyles='dashed', 
                           linewidths=2, alpha=0.7)
                
                # Mark the best fit point
                plt.plot(best_dw, best_fraction, 'ro', markersize=10, markeredgecolor='white')
                
                # Add colorbar
                cbar = plt.colorbar()
                cbar.set_label('Goodness of Fit (lower is better)', fontsize=12)
                
                plt.xlabel('Debye-Waller Factor', fontsize=14)
                plt.ylabel('Fraction', fontsize=14)
                plt.title(f'Focused View of Optimal Region (within {highlight_percent}%)', fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Could not create focused view: {e}")
    
    # Prepare return dictionary with sweep results
    sweep_results = {
        'dw_values': dw_values,
        'fraction_values': fraction_values,
        'gf_values': gf_values,
        'parameters': all_params
    }
    
    return best_params, best_cov, sweep_results

def extract_peak_data(peak_results):
    """
    Extract peak Q values and intensities from find_peaks_by_sections results
    and convert them to a numpy array.
    
    Parameters:
    -----------
    peak_results : dict
        Dictionary containing peak information from find_peaks_by_sections
        
    Returns:
    --------
    peak_data : ndarray of shape (n, 2)
        Array with [Q, Intensity] values for each peak
    """
    if len(peak_results['q_values']) == 0:
        return np.array([]).reshape(0, 2)
    
    # Create a numpy array with q values and intensities
    q_values = peak_results['q_values']
    intensities = peak_results['intensities']
    
    # Stack into a 2D array
    peak_data = np.column_stack((q_values, intensities))
    
    # Sort by q values (should already be sorted, but just to be sure)
    peak_data = peak_data[np.argsort(peak_data[:, 0])]
    
    return peak_data

import numpy as np
from scipy.integrate import trapezoid
from typing import Dict, List, Tuple, Optional, Union

def extract_peak_intensities(data: np.ndarray, 
                            peak_results: Dict, 
                            integration_width: Union[float, List[float], np.ndarray] = None,
                            background_subtracted: bool = True,
                            local_background: bool = True,
                            integration_method: str = 'sum') -> np.ndarray:
    """
    Extract peak positions (Q) and their integrated intensities as a simple 2D array.
    Uses either the peak widths or a specified integration width for each peak.
    Integration can be done using simple summation or trapezoidal rule.
    
    Parameters:
    -----------
    data : ndarray of shape (n, 2)
        Array with [Q, Intensity] values from the original dataset
    peak_results : dict
        Dictionary containing peak information from find_peaks_by_sections
    integration_width : float or list or ndarray, optional
        Width in Q units to use for integration around each peak.
        - If float: Same width for all peaks
        - If list/array: Custom width for each peak (must match number of peaks)
        - If None: Use peak widths from peak_results if available, otherwise default to 0.1
    background_subtracted : bool, optional
        Whether to return background-subtracted intensities (True) or raw intensities (False)
        Default is True (background-subtracted)
    local_background : bool, optional
        Whether to subtract local background from integrated intensities (default: True)
        Only used if background_subtracted=True
    integration_method : str, optional
        Method to use for calculating integrated intensities:
        - 'sum': Simple summation of intensity values (default)
        - 'trapz': Trapezoidal rule integration
        
    Returns:
    --------
    peak_intensities : ndarray of shape (n, 2)
        Array with [Q, Integrated_Intensity] values for each peak
    """
    # Check if peak_results has detected peaks
    if 'q_values' not in peak_results or len(peak_results['q_values']) == 0:
        return np.array([]).reshape(0, 2)
    
    # Extract peak data
    q_values = peak_results['q_values']
    num_peaks = len(q_values)
    
    # Determine integration widths
    if integration_width is None:
        if 'widths' in peak_results and len(peak_results['widths']) == num_peaks:
            # Use provided peak widths, but ensure they're all positive
            integration_widths = np.maximum(peak_results['widths'], 0.05)
        else:
            # Default to fixed width of 0.1 for all peaks
            integration_widths = np.full(num_peaks, 0.1)
    elif isinstance(integration_width, (int, float)):
        # Same width for all peaks
        integration_widths = np.full(num_peaks, float(integration_width))
    elif isinstance(integration_width, (list, np.ndarray)):
        # Check that provided widths match number of peaks
        if len(integration_width) != num_peaks:
            raise ValueError(f"Number of integration widths ({len(integration_width)}) must match number of peaks ({num_peaks})")
        integration_widths = np.array(integration_width, dtype=float)
    else:
        raise TypeError("integration_width must be a float, list, numpy array, or None")
    
    # Create arrays to store results
    integrated_intensities = np.zeros(num_peaks)
    
    # Sort data by q value to ensure correct integration
    sorted_data = data[np.argsort(data[:, 0])]
    q_data = sorted_data[:, 0]
    intensity_data = sorted_data[:, 1]
    
    # Process each peak
    for i, (q_peak, width) in enumerate(zip(q_values, integration_widths)):
        # Define integration range around peak
        half_width = width / 2
        q_min = q_peak - half_width
        q_max = q_peak + half_width
        
        # Find data points within integration range
        in_range = (q_data >= q_min) & (q_data <= q_max)
        
        # Skip if no data points in range
        if not np.any(in_range):
            integrated_intensities[i] = 0
            continue
        
        # Extract data points for integration
        q_range = q_data[in_range]
        intensity_range = intensity_data[in_range]
        
        # Determine background if requested
        if background_subtracted and local_background:
            # Use linear interpolation between edges of integration range
            # First, check if we need to find points outside the range
            if q_min <= q_data[0] or q_max >= q_data[-1]:
                # We're at the edge of the data, use the closest edge value as background
                if q_min <= q_data[0]:
                    left_bg = intensity_range[0]
                else:
                    # Find closest point just before q_min
                    idx_left = np.where(q_data < q_min)[0]
                    if len(idx_left) > 0:
                        left_bg = intensity_data[idx_left[-1]]
                    else:
                        left_bg = intensity_range[0]
                
                if q_max >= q_data[-1]:
                    right_bg = intensity_range[-1]
                else:
                    # Find closest point just after q_max
                    idx_right = np.where(q_data > q_max)[0]
                    if len(idx_right) > 0:
                        right_bg = intensity_data[idx_right[0]]
                    else:
                        right_bg = intensity_range[-1]
            else:
                # Find closest points just outside the range
                idx_left = np.where(q_data < q_min)[0][-1]
                idx_right = np.where(q_data > q_max)[0][0]
                left_bg = intensity_data[idx_left]
                right_bg = intensity_data[idx_right]
            
            # Linear interpolation for background
            slope = (right_bg - left_bg) / (q_max - q_min)
            background = left_bg + slope * (q_range - q_min)
        else:
            # No background subtraction, set background to zero
            background = np.zeros_like(q_range)
        
        # Calculate integrated intensities based on the selected method
        if background_subtracted:
            # Subtract background before integration
            subtracted_intensity = intensity_range - background
            
            if integration_method == 'trapz':
                # Use trapezoidal rule for integration
                from scipy.integrate import trapz
                integrated_intensity = trapz(subtracted_intensity, q_range)
            else:  # 'sum' (default)
                # Simple summation of intensity values
                # For more accurate summation, we scale by the average step size
                if len(q_range) > 1:
                    avg_step = (q_range[-1] - q_range[0]) / (len(q_range) - 1)
                    integrated_intensity = np.sum(subtracted_intensity) * avg_step
                else:
                    integrated_intensity = subtracted_intensity[0]
        else:
            # Just integrate the raw intensity
            if integration_method == 'trapz':
                from scipy.integrate import trapz
                integrated_intensity = trapz(intensity_range, q_range)
            else:  # 'sum' (default)
                if len(q_range) > 1:
                    avg_step = (q_range[-1] - q_range[0]) / (len(q_range) - 1)
                    integrated_intensity = np.sum(intensity_range) * avg_step
                else:
                    integrated_intensity = intensity_range[0]
        
        # Store the integrated intensity
        integrated_intensities[i] = integrated_intensity
    
    # Create the final result array with [Q, Integrated_Intensity]
    peak_intensities = np.column_stack((q_values, integrated_intensities))
    
    # Sort by Q value to ensure consistent order
    peak_intensities = peak_intensities[np.argsort(peak_intensities[:, 0])]
    
    return peak_intensities

def pitchcalc_from_results(peak_results: Dict) -> Tuple[Optional[float], np.ndarray]:
    """
    Calculate the pitch from peak detection results using 2π/mean(spacing).
    Works with both old and new formats of peak_results.
    
    Parameters:
    -----------
    peak_results : dict
        Dictionary containing peak information from find_peaks_by_sections
        Can contain either:
        - 'q_values' key (old format)
        - 'peaks' array with [q, intensity, integral] (new format)
        
    Returns:
    --------
    pitch : float or None
        Calculated pitch value, or None if fewer than 2 peaks detected
    spacing : ndarray
        Array of spacings between adjacent peaks
    """
    # Check which format we're dealing with
    if 'peaks_max' in peak_results and isinstance(peak_results['peaks_max'], np.ndarray) and peak_results['peaks_max'].shape[1] >= 1:
        # New format - extract q_values from peaks array (first column)
        q_values = peak_results['peaks_max'][:, 0]
    elif 'q_values' in peak_results:
        # Old format - use q_values directly
        q_values = peak_results['q_values']
    else:
        # No valid peaks found
        return None, np.array([])
    
    # Check if we have enough peaks
    if len(q_values) < 2:
        return None, np.array([])
    
    # Sort q values (should already be sorted, but to be safe)
    q_values = np.sort(q_values)
    
    # Calculate spacings between adjacent peaks
    spacings = np.diff(q_values)
    
    # Calculate pitch as 2π/mean(spacing)
    pitch = 2 * np.pi / np.mean(spacings)
    
    return pitch, spacings

def FreeFormTrapezoid(Coord,Qx,Qz,Trapnumber):
    H1 = Coord[0,3]
    H2 = Coord[0,3]
    form=np.zeros([len(Qx[:])]) # initialize structure of the amplitude - (labeled form here)
    for i in range(int(Trapnumber)): # edit this to remove the need for the trapnumber variable
        H2 = H2+Coord[i,2]
        if i > 0:
            H1 = H1+Coord[i-1,2] 
        x1 = Coord[i,0]
        x4 = Coord[i,1]
        x2 = Coord[i+1,0]
        x3 = Coord[i+1,1]
        if x2==x1:
            x2=x2-0.000001
        if x4==x3:
            x4=x4-0.000001
        SL = Coord[i,2]/(x2-x1)
        SR = -Coord[i,2]/(x4-x3)
        
        A1 = (np.exp(1j*Qx*((H1-SR*x4)/SR))/(Qx/SR+Qz))*(np.exp(-1j*H2*(Qx/SR+Qz))-np.exp(-1j*H1*(Qx/SR+Qz)))
        A2 = (np.exp(1j*Qx*((H1-SL*x1)/SL))/(Qx/SL+Qz))*(np.exp(-1j*H2*(Qx/SL+Qz))-np.exp(-1j*H1*(Qx/SL+Qz)))
        form=form+(1j/Qx)*(A1-A2)*Coord[i,4]
    return form





def SimInt_BCP(Qx,Pitch_mean,Fraction,I0,DW,Bk):
    # The background is assumed to be 1
    Trapnumber=1
    Qz=np.zeros_like(Qx)
    TPAR=np.zeros([Trapnumber+1,2])
   
    TPAR[0,0]=Fraction*Pitch_mean; TPAR[0,1]=20 
    TPAR[1,0]=Fraction*Pitch_mean; TPAR[1,1]=0
    Coord= CD.SymCoordAssign_SingleMaterial(TPAR)
    F1 = FreeFormTrapezoid(Coord[:,:,0],Qx,Qz,Trapnumber) 
    M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(DW,2)),0.5)
    Formfactor=F1*M
    Formfactor=abs(Formfactor)
    SimInt = np.power(Formfactor,2)*I0+Bk  #+SPAR[2]
    return SimInt

def SimIntLog_BCP(Qx,Pitch_mean,Fraction,I0,DW,Bk):
    # The background is assumed to be 1
    Trapnumber=1
    Qz=np.zeros_like(Qx)
    TPAR=np.zeros([Trapnumber+1,2])
    SLD=np.zeros([Trapnumber+1,1])
    TPAR[0,0]=Fraction*Pitch_mean; TPAR[0,1]=20; SLD[0,0]=1; # Assumes binary (0,1) contrast
    TPAR[1,0]=Fraction*Pitch_mean; TPAR[1,1]=0;SLD[1,0]=1;
    Coord= CD.SymCoordAssign_SingleMaterial(TPAR)
    F1 = FreeFormTrapezoid(Coord[:,:,0],Qx,Qz,Trapnumber) 
    M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(DW,2)),0.5)
    Formfactor=F1*M
    Formfactor=abs(Formfactor)
    SimInt = np.power(Formfactor,2)*I0+Bk  #+SPAR[2]
    return np.log(SimInt)


def BCPSimFit(data, Pitch_mean, peaks, Fraction, DW, I0,Bk, logfit=False):
    param_init=[Pitch_mean,Fraction,I0, DW,Bk]
    bounds=([Pitch_mean*0.99,Fraction-0.1,I0*0.01, DW-15,Bk*0.95], [Pitch_mean*1.01,Fraction+0.1,I0*100, DW+15,Bk*1.05])
    LD=np.log(data[:,1][peaks])
    if logfit==False:
        params,params_cov=scipy.optimize.curve_fit(SimInt_BCP, data[:,0][peaks], data[:,1][peaks],p0=param_init,bounds=bounds)
        # plot results
        plt.plot(data[:,0],data[:,1], label='Exp Data')
        plt.plot(data[:,0][peaks], SimInt_BCP(data[:,0][peaks], params[0], params[1],params[2],params[3],params[4]),color='r',label='Fitted function')
        plt.plot(data[:,0][peaks],data[:,1][peaks],color='r',marker='o',alpha =0.4)
        plt.yscale('log')
        plt.xlabel('q ($A^{-1}$)', fontsize =16)
        plt.ylabel('Intensity (a.u)', fontsize =16)
        plt.legend(loc='upper right')
    elif logfit== True:
        params,params_cov=scipy.optimize.curve_fit(SimIntLog_BCP, data[:,0][peaks], LD,p0=param_init,bounds=bounds)
    return params, params_cov

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from typing import Tuple, Optional, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from typing import Tuple, Optional, Dict, Any, Union

def BCPSimFit_V2(data: np.ndarray, 
                Pitch_mean: float, 
                Fraction: float, 
                DW: float, 
                I0: float,
                Bk: float, 
                additional_data: Optional[np.ndarray] = None,
                logfit: bool = False,
                plot_results: bool = True,
                figsize: Tuple[int, int] = (10, 6),
                title: Optional[str] = None,
                fit_pitch: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit scattering data with a block copolymer (BCP) model and visualize the results.
    
    Parameters:
    -----------
    data : np.ndarray of shape (n, 2)
        Array with [Q, Intensity] values to fit
    Pitch_mean : float
        Pitch value for the model (fixed if fit_pitch=False)
    Fraction : float
        Initial fraction value for the model (typically between 0 and 1)
    DW : float
        Initial Debye-Waller factor for the model
    I0 : float
        Initial intensity scaling factor
    Bk : float
        Initial background value
    additional_data : np.ndarray of shape (m, 2), optional
        Additional [Q, Intensity] data to plot alongside the fitted data
    logfit : bool, optional
        Whether to fit in log space (default: False)
    plot_results : bool, optional
        Whether to plot the fitting results (default: True)
    figsize : tuple of int, optional
        Figure size as (width, height) in inches (default: (10, 6))
    title : str, optional
        Custom title for the plot (default: None)
    fit_pitch : bool, optional
        Whether to fit the Pitch parameter or keep it fixed (default: False)
        
    Returns:
    --------
    params : np.ndarray
        Optimized parameters [Pitch, Fraction, I0, DW, Bk]
    params_cov : np.ndarray
        Covariance matrix of the optimized parameters
        
    Raises:
    -------
    ValueError
        If data is not properly formatted or has wrong dimensions
    RuntimeError
        If the fitting process fails to converge
    """
    # Validate input data
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be a numpy array of shape (n, 2) containing [Q, Intensity] values")
    
    if additional_data is not None and (not isinstance(additional_data, np.ndarray) or 
                                       additional_data.ndim != 2 or 
                                       additional_data.shape[1] != 2):
        raise ValueError("additional_data must be a numpy array of shape (m, 2) containing [Q, Intensity] values")
    
    # Define wrapper functions to handle fixed Pitch
    if not fit_pitch:
        # Create wrappers that fix the Pitch parameter
        def SimInt_BCP_fixed_pitch(q, Fraction, I0, DW, Bk):
            return SimInt_BCP(q, Pitch_mean, Fraction, I0, DW, Bk)
            
        def SimIntLog_BCP_fixed_pitch(q, Fraction, I0, DW, Bk):
            return SimIntLog_BCP(q, Pitch_mean, Fraction, I0, DW, Bk)
        
        # Initial parameters and bounds (without Pitch)
        param_init = [Fraction, I0, DW, Bk]
        bounds = ([Fraction-0.1, I0*0.01, DW-15, Bk*0.95], 
                 [Fraction+0.1, I0*100, DW+15, Bk*1.05])
    else:
        # Use the full model with Pitch as a fitted parameter
        # Initial parameters and bounds
        param_init = [Pitch_mean, Fraction, I0, DW, Bk]
        bounds = ([Pitch_mean*0.99, Fraction-0.1, I0*0.01, DW-15, Bk*0.95], 
                 [Pitch_mean*1.01, Fraction+0.1, I0*100, DW+15, Bk*1.05])
    
    # Check for valid bounds
    for i, (lower, upper) in enumerate(zip(bounds[0], bounds[1])):
        if lower >= upper:
            param_names = ['Fraction', 'I0', 'DW', 'Bk'] if not fit_pitch else ['Pitch', 'Fraction', 'I0', 'DW', 'Bk']
            raise ValueError(f"Invalid bounds for parameter {param_names[i]}: lower bound ({lower}) must be less than upper bound ({upper})")
    
    # Compute log of intensities for log fitting
    LD = np.log(data[:, 1])
    
    # Fit the data
    try:
        if not fit_pitch:
            # Use fixed pitch functions
            if not logfit:
                params_fit, params_cov = scipy.optimize.curve_fit(
                    SimInt_BCP_fixed_pitch, data[:, 0], data[:, 1],
                    p0=param_init, bounds=bounds
                )
                # Insert fixed Pitch at the beginning of the parameter array
                params = np.insert(params_fit, 0, Pitch_mean)
            else:
                params_fit, params_cov = scipy.optimize.curve_fit(
                    SimIntLog_BCP_fixed_pitch, data[:, 0], LD,
                    p0=param_init, bounds=bounds
                )
                # Insert fixed Pitch at the beginning of the parameter array
                params = np.insert(params_fit, 0, Pitch_mean)
        else:
            # Fit with Pitch as a variable
            if not logfit:
                params, params_cov = scipy.optimize.curve_fit(
                    SimInt_BCP, data[:, 0], data[:, 1],
                    p0=param_init, bounds=bounds
                )
            else:
                params, params_cov = scipy.optimize.curve_fit(
                    SimIntLog_BCP, data[:, 0], LD,
                    p0=param_init, bounds=bounds
                )
    except Exception as e:
        raise RuntimeError(f"Fitting failed: {str(e)}")
    
    # Plot the results if requested
    if plot_results:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot additional data first if provided
        if additional_data is not None:
            plt.plot(additional_data[:, 0], additional_data[:, 1], 'b-', 
                    label='Additional Data', alpha=0.7, linewidth=1)
        
        # Plot experimental data
        plt.plot(data[:, 0], data[:, 1], 'k-', label='Exp Data', linewidth=1)
        
        # Plot fitted function
        plt.plot(data[:, 0], SimInt_BCP(data[:, 0], params[0], params[1], params[2], params[3], params[4]),
                'r-', label='Fitted Function', linewidth=2)
        
        # Plot experimental data points
        plt.scatter(data[:, 0], data[:, 1], color='r', marker='o', alpha=0.4, s=20)
        
        # Set log scale for y-axis
        plt.yscale('log')
        
        # Add grid
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Set labels and title
        plt.xlabel('q (Å$^{-1}$)', fontsize=16)
        plt.ylabel('Intensity (a.u)', fontsize=16)
        
        if title:
            plt.title(title, fontsize=16)
        else:
            pitch_status = "Variable" if fit_pitch else "Fixed"
            plt.title(f'BCP Model Fit (Pitch: {pitch_status})', fontsize=16)
        
        # Add a text box with fit parameters
        param_names = ['Pitch', 'Fraction', 'I0', 'DW', 'Bk']
        param_text = "Fitted Parameters:\n"
        for i, (name, val) in enumerate(zip(param_names, params)):
            if name == 'Pitch' and not fit_pitch:
                param_text += f"{name}: {val:.4e} (fixed)\n"
            else:
                param_text += f"{name}: {val:.4e}\n"
        
        # Add text box with parameters
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.05, param_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # Add legend
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    return params, params_cov


def BCPSimFit_DWSweep(data: np.ndarray, 
                      Pitch_mean: float, 
                      Fraction: float, 
                      DW_start: float, 
                      DW_end: float, 
                      DW_steps: int, 
                      I0: float, 
                      Bk: float, 
                      additional_data: Optional[np.ndarray] = None,
                      highlight_percent: float = 5, 
                      logfit: bool = False, 
                      plot_results: bool = True,
                      fit_pitch: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit scattering data while sweeping the Debye-Waller (DW) parameter over a range and find the optimal value.
    
    Parameters:
    -----------
    data : np.ndarray of shape (n, 2)
        Array with [Q, Intensity] values to fit
    Pitch_mean : float
        Pitch value for the model (fixed if fit_pitch=False)
    Fraction : float
        Initial fraction value for the model (typically between 0 and 1)
    DW_start : float
        Lower bound for DW parameter sweep
    DW_end : float
        Upper bound for DW parameter sweep
    DW_steps : int
        Number of steps between DW_start and DW_end
    I0 : float
        Initial intensity scaling factor
    Bk : float
        Initial background value
    additional_data : np.ndarray of shape (m, 2), optional
        Additional [Q, Intensity] data to plot alongside the fitted data
    highlight_percent : float, optional
        Percentage difference from best fit to highlight (default: 5%)
    logfit : bool, optional
        Whether to fit in log space (default: False)
    plot_results : bool, optional
        Whether to plot the results (default: True)
    fit_pitch : bool, optional
        Whether to fit the Pitch parameter or keep it fixed (default: False)
        
    Returns:
    --------
    best_params : np.ndarray
        Best fit parameters [Pitch, Fraction, I0, DW, Bk]
    best_cov : np.ndarray or None
        Covariance matrix for best fit (may be None if not available)
    dw_values : np.ndarray
        Array of DW values used in sweep
    gf_values : np.ndarray
        Goodness of fit values for each DW value
        
    Raises:
    -------
    ValueError
        If input data is not properly formatted or if all fits fail
    RuntimeError
        If the fitting process fails completely
    """
    # Validate input data
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be a numpy array of shape (n, 2) containing [Q, Intensity] values")
    
    if additional_data is not None and (not isinstance(additional_data, np.ndarray) or 
                                       additional_data.ndim != 2 or 
                                       additional_data.shape[1] != 2):
        raise ValueError("additional_data must be a numpy array of shape (m, 2) containing [Q, Intensity] values")
    
    if DW_steps < 2:
        raise ValueError("DW_steps must be at least 2")
    
    if DW_start >= DW_end:
        raise ValueError("DW_start must be less than DW_end")

    # Create array of DW values to sweep
    dw_values = np.linspace(DW_start, DW_end, DW_steps)
    
    # Initialize arrays to store fit results
    gf_values = np.zeros(DW_steps)
    all_params = []
    all_covs = []
    
    # Define goodness of fit (GF) function - lower is better
    def calc_gf(y_true, y_pred):
        if logfit:
            # For log fit, use mean squared error in log space
            y_true_log = np.log(y_true)
            y_pred_log = np.log(y_pred)
            return np.mean((y_true_log - y_pred_log) ** 2)
        else:
            # For linear fit, use normalized chi-squared
            # (sum of squared relative errors)
            return np.mean(((y_true - y_pred) / y_true) ** 2)
    
    LD = np.log(data[:, 1])
    
    # Sweep through DW values
    for i, dw in enumerate(dw_values):
        try:
            if fit_pitch:
                # Set initial parameters and bounds for fitting Pitch, Fraction, I0, and Bk
                param_init = [Pitch_mean, Fraction, I0, Bk]
                bounds = ([Pitch_mean * 0.99, Fraction - 0.1, I0 * 0.01, Bk * 0.95], 
                          [Pitch_mean * 1.01, Fraction + 0.1, I0 * 100, Bk * 1.05])
                
                if not logfit:
                    # Create a wrapper function with fixed DW
                    def wrapper(q, Pitch, Fraction, I0, Bk):
                        return SimInt_BCP(q, Pitch, Fraction, I0, dw, Bk)
                    
                    params, params_cov = scipy.optimize.curve_fit(
                        wrapper, data[:, 0], data[:, 1], p0=param_init, bounds=bounds
                    )
                    
                    # Insert the fixed DW value into the parameter list at position 3
                    full_params = np.insert(params, 3, dw)
                    
                    # Calculate goodness of fit
                    y_pred = SimInt_BCP(data[:, 0], *full_params)
                    gf = calc_gf(data[:, 1], y_pred)
                else:
                    # Create a wrapper function with fixed DW for log fitting
                    def wrapper(q, Pitch, Fraction, I0, Bk):
                        return SimIntLog_BCP(q, Pitch, Fraction, I0, dw, Bk)
                    
                    params, params_cov = scipy.optimize.curve_fit(
                        wrapper, data[:, 0], LD, p0=param_init, bounds=bounds
                    )
                    
                    # Insert the fixed DW value into the parameter list at position 3
                    full_params = np.insert(params, 3, dw)
                    
                    # Calculate goodness of fit
                    y_pred = SimInt_BCP(data[:, 0], *full_params)
                    gf = calc_gf(data[:, 1], y_pred)
            else:
                # Fixed Pitch: set initial parameters and bounds for fitting only Fraction, I0, and Bk
                param_init = [Fraction, I0, Bk]
                bounds = ([Fraction - 0.1, I0 * 0.01, Bk * 0.95], 
                          [Fraction + 0.1, I0 * 100, Bk * 1.05])
                
                if not logfit:
                    # Create a wrapper function with fixed Pitch and DW
                    def wrapper(q, Fraction, I0, Bk):
                        return SimInt_BCP(q, Pitch_mean, Fraction, I0, dw, Bk)
                    
                    params, params_cov = scipy.optimize.curve_fit(
                        wrapper, data[:, 0], data[:, 1], p0=param_init, bounds=bounds
                    )
                    
                    # Insert Pitch and DW into the parameter list
                    full_params = np.array([Pitch_mean, params[0], params[1], dw, params[2]])
                    
                    # Calculate goodness of fit
                    y_pred = SimInt_BCP(data[:, 0], *full_params)
                    gf = calc_gf(data[:, 1], y_pred)
                else:
                    # Create a wrapper function with fixed Pitch and DW for log fitting
                    def wrapper(q, Fraction, I0, Bk):
                        return SimIntLog_BCP(q, Pitch_mean, Fraction, I0, dw, Bk)
                    
                    params, params_cov = scipy.optimize.curve_fit(
                        wrapper, data[:, 0], LD, p0=param_init, bounds=bounds
                    )
                    
                    # Insert Pitch and DW into the parameter list
                    full_params = np.array([Pitch_mean, params[0], params[1], dw, params[2]])
                    
                    # Calculate goodness of fit
                    y_pred = SimInt_BCP(data[:, 0], *full_params)
                    gf = calc_gf(data[:, 1], y_pred)
                
            # Store results
            gf_values[i] = gf
            all_params.append(full_params)
            all_covs.append(params_cov)
        except Exception as e:
            print(f"Fit failed for DW={dw:.4f}: {e}")
            gf_values[i] = np.nan
            all_params.append(None)
            all_covs.append(None)
    
    # Find best fit
    valid_indices = ~np.isnan(gf_values)
    if not np.any(valid_indices):
        raise ValueError("All fits failed. Try adjusting the parameter ranges or the data.")
    
    best_idx = np.argmin(gf_values[valid_indices])
    best_idx = np.where(valid_indices)[0][best_idx]
    best_params = all_params[best_idx]
    best_cov = all_covs[best_idx]
    best_dw = dw_values[best_idx]
    best_gf = gf_values[best_idx]
    
    # Print best fit info
    pitch_status = "variable" if fit_pitch else "fixed"
    print(f"Best fit found at DW = {best_dw:.4f} with goodness of fit = {best_gf:.4e} (Pitch: {pitch_status})")
    print(f"Best parameters: Pitch = {best_params[0]:.4f}" + (" (fixed)" if not fit_pitch else "") + 
          f", Fraction = {best_params[1]:.4f}, I0 = {best_params[2]:.4e}, " +
          f"DW = {best_params[3]:.4f}, Bk = {best_params[4]:.4e}")
    
    if plot_results:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot additional data first if provided
        if additional_data is not None:
            ax1.plot(additional_data[:, 0], additional_data[:, 1], 'b-', 
                    label='Additional Data', alpha=0.7, linewidth=1)
        
        # Plot original data and best fit
        ax1.plot(data[:, 0], data[:, 1], 'k-', label='Exp Data', linewidth=1)
        ax1.plot(data[:, 0], SimInt_BCP(data[:, 0], *best_params), 
                color='r', label=f'Best Fit (DW={best_dw:.2f})', linewidth=2)
        ax1.scatter(data[:, 0], data[:, 1], color='r', alpha=0.4, s=20)
        ax1.set_yscale('log')
        ax1.set_xlabel('q (Å$^{-1}$)', fontsize=14)
        ax1.set_ylabel('Intensity (a.u)', fontsize=14)
        ax1.set_title(f'Best Fit Result (Pitch: {pitch_status})', fontsize=16)
        ax1.legend(loc='upper right')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Plot GF vs DW
        # Calculate highlight threshold
        threshold = best_gf * (1 + highlight_percent / 100)
        within_threshold = gf_values <= threshold
        
        # Plot points outside threshold
        outside_points = ~within_threshold & valid_indices
        if np.any(outside_points):
            ax2.scatter(dw_values[outside_points], gf_values[outside_points], 
                       color='blue', alpha=0.7, s=30, label='Outside threshold')
        
        # Plot points within threshold
        within_points = within_threshold & valid_indices
        if np.any(within_points):
            ax2.scatter(dw_values[within_points], gf_values[within_points], 
                       color='green', alpha=0.9, s=50, label=f'Within {highlight_percent}%')
        
        # Highlight best point
        ax2.scatter(best_dw, best_gf, color='red', s=100, edgecolor='black', linewidth=1.5, 
                   label=f'Best Fit (DW={best_dw:.2f}, GF={best_gf:.2e})')
        
        # Add a line connecting all points if we have enough valid points
        if np.sum(valid_indices) > 1:
            # Sort points for proper line connection
            sort_idx = np.argsort(dw_values[valid_indices])
            sorted_dw = dw_values[valid_indices][sort_idx]
            sorted_gf = gf_values[valid_indices][sort_idx]
            ax2.plot(sorted_dw, sorted_gf, 'k-', alpha=0.5)
        
        ax2.set_xlabel('Debye-Waller Factor', fontsize=14)
        ax2.set_ylabel('Goodness of Fit (lower is better)', fontsize=14)
        ax2.set_title('DW Parameter Sweep', fontsize=16)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')
        
        # Show the best parameters
        param_names = ['Pitch', 'Fraction', 'I0', 'DW', 'Bk']
        param_text = "Best Parameters:\n"
        for i, (name, val) in enumerate(zip(param_names, best_params)):
            if name == 'Pitch' and not fit_pitch:
                param_text += f"{name}: {val:.4e} (fixed)\n"
            else:
                param_text += f"{name}: {val:.4e}\n"
        
        # Add text box with best parameters
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.05, param_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
        # Also create a focused view around the best DW value if there are enough points within threshold
        if np.sum(within_points) > 1:
            plt.figure(figsize=(8, 6))
            
            # Find range of DW values within threshold
            dw_within = dw_values[within_points]
            min_dw = np.min(dw_within)
            max_dw = np.max(dw_within)
            
            # Create a more detailed range around the best value
            padding = (max_dw - min_dw) * 0.2
            detailed_dw = np.linspace(max(min_dw - padding, DW_start), 
                                      min(max_dw + padding, DW_end), 100)
            
            # Interpolate to create a smooth curve if we have enough points
            if np.sum(valid_indices) > 3:
                try:
                    from scipy.interpolate import interp1d
                    # First sort the data points by DW value
                    sort_idx = np.argsort(dw_values[valid_indices])
                    sorted_dw = dw_values[valid_indices][sort_idx]
                    sorted_gf = gf_values[valid_indices][sort_idx]
                    
                    interp_func = interp1d(sorted_dw, sorted_gf, kind='cubic', 
                                          bounds_error=False, fill_value='extrapolate')
                    smooth_gf = interp_func(detailed_dw)
                    plt.plot(detailed_dw, smooth_gf, 'k-', alpha=0.7, label='Interpolated Curve')
                except Exception as e:
                    print(f"Interpolation failed: {e}")
                    # If interpolation fails, just connect the points
                    plt.plot(sorted_dw, sorted_gf, 'k-', alpha=0.5, label='Trend Line')
            elif np.sum(valid_indices) > 1:
                # If we have at least 2 points, draw a simple line
                sort_idx = np.argsort(dw_values[valid_indices])
                sorted_dw = dw_values[valid_indices][sort_idx]
                sorted_gf = gf_values[valid_indices][sort_idx]
                plt.plot(sorted_dw, sorted_gf, 'k-', alpha=0.5, label='Trend Line')
            
            # Plot points within threshold
            plt.scatter(dw_values[within_points], gf_values[within_points], 
                       color='green', alpha=0.9, s=50, label=f'Within {highlight_percent}%')
            
            # Highlight best point
            plt.scatter(best_dw, best_gf, color='red', s=100, edgecolor='black', linewidth=1.5,
                       label=f'Best Fit (DW={best_dw:.2f})')
            
            plt.xlabel('Debye-Waller Factor', fontsize=14)
            plt.ylabel('Goodness of Fit (lower is better)', fontsize=14)
            plt.title(f'DW Parameter Sweep (Optimal Region, Pitch: {pitch_status})', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='best')
            
            # Add horizontal line at threshold value
            plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                       label=f'{highlight_percent}% Threshold')
            
            plt.tight_layout()
            plt.show()
    
    return best_params, best_cov, dw_values, gf_values


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from typing import Tuple, Optional, List, Dict, Any, Union
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

def BCPSimFit_2DParameterSweep(data: np.ndarray, 
                              Pitch_mean: float,
                              Fraction_start: float,
                              Fraction_end: float,
                              Fraction_steps: int,
                              DW_start: float, 
                              DW_end: float, 
                              DW_steps: int, 
                              I0: float, 
                              Bk: float, 
                              additional_data: Optional[np.ndarray] = None,
                              highlight_percent: float = 5, 
                              logfit: bool = False, 
                              plot_results: bool = True,
                              fit_pitch: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Fit scattering data while sweeping both the Debye-Waller (DW) and Fraction parameters
    over a 2D grid of values to find the optimal combination.
    
    Parameters:
    -----------
    data : np.ndarray of shape (n, 2)
        Array with [Q, Intensity] values to fit
    Pitch_mean : float
        Pitch value for the model (fixed if fit_pitch=False)
    Fraction_start : float
        Lower bound for Fraction parameter sweep
    Fraction_end : float
        Upper bound for Fraction parameter sweep
    Fraction_steps : int
        Number of steps between Fraction_start and Fraction_end
    DW_start : float
        Lower bound for DW parameter sweep
    DW_end : float
        Upper bound for DW parameter sweep
    DW_steps : int
        Number of steps between DW_start and DW_end
    I0 : float
        Initial intensity scaling factor
    Bk : float
        Initial background value
    additional_data : np.ndarray of shape (m, 2), optional
        Additional [Q, Intensity] data to plot alongside the fitted data
    highlight_percent : float, optional
        Percentage difference from best fit to highlight (default: 5%)
    logfit : bool, optional
        Whether to fit in log space (default: False)
    plot_results : bool, optional
        Whether to plot the results (default: True)
    fit_pitch : bool, optional
        Whether to fit the Pitch parameter or keep it fixed (default: False)
        
    Returns:
    --------
    best_params : np.ndarray
        Best fit parameters [Pitch, Fraction, I0, DW, Bk]
    best_cov : np.ndarray or None
        Covariance matrix for best fit (may be None if not available)
    sweep_results : dict
        Dictionary containing:
        - 'dw_values': Array of DW values used in sweep
        - 'fraction_values': Array of Fraction values used in sweep
        - 'gf_values': 2D array of goodness of fit values for each parameter combination
        - 'parameters': 2D array of parameter sets for each combination
        
    Raises:
    -------
    ValueError
        If input data is not properly formatted or if all fits fail
    RuntimeError
        If the fitting process fails completely
    """
    # Validate input data
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be a numpy array of shape (n, 2) containing [Q, Intensity] values")
    
    if additional_data is not None and (not isinstance(additional_data, np.ndarray) or 
                                       additional_data.ndim != 2 or 
                                       additional_data.shape[1] != 2):
        raise ValueError("additional_data must be a numpy array of shape (m, 2) containing [Q, Intensity] values")
    
    if DW_steps < 2 or Fraction_steps < 2:
        raise ValueError("DW_steps and Fraction_steps must both be at least 2")
    
    if DW_start >= DW_end:
        raise ValueError("DW_start must be less than DW_end")
        
    if Fraction_start >= Fraction_end:
        raise ValueError("Fraction_start must be less than Fraction_end")

    # Create arrays of parameter values to sweep
    dw_values = np.linspace(DW_start, DW_end, DW_steps)
    fraction_values = np.linspace(Fraction_start, Fraction_end, Fraction_steps)
    
    # Create meshgrid for 2D parameter space
    DW_grid, Fraction_grid = np.meshgrid(dw_values, fraction_values)
    
    # Initialize arrays to store fit results
    gf_values = np.full((Fraction_steps, DW_steps), np.nan)
    all_params = np.full((Fraction_steps, DW_steps), None, dtype=object)
    all_covs = np.full((Fraction_steps, DW_steps), None, dtype=object)
    
    # Define goodness of fit (GF) function - lower is better
    def calc_gf(y_true, y_pred):
        if logfit:
            # For log fit, use mean squared error in log space
            y_true_log = np.log(y_true)
            y_pred_log = np.log(y_pred)
            return np.mean((y_true_log - y_pred_log) ** 2)
        else:
            # For linear fit, use normalized chi-squared
            # (sum of squared relative errors)
            return np.mean(((y_true - y_pred) / y_true) ** 2)
    
    LD = np.log(data[:, 1])
    
    # Track progress
    total_fits = Fraction_steps * DW_steps
    successful_fits = 0
    
    print(f"Starting 2D parameter sweep with {total_fits} combinations...")
    
    # Sweep through parameter combinations
    for i, fraction in enumerate(fraction_values):
        for j, dw in enumerate(dw_values):
            try:
                if fit_pitch:
                    # Set initial parameters and bounds for fitting Pitch, I0, and Bk
                    # (Fraction and DW are fixed)
                    param_init = [Pitch_mean, I0, Bk]
                    bounds = ([Pitch_mean * 0.99, I0 * 0.01, Bk * 0.95], 
                              [Pitch_mean * 1.01, I0 * 100, Bk * 1.05])
                    
                    if not logfit:
                        # Create a wrapper function with fixed Fraction and DW
                        def wrapper(q, Pitch, I0, Bk):
                            return SimInt_BCP(q, Pitch, fraction, I0, dw, Bk)
                        
                        params, params_cov = scipy.optimize.curve_fit(
                            wrapper, data[:, 0], data[:, 1], p0=param_init, bounds=bounds
                        )
                        
                        # Insert the fixed parameters into the parameter list
                        full_params = np.array([params[0], fraction, params[1], dw, params[2]])
                        
                        # Calculate goodness of fit
                        y_pred = SimInt_BCP(data[:, 0], *full_params)
                        gf = calc_gf(data[:, 1], y_pred)
                    else:
                        # Create a wrapper function with fixed Fraction and DW for log fitting
                        def wrapper(q, Pitch, I0, Bk):
                            return SimIntLog_BCP(q, Pitch, fraction, I0, dw, Bk)
                        
                        params, params_cov = scipy.optimize.curve_fit(
                            wrapper, data[:, 0], LD, p0=param_init, bounds=bounds
                        )
                        
                        # Insert the fixed parameters into the parameter list
                        full_params = np.array([params[0], fraction, params[1], dw, params[2]])
                        
                        # Calculate goodness of fit
                        y_pred = SimInt_BCP(data[:, 0], *full_params)
                        gf = calc_gf(data[:, 1], y_pred)
                else:
                    # Fixed Pitch: set initial parameters and bounds for fitting only I0 and Bk
                    # (Pitch, Fraction, and DW are all fixed)
                    param_init = [I0, Bk]
                    bounds = ([I0 * 0.01, Bk * 0.95], 
                              [I0 * 100, Bk * 1.05])
                    
                    if not logfit:
                        # Create a wrapper function with fixed Pitch, Fraction, and DW
                        def wrapper(q, I0, Bk):
                            return SimInt_BCP(q, Pitch_mean, fraction, I0, dw, Bk)
                        
                        params, params_cov = scipy.optimize.curve_fit(
                            wrapper, data[:, 0], data[:, 1], p0=param_init, bounds=bounds
                        )
                        
                        # Insert the fixed parameters into the parameter list
                        full_params = np.array([Pitch_mean, fraction, params[0], dw, params[1]])
                        
                        # Calculate goodness of fit
                        y_pred = SimInt_BCP(data[:, 0], *full_params)
                        gf = calc_gf(data[:, 1], y_pred)
                    else:
                        # Create a wrapper function with fixed Pitch, Fraction, and DW for log fitting
                        def wrapper(q, I0, Bk):
                            return SimIntLog_BCP(q, Pitch_mean, fraction, I0, dw, Bk)
                        
                        params, params_cov = scipy.optimize.curve_fit(
                            wrapper, data[:, 0], LD, p0=param_init, bounds=bounds
                        )
                        
                        # Insert the fixed parameters into the parameter list
                        full_params = np.array([Pitch_mean, fraction, params[0], dw, params[1]])
                        
                        # Calculate goodness of fit
                        y_pred = SimInt_BCP(data[:, 0], *full_params)
                        gf = calc_gf(data[:, 1], y_pred)
                
                # Store results
                gf_values[i, j] = gf
                all_params[i, j] = full_params
                all_covs[i, j] = params_cov
                successful_fits += 1
                
                # Print progress every 10% completion
                progress = (i * DW_steps + j + 1) / total_fits * 100
                if (i * DW_steps + j + 1) % max(1, total_fits // 10) == 0:
                    print(f"Progress: {progress:.1f}% ({successful_fits} successful fits)")
                
            except Exception as e:
                # Quietly continue on failure - we'll handle the NaN values later
                pass
    
    # Find best fit
    valid_mask = ~np.isnan(gf_values)
    if not np.any(valid_mask):
        raise ValueError("All fits failed. Try adjusting the parameter ranges or the data.")
    
    best_idx = np.nanargmin(gf_values.flatten())
    best_i, best_j = np.unravel_index(best_idx, gf_values.shape)
    best_params = all_params[best_i, best_j]
    best_cov = all_covs[best_i, best_j]
    best_fraction = fraction_values[best_i]
    best_dw = dw_values[best_j]
    best_gf = gf_values[best_i, best_j]
    
    print(f"\nCompleted {successful_fits} of {total_fits} fits successfully.")
    
    # Print best fit info
    pitch_status = "variable" if fit_pitch else "fixed"
    print(f"Best fit found at Fraction = {best_fraction:.4f}, DW = {best_dw:.4f} with goodness of fit = {best_gf:.4e}")
    print(f"Best parameters: Pitch = {best_params[0]:.4f}" + (" (fixed)" if not fit_pitch else "") + 
          f", Fraction = {best_params[1]:.4f}, I0 = {best_params[2]:.4e}, " +
          f"DW = {best_params[3]:.4f}, Bk = {best_params[4]:.4e}")
    
    if plot_results:
        # Create a figure with three subplots (1 for fit, 2 for parameter space visualization)
        fig = plt.figure(figsize=(18, 6))
        
        # Create a 1x3 grid for subplots
        gs = fig.add_gridspec(1, 3)
        
        # First subplot: Best fit
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot additional data first if provided
        if additional_data is not None:
            ax1.plot(additional_data[:, 0], additional_data[:, 1], 'b-', 
                    label='Additional Data', alpha=0.7, linewidth=1)
        
        # Plot original data and best fit
        ax1.plot(data[:, 0], data[:, 1], 'k-', label='Exp Data', linewidth=1)
        ax1.plot(data[:, 0], SimInt_BCP(data[:, 0], *best_params), 
                color='r', label=f'Best Fit', linewidth=2)
        ax1.scatter(data[:, 0], data[:, 1], color='r', alpha=0.4, s=20)
        ax1.set_yscale('log')
        ax1.set_xlabel('q (Å$^{-1}$)', fontsize=14)
        ax1.set_ylabel('Intensity (a.u)', fontsize=14)
        ax1.set_title(f'Best Fit Result', fontsize=16)
        ax1.legend(loc='upper right')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Show the best parameters
        param_names = ['Pitch', 'Fraction', 'I0', 'DW', 'Bk']
        param_text = "Best Parameters:\n"
        for i, (name, val) in enumerate(zip(param_names, best_params)):
            if name == 'Pitch' and not fit_pitch:
                param_text += f"{name}: {val:.4e} (fixed)\n"
            else:
                param_text += f"{name}: {val:.4e}\n"
        
        # Add text box with best parameters
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.05, param_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # Second subplot: 2D heatmap of goodness of fit
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Create a mask for invalid values
        masked_gf = np.ma.masked_invalid(gf_values)
        
        # Calculate threshold for highlighting
        threshold = best_gf * (1 + highlight_percent / 100)
        
        # Create a custom colormap that highlights the best values
        cmap = plt.cm.viridis.copy()
        cmap.set_bad('gray', 0.5)  # Gray color for masked values
        
        # Plot heatmap
        pcm = ax2.pcolormesh(DW_grid, Fraction_grid, masked_gf, 
                           norm=Normalize(vmin=np.nanmin(gf_values), vmax=threshold*2),
                           cmap=cmap, shading='auto')
        
        # Add a colorbar
        cbar = fig.colorbar(pcm, ax=ax2)
        cbar.set_label('Goodness of Fit (lower is better)', fontsize=12)
        
        # Add contour lines for the threshold
        if np.any(masked_gf < threshold):
            ax2.contour(DW_grid, Fraction_grid, masked_gf, 
                       levels=[threshold], colors='red', linestyles='dashed', 
                       linewidths=2, alpha=0.7)
        
        # Mark the best fit point
        ax2.plot(best_dw, best_fraction, 'ro', markersize=10, markeredgecolor='white')
        
        ax2.set_xlabel('Debye-Waller Factor', fontsize=14)
        ax2.set_ylabel('Fraction', fontsize=14)
        ax2.set_title('2D Parameter Space Goodness of Fit', fontsize=16)
        
        # Third subplot: 3D surface plot
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        
        # Create X, Y for plotting
        X, Y = np.meshgrid(dw_values, fraction_values)
        
        # Plot the 3D surface
        surf = ax3.plot_surface(X, Y, masked_gf, cmap='viridis', 
                               linewidth=0, antialiased=True, alpha=0.8)
        
        # Add a contour plot at the bottom
        cset = ax3.contourf(X, Y, masked_gf, zdir='z', offset=np.nanmin(gf_values),
                           cmap='viridis', alpha=0.5)
        
        # Mark the best fit point
        ax3.scatter([best_dw], [best_fraction], [best_gf], color='red', s=100, 
                   edgecolor='white', linewidth=1.5)
        
        ax3.set_xlabel('Debye-Waller Factor', fontsize=12)
        ax3.set_ylabel('Fraction', fontsize=12)
        ax3.set_zlabel('Goodness of Fit', fontsize=12)
        ax3.set_title('3D Parameter Space Visualization', fontsize=16)
        
        fig.tight_layout()
        plt.show()
        
        # Also create a focused view around the best parameter combination
        # Calculate the region of interest (within threshold)
        try:
            within_threshold = (gf_values <= threshold) & (~np.isnan(gf_values))
            if np.any(within_threshold):
                plt.figure(figsize=(10, 8))
                
                # Find indices within threshold
                i_indices, j_indices = np.where(within_threshold)
                
                # Get min/max values with padding
                min_i, max_i = max(0, np.min(i_indices) - 1), min(Fraction_steps - 1, np.max(i_indices) + 1)
                min_j, max_j = max(0, np.min(j_indices) - 1), min(DW_steps - 1, np.max(j_indices) + 1)
                
                # Extract the focused region of the parameter grid
                focused_gf = gf_values[min_i:max_i+1, min_j:max_j+1]
                focused_fraction = fraction_values[min_i:max_i+1]
                focused_dw = dw_values[min_j:max_j+1]
                
                # Create new meshgrid for the focused region
                focused_X, focused_Y = np.meshgrid(focused_dw, focused_fraction)
                
                # Create a mask for invalid values
                masked_focused_gf = np.ma.masked_invalid(focused_gf)
                
                # Plot heatmap for the focused region
                plt.pcolormesh(focused_X, focused_Y, masked_focused_gf, 
                              norm=Normalize(vmin=np.nanmin(gf_values), vmax=threshold*1.2),
                              cmap=cmap, shading='auto')
                
                # Add contour for the threshold
                plt.contour(focused_X, focused_Y, masked_focused_gf, 
                           levels=[threshold], colors='red', linestyles='dashed', 
                           linewidths=2, alpha=0.7)
                
                # Mark the best fit point
                plt.plot(best_dw, best_fraction, 'ro', markersize=10, markeredgecolor='white')
                
                # Add colorbar
                cbar = plt.colorbar()
                cbar.set_label('Goodness of Fit (lower is better)', fontsize=12)
                
                plt.xlabel('Debye-Waller Factor', fontsize=14)
                plt.ylabel('Fraction', fontsize=14)
                plt.title(f'Focused View of Optimal Region (within {highlight_percent}%)', fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Could not create focused view: {e}")
    
    # Prepare return dictionary with sweep results
    sweep_results = {
        'dw_values': dw_values,
        'fraction_values': fraction_values,
        'gf_values': gf_values,
        'parameters': all_params
    }
    
    return best_params, best_cov, sweep_results


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from typing import Tuple, Optional, Dict, List, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from typing import Tuple, Optional, Dict, List, Any, Union

def create_3D_parameter_plot(sweep_results: Dict[str, Any], 
                             best_params: np.ndarray = None,
                             elev: float = 30, 
                             azim: float = -60,
                             figsize: Tuple[int, int] = (10, 8),
                             custom_cmap: str = 'viridis',
                             intensity_scale: Optional[Tuple[float, float]] = None,
                             highlight_threshold: Optional[float] = None,
                             title: Optional[str] = None,
                             show_contour: bool = True,
                             alpha_surface: float = 0.8,
                             zlabel: str = 'Goodness of Fit',
                             invert_z: bool = False,
                             add_colorbar: bool = True,
                             add_best_point: bool = True,
                             contour_offset: Optional[float] = None,
                             contour_colors: str = 'viridis',
                             contour_alpha: float = 0.5,
                             contour_levels: int = 10,
                             lightdir: Optional[Tuple[float, float, float]] = None):
    """
    Create a 3D surface plot with customizable perspective from parameter sweep results.
    
    Parameters:
    -----------
    sweep_results : dict
        Dictionary containing parameter sweep results, with keys:
        - 'dw_values': Array of DW values used in sweep
        - 'fraction_values': Array of Fraction values used in sweep
        - 'gf_values': 2D array of goodness of fit values for each parameter combination
    best_params : np.ndarray, optional
        Best fit parameters [Pitch, Fraction, I0, DW, Bk], used to mark best point
    elev : float, optional
        Elevation angle in degrees (default: 30)
    azim : float, optional
        Azimuth angle in degrees (default: -60)
    figsize : tuple(int, int), optional
        Figure size as (width, height) in inches (default: (10, 8))
    custom_cmap : str, optional
        Name of the colormap to use (default: 'viridis')
    intensity_scale : tuple(float, float), optional
        Custom scale for the intensity (vmin, vmax). If None, auto-scaling will be used.
    highlight_threshold : float, optional
        Threshold value for highlighting region of interest. If None, no threshold highlighting.
    title : str, optional
        Custom title for the plot. If None, a default title will be used.
    show_contour : bool, optional
        Whether to show contour plot at the bottom (default: True)
    alpha_surface : float, optional
        Alpha transparency for the 3D surface (default: 0.8)
    zlabel : str, optional
        Label for the z-axis (default: 'Goodness of Fit')
    invert_z : bool, optional
        Whether to invert the z-axis to show lower values at the top (default: False)
    add_colorbar : bool, optional
        Whether to add a colorbar to the plot (default: True)
    add_best_point : bool, optional
        Whether to highlight the best fit point (default: True)
    contour_offset : float, optional
        Offset for the contour plot at the bottom. If None, minimum value is used.
    contour_colors : str, optional
        Colormap for the contour plot (default: same as surface plot)
    contour_alpha : float, optional
        Alpha transparency for the contour plot (default: 0.5)
    contour_levels : int, optional
        Number of contour levels to show (default: 10)
    lightdir : tuple(float, float, float), optional
        Direction of the light source as (x, y, z). If None, default lighting is used.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The created axes with the 3D plot
    """
    # Extract parameter sweep data
    dw_values = sweep_results['dw_values']
    fraction_values = sweep_results['fraction_values']
    gf_values = sweep_results['gf_values']
    
    # Create a mask for invalid values
    masked_gf = np.ma.masked_invalid(gf_values)
    
    # Extract best point if available
    best_dw = None
    best_fraction = None
    best_gf = None
    if best_params is not None and len(best_params) >= 4:
        best_fraction = best_params[1]
        best_dw = best_params[3]
        
        # Find the closest grid point to the best parameters
        best_i = np.abs(fraction_values - best_fraction).argmin()
        best_j = np.abs(dw_values - best_dw).argmin()
        best_gf = gf_values[best_i, best_j]
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D surface
    X, Y = np.meshgrid(dw_values, fraction_values)
    
    # Determine intensity scale (vmin, vmax) for the plot
    if intensity_scale is None:
        vmin = np.nanmin(masked_gf)
        vmax = np.nanmax(masked_gf)
    else:
        vmin, vmax = intensity_scale
        
    # Set colormap
    cmap = plt.get_cmap(custom_cmap)
    
    # Plot the 3D surface
    surf = ax.plot_surface(X, Y, masked_gf, cmap=cmap, 
                          linewidth=0, antialiased=True, alpha=alpha_surface,
                          norm=Normalize(vmin=vmin, vmax=vmax))
    
    # Add contour at the bottom if requested
    if show_contour:
        # Determine contour offset
        if contour_offset is None:
            contour_offset = np.nanmin(masked_gf)
        
        # Create contour levels
        levels = np.linspace(vmin, vmax, contour_levels)
        
        # Add contour
        cset = ax.contourf(X, Y, masked_gf, zdir='z', offset=contour_offset,
                          cmap=contour_colors, alpha=contour_alpha, levels=levels)
    
    # Mark the best fit point if available and requested
    if add_best_point and best_dw is not None and best_fraction is not None and best_gf is not None:
        ax.scatter([best_dw], [best_fraction], [best_gf], color='red', s=100, 
                  edgecolor='white', linewidth=1.5)
    
    # Set plot labels
    ax.set_xlabel('Debye-Waller Factor', fontsize=12)
    ax.set_ylabel('Fraction', fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    
    # Set custom view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Invert z-axis if requested (to show lower values at the top)
    if invert_z:
        ax.invert_zaxis()
    
    # Set custom lighting if requested
    if lightdir is not None:
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=lightdir[0], altdeg=lightdir[1])
        # Apply lighting to the surface (this would require reconfiguring the surface plot)
        # In this case, just update the light position parameters
        surf._shade_colors = ls.shade_normals
    
    # Add colorbar if requested
    if add_colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Goodness of Fit (lower is better)', fontsize=12)
    
    # Set title
    if title is None:
        title = '3D Parameter Space Visualization'
    ax.set_title(title, fontsize=16)
    
    # Add threshold contour if requested
    if highlight_threshold is not None:
        # Add a horizontal plane at the threshold value
        try:
            xl, xh = ax.get_xlim()
            yl, yh = ax.get_ylim()
            xx, yy = np.meshgrid(np.linspace(xl, xh, 10), np.linspace(yl, yh, 10))
            zz = np.full_like(xx, highlight_threshold)
            
            # Plot the plane with some transparency
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')
            
            # Add text annotation for the threshold
            ax.text(
                (xl + xh) / 2,
                yl,
                highlight_threshold,
                f'Threshold: {highlight_threshold:.4e}',
                color='red', fontsize=10
            )
        except Exception as e:
            print(f"Could not add threshold plane: {e}")
    
    plt.tight_layout()
    
    return fig, ax

import numpy as np
from typing import Tuple, Union, Dict

def split_peak_data(peak_data: Union[np.ndarray, Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a peak data array with [Q, Intensity, integrated_intensity] columns
    into two separate arrays: [Q, Intensity] and [Q, integrated_intensity].
    
    Parameters:
    -----------
    peak_data : ndarray of shape (n, 3) or dict
        Either:
        - A numpy array with columns [Q, Intensity, integrated_intensity]
        - A dictionary with a 'peaks' key containing such an array
    
    Returns:
    --------
    peak_heights : ndarray of shape (n, 2)
        Array with columns [Q, Intensity] for peak heights
    peak_integrals : ndarray of shape (n, 2)
        Array with columns [Q, integrated_intensity] for integrated intensities
    
    Raises:
    -------
    ValueError
        If input array doesn't have the expected shape
    """
    # Handle dictionary input (extract 'peaks' array)
    if isinstance(peak_data, dict):
        if 'peaks' in peak_data and isinstance(peak_data['peaks'], np.ndarray):
            peak_array = peak_data['peaks']
        else:
            raise ValueError("Input dictionary must contain a 'peaks' key with a numpy array")
    else:
        peak_array = peak_data
    
    # Check input shape
    if not isinstance(peak_array, np.ndarray):
        raise ValueError("Input must be a numpy array or a dictionary containing a 'peaks' array")
    
    if peak_array.ndim != 2 or peak_array.shape[1] < 3:
        raise ValueError(f"Input array must have shape (n, 3+), but got {peak_array.shape}")
    
    # Extract columns
    q_values = peak_array[:, 0]
    intensities = peak_array[:, 1]
    integrated_intensities = peak_array[:, 2]
    
    # Create output arrays
    peak_heights = np.column_stack((q_values, intensities))
    peak_integrals = np.column_stack((q_values, integrated_intensities))
    
    return peak_heights, peak_integrals