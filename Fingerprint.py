

import numpy as np
import os
from typing import Optional, Union

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple

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

def find_peaks_by_sections(data, sections):
    """
    Find peaks in different sections of data with different parameters.
    
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
    
    Returns:
    --------
    peak_results : dict
        Dictionary containing:
        - 'q_values': Q values of detected peaks
        - 'intensities': Intensity values of detected peaks
        - 'widths': Width values of detected peaks in Q units
        - 'indices': Original indices of peaks in the data array
    """
    # Initialize lists to store results
    all_q_peaks = []
    all_intensities = []
    all_widths = []
    all_indices = []
    
    # Process each section with its own parameters
    for section in sections:
        q_min, q_max = section['q_range']
        
        # Find indices corresponding to the Q range
        section_mask = (data[:, 0] >= q_min) & (data[:, 0] <= q_max)
        section_indices = np.where(section_mask)[0]
        
        if len(section_indices) == 0:
            continue
        
        # Extract the section data
        section_data = data[section_mask]
        
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
        q_peaks = data[global_indices, 0]
        intensities = data[global_indices, 1]
        
        # Calculate widths in Q units rather than sample units
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
            widths_q = np.zeros(len(q_peaks))
        
        # Add results to master lists
        all_q_peaks.extend(q_peaks)
        all_intensities.extend(intensities)
        all_widths.extend(widths_q)
        all_indices.extend(global_indices)
    
    # Convert lists to arrays and sort by Q value
    if all_q_peaks:
        sort_idx = np.argsort(all_q_peaks)
        return {
            'q_values': np.array(all_q_peaks)[sort_idx],
            'intensities': np.array(all_intensities)[sort_idx],
            'widths': np.array(all_widths)[sort_idx],
            'indices': np.array(all_indices)[sort_idx]
        }
    else:
        return {
            'q_values': np.array([]),
            'intensities': np.array([]),
            'widths': np.array([]),
            'indices': np.array([])
        }

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

def plot_peaks_over_data(data, peak_results, sections=None, figsize=(12, 8), plot_widths=True):
    """
    Plot detected peaks over the original intensity vs. Q data with different sections highlighted.
    Uses log scale for the y-axis, with section shading covering the entire y-axis and
    section headings at the top of the figure.
    
    Parameters:
    -----------
    data : ndarray of shape (n, 2)
        Array with [Q, Intensity] values
    peak_results : dict
        Dictionary containing peak information from find_peaks_by_sections
    sections : list of dicts, optional
        Section definitions used for peak finding, to highlight different regions
    figsize : tuple, optional
        Figure size (width, height) in inches
    plot_widths : bool, optional
        Whether to indicate peak widths on the plot
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
        The created figure and axes for further customization if needed
    """
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
        trans = ax.transAxes.inverted()  # Transform from data to axes coordinates
        inv_trans = trans.inverted()     # Transform from axes to data coordinates
        
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
    if len(peak_results['q_values']) > 0:
        ax.plot(peak_results['q_values'], peak_results['intensities'], 'ro', 
                markersize=8, label='Detected Peaks')
        
        # Annotate each peak with its Q value
        for i, (q, intensity) in enumerate(zip(peak_results['q_values'], peak_results['intensities'])):
            # In log scale, position the text above the peak
            log_intensity = np.log10(intensity)
            text_y = 10**(log_intensity + 0.1)  # Move further up in log space
            
            ax.annotate(f'{q:.2f}', (q, text_y), xytext=(0, 5), 
                        textcoords='offset points', fontsize=9, ha='center')
        
        # Visualize peak widths if requested
        if plot_widths and len(peak_results['widths']) > 0:
            for i, (q, intensity, width) in enumerate(zip(peak_results['q_values'], 
                                                         peak_results['intensities'], 
                                                         peak_results['widths'])):
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

def pitchcalc_from_results(peak_results):
    """
    Calculate the pitch from peak detection results using 2π/mean(spacing).
    
    Parameters:
    -----------
    peak_results : dict
        Dictionary containing peak information from find_peaks_by_sections
        
    Returns:
    --------
    pitch : float
        Calculated pitch value
    spacing : ndarray
        Array of spacings between adjacent peaks
    """
    if len(peak_results['q_values']) < 2:
        return None, np.array([])
    
    # Sort q values (should already be sorted, but to be safe)
    q_values = np.sort(peak_results['q_values'])
    
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