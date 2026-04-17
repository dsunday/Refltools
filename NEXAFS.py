
import kkcalc #this is the only library that you might not already have.
from kkcalc import data
from kkcalc import kk

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rc, gridspec
import os
import tempfile
import json
import pickle
import re
from typing import Dict, Tuple, Optional, Union, List
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path

from scipy.optimize import curve_fit, least_squares, minimize, differential_evolution, dual_annealing

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
    
    # Prepare input data for kkcalc. The loader in kkcalc expects
    # whitespace-separated columns, so handle common CSV inputs by
    # converting them to a temporary space-delimited file.
    prepared_input = input_file
    temp_file = None
    try:
        if isinstance(input_file, (str, os.PathLike)):
            path = Path(input_file)

            # Detect comma-separated content (including CSV files)
            needs_conversion = path.suffix.lower() == '.csv'
            if not needs_conversion:
                try:
                    with open(path, 'r') as fh:
                        for line in fh:
                            stripped = line.strip()
                            if not stripped or stripped.startswith('#'):
                                continue
                            needs_conversion = ',' in stripped
                            break
                except FileNotFoundError:
                    pass

            if needs_conversion:
                df = pd.read_csv(path)
                if df.shape[1] < 2:
                    raise ValueError(
                        f"Expected at least two columns in {path}, found {df.shape[1]}"
                    )
                array = df.iloc[:, :2].to_numpy(dtype=float)
                array = array[np.all(np.isfinite(array), axis=1)]

                temp = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
                np.savetxt(temp.name, array, fmt='%.12e')
                temp_file = temp.name
                temp.close()
                prepared_input = temp_file
            else:
                prepared_input = str(path)
        else:
            # Assume array-like input
            array = np.asarray(input_file, dtype=float)
            if array.ndim != 2 or array.shape[1] < 2:
                raise ValueError(
                    "input_file must be a path or an array with at least two columns"
                )
            array = array[:, :2]
            array = array[np.all(np.isfinite(array), axis=1)]
            temp = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
            np.savetxt(temp.name, array, fmt='%.12e')
            temp_file = temp.name
            temp.close()
            prepared_input = temp_file

        # Calculate the real part using Kramers-Kronig transform
        output = kk.kk_calculate_real(
            prepared_input,
            chemical_formula,
            load_options=None,
            input_data_type='Beta',
            merge_points=merge_points,
            add_background=False,
            fix_distortions=False,
            curve_tolerance=0.05,
            curve_recursion=100
        )
    finally:
        if temp_file is not None:
            try:
                os.remove(temp_file)
            except OSError:
                pass
    
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

def process_nexafs_to_SLD_from_array(energy, intensity, chemical_formula, density,
                                      x_min=None, x_max=None):
    """
    Process a NEXAFS spectrum supplied as in-memory arrays and return the
    refractive index components and SLD.

    This is a convenience wrapper around process_nexafs_to_SLD for use when
    the spectrum is already loaded into numpy arrays rather than sitting in a
    file — for example, after calling load_spectrum_data() on a CSV file.

    The arrays are written to a temporary file internally so that kkcalc can
    process them; the file is deleted automatically on return.

    Parameters
    ----------
    energy : array-like
        Energy values in eV.
    intensity : array-like
        Absorption intensity values corresponding to each energy point.
        These are treated as Beta (the imaginary part of the refractive index
        decrement) by kkcalc, consistent with process_nexafs_to_SLD.
    chemical_formula : str
        Chemical formula of the material, e.g. 'C8H8' for polystyrene.
        Used by kkcalc to stitch Henke background scattering factors.
    density : float
        Mass density of the material in g/cm³.
    x_min : float, optional
        Lower merge point in eV — the energy at which the measured spectrum
        begins to replace the Henke background tabulation.  Should be just
        below the first feature of interest in the spectrum.
    x_max : float, optional
        Upper merge point in eV — the energy at which the measured spectrum
        hands back to the Henke background.  Should be just above the last
        feature of interest.

    Returns
    -------
    DeltaBeta : numpy.ndarray, shape (n, 3)
        Columns: Energy (eV), Delta (dimensionless), Beta (dimensionless).
        Spans the full kkcalc output energy grid, which extends well beyond
        the input energy range due to Henke background stitching.
    SLD : numpy.ndarray, shape (n, 4)
        Columns: Energy (eV), SLD_real, SLD_imag, Wavelength (Å).
        SLD values are in units of 10^-6 Å^-2.

    Examples
    --------
    Load a CSV spectrum with load_spectrum_data then transform to SLD:

    >>> from NEXAFS import load_spectrum_data, process_nexafs_to_SLD_from_array
    >>> energy, intensity = load_spectrum_data('my_carbon_spectrum.csv')
    >>> DeltaBeta, SLD = process_nexafs_to_SLD_from_array(
    ...     energy, intensity,
    ...     chemical_formula='C8H8',
    ...     density=1.05,
    ...     x_min=270.0,
    ...     x_max=330.0
    ... )

    Pass a numpy array directly:

    >>> import numpy as np
    >>> data = np.loadtxt('spectrum.txt')
    >>> DeltaBeta, SLD = process_nexafs_to_SLD_from_array(
    ...     data[:, 0], data[:, 1],
    ...     chemical_formula='C8H8',
    ...     density=1.05
    ... )
    """
    import tempfile

    energy    = np.asarray(energy,    dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if energy.shape != intensity.shape or energy.ndim != 1:
        raise ValueError(
            "energy and intensity must be 1-D arrays of the same length. "
            f"Got shapes {energy.shape} and {intensity.shape}."
        )

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, prefix='nexafs_kk_'
    )
    try:
        for e, i in zip(energy, intensity):
            tmp.write(f"{e:.6f}  {i:.10e}\n")
        tmp.close()
        return process_nexafs_to_SLD(tmp.name, chemical_formula, density, x_min, x_max)
    finally:
        os.unlink(tmp.name)

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

import numpy as np
from scipy import interpolate, special
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt

def erf(x):
    """
    Error function wrapper for numpy's error function.
    """
    return special.erf(x)

def edge_bl_func(x, location, height, width, decay):
    """
    Python equivalent of the Igor Pro Edge_BLFunc function with explicit parameters.
    
    Parameters:
    x : float or numpy array
        The x values at which to evaluate the function
    location : float
        The location parameter (equivalent to s.cwave[0] in Igor)
    height : float
        The height parameter (equivalent to s.cwave[1] in Igor)
    width : float
        The width parameter (equivalent to s.cwave[2] in Igor)
    decay : float
        The decay parameter (equivalent to s.cwave[3] in Igor)
    
    Returns:
    float or numpy array
        The function value(s) at x
    """
    # Determine step (1 if x >= location + width, 0 otherwise)
    step = np.where(x >= location + width, 1, 0)
    
    # Calculate the error function component
    erf_component = (1 + erf((x - location) * 2 * math.log(2) / width))
    
    # Calculate the decay component
    decay_component = (1 + step * (np.exp(-decay * (x - location - width)) - 1))
    
    # Calculate the final result
    result = (height / 2) * erf_component * decay_component
    
    return result

def gaussian(x, height, center, width):
    """
    Calculate a Gaussian peak.
    
    Parameters:
    x : numpy array
        Energy values
    height : float
        Peak height
    center : float
        Peak center energy
    width : float
        Peak width (FWHM)
    
    Returns:
    numpy array
        Gaussian peak values
    """
    sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)



def load_spectrum_data(data: Union[str, np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Load spectrum data from various formats.
    
    Parameters:
    data : str, numpy array, or pandas DataFrame
        - str: filepath to load data from
        - numpy array: 2D array with columns [energy, intensity]
        - pandas DataFrame: DataFrame with energy and intensity columns
    
    Returns:
    numpy.ndarray
        Array with shape (n_points, 2) containing [energy, intensity]
    """
    if isinstance(data, str):
        # Try to load from file
        try:
            # Try pandas first (handles various formats)
            df = pd.read_csv(data, sep=None, engine='python')
            if df.shape[1] >= 2:
                energy = df.iloc[:, 0].values
                intensity = df.iloc[:, 1].values
            else:
                raise ValueError("Data file must have at least 2 columns")
        except:
            # Fallback to numpy
            arr = np.loadtxt(data)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("Data must have shape (n_points, 2) with energy and intensity")
            energy = arr[:, 0]
            intensity = arr[:, 1]
    
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] >= 2:
            energy = data.iloc[:, 0].values
            intensity = data.iloc[:, 1].values
        else:
            raise ValueError("DataFrame must have at least 2 columns")
    
    elif isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Data must have shape (n_points, 2) with energy and intensity")
        energy = data[:, 0]
        intensity = data[:, 1]
    
    else:
        raise ValueError("Data must be filepath, numpy array, or pandas DataFrame")
    
    return np.column_stack((energy, intensity))

def plot_spectra(spectra: Union[np.ndarray, List[np.ndarray]],
                 labels: Optional[List[str]] = None,
                 energy_min: Optional[float] = None,
                 energy_max: Optional[float] = None,
                 stacked: bool = False,
                 figsize: Optional[Tuple[float, float]] = None,
                 title: Optional[str] = None,
                 xlabel: str = 'Energy (eV)',
                 ylabel: str = 'Intensity (a.u.)',
                 colors: Optional[List[str]] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 save_path: Optional[str] = None,
                 dpi: int = 300,
                 show: bool = True) -> plt.Figure:
    """
    Plot one or more NEXAFS spectra with customizable energy range and layout.
    
    Parameters:
    -----------
    spectra : numpy.ndarray or list of numpy.ndarray
        Single spectrum array with shape (n_points, 2) or list of such arrays.
        Each array should have columns [energy, intensity]
    labels : list of str, optional
        Labels for each spectrum. If None, auto-generates "Spectrum 1", "Spectrum 2", etc.
    energy_min : float, optional
        Minimum energy to display. If None, uses minimum across all spectra.
    energy_max : float, optional
        Maximum energy to display. If None, uses maximum across all spectra.
    stacked : bool, default False
        If True, creates vertical subplots for each spectrum.
        If False, overlays all spectra on a single plot.
    figsize : tuple, optional
        Figure size (width, height). If None, uses (12, 8) for overlaid or (12, 4*n) for stacked.
    title : str, optional
        Plot title. If None, no title is shown.
    xlabel : str, default 'Energy (eV)'
        X-axis label
    ylabel : str, default 'Intensity (a.u.)'
        Y-axis label
    colors : list of str, optional
        List of colors for each spectrum. If None, auto-generates colors.
        Can be matplotlib color names (e.g., 'red', 'blue') or hex codes (e.g., '#FF0000').
        If provided, length must match number of spectra.
    xlim : tuple, optional
        X-axis limits as (xmin, xmax). If None, uses energy_min and energy_max.
    ylim : tuple, optional
        Y-axis limits as (ymin, ymax). If None, auto-scales based on data.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default 300
        Resolution for saved figure
    show : bool, default True
        Whether to display the figure. Set to False to prevent duplicate displays in Jupyter notebooks.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
        
    Examples:
    ---------
    >>> # Single spectrum
    >>> plot_spectra(Peptoid1, energy_min=280, energy_max=320)
    
    >>> # Multiple spectra overlaid
    >>> plot_spectra([Peptoid1, Peptoid2, Peptoid3], labels=['P1', 'P2', 'P3'])
    
    >>> # Multiple spectra stacked
    >>> plot_spectra([Peptoid1, Peptoid2], stacked=True)
    
    >>> # Specify colors and axis limits
    >>> plot_spectra([Peptoid1, Peptoid2], colors=['red', 'blue'], xlim=(280, 300), ylim=(0, 1))
    """
    
    # Normalize input to list of arrays
    if isinstance(spectra, np.ndarray):
        spectra_list = [spectra]
    else:
        spectra_list = spectra
    
    n_spectra = len(spectra_list)
    
    # Generate default labels if not provided
    if labels is None:
        labels = [f'Spectrum {i+1}' for i in range(n_spectra)]
    elif len(labels) != n_spectra:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of spectra ({n_spectra})")
    
    # Determine energy range
    all_energies = np.concatenate([spec[:, 0] for spec in spectra_list])
    if energy_min is None:
        energy_min = np.min(all_energies)
    if energy_max is None:
        energy_max = np.max(all_energies)
    
    # Filter data to energy range
    filtered_spectra = []
    for spec in spectra_list:
        energy = spec[:, 0]
        intensity = spec[:, 1]
        mask = (energy >= energy_min) & (energy <= energy_max)
        filtered_spectra.append(spec[mask])
    
    # Determine figure size
    if figsize is None:
        if stacked:
            figsize = (12, 4 * n_spectra)
        else:
            figsize = (12, 8)
    
    # Create figure and axes
    if stacked:
        fig, axes = plt.subplots(n_spectra, 1, figsize=figsize, sharex=True)
        if n_spectra == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax] * n_spectra
    
    # Generate colors for overlaid plot if not provided
    if colors is None:
        if not stacked and n_spectra > 1:
            if n_spectra <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, max(n_spectra, 3)))
            else:
                colors = plt.cm.tab20(np.linspace(0, 1, n_spectra))
        else:
            colors = ['C0'] * n_spectra
    elif len(colors) != n_spectra:
        raise ValueError(f"Number of colors ({len(colors)}) must match number of spectra ({n_spectra})")
    
    # Plot each spectrum
    for i, (spec, label, color) in enumerate(zip(filtered_spectra, labels, colors)):
        ax = axes[i]
        energy = spec[:, 0]
        intensity = spec[:, 1]
        ax.plot(energy, intensity, 'o-', color=color, markersize=3, 
                linewidth=1.5, label=label, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=18)
        if stacked:
            ax.legend(loc='best')
    
    # Set common properties
    axes[-1].set_xlabel(xlabel, fontsize=18)
    if xlim is not None:
        axes[-1].set_xlim(xlim)
    else:
        axes[-1].set_xlim(energy_min, energy_max)
    
    # Apply ylim if provided
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)
    
    # Add legend for overlaid plot
    if not stacked and n_spectra > 1:
        axes[0].legend(loc='best')
    
    # Add title
    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.995)
    
    fig.tight_layout()
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show figure if requested
    if show:
        plt.show()
        plt.close(fig)
    
    #return fig



def fit_spectra_with_edge(target_spectrum, spectrum1, spectrum2, 
                         initial_guess=None, include_edge=True,
                         bounds=None):
    """
    Fits a linear combination of two spectra plus an optional step edge 
    to match a target spectrum.
    
    Parameters:
    -----------
    target_spectrum : ndarray, shape (n, 2)
        Target spectrum with columns [Energy, Intensity]
    spectrum1 : ndarray, shape (m, 2)
        First input spectrum with columns [Energy, Intensity]
    spectrum2 : ndarray, shape (k, 2)
        Second input spectrum with columns [Energy, Intensity]
    initial_guess : list or None, optional
        Initial guess for parameters:
        [fraction, edge_location, edge_height, edge_width, edge_decay]
        If None, defaults to [0.5, (energy_range_mid), 0.1, 1.0, 0.5]
    include_edge : bool, optional
        Whether to include the step edge in the fit
    bounds : list of tuples or None, optional
        Bounds for parameters as [(min_frac, max_frac), 
        (min_loc, max_loc), (min_height, max_height), 
        (min_width, max_width), (min_decay, max_decay)]
        If None, uses default bounds
        
    Returns:
    --------
    params : list
        The optimal parameters [fraction, edge_location, edge_height, edge_width, edge_decay]
        (if include_edge=False, the edge parameters will be zeros)
    combined_spectrum : ndarray, shape (n, 2)
        The optimal combination of spectrum1, spectrum2, and edge
    components : dict
        Dictionary with separate components: 
        {'spec1_contrib', 'spec2_contrib', 'edge_contrib'}
    """
    # Get common energy grid (use target's energy grid)
    target_energies = target_spectrum[:, 0]
    
    # Set default initial guess if not provided
    if initial_guess is None:
        energy_mid = (target_energies.min() + target_energies.max()) / 2
        initial_guess = [0.5, energy_mid, 0.1, 1.0, 0.5]
    
    # Create interpolation functions for both input spectra
    interp1 = interpolate.interp1d(spectrum1[:, 0], spectrum1[:, 1], 
                                  bounds_error=False, fill_value=0)
    interp2 = interpolate.interp1d(spectrum2[:, 0], spectrum2[:, 1], 
                                  bounds_error=False, fill_value=0)
    
    # Interpolate input spectra onto target energy grid
    intensity1 = interp1(target_energies)
    intensity2 = interp2(target_energies)
    
    # Define objective function to minimize (sum of squared differences)
    if include_edge:
        def objective(params):
            fraction, edge_loc, edge_height, edge_width, edge_decay = params
            # Calculate linear combination of spectra
            spec_combined = fraction * intensity1 + (1 - fraction) * intensity2
            # Add edge function
            edge_contribution = edge_bl_func(target_energies, edge_loc, edge_height, edge_width, edge_decay)
            combined_intensity = spec_combined + edge_contribution
            # Return sum of squared differences
            return np.sum((combined_intensity - target_spectrum[:, 1])**2)
        
        # Set default bounds if not provided
        if bounds is None:
            energy_min, energy_max = target_energies.min(), target_energies.max()
            bounds = [
                (0, 1),                     # fraction between 0 and 1
                (energy_min, energy_max),   # edge location within energy range
                (0, 1),                     # edge height between 0 and 1
                (0.1, 10),                  # edge width (reasonable range)
                (0, 10)                     # edge decay (reasonable range)
            ]
    else:
        # Simplified objective function without edge
        def objective(params):
            fraction = params[0]
            combined_intensity = fraction * intensity1 + (1 - fraction) * intensity2
            return np.sum((combined_intensity - target_spectrum[:, 1])**2)
        
        # Use only the first parameter if no edge
        initial_guess = [initial_guess[0]]
        if bounds is None:
            bounds = [(0, 1)]  # Just the fraction bound
    
    # Perform optimization
    result = minimize(objective, initial_guess, bounds=bounds)
    optimal_params = result.x
    
    # Ensure we return a consistent parameter list format
    if not include_edge:
        optimal_params = np.append(optimal_params, [0, 0, 0, 0])
    
    # Calculate final combined spectrum and components
    fraction = optimal_params[0]
    spec1_contrib = fraction * intensity1
    spec2_contrib = (1 - fraction) * intensity2
    
    if include_edge:
        edge_loc, edge_height, edge_width, edge_decay = optimal_params[1:5]
        edge_contrib = edge_bl_func(target_energies, edge_loc, edge_height, edge_width, edge_decay)
    else:
        edge_contrib = np.zeros_like(target_energies)
    
    combined_intensity = spec1_contrib + spec2_contrib + edge_contrib
    combined_spectrum = np.column_stack((target_energies, combined_intensity))
    
    # Create components dictionary
    components = {
        'spec1_contrib': np.column_stack((target_energies, spec1_contrib)),
        'spec2_contrib': np.column_stack((target_energies, spec2_contrib)),
        'edge_contrib': np.column_stack((target_energies, edge_contrib))
    }
    
    return optimal_params, combined_spectrum, components



def plot_spectral_fit_with_edge(target_spectrum, spectrum1, spectrum2, 
                              optimal_params, combined_spectrum, components,
                              labels=None, title=None, figsize=(12, 8)):
    """
    Creates a comprehensive plot showing the fit results with the edge function component.
    
    Parameters:
    -----------
    target_spectrum : ndarray, shape (n, 2)
        Target spectrum with columns [Energy, Intensity]
    spectrum1 : ndarray, shape (m, 2)
        First input spectrum with columns [Energy, Intensity]
    spectrum2 : ndarray, shape (k, 2)
        Second input spectrum with columns [Energy, Intensity]
    optimal_params : list
        The optimal parameters [fraction, edge_location, edge_height, edge_width, edge_decay]
    combined_spectrum : ndarray, shape (n, 2)
        The optimal combination of spectrum1, spectrum2, and edge
    components : dict
        Dictionary with separate components: 
        {'spec1_contrib', 'spec2_contrib', 'edge_contrib'}
    labels : list, optional
        List of labels for the spectra in the order [target, spectrum1, spectrum2, edge]
        Default: ['Target', 'Spectrum 1', 'Spectrum 2', 'Edge']
    title : str, optional
        Title for the figure
    figsize : tuple, optional
        Figure size as (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axes : list of matplotlib.axes.Axes
        The axes objects
    """
    if labels is None:
        labels = ['Target', 'Spectrum 1', 'Spectrum 2', 'Edge']
    
    # Extract parameters
    fraction = optimal_params[0]
    has_edge = np.any(components['edge_contrib'][:, 1] != 0)
    
    # Calculate the fit quality
    residuals = target_spectrum[:, 1] - combined_spectrum[:, 1]
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Set up the figure
    if has_edge:
        fig, axes = plt.subplots(2, 2, figsize=figsize, 
                                 gridspec_kw={'height_ratios': [3, 1.5]})
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = [axes[0], axes[1], None, None]
    
    # Left top plot: Reference spectra
    axes[0].plot(spectrum1[:, 0], spectrum1[:, 1], 'b-', label=labels[1])
    axes[0].plot(spectrum2[:, 0], spectrum2[:, 1], 'g-', label=labels[2])
    axes[0].set_xlabel('Energy')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Reference Spectra')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right top plot: Target vs. Fitted
    axes[1].plot(target_spectrum[:, 0], target_spectrum[:, 1], 'k-', label=labels[0])
    axes[1].plot(combined_spectrum[:, 0], combined_spectrum[:, 1], 'r--', 
                label=f'Fit (RMSE: {rmse:.5f})')
    
    # Add fit parameters to the legend
    edge_text = ''
    if has_edge:
        edge_loc, edge_height, edge_width, edge_decay = optimal_params[1:5]
        edge_text = f'\nEdge: loc={edge_loc:.2f}, h={edge_height:.2f}, w={edge_width:.2f}, d={edge_decay:.2f}'
    
    axes[1].text(0.02, 0.02, 
                f'Fraction: {fraction:.3f} × {labels[1]} + {1-fraction:.3f} × {labels[2]}{edge_text}',
                transform=axes[1].transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))
    
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('Target vs. Fitted Spectrum')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if has_edge:
        # Left bottom plot: Component contributions
        axes[2].plot(components['spec1_contrib'][:, 0], components['spec1_contrib'][:, 1], 'b-', 
                    label=f'{fraction:.3f} × {labels[1]}')
        axes[2].plot(components['spec2_contrib'][:, 0], components['spec2_contrib'][:, 1], 'g-', 
                    label=f'{1-fraction:.3f} × {labels[2]}')
        axes[2].plot(components['edge_contrib'][:, 0], components['edge_contrib'][:, 1], 'm-', 
                    label=f'{labels[3]}')
        axes[2].set_xlabel('Energy')
        axes[2].set_ylabel('Intensity')
        axes[2].set_title('Component Contributions')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Right bottom plot: Residuals
        axes[3].plot(target_spectrum[:, 0], residuals, 'k-')
        axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[3].set_xlabel('Energy')
        axes[3].set_ylabel('Residuals')
        axes[3].set_title('Fit Residuals')
        axes[3].grid(True, alpha=0.3)
    
    # Adjust layout and add overall title if provided
    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=0.9)
    
    return fig, axes

def plot_spectra_with_fixed_params(target_spectrum, spectrum1, spectrum2, 
                                  params, include_edge=True,
                                  labels=None, title=None, figsize=(10, 6)):
    """
    Plots the target spectrum against a linear combination with user-specified parameters.
    
    Parameters:
    -----------
    target_spectrum : ndarray, shape (n, 2)
        Target spectrum with columns [Energy, Intensity]
    spectrum1 : ndarray, shape (m, 2)
        First input spectrum with columns [Energy, Intensity]
    spectrum2 : ndarray, shape (k, 2)
        Second input spectrum with columns [Energy, Intensity]
    params : list
        The parameters [fraction, edge_location, edge_height, edge_width, edge_decay]
    include_edge : bool, optional
        Whether to include the step edge in the plot
    labels : list, optional
        List of labels for the spectra in the order [target, spectrum1, spectrum2, edge]
        Default: ['Target', 'Spectrum 1', 'Spectrum 2', 'Edge']
    title : str, optional
        Title for the figure
    figsize : tuple, optional
        Figure size as (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The axes object
    rmse : float
        Root mean square error between target and calculated intensities
    """
    if labels is None:
        labels = ['Target', 'Spectrum 1', 'Spectrum 2', 'Edge']
        
    # Get target energy grid
    target_energies = target_spectrum[:, 0]
    
    # Create interpolation functions for both input spectra
    interp1 = interpolate.interp1d(spectrum1[:, 0], spectrum1[:, 1], 
                                   bounds_error=False, fill_value=0)
    interp2 = interpolate.interp1d(spectrum2[:, 0], spectrum2[:, 1], 
                                   bounds_error=False, fill_value=0)
    
    # Interpolate input spectra onto target energy grid
    intensity1 = interp1(target_energies)
    intensity2 = interp2(target_energies)
    
    # Extract the parameters
    fraction = params[0]
    spec1_contrib = fraction * intensity1
    spec2_contrib = (1 - fraction) * intensity2
    
    # Calculate edge contribution if included
    if include_edge and len(params) >= 5:
        edge_loc, edge_height, edge_width, edge_decay = params[1:5]
        edge_contrib = edge_bl_func(target_energies, edge_loc, edge_height, edge_width, edge_decay)
    else:
        edge_contrib = np.zeros_like(target_energies)
        
    # Calculate combined intensity
    combined_intensity = spec1_contrib + spec2_contrib + edge_contrib
    
    # Calculate residuals and RMSE
    residuals = target_spectrum[:, 1] - combined_intensity
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the spectra
    ax.plot(target_spectrum[:, 0], target_spectrum[:, 1], 'k-', label=labels[0])
    ax.plot(target_energies, combined_intensity, 'r--', 
            label=f'Calculated (RMSE: {rmse:.5f})')
    
    # Add component spectra
    ax.plot(target_energies, spec1_contrib, 'b-', alpha=0.4, 
            label=f'{fraction:.3f} × {labels[1]}')
    ax.plot(target_energies, spec2_contrib, 'g-', alpha=0.4, 
            label=f'{1-fraction:.3f} × {labels[2]}')
    
    if include_edge and np.any(edge_contrib != 0):
        ax.plot(target_energies, edge_contrib, 'm-', alpha=0.4, 
                label=f'{labels[3]} (loc={params[1]:.2f}, h={params[2]:.2f})')
    
    # Add parameter text box
    param_text = f'Fraction: {fraction:.3f}'
    if include_edge and len(params) >= 5:
        param_text += f'\nEdge: loc={params[1]:.2f}, h={params[2]:.2f}, w={params[3]:.2f}, d={params[4]:.2f}'
    
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Energy')
    ax.set_ylabel('Intensity')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Spectra Comparison with Fixed Parameters')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax, rmse

import tempfile
import os
 
 
def imag_SLD_to_beta(SLD_array):
    """
    Convert imaginary SLD values from a reflectometry fit to the optical constant beta (β).
 
    This is the inverse of the SLD_imag calculation in DeltaBetatoSLD:
        SLD_imag [10^-6 Å^-2] = 2π * β / λ²  * 10^6
    so:
        β = SLD_imag * λ² / (2π * 10^6)
 
    Parameters
    ----------
    SLD_array : numpy.ndarray
        Array with columns [Energy (eV), SLD_real, SLD_imag, Wavelength (Å)].
        SLD values are in units of 10^-6 Å^-2.
        This is the format returned by DeltaBetatoSLD / process_nexafs_to_SLD,
        or from reflectometry fit results stored in the same convention.
 
    Returns
    -------
    numpy.ndarray
        Array with columns [Energy (eV), Beta (dimensionless)].
    """
    energy     = SLD_array[:, 0]         # eV
    SLD_imag   = SLD_array[:, 2]         # 10^-6 Å^-2
    wavelength = SLD_array[:, 3]         # Å
 
    # Invert SLD_imag = 2π * β / λ² * 1e6
    beta = SLD_imag * wavelength**2 / (2.0 * np.pi * 1e6)
 
    return np.column_stack((energy, beta))
 
 
def beta_to_delta_kk(energy_beta, chemical_formula, density,
                     merge_points=None,
                     add_background=False,
                     fix_distortions=False,
                     curve_tolerance=0.05,
                     curve_recursion=100):
    """
    Apply the Kramers-Kronig transform to a beta (β) spectrum to obtain delta (δ),
    using kkcalc.  The input is an in-memory numpy array; it is written to a
    temporary file internally so that kkcalc's file-based API can be used.
 
    Parameters
    ----------
    energy_beta : numpy.ndarray
        Array with columns [Energy (eV), Beta (dimensionless)], as returned by
        imag_SLD_to_beta().
    chemical_formula : str
        Chemical formula of the material, e.g. 'C8H8' for polystyrene.
        Used by kkcalc to stitch background scattering factors.
    density : float
        Mass density of the material in g/cm³.
    merge_points : list of two floats, optional
        [E_min, E_max] energy range (eV) within which the measured beta data
        replaces the tabulated background.  Typically set to bracket the
        absorption edge region.  If None, kkcalc chooses merge points
        automatically.
    add_background : bool, optional
        Whether kkcalc should add back a removed background (default False).
    fix_distortions : bool, optional
        Whether to apply kkcalc's distortion correction (default False).
        Requires scipy.
    curve_tolerance : float, optional
        Tolerance for the KK transform recursion (default 0.05).
    curve_recursion : int, optional
        Maximum recursion depth for the KK transform (default 100).
 
    Returns
    -------
    numpy.ndarray
        Array with columns [Energy (eV), Delta (dimensionless), Beta (dimensionless)].
        The energy axis matches the output grid produced by kkcalc (which may differ
        from the input grid due to stitching with tabulated data).
    """
    from kkcalc import data as kkdata
    from kkcalc import kk
 
    # Parse formula and compute formula mass for unit conversion
    stoichiometry = kkdata.ParseChemicalFormula(chemical_formula)
    formula_mass  = kkdata.calculate_FormulaMass(stoichiometry)
 
    # Write beta spectrum to a temporary two-column text file
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, prefix='kkcalc_beta_'
    )
    try:
        for row in energy_beta:
            tmp.write(f"{row[0]:.6f}  {row[1]:.10e}\n")
        tmp.close()
 
        # Run the Kramers-Kronig transform
        # Output columns from kk_calculate_real are [Energy, f1_ASF, f2_ASF]
        output = kk.kk_calculate_real(
            tmp.name,
            chemical_formula,
            load_options=None,
            input_data_type='Beta',
            merge_points=merge_points,
            add_background=add_background,
            fix_distortions=fix_distortions,
            curve_tolerance=curve_tolerance,
            curve_recursion=curve_recursion
        )
    finally:
        os.unlink(tmp.name)
 
    # Convert ASF columns to refractive index (delta, beta)
    delta_col = kkdata.convert_data(
        output[:, [0, 1]],
        'ASF',
        'refractive_index',
        Density=density,
        Formula_Mass=formula_mass
    )
    beta_col = kkdata.convert_data(
        output[:, [0, 2]],
        'ASF',
        'refractive_index',
        Density=density,
        Formula_Mass=formula_mass
    )
 
    # delta_col[:,0] and beta_col[:,0] are the same energy axis
    energy = delta_col[:, 0]
    delta  = delta_col[:, 1]
    beta   = beta_col[:, 1]
 
    return np.column_stack((energy, delta, beta))
 
 
def delta_to_real_SLD(energy_delta):
    """
    Convert delta (δ) values to real SLD.
 
    Uses the same formula as DeltaBetatoSLD:
        SLD_real [10^-6 Å^-2] = 2π * δ / λ²  * 10^6
 
    Parameters
    ----------
    energy_delta : numpy.ndarray
        Array with columns [Energy (eV), Delta (dimensionless)].
 
    Returns
    -------
    numpy.ndarray
        Array with columns [Energy (eV), SLD_real (10^-6 Å^-2), Wavelength (Å)].
    """
    energy     = energy_delta[:, 0]
    delta      = energy_delta[:, 1]
    wavelength = EnergytoWavelength(energy)
 
    SLD_real = 2.0 * np.pi * delta / (wavelength**2) * 1e6
 
    return np.column_stack((energy, SLD_real, wavelength))
 
 
def imag_SLD_to_real_SLD(SLD_array, chemical_formula, density,
                          merge_points=None,
                          add_background=False,
                          fix_distortions=False,
                          curve_tolerance=0.05,
                          curve_recursion=100):
    """
    Full pipeline: convert imaginary SLD from a reflectometry fit into the
    corresponding real SLD via the Kramers-Kronig transform.
 
    Pipeline:
        imaginary SLD  →  β  →  KK transform  →  δ  →  real SLD
 
    Parameters
    ----------
    SLD_array : numpy.ndarray
        Array with columns [Energy (eV), SLD_real, SLD_imag, Wavelength (Å)],
        as produced by DeltaBetatoSLD() or stored from a reflectometry fit.
        SLD values are in units of 10^-6 Å^-2.
        Only the imaginary SLD column is used as input to the KK transform.
    chemical_formula : str
        Chemical formula of the material, e.g. 'C8H8' for polystyrene.
    density : float
        Mass density of the material in g/cm³.
    merge_points : list of two floats, optional
        [E_min, E_max] energy range (eV) over which the measured data is
        merged into the background scattering factors.  Should bracket the
        absorption edge of interest.  If None, kkcalc selects merge points
        automatically.
    add_background : bool, optional
        Whether kkcalc should reconstruct a removed background (default False).
    fix_distortions : bool, optional
        Apply kkcalc's linear distortion correction (default False).
    curve_tolerance : float, optional
        KK transform recursion tolerance (default 0.05).
    curve_recursion : int, optional
        KK transform maximum recursion depth (default 100).
 
    Returns
    -------
    tuple
        (DeltaBeta, SLD_kk) where:
 
        DeltaBeta : numpy.ndarray, shape (n, 3)
            Columns [Energy (eV), Delta, Beta] on the kkcalc output energy grid.
 
        SLD_kk : numpy.ndarray, shape (n, 4)
            Columns [Energy (eV), SLD_real_kk, SLD_imag_kk, Wavelength (Å)]
            with SLD values in 10^-6 Å^-2.
            SLD_real_kk is the KK-derived real SLD from the fitted imaginary SLD.
            SLD_imag_kk is the beta back-converted to SLD units on the kkcalc
            output energy grid (consistent with the SLD_real_kk column).
 
    Notes
    -----
    The kkcalc output energy grid typically extends well beyond the input range
    (it stitches in Henke tabulated data), so the returned arrays will usually
    span a much wider energy range than the input SLD_array.  Use merge_points
    to control where the measured data is spliced in.
 
    Example
    -------
    >>> # SLD_fit comes from a reflectometry fit at the carbon edge
    >>> DeltaBeta_kk, SLD_kk = imag_SLD_to_real_SLD(
    ...     SLD_fit,
    ...     chemical_formula='C8H8',
    ...     density=1.05,
    ...     merge_points=[270.0, 320.0]
    ... )
    >>> # Plot result over the carbon edge
    >>> mask = (SLD_kk[:, 0] >= 270) & (SLD_kk[:, 0] <= 320)
    >>> plt.plot(SLD_kk[mask, 0], SLD_kk[mask, 1], label='Real SLD (KK)')
    >>> plt.plot(SLD_kk[mask, 0], SLD_kk[mask, 2], label='Imag SLD (fit)')
    """
    # Step 1: imaginary SLD → beta
    energy_beta = imag_SLD_to_beta(SLD_array)
 
    # Step 2: beta → delta via KK transform (also returns consistent beta)
    DeltaBeta = beta_to_delta_kk(
        energy_beta,
        chemical_formula,
        density,
        merge_points=merge_points,
        add_background=add_background,
        fix_distortions=fix_distortions,
        curve_tolerance=curve_tolerance,
        curve_recursion=curve_recursion
    )
    # DeltaBeta columns: [Energy, Delta, Beta]
 
    # Step 3: delta → real SLD, beta → imag SLD (reuse DeltaBetatoSLD)
    SLD_kk = DeltaBetatoSLD(DeltaBeta)
 
    return DeltaBeta, SLD_kk

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


def plot_sld_four_panel(sld_data_dict, metadata_dict, spectrum_data_dict=None,
                        energy_range=(280, 300), marker_color_file=None, figsize=(14, 10),
                        save_path=None, full_range_only=False):
    """
    Create a four-panel plot showing SLD data from reflectivity fits and optionally NEXAFS spectrum data.
    
    Parameters:
    -----------
    sld_data_dict : dict
        Dictionary mapping scan names to SLD arrays from load_material_sld_array.
        Format: {'scan1': array1, 'scan2': array2, ...}
        Each array should have shape (n_points, 3) with columns [Energy, SLD_Real, SLD_Imag]
    metadata_dict : dict
        Dictionary mapping scan names to metadata.
        Format: {'scan1': {'material': 'Carbon', 'process': 'Bake', 'sample': 'SOC'}, ...}
    spectrum_data_dict : dict, optional
        Dictionary mapping scan names to spectrum arrays from load_spectrum_data.
        Format: {'scan1': array1, 'scan2': array2, ...}
        Each array should have shape (n_points, 2) with columns [energy, intensity]
        If None or not provided, spectrum data will not be plotted.
    energy_range : tuple, optional
        Tuple (min, max) for truncated energy range in right panels. Default: (280, 300)
    marker_color_file : str, optional
        Path to pickle file for saving marker/color assignments.
        If None, uses 'marker_color_assignments.pkl' in the same directory as NEXAFS.py
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default: (14, 10)
    save_path : str, optional
        Path to save the figure. If provided, the figure will be saved to this path.
        Supports common image formats (png, pdf, svg, etc.). If None, figure is not saved.
        Default: None
    full_range_only : bool, optional
        If True, only plot the full energy range panels (one column, two rows).
        If False, plot all four panels (two columns, two rows). Default: False
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of axes objects for the four panels
    """
    # Set default marker/color file path (use current working directory, typically the notebook directory)
    if marker_color_file is None:
        marker_color_file = os.path.join(os.getcwd(), 'marker_color_assignments.pkl')
    
    # Load existing marker/color assignments if file exists
    marker_assignments = {}  # process -> marker
    color_assignments = {}   # sample -> color
    
    # Initialize color_assignments to empty dict (will be set below if file exists)
    if os.path.exists(marker_color_file):
        try:
            with open(marker_color_file, 'rb') as f:
                saved_data = pickle.load(f)
                marker_assignments = saved_data.get('markers', {})
                color_assignments_loaded = saved_data.get('colors', {})
                
                # Migrate from old format (process, sample) tuples to new format (sample only)
                # Check if we have old format (tuple keys)
                if color_assignments_loaded:
                    # Check if first key is a tuple (old format)
                    first_key = next(iter(color_assignments_loaded.keys()), None)
                    if first_key is not None and isinstance(first_key, tuple):
                        # Old format: migrate to new format by extracting unique samples
                        color_assignments = {}
                        for key, color in color_assignments_loaded.items():
                            if isinstance(key, tuple):
                                # Old format: (process, sample) -> extract sample
                                sample = key[1]
                                # Only keep the first occurrence of each sample
                                if sample not in color_assignments:
                                    color_assignments[sample] = color
                            else:
                                # Already new format
                                color_assignments[key] = color
                    else:
                        # Already new format
                        color_assignments = color_assignments_loaded
                else:
                    # Empty dictionary
                    color_assignments = {}
        except Exception as e:
            print(f"Warning: Could not load marker/color assignments: {e}")
            color_assignments = {}
    
    # Available markers and colors
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '8', '<', '>']
    colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
    colors.extend(plt.cm.Set3(np.linspace(0, 1, 12)).tolist())
    
    # Assign markers and colors for all scans (check both SLD and spectrum data)
    marker_idx = 0
    # Start color_idx from the number of already assigned samples
    color_idx = len(color_assignments)
    
    # Collect all scan names from both dictionaries
    if spectrum_data_dict is None:
        spectrum_data_dict = {}
    all_scan_names = set(sld_data_dict.keys()) | set(spectrum_data_dict.keys())
    
    for scan_name in all_scan_names:
        if scan_name not in metadata_dict:
            continue
        
        metadata = metadata_dict[scan_name]
        process = metadata.get('process', 'Unknown')
        sample = metadata.get('sample', 'Unknown')
        
        # Assign marker shape based on process (cycle through shapes for different processes)
        if process not in marker_assignments:
            marker_assignments[process] = markers[marker_idx % len(markers)]
            marker_idx += 1
        
        # Assign color based on sample (different samples get different colors)
        if sample not in color_assignments:
            color_assignments[sample] = colors[color_idx % len(colors)]
            color_idx += 1
    
    # Save updated assignments
    try:
        with open(marker_color_file, 'wb') as f:
            pickle.dump({'markers': marker_assignments, 'colors': color_assignments}, f)
    except Exception as e:
        print(f"Warning: Could not save marker/color assignments: {e}")
    
    # Create layout based on full_range_only option
    if full_range_only:
        # One column, two rows (full range only)
        fig, axes = plt.subplots(2, 1, figsize=(figsize[0]/2, figsize[1]))
        ax_top = axes[0]      # Full range, real
        ax_bottom = axes[1]   # Full range, imag
        ax_top_left = ax_top
        ax_bottom_left = ax_bottom
        ax_top_right = None
        ax_bottom_right = None
    else:
        # Two columns, two rows (four panels)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax_top_left = axes[0, 0]      # Full range, real
        ax_bottom_left = axes[1, 0]   # Full range, imag
        ax_top_right = axes[0, 1]     # Truncated range, real
        ax_bottom_right = axes[1, 1]  # Truncated range, imag
    
    # Plot SLD data
    for scan_name, sld_array in sld_data_dict.items():
        if scan_name not in metadata_dict:
            continue
        
        metadata = metadata_dict[scan_name]
        process = metadata.get('process', 'Unknown')
        sample = metadata.get('sample', 'Unknown')
        material = metadata.get('material', 'Unknown')
        
        marker = marker_assignments[process]
        color = color_assignments[sample]
        
        # Determine marker style based on process
        # "Bake" processes use open markers, "UV" processes use closed markers
        if process == "Bake":
            marker_style = {'marker': marker, 'fillstyle': 'none', 'markeredgecolor': color, 
                          'markerfacecolor': 'none', 'markeredgewidth': 1.5, 'color': color}
        else:
            # Default to closed markers for UV and other processes
            marker_style = {'marker': marker, 'color': color}
        
        # Create label
        label = f"{material} - {process} - {sample}"
        
        # Extract energy, real, and imaginary components
        energy = sld_array[:, 0]
        sld_real = sld_array[:, 1]
        sld_imag = sld_array[:, 2]
        
        # Filter for truncated range
        mask_truncated = (energy >= energy_range[0]) & (energy <= energy_range[1])
        energy_truncated = energy[mask_truncated]
        sld_real_truncated = sld_real[mask_truncated]
        sld_imag_truncated = sld_imag[mask_truncated]
        
        # Plot real component - full range
        ax_top_left.plot(energy, sld_real, label=label, markersize=6, linewidth=1.5, 
                        alpha=0.8, **marker_style)
        
        # Plot real component - truncated range (only if right panel exists)
        if ax_top_right is not None:
            ax_top_right.plot(energy_truncated, sld_real_truncated, label=label, markersize=6, 
                             linewidth=1.5, alpha=0.8, **marker_style)
        
        # Plot imaginary component - full range
        ax_bottom_left.plot(energy, sld_imag, label=label, markersize=6, linewidth=1.5, 
                           alpha=0.8, **marker_style)
        
        # Plot imaginary component - truncated range (only if right panel exists)
        if ax_bottom_right is not None:
            ax_bottom_right.plot(energy_truncated, sld_imag_truncated, label=label, markersize=6, 
                                linewidth=1.5, alpha=0.8, **marker_style)
    
    # Plot spectrum data (imaginary only, as lines) if provided
    if spectrum_data_dict:
        for scan_name, spectrum_array in spectrum_data_dict.items():
            if scan_name not in metadata_dict:
                continue
            
            metadata = metadata_dict[scan_name]
            process = metadata.get('process', 'Unknown')
            sample = metadata.get('sample', 'Unknown')
            material = metadata.get('material', 'Unknown')
            
            color = color_assignments.get(sample, 'gray')
            
            # Create label
            label = f"{material} - {process} - {sample} (NEXAFS)"
            
            # Extract energy and intensity
            energy_spec = spectrum_array[:, 0]
            intensity = spectrum_array[:, 1]
            
            # Filter for truncated range
            mask_truncated = (energy_spec >= energy_range[0]) & (energy_spec <= energy_range[1])
            energy_spec_truncated = energy_spec[mask_truncated]
            intensity_truncated = intensity[mask_truncated]
            
            # Plot as lines only (no markers) on imaginary panels
            ax_bottom_left.plot(energy_spec, intensity, color=color, 
                               label=label, linewidth=2, alpha=0.7, linestyle='--')
            
            # Plot truncated range (only if right panel exists)
            if ax_bottom_right is not None:
                ax_bottom_right.plot(energy_spec_truncated, intensity_truncated, color=color, 
                                    label=label, linewidth=2, alpha=0.7, linestyle='--')
    
    # Formatting
    # Top - Full range, real
    ax_top_left.set_xlabel('Energy (eV)')
    ax_top_left.set_ylabel('Real SLD (10$^{-6}$ Å$^{-2}$)')
    ax_top_left.set_title('Real SLD - Full Energy Range')
    ax_top_left.grid(True, alpha=0.3)
    ax_top_left.legend(fontsize=8, loc='best')
    
    # Bottom - Full range, imag
    ax_bottom_left.set_xlabel('Energy (eV)')
    ax_bottom_left.set_ylabel('Imaginary SLD (10$^{-6}$ Å$^{-2}$)')
    ax_bottom_left.set_title('Imaginary SLD - Full Energy Range')
    ax_bottom_left.grid(True, alpha=0.3)
    ax_bottom_left.legend(fontsize=8, loc='best')
    
    # Top right - Truncated range, real (only if right panel exists)
    if ax_top_right is not None:
        ax_top_right.set_xlabel('Energy (eV)')
        ax_top_right.set_ylabel('Real SLD (10$^{-6}$ Å$^{-2}$)')
        ax_top_right.set_title(f'Real SLD - {energy_range[0]}-{energy_range[1]} eV')
        ax_top_right.grid(True, alpha=0.3)
        ax_top_right.legend(fontsize=8, loc='best')
        ax_top_right.set_xlim(energy_range)
    
    # Bottom right - Truncated range, imag (only if right panel exists)
    if ax_bottom_right is not None:
        ax_bottom_right.set_xlabel('Energy (eV)')
        ax_bottom_right.set_ylabel('Imaginary SLD (10$^{-6}$ Å$^{-2}$)')
        ax_bottom_right.set_title(f'Imaginary SLD - {energy_range[0]}-{energy_range[1]} eV')
        ax_bottom_right.grid(True, alpha=0.3)
        ax_bottom_right.legend(fontsize=8, loc='best')
        ax_bottom_right.set_xlim(energy_range)
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save figure to {save_path}: {e}")
    
    # Ensure figure is properly rendered (helps with copying/display)
    try:
        fig.canvas.draw()
    except:
        pass  # If canvas doesn't support draw, continue anyway
    
    return fig, axes

    
