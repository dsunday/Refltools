import numpy as np
import matplotlib.pyplot as plt
from refnx.reflect import SLD, Structure, ReflectModel, MaterialSLD

def estimate_points_per_range(angle_ranges, film_thickness, energy_ev, 
                             points_per_fringe=10, 
                             low_angle_points=None):
    """
    Estimate the number of measurement points needed for each angular range
    based on film thickness, energy, and desired points per fringe.
    
    Parameters:
    -----------
    angle_ranges : list of tuples
        List of (min_angle, max_angle) tuples in degrees
    film_thickness : float
        Total film thickness in Angstroms
    energy_ev : float
        Energy in eV
    points_per_fringe : int, optional
        Number of points desired per interference fringe, default is 10
    low_angle_points : int, optional
        Specific number of points for the lowest angle range
        If None, will calculate based on points_per_fringe
        
    Returns:
    --------
    points_per_range : list of int
        Number of points for each angular range
    """
    # Calculate wavelength from energy
    wavelength = 12398.0 / energy_ev
    
    points_per_range = []
    for i, (min_angle, max_angle) in enumerate(angle_ranges):
        # Convert angles to q values
        q_min = 4 * np.pi * np.sin(np.radians(min_angle)) / wavelength
        q_max = 4 * np.pi * np.sin(np.radians(max_angle)) / wavelength
        
        # Calculate fringe spacing (delta_q) based on film thickness
        # For a film of thickness d, fringes occur at intervals of delta_q = 2π/d
        delta_q = 2 * np.pi / film_thickness
        
        # Calculate number of fringes in this q range
        num_fringes = (q_max - q_min) / delta_q
        
        # Calculate points needed for this range
        if i == 0 and low_angle_points is not None:
            # Use specified number of points for low angle range
            points_needed = low_angle_points
        else:
            # Calculate based on points per fringe
            points_needed = int(np.ceil(num_fringes * points_per_fringe))
            
        # Ensure at least a minimum number of points (e.g., 5)
        points_needed = max(points_needed, 5)
        
        points_per_range.append(points_needed)
    
    return points_per_range

def create_q_array_auto(angle_ranges, film_thickness, energy_ev, 
                       points_per_fringe=10, 
                       low_angle_points=None):
    """
    Automatically create a q array based on film thickness and desired points per fringe.
    
    Parameters:
    -----------
    angle_ranges : list of tuples
        List of (min_angle, max_angle) tuples in degrees
    film_thickness : float
        Total film thickness in Angstroms
    energy_ev : float
        Energy in eV
    points_per_fringe : int, optional
        Number of points desired per interference fringe, default is 10
    low_angle_points : int, optional
        Specific number of points for the lowest angle range
        
    Returns:
    --------
    q_array : ndarray
        Array of q values in inverse Angstroms
    theta_array : ndarray
        Array of theta values in degrees
    points_per_range : list of int
        Number of points calculated for each range
    """
    # Calculate number of points per range
    points_per_range = estimate_points_per_range(
        angle_ranges, film_thickness, energy_ev, 
        points_per_fringe, low_angle_points
    )
    
    # Calculate wavelength
    wavelength = 12398.0 / energy_ev
    
    # Create theta array from angle ranges
    theta_sections = []
    for (min_angle, max_angle), n_points in zip(angle_ranges, points_per_range):
        # Create linear space for each angular range
        theta_section = np.linspace(min_angle, max_angle, n_points)
        theta_sections.append(theta_section)
    
    # Combine all theta sections
    theta_array = np.concatenate(theta_sections)
    
    # Convert theta to q: q = 4π sin(θ) / λ
    q_array = 4 * np.pi * np.sin(np.radians(theta_array)) / wavelength
    
    return q_array, theta_array, points_per_range

def estimate_measurement_time(points_per_range, time_per_point, total_overhead, point_overhead):
    """
    Estimate the measurement time based on points and time parameters.
    
    Parameters:
    -----------
    points_per_range : list of int
        Number of points for each angular range
    time_per_point : list of float
        Time per point (in seconds) for each angular range
    total_overhead : float
        Total overhead time (in seconds) for the entire measurement
    point_overhead : float
        Time overhead (in seconds) per data point for motor movement
        
    Returns:
    --------
    total_time : float
        Total estimated measurement time in seconds (including overheads)
    time_per_range : list of float
        Estimated time for each angular range in seconds (including point overhead)
    """
    if len(points_per_range) != len(time_per_point):
        raise ValueError("Number of ranges must match number of time_per_point values")
    
    # Calculate time for each range (including per-point overhead)
    time_per_range = []
    counting_time = 0
    
    for points, time_point in zip(points_per_range, time_per_point):
        # Calculate time for this range including per-point movement overhead
        range_time = points * (time_point + point_overhead)
        time_per_range.append(range_time)
        counting_time += range_time
    
    # Add total overhead to get total time
    total_time = counting_time + total_overhead
    
    return total_time, time_per_range

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
        
    Returns:
    --------
    formatted_time : str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hrs"

def compare_point_densities(angle_ranges, film_thickness, energy_ev, model,
                           points_per_fringe_options=[5, 10, 15],
                           low_angle_points=None,
                           time_per_point=None,
                           total_overhead=None,
                           point_overhead=None):
    """
    Compare different point density selections for reflectivity measurements.
    
    Parameters:
    -----------
    angle_ranges : list of tuples
        List of (min_angle, max_angle) tuples in degrees
    film_thickness : float
        Total film thickness in Angstroms
    energy_ev : float
        Energy in eV
    model : ReflectModel
        The reflectivity model to use for simulations
    points_per_fringe_options : list of int, optional
        Different options for points per fringe to compare
    low_angle_points : int, optional
        Specific number of points for the lowest angle range
    time_per_point : list of float, optional
        Time per point (in seconds) for each angular range
        If provided, will include time estimates
    total_overhead : float, optional
        Total overhead time (in seconds) for the entire measurement
    point_overhead : float, optional
        Time overhead (in seconds) per data point for motor movement
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure for further customization if needed
    """
    # Calculate wavelength
    wavelength = 12398.0 / energy_ev
    
    # Check if time estimation is requested
    include_time = (time_per_point is not None) and (total_overhead is not None)
    
    # Create figure for comparison
    fig, axes = plt.subplots(len(points_per_fringe_options), 1, 
                            figsize=(12, 4*len(points_per_fringe_options)),
                            sharex=True)
    
    if len(points_per_fringe_options) == 1:
        axes = [axes]  # Make sure axes is a list even for a single subplot
    
    # For total points comparison
    total_points_data = []
    
    # Loop through different point densities
    for i, ppf in enumerate(points_per_fringe_options):
        # Get q array
        q_values, theta_values, points_per_range = create_q_array_auto(
            angle_ranges, film_thickness, energy_ev, 
            points_per_fringe=ppf, 
            low_angle_points=low_angle_points
        )
        
        # Calculate reflectivity using the provided model
        reflectivity = model(q_values)
        
        # Plot as continuous line
        axes[i].semilogy(q_values, reflectivity, 'b-', alpha=0.3, label='Reflectivity Curve')
        
        # Plot individual points with different colors for each range
        start_idx = 0
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        total_points = 0
        
        for j, n_points in enumerate(points_per_range):
            end_idx = start_idx + n_points
            range_q = q_values[start_idx:end_idx]
            range_r = reflectivity[start_idx:end_idx]
            
            color = colors[j % len(colors)]
            axes[i].semilogy(range_q, range_r, 'o', color=color, 
                           label=f'Range {j+1}: {n_points} points')
            
            total_points += n_points
            start_idx = end_idx
        
        # Calculate time estimates if requested
        if include_time:
            total_time, time_per_range = estimate_measurement_time(
                points_per_range, time_per_point, total_overhead, point_overhead)
            total_time_formatted = format_time(total_time)
            time_per_range_formatted = [format_time(t) for t in time_per_range]
            counting_time = sum(time_per_range)
            counting_time_formatted = format_time(counting_time)
            
            # Calculate pure counting time (without point overhead)
            pure_counting_times = []
            pure_counting_total = 0
            for points, time_point in zip(points_per_range, time_per_point):
                pure_time = points * time_point
                pure_counting_times.append(pure_time)
                pure_counting_total += pure_time
            pure_counting_times_formatted = [format_time(t) for t in pure_counting_times]
            pure_counting_total_formatted = format_time(pure_counting_total)
        else:
            total_time = None
            time_per_range = None
            total_time_formatted = None
            time_per_range_formatted = None
            counting_time = None
            counting_time_formatted = None
            pure_counting_times = None
            pure_counting_times_formatted = None
            pure_counting_total = None
            pure_counting_total_formatted = None
        
        # Record data for this density option
        total_points_data.append((
            ppf, 
            total_points, 
            points_per_range, 
            total_time_formatted, 
            time_per_range_formatted,
            counting_time_formatted,
            pure_counting_times_formatted,
            pure_counting_total_formatted
        ))
        
        # Set up plot
        axes[i].set_ylabel('Reflectivity')
        title = f'{ppf} points per fringe (Total: {total_points} points'
        if include_time:
            title += f', Time: {total_time_formatted}'
        title += ')'
        axes[i].set_title(title)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i].legend(loc='upper right')
    
    # Add overall labels
    axes[-1].set_xlabel('Q (Å⁻¹)')
    plt.suptitle(f'Comparison of Point Densities (Film: {film_thickness} Å, Energy: {energy_ev} eV)', 
                fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Print comparison table
    print("\nPoint Density Comparison Table:")
    
    if include_time:
        print("-" * 120)
        header = f"{'Points/':^10} | {'Total':^10} | {'Pure Count':^12} | {'Count+Move':^12} | {'Total':^12} | {'Range Breakdown (Points / Time)':^60}"
        subheader = f"{'Fringe':^10} | {'Points':^10} | {'Time':^12} | {'Time':^12} | {'Time':^12} | {' ':^60}"
        print(header)
        print(subheader)
        print("-" * 120)
        
        for ppf, total, dist, t_total, t_ranges, count_move_time, pure_times, pure_total in total_points_data:
            dist_time_str = " | ".join([f"{p} pts ({pt}/{t})" for p, pt, t in zip(dist, pure_times, t_ranges)])
            print(f"{ppf:^10} | {total:^10} | {pure_total:^12} | {count_move_time:^12} | {t_total:^12} | {dist_time_str}")
        
        print("-" * 120)
        print("Range times shown as: Pure counting time / Counting+movement time")
    else:
        print("-" * 60)
        print(f"{'Points/Fringe':^15} | {'Total Points':^15} | {'Distribution':^25}")
        print("-" * 60)
        
        for ppf, total, dist, _, _, _, _, _ in total_points_data:
            dist_str = ", ".join([str(p) for p in dist])
            print(f"{ppf:^15} | {total:^15} | {dist_str:^25}")
        
        print("-" * 60)
    
    if low_angle_points is not None:
        print(f"* First range fixed at {low_angle_points} points")
    
    # Print time parameters if available
    if include_time:
        print("\nTime Parameters:")
        print(f"- Total setup overhead: {format_time(total_overhead)}")
        print(f"- Per-point movement overhead: {format_time(point_overhead)}")
        print("- Count time per point by range:")
        for i, t in enumerate(time_per_point):
            print(f"  * Range {i+1}: {format_time(t)} per point")
    
    # Calculate fringe delta_q for reference
    delta_q = 2 * np.pi / film_thickness
    print(f"\nFringe spacing: ΔQ = {delta_q:.6f} Å⁻¹ (for {film_thickness} Å film)")
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Example: Three angular ranges
    angle_ranges = [(0.05, 12), (11,20), (19, 30),(29,50)]
    
    # Film parameters
    film_thickness = 250.0  # Angstroms - a relatively thick film
    energy_ev = 290.0  # 8 keV X-rays
    
    # Create a simple model for comparison
    air = SLD(0.0, name='air')
    film = SLD(3.47, name='film')
    substrate = SLD(2.07, name='substrate')
    structure = air(0, 0) | film(film_thickness, 3.0) | substrate(0, 3.0)
    model = ReflectModel(structure, scale=1.0, bkg=1e-7, dq=2.0)
    
    # Time parameters (seconds)
    time_per_point = [0.1, 0.2, 0.5,1]  # Counting time per point for each range
    total_overhead = 120  # 10 minutes total setup overhead
    point_overhead = 2  # 5 seconds per point for motor movement
    
    # Compare different point densities
    fig = compare_point_densities(
        angle_ranges, 
        film_thickness, 
        energy_ev,
        model,
        points_per_fringe_options=[8, 9, 10],
        low_angle_points=15,  # Specific number for low angle range
        time_per_point=time_per_point,
        total_overhead=total_overhead,
        point_overhead=point_overhead
    )
    
    plt.show()