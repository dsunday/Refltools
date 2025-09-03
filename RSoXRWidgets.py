import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import re
from matplotlib.colors import to_hex
import matplotlib.cm as cm


class ReflectivityViewer:
    def __init__(self, data_dir=".", file_prefix="", file_pattern="*eV.dat"):
        """
        Initialize the reflectivity curves viewer widget
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the reflectivity data files
        file_prefix : str
            Prefix for the data files (e.g., "Prefix_")
        file_pattern : str
            Pattern to match the data files
        """
        self.data_dir = data_dir
        
        # Create the complete pattern for file searching
        self.pattern = os.path.join(data_dir, f"{file_prefix}{file_pattern}")
        
        # Find all matching files
        self.files = sorted(glob.glob(self.pattern))
        
        # Extract energies from filenames
        self.energies = []
        for file in self.files:
            # Extract energy using regex
            # Look for a number followed by "eV" in the filename
            energy_match = re.search(r'(\d+(?:\.\d+)?)eV', os.path.basename(file))
            if energy_match:
                self.energies.append(float(energy_match.group(1)))
            else:
                # If no match, use a default placeholder
                self.energies.append(0.0)
        
        # Create a dictionary of energy to file mapping
        self.energy_to_file = dict(zip(self.energies, self.files))
        
        # Sort energies
        self.energies.sort()
        
        # Initialize the widget components
        self.setup_widgets()
    
    def load_data(self, filename):
        """
        Load reflectivity data from a file
        
        Parameters:
        -----------
        filename : str
            Path to the data file
            
        Returns:
        --------
        data : numpy array
            Loaded data with columns [Q, R, error (if available)]
        """
        try:
            data = np.loadtxt(filename)
            return data
        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")
            return None
    
    def setup_widgets(self):
        """Create and setup the interactive widgets"""
        # Create dropdown for energy selection
        energy_options = [(f"{energy:.1f} eV", energy) for energy in self.energies]
        self.energy_dropdown = widgets.Dropdown(
            options=energy_options,
            description='Energy:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Create slider for energy selection (as an alternative)
        if self.energies:
            self.energy_slider = widgets.FloatSlider(
                value=self.energies[0],
                min=min(self.energies),
                max=max(self.energies),
                step=0.1,
                description='Energy:',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
                layout=widgets.Layout(width='500px')
            )
        else:
            self.energy_slider = widgets.FloatSlider(
                value=0.0,
                min=0.0,
                max=1.0,
                description='Energy:',
                disabled=True
            )
        
        # Create button for previous and next
        self.prev_button = widgets.Button(
            description='Previous',
            disabled=False,
            button_style='',
            tooltip='Go to previous energy',
            icon='arrow-left'
        )
        
        self.next_button = widgets.Button(
            description='Next',
            disabled=False,
            button_style='',
            tooltip='Go to next energy',
            icon='arrow-right'
        )
        
        # Create checkboxes for plot options
        self.log_scale = widgets.Checkbox(
            value=True,
            description='Log Y-Scale',
            disabled=False,
            layout=widgets.Layout(width='150px')
        )
        
        self.normalize = widgets.Checkbox(
            value=False,
            description='Normalize',
            disabled=False,
            layout=widgets.Layout(width='150px')
        )
        
        # Create plot output
        self.plot_output = widgets.Output()
        
        # Connect event handlers
        self.energy_dropdown.observe(self.update_plot, names='value')
        self.energy_slider.observe(self.on_slider_change, names='value')
        self.prev_button.on_click(self.on_prev_button_click)
        self.next_button.on_click(self.on_next_button_click)
        self.log_scale.observe(self.update_plot, names='value')
        self.normalize.observe(self.update_plot, names='value')
        
        # Create widget layout
        self.create_layout()
    
    def create_layout(self):
        """Create the layout for the widgets"""
        # Navigation controls
        nav_controls = widgets.HBox([self.prev_button, self.next_button])
        
        # Top controls
        top_controls = widgets.VBox([
            widgets.HBox([widgets.Label('Select Energy:'), self.energy_dropdown]),
            widgets.HBox([widgets.Label('Energy Slider:'), self.energy_slider]),
            nav_controls
        ])
        
        # Plot options
        plot_options = widgets.HBox([self.log_scale, self.normalize])
        
        # Combine all widgets
        self.widget = widgets.VBox([
            top_controls,
            plot_options,
            self.plot_output
        ])
    
    def on_slider_change(self, change):
        """Handle slider value change by finding the closest energy"""
        slider_value = change.new
        
        # Find the closest energy value to the slider value
        closest_energy = min(self.energies, key=lambda x: abs(x - slider_value))
        
        # Update the dropdown to match (without triggering the dropdown's observe)
        self.energy_dropdown.unobserve(self.update_plot, names='value')
        self.energy_dropdown.value = closest_energy
        self.energy_dropdown.observe(self.update_plot, names='value')
        
        # Update the plot
        self.update_plot(None)
    
    def on_prev_button_click(self, b):
        """Go to the previous energy in the list"""
        current_index = self.energies.index(self.energy_dropdown.value)
        if current_index > 0:
            new_index = current_index - 1
            new_energy = self.energies[new_index]
            
            # Update dropdown and slider
            self.energy_dropdown.value = new_energy
            self.energy_slider.value = new_energy
    
    def on_next_button_click(self, b):
        """Go to the next energy in the list"""
        current_index = self.energies.index(self.energy_dropdown.value)
        if current_index < len(self.energies) - 1:
            new_index = current_index + 1
            new_energy = self.energies[new_index]
            
            # Update dropdown and slider
            self.energy_dropdown.value = new_energy
            self.energy_slider.value = new_energy
    
    def update_plot(self, change):
        """Update the plot based on the current selection"""
        # Clear previous output
        self.plot_output.clear_output(wait=True)
        
        with self.plot_output:
            # Get the selected energy
            energy = self.energy_dropdown.value
            
            # Get the corresponding file
            if energy in self.energy_to_file:
                filename = self.energy_to_file[energy]
                
                # Load the data
                data = self.load_data(filename)
                
                if data is not None and data.shape[0] > 0:
                    # Extract Q and R values
                    q_values = data[:, 0]
                    r_values = data[:, 1]
                    
                    # Get plot options
                    use_log_scale = self.log_scale.value
                    do_normalize = self.normalize.value
                    
                    # Normalize if requested
                    if do_normalize and np.max(r_values) > 0:
                        r_values = r_values / np.max(r_values)
                    
                    # Create the plot
                    plt.figure(figsize=(10, 6))
                    
                    # Plot the data
                    plt.plot(q_values, r_values, 'o-', markersize=4)
                    
                    # Set scale and limits
                    if use_log_scale:
                        plt.yscale('log')
                        
                        # Adjust y-scale to have the minimum below the lowest data point
                        non_zero_mask = r_values > 0
                        if np.any(non_zero_mask):  # Check if there are any non-zero values
                            min_r_value = np.min(r_values[non_zero_mask])  # Find minimum non-zero value
                            
                            # Set y-limit to show slightly more than one order of magnitude below the minimum
                            plt.ylim(bottom=min_r_value / 10)
                    
                    # Add error bars if available
                    if data.shape[1] > 2:
                        error_values = data[:, 2]
                        plt.errorbar(q_values, r_values, yerr=error_values, fmt='none', alpha=0.5)
                    
                    # Set labels and title
                    plt.xlabel('Q (Å⁻¹)')
                    plt.ylabel('Reflectivity' + (' (Normalized)' if do_normalize else ''))
                    plt.title(f'Reflectivity Curve - {energy:.1f} eV')
                    
                    # Add grid
                    plt.grid(True, which='both', linestyle='--', alpha=0.7)
                    
                    # Add file information
                    plt.figtext(0.02, 0.02, f'File: {os.path.basename(filename)}', fontsize=8)
                    
                    # Display the plot
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"No valid data found in file: {filename}")
            else:
                print(f"No file found for energy {energy:.1f} eV")
    
    def display(self):
        """Display the widget"""
        display(self.widget)
        
        # Initialize the plot if we have files
        if self.energies:
            self.update_plot(None)
        else:
            with self.plot_output:
                print("No reflectivity data files found matching the pattern.")


# Example usage:
def create_reflectivity_viewer(data_dir=".", file_prefix="", file_pattern="*eV.dat"):
    """
    Create and display a reflectivity viewer widget
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the reflectivity data files
    file_prefix : str
        Prefix for the data files (e.g., "Prefix_")
    file_pattern : str
        Pattern to match the data files
    """
    viewer = ReflectivityViewer(data_dir=data_dir, file_prefix=file_prefix, file_pattern=file_pattern)
    viewer.display()
    return viewer

# Show example usage
print("Example usage:")
print("viewer = create_reflectivity_viewer(data_dir='data', file_prefix='Prefix_')")
print("# Or with default parameters (current directory, no specific prefix):")
print("viewer = create_reflectivity_viewer()")


class MultiCurveViewer:
    """
    Widget for viewing multiple reflectivity curves simultaneously
    """
    
    def __init__(self, data_dir=".", file_prefix="", file_pattern="*eV.dat"):
        """
        Initialize the multiple reflectivity curves viewer widget
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the reflectivity data files
        file_prefix : str
            Prefix for the data files (e.g., "Prefix_")
        file_pattern : str
            Pattern to match the data files
        """
        self.data_dir = data_dir
        
        # Create the complete pattern for file searching
        self.pattern = os.path.join(data_dir, f"{file_prefix}{file_pattern}")
        
        # Find all matching files
        self.files = sorted(glob.glob(self.pattern))
        
        # Extract energies from filenames
        self.energies = []
        for file in self.files:
            # Extract energy using regex
            # Look for a number followed by "eV" in the filename
            energy_match = re.search(r'(\d+(?:\.\d+)?)eV', os.path.basename(file))
            if energy_match:
                self.energies.append(float(energy_match.group(1)))
            else:
                # If no match, use a default placeholder
                self.energies.append(0.0)
        
        # Create a dictionary of energy to file mapping
        self.energy_to_file = dict(zip(self.energies, self.files))
        
        # Sort energies
        self.energies.sort()
        
        # Track which energies are selected
        self.selected_energies = set()
        
        # Initialize the widget components
        self.setup_widgets()
    
    def load_data(self, filename):
        """
        Load reflectivity data from a file
        
        Parameters:
        -----------
        filename : str
            Path to the data file
            
        Returns:
        --------
        data : numpy array
            Loaded data with columns [Q, R, error (if available)]
        """
        try:
            data = np.loadtxt(filename)
            return data
        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")
            return None
    
    def setup_widgets(self):
        """Create and setup the interactive widgets"""
        # Create a list of energies with checkboxes
        self.energy_checkboxes = []
        
        # Create a layout for the checkboxes (4 columns)
        items_per_column = max(1, len(self.energies) // 4 + (1 if len(self.energies) % 4 else 0))
        checkbox_columns = []
        
        # Build the checkboxes in columns
        for i in range(0, len(self.energies), items_per_column):
            column_checkboxes = []
            for energy in self.energies[i:i+items_per_column]:
                checkbox = widgets.Checkbox(
                    value=False,
                    description=f"{energy:.1f} eV",
                    disabled=False,
                    indent=False,
                    layout=widgets.Layout(width='120px')
                )
                checkbox.energy = energy  # Store the energy value as an attribute
                checkbox.observe(self.on_checkbox_change, names='value')
                column_checkboxes.append(checkbox)
                self.energy_checkboxes.append(checkbox)
                
            checkbox_columns.append(widgets.VBox(column_checkboxes))
        
        # Arrange the checkboxes in a grid
        self.energy_selection = widgets.HBox(checkbox_columns)
        
        # Create buttons for quick selection
        self.select_all_button = widgets.Button(
            description='Select All',
            disabled=False,
            button_style='',
            tooltip='Select all energies',
            layout=widgets.Layout(width='100px')
        )
        
        self.clear_all_button = widgets.Button(
            description='Clear All',
            disabled=False,
            button_style='',
            tooltip='Clear all selections',
            layout=widgets.Layout(width='100px')
        )
        
        # Create checkboxes for plot options
        self.log_scale = widgets.Checkbox(
            value=True,
            description='Log Y-Scale',
            disabled=False,
            layout=widgets.Layout(width='150px')
        )
        
        self.normalize = widgets.Checkbox(
            value=False,
            description='Normalize',
            disabled=False,
            layout=widgets.Layout(width='150px')
        )
        
        self.show_legend = widgets.Checkbox(
            value=True,
            description='Show Legend',
            disabled=False,
            layout=widgets.Layout(width='150px')
        )
        
        self.use_colors = widgets.Checkbox(
            value=True,
            description='Use Colors',
            disabled=False,
            layout=widgets.Layout(width='150px')
        )
        
        # Create plot output
        self.plot_output = widgets.Output()
        
        # Connect event handlers
        self.select_all_button.on_click(self.on_select_all)
        self.clear_all_button.on_click(self.on_clear_all)
        self.log_scale.observe(self.update_plot, names='value')
        self.normalize.observe(self.update_plot, names='value')
        self.show_legend.observe(self.update_plot, names='value')
        self.use_colors.observe(self.update_plot, names='value')
        
        # Create widget layout
        self.create_layout()
    
    def create_layout(self):
        """Create the layout for the widgets"""
        # Selection controls
        selection_controls = widgets.HBox([
            widgets.Label('Select Energies:'),
            self.select_all_button,
            self.clear_all_button
        ])
        
        # Top controls
        top_controls = widgets.VBox([
            selection_controls,
            self.energy_selection
        ])
        
        # Plot options
        plot_options = widgets.HBox([
            self.log_scale, 
            self.normalize, 
            self.show_legend, 
            self.use_colors
        ])
        
        # Combine all widgets
        self.widget = widgets.VBox([
            top_controls,
            plot_options,
            self.plot_output
        ])
    
    def on_checkbox_change(self, change):
        """Handle checkbox state change"""
        checkbox = change.owner
        if hasattr(checkbox, 'energy'):
            energy = checkbox.energy
            
            if change.new:  # If checked
                self.selected_energies.add(energy)
            else:  # If unchecked
                if energy in self.selected_energies:
                    self.selected_energies.remove(energy)
            
            # Update the plot
            self.update_plot(None)
    
    def on_select_all(self, b):
        """Select all energies"""
        for checkbox in self.energy_checkboxes:
            checkbox.value = True
    
    def on_clear_all(self, b):
        """Clear all selections"""
        for checkbox in self.energy_checkboxes:
            checkbox.value = False
    
    def update_plot(self, change):
        """Update the plot based on the current selections"""
        # Clear previous output
        self.plot_output.clear_output(wait=True)
        
        with self.plot_output:
            if not self.selected_energies:
                print("Please select at least one energy to display.")
                return
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Get plot options
            use_log_scale = self.log_scale.value
            do_normalize = self.normalize.value
            show_legend = self.show_legend.value
            use_colors = self.use_colors.value
            
            # Create a color map for multiple energies
            colormap = cm.viridis
            energy_colors = {}
            
            # Get min and max energy for color scaling
            min_energy = min(self.energies)
            max_energy = max(self.energies)
            energy_range = max_energy - min_energy
            
            if energy_range == 0:  # Prevent division by zero
                energy_range = 1.0
            
            # Sort selected energies for consistent plotting
            sorted_energies = sorted(self.selected_energies)
            
            # Process each selected energy
            for i, energy in enumerate(sorted_energies):
                if energy in self.energy_to_file:
                    filename = self.energy_to_file[energy]
                    
                    # Load the data
                    data = self.load_data(filename)
                    
                    if data is not None and data.shape[0] > 0:
                        # Extract Q and R values
                        q_values = data[:, 0]
                        r_values = data[:, 1]
                        
                        # Normalize if requested
                        if do_normalize and np.max(r_values) > 0:
                            r_values = r_values / np.max(r_values)
                        
                        # Choose color based on energy if use_colors is True
                        if use_colors:
                            # Use color based on energy position in the sorted list
                            # This ensures a good spread of colors across the spectrum
                            if len(sorted_energies) > 1:
                                norm_position = i / (len(sorted_energies) - 1)
                            else:
                                norm_position = 0.5  # Single energy gets middle color
                                
                            color = to_hex(colormap(norm_position))
                            energy_colors[energy] = color
                        else:
                            # Use default color cycle
                            color = None
                        
                        # Plot the data
                        label = f"{energy:.1f} eV"
                        plt.plot(q_values, r_values, 'o-', markersize=3, label=label, color=color)
                        
                        # Add error bars if available
                        if data.shape[1] > 2:
                            error_values = data[:, 2]
                            plt.errorbar(q_values, r_values, yerr=error_values, fmt='none', alpha=0.3, color=color)
            
            # Set y-scale
            if use_log_scale:
                plt.yscale('log')
                
                # Find the minimum non-zero value across all datasets
                global_min = float('inf')
                for energy in sorted_energies:
                    if energy in self.energy_to_file:
                        data = self.load_data(self.energy_to_file[energy])
                        if data is not None and data.shape[0] > 0:
                            r_values = data[:, 1]
                            
                            # Normalize if requested
                            if do_normalize and np.max(r_values) > 0:
                                r_values = r_values / np.max(r_values)
                            
                            # Find minimum non-zero value
                            non_zero_mask = r_values > 0
                            if np.any(non_zero_mask):
                                min_value = np.min(r_values[non_zero_mask])
                                global_min = min(global_min, min_value)
                
                # Set y-limit if we found a valid minimum
                if global_min != float('inf'):
                    plt.ylim(bottom=global_min / 10)
            
            # Add legend if requested
            if show_legend:
                plt.legend(loc='best')
            
            # Set labels and title
            plt.xlabel('Q (Å⁻¹)')
            if do_normalize:
                plt.ylabel('Normalized Reflectivity')
            else:
                plt.ylabel('Reflectivity')
            
            if len(sorted_energies) == 1:
                energy = next(iter(sorted_energies))
                plt.title(f'Reflectivity Curve - {energy:.1f} eV')
            else:
                plt.title(f'Reflectivity Curves - Multiple Energies')
            
            # Add grid
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # Display the plot
            plt.tight_layout()
            plt.show()
    
    def display(self):
        """Display the widget"""
        display(self.widget)
        
        # Initialize with a message
        with self.plot_output:
            print("Please select at least one energy to display.")


# Example usage:
def create_multi_curve_viewer(data_dir=".", file_prefix="", file_pattern="*eV.dat"):
    """
    Create and display a widget for viewing multiple reflectivity curves
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the reflectivity data files
    file_prefix : str
        Prefix for the data files (e.g., "Prefix_")
    file_pattern : str
        Pattern to match the data files
    """
    viewer = MultiCurveViewer(data_dir=data_dir, file_prefix=file_prefix, file_pattern=file_pattern)
    viewer.display()
    return viewer

# Show example usage
print("Example usage:")
print("viewer = create_multi_curve_viewer(data_dir='data', file_prefix='Prefix_')")
print("# Or with default parameters (current directory, no specific prefix):")
print("viewer = create_multi_curve_viewer()")


import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import glob
from copy import deepcopy
import pandas as pd
from scipy.signal import savgol_filter

class RSoXRTrimWidget:
    """
    Interactive widget for trimming RSoXR data groups with stitching preview
    Enhanced to support manually edited groups and background subtraction
    """
    
    def __init__(self, processor, scan_groups=None, data_directory='.', default_output_dir='trimmed_output', open_beam=None):
        """
        Initialize the widget
        
        Parameters:
        -----------
        processor : RSoXRProcessor
            The initialized RSoXR processor instance
        scan_groups : list, optional
            List of scan groups. If provided, these groups will be used directly.
            If None, auto-detect scan groups from data_directory.
        data_directory : str
            Directory containing the data files (used only if scan_groups not provided)
        default_output_dir : str
            Default output directory for saved files
        open_beam : str, optional
            Path to open beam file. If provided, will automatically load open beam data
        """
        self.processor = processor
        self.data_directory = data_directory
        self.default_output_dir = default_output_dir
        
        if scan_groups is None:
            # Auto-detect scan groups if not provided
            self.scan_groups = self.processor.auto_group_scans(
                data_directory=data_directory, 
                save_table=False
            )
            self.groups_source = "autoscan"
        else:
            # Use provided groups (assumed to be manually edited)
            self.scan_groups = scan_groups
            self.groups_source = "provided"
        
        # Set default trim values to min and max points of each section
        self._set_default_trim_values()
            
        # Store the original trims for each group
        self.original_trims = {}
        for i, group in enumerate(self.scan_groups):
            self.original_trims[i] = deepcopy(group['trims'])
        
        # Store the current trims for each group (may be modified by user)
        self.current_trims = deepcopy(self.original_trims)
        
        # Initialize background subtraction storage for each group/file
        # Each entry will be a constant background value to subtract
        self.current_backgrounds = {}
        for i, group in enumerate(self.scan_groups):
            self.current_backgrounds[i] = [0.0] * len(group['files'])  # Default: no background subtraction
        
        # Store loaded scan data for the current group
        self.current_group_data = None
        
        # Store current file data
        self.current_data = None
        
        # Create the widget UI
        self._create_widgets()
        
        # Load open beam data if provided
        if open_beam is not None:
            self._load_open_beam_on_init(open_beam)

    
    def _set_default_trim_values(self):
        """
        Set default trim values to use min and max points of each section
        (0 for start, -1 for end, which means use all data points)
        """
        for group in self.scan_groups:
            # Set trim values to use the full range of each file
            group['trims'] = [(0, -1)] * len(group['files'])
    
    def _create_widgets(self):
        """Create the interactive widgets"""
        # Select group widget
        group_options = []
        for i, group in enumerate(self.scan_groups):
            group_options.append(f"{i+1}: {group['sample_name']} ({group['energy']:.1f} eV)")
        
        self.group_select = widgets.Dropdown(
            options=group_options,
            description='Group:',
            disabled=False,
            layout=widgets.Layout(width='50%')
        )
        
        # Select file within group widget
        self.file_select = widgets.Dropdown(
            options=[],
            description='File:',
            disabled=False,
            layout=widgets.Layout(width='50%')
        )
        
        # Trim start and end value display with manual adjustment
        self.trim_start_display = widgets.IntText(
            value=0,
            description='Trim Start:',
            disabled=False,
            layout=widgets.Layout(width='20%')
        )
        
        self.trim_end_display = widgets.IntText(
            value=0,
            description='Trim End:',
            disabled=False,
            layout=widgets.Layout(width='20%')
        )
        
        # Add buttons to apply manual trim values
        self.apply_trim_button = widgets.Button(
            description='Apply Trim Values',
            disabled=False,
            button_style='primary',
            tooltip='Apply manually entered trim values',
            icon='check',
            layout=widgets.Layout(width='20%')
        )
        
        # Background subtraction controls
        self.background_value = widgets.FloatText(
            value=0.0,
            description='Background:',
            disabled=False,
            layout=widgets.Layout(width='20%'),
            step=0.001,
            tooltip='Constant background value to subtract from this section'
        )
        
        self.apply_background_button = widgets.Button(
            description='Apply Background',
            disabled=False,
            button_style='info',
            tooltip='Apply background subtraction value',
            icon='minus',
            layout=widgets.Layout(width='20%')
        )
        
        # Display selected vs. original trim info
        self.trim_info = widgets.HTML(
            value="",
            description='',
            layout=widgets.Layout(width='40%')
        )
        
        # Background info display
        self.background_info = widgets.HTML(
            value="",
            description='',
            layout=widgets.Layout(width='40%')
        )
        
        # Reset button for current file
        self.reset_file_button = widgets.Button(
            description='Reset This File',
            disabled=False,
            button_style='warning',
            tooltip='Reset trims and background for the current file',
            icon='undo'
        )
        
        # Reset all button for current group
        self.reset_group_button = widgets.Button(
            description='Reset Group',
            disabled=False,
            button_style='danger',
            tooltip='Reset all trims and backgrounds for the current group',
            icon='refresh'
        )
        
        # New button for previewing stitching between curves
        self.preview_stitching_button = widgets.Button(
            description='Preview Stitching',
            disabled=False,
            button_style='info',
            tooltip='Preview how the scans stitch together with current trim and background values',
            icon='eye'
        )
        
        # Process button to run data reduction with current trims
        self.process_button = widgets.Button(
            description='Process Group',
            disabled=False,
            button_style='success',
            tooltip='Process this group with current trim and background settings',
            icon='check'
        )
        
        # Save trims to file button
        self.save_button = widgets.Button(
            description='Save Settings',
            disabled=False,
            button_style='info',
            tooltip='Save trim and background values to CSV file',
            icon='save'
        )
        
        # Processing options
        self.normalize_checkbox = widgets.Checkbox(
            value=True,
            description='Normalize',
            disabled=False
        )
        
        self.smooth_checkbox = widgets.Checkbox(
            value=False,
            description='Smooth Data',
            disabled=False
        )
        
        self.remove_zeros_checkbox = widgets.Checkbox(
            value=True,
            description='Remove Zeros',
            disabled=False
        )
        
        # Log scale for y-axis
        self.log_scale_checkbox = widgets.Checkbox(
            value=True,
            description='Log Scale',
            disabled=False
        )
        
        # Convert to photons
        self.convert_photons_checkbox = widgets.Checkbox(
            value=False,
            description='Convert to Photons',
            disabled=False
        )
        
        # Smoothing parameters
        self.smoothing_window = widgets.IntText(
            value=15,
            description='Window Size:',
            disabled=False,
            layout=widgets.Layout(width='250px'),
            style={'description_width': 'initial'}
        )
        
        # Add helper text for window size
        self.window_info = widgets.HTML(
            value="<i style='font-size:0.9em;'>Must be odd and ≥ 5</i>",
            layout=widgets.Layout(width='200px')
        )
        
        self.smoothing_order = widgets.IntText(
            value=2,
            description='Polynomial Order:',
            disabled=False,
            layout=widgets.Layout(width='250px'),
            style={'description_width': 'initial'}
        )
        
        # Add helper text for polynomial order
        self.order_info = widgets.HTML(
            value="<i style='font-size:0.9em;'>Must be ≤ window size-1</i>",
            layout=widgets.Layout(width='200px')
        )
        
        # Preview smoothing button
        self.preview_smooth_button = widgets.Button(
            description='Preview Smoothing',
            disabled=False,
            button_style='primary',
            tooltip='Preview smoothing with current parameters',
            icon='eye',
            layout=widgets.Layout(width='200px')
        )
        
        # Output directory - use the provided default
        self.output_dir = widgets.Text(
            value=self.default_output_dir,
            description='Output Directory:',
            disabled=False,
            layout=widgets.Layout(width='50%')
        )
        
        # Pre-process and preview checkboxes
        self.preview_remove_zeros = widgets.Checkbox(
            value=False,
            description='Preview Zero Removal',
            disabled=False
        )
        
        # Show background subtraction in preview
        self.preview_background = widgets.Checkbox(
            value=True,
            description='Preview Background Subtraction',
            disabled=False,
            tooltip='Show effect of background subtraction in plots'
        )
        
        # Group source info
        group_source_text = "✓ Using provided scan groups (manually edited)" if self.groups_source == "provided" else "Using autoscanned groups"
        self.group_source_info = widgets.HTML(
            value=f"<i style='color:#666;'>{group_source_text} | Total groups: {len(self.scan_groups)}</i>",
            layout=widgets.Layout(width='100%')
        )
        
        # Arrange widgets in containers
        self.group_file_container = widgets.HBox([self.group_select, self.file_select])
        
        # Modified trim display container to include background controls
        self.trim_display_container = widgets.HBox([
            self.trim_start_display, 
            self.trim_end_display, 
            self.apply_trim_button,
            self.trim_info
        ])
        
        self.background_container = widgets.HBox([
            self.background_value,
            self.apply_background_button,
            self.background_info
        ])
        
        # Moved smoothing options to the left side, stacked vertically
        self.smoothing_container = widgets.VBox([
            self.smooth_checkbox,
            self.smoothing_window,
            self.window_info,
            self.smoothing_order,
            self.order_info,
            self.preview_smooth_button
        ], layout=widgets.Layout(
            margin='0 20px 0 0'  # Add some right margin
        ))
        
        # Other processing options in another column
        self.processing_options_container = widgets.VBox([
            self.normalize_checkbox,
            self.remove_zeros_checkbox,
            self.preview_remove_zeros,
            self.log_scale_checkbox,
            self.convert_photons_checkbox,
            self.preview_background
        ])
        
        self.button_container = widgets.HBox([
            self.reset_file_button, 
            self.reset_group_button,
            self.preview_stitching_button,
            self.process_button,
            self.save_button
        ])
        
        self.open_beam_controls = self.add_open_beam_controls_to_widget()
        
        # Output area
        self.output_area = widgets.Output()
        
        # Main container with new layout (added group source info)
        self.main_container = widgets.VBox([
            widgets.HTML(value="<h2>RSoXR Interactive Trimming Tool with Background Subtraction</h2>"),
            self.group_source_info,  # Shows group source
            self.group_file_container,
            self.trim_display_container,
            self.background_container,  # New: Background subtraction controls
            self.open_beam_controls,
            widgets.HBox([
                self.smoothing_container,  # Left side
                self.processing_options_container  # Right side
            ]),
            self.output_dir,
            self.button_container,
            self.output_area
        ])
        
        # Register callbacks
        self.group_select.observe(self._on_group_change, names='value')
        self.file_select.observe(self._on_file_change, names='value')
        self.reset_file_button.on_click(self._on_reset_file)
        self.reset_group_button.on_click(self._on_reset_group)
        self.process_button.on_click(self._on_process_group)
        self.save_button.on_click(self._on_save_settings)
        self.log_scale_checkbox.observe(self._on_log_scale_change, names='value')
        self.preview_smooth_button.on_click(self._on_preview_smooth)
        self.apply_trim_button.on_click(self._on_apply_trim)
        self.apply_background_button.on_click(self._on_apply_background)
        self.preview_remove_zeros.observe(self._on_log_scale_change, names='value')
        self.preview_background.observe(self._on_log_scale_change, names='value')
        self.preview_stitching_button.on_click(self._on_preview_stitching)
        
        # Initialize by selecting the first group
        if group_options:
            self._on_group_change({'new': group_options[0]})
    
    def _apply_background_subtraction(self, data, background_value):
        """Apply background subtraction to data"""
        if background_value == 0.0:
            return data
        
        corrected_data = data.copy()
        corrected_data[:, 1] = corrected_data[:, 1] - background_value
        return corrected_data
    
    def _on_apply_background(self, b):
        """Apply manually entered background value"""
        # Get current indices
        group_str = self.group_select.value
        group_idx = int(group_str.split(':')[0]) - 1
        
        file_str = self.file_select.value
        file_idx = int(file_str.split(':')[0]) - 1
        
        # Get the manually entered background value
        background_value = self.background_value.value
        
        # Update the background values
        self.current_backgrounds[group_idx][file_idx] = background_value
        
        # Update the UI
        self._on_file_change({'new': self.file_select.value})
        
        print(f"Applied background subtraction: {background_value}")
    
    def _load_group_data(self, group_idx):
        """Load all data files for the current group"""
        self.current_group_data = []
        
        for i, filename in enumerate(self.scan_groups[group_idx]['files']):
            try:
                data = self.processor.load_data_file(filename)
                self.current_group_data.append(data)
            except Exception as e:
                print(f"Error loading data for {os.path.basename(filename)}: {str(e)}")
                self.current_group_data.append(None)
    
    def _on_group_change(self, change):
        """Handle group selection change"""
        # Extract group index from selection string (format: "1: name (energy eV)")
        group_str = change['new']
        group_idx = int(group_str.split(':')[0]) - 1
        
        # Update file selection dropdown with files from this group
        file_options = []
        for i, filename in enumerate(self.scan_groups[group_idx]['files']):
            # Get detector type if available
            detector_type = ""
            if group_idx < len(self.scan_groups) and i < len(self.scan_groups[group_idx]['metadata']):
                metadata = self.scan_groups[group_idx]['metadata'][i]
                if metadata and 'detector' in metadata:
                    detector_type = f" ({metadata['detector']})"
            
            file_options.append(f"{i+1}: {os.path.basename(filename)}{detector_type}")
        
        self.file_select.options = file_options
        
        # Clear current data
        self.current_data = None
        self.current_group_data = None
        
        # Select the first file
        if file_options:
            self.file_select.value = file_options[0]

    def _on_file_change(self, change):
        """Handle file selection change"""
        # Clear current data to force reload
        self.current_data = None
        
        # Extract indices from selection strings
        group_str = self.group_select.value
        group_idx = int(group_str.split(':')[0]) - 1
        
        file_str = self.file_select.value
        file_idx = int(file_str.split(':')[0]) - 1
        
        # Get current trim and background
        current_trim = self.current_trims[group_idx][file_idx]
        original_trim = self.original_trims[group_idx][file_idx]
        current_background = self.current_backgrounds[group_idx][file_idx]
        
        # Load data for visualization
        filename = self.scan_groups[group_idx]['files'][file_idx]
        try:
            self.current_data = self.processor.load_data_file(filename)
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                print(f"Error loading file: {str(e)}")
            self.current_data = None
            return
        
        # Update trim display
        self.trim_start_display.value = current_trim[0]
        if current_trim[1] == -1:
            self.trim_end_display.value = len(self.current_data)
        else:
            # Convert negative index to positive for display
            self.trim_end_display.value = len(self.current_data) + current_trim[1] if current_trim[1] < 0 else current_trim[1]
        
        # Update background display
        self.background_value.value = current_background
        
        # Update trim info display
        if current_trim != original_trim:
            self.trim_info.value = f"<span style='color:orange'>Modified: Original trim was {original_trim}</span>"
        else:
            self.trim_info.value = ""
        
        # Update background info display
        if current_background != 0.0:
            self.background_info.value = f"<span style='color:blue'>Background: -{current_background:.3f}</span>"
        else:
            self.background_info.value = ""
        
        # Plot the file with current trim and background
        self._plot_current_file()
    
    def _plot_current_file(self):
        """Plot the currently selected file with trim indicators and background subtraction"""
        # Get the selected indices
        group_str = self.group_select.value
        group_idx = int(group_str.split(':')[0]) - 1
        
        file_str = self.file_select.value
        file_idx = int(file_str.split(':')[0]) - 1
        
        # Get the filename and load the data if not already loaded
        filename = self.scan_groups[group_idx]['files'][file_idx]
        if self.current_data is None:
            try:
                self.current_data = self.processor.load_data_file(filename)
            except Exception as e:
                with self.output_area:
                    clear_output(wait=True)
                    print(f"Error loading file: {str(e)}")
                return
        
        data = self.current_data
        
        # Get current trim and background
        trim_start, trim_end = self.current_trims[group_idx][file_idx]
        background_value = self.current_backgrounds[group_idx][file_idx]
        
        # Clear the output area and create a new figure
        with self.output_area:
            clear_output(wait=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Process data for preview if requested
            plot_data = data
            if self.preview_remove_zeros.value:
                # Remove zero intensity points
                non_zero_mask = data[:, 1] > 0
                if not all(non_zero_mask):
                    print(f"Preview: Removing {len(data) - np.sum(non_zero_mask)} zero data points")
                    plot_data = data[non_zero_mask]
            
            # Apply background subtraction if preview is enabled
            if self.preview_background.value and background_value != 0.0:
                plot_data_bg = self._apply_background_subtraction(plot_data, background_value)
                # Plot original data
                ax.plot(plot_data[:, 0], plot_data[:, 1], 'b-', marker='o', alpha=0.5, markersize=3, label='Raw Data')
                # Plot background-subtracted data
                ax.plot(plot_data_bg[:, 0], plot_data_bg[:, 1], 'g-', marker='s', alpha=0.7, markersize=3, label=f'Background Corrected (-{background_value:.3f})')
                plot_data = plot_data_bg  # Use corrected data for trimming visualization
            else:
                # Plot the data
                ax.plot(plot_data[:, 0], plot_data[:, 1], 'b-', marker='o', alpha=0.7, markersize=4, label='Raw Data')
            
            # Highlight the trimmed data
            end_idx = len(data) + trim_end if trim_end < 0 else trim_end
            trimmed_data = data[trim_start:end_idx]
            
            # Apply background subtraction to trimmed data if preview is enabled
            if self.preview_background.value and background_value != 0.0:
                trimmed_data = self._apply_background_subtraction(trimmed_data, background_value)
            
            if len(trimmed_data) > 0:
                ax.plot(trimmed_data[:, 0], trimmed_data[:, 1], 'r-', marker='x', alpha=0.8, markersize=5, label='Selected Data')
            
            # Set y-scale based on checkbox
            if self.log_scale_checkbox.value:
                ax.set_yscale('log')
            else:
                ax.set_yscale('linear')
                
            # Add labels and title
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Intensity')
            ax.set_title(f"File: {os.path.basename(filename)}")
            
            # Add trim and background info to the plot
            info_text = f"Trim: [{trim_start}, {trim_end if trim_end != -1 else 'end'}]"
            if background_value != 0.0:
                info_text += f"\nBackground: -{background_value:.3f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='left',
                   bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
            
            # Add a legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
    
    def _on_apply_trim(self, b):
        """Apply manually entered trim values"""
        # Get current indices
        group_str = self.group_select.value
        group_idx = int(group_str.split(':')[0]) - 1
        
        file_str = self.file_select.value
        file_idx = int(file_str.split(':')[0]) - 1
        
        # Get the manually entered values
        start_trim = self.trim_start_display.value
        end_trim = self.trim_end_display.value
        
        # Validate start trim
        if start_trim < 0:
            start_trim = 0
            print(f"Warning: Start trim must be non-negative. Setting to 0.")
        elif start_trim >= len(self.current_data):
            start_trim = len(self.current_data) - 1
            print(f"Warning: Start trim too large. Setting to {start_trim}.")
            
        # Convert end trim to negative index if needed
        if end_trim == len(self.current_data):
            end_trim = -1
        elif end_trim > 0 and end_trim < len(self.current_data):
            end_trim = end_trim - len(self.current_data)
        
        # Validate end trim
        if end_trim != -1 and start_trim >= len(self.current_data) + end_trim:
            end_trim = -1
            print(f"Warning: End trim would result in no data. Setting to end of array.")
            
        # Update the trim values
        self.current_trims[group_idx][file_idx] = (start_trim, end_trim)
        
        # Update the UI
        self._on_file_change({'new': self.file_select.value})
        
        print(f"Applied trim: [{start_trim}, {end_trim if end_trim != -1 else 'end'}]")
    
    def _on_reset_file(self, b):
        """Reset trim and background for the current file to original values"""
        # Get current indices
        group_str = self.group_select.value
        group_idx = int(group_str.split(':')[0]) - 1
        
        file_str = self.file_select.value
        file_idx = int(file_str.split(':')[0]) - 1
        
        # Reset to original trim and clear background
        self.current_trims[group_idx][file_idx] = self.original_trims[group_idx][file_idx]
        self.current_backgrounds[group_idx][file_idx] = 0.0
        
        # Update the UI
        self._on_file_change({'new': self.file_select.value})
        
        with self.output_area:
            print(f"Reset trim values for file {file_idx+1} to original: {self.original_trims[group_idx][file_idx]}")
            print(f"Reset background to 0.0")
    
    def _on_reset_group(self, b):
        """Reset all trims and backgrounds for the current group to original values"""
        # Get current group index
        group_str = self.group_select.value
        group_idx = int(group_str.split(':')[0]) - 1
        
        # Reset to original trims and clear backgrounds
        self.current_trims[group_idx] = deepcopy(self.original_trims[group_idx])
        self.current_backgrounds[group_idx] = [0.0] * len(self.scan_groups[group_idx]['files'])
        
        # Update the UI
        self._on_file_change({'new': self.file_select.value})
        
        with self.output_area:
            clear_output(wait=True)
            print(f"Reset all trim values for group {group_idx+1} to original values")
            print(f"Reset all background values to 0.0")
            print("Use the 'Preview Stitching' button to see how the original values affect the stitching.")

    def _on_preview_stitching(self, b):
        """Preview the stitching between different scans with current trim and background values"""
        try:
            # Get current group
            group_str = self.group_select.value
            group_idx = int(group_str.split(':')[0]) - 1
            
            group = self.scan_groups[group_idx]
            energy = group['energy']
            
            # Ensure all data is loaded
            if not self.current_group_data or any(data is None for data in self.current_group_data):
                self._load_group_data(group_idx)
            
            # Check if we have valid data
            if not self.current_group_data:
                with self.output_area:
                    clear_output(wait=True)
                    print("Error: No data loaded for current group. Please select a valid group.")
                return
            
            # Prepare trimmed and background-corrected data
            scans = []
            for i, data in enumerate(self.current_group_data):
                if data is None:
                    print(f"Warning: Data for scan {i+1} is not available")
                    continue
                    
                # Apply current trim
                trim_start, trim_end = self.current_trims[group_idx][i]
                end_idx = len(data) + trim_end if trim_end < 0 else trim_end
                trimmed_data = data[trim_start:end_idx].copy()  # Make a copy to avoid modifying original
                
                # Apply background subtraction
                background_value = self.current_backgrounds[group_idx][i]
                if background_value != 0.0:
                    trimmed_data = self._apply_background_subtraction(trimmed_data, background_value)
                
                # Remove zeros if requested
                if self.remove_zeros_checkbox.value:
                    non_zero_mask = trimmed_data[:, 1] > 0
                    if not all(non_zero_mask):
                        print(f"Removing {len(trimmed_data) - np.sum(non_zero_mask)} zero points from scan {i+1}")
                        trimmed_data = trimmed_data[non_zero_mask]
                
                if len(trimmed_data) > 0:  # Only add non-empty scans
                    scans.append(trimmed_data)
            
            # Preview the stitching
            with self.output_area:
                clear_output(wait=True)
                
                if len(scans) == 0:
                    print("Error: No valid scan data available for stitching preview.")
                    return
                
                print(f"Generating stitching preview for group {group_idx+1}: {group['sample_name']} at {energy:.1f} eV")
                
                # Check if open beam normalization is available and get the intensity
                if (hasattr(self.processor, 'open_beam_data') and 
                    self.processor.open_beam_data is not None and
                    len(self.processor.open_beam_data) > 0):
                    
                    open_beam_intensity = self.processor.get_open_beam_intensity(energy)
                    if open_beam_intensity is not None:
                        # Check if open beam normalization is currently enabled
                        if getattr(self.processor, 'use_open_beam_normalization', False):
                            print(f"Open beam I₀({energy:.1f} eV) = {open_beam_intensity:.3e} (ENABLED)")
                        else:
                            print(f"Open beam I₀({energy:.1f} eV) = {open_beam_intensity:.3e} (available)")
                    else:
                        print(f"Open beam data available but could not interpolate intensity at {energy:.1f} eV")
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Left plot: Raw scans (trimmed and background corrected)
                for i, scan in enumerate(scans):
                    if len(scan) > 0:
                        ax1.plot(scan[:, 0], scan[:, 1], marker='o', linestyle='-', 
                                color=f'C{i}', label=f'Scan {i+1}', markersize=3)
                
                # Right plot: Stitching preview with scaling
                combined = np.empty((0, 2))
                
                for i, scan in enumerate(scans):
                    if len(scan) == 0:
                        continue
                        
                    if i == 0:
                        # First scan doesn't need scaling
                        ax2.plot(scan[:, 0], scan[:, 1], marker='o', linestyle='-', 
                                color=f'C{i}', label=f'Scan {i+1}', markersize=3)
                        combined = scan.copy()
                    else:
                        # Find overlapping region and calculate scaling factor
                        prev_max_angle = combined[:, 0].max()
                        curr_min_angle = scan[:, 0].min()
                        
                        # Store original data for plotting
                        orig_scan = scan.copy()
                        
                        # Calculate scaling factor if there's overlap
                        scale = 1.0
                        if curr_min_angle <= prev_max_angle:
                            # There's overlap - calculate scaling factor
                            overlap_mask_combined = combined[:, 0] >= curr_min_angle
                            overlap_mask_scan = scan[:, 0] <= prev_max_angle
                            
                            if np.any(overlap_mask_combined) and np.any(overlap_mask_scan):
                                overlap_combined = combined[overlap_mask_combined]
                                overlap_scan = scan[overlap_mask_scan]
                                
                                if len(overlap_combined) > 0 and len(overlap_scan) > 0:
                                    # Use median ratio for robust scaling
                                    ratios = []
                                    for angle in overlap_scan[:, 0]:
                                        # Find closest angle in combined data
                                        idx = np.argmin(np.abs(overlap_combined[:, 0] - angle))
                                        scan_idx = np.argmin(np.abs(overlap_scan[:, 0] - angle))
                                        if (overlap_combined[idx, 1] > 0 and 
                                            overlap_scan[scan_idx, 1] > 0):  # Avoid division by zero
                                            ratios.append(overlap_combined[idx, 1] / overlap_scan[scan_idx, 1])
                                    
                                    if len(ratios) > 0:
                                        scale = np.median(ratios)
                        
                        print(f"Scan {i+1}: scaling factor = {scale:.3f}")
                        
                        # Apply scaling
                        scan[:, 1] = scan[:, 1] * scale
                        
                        # Plot the unscaled data (dashed line)
                        ax2.plot(orig_scan[:, 0], orig_scan[:, 1], marker='', linestyle='--', 
                                color=f'C{i}', alpha=0.4, label=f'Scan {i+1} (Unscaled)')
                        
                        # Plot the scaled data
                        ax2.plot(scan[:, 0], scan[:, 1], marker='o', linestyle='-', 
                                color=f'C{i}', label=f'Scan {i+1} (Scaled)', markersize=3)
                        
                        # Concatenate with existing data
                        combined = np.concatenate((combined, scan))
                
                # Set y-scale based on checkbox
                if self.log_scale_checkbox.value:
                    ax1.set_yscale('log')
                    ax2.set_yscale('log')
                else:
                    ax1.set_yscale('linear')
                    ax2.set_yscale('linear')
                
                # Set labels and titles
                ax1.set_xlabel('Angle (degrees)')
                ax1.set_ylabel('Intensity')
                ax1.set_title(f'Raw Scans (Trimmed & Background Corrected) - {energy:.1f} eV')
                ax1.legend()
                ax1.grid(True, which="both", ls="--", alpha=0.3)
                
                ax2.set_xlabel('Angle (degrees)')
                ax2.set_ylabel('Intensity')
                ax2.set_title(f'Stitching Preview - {energy:.1f} eV')
                ax2.legend()
                ax2.grid(True, which="both", ls="--", alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
                print("Stitching preview complete!")
                print("Overlapping regions are highlighted in the right plot.")
                print("Dotted lines show unscaled data, solid lines show scaled data.")
                print("Background subtraction has been applied where specified.")
                print("To adjust the overlap regions, modify the trim values for each file.")
                print("Note: This is only a preview. Click 'Process Group' to apply the final processing.")
        
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                print(f"Error generating stitching preview: {str(e)}")
                import traceback
                traceback.print_exc()
                
                
    def _validate_smoothing_params(self):
        """Validate smoothing parameters and adjust if necessary"""
        window = self.smoothing_window.value
        order = self.smoothing_order.value
        
        # Window must be odd and >= 5
        if window < 5:
            self.smoothing_window.value = 5
            window = 5
            print("Window size must be at least 5. Adjusted to 5.")
        if window % 2 == 0:
            self.smoothing_window.value = window + 1
            window = window + 1
            print(f"Window size must be odd. Adjusted to {window}.")
            
        # Order must be less than window size
        if order >= window:
            self.smoothing_order.value = window - 1
            order = window - 1
            print(f"Polynomial order must be less than window size. Adjusted to {order}.")
            
        # Order must be at least 1
        if order < 1:
            self.smoothing_order.value = 1
            order = 1
            print("Polynomial order must be at least 1. Adjusted to 1.")
            
        return window, order
    
    def _on_preview_smooth(self, b):
        """Preview smoothing with current parameters"""
        # Get the current data
        if self.current_data is None or len(self.current_data) == 0:
            print("No data available for smoothing preview")
            return
            
        # Validate smoothing parameters
        window_size, polynomial_order = self._validate_smoothing_params()
        
        try:
            # Apply smoothing to the intensity data
            data = self.current_data
            
            # Apply trimming first
            group_str = self.group_select.value
            group_idx = int(group_str.split(':')[0]) - 1
            file_str = self.file_select.value
            file_idx = int(file_str.split(':')[0]) - 1
            trim_start, trim_end = self.current_trims[group_idx][file_idx]
            end_idx = len(data) + trim_end if trim_end < 0 else trim_end
            trimmed_data = data[trim_start:end_idx]
            
            # Apply background subtraction
            background_value = self.current_backgrounds[group_idx][file_idx]
            if background_value != 0.0:
                trimmed_data = self._apply_background_subtraction(trimmed_data, background_value)
            
            # Remove zeros if needed
            if len(trimmed_data) < window_size:
                print(f"Error: Not enough data points ({len(trimmed_data)}) for smoothing window ({window_size})")
                return
                
            # Apply Savitzky-Golay smoothing
            smoothed_intensity = savgol_filter(trimmed_data[:, 1], window_size, polynomial_order)
            
            # Create smoothed data array
            smoothed_data = trimmed_data.copy()
            smoothed_data[:, 1] = smoothed_intensity
            
            # Plot comparison
            with self.output_area:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot original trimmed data
                ax.plot(trimmed_data[:, 0], trimmed_data[:, 1], 'b-o', alpha=0.6, 
                       markersize=3, label='Original (Trimmed & BG Corrected)')
                
                # Plot smoothed data
                ax.plot(smoothed_data[:, 0], smoothed_data[:, 1], 'r-', linewidth=2, 
                       label=f'Smoothed (window={window_size}, order={polynomial_order})')
                
                # Set y-scale based on checkbox
                if self.log_scale_checkbox.value:
                    ax.set_yscale('log')
                else:
                    ax.set_yscale('linear')
                
                ax.set_xlabel('Angle (degrees)')
                ax.set_ylabel('Intensity')
                ax.set_title('Smoothing Preview')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.show()
                
                print(f"Smoothing preview with window size {window_size} and polynomial order {polynomial_order}")
                print("This shows how the data will look after smoothing is applied during processing.")
                print("Click 'Process Group' to apply smoothing to the final data.")
                print("Click any file in the dropdown to return to the trimming view.")
                
        except Exception as e:
            print(f"Error previewing smoothing: {str(e)}")
            print("Tip: Try adjusting the window size or polynomial order.")
    
    def _on_process_group(self, b):
        """Process the current group with current trim and background settings"""
        # Get current group index
        group_str = self.group_select.value
        group_idx = int(group_str.split(':')[0]) - 1
        
        # Get the current group
        group = deepcopy(self.scan_groups[group_idx])
        
        # Update the group with current trims
        group['trims'] = self.current_trims[group_idx]
        
        # Add background values to the group
        group['backgrounds'] = self.current_backgrounds[group_idx]
        
        # Create output directory if it doesn't exist
        output_dir = self.output_dir.value
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate smoothing parameters if smoothing is enabled
        if self.smooth_checkbox.value:
            window_size, polynomial_order = self._validate_smoothing_params()
        else:
            window_size, polynomial_order = None, 2
        
        # Process the group with current settings
        with self.output_area:
            clear_output(wait=True)
            print(f"Processing group {group_idx+1}: {group['sample_name']} at {group['energy']:.1f} eV")
            print(f"Using trims: {group['trims']}")
            print(f"Using backgrounds: {group['backgrounds']}")
            print(f"Output directory: {output_dir}")
            
            try:
                # Generate output filename
                output_filename = f"{group['sample_name']}_{group['energy']:.1f}eV.dat"
                
                # Check if the processor has the updated method
                if hasattr(self.processor, 'reduce_data_with_background'):
                    # Use the updated processor with background subtraction support
                    result = self.processor.process_scan_set(
                        scan_group=group,
                        output_filename=output_filename,
                        normalize=self.normalize_checkbox.value,
                        plot=True,
                        convert_to_photons=self.convert_photons_checkbox.value,
                        smooth_data=self.smooth_checkbox.value,
                        savgol_window=window_size,
                        savgol_order=polynomial_order,
                        remove_zeros=self.remove_zeros_checkbox.value,
                        estimate_thickness=False,  # Disabled thickness estimation
                        output_dir=output_dir
                    )
                    
                    print("\nProcessing completed successfully!")
                    print(f"Data saved to {os.path.join(output_dir, output_filename)}")
                    
                    # Show background subtraction summary
                    bg_applied = [bg for bg in group['backgrounds'] if bg != 0.0]
                    if bg_applied:
                        print(f"Background subtraction applied to {len(bg_applied)} files")
                    else:
                        print("No background subtraction was applied")
                        
                else:
                    # Fall back to manual processing demonstration
                    print("\nWARNING: Background subtraction not yet implemented in processor.")
                    print("To enable background subtraction, add the updated methods to RSoXRProcessor:")
                    print("1. Updated process_scan_set method")
                    print("2. reduce_data_with_background method") 
                    print("3. _plot_processed_data_with_background method")
                    print("\nFor now, showing what would be processed:")
                    
                    # Show the processing steps that would be applied
                    print("\nProcessing steps that would be applied:")
                    print("1. Load each scan file")
                    print("2. Apply trim values")
                    print("3. Apply background subtraction")
                    print("4. Remove zeros (if enabled)")
                    print("5. Apply smoothing (if enabled)")
                    print("6. Normalize and stitch scans together")
                    print("7. Save final result")
                    
                    # Show the processing settings
                    print(f"\nProcessing settings:")
                    print(f"  Normalize: {self.normalize_checkbox.value}")
                    print(f"  Remove zeros: {self.remove_zeros_checkbox.value}")
                    print(f"  Convert to photons: {self.convert_photons_checkbox.value}")
                    print(f"  Smooth data: {self.smooth_checkbox.value}")
                    if self.smooth_checkbox.value:
                        print(f"    Window size: {window_size}")
                        print(f"    Polynomial order: {polynomial_order}")
                    
                    # Show background subtraction summary
                    print(f"\nBackground subtraction summary:")
                    for i, (filename, bg_val) in enumerate(zip(group['files'], group['backgrounds'])):
                        if bg_val != 0.0:
                            print(f"  File {i+1} ({os.path.basename(filename)}): -{bg_val:.3f}")
                        else:
                            print(f"  File {i+1} ({os.path.basename(filename)}): No background subtraction")
                    
                    print(f"\nThe processed data would be saved as: {os.path.join(output_dir, output_filename)}")
                
            except Exception as e:
                print(f"Error processing group: {str(e)}")
                print("If background subtraction is enabled, ensure the processor has been updated with the new methods.")

    def _on_save_settings(self, b):
        """Save current trim and background settings to a CSV file"""
        try:
            # Create a list to collect trim and background data
            settings_data = []
            
            for group_idx, group in enumerate(self.scan_groups):
                for file_idx, filename in enumerate(group['files']):
                    original_trim = self.original_trims[group_idx][file_idx]
                    current_trim = self.current_trims[group_idx][file_idx]
                    background_value = self.current_backgrounds[group_idx][file_idx]
                    
                    # Add row to data
                    settings_data.append({
                        'Group': group_idx + 1,
                        'Sample': group['sample_name'],
                        'Energy (eV)': group['energy'],
                        'File Index': file_idx + 1,
                        'Filename': os.path.basename(filename),
                        'Original Trim Start': original_trim[0],
                        'Original Trim End': original_trim[1],
                        'Current Trim Start': current_trim[0],
                        'Current Trim End': current_trim[1],
                        'Background Value': background_value,
                        'Trim Modified': original_trim != current_trim,
                        'Background Applied': background_value != 0.0,
                        'Groups Source': self.groups_source
                    })
            
            # Create a DataFrame
            df = pd.DataFrame(settings_data)
            
            # Create output directory if it doesn't exist
            output_dir = self.output_dir.value
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to CSV
            output_file = os.path.join(output_dir, 'trim_and_background_settings.csv')
            df.to_csv(output_file, index=False)
            
            with self.output_area:
                clear_output(wait=True)
                print(f"Trim and background settings saved to {output_file}")
                
                # Also display a summary
                print("\nSummary of current configuration:")
                print(f"Groups source: {self.groups_source}")
                print(f"Total groups: {len(self.scan_groups)}")
                
                print("\nSummary of modifications:")
                modified_df = df[(df['Trim Modified']) | (df['Background Applied'])]
                if len(modified_df) > 0:
                    print("Files with modifications:")
                    for _, row in modified_df.iterrows():
                        changes = []
                        if row['Trim Modified']:
                            changes.append(f"Trim: [{row['Current Trim Start']}, {row['Current Trim End']}]")
                        if row['Background Applied']:
                            changes.append(f"Background: -{row['Background Value']:.3f}")
                        print(f"  Group {int(row['Group'])}, File {int(row['File Index'])}: {', '.join(changes)}")
                else:
                    print("No modifications were made from original values.")
                    
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                print(f"Error saving settings: {str(e)}")
    
    def _on_log_scale_change(self, change):
        """Handle log scale checkbox change"""
        # Redraw the plot with new scale
        self._plot_current_file()
        
    def add_open_beam_controls_to_widget(self):
        """
        Add open beam controls to the RSoXRTrimWidget
        This method should be called during widget initialization
        """
        import ipywidgets as widgets
        from IPython.display import clear_output
        
        # Open beam file selection
        self.open_beam_file_input = widgets.Text(
            value='',
            placeholder='Path to open beam file',
            description='Open Beam File:',
            disabled=False,
            layout=widgets.Layout(width='70%')
        )
        
        # Load open beam button
        self.load_open_beam_button = widgets.Button(
            description='Load Open Beam',
            disabled=False,
            button_style='info',
            tooltip='Load open beam intensity data',
            icon='upload'
        )
        
        # Open beam normalization checkbox
        self.use_open_beam_checkbox = widgets.Checkbox(
            value=False,
            description='Use Open Beam Normalization',
            disabled=True,  # Disabled until open beam data is loaded
            tooltip='Use open beam intensity for normalization instead of standard normalization'
        )
        
        # Open beam status display
        self.open_beam_status = widgets.HTML(
            value="<b>Open Beam Status:</b> No data loaded",
            description='',
            layout=widgets.Layout(width='100%')
        )
        
        # Set up event handlers
        self.load_open_beam_button.on_click(self._on_load_open_beam)
        self.use_open_beam_checkbox.observe(self._on_open_beam_checkbox_change, names='value')
        
        # Create the open beam controls widget
        self.open_beam_controls = widgets.VBox([
            widgets.HTML("<h4>Open Beam Normalization</h4>"),
            widgets.HBox([self.open_beam_file_input, self.load_open_beam_button]),
            self.use_open_beam_checkbox,
            self.open_beam_status
        ])
        
        return self.open_beam_controls


    def _on_load_open_beam(self, b):
        """Handle open beam file loading in widget"""
        file_path = self.open_beam_file_input.value.strip()
        
        if not file_path:
            with self.output_area:
                print("Please enter a path to the open beam file.")
            return
        
        if not os.path.exists(file_path):
            with self.output_area:
                print(f"Error: File not found: {file_path}")
            return
        
        with self.output_area:
            clear_output(wait=True)
            print(f"Loading open beam data from: {file_path}")
            
            # Load the open beam file
            success = self.processor.load_open_beam_file(file_path)
            
            if success:
                # Update status and enable checkbox
                energy_range = f"{self.processor.open_beam_data['energy'].min():.1f} - {self.processor.open_beam_data['energy'].max():.1f} eV"
                self.open_beam_status.value = f"<b>Open Beam Status:</b> ✓ Loaded ({len(self.processor.open_beam_data)} points, {energy_range})"
                self.use_open_beam_checkbox.disabled = False
                
                print("Open beam data loaded successfully!")
                print("You can now enable 'Use Open Beam Normalization' in the processing options.")
            else:
                self.open_beam_status.value = "<b>Open Beam Status:</b> ✗ Failed to load"
                self.use_open_beam_checkbox.disabled = True
                self.use_open_beam_checkbox.value = False


    def _on_open_beam_checkbox_change(self, change):
        """Handle open beam normalization checkbox change in widget"""
        use_open_beam = change['new']
        
        # Update the processor setting
        success = self.processor.set_open_beam_normalization(use_open_beam)
        
        if not success:
            # Revert checkbox if setting failed
            self.use_open_beam_checkbox.value = False
        
        with self.output_area:
            if success:
                norm_type = "open beam" if use_open_beam else "standard"
                print(f"Normalization mode set to: {norm_type}")
            else:
                print("Failed to set open beam normalization. Please load open beam data first.")

    def _load_open_beam_on_init(self, open_beam_path):
        """
        Load open beam data during widget initialization
        
        Parameters:
        -----------
        open_beam_path : str
            Path to the open beam file
        """
        try:
            print(f"Loading open beam data from initialization parameter: {open_beam_path}")
            
            # Load the open beam file using the processor
            success = self.processor.load_open_beam_file(open_beam_path)
            
            if success:
                # Update the open beam controls UI
                self.open_beam_file_input.value = open_beam_path
                
                energy_range = f"{self.processor.open_beam_data['energy'].min():.1f} - {self.processor.open_beam_data['energy'].max():.1f} eV"
                self.open_beam_status.value = f"<b>Open Beam Status:</b> ✓ Loaded ({len(self.processor.open_beam_data)} points, {energy_range})"
                
                self.use_open_beam_checkbox.disabled = False
                
                print(f"✓ Open beam data loaded successfully during initialization!")
                print(f"  Energy range: {energy_range}")
                print(f"  Data points: {len(self.processor.open_beam_data)}")
                
            else:
                print(f"✗ Failed to load open beam data from: {open_beam_path}")
                
        except Exception as e:
            print(f"Error loading open beam data during initialization: {str(e)}")
        
    def display(self):
        """Display the widget"""
        display(self.main_container)
        print(f"RSoXR Trim Widget initialized with default output directory: {self.default_output_dir}")
        print(f"Default trim values set to use full range of each data section (0, -1)")
        print(f"Loaded {len(self.scan_groups)} scan groups")