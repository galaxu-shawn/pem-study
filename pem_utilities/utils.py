"""
Utility Functions for PerceiveEM

This module provides various utility functions for electromagnetic simulation and visualization,
including color mapping, unit conversion, mesh creation, image processing, and actor/traffic
generation for simulation scenarios.

Created on Fri Jan 26 15:00:00 2024

@author: asligar
"""

import os
import numpy as np
from matplotlib import colormaps as cm
import cv2
import copy
from PIL import Image
from scipy.signal import savgol_filter
import pyvista as pv
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from pem_utilities.router import Pick_Path
from pem_utilities.rotation import euler_to_rot
from pathlib import Path


def color_mapper(color='black', random=False, cmap='gist_rainbow'):
    """
    Map a color name to its RGB representation or generate random colors.
    
    This function provides a standardized way to convert color names to RGB tuples,
    with support for random color generation using matplotlib colormaps.

    Parameters
    ----------
    color : str or tuple or list, optional
        The name of the color to be mapped, or RGB values as tuple/list.
        Defaults to 'black'.
    random : bool, optional
        If True, generates a random color from the specified colormap.
        Defaults to False.
    cmap : str, optional
        The matplotlib colormap name to use for random color generation.
        Defaults to 'gist_rainbow'.

    Returns
    -------
    tuple
        RGB color values as a tuple of floats in range [0, 1].
        
    Examples
    --------
    >>> color_mapper('red')
    (1.0, 0.0, 0.0)
    >>> color_mapper(random=True, cmap='viridis')
    (0.267004, 0.004874, 0.329415)  # Random values
    """
    # Handle different input types
    if isinstance(color, str):
        cmap = cm.get_cmap(cmap)
    elif isinstance(color, tuple):
        return color
    elif isinstance(color, list):
        color = tuple(color)
        return color

    if random:
        # Generate random color from colormap
        color = cmap(np.random.rand())
        return color[:3]
    else:
        color = color.lower()

        # Predefined color dictionary
        colors = {
            'red': (1., 0., 0.), 'orange': (1., .5, 0.), 'yellow': (1., 1., 0.),
            'green': (0., 1., 0.), 'blue': (0., 0., 1.), 'inidgo': (0.8, .6, 0.),
            'violet': (0.6, .6, 0.), 'black': (0., 0., 0.), 'white': (1., 1., 1.),
            'grey': (.5, .5, .5), 'lightgrey': (.76, .76, .76), 'brown': (.5, .5, 0),
            'purple': (.5, 0., .5)
        }
        
        if color in colors.keys():
            return colors[color]
        else:
            print('requested color ' + color + ' does not exists')
            return colors['black']


def convert_units(value, old_units, new_units):
    """
    Convert values between common engineering units used in HFSS and electromagnetic simulations.
    
    This function provides unit conversion for length measurements commonly used in
    electromagnetic simulation software like HFSS.

    Parameters
    ----------
    value : float
        The numeric value to be converted.
    old_units : str
        The current unit of the input value. Supported units:
        'nm', 'um', 'mm', 'meter', 'cm', 'ft', 'in', 'mil', 'uin'
    new_units : str
        The target unit for conversion. Uses same units as old_units.

    Returns
    -------
    float
        The value converted to the new unit system.
        
    Examples
    --------
    >>> convert_units(1000, 'mm', 'meter')
    1.0
    >>> convert_units(1, 'in', 'mm')
    25.4
    """
    # Unit conversion factors to meters
    unit_conv = {
        "nm": .000000001,    # nanometers
        "um": .000001,       # micrometers
        "mm": .001,          # millimeters
        "meter": 1.0,        # meters (base unit)
        "cm": .01,           # centimeters
        "ft": .3048,         # feet
        "in": .0254,         # inches
        "mil": .0000254,     # mils (thousandths of an inch)
        "uin": .0000000254   # microinches
    }

    sf = 1.0  # scale factor

    # Get conversion factors for both units
    base_units = None
    target_units = None
    if old_units.lower() in unit_conv:
        base_units = unit_conv[old_units.lower()]
    if new_units.lower() in unit_conv:
        target_units = unit_conv[new_units.lower()]

    # Calculate scale factor if both units are valid
    if base_units is not None and target_units is not None:
        sf = base_units / target_units

    # Apply conversion if units are different
    if old_units != new_units:
        nu_value = value * sf
    else:
        nu_value = value

    return nu_value


def create_position_array_from_yaw_pos(time_stamps, init_pos=np.zeros(3), yaw=0, init_vel=0, end_vel=None):
    """
    Generate a trajectory array based on yaw angle and velocity parameters.
    
    Creates a linear trajectory in 3D space based on initial position, yaw angle,
    and velocity parameters. Useful for simulating moving objects in electromagnetic
    scenarios.

    Parameters
    ----------
    time_stamps : array_like
        Array of time values for the trajectory.
    init_pos : array_like, optional
        Initial position [x, y, z] in 3D space. Defaults to origin [0, 0, 0].
    yaw : float, optional
        Yaw angle in degrees defining the direction of motion. Defaults to 0.
    init_vel : float, optional
        Initial velocity magnitude. Defaults to 0.
    end_vel : float, optional
        Final velocity magnitude. If None, uses init_vel. Defaults to None.

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_timestamps, 3) containing [x, y, z] positions
        for each time stamp.
        
    Examples
    --------
    >>> times = np.linspace(0, 10, 11)
    >>> positions = create_position_array_from_yaw_pos(times, yaw=45, init_vel=5)
    >>> positions.shape
    (11, 3)
    """
    # Extract initial position components
    init_x_pos = init_pos[0]
    init_y_pos = init_pos[1]
    init_z_pos = init_pos[2]

    # Set end velocity if not provided
    if not end_vel:
        end_vel = init_vel

    num_time_stamps = len(time_stamps)
    
    # Calculate x positions using average velocity and yaw angle
    xpos = np.linspace(
        init_x_pos,
        init_x_pos + .5 * (np.cos(np.deg2rad(yaw)) * init_vel + np.cos(np.deg2rad(yaw)) * end_vel) * time_stamps[-1], 
        num=num_time_stamps
    )
    
    # Calculate y positions using average velocity and yaw angle
    ypos = np.linspace(
        init_y_pos,
        init_y_pos + .5 * (np.sin(np.deg2rad(yaw)) * init_vel + np.sin(np.deg2rad(yaw) * end_vel)) * time_stamps[-1], 
        num=num_time_stamps
    )
    
    # Z position remains constant
    zpos = np.zeros(num_time_stamps) + init_z_pos
    
    # Stack coordinates into position array [time_stamps][xyz]
    pos_xyz = np.stack((xpos, ypos, zpos), axis=1)
    return pos_xyz


def create_mesh_from_image(data,
                           center_idx_in_data=None,
                           center_location_in_display=[0, 0],
                           x_size=10,
                           y_size=10,
                           rPixels_extents=None,
                           dPixels_extents=None,
                           z_offset=0,
                           all_positions=None):
    """
    Convert 2D image data into a PyVista mesh for 3D visualization.
    
    This function creates a 3D surface mesh from 2D array data (such as radar images
    or field distributions) for visualization in PyVista. The data is converted to
    logarithmic scale and mapped onto a spatial grid.

    Parameters
    ----------
    data : numpy.ndarray
        2D array containing the image data to be converted to mesh.
    center_idx_in_data : list or None, optional
        [doppler_center, range_center] indices in the data array. If None,
        automatically calculated as the center of the array.
    center_location_in_display : list, optional
        [x, y] coordinates for the center of the mesh in display space.
        Defaults to [0, 0].
    x_size : float, optional
        Physical size of the mesh in the x-direction. Defaults to 10.
    y_size : float, optional
        Physical size of the mesh in the y-direction. Defaults to 10.
    rPixels_extents : int or None, optional
        Number of range pixels to include. If None, uses full range dimension.
    dPixels_extents : int or None, optional
        Number of Doppler pixels to include. If None, uses full Doppler dimension.
    z_offset : float, optional
        Vertical offset for the mesh. Defaults to 0.
    all_positions : array_like or None, optional
        Predefined positions for mesh points. Currently unused.

    Returns
    -------
    pyvista.PolyData
        A PyVista surface mesh with 'image_mag' field containing the processed data.
        
    Notes
    -----
    The function applies 20*log10 transformation to the data and clips values
    between -190 and -130 dB for visualization purposes.
    """
    # Set default pixel extents if not provided
    if rPixels_extents is None:
        rPixels_extents = data.shape[1]
    if dPixels_extents is None:
        dPixels_extents = data.shape[0]

    # Calculate center indices if not provided
    if center_idx_in_data is None:
        # Get center of image, and correct extents if pixels are even
        center_idx_d = 0
        if data.shape[0] % 2 == 0:  # even number of doppler bins
            center_idx_d = int(data.shape[0] / 2)
            if dPixels_extents >= data.shape[0]:
                dPixels_extents = dPixels_extents - 1
        else:  # odd number of doppler bins
            center_idx_d = int(data.shape[0] / 2) + 1
            
        if data.shape[1] % 2 == 0:  # even number of range bins
            center_idx_r = int(data.shape[1] / 2)
            if rPixels_extents >= data.shape[1]:
                rPixels_extents = rPixels_extents - 1
        else:  # odd number of range bins
            center_idx_r = int(data.shape[1] / 2) + 1

        center_idx_in_data = [center_idx_d, center_idx_r]

    # Calculate slice indices for data extraction
    doppler_start_idx = center_idx_in_data[0] - int(dPixels_extents / 2)
    doppler_stop_idx = center_idx_in_data[0] + int(dPixels_extents / 2)
    range_start_idx = center_idx_in_data[1] - int(rPixels_extents / 2)
    range_stop_idx = center_idx_in_data[1] + int(rPixels_extents / 2)

    # Limit slice indices to array bounds
    if doppler_start_idx < 0:
        doppler_start_idx = 0
    if range_start_idx < 0:
        range_start_idx = 0
    if doppler_stop_idx >= data.shape[0]:
        doppler_stop_idx = data.shape[0] - 1
    if range_stop_idx >= data.shape[1]:
        range_stop_idx = data.shape[1] - 1

    # Extract and process data slice
    data_for_view = data[doppler_start_idx:doppler_stop_idx, range_start_idx:range_stop_idx]
    data_for_view = (data)  # Use full data for now
    
    # Convert to dB scale with magnitude limiting
    mag = np.fmin(np.fmax(20 * np.log10(np.fmax(np.abs(data_for_view.T), 1.e-30)), -190), -130)
    mag = np.ndarray.flatten(mag, order='C')

    # Create spatial grid for plotting
    x = np.linspace(-x_size / 2, x_size / 2, num=data_for_view.shape[0]) + center_location_in_display[0]
    y = np.linspace(-y_size / 2, y_size / 2, num=data_for_view.shape[1]) + center_location_in_display[1]
    xv, yv = np.meshgrid(x, y)
    xy = np.array((xv.ravel(), yv.ravel())).T
    z = z_offset * np.ones((len(xy), 1))
    xyz = np.hstack((xy, z))
    
    # Create PyVista mesh
    fields_mesh = pv.PolyData(xyz)
    fields_mesh['image_mag'] = mag
    fields_mesh_surface = fields_mesh.delaunay_2d()
    
    return fields_mesh_surface


def create_mesh_from_image2(data, z_offset=0, all_positions=None, limits=[None, None], 
                           function='dB', create_surface=True):
    """
    Create a PyVista mesh from data with flexible mathematical transformations.
    
    This is an enhanced version of create_mesh_from_image that provides more
    control over data processing and supports predefined position arrays.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array to be converted to mesh.
    z_offset : float, optional
        Vertical offset for mesh positioning. Defaults to 0.
    all_positions : numpy.ndarray or None, optional
        Predefined [x, y] positions for mesh points. If None, uses grid positions.
    limits : list, optional
        [min_value, max_value] for data clipping. None values use data extremes.
        Defaults to [None, None].
    function : str, optional
        Mathematical transformation to apply to data:
        'dB' or 'db20': 20*log10(abs(data))
        'db10': 10*log10(abs(data))
        'real': real part of data
        'imag': imaginary part of data
        'phase': phase angle of data
        Other: absolute value
        Defaults to 'dB'.
    create_surface : bool, optional
        Whether to create a triangulated surface using Delaunay triangulation.
        Defaults to True.

    Returns
    -------
    pyvista.PolyData
        PyVista mesh with 'image_mag' field containing processed data.
        
    Notes
    -----
    This function handles complex data types and provides various mathematical
    transformations commonly used in electromagnetic field visualization.
    """
    # Check for non-finite values in input data
    if not np.isfinite(data).all():
        pass  # Could add warning or handling here
    
    # Apply mathematical transformation based on function parameter
    if function.lower() == 'db':
        data = 20 * np.log10(np.fmax(np.abs(data), 1.e-30))
    elif function.lower() == 'db10':
        data = 10 * np.log10(np.fmax(np.abs(data), 1.e-30))
    elif function.lower() == 'db20':
        data = 20 * np.log10(np.fmax(np.abs(data), 1.e-30))
    elif function.lower() == 'real':
        data = np.real(data)
    elif function.lower() == 'imag':
        data = np.fmax(np.imag(data), 1.e-30) 
    elif function.lower() == 'phase':
        data = np.angle(data)
    else:
        data = np.fmax(np.abs(data), 1.e-30)

    # Set limits based on data if not provided
    if limits[0] is None:
        limits[0] = np.min(data)
    if limits[1] is None:
        limits[1] = np.max(data)

    # Check for non-finite values after transformation
    if not np.isfinite(data).all():
        pass  # Could add warning or handling here
    
    # Flatten data for mesh creation
    data = np.ndarray.flatten(data, order='C')

    # Create 3D positions by adding z-offset
    z = z_offset * np.ones((len(all_positions), 1))
    xyz = np.hstack((all_positions, z))
    
    # Create PyVista mesh
    fields_mesh = pv.PolyData(xyz)
    fields_mesh['image_mag'] = data
    
    # Optionally create triangulated surface
    if create_surface:
        fields_mesh = fields_mesh.delaunay_2d()
        
    return fields_mesh


def apply_math_function(values, function='dB'):
    """
    Apply mathematical transformations to complex or real-valued data arrays.
    
    This utility function provides standardized mathematical operations commonly
    used in electromagnetic field processing and visualization.

    Parameters
    ----------
    values : numpy.ndarray
        Input array of values (can be complex or real).
    function : str, optional
        Mathematical transformation to apply:
        'dB': 20*log10(abs(values))
        'abs': absolute value
        'real': real part
        'imag': imaginary part  
        'phase': phase angle
        Other: returns values unchanged
        Defaults to 'dB'.

    Returns
    -------
    numpy.ndarray
        Transformed values with specified mathematical function applied.
        
    Examples
    --------
    >>> data = np.array([1+1j, 2+2j, 3+3j])
    >>> apply_math_function(data, 'dB')
    array([3.0103, 9.0309, 13.0103])
    """
    new_values = None
    
    if function.lower() == 'db':
        new_values = 20 * np.log10(np.fmax(np.abs(values), 1.e-15))
    elif function.lower() == 'abs':
        new_values = np.fmax(np.abs(values), 1.e-15)
    elif function.lower() == 'real':
        new_values = np.fmax(np.real(values), 1.e-15)
    elif function.lower() == 'imag':
        new_values = np.fmax(np.imag(values), 1.e-15)
    elif function.lower() == 'phase':
        new_values = np.fmax(np.angle(values), 1.e-15)
    else:
        new_values = values
        
    return new_values


def generate_rgb_from_array(data, function='dB',
                            plot_min=None, plot_max=None,
                            plot_dynamic_range=None,
                            resize_window=None,
                            colormap='jet',
                            smooth_image=False,
                            show_image=False,
                            return_rgba=False):
    """
    Convert numerical data array to RGB image for visualization and display.
    
    This function transforms numerical data (such as radar images or field
    distributions) into RGB format suitable for display or saving as images.
    Supports various mathematical transformations and colormap options.

    Parameters
    ----------
    data : array_like
        Input data array to be converted to RGB. Can be list or numpy array.
        If dimension > 2, only the last 2 dimensions are used.
    function : str, optional
        Mathematical transformation to apply (see apply_math_function).
        Defaults to 'dB'.
    plot_min : float or None, optional
        Minimum value for color scaling. If None, uses data minimum.
    plot_max : float or None, optional
        Maximum value for color scaling. If None, uses data maximum.
    plot_dynamic_range : float or None, optional
        Dynamic range in dB. If provided, plot_min = plot_max - dynamic_range.
    resize_window : tuple or None, optional
        (width, height) for display window resizing when show_image=True.
    colormap : str or matplotlib.colors.Colormap, optional
        Colormap for visualization. Defaults to 'jet'.
    smooth_image : bool, optional
        Whether to apply Savitzky-Golay smoothing filter. Defaults to False.
    show_image : bool, optional
        Whether to display the image using OpenCV. Defaults to False.

    Returns
    -------
    numpy.ndarray
        RGB image array of shape (height, width, 3) with values in range [0, 255].
        
    Notes
    -----
    For data with dimension > 2, only the last 2 dimensions are used for display.
    The first dimensions are assumed to be time, transmitter, or receiver indices.
    """
    # Handle colormap input
    if isinstance(colormap, str):
        try:
            colormap = cm.get_cmap(colormap)
        except:
            print(f'WARNING: {colormap} colormap does not exist')
            colormap = cm.get_cmap('magma')

    # Convert list to numpy array if needed
    if isinstance(data, list):
        data = np.array(data)

    # Validate and process data dimensions
    if data.ndim < 2:
        print('The data array must be at least 2D')
        return False
    elif data.ndim > 2:
        # Extract only the last 2 dimensions for display
        # First dims may be time step, Tx number, Rx number, etc.
        slc = [0] * (data.ndim - 2)  # Select first index for higher dimensions
        slc += [slice(None), slice(None)]  # Keep all indices for last 2 dims
        data = data[slc]
    
    # Apply mathematical transformation
    data = apply_math_function(data, function)
    data_min = np.min(data)
    data_max = np.max(data)

    # Set plot limits
    if plot_min is None:
        plot_min = np.min(data)
    elif plot_min > data_max:
        print(f'plot_min is greater than the data max value of {data_max}, resetting to data min')
        plot_min = np.min(data)

    # Floor the minimum plottable value
    if plot_min < -300:
        plot_min = -300

    if plot_max is None:
        plot_max = np.max(data)
        if plot_dynamic_range is not None:
            plot_min = plot_max - plot_dynamic_range
    elif plot_max < data_min:
        print(f'plot_max is less than data min, resetting to data max')
        plot_max = np.max(data)
    elif plot_dynamic_range is not None:
        plot_min = plot_max - plot_dynamic_range

    # Handle empty data case
    if data is None:
        data_in = np.ones((100, 100))
    else:
        # Clip data to plot range and transpose for proper orientation
        data_in = np.fmin(np.fmax(data.T, plot_min), plot_max)

    # Normalize data to [0, 1] range for colormap
    if np.ptp(data_in) == 0:
        plot_data = np.zeros(data_in.shape)
    else:
        plot_data = (data_in - np.min(data_in)) / np.ptp(data_in)
    plot_data = np.flipud(plot_data)  # Flip for proper display orientation
    
    # Apply smoothing if requested
    if smooth_image:
        try:
            plot_data = savgol_filter(plot_data, 25, 3)
        except:
            print('smoothing of radar output image failed')
    
    # Apply colormap and convert to RGB
    img_plot_data_rgba = np.uint8(colormap(plot_data) * 255)
    img_plot_data_rgb = img_plot_data_rgba[:, :, :3]  # Remove alpha channel

    # Convert RGB to BGR for OpenCV compatibility
    img_plot_data_bgr = np.moveaxis(
        np.stack([img_plot_data_rgb[:, :, 2], img_plot_data_rgb[:, :, 1], img_plot_data_rgb[:, :, 0]]), 0, -1)
    
    # Create PIL image and convert to frame
    img_plot = Image.fromarray(img_plot_data_bgr)
    frame = np.array(img_plot).astype('uint8')
    
    # Display image if requested
    if show_image:
        cv2.waitKey(5)  # Small delay to prevent freezing
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        if resize_window is not None:
            cv2.resizeWindow("output", resize_window[0], resize_window[1])
        cv2.imshow("output", frame)

    if return_rgba:
        return img_plot_data_rgba
    else:
        return img_plot_data_rgb


def generate_rgba_from_array(data, function='dB',
                            plot_min=None, plot_max=None,
                            plot_dynamic_range=None,
                            resize_window=None,
                            colormap='jet',
                            smooth_image=False,
                            show_image=False):
    image_rgba =generate_rgb_from_array(data, function=function,
                                        plot_min=plot_min, plot_max=plot_max,
                                        plot_dynamic_range=plot_dynamic_range,
                                        resize_window=resize_window,
                                        colormap=colormap,
                                        smooth_image=smooth_image,
                                        show_image=show_image,
                                        return_rgba=True)
    return image_rgba

def create_crowd(all_actors, num_peds=2, xyz=[0, 0, 0], radius=10, name='ped_crowd'):
    """
    Create a crowd of pedestrian actors randomly distributed in a circular area.
    
    This function generates multiple pedestrian actors from the CMU motion capture
    database, randomly positioning them within a specified radius and applying
    random orientations and scales.

    Parameters
    ----------
    all_actors : object
        Actor manager object that handles adding and managing simulation actors.
    num_peds : int, optional
        Number of pedestrian actors to create. Defaults to 2.
    xyz : list, optional
        [x, y, z] center coordinates for the crowd. Defaults to [0, 0, 0].
    radius : float, optional
        Radius in meters for random pedestrian placement. Defaults to 10.
    name : str, optional
        Base name for the pedestrian actors. Defaults to 'ped_crowd'.

    Returns
    -------
    list
        List of actor names that were created.
        
    Notes
    -----
    Pedestrians are randomly selected from available CMU database models and
    placed with random positions, orientations, and scale factors between 0.8-1.1.
    """
    xyz = np.array(xyz)
    all_names = []
    
    # Dynamically resolve the path to the model directory
    script_path = Path(__file__).resolve()
    model_path = os.path.join(script_path.parent.parent, 'models')
    
    for each in range(num_peds):
        # Random scale factor for pedestrian size variation
        scale = np.random.uniform(.8, 1.1, 1)[0]

        # Available pedestrian models from CMU database
        possible_models = ['77_01', '77_02', '77_03', '77_04', '82_05', '138_12', '140_06', '140_07']
        model_to_use = np.random.choice(possible_models)

        # Construct full model path
        model_path_full = os.path.join(model_path, 'CMU_Database', model_to_use + '.daez')
        
        # Add actor to simulation
        actor_pedestrian_name = all_actors.add_actor(
            name=name,
            filename=model_path_full,
            target_ray_spacing=0.1,
            scale_mesh=scale
        )
        all_names.append(actor_pedestrian_name)
        
        # Random position within specified radius
        x_rand = np.random.uniform(xyz[0] - radius, xyz[0] + radius, 1)[0]
        y_rand = np.random.uniform(xyz[1] - radius, xyz[1] + radius, 1)[0]
        all_actors.actors[actor_pedestrian_name].coord_sys.pos = [x_rand, y_rand, xyz[2]]
        
        # Random orientation
        rotation_phi = np.random.uniform(0, 360, 1)[0]
        rot = euler_to_rot(phi=rotation_phi, order='zyz', deg=True)
        all_actors.actors[actor_pedestrian_name].coord_sys.rot = rot
        all_actors.actors[actor_pedestrian_name].coord_sys.update()

    return all_names


def create_pack_of_peds(all_actors, peds_dict={}, num_peds=1, pre_picked=None, 
                       surface_meshes=None, name='ped_pack'):
    """
    Create a pack of pedestrians that follow predefined paths with slight variations.
    
    This function creates pedestrian actors that move along similar paths with
    small random offsets, simulating a group of people walking together.

    Parameters
    ----------
    all_actors : object
        Actor manager object for handling simulation actors.
    peds_dict : dict, optional
        Dictionary to store pedestrian-path associations. Defaults to {}.
    num_peds : int, optional
        Number of pedestrians to create in the pack. Defaults to 1.
    pre_picked : array_like or None, optional
        Predefined path waypoints as array of [x, y, z] coordinates.
    surface_meshes : list or None, optional
        List of surface meshes for path snapping and collision detection.
    name : str, optional
        Base name for pedestrian actors. Defaults to 'ped_pack'.

    Returns
    -------
    dict
        Updated peds_dict with new pedestrian-path associations.
        
    Notes
    -----
    Each pedestrian gets a slightly offset path from the original with random
    speed variations between 0.9-1.5 m/s and scale factors between 0.8-1.1.
    """
    # Dynamically resolve the path to the model directory
    script_path = Path(__file__).resolve()
    model_path = os.path.join(script_path.parent.parent, 'models')
    
    print('Creating pedestrians (moving pack)')
    pre_picked_orig = np.array(pre_picked)
    pre_picked = np.array(pre_picked)
    
    for each in range(num_peds):
        # Create path manager for this pedestrian
        path_picked_ped = Pick_Path()
        pre_picked = copy.copy(pre_picked_orig)
        
        # Add small random offsets to create path variation
        x_rand = np.random.uniform(-each/2, each/2, 1)[0]
        y_rand = np.random.uniform(-each/2, each/2, 1)[0]
        pre_picked[:, 0] = pre_picked_orig[:, 0] + x_rand
        pre_picked[:, 1] = pre_picked_orig[:, 1] + y_rand
        
        # Random speed variation
        speed = np.random.uniform(.9, 1.5, 1)[0]
        
        # Create custom path with surface snapping
        path_picked_ped.custom_path(
            mesh_list=surface_meshes, 
            pre_picked=pre_picked, 
            speed=speed,
            upsample_selection=101,
            z_offset=0, 
            snap_to_surface=True
        )
        
        # Random scale for pedestrian size
        scale = np.random.uniform(.8, 1.1, 1)[0]
        
        # Add pedestrian actor with walking animation
        actor_pedestrian_name = all_actors.add_actor(
            name=name,
            filename=os.path.join(model_path, 'Walking_speed50_armspace50.dae'),
            target_ray_spacing=0.1,
            scale_mesh=scale
        )
        
        # Disable linear velocity updates (using custom path instead)
        all_actors.actors[actor_pedestrian_name].use_linear_velocity_equation_update = False
        
        # Store pedestrian-path association
        peds_dict[actor_pedestrian_name] = path_picked_ped
    
    return peds_dict


def create_traffic(all_actors, traffic_dict={}, num_cars=1, speed=10,
                   vehicle_separation=10, add_randomness=False,
                   pre_picked=None, surface_meshes=None, name='traffic'):
    """
    Create a line of vehicle traffic following a predefined route.
    
    This function generates vehicle actors that follow a path with specified
    spacing and speed parameters, useful for creating realistic traffic scenarios
    in electromagnetic simulations.

    Parameters
    ----------
    all_actors : object
        Actor manager object for handling simulation actors.
    traffic_dict : dict, optional
        Dictionary to store vehicle-path associations. Defaults to {}.
    num_cars : int, optional
        Number of vehicles to create. Defaults to 1.
    speed : float, optional
        Base speed for vehicles in m/s. Defaults to 10.
    vehicle_separation : float, optional
        Distance between vehicles in meters. Defaults to 10.
    add_randomness : bool, optional
        Whether to add random variations to positioning and speed. Defaults to False.
    pre_picked : array_like or None, optional
        Predefined path waypoints as array of [x, y, z] coordinates.
    surface_meshes : list or None, optional
        List of surface meshes for path snapping.
    name : str, optional
        Base name for vehicle actors. Defaults to 'traffic'.

    Returns
    -------
    dict
        Updated traffic_dict with new vehicle-path associations.
        
    Notes
    -----
    Vehicles are spaced along the path direction with the first vehicle at the
    original path and subsequent vehicles offset backwards. Random variations
    include position offsets and speed changes when add_randomness=True.
    """
    # Dynamically resolve the path to the model directory
    script_path = Path(__file__).resolve()
    model_path = os.path.join(script_path.parent.parent, 'models')
    
    print('Creating traffic')
    pre_picked = np.array(pre_picked)
    
    # Calculate path direction vector for vehicle spacing
    vector = pre_picked[1] - pre_picked[0]
    norm = -1 * vector / np.linalg.norm(vector, axis=0)  # Negative for backward spacing
    
    pre_picked_orig = copy.copy(pre_picked)
    original_vehicle_separation = vehicle_separation
    original_speed = speed
    
    for each in range(num_cars):
        previous_speed = original_speed
        path_picked_veh = Pick_Path()
        
        if each > 0:
            # Create extended path for vehicles behind the leader
            new_picked = np.zeros((pre_picked_orig.shape[0] + 1, pre_picked_orig.shape[1]))
            
            if add_randomness:
                # Add random variations to separation and speed
                vehicle_separation = np.random.uniform(
                    original_vehicle_separation * .9, 
                    original_vehicle_separation * 1.1, 1
                )[0]
                # Vehicles can only be slower than the car in front
                speed = np.random.uniform(original_speed * .975, previous_speed, 1)[0]
                previous_speed = speed
                
                # Add random lateral offset
                new_first_point = norm * each * vehicle_separation
                rand_x = np.random.uniform(-0.5, 0.5, 1)[0]
                rand_y = np.random.uniform(-0.5, 0.5, 1)[0]
                new_picked[0] = pre_picked_orig[0] + new_first_point + rand_x
                new_picked[1:] = pre_picked_orig + rand_y
            else:
                # Deterministic spacing
                vehicle_separation = original_vehicle_separation
                speed = original_speed
                new_first_point = norm * each * vehicle_separation
                new_picked[0] = pre_picked_orig[0] + new_first_point
                new_picked[1:] = pre_picked_orig
        else:
            # First vehicle uses original path
            new_picked = pre_picked_orig
        
        # Create custom path for this vehicle
        path_picked_veh.custom_path(
            mesh_list=surface_meshes, 
            pre_picked=new_picked, 
            speed=speed,
            upsample_selection=101,
            z_offset=0, 
            snap_to_surface=True
        )

        # Add vehicle actor (Audi A1 model)
        actor_vehicle_name = all_actors.add_actor(
            name=name,
            filename=os.path.join(model_path, 'Audi_A1_2010/Audi_A1_2010.json')
        )
        
        # Disable linear velocity updates (using custom path instead)
        all_actors.actors[actor_vehicle_name].use_linear_velocity_equation_update = False

        # Store vehicle-path association
        traffic_dict[actor_vehicle_name] = path_picked_veh

    return traffic_dict


def create_traffic_grid(all_actors, traffic_dict={}, num_cars_per_column=1,
                        num_car_to_driver_side=1, num_cars_to_passenger_side=1,
                        speed=10, vehicle_separation=10, horizontal_seperation=3.5,
                        add_randomness=False, pre_picked=None, surface_meshes=None,
                        name='traffic'):
    """
    Create a grid of vehicle traffic with multiple lanes following parallel paths.
    
    This function creates a more complex traffic scenario with vehicles in multiple
    lanes (driver side, main lane, passenger side) to simulate realistic multi-lane
    traffic conditions.

    Parameters
    ----------
    all_actors : object
        Actor manager object for handling simulation actors.
    traffic_dict : dict, optional
        Dictionary to store vehicle-path associations. Defaults to {}.
    num_cars_per_column : int, optional
        Number of vehicles in the main column. Defaults to 1.
    num_car_to_driver_side : int, optional
        Number of vehicles in the driver-side lane. Defaults to 1.
    num_cars_to_passenger_side : int, optional
        Number of vehicles in the passenger-side lane. Defaults to 1.
    speed : float, optional
        Base speed for vehicles in m/s. Defaults to 10.
    vehicle_separation : float, optional
        Longitudinal distance between vehicles in meters. Defaults to 10.
    horizontal_seperation : float, optional
        Lateral distance between lanes in meters. Defaults to 3.5.
    add_randomness : bool, optional
        Whether to add random variations. Defaults to False.
    pre_picked : array_like or None, optional
        Predefined path waypoints for the main lane.
    surface_meshes : list or None, optional
        List of surface meshes for path snapping.
    name : str, optional
        Base name for vehicle actors. Defaults to 'traffic'.

    Returns
    -------
    dict
        Updated traffic_dict with all vehicle-path associations.
        
    Notes
    -----
    The function creates perpendicular vectors to the main path direction to
    establish parallel lanes for multi-lane traffic simulation.
    """
    # Create main column of cars
    traffic_dict.update(create_traffic(
        all_actors, traffic_dict=traffic_dict, num_cars=num_cars_per_column,
        speed=speed, vehicle_separation=vehicle_separation,
        add_randomness=add_randomness, pre_picked=pre_picked,
        surface_meshes=surface_meshes, name=name
    ))

    pre_picked_orig = np.array(pre_picked)
    
    # Calculate perpendicular vectors for lane separation
    if num_cars_to_passenger_side > 0 or num_car_to_driver_side > 0:
        pre_picked = np.array(pre_picked)
        
        # Create vectors perpendicular to the path for lane positioning
        path_vector = np.diff(pre_picked, axis=0)
        norm = np.linalg.norm(path_vector, axis=1)
        
        # Normalize path vectors
        for n in range(len(path_vector)):
            path_vector[n] = path_vector[n] / norm[n]
        
        # Extend path vector array to match waypoint count
        path_vector_norm = np.zeros((path_vector.shape[0] + 1, path_vector.shape[1]))
        path_vector_norm[-1] = path_vector[-1]  # Use last vector for final point
        path_vector_norm[:-1] = path_vector
        
        up_vec = np.array([0, 0, 1])  # Vertical reference vector

        # Calculate perpendicular vectors for lane separation
        passenger_vector = np.cross(path_vector_norm, up_vec)  # Right side
        driver_vector = np.cross(up_vec, path_vector_norm)     # Left side

    # Create driver-side lane traffic
    if num_car_to_driver_side > 0:
        pre_picked = pre_picked_orig + horizontal_seperation * driver_vector
        traffic_dict.update(create_traffic(
            all_actors, traffic_dict=traffic_dict, num_cars=num_car_to_driver_side,
            speed=speed, vehicle_separation=vehicle_separation,
            add_randomness=add_randomness, pre_picked=pre_picked,
            surface_meshes=surface_meshes, name=name
        ))

    # Create passenger-side lane traffic
    if num_cars_to_passenger_side > 0:
        pre_picked = pre_picked_orig + horizontal_seperation * passenger_vector
        traffic_dict.update(create_traffic(
            all_actors, traffic_dict=traffic_dict, num_cars=num_cars_to_passenger_side,
            speed=speed, vehicle_separation=vehicle_separation,
            add_randomness=add_randomness, pre_picked=pre_picked,
            surface_meshes=surface_meshes, name=name
        ))

    return traffic_dict


def adjust_z_elevation(mesh=None, pos=[0, 0, 0], adjust_only_once=False, 
                      smooth=True, constant_offset=0):
    """
    Adjust z-coordinates of positions to align with surface mesh elevation.
    
    This function performs terrain following by adjusting z-coordinates of actor
    positions to match the underlying surface mesh elevation, ensuring actors
    appear to be standing/moving on the terrain surface.

    Parameters
    ----------
    mesh : pyvista.PolyData or list or None, optional
        Surface mesh(es) to use for elevation reference. If None, returns
        original positions. Can be single mesh or list of meshes.
    pos : array_like, optional
        Position(s) to adjust. Can be single [x, y, z] point or array of points.
        Defaults to [0, 0, 0].
    adjust_only_once : bool, optional
        If True, uses maximum elevation along entire path for all points.
        If False, adjusts each point individually based on local elevation.
        Defaults to False.
    smooth : bool, optional
        Whether to apply Gaussian smoothing to elevation profile when 
        adjust_only_once=False. Helps reduce abrupt elevation changes.
        Defaults to True.
    constant_offset : float, optional
        Additional vertical offset to add to all adjusted elevations.
        Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Position array with adjusted z-coordinates to match surface elevation.
        
    Notes
    -----
    The function uses ray tracing from above and below the mesh bounds to find
    intersection points with the surface. For multiple meshes, it uses the
    maximum elevation found across all meshes.
        
    Examples
    --------
    >>> terrain_mesh = pv.read('terrain.vtk')
    >>> actor_pos = [[0, 0, 5], [1, 1, 5], [2, 2, 5]]
    >>> adjusted_pos = adjust_z_elevation(terrain_mesh, actor_pos)
    """
    def get_z_surface_location(pos, mesh, buffer=10):
        """
        Find surface elevation at given x,y position using ray tracing.
        
        Parameters
        ----------
        pos : array_like
            [x, y, z] position to query.
        mesh : pyvista.PolyData
            Surface mesh for ray tracing.
        buffer : float, optional
            Vertical buffer above and below mesh bounds for ray tracing.
            
        Returns
        -------
        float or False
            Surface elevation at position, or False if no intersection found.
        """
        # Define ray endpoints with buffer above and below mesh
        start_z = mesh.bounds[4] - buffer  # mesh.bounds[4] is z_min
        stop_z = mesh.bounds[5] + buffer   # mesh.bounds[5] is z_max
        start_ray = [pos[0], pos[1], start_z]
        stop_ray = [pos[0], pos[1], stop_z]
        
        # Perform ray tracing to find surface intersections
        points, ind = mesh.ray_trace(start_ray, stop_ray)
        
        if len(points) == 0:
            return False
        else:
            # Return highest intersection point (handles overhangs)
            return np.max(points[:, 2])

    # Return original position if no mesh provided
    if mesh is None:
        return np.array(pos)
    
    # Convert single mesh to list for consistent processing
    if not isinstance(mesh, list):
        mesh = [mesh]

    # Convert position to numpy array for processing
    if isinstance(pos, list):
        pos = np.array(pos)
    
    if pos.ndim == 1:
        # Single position case
        all_z_pos = []
        for each in mesh:
            z_surface_location = get_z_surface_location(pos, each)
            if z_surface_location:
                all_z_pos.append(z_surface_location)
        
        if len(all_z_pos) == 0:
            return pos
        
        # Use maximum elevation from all meshes
        z_surface_location = np.max(np.array(all_z_pos)) + constant_offset
        pos[2] = z_surface_location
        return pos
    
    elif pos.ndim == 2 and adjust_only_once:
        # Multiple positions, single adjustment case
        all_z_pos = []
        for position in pos:
            for each in mesh:
                z_surface_location = get_z_surface_location(position, each)
                if z_surface_location:
                    all_z_pos.append(z_surface_location)
        
        if len(all_z_pos) == 0:
            return pos
        
        # Use single maximum elevation for all positions
        max_elevation = np.max(np.array(all_z_pos)) + constant_offset
        pos[:, 2] = max_elevation
        return pos
    
    elif pos.ndim == 2 and not adjust_only_once:
        # Multiple positions, individual adjustment case
        for pos_idx in range(len(pos)):
            all_z_pos = []
            for each in mesh:
                z_surface_location = get_z_surface_location(pos[pos_idx], each)
                if z_surface_location:
                    all_z_pos.append(z_surface_location)
            
            if len(all_z_pos) == 0:
                continue  # Skip this position if no elevation found
            
            # Set elevation for this specific position
            z_surface_location = np.max(np.array(all_z_pos)) + constant_offset
            pos[pos_idx, 2] = z_surface_location
        
        # Apply smoothing to elevation profile if requested
        if smooth:
            if len(pos) > 15:  # Only smooth if sufficient points
                pos[:, 2] = gaussian_filter1d(pos[:, 2], sigma=5)
        
        return pos


