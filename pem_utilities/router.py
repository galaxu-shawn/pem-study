"""
Path Generation Module for 3D Simulations

This module provides functionality for creating custom paths in 3D environments using 
interactive point selection or pre-defined coordinates. The paths are interpolated and 
can be used for object movement in simulations, particularly for pedestrian or vehicle 
trajectory planning.

Dependencies:
    - numpy: Numerical computations and array operations
    - pyvista: 3D visualization and mesh operations
    - scipy: Interpolation functions
    - utilities.rotation: Custom rotation utilities (euler_to_rot, vec_to_rot, look_at)
"""

import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from pem_utilities.rotation import euler_to_rot, vec_to_rot, look_at


class Pick_Path:
    """
    A class for creating and managing custom 3D paths for simulation objects.
    
    This class provides functionality to define paths either through interactive 
    3D point selection using PyVista's plotting interface or by providing 
    pre-defined coordinates. The resulting path includes position and rotation 
    interpolation functions suitable for animation and simulation.
    
    Attributes:
        pos_func (scipy.interpolate.interp1d): Interpolation function for positions
        rot_func (scipy.interpolate.interp1d): Interpolation function for rotations
        time_stamps (numpy.ndarray): Array of time values for the path
        total_travel_time (float): Total time required to traverse the path
        dist_xyz (numpy.ndarray): Distance components in each axis
        total_dist_travled (float): Total distance traveled along the path
        selected_points (numpy.ndarray): Array of selected/input path points
    """
    
    def __init__(self):
        """
        Initialize a new Pick_Path instance.
        
        All attributes are set to None initially and will be populated
        when custom_path() is called.
        """
        self.pos_func = None
        self.rot_func = None
        self.time_stamps = None
        self.total_travel_time = None
        self.dist_xyz = None
        self.total_dist_travled = None
        self.selected_points = None
    
    def custom_path(self, mesh_list=None, pre_picked=None, speed=10, opacity=1.0, 
                   upsample_selection=101, z_offset=0.0, snap_to_surface=True, show_model=True):
        """
        Define a custom path for simulation objects using interactive selection or pre-defined points.
        
        This method creates a smooth interpolated path from either user-selected points in a 3D 
        environment or from pre-defined coordinates. The path includes both position and rotation 
        interpolation functions based on travel speed and direction.
        
        Parameters:
            mesh_list (list or pyvista.PolyData, optional): 
                3D meshes to display for interactive point selection. Can be a single mesh 
                or list of meshes. Default is None.
            pre_picked (array-like, optional): 
                Pre-defined path points as [x, y] or [x, y, z] coordinates. If provided, 
                skips interactive selection. Default is None.
            speed (float, optional): 
                Travel speed in units per second for timing calculations. Default is 10.
            opacity (float, optional): 
                Opacity of displayed meshes (0.0 to 1.0). Default is 1.0.
            upsample_selection (int, optional): 
                Number of interpolated points in the final path. Default is 101.
            z_offset (float, optional): 
                Vertical offset to apply to all path points. Default is 0.0.
            snap_to_surface (bool, optional): 
                Whether to snap path points to mesh surface using ray tracing. Default is True.
            show_model (bool, optional): 
                Whether to display the 3D model during interactive selection. Default is True.
        
        Returns:
            None: Results are stored in instance attributes (pos_func, rot_func, etc.)
        
        Raises:
            ValueError: If both mesh_list and pre_picked are None.
            
        Note:
            When using interactive mode (pre_picked=None), the PyVista plotter window will 
            open allowing users to click points to define the path. The path will be 
            interpolated and smoothed based on the selected points.
        """
        
        # Ensure mesh_list is always a list for consistent processing
        if mesh_list is not None and not isinstance(mesh_list, list):
            mesh_list = [mesh_list]

        # Initialize PyVista plotter and mesh container
        plotter = pv.Plotter()
        total_mesh = pv.PolyData()

        # Validate input parameters
        if mesh_list is None and pre_picked is None:
            raise ValueError("Either mesh_list or pre_picked must be provided.")
        
        # Process mesh data and set up 3D environment bounds
        if mesh_list is not None and show_model:
            # Combine all meshes and add to plotter for interactive selection
            for mesh in mesh_list:
                total_mesh += mesh
            plotter.add_mesh(total_mesh, opacity=opacity, pickable=True)
            bounds = plotter.bounds
            zmin = bounds[4]  # Minimum z-coordinate
            zmax = bounds[5]  # Maximum z-coordinate
        elif mesh_list is not None and not show_model:
            # Combine meshes without displaying (for bounds calculation only)
            for index, mesh in enumerate(mesh_list):
                total_mesh += mesh
            bounds = total_mesh.bounds
            zmin = bounds[4]
            zmax = bounds[5]
            
        # Handle point selection: interactive vs pre-defined
        if pre_picked is None and show_model:
            # Interactive point selection using PyVista
            plotter.enable_path_picking()
            plotter.show_grid()
            plotter.show()

            # Extract selected points from plotter
            selected_points = plotter.picked_path
            selected_points = np.array(selected_points.points)
            
            # Remove duplicate points while preserving order
            if len(selected_points) > 1:
                selected_points, ind = np.unique(selected_points, return_index=True, axis=0)
                selected_points = selected_points[np.argsort(ind)]
            
            # Ensure we have at least 2D array structure
            selected_points = np.atleast_2d(selected_points)
            
            # Handle single point case by duplicating the point
            if len(selected_points) == 1:
                selected_points = np.stack((selected_points[0], selected_points[0]))
            
            # Apply vertical offset
            selected_points[:, 2] = selected_points[:, 2] + z_offset
            print('Selected points:')
            print(repr(selected_points))
        else:
            # Use pre-defined points
            selected_points = np.array(pre_picked)
            
            # Handle 2D input by adding z-coordinate
            if selected_points.shape[1] == 2:
                # Vectorized approach: extract x,y and set z=0
                selected_points = np.column_stack((
                    selected_points[:, 0], 
                    selected_points[:, 1], 
                    np.zeros(len(selected_points))
                ))
            
            # Apply vertical offset
            selected_points[:, 2] = selected_points[:, 2] + z_offset

        # Store selected points for reference
        self.selected_points = selected_points

        # Initialize output data array
        data = np.zeros((upsample_selection, selected_points.shape[1]))

        # Calculate timing based on distances and speed
        time_at_selected = [0]
        orig_diff = np.diff(selected_points, axis=0)  # Point-to-point differences
        orig_dist_at_step = np.linalg.norm(orig_diff, axis=1)  # Euclidean distances
        total_dist_orig_traveled = 0
        
        # Accumulate travel times based on distance and speed
        for dist in orig_dist_at_step:
            total_dist_orig_traveled += dist
            if dist != 0.0:
                time_at_selected.append(total_dist_orig_traveled / speed)
            else:  
                # Handle case where points are identical (single point scenario)
                time_at_selected.append(1)

        # Determine interpolation method based on number of points
        interp_type = 'quadratic'
        if len(selected_points) <= 2:
            interp_type = 'linear'

        # Create interpolation function for smooth path
        interp_func = interp1d(time_at_selected, selected_points, kind=interp_type, axis=0)
        
        # Generate uniform time samples for output
        new_t = np.linspace(0, time_at_selected[-1], num=upsample_selection)
        
        # Interpolate positions at uniform time intervals
        new_data = interp_func(new_t)

        # Optional: Snap interpolated points to mesh surface
        if snap_to_surface and mesh_list is not None:
            for n, t in enumerate(new_t):
                # Define vertical ray for surface intersection
                start_ray = [new_data[n, 0], new_data[n, 1], zmin - 10]
                stop_ray = [new_data[n, 0], new_data[n, 1], zmax + 10]
                
                # Perform ray tracing to find surface intersection
                points, ind = total_mesh.ray_trace(start_ray, stop_ray)
                
                if len(points) == 0:
                    # No intersection found, use minimum z-value
                    new_data[n, 2] = zmin + z_offset
                else:
                    # Use lowest intersection point (surface)
                    z_surface_location = np.min(points[:, 2])
                    new_data[n, 2] = z_surface_location + z_offset

        # Calculate rotation matrices for each path segment
        all_rot = []
        for n in range(len(new_data) - 1):
            # Calculate rotation to look from current point to next point
            rot = look_at(new_data[n], new_data[n + 1])
            all_rot.append(rot)
        
        # Use the same rotation for the last point as the second-to-last
        all_rot.append(rot)

        # Create interpolation functions for both position and rotation
        time_stamps = np.linspace(0, time_at_selected[-1], num=upsample_selection)
        
        # Position interpolation with extrapolation handling
        pos_interp_func = interp1d(
            time_stamps, new_data, kind=interp_type, axis=0, 
            fill_value=new_data[-1], bounds_error=False
        )
        
        # Rotation interpolation with extrapolation handling
        rot_interp_func = interp1d(
            time_stamps, all_rot, kind=interp_type, axis=0, 
            fill_value=all_rot[-1], bounds_error=False
        )

        # Store results in instance attributes
        self.pos_func = pos_interp_func
        self.rot_func = rot_interp_func
        self.time_stamps = time_stamps
        self.total_travel_time = time_at_selected[-1]
        self.total_dist_orig_traveled = total_dist_orig_traveled