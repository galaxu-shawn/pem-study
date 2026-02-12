"""
3D Model Visualization Utility for Electromagnetic Simulations

This module provides a comprehensive visualization framework for electromagnetic simulation
results using PyVista for 3D rendering and Matplotlib for 2D plot overlays. It supports
real-time visualization of radar/antenna systems, point clouds, and various data domains
including range-Doppler, frequency-pulse, and spatial field distributions.

Key Features:
- 3D scene rendering with PyVista
- Animated video generation with customizable camera controls
- Real-time data overlay with Matplotlib integration
- Point cloud visualization for radar targets
- Support for multiple coordinate systems and transformations
- Flexible camera positioning and movement

Created on Fri Jan 26 15:00:00 2024

@author: asligar
"""
import numpy as np
import pyvista as pv
import copy
import matplotlib.pyplot as plt
from pem_utilities.utils import color_mapper

# Import for screen resolution detection
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

def get_screen_resolution():
    """
    Detect the primary monitor's screen resolution.
    
    Returns:
        tuple: (width, height) of the screen in pixels, adjusted to be divisible by 16
        
    Note:
        Falls back to (1920, 1080) if detection fails.
        Dimensions are rounded down to nearest multiple of 16 for FFMPEG compatibility.
    """
    def round_to_16(value):
        """Round down to nearest multiple of 16 for FFMPEG compatibility."""
        return (value // 16) * 16
    
    try:
        if TKINTER_AVAILABLE:
            # Use tkinter to get screen resolution
            root = tk.Tk()
            root.withdraw()  # Hide the window
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            
            # Adjust to be divisible by 16
            width = round_to_16(width)
            height = round_to_16(height)
            return (width, height)
        else:
            # Fallback: try using PyQt/PySide if available
            try:
                from PySide6.QtWidgets import QApplication
                from PySide6.QtCore import Qt
                app = QApplication.instance()
                if app is None:
                    app = QApplication([])
                screen = app.primaryScreen()
                size = screen.size()
                width = round_to_16(size.width())
                height = round_to_16(size.height())
                return (width, height)
            except ImportError:
                pass
            
            # Second fallback: try PyQt5
            try:
                return (800, 592)  # 592 = 37*16, closest to 600
            except ImportError:
                pass
                
    except Exception as e:
        print(f"Warning: Could not detect screen resolution ({e}). Using default 1920x1088.")
    
    # Final fallback to common resolution (already divisible by 16)
    return (1920, 1088)  # 1088 = 68*16, closest to 1080


class ModelVisualization:
    """
    A comprehensive 3D visualization framework for electromagnetic simulation results.
    
    This class provides real-time 3D visualization capabilities with support for:
    - Scene actors (antennas, structures, objects)
    - Animated camera controls
    - Data overlay plots (range-Doppler, field strength, etc.)
    - Point cloud rendering for radar targets
    - Video generation for simulation sequences
    
    The visualization system integrates PyVista for 3D rendering with Matplotlib
    for 2D data overlays, creating a unified interface for complex EM simulations.
    """
    
    def __init__(self,
                 all_scene_actors,
                 show_antennas=True,
                 output_movie_name='output_geometry.mp4',
                 fps=10,
                 overlay_results=True,
                 vel_domain=None,
                 rng_domain=None,
                 freq_domain=None,
                 pulse_domain=None,
                 sim_time_domain=None,
                 x_domain=None,
                 y_domain=None,
                 angle_domain=None,
                 shape=(100,100),
                 output_video_size=None,
                 figure_size=(0.275, 0.375),
                 cmap='jet',
                 camera_orientation=None,
                 camera_attachment=None,
                 camera_position=None):
        """
        Initialize the 3D visualization framework.
        
        Args:
            all_scene_actors (dict or object): Scene actors containing meshes and properties.
                Can be a dictionary of actors or an object with an 'actors' attribute.
            show_antennas (bool, optional): Whether to display antenna elements. Defaults to True.
            output_movie_name (str, optional): Output video filename. Defaults to 'output_geometry.mp4'.
            fps (int, optional): Video frame rate. Defaults to 10.
            overlay_results (bool, optional): Enable matplotlib data overlay. Defaults to True.
            vel_domain (array-like, optional): Velocity domain for range-Doppler plots.
            rng_domain (array-like, optional): Range domain for various plot types.
            freq_domain (array-like, optional): Frequency domain for frequency-pulse plots.
            pulse_domain (array-like, optional): Pulse time domain for time-based plots.
            sim_time_domain (array-like, optional): Simulation time domain for temporal plots.
            x_domain (array-like, optional): X spatial domain for field plots.
            y_domain (array-like, optional): Y spatial domain for field plots.
            angle_domain (array-like, optional): Angular domain for range-angle plots.
            shape (tuple, optional): Default plot shape (height, width). Defaults to (100,100).
            output_video_size (tuple, optional): Video resolution (width, height). If None, 
                automatically detects screen resolution. Defaults to None.
            figure_size (tuple, optional): Matplotlib overlay size as fraction of screen. Defaults to (0.275, 0.375).
            cmap (str, optional): Colormap for visualizations. Defaults to 'jet'.
            camera_orientation (str or dict, optional): Camera orientation preset or custom settings.
                String options: 'scene_top', 'follow', 'follow2-8', 'side', 'top', 'front', 'radar'
            camera_attachment (str, optional): Name of actor to attach camera to.
            camera_position (tuple, optional): Fixed camera position (x, y, z).
        
        Raises:
            ValueError: If scene actors are invalid or incompatible domains are specified.
        """
        # Automatically detect screen resolution if not provided
        if output_video_size is None:
            output_video_size = get_screen_resolution()
            print(f"Auto-detected screen resolution: {output_video_size[0]}x{output_video_size[1]}")
        
        # Initialize matplotlib components
        self.ax = None
        self.fig = None
        self.h_chart = None
        
        # Store visualization parameters
        self.cmap = cmap
        self.show_antennas = show_antennas
        
        # Initialize PyVista plotter with video recording
        self.pl = pv.Plotter(window_size=output_video_size)
        self.pl.open_movie(output_movie_name, framerate=fps)

        # Camera configuration
        # camera_orientation: controls predefined camera movements and positions
        # camera_attachment: actor name for camera to follow during animation
        # camera_position: fixed camera position override
        self.camera_orientation = camera_orientation
        self.camera_attachment = camera_attachment
        self.camera_position = camera_position

        # Animation and scene tracking
        self.scene_idx_counter = 0  # Frame counter for animation sequences
        self.point_cloud_mesh = None  # Mesh for radar target point clouds
        self.point_cloud_actor = None  # PyVista actor for point cloud rendering
        
        # Process and add scene actors to the visualization
        if isinstance(all_scene_actors, dict):
            # Direct dictionary of actors
            for actor in all_scene_actors:
                self._add_parts_to_scene(all_scene_actors[actor])
        elif hasattr(all_scene_actors, 'actors'):
            # Object with actors attribute
            all_scene_actors = all_scene_actors.actors
            for actor in all_scene_actors:
                self._add_parts_to_scene(all_scene_actors[actor])
        else:
            print('No actors found in scene')

        # Add coordinate axes to the scene for reference
        self.pl.add_axes(line_width=2, xlabel='X', ylabel='Y', zlabel='Z')

        # Initialize matplotlib overlay for data visualization
        if overlay_results:
            self._setup_data_overlay(
                vel_domain, rng_domain, freq_domain, pulse_domain,
                sim_time_domain, x_domain, y_domain, angle_domain, shape, figure_size
            )

        # Store scene actors reference and set initial camera view
        self.all_scene_actors = all_scene_actors
        self._camera_view()

    def _setup_data_overlay(self, vel_domain, rng_domain, freq_domain, pulse_domain,
                           sim_time_domain, x_domain, y_domain, angle_domain, shape, figure_size):
        """
        Initialize matplotlib overlay for real-time data visualization.
        
        This method sets up the appropriate plot type based on the provided domain parameters.
        Supported plot types include range-Doppler, frequency-pulse, spatial field distributions,
        range-angle, and temporal range profiles.
        
        Args:
            vel_domain (array-like): Velocity values for range-Doppler plots
            rng_domain (array-like): Range values for various plot types
            freq_domain (array-like): Frequency values for frequency-pulse plots
            pulse_domain (array-like): Pulse time values for time-based plots
            sim_time_domain (array-like): Simulation time values for temporal plots
            x_domain (array-like): X spatial coordinates for field plots
            y_domain (array-like): Y spatial coordinates for field plots
            angle_domain (array-like): Angular values for range-angle plots
            shape (tuple): Default plot dimensions (height, width)
            figure_size (tuple): Matplotlib overlay size as fraction of screen
        """
        # Create matplotlib figure for data overlay
        self.fig, self.ax = plt.subplots(tight_layout=True)

        # Determine plot type based on available domains and initialize with random data
        if vel_domain is not None and rng_domain is not None:
            # Range-Doppler plot for radar applications
            results = np.random.rand(len(vel_domain), len(rng_domain))
            self.mpl_ax_handle = self.ax.imshow(results, vmin=-250, vmax=-100, cmap=self.cmap, aspect='auto',
                                           extent=[vel_domain[0], vel_domain[-1], rng_domain[0], rng_domain[-1]])
            self.ax.set_xlabel('Doppler velocity (m/s)', fontsize=12)
            self.ax.set_ylabel('Range (m)', fontsize=12)
            self.ax.set_title('Range vs. Doppler', fontsize=12)
            self.ax.tick_params(axis='both', labelsize=12)
            
        elif pulse_domain is not None and rng_domain is not None:
            # Range-Pulse plot for temporal radar analysis
            results = np.random.rand(len(pulse_domain), len(rng_domain))
            self.mpl_ax_handle = self.ax.imshow(results, vmin=-250, vmax=-100, cmap=self.cmap, aspect='auto',
                                           extent=[pulse_domain[0], pulse_domain[-1], rng_domain[0], rng_domain[-1]])
            self.ax.set_xlabel('Frame Pulse Time (s)', fontsize=12)
            self.ax.set_ylabel('Range (m)', fontsize=12)
            self.ax.set_title('Range vs. Pulse', fontsize=12)
            self.ax.tick_params(axis='both', labelsize=12)
            
        elif freq_domain is not None and pulse_domain is not None:
            # Frequency-Pulse plot for signal analysis
            results = np.random.rand(len(pulse_domain), len(freq_domain))
            self.mpl_ax_handle = self.ax.imshow(results, cmap=self.cmap, aspect='auto',
                                           extent=[pulse_domain[0], pulse_domain[-1], freq_domain[0], freq_domain[-1]])
            self.ax.set_xlabel('Pulse (s)', fontsize=12)
            self.ax.set_ylabel('Freq (Hz)', fontsize=12)
            self.ax.set_title('Freq vs. Pulse', fontsize=12)
            self.ax.tick_params(axis='both', labelsize=12)

        elif x_domain is not None and y_domain is not None:
            # 2D spatial field distribution
            results = np.random.rand(len(x_domain), len(y_domain))
            self.mpl_ax_handle = self.ax.imshow(results.T, cmap=self.cmap, aspect='auto',
                                                extent=[x_domain[0], x_domain[-1], y_domain[0], y_domain[-1]])
            self.ax.set_xlabel('X (m)', fontsize=12)
            self.ax.set_ylabel('Y (m)', fontsize=12)
            self.ax.set_title('Field Strength', fontsize=12)
            self.ax.tick_params(axis='both', labelsize=12)
            
        elif x_domain is not None and y_domain is None:
            # 1D line plot for spatial profiles
            results = np.random.rand(len(x_domain))
            self.mpl_ax_handle, = self.ax.plot(results)
            
        elif rng_domain is not None and angle_domain is not None:
            # Range-Angle plot for beam pattern analysis
            results = np.random.rand(len(angle_domain), len(rng_domain))
            self.mpl_ax_handle = self.ax.imshow(results, cmap=self.cmap,
                                           extent=[angle_domain[0], angle_domain[-1], rng_domain[0], rng_domain[-1]],
                                           origin='lower')
            self.ax.set_xlabel('Angle (deg)')
            self.ax.set_ylabel('Range (m)')
            self.ax.set_title('Range vs. Angle')
            
        elif rng_domain is not None and sim_time_domain is not None:
            # Range profile vs simulation time for temporal analysis
            results = np.random.rand(len(sim_time_domain), len(rng_domain))
            self.mpl_ax_handle = self.ax.imshow(results, cmap=self.cmap,
                                           extent=[sim_time_domain[0], sim_time_domain[-1], rng_domain[0], rng_domain[-1]],
                                           origin='lower', aspect='auto')
            self.ax.set_xlabel('Simulation Time (s)')
            self.ax.set_ylabel('Range (m)')
            self.ax.set_title('Range vs. Simulation Time')
        else:
            # Default generic 2D plot
            results = np.random.rand(shape[0], shape[1])
            self.mpl_ax_handle = self.ax.imshow(results.T, cmap=self.cmap, aspect='auto')

        # Add matplotlib chart to PyVista plotter as overlay
        self.h_chart = pv.ChartMPL(self.fig, size=figure_size, loc=((1-figure_size[0])*.95, (1-figure_size[1])*.95))
        self.pl.add_chart(self.h_chart)

    def _add_parts_to_scene(self, actor):
        """
        Recursively add actor parts and meshes to the 3D scene.
        
        This method processes actor hierarchies, applying appropriate colors, textures,
        and transparency settings. It handles special cases for antennas and generators,
        and recursively processes child parts.
        
        Args:
            actor: Scene actor object containing mesh data, colors, and part hierarchy
            
        Note:
            - Antenna actors are only added if show_antennas is True
            - Generator actors use special color schemes and update mechanisms
            - Color inheritance follows a hierarchy: part color > actor color > random
        """
        # Determine top-level actor color for inheritance
        top_level_actor_color = None
        if actor.color is not None:
            top_level_actor_color = color_mapper(color=actor.color)

        # Handle antenna actors with special rendering options
        if actor.is_antenna and self.show_antennas:
            options = {}
            options['cmap'] = 'jet'  # Use jet colormap for antenna pattern visualization
            if actor.mesh is not None:
                actor.pv_actor = self.pl.add_mesh(actor.mesh, **options)

        # Handle non-antenna top-level meshes
        elif actor.mesh is not None and not actor.is_antenna:
            options = {}
            c = color_mapper(random=True, cmap=self.cmap)
            options['color'] = c
            if actor.transparency is not None:
                options['use_transparency'] = True
                options['opacity'] = actor.transparency  # Fixed: was referencing wrong transparency
            actor.pv_actor = self.pl.add_mesh(actor.mesh, **options)

        # Process all parts within the actor
        for part in actor.parts.keys():
            if actor.parts[part].mesh:
                options = {}
                
                # Check for texture mapping first
                if hasattr(actor.parts[part].mesh, 'textures'):
                    options['texture'] = actor.parts[part].mesh.textures
                    
                # Check for material or color data in mesh
                elif 'material' in actor.parts[part].mesh.cell_data.keys():
                    # VTP files might have material index mapping
                    actor.parts[part].mesh.set_active_scalars('material')
                elif 'color' in actor.parts[part].mesh.cell_data.keys():
                    actor.parts[part].mesh.set_active_scalars('color')
                    
                # Apply color hierarchy: part > actor > type-specific > random
                else:
                    if actor.parts[part].color is not None:
                        c = color_mapper(color=actor.parts[part].color)
                        options['color'] = c
                    elif top_level_actor_color is not None:
                        c = top_level_actor_color
                        options['color'] = c
                    elif actor.actor_type == 'generator':
                        c = 'bone'  # Special colormap for generator objects
                        options['cmap'] = c
                    else:
                        c = color_mapper(random=True, cmap=self.cmap)
                        options['color'] = c
                        
                    # Apply transparency if specified
                    if actor.transparency is not None:
                        options['use_transparency'] = True
                        options['opacity'] = actor.transparency

                # Add the mesh to the scene
                actor.pv_actor = self.pl.add_mesh(actor.parts[part].mesh, **options)
                
            # Recursively process child parts
            if len(actor.parts[part].parts) > 0:
                for child_part in actor.parts[part].parts:
                    self._add_parts_to_scene(actor.parts[part].parts[child_part])

    def _update_parts_in_scene(self, actor):
        """
        Update actor positions and transformations in the 3D scene.
        
        This method applies incremental transformations to all actor parts based on their
        current coordinate systems. It handles special cases for antennas and generators,
        ensuring proper transformation updates during animation sequences.
        
        Args:
            actor: Scene actor object to update with current coordinate system state
            
        Note:
            - Computes relative transforms to avoid accumulation errors
            - Generator actors can use absolute transforms for dynamic updates
            - Recursively updates all child parts in the actor hierarchy
        """
        # Update antenna meshes if enabled
        if actor.is_antenna and self.show_antennas:
            T = actor.coord_sys.transform4x4  # Current 4x4 transformation matrix
            previous_T = actor.previous_transform  # Previous frame's transformation
            # Compute incremental transform to avoid accumulation errors
            total_transform = np.matmul(T, np.linalg.inv(previous_T))
            
            if hasattr(actor, 'mesh'):
                if actor.mesh is not None:
                    actor.mesh.transform(total_transform, inplace=True)  # Apply position update
            actor.previous_transform = T  # Store current transform for next iteration

        # Update top-level actor meshes (non-antenna)
        if actor.mesh is not None:
            T = actor.coord_sys.transform4x4  # Current 4x4 transformation matrix
            previous_T = actor.previous_transform  # Previous frame's transformation
            # Compute incremental transform to avoid accumulation errors
            total_transform = np.matmul(T, np.linalg.inv(previous_T))
            
            if hasattr(actor, 'mesh'):
                if actor.mesh is not None:
                    actor.mesh.transform(total_transform, inplace=True)  # Apply position update
            actor.previous_transform = T  # Store current transform for next iteration

        # Update all parts within the actor
        for part in actor.parts.keys():
            T = actor.parts[part].coord_sys.transform4x4  # Current part transformation
            previous_T = actor.parts[part].previous_transform  # Previous part transformation
            # Compute incremental transform for this part
            total_transform = np.matmul(T, np.linalg.inv(previous_T))
            
            if hasattr(actor.parts[part], 'mesh'):
                if actor.parts[part].mesh is not None:
                    # Handle different update strategies for generators vs regular actors
                    if actor.actor_type != 'generator' or not actor.dynamic_generator_updates:
                        # Standard incremental transform update
                        actor.parts[part].mesh.transform(total_transform, inplace=True)
                    else:
                        # Generator objects: apply absolute transform and update dataset
                        # Generators typically receive new data at original locations
                        actor.parts[part].mesh.transform(total_transform, inplace=True)
                        self.pl.actors[actor.pv_actor.name].mapper.dataset = actor.parts[part].mesh

            # Store current transform for next iteration
            actor.parts[part].previous_transform = T
            
            # Recursively update child parts
            if len(actor.parts[part].parts) > 0:
                for child_part in actor.parts[part].parts:
                    self._update_parts_in_scene(actor.parts[part].parts[child_part])

    def update_frame(self, write_frame=True, plot_data=None, plot_limits=None, update_camera_view=True,return_raw_image=False):
        """
        Update the visualization for a single animation frame.
        
        This method updates both the 3D scene geometry and the 2D data overlay,
        then optionally writes the frame to the output video or displays it interactively.
        
        Args:
            write_frame (bool, optional): Whether to write frame to video file. Defaults to True.
            plot_data (array-like, optional): New data for the matplotlib overlay plot.
                Can be 1D (line plot) or 2D (image plot).
            plot_limits (tuple, optional): Color scale limits (vmin, vmax) for the plot.
            update_camera_view (bool, optional): Whether to update camera position. Defaults to True.
                
        Note:
            - Automatically detects plot data dimensionality and updates accordingly
            - Camera view is updated on first frame and when explicitly requested
            - All scene actors are updated with current transformations
        """
        # Update matplotlib overlay data if provided
        if plot_data is not None:
            if isinstance(plot_data, list):
                plot_data = np.array(plot_data)
                
            if plot_data.ndim == 1:
                # Handle 1D line plot updates
                self.mpl_ax_handle.set_ydata(plot_data)
                self.ax.set_ylim(np.min(plot_data), np.max(plot_data))
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                # Handle 2D image plot updates
                self.mpl_ax_handle.set_data(plot_data)
                if plot_limits is not None:
                    self.mpl_ax_handle.set_clim(vmin=plot_limits[0], vmax=plot_limits[1])

        # Update all scene actors with current transformations
        for actor in list(self.all_scene_actors.keys()):
            self._update_parts_in_scene(self.all_scene_actors[actor])

        # Update camera view on first frame or when explicitly requested
        if self.scene_idx_counter == 0 or update_camera_view:
            self._camera_view()

        # Write frame to video or display interactively
        if write_frame:
            self.pl.write_frame()
        else:
            self.pl.show()

            
        # Increment frame counter for animation tracking
        self.scene_idx_counter += 1

        if return_raw_image:
            return self.pl.screenshot(return_img=True)

    def _camera_view(self, camera_attachment=None):
        """
        Configure and update camera position and orientation.
        
        This method handles various camera modes including fixed positions, predefined
        orientations, and dynamic camera following. It supports both string-based
        presets and custom dictionary configurations.
        
        Args:
            camera_attachment (str, optional): Actor name to attach camera to.
                Overrides the instance's camera_attachment if provided.
                
        Camera Orientation Presets:
            - 'scene_top': Top-down view of entire scene
            - 'follow': Close third-person follow camera
            - 'follow2-8': Various follow distances and angles
            - 'side': Side view perspective
            - 'top': Overhead view with forward orientation
            - 'front': Front-facing view
            - 'radar': First-person radar perspective
            
        Note:
            - Fixed camera positions override all other settings
            - Camera transformations are relative to attached actor's coordinate system
            - Custom orientations can be specified via dictionary with required keys
        """
        # Use fixed camera position if specified
        if self.camera_position is not None:
            self.pl.camera_position = self.camera_position
            return

        # Determine camera attachment (allow override for dynamic changes)
        if camera_attachment is None:
            camera_attachment = self.camera_attachment

        # Early return if no camera configuration is specified
        if camera_attachment is None and self.camera_orientation is None:
            return
            
        # Validate camera attachment exists in scene
        if camera_attachment not in self.all_scene_actors.keys():
            print(f"Camera attachment {camera_attachment} not found in scene")
            return
            
        # Get transformation matrix from attached actor
        cam_transform = self.all_scene_actors[camera_attachment].coord_sys.transform4x4
        cam_pos = cam_transform[0:3, 3]  # Extract position
        cam_rot = cam_transform[0:3, 0:3]  # Extract rotation matrix

        # Apply camera orientation settings
        if self.camera_orientation is not None:
            if isinstance(self.camera_orientation, str):
                # Handle predefined string-based camera orientations
                orientation = self.camera_orientation.lower()
                
                if orientation == 'scene_top':
                    cam_offset = [0, 0, 100]  # High altitude overview
                    focal_offset = [0, 0, 0]
                    self.pl.camera_position = 'xy'  # Use PyVista's XY plane view
                    
                elif orientation == 'follow':
                    cam_offset = [-10, 0, 3]  # Standard third-person follow
                    focal_offset = [25, 0, 0.5]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 80
                    
                elif orientation == 'follow2':
                    cam_offset = [-20, 0, 6]  # Medium distance follow
                    focal_offset = [30, 0, 0.5]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 80
                    
                elif orientation == 'follow3':
                    cam_offset = [-30, 0, 9]  # Long distance follow
                    focal_offset = [50, 0, 0.5]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 80
                    
                elif orientation == 'follow4':
                    cam_offset = [-2, 0, 1]  # Close follow
                    focal_offset = [10, 0, 0.1]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 88
                    
                elif orientation == 'follow5':
                    cam_offset = [-2, 0, 2]  # Close elevated follow
                    focal_offset = [10, 0, 0.1]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 80
                    
                elif orientation == 'follow6':
                    cam_offset = [-1, 0, 1.5]  # Very close follow
                    focal_offset = [10, 0, -0.1]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 75
                    
                elif orientation == 'follow7':
                    cam_offset = [-150, 0, 50]  # Aerial follow view
                    focal_offset = [50, 0, 0.5]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 60
                    
                elif orientation == 'follow8':
                    cam_offset = [-0.5, 0, 0.25]  # Extremely close follow
                    focal_offset = [1, 0, 0]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 75
                    
                elif orientation == 'side':
                    cam_offset = [1, -8, 1]  # Side perspective view
                    focal_offset = [0, 25, 0.5]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    
                elif orientation == 'top':
                    cam_offset = [0, 0, 15]  # Overhead view
                    focal_offset = [0, 0, 0]
                    self.pl.camera.up = (1.0, 0.0, 0.0)  # Forward as up direction
                    
                elif orientation == 'front':
                    cam_offset = [14, 0, 3]  # Front-facing view
                    focal_offset = [-10, 0, 0]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    
                elif orientation == 'radar':
                    cam_offset = [0, 0, 0]  # First-person radar view
                    focal_offset = [25, 0, 0]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    self.pl.camera.view_angle = 80
                    
                else:
                    # Default fallback orientation
                    cam_offset = [-12, 0, 3]
                    focal_offset = [25, 0, 0.5]
                    self.pl.camera.up = (0.0, 0.0, 1.0)
                    
            elif isinstance(self.camera_orientation, dict):
                # Handle custom dictionary-based camera configuration
                cam_offset = self.camera_orientation['cam_offset']
                focal_offset = self.camera_orientation['focal_offset']
                self.pl.camera.up = self.camera_orientation['up']
                self.pl.camera.view_angle = self.camera_orientation['view_angle']

            # Transform camera offset from local to world coordinates
            cam_offset = (cam_offset[0] * cam_rot[:3, 0] + 
                         cam_offset[1] * cam_rot[:3, 1] + 
                         cam_offset[2] * cam_rot[:3, 2])
            camera_pos = cam_pos + cam_offset
            
            # Transform focal point offset from local to world coordinates
            focal_cam_offset = (focal_offset[0] * cam_rot[:3, 0] + 
                               focal_offset[1] * cam_rot[:3, 1] + 
                               focal_offset[2] * cam_rot[:3, 2])
            focal_pos = cam_pos + focal_cam_offset
            
            # Apply final camera position and focal point
            self.pl.camera.position = camera_pos
            self.pl.camera.focal_point = focal_pos

    def add_point_cloud_to_scene(self,
                                 target_list,
                                 tx_pos=[0,0,0],
                                 tx_rot=np.eye(3),
                                 point_cloud_mesh=None,
                                 color_min=-10,
                                 color_max=10,
                                 color_mode='p_received',
                                 size_mode='p_received',
                                 shape='sphere',
                                 max_radius=10):
        """
        Add or update a point cloud visualization for radar targets in the scene.
        
        This method creates 3D geometric representations of radar targets with properties
        like received power, velocity, and RCS mapped to visual attributes such as color
        and size. Supports multiple visualization modes and shape options.
        
        Args:
            target_list (dict): Dictionary of target data with structure:
                target_name: {
                    'xpos': float, 'ypos': float, 'zpos': float,  # Target position
                    'p_received': float,  # Received power
                    'velocity': float,    # Target velocity
                    'rcs': float         # Radar cross section
                }
            tx_pos (list, optional): Transmitter position [x, y, z]. Defaults to [0,0,0].
            tx_rot (array-like, optional): Transmitter rotation matrix (3x3). Defaults to identity.
            point_cloud_mesh (unused): Legacy parameter, not currently used.
            color_min (float, optional): Minimum value for color scale. Defaults to -10.
            color_max (float, optional): Maximum value for color scale. Defaults to 10.
            color_mode (str, optional): Property to map to color: 'p_received', 'velocity', 'rcs'. 
                Defaults to 'p_received'.
            size_mode (str, optional): Property to map to size: 'p_received', 'velocity', 'rcs'. 
                Defaults to 'p_received'.
            shape (str, optional): Target shape: 'sphere', 'cube', 'point'. Defaults to 'sphere'.
            max_radius (float, optional): Maximum radius for size scaling. Defaults to 10.
            
        Note:
            - First call creates the point cloud actor, subsequent calls update the existing one
            - Size and color mappings are automatically normalized across all targets
            - Transformations are applied to position targets relative to the transmitter
        """
        cmap = 'jet'  # Colormap for point cloud visualization
        
        # Create 4x4 transformation matrix for transmitter coordinate system
        T = np.concatenate((np.asarray(tx_rot), np.asarray(tx_pos).reshape((-1, 1))), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)

        # Initialize containers for point cloud geometry and transforms
        target_geo = []         # Individual target geometries
        target_transforms = []  # Individual transform pairs
        total_transforms = []   # Combined transforms
        point_cloud = None      # Final combined point cloud mesh

        # Extract all target properties for normalization
        all_p_received = []
        all_vel = []
        all_rcs = []
        for target in target_list:
            all_p_received.append(target_list[target]['p_received'])
            all_vel.append(target_list[target]['velocity'])
            all_rcs.append(target_list[target]['rcs'])
            
        # Calculate normalization ranges for each property
        p_received_max = np.max(10*np.log10(np.abs(np.array(all_p_received))))
        p_received_min = np.min(10*np.log10(np.abs(np.array(all_p_received))))
        vel_max = np.max(np.array(all_vel))
        vel_min = np.min(np.array(all_vel))
        rcs_max = np.max(np.array(all_rcs))
        rcs_min = np.min(np.array(all_rcs))

        # Create individual target geometries
        for target in target_list:
            # Extract target position and create local transform
            pos = [target_list[target]['xpos'], target_list[target]['ypos'], target_list[target]['zpos']]
            target_transform = np.eye(4)
            target_transform[0:3, 3] = pos

            # Extract target properties
            rcs = target_list[target]['rcs']
            p_received = target_list[target]['p_received']
            vel = target_list[target]['velocity']

            # Determine color value based on selected mode
            color_val = 1  # Default color value
            if color_mode == 'p_received':
                color_val = p_received
            elif color_mode == 'velocity':
                color_val = vel
            elif color_mode == 'rcs':
                color_val = rcs

            # Determine size value based on selected mode with normalization
            size_val = 1  # Default size value
            if size_mode == 'p_received':
                # Convert to dB and normalize to [0, 1] range
                p_rec_db = 10*np.log10(np.abs(p_received))
                p_rec_db_norm = (p_rec_db - p_received_min) / (p_received_max - p_received_min)
                size_val = p_rec_db_norm * max_radius
            elif size_mode == 'velocity':
                # Normalize velocity magnitude to [0, 1] range
                vel_abs = np.abs(vel)
                vel_norm = (vel_abs - vel_min) / (vel_max - vel_min)
                size_val = vel_norm * max_radius
            elif size_mode == 'rcs':
                size_val = rcs

            # Create target geometry based on specified shape
            if shape == 'cube':
                sphere = pv.Cube(center=(0, 0, 0), x_length=size_val, y_length=size_val, z_length=size_val)
            elif shape == 'point':
                # TODO: Implement point-based visualization for large datasets
                sphere = pv.Cube(center=(0, 0, 0), x_length=size_val, y_length=size_val, z_length=size_val)
            else:
                # Default sphere geometry
                sphere = pv.Sphere(radius=size_val, center=(0, 0, 0))
                
            # Assign color property to all points in the geometry
            sphere[color_mode] = np.ones(sphere.n_points) * color_val
            
            # Store geometry and transformation information
            target_geo.append(sphere)
            target_transforms.append([T, target_transform])
            total_transforms.append(np.matmul(T, target_transform))

        # Apply transformations and combine all target geometries
        targets_geo_transforms = zip(target_geo, total_transforms)
        for geo, trans in targets_geo_transforms:
            # Apply combined transform to position target in world coordinates
            geo.transform(trans, inplace=True)
            geo_copy = copy.deepcopy(geo)
            
            # Combine into single point cloud mesh
            if point_cloud is None:
                point_cloud = geo_copy
            else:
                point_cloud += geo_copy
                
        # Store the combined point cloud mesh
        self.point_cloud_mesh = point_cloud
        
        # Add to scene or update existing point cloud
        if (self.point_cloud_actor is None and point_cloud is not None):
            # First time: create new point cloud actor
            self.point_cloud_actor = self.pl.add_mesh(self.point_cloud_mesh,
                                                       opacity=1,
                                                       clim=[color_min, color_max],
                                                       cmap=cmap)
        else:
            # Update existing point cloud with new data
            try:
                self.point_cloud_actor.mapper.dataset = self.point_cloud_mesh
            except Exception as e:
                print("Something went wrong updating point cloud - no points in cloud?")
                return None

    def close(self):
        """
        Clean up and close the visualization framework.
        
        This method properly closes the PyVista plotter, finalizes any video recording,
        and releases associated resources. Should be called when visualization is complete.
        
        Note:
            - Automatically finalizes video recording if enabled
            - Closes all visualization windows
            - Releases memory and file handles
        """
        self.pl.close()
