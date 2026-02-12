# -*- coding: utf-8 -*-
"""
SAR Setup Backend Module

This module provides a backend class for setting up and running Synthetic Aperture Radar (SAR)
simulations. It handles radar configuration, scene setup, waveform generation, and image processing
for SAR imaging applications.

The module supports various imaging methods including range-doppler processing and advanced SAR
algorithms like polar format, omega-k, and backprojection. It can simulate both point source
and planewave radar configurations with configurable observer positions and target orientations.

Created on Mon Mar 29 10:51:32 2021
@author: asligar

Dependencies:
    - numpy: Numerical computations
    - scipy: Scientific computing
    - matplotlib: Plotting and visualization
    - pyvista: 3D visualization
    - tqdm: Progress bars
    - api_core: Core radar simulation API
    - utilities.*: Various utility modules for radar processing
"""

from tqdm import tqdm
import numpy as np
import os
import sys
import scipy
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as walltime
#
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
# Import utility modules for various radar processing functions
from pem_utilities.rotation import euler_to_rot, vec_to_rot, look_at, rot_to_euler
from pem_utilities.post_processing_radar_imaging import platform_dict, img_plane_dict, SARImageProcessor
from pem_utilities.antenna_device import add_single_tx_rx, Waveform, AntennaDevice
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.utils import apply_math_function
from pem_utilities.domain_transforms import DomainTransforms

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


class SAR_Setup_Backend:
    """
    Backend class for SAR (Synthetic Aperture Radar) simulation setup and execution.
    
    This class provides a comprehensive framework for setting up SAR simulations including:
    - Radar configuration and waveform generation
    - Scene setup with targets and observers
    - Ray tracing simulation with configurable parameters
    - Multiple SAR imaging algorithms
    - Debug output and visualization capabilities
    
    The class supports both traditional range-doppler processing and advanced SAR imaging
    methods such as polar format algorithm, omega-k algorithm, and backprojection.
    """
    
    def __init__(self,
                 target_center=[0, 0, 0],
                 azimuth_aspect_deg=1.7,
                 use_planewave=False,
                 azimuth_observer_deg=0,
                 elevation_observer_deg=45,
                 distance_observer_m=1000,
                 image_method='fft',
                 export_debug=False):
        """
        Initialize the SAR setup backend with configuration parameters.
        
        Args:
            target_center (list): XYZ coordinates of target center [x, y, z] in meters
            azimuth_aspect_deg (float): Azimuth aspect angle in degrees for synthetic aperture
            use_planewave (bool): If True, use planewave source; if False, use point source
            azimuth_observer_deg (float): Observer azimuth angle in degrees (0° = +X axis)
            elevation_observer_deg (float): Observer elevation angle in degrees (0° = XY plane)
            distance_observer_m (float): Distance from observer to target center in meters
            image_method (str, optional): SAR imaging method ('range_doppler', 'polar_format', 
                                        'omega_k', 'backprojection', 'isar')
            export_debug (bool): Enable debug output including camera views and logs
        """
        # Radar waveform parameters
        self._center_freq = 10e9          # Center frequency in Hz (10 GHz)
        self.num_freqs = 1400            # Number of frequency samples
        self.bandwidth = 300e6           # Bandwidth in Hz (300 MHz)
        self.num_pulse_CPI = 1400        # Number of pulses in Coherent Processing Interval

        # Imaging configuration
        self.image_method = image_method              # SAR imaging algorithm to use
        self.image_plane_dict = None                  # Image plane parameters (set during processing)
        self.range_pixels = self.num_freqs           # Range dimension pixels
        self.doppler_pixels = self.num_pulse_CPI     # Doppler dimension pixels
        
        # Domain transform objects for range and cross-range processing
        self.domain_transforms_down_range = None     # Range domain transforms
        self.domain_transforms_cross_range = None    # Cross-range domain transforms
        
        # Window specifications for range and doppler processing
        self.d_specs = 'hann'            # Doppler window specification
        self.r_specs = 'hann'            # Range window specification
        
        # Aspect angle tracking
        self.previous_aspect_deg = None              # Previous aspect angle for change detection
        self.azimuth_aspect_deg = azimuth_aspect_deg # Current azimuth aspect angle
        
        # Antenna beam parameters
        self.beam_width_h = 60           # Horizontal beamwidth in degrees
        self.beam_width_v = 60           # Vertical beamwidth in degrees
        
        # Scene geometry
        self.target_center = target_center           # Target center coordinates
        
        # Observer position parameters (used when tx/rx positioning is active)
        self.azimuth_observer_deg = azimuth_observer_deg     # Observer azimuth
        self.elevation_observer_deg = elevation_observer_deg # Observer elevation
        self.distance_observer_m = distance_observer_m       # Observer distance
        
        # Ray tracing simulation options
        self.ray_shoot_method = 'grid'  # Ray shooting method ('grid' or 'sbr')
        self.go_blockage = -1            # GO blockage setting (-1: disabled, ≥0: enabled)
        self.max_num_refl = 3            # Maximum number of reflections
        self.max_num_trans = 0           # Maximum number of transmissions
        self._ray_spacing = 0.1
        lambda_center = 2.99792458e8 / self._center_freq  # Wavelength at center frequency
        self._ray_density = np.sqrt(2) * lambda_center / self._ray_spacing # self consisten for initialization
        self.max_batches=100
        self.gpu_quota = 0.9
        self.gpu_device = 0
        self.range_filter = None         # Range filter specification
        self.enhanced_ray_processing = False  # Enable enhanced ray processing, only supported for 26.1 and higher
        self.skip_terminal_bnc_po_blockage = False # Skip terminal blockage for PO if True
        # Timing parameters
        self.cpi_duration = 1            # CPI duration in seconds (1s for easy calculations)
        
        # Source configuration
        self.use_planewave = use_planewave    # Planewave vs point source flag
        self.polarization = 'VV'              # Antenna polarization
        
        # Output configuration
        self.output_path = '../output/'       # Output directory path
        
        # Image scaling factors
        self.scale_factor_range_image = 1     # Range dimension scaling factor
        self.scale_factor_doppler_image = 1   # Doppler dimension scaling factor
        
        # Debug options
        self.export_debug = export_debug      # Debug export flag
        
        # Runtime tracking
        self.number_of_runs = 0              # Counter for simulation runs (for parameter display)

        self.sim_performance_time = 0          # simulation time track
        
    # Ray spacing properties
    @property
    def ray_spacing(self):
        """
        Get the current ray spacing in meters.
        
        Returns:
            float: Ray spacing value in meters
        """
        return self._ray_spacing

    @ray_spacing.setter
    def ray_spacing(self, value):
        """
        Set the ray spacing in meters.
        
        Args:
            value (float): Ray spacing in meters
            
        Note:
            Setting ray_spacing will clear ray_density to None
        """

        lambda_center = 2.99792458e8 / self._center_freq
        self._ray_density = np.sqrt(2) * lambda_center / value
        self._ray_spacing = value


    @property
    def ray_density(self):
        """
        Get the current ray density.
        
        Returns:
            float or None: Ray density value, or None if using ray_spacing
        """
        return self._ray_density

    @ray_density.setter
    def ray_density(self, value):
        """
        Set the ray density, which overrides ray_spacing.
        
        Args:
            value (float): Ray density value
            
        Note:
            Setting ray_density will clear ray_spacing to None
        """

        lambda_center = 2.99792458e8 / self._center_freq
        self._ray_spacing = np.sqrt(2) * lambda_center / value
        self._ray_density = value


    @property
    def center_freq(self):
        """
        Get the current ray density.
        
        Returns:
            float or None: Ray density value, or None if using ray_spacing
        """
        return self._center_freq

    @center_freq.setter
    def center_freq(self, value):
        """

        """

        lambda_center = 2.99792458e8 / value
        self._ray_density = np.sqrt(2) * lambda_center / self._ray_spacing
        self._center_freq = value
    


    def create_scene(self, all_actors, target_ref_actor_name=None):
        """
        Create the simulation scene with actors for radar and target reference.
        
        This method sets up the basic scene structure including:
        - Target reference actor (scene center)
        - Radar actor (observer position)
        - Initial radar positioning based on observer parameters
        
        Args:
            all_actors: Actor collection object for scene management
            target_ref_actor_name (str, optional): Name of existing target reference actor.
                                                  If None, creates a new actor.
        """
        self.all_actors = all_actors
        
        # Set up target reference actor (scene center)
        if target_ref_actor_name is None:
            self.actor_scene_ref_name = self.all_actors.add_actor()
        else:
            self.actor_scene_ref_name = target_ref_actor_name
        
        # Create radar actor (initially empty)
        self.actor_radar_name = self.all_actors.add_actor()
        
        # Calculate initial radar position based on observer parameters
        if not self.use_planewave:
            # For point source: position based on spherical coordinates
            self.radar_pos = self.convert_az_el_dist_to_xyz()
        else:
            # For planewave: position at scene center
            self.radar_pos = self.target_center

    def intialize_solver(self):
        """
        Initialize the radar solver with waveform, antenna, and simulation parameters.
        
        This method performs the complete solver setup including:
        - Waveform configuration for radar operation
        - Antenna device setup with transmitter and receiver
        - Simulation options configuration
        - Domain transforms setup for processing
        - Range-doppler response activation
        - Debug system initialization
        
        The method configures all necessary parameters for ray tracing simulation
        and prepares the system for radar response computation.
        """
        # Configure waveform based on imaging method
        if self.image_method is not None and (self.image_method == 'range_doppler' or self.image_method == 'fft'):
            output = 'RangeDoppler'
        else:
            output = 'FreqPulse'
        
        # Waveform dictionary with all radar parameters
        waveform_dict = {
            "mode": "PulsedDoppler",           # Pulsed Doppler radar mode
            "output": output,                   # Output format
            "center_freq": self.center_freq,   # Center frequency
            "bandwidth": self.bandwidth,       # Signal bandwidth
            "num_freq_samples": self.num_freqs,         # Number of frequency samples
            "cpi_duration": self.cpi_duration,          # CPI duration
            "num_pulse_CPI": self.num_pulse_CPI,        # Pulses per CPI
            "tx_multiplex": "INDIVIDUAL",               # Transmitter multiplexing
            "mode_delay": "CENTER_CHIRP"                # Timing reference
        }
        
        # Calculate derived parameters
        pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
        wavelength = 3e8 / self.center_freq
        self.mode_name = 'mode1'
        waveform = Waveform(waveform_dict)
        
        # Add single transmitter/receiver radar to the scene
        self.ant_device = add_single_tx_rx(
            self.all_actors,
            waveform,
            self.mode_name,
            parent_h_node=self.all_actors.actors[self.actor_radar_name].h_node,
            beamwidth_H=self.beam_width_h,
            beamwidth_V=self.beam_width_v,
            planewave=self.use_planewave,
            polarization=self.polarization,
            range_pixels=self.range_pixels,
            doppler_pixels=self.doppler_pixels,
            r_specs=self.r_specs,
            d_specs=self.d_specs,
            scale_pattern=200
        )

        if self.enhanced_ray_processing and int(pem_api_manager.version) <= 252:
            print("Warning: Enhanced Ray Processing is not supported in 25.2 and earlier, ignoring setting.")
        elif self.enhanced_ray_processing:
            pem.setEnhancedRayProcessing(self.ant_device.modes[self.ant_device.active_mode_name],self.max_num_refl)
        # Calculate ray spacing based on ray density

        if self.skip_terminal_bnc_po_blockage:
            pem_api_manager.isOK(pem.setPrivateKey("SkipTerminalBncPOBlockage", "true")) # we can skip terminal blockage for PO if we want
        # Configure simulation options
        sim_options = SimulationOptions(center_freq=self.ant_device.waveforms[self.mode_name].center_freq)
        # sim_options.ray_spacing = ray_spacing
        sim_options.max_reflections = self.max_num_refl
        sim_options.max_transmissions = self.max_num_trans
        sim_options.go_blockage = self.go_blockage
        sim_options.field_of_view = 180
        sim_options.bounding_box = -1
        sim_options.ray_shoot_method = self.ray_shoot_method  # Set ray shooting method
        sim_options.ray_density = self._ray_density  # Set ray density
        sim_options.max_batches = self.max_batches
        # Note: GPU device selection can be done with sim_options.gpu_device = 0
        sim_options.gpu_device = self.gpu_device
        sim_options.gpu_quota = self.gpu_quota
        sim_options.auto_configure_simulation()
        
        # Get response domains for post-processing and axis scaling
        which_mode = self.ant_device.modes[self.mode_name]
        self.ant_device.waveforms[self.mode_name].get_response_domains(which_mode)
        
        # Extract domains for each dimension
        self.vel_domain = self.ant_device.waveforms[self.mode_name].vel_domain     # Velocity domain
        self.rng_domain = self.ant_device.waveforms[self.mode_name].rng_domain     # Range domain
        self.freq_domain = self.ant_device.waveforms[self.mode_name].freq_domain   # Frequency domain
        self.pulse_domain = self.ant_device.waveforms[self.mode_name].pulse_domain # Pulse domain
        
        # Initialize domain transform objects
        self.domain_transforms_down_range = DomainTransforms(
            freq_domain=self.freq_domain,
            center_freq=self.ant_device.waveforms[self.mode_name].center_freq
        )
        self.domain_transforms_cross_range = DomainTransforms(
            center_freq=self.ant_device.waveforms[self.mode_name].center_freq,
            aspect_domain=np.linspace(0, self.azimuth_aspect_deg, self.num_pulse_CPI)
        )
        
        # Update actor coordinate systems
        self.all_actors.actors[self.actor_radar_name].coord_sys.update()
        
        # Set scene rotation based on azimuth aspect
        self.all_actors.actors[self.actor_scene_ref_name].coord_sys.ang = (
            0, 0, np.deg2rad(self.azimuth_aspect_deg / self.cpi_duration)
        )
        self.all_actors.actors[self.actor_scene_ref_name].update_rot_based_on_ang_vel = False
        self.all_actors.actors[self.actor_scene_ref_name].coord_sys.update()
        
        # Configure range-doppler response for non-planewave sources
        if not self.use_planewave:
            # Calculate vector and distance to target for image centering
            vector_to_target = np.array(self.convert_az_el_dist_to_xyz()) - np.array(self.target_center)
            vector_to_target_n = vector_to_target / np.linalg.norm(vector_to_target)
            dist_to_center = np.sqrt(vector_to_target[0]**2 + vector_to_target[1]**2 + vector_to_target[2]**2)
            
            # Set radial velocity (currently zero for rotating targets)
            radial_velocity = 0
        else:
            # For planewave sources
            dist_to_center = 0
            radial_velocity = 0
        
        # Set pixel reference point
        ref_pixel = RssPy.ImagePixelReference.MIDDLE
        
        # Activate range-doppler response if using traditional method
        if self.image_method == 'fft':
            # Calculate pixel counts with scaling factors
            num_pixels_range = self.ant_device.range_pixels
            num_pixels_doppler = self.ant_device.doppler_pixels
            
            if self.scale_factor_range_image != 1:
                num_pixels_range = int(num_pixels_range * self.scale_factor_range_image)
            if self.scale_factor_doppler_image != 1:
                num_pixels_doppler = int(num_pixels_doppler * self.scale_factor_doppler_image)
            
            # Activate range-doppler processing
            pem_api_manager.isOK(pem.activateRangeDopplerResponse(
                self.ant_device.modes[self.mode_name],
                num_pixels_range,
                num_pixels_doppler,
                ref_pixel,
                dist_to_center,
                ref_pixel,
                radial_velocity,
                self.ant_device.r_specs,
                self.ant_device.d_specs
            ))
        
        # Apply range filter if specified (not supported with planewave)
        if self.range_filter is not None and not self.use_planewave:
            if isinstance(self.range_filter, (int, float)):
                # Simple symmetric range filter
                range_min = self.range_filter / 2
                range_max = self.range_filter / 2
            else:
                # Calculate range based on domain
                center_of_range_domain = (self.rng_domain[0] + self.rng_domain[-1]) / 2
                range_min = center_of_range_domain - self.rng_domain[-1] / 2
                range_max = center_of_range_domain + self.rng_domain[-1] / 2
            
            # Create high-pass and low-pass range filters
            rngFilter_high = RssPy.RangeFilter()
            rngFilter_low = RssPy.RangeFilter()
            
            # Configure filter cutoffs
            rangeCutoff = dist_to_center - range_min
            rngFilter_high.setIdealAbrupt(RssPy.FilterPassType.HIGH_PASS, rangeCutoff)
            rangeCutoff = dist_to_center + range_max
            rngFilter_low.setIdealAbrupt(RssPy.FilterPassType.LOW_PASS, rangeCutoff)
            
            # Apply filters to the mode
            pem.addRangeFilter(self.ant_device.modes[self.mode_name], rngFilter_high)
            pem.addRangeFilter(self.ant_device.modes[self.mode_name], rngFilter_low)
        
        # Display maximum range
        print(f'Max Range: {self.rng_domain[-1]} m')
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Initialize debug systems if enabled
        if self.export_debug:
            if not self.use_planewave:
                self.debug_camera = DebuggingCamera(
                    hMode=self.ant_device.modes[self.mode_name],
                    display_mode='coatings',
                    output_size=(512, 512),
                    background_color=255,
                    frame_rate=10
                )
            self.debug_logs = DebuggingLogs(output_directory=self.output_path)
            self.debug_logs.write_scene_summary(file_name='out.json')

    def run_simulation(self, function='dB'):
        """
        Execute the radar simulation and generate SAR image.
        
        This method performs the complete simulation workflow:
        - Updates radar and target positions
        - Computes radar response using ray tracing
        - Applies selected SAR imaging algorithm
        - Returns processed SAR image
        
        Args:
            function (str): Mathematical function to apply to image data.
                          Options: 'dB', 'abs', 'real', 'imag', or other supported functions
        
        Returns:
            numpy.ndarray: Processed SAR image as 2D array
        """
        # Calculate current radar position and orientation
        pos = self.convert_az_el_dist_to_xyz()
        rot = look_at(pos, self.target_center, correct_rotation_matrix=False)
        
        # Override position for planewave mode
        if self.use_planewave:
            pos = self.target_center
        
        # Update radar actor position and orientation
        self.all_actors.actors[self.actor_radar_name].coord_sys.pos = pos
        self.all_actors.actors[self.actor_radar_name].coord_sys.rot = rot
        self.all_actors.actors[self.actor_radar_name].coord_sys.update()
        
        # Update scene rotation speed if azimuth aspect has changed
        self.all_actors.actors[self.actor_scene_ref_name].coord_sys.ang = (
            0, 0, np.deg2rad(self.azimuth_aspect_deg / self.cpi_duration)
        )
        self.all_actors.actors[self.actor_scene_ref_name].coord_sys.update()
        
        # Calculate azimuth domain for synthetic aperture
        phi = self.azimuth_observer_deg
        phi_domain = np.linspace(phi, phi + self.azimuth_aspect_deg, self.num_pulse_CPI)
        
        # Display imaging parameters on first run or when aspect changes
        # if self.number_of_runs == 0 or self.previous_aspect_deg != self.azimuth_aspect_deg:
        #     self.domain_transforms_cross_range.aspect_angle = self.azimuth_aspect_deg
        #     self.domain_transforms_cross_range.num_aspect_angle = self.num_pulse_CPI
        #     self.previous_aspect_deg = self.azimuth_aspect_deg
            
        #     print("######---->Image Parameters for Waveform<----######")
        #     print(f'Maximum Down Range: {self.domain_transforms_down_range.range_period:.2f} m')
        #     print(f'Range Resolution: {self.domain_transforms_down_range.range_resolution:.2f} m')
        #     print('------')
        #     print(f'Cross Range Window: {self.domain_transforms_cross_range.range_period:.2f} m')
        #     print(f'Cross Range Resolution: {self.domain_transforms_cross_range.range_resolution:.2f} deg')
        #     print("######--------######")
        
        # Calculate elevation domain (constant for current implementation)
        theta_domain = 90 - np.linspace(self.elevation_observer_deg, self.elevation_observer_deg, self.num_pulse_CPI)
        
        # Convert spherical coordinates to cartesian for platform motion
        xyz_domain = np.array([
            np.sin(np.deg2rad(theta_domain)) * np.cos(np.deg2rad(phi_domain)),
            np.sin(np.deg2rad(theta_domain)) * np.sin(np.deg2rad(phi_domain)),
            np.cos(np.deg2rad(theta_domain))
        ]).T
        
        # Execute radar simulation
        start_time = walltime.time()
        pem_api_manager.isOK(pem.computeResponseSync())
        self.sim_performance_time = walltime.time() - start_time
        # Determine response type based on imaging method
        if self.image_method is not None and  (self.image_method == 'fft' or self.image_method == 'range_doppler'):
            response_type = RssPy.ResponseType.RANGE_DOPPLER
        else:
            response_type = RssPy.ResponseType.FREQ_PULSE
        
        # Retrieve simulation response
        (ret, response) = pem.retrieveResponse(self.ant_device.modes[self.mode_name], response_type)
        
        # Extract response data (format: [tx_idx, rx_idx, pulse_idx, freq_idx])
        response = np.array(response[0, 0])
        
        # Process image based on selected method
        if self.image_method is None:
            return apply_math_function(response, function=function)
        elif self.image_method is not None and (self.image_method != 'fft' and self.image_method != 'range_doppler'):
            # Advanced SAR imaging algorithms (experimental)
            
            # Standardize function parameter
            if function.lower() == "db":
                function = 'db'
            else:
                function = 'abs'
            
            # Create platform and image plane dictionaries
            platform = platform_dict(xyz_domain, self.freq_domain, self.bandwidth, 
                                    self.num_freqs, self.num_pulse_CPI)
            self.image_plane_dict = img_plane_dict(platform, res_factor=1.0, 
                                                 n_hat=np.array([0, 0, 1]), aspect=1, upsample=True)
            
            # Initialize SAR image processor
            processor = SARImageProcessor(phase_history=response, platform_params=platform,
                                        image_plane=self.image_plane_dict)
            
            # Generate image using specified algorithm
            if 'polar_format' in self.image_method.lower():
                image = processor.process_image('polar_format')
            elif 'omega_k' in self.image_method.lower():
                image = processor.process_image('omega_k')
            elif 'backprojection' in self.image_method.lower():
                image = processor.process_image('backprojection')
            else:
                image = processor.process_image('isar')
            
            # Apply mathematical function to processed image
            sar_image = apply_math_function(image, function=function)
        else:
            # Traditional range-doppler processing
            sar_image = apply_math_function(response, function=function)
        
        # Apply image scaling if specified
        if self.scale_factor_range_image != 1:
            mid_pixel_range = int(sar_image.shape[1] / 2)
            num_pixels_range_offset = int(sar_image.shape[1] / self.scale_factor_range_image / 2)
            sar_image = sar_image[:, mid_pixel_range - num_pixels_range_offset:
                                    mid_pixel_range + num_pixels_range_offset]
        
        if self.scale_factor_doppler_image != 1:
            mid_pixel_doppler = int(sar_image.shape[0] / 2)
            num_pixels_doppler_offset = int(sar_image.shape[0] / self.scale_factor_doppler_image / 2)
            sar_image = sar_image[mid_pixel_doppler - num_pixels_doppler_offset:
                                mid_pixel_doppler + num_pixels_doppler_offset, :]
        
        # Generate debug output if enabled
        if self.export_debug:
            self.debug_logs.write_scene_summary(file_name='out.json')
            if not self.use_planewave:
                self.debug_camera.generate_image()
        
        # Increment run counter
        self.number_of_runs += 1
        
        return sar_image

    def export_debug_camera(self):
        """
        Export debug camera views as animated GIF.
        
        This method generates a GIF animation from the debug camera views
        captured during simulation runs. Only available when debug export
        is enabled and not using planewave mode.
        """
        if self.export_debug and not self.use_planewave:
            self.debug_camera.write_camera_to_gif(
                file_name=os.path.join(self.output_path, 'camera.gif')
            )

    def convert_az_el_dist_to_xyz(self):
        """
        Convert spherical coordinates (azimuth, elevation, distance) to Cartesian coordinates.
        
        This method transforms the observer position from spherical coordinates
        relative to the target center into absolute Cartesian coordinates.
        
        Returns:
            numpy.ndarray: XYZ coordinates as [x, y, z] array in meters
        
        Note:
            - Azimuth: 0° = +X axis, increases counter-clockwise when viewed from +Z
            - Elevation: 0° = XY plane, 90° = +Z axis
            - Distance: Radial distance from target center
        """
        if self.distance_observer_m==0:
            self.distance_observer_m = 1e-6  # Avoid multiplication by zero
        x = (self.distance_observer_m * np.cos(np.deg2rad(self.azimuth_observer_deg)) * 
             np.cos(np.deg2rad(self.elevation_observer_deg)) + self.target_center[0])
        y = (self.distance_observer_m * np.sin(np.deg2rad(self.azimuth_observer_deg)) * 
             np.cos(np.deg2rad(self.elevation_observer_deg)) + self.target_center[1])
        z = (self.distance_observer_m * np.sin(np.deg2rad(self.elevation_observer_deg)) + 
             self.target_center[2])
        return np.array([x, y, z])

    def generate_equal_space_samples_uv(self, phi_spacing_deg=5, theta_spacing_deg=10, 
                                       theta_start=0, theta_stop=90, show_plot=False):
        """
        Generate uniformly spaced points on a unit sphere using standard spherical coordinates.
        
        This method creates observation points for multi-static radar simulations or
        coverage analysis by sampling the sphere at regular angular intervals.
        
        Args:
            phi_spacing_deg (float): Spacing in degrees for azimuthal angle (around equator)
            theta_spacing_deg (float): Spacing in degrees for polar angle (from pole)
            theta_start (float): Starting polar angle in degrees (0° at North pole)
            theta_stop (float): Ending polar angle in degrees (90° at equator, 180° at South pole)
            show_plot (bool): Whether to display 3D visualization of points
        
        Returns:
            tuple: (points, point_az_el) where:
                - points: XYZ coordinates of points on sphere (N×3 array)
                - point_az_el: Same points in azimuth/elevation format (M×2 array)
        
        Note:
            Points at exactly 90° elevation are excluded from point_az_el to avoid
            singularities in azimuth calculation.
        """
        # Generate angular grids
        points = []
        phi = np.linspace(0, 360, num=int(360 / phi_spacing_deg) + 1)
        theta = np.linspace(theta_start, theta_stop, 
                          num=max(int((theta_stop - theta_start) / theta_spacing_deg) + 1, 1))
        
        # Convert spherical to Cartesian coordinates
        for p in phi:
            for t in theta:
                x = np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(p))
                y = np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(p))
                z = np.cos(np.deg2rad(t))
                points.append([x, y, z])
        points = np.array(points)
        
        # Convert to azimuth/elevation format for simulation use
        point_az_el = []
        for pt in points:
            x, y, z = pt
            azimuth = np.rad2deg(np.arctan2(y, x))
            elevation = np.rad2deg(np.arcsin(z))
            
            # Exclude points at zenith to avoid azimuth singularity
            if not np.isclose(elevation, 90.0):
                point_az_el.append([azimuth, elevation])
        point_az_el = np.array(point_az_el)
        
        # Generate visualization if requested
        if show_plot:
            self._plot_points(points)
        
        return points, point_az_el

    def generate_equal_space_samples_spherical(self, num_samples, theta_start=0, theta_stop=90, 
                                             show_plot=False):
        """
        Generate uniformly distributed points on sphere using Fibonacci lattice sampling.
        
        This method provides better uniformity than regular grid sampling by using
        the Fibonacci spiral pattern, which minimizes clustering and gaps.
        
        Args:
            num_samples (int): Number of points to generate
            theta_start (float): Starting polar angle in degrees (0° is North pole)
            theta_stop (float): Ending polar angle in degrees (180° is South pole)
            show_plot (bool): Whether to display 3D visualization of points
        
        Returns:
            tuple: (points, point_az_el) where:
                - points: XYZ coordinates of points on sphere (N×3 array)  
                - point_az_el: Same points in azimuth/elevation format (N×2 array)
        
        Note:
            The Fibonacci lattice provides optimal uniform distribution for
            arbitrary numbers of points, avoiding the clustering issues
            common with regular angular grids.
        """
        # Convert theta range to z-coordinate range
        z_start = np.cos(np.deg2rad(theta_stop))
        z_stop = np.cos(np.deg2rad(theta_start))
        
        # Calculate sampling density adjustment
        theta_range_ratio = (theta_stop - theta_start) / 180.0
        adjusted_num_samples = int(num_samples / theta_range_ratio)
        
        # Use Fibonacci lattice for uniform sampling
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
        
        for i in range(adjusted_num_samples):
            # Calculate z coordinate (normalized height)
            z = 1 - (i / float(adjusted_num_samples - 1)) * 2
            
            # Check if point falls within desired theta range
            if z_start <= z <= z_stop:
                radius = np.sqrt(1 - z * z)  # Radius at this z level
                theta = phi * i  # Golden angle increment
                
                # Convert to Cartesian coordinates
                x = np.cos(theta) * radius
                y = np.sin(theta) * radius
                points.append([x, y, z])
                
                # Stop when we have enough points
                if len(points) >= num_samples:
                    break
        
        points = np.array(points)
        
        # Convert to azimuth/elevation format for simulation use
        point_az_el = []
        for pt in points:
            x, y, z = pt
            azimuth = np.rad2deg(np.arctan2(y, x))
            elevation = np.rad2deg(np.arcsin(z))
            point_az_el.append([azimuth, elevation])
        point_az_el = np.array(point_az_el)
        
        # Generate visualization if requested
        if show_plot:
            self._plot_points(points)
        
        return points, point_az_el

    def _plot_points(self, points_xyz):
        """
        Plot 3D visualization of sampling points on unit sphere.
        
        This private method creates a 3D scatter plot showing the distribution
        of observation points on a unit sphere with a wireframe reference.
        
        Args:
            points_xyz (numpy.ndarray): XYZ coordinates of points to plot (N×3 array)
        
        Note:
            This method is used internally by the sampling methods when
            show_plot=True is specified.
        """
        # Convert to spherical coordinates for reference (currently unused)
        r = np.ones(points_xyz.shape[0])
        theta = np.arccos(points_xyz[:, 1])  # Polar angle
        phi = np.arctan2(points_xyz[:, 2], points_xyz[:, 0])  # Azimuthal angle
        
        # Create 3D plot
        fig = plt.figure(figsize=(6, 6))
        ax2 = fig.add_subplot(111, projection='3d')
        
        # Plot sampling points
        ax2.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], 
                   c='blue', s=15, label='Sampling Points')
        
        # Add wireframe reference sphere for context
        u_sphere = np.linspace(0, 2 * np.pi, 20)
        v_sphere = np.linspace(0, np.pi, 10)
        u_sphere, v_sphere = np.meshgrid(u_sphere, v_sphere)
        x_sphere = np.sin(v_sphere) * np.cos(u_sphere)
        y_sphere = np.sin(v_sphere) * np.sin(u_sphere)
        z_sphere = np.cos(v_sphere)
        ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
        
        # Configure plot appearance
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Uniform Sampling Observation Points')
        
        # Set equal aspect ratio for proper sphere visualization
        ax2.set_box_aspect([
            (np.max(x_sphere) - np.min(x_sphere)) / 2,
            (np.max(y_sphere) - np.min(y_sphere)) / 2,
            (np.max(z_sphere) - np.min(z_sphere)) / 2
        ])
        
        plt.tight_layout()
        plt.show()

