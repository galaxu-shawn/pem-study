# -*- coding: utf-8 -*-
"""
SAR Imaging of Multiple Ships with Ocean Surface

This script generates Synthetic Aperture Radar (SAR) images of multiple ships
positioned on a dynamic ocean surface. Each ship can be configured with:
- Position, heading, and velocity
- Ship draft (depth below waterline)
- Wake generation

The script supports:
- Multiple ship configurations in a single scene
- Ocean surface with configurable wave properties
- Ship wake generation and visualization
- Time-series animation output
- Interactive plot with adjustable dynamic range

@author: asligar
Created on Mon Mar 29 10:51:32 2021
"""

# Standard library imports
import os

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm

# PEM utilities - Core functionality
from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API

# PEM utilities - SAR and domain transforms
from pem_utilities.domain_transforms import DomainTransforms
from pem_utilities.sar_setup_backend import SAR_Setup_Backend

# PEM utilities - Scene setup
from pem_utilities.actor import Actors
from pem_utilities.materials import MaterialManager, MatData
from pem_utilities.rotation import euler_to_rot
from pem_utilities.mesh_utilities import get_z_elevation_from_mesh

# PEM utilities - Ocean surface and visualization
from pem_utilities.seastate3 import OceanPresets, Wake
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.post_processing import DynamicImageViewer


# ============================================================================
# Initialize API and Paths
# ============================================================================
paths = get_repo_paths()
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem
RssPy = pem_api_manager.RssPy 

# ============================================================================
# CONFIGURATION: Ship Definitions
# ============================================================================
# Each ship is defined with the following parameters:
# - filename: STL model filename from models/Ships folder
# - ship_draft: Depth below waterline (meters) - controls vertical position
# - velocity: Ship speed (m/s) - affects wake generation
# - position: (x, y) coordinates in scene (meters)
# - heading: Ship orientation (degrees, 0=North, 90=East)

all_ships = [
    {"filename": "cargoship.stl", "ship_draft": 30.0, "velocity": 9.0, 
     "position": (0, 0), "heading": 45},
    {"filename": "explorer_ship_meter.stl", "ship_draft": 8.0, "velocity": 12.0, 
     "position": (200, 0), "heading": 90},
    {"filename": "fishing_boat.stl", "ship_draft": 2.5, "velocity": 4, 
     "position": (200, 200), "heading": 10},
    {"filename": "HMS_Daring_Type_45.stl", "ship_draft": 10.5, "velocity": 8.0, 
     "position": (50, -300), "heading": 135},
    {"filename": "mo1_hp_reduced.stl", "ship_draft": 1.5, "velocity": 8.0, 
     "position": (-100, 200), "heading": -45},
    {"filename": "power_yacht.stl", "ship_draft": 1.25, "velocity": 10, 
     "position": (400, 0), "heading": -90},
    {"filename": "sail_boat_38ft.stl", "ship_draft": 1.0, "velocity": 2.4, 
     "position": (400, 400), "heading": 75},
    {"filename": "Tanker_Fixed.stl", "ship_draft": 15, "velocity": 8, 
     "position": (-200, 300), "heading": 195},
    {"filename": "Trawler_Fixed.stl", "ship_draft": 5, "velocity": 3.5, 
     "position": (-400, 400), "heading": -20}
]

# Select which ships to include in the simulation
# Examples: all_ships[:3] for first 3 ships, all_ships[:1] for just one ship
which_ships = all_ships # expects a list

# ============================================================================
# CONFIGURATION: Radar Parameters
# ============================================================================
center_freq = 10e9  # Hz (10 GHz X-band radar)

# Ray tracing settings0
ray_density = 0.1  # Rays per wavelength (higher = more accurate but slower)
ray_shoot_method = 'sbr'  # 'grid' or 'sbr' (Shooting and Bouncing Rays)
                          # Note: 'sbr' is beta in 25.2, fully supported in 26.1+

# SAR image domain parameters
desired_max_range = 1000        # Maximum range extent (meters)
desired_max_cross_range = 1000  # Maximum cross-range extent (meters)
resolution = 1.0                # Spatial resolution (meters) for both domains

# Output image size
output_size_num_pixels_range = 1024         # Pixels in range dimension
output_size_num_pixels_cross_range = 1024   # Pixels in cross-range dimension


# ============================================================================
# CONFIGURATION: Ocean Surface Settings
# ============================================================================
include_ocean = True  # Set to False to disable ocean surface
include_wake = True   # Set to False to disable ship wake generation

# Ocean mesh parameters
num_grid = 500       # Grid resolution (number of points per dimension)
scene_length = 1000  # Physical size of ocean patch (meters)

# ============================================================================
# Compute SAR Waveform Parameters
# ============================================================================
# Calculate the required bandwidth and aspect angles needed to achieve
# the desired range and cross-range resolution

# Define spatial domains
num_range_bins = int(desired_max_range / resolution)
num_cross_range_bins = int(desired_max_cross_range / resolution)

range_domain = np.linspace(0, resolution * (num_range_bins - 1), num=num_range_bins)
crossrange_domain = np.linspace(0, resolution * (num_cross_range_bins - 1), 
                                num=num_cross_range_bins)

# Initialize domain transform objects to compute waveform parameters
dt_down_range = DomainTransforms(range_domain=range_domain, center_freq=center_freq)
dt_cross_range = DomainTransforms(range_domain=crossrange_domain, center_freq=center_freq)

# Extract computed waveform parameters
bandwidth = dt_down_range.bandwidth
num_freqs = dt_down_range.num_freq
aspect_ang_phi = dt_cross_range.aspect_angle
num_aspect_angle = dt_cross_range.num_aspect_angle

# Display computed parameters
print(f'\n{"="*70}')
print(f'Computed SAR Waveform Parameters')
print(f'{"="*70}')
print(f'Target Resolution: {resolution} m (range and cross-range)')
print(f'Max Range: {desired_max_range} m')
print(f'Max Cross-Range: {desired_max_cross_range} m')
print(f'\nWaveform Configuration:')
print(f'  Bandwidth: {bandwidth/1e6:.2f} MHz')
print(f'  Number of Frequencies: {num_freqs}')
print(f'  Aspect Angle Span: {aspect_ang_phi:.2f} deg')
print(f'  Number of Aspect Angles: {num_aspect_angle}')
print(f'{"="*70}\n')


# ============================================================================
# Setup Materials
# ============================================================================
mat_manager = MaterialManager()

# Create custom seawater material with surface roughness
# These parameters define the electromagnetic properties of seawater and
# its surface roughness for realistic radar backscatter simulation
height_standard_dev = 17  # Surface roughness height std dev (mm)
corr_length = 0.05        # Correlation length (meters)
roughness = height_standard_dev * 1e-3 / corr_length  # Dimensionless roughness

seawater_material = MatData.from_dict({
    "thickness": -1,              # -1 = infinite thickness (half-space)
    "relEpsReal": 40,             # Real part of relative permittivity
    "relEpsImag": 0.0,            # Imaginary part of relative permittivity
    "conductivity": 1.0,          # Conductivity (S/m)
    "height_standard_dev": height_standard_dev,
    "roughness": roughness
})
mat_manager.create_material('my_seawater', seawater_material)


# ============================================================================
# Initialize Scene and Actors
# ============================================================================
# Create actor container for organizing all scene elements
all_actors = Actors()

# Create an empty reference actor for the scene
# All other actors will be positioned relative to this reference, making
# it easy to rotate or reposition the entire scene without changing individual actors
actor_scene_ref_name = all_actors.add_actor()


# ============================================================================
# Create Ocean Surface
# ============================================================================
if include_ocean:
    # Generate ocean surface using Beaufort scale preset
    # Beaufort 6 = Strong breeze (significant wave height ~3-4m)
    # 	wind 11–14m/s with large waves beginning to form, many whitecaps everywhere, and more spray
    ocean = OceanPresets.from_beaufort(4, num_grid=num_grid, scene_length=scene_length)
    ocean.wind_direction = (0.8, 0.2)  # Set wind direction vector
    
    # Alternative: Create custom ocean configuration. Below allows fine-tuning
    # and custom wave components. Using OceanPresets is simpler for common sea states.
    # Uncomment and modify these lines for custom ocean parameters:
    #
    # from pem_utilities.seastate3 import OceanSurface, OceanConfig
    # ocean_config = OceanConfig(num_grid=num_grid, scene_length=scene_length)
    # ocean_config.wind_speed = 12.0
    # ocean_config.wave_amplitude = 1.5
    # ocean_config.wind_direction = (0.8, 0.2)
    # ocean_config.choppiness = 0.35
    # ocean_config.random_seed = 11
    # ocean = OceanSurface(ocean_config=ocean_config)
    

    # Add custom swell components:
    ocean.add_swell(amplitude=0.5, wavelength=120, direction=(0.6, 0.4))

# ============================================================================
# Load and Position Ships
# ============================================================================
model_path = os.path.join(paths.models, 'Ships')

# Dictionary to store ship properties for later use
all_ship_dict = {}

print(f'\nLoading {len(which_ships)} ship(s)...')
for ship_config in which_ships:
    # Extract ship configuration parameters
    ship_file = os.path.join(model_path, ship_config['filename'])
    ship_velocity = ship_config['velocity']
    ship_orientation = ship_config['heading']
    ship_position = ship_config['position']
    ship_draft = ship_config['ship_draft']
    
    # Add ship actor to the scene
    # mat_idx=0 indicates Perfect Electric Conductor (PEC)
    ship_name = all_actors.add_actor(
        name='ship',
        filename=ship_file,
        color=(0.5, 0.5, 0.5),
        mat_idx=0,  # PEC material
        parent_h_node=all_actors.actors[actor_scene_ref_name].h_node # reference to our scene actor (which is just a easy reference to the origin)
    )
    
    # Position the ship
    # Still ocean surface is at z=0; draft moves ship below waterline, 
    # we can adjust this position later to account for wave amplitude
    all_actors.actors[ship_name].coord_sys.pos = [
        ship_position[0], 
        ship_position[1], 
        -ship_draft
    ]
    
    # Set ship orientation (rotation around z-axis)
    all_actors.actors[ship_name].coord_sys.rot = euler_to_rot(
        phi=ship_orientation, 
        theta=0, 
        psi=0
    )
    # all_actors.actors[ship_name].velocity_mag = ship_velocity
    all_actors.actors[ship_name].coord_sys.update()
    
    # Extract ship dimensions from mesh bounds
    ship_mesh = all_actors.actors[ship_name].get_mesh()[0]
    ship_length = ship_mesh.bounds[1] - ship_mesh.bounds[0]  # X-dimension
    ship_width = ship_mesh.bounds[3] - ship_mesh.bounds[2]   # Y-dimension
    
    # Store ship properties for dynamic positioning and wake generation
    all_ship_dict[ship_name] = {
        'velocity': ship_velocity,
        'position': ship_position,
        'orientation': ship_orientation,
        'draft': ship_draft,
        'length': ship_length,
        'width': ship_width
    }
    
    # Generate ship wake if ocean is enabled
    if include_ocean and include_wake:
        wake = Wake.create(
            ship_length=ship_length,
            ship_velocity=ship_velocity,
            scene_length=scene_length,
            num_grid=num_grid,
            ship_position=ship_position,
            ship_heading=ship_orientation,
            beam_ship=ship_width,
            draft_ship=ship_draft
        )
        wake.set_static(True)  # Static wake (not time-evolving)
        ocean.add_wake(wake)
    
    print(f'  Loaded: {ship_config["filename"]} at {ship_position}')


# Add ocean surface actor to the scene. As a genererator actor, calls to update_actor()
# with time value will generate a new mesh. i.e. --> all_actors.actors[ocean_name].update_actor(time=time)
ocean_name = all_actors.add_actor(
    name='ocean',
    generator=ocean,
    mat_idx=mat_manager.get_index('my_seawater'),
    parent_h_node=all_actors.actors[actor_scene_ref_name].h_node
)


# ============================================================================
# Configure SAR System
# ============================================================================
# The SAR system is configured with a fixed azimuth (radar stays in one position)
# and the scene is rotated to generate different aspect angles. This approach
# ensures the square ocean patch remains properly aligned without edge effects.

sar = SAR_Setup_Backend(
    azimuth_observer_deg=0,          # Intial Fixed radar azimuth position (can be updated with sar.azimuth_observer_deg=X or sar.elevation_observer_deg=K)
    elevation_observer_deg=30,       # Radar elevation angle (grazing angle)
    use_planewave=True,              # Use plane wave illumination
    azimuth_aspect_deg=aspect_ang_phi  # Angular span for synthetic aperture
)

# Create the scene with the configured actors
sar.create_scene(all_actors, target_ref_actor_name=actor_scene_ref_name)

# Output and simulation settings
sar.output_path = paths.output

# Ray tracing parameters
sar.go_blockage = -1              # -1 = no GO blockage; >=0 for GO blockage
sar.max_num_refl = 1              # Maximum number of reflections per ray
sar.max_num_trans = 1             # Maximum number of transmissions per ray
sar.ray_density = ray_density
sar.ray_shoot_method = ray_shoot_method
sar.skip_terminal_bnc_po_blockage = True  # Skip terminal blockage for PO to speed up simulation
# Radar waveform parameters
sar.center_freq = center_freq
sar.num_freqs = num_freqs
sar.bandwidth = bandwidth
sar.num_pulse_CPI = num_aspect_angle  # Coherent Processing Interval (CPI) pulse count
sar.gpu_device=0
sar.gpu_quota=0.95
# Output image dimensions
sar.range_pixels = output_size_num_pixels_range
sar.doppler_pixels = output_size_num_pixels_cross_range

# Initialize the solver
sar.intialize_solver()
print('\nSAR system initialized successfully.')

# ============================================================================
# Compute Ship Rocking Parameters Based on Sea State
# ============================================================================
if include_ocean:
    # Extract wave parameters from ocean configuration
    # Wave period is estimated from dominant wavelength and ocean properties
    dominant_wavelength = ocean.scene_length / 20  # Estimate from scene
    wave_speed = np.sqrt(9.81 * dominant_wavelength / (2 * np.pi))  # Deep water dispersion
    wave_period = dominant_wavelength / wave_speed
    
    # Beaufort 3 parameters (from the ocean preset used)
    # Significant wave height for Beaufort 3 is approximately 0.5-1.25m
    estimated_wave_height = 0.8  # meters (typical for Beaufort 3)
    
    # Wind direction vector (normalized)
    wind_dir = np.array(ocean.wind_direction)
    wind_dir = wind_dir / np.linalg.norm(wind_dir)
    wind_angle = np.arctan2(wind_dir[1], wind_dir[0])  # Angle in radians
    
    print(f'\n{"="*70}')
    print(f'Ship Rocking Parameters')
    print(f'{"="*70}')
    print(f'Sea State: Beaufort 3')
    print(f'Estimated Wave Height: {estimated_wave_height:.2f} m')
    print(f'Wave Period: {wave_period:.2f} s')
    print(f'Wind Direction: {np.rad2deg(wind_angle):.1f} deg')
    print(f'{"="*70}\n')
    
    # Calculate angular velocity for each ship
    for ship_name in all_ship_dict.keys():
        ship_heading = all_ship_dict[ship_name]['orientation']  # degrees
        ship_length = all_ship_dict[ship_name]['length']
        ship_width = all_ship_dict[ship_name]['width']
        
        # Convert ship heading to radians
        ship_heading_rad = np.deg2rad(ship_heading)
        
        # Calculate relative angle between wave direction and ship heading
        relative_angle = wind_angle - ship_heading_rad
        
        # Roll motion (rocking along ship's longitudinal axis - X in ship frame)
        # Maximum when waves hit ship from the side (beam seas)
        roll_factor = np.abs(np.sin(relative_angle))
        
        # Pitch motion (rocking along ship's transverse axis - Y in ship frame)
        # Maximum when waves hit ship from front/back (head/following seas)
        pitch_factor = np.abs(np.cos(relative_angle))
        
        # Roll amplitude depends on ship beam (width) and wave height
        # Smaller ships and beam seas cause more roll
        roll_amplitude_deg = (estimated_wave_height / ship_width) * 15 * roll_factor  # degrees
        
        # Pitch amplitude depends on ship length and wave period
        # Shorter ships in head seas pitch more
        pitch_amplitude_deg = (estimated_wave_height / ship_length) * 10 * pitch_factor  # degrees
        
        # Angular velocity in radians/sec (sinusoidal motion)
        # For sinusoidal motion: theta(t) = A * sin(omega * t)
        # Angular velocity: omega_angular = A * omega * cos(omega * t)
        # Maximum angular velocity = A * omega where omega = 2*pi/T
        omega = 2 * np.pi / wave_period
        
        # Convert amplitudes to radians
        roll_amplitude_rad = np.deg2rad(roll_amplitude_deg)
        pitch_amplitude_rad = np.deg2rad(pitch_amplitude_deg)
        
        # Store rocking parameters for time-based calculation
        all_ship_dict[ship_name]['roll_amplitude'] = roll_amplitude_rad
        all_ship_dict[ship_name]['pitch_amplitude'] = pitch_amplitude_rad
        all_ship_dict[ship_name]['wave_omega'] = omega
        all_ship_dict[ship_name]['relative_wave_angle'] = relative_angle
        
        print(f'Ship: {ship_name}')
        print(f'  Heading: {ship_heading:.1f} deg')
        print(f'  Relative Wave Angle: {np.rad2deg(relative_angle):.1f} deg')
        print(f'  Roll Amplitude: {roll_amplitude_deg:.2f} deg')
        print(f'  Pitch Amplitude: {pitch_amplitude_deg:.2f} deg')
        print(f'  Max Roll Rate: {roll_amplitude_rad * omega:.4f} rad/s')
        print(f'  Max Pitch Rate: {pitch_amplitude_rad * omega:.4f} rad/s\n')

# ============================================================================
# Setup Visualization and Animation
# ============================================================================
# Configure visualization parameters
output_movie_name = os.path.join(sar.output_path, 'out_vis_seastate.mp4')
fps = 0.1  # Frames per second for output video
stop_time = 280  # Total simulation time (seconds)

# Initialize 3D visualization with SAR image display (only required for simulation, not used in simulation)
modeler = ModelVisualization(
    all_actors,
    show_antennas=False,
    fps=fps,
    output_movie_name=output_movie_name,
    figure_size=(0.4, 0.4), # from 1.0 (full screen) to smaller sizes of the figure displayed in upper right of modeler
    shape=(len(sar.rng_domain), len(sar.vel_domain)),
    cmap='Greys_r'  # Grayscale colormap for SAR image
)

# Add visualization enhancements
modeler.pl.add_axes_at_origin(labels_off=True)
modeler.pl.show_grid()

# ============================================================================
# Run Time-Series Simulation
# In this example we are just going to animate the ocean surface waves
# ship positions are updated based on ocean surface elevation but constant in xy
# we could add ship motion later if desired
# ============================================================================
print(f'\n{"="*70}')
print(f'Running time-series simulation...')
print(f'Duration: {stop_time} seconds at {fps} fps ({int(stop_time*fps)} frames)')
print(f'{"="*70}\n')
all_images = []
time_stamps = np.linspace(0, stop_time, num=int(stop_time * fps))
for time in tqdm(time_stamps, desc='Processing frames'):
    # Update ocean surface for current time (this is the only thing that changes in our scene. the ships are constant xy location, only z changes with ocean waves)
    if include_ocean:
        all_actors.actors[ocean_name].update_actor(time=time)
        ocean_mesh = all_actors.actors[ocean_name].get_mesh()[0]
    
        # Update each ship's vertical position and angular velocity to match ocean surface
        for ship_name in all_ship_dict.keys():
            # Get ship properties
            ship_draft = all_ship_dict[ship_name]['draft']
            xy = all_actors.actors[ship_name].coord_sys.pos[:2]
            
            # Get ocean surface elevation at ship position
            minmax_location = get_z_elevation_from_mesh(xy, ocean_mesh, return_min_and_max=True)
            max_z = -ship_draft
            if minmax_location[1] is not None:
                max_z = np.max(minmax_location) - ship_draft
            
            # Calculate time-varying angular velocity for ship rocking
            # Ships rock with sinusoidal motion based on wave period
            omega = all_ship_dict[ship_name]['wave_omega']
            roll_amp = all_ship_dict[ship_name]['roll_amplitude']
            pitch_amp = all_ship_dict[ship_name]['pitch_amplitude']
            
            # Angular velocity for sinusoidal motion: omega_angular = A * omega * cos(omega * t)
            # Roll is around ship's X-axis (longitudinal), Pitch is around Y-axis (transverse)
            roll_angular_vel = roll_amp * omega * np.cos(omega * time)
            pitch_angular_vel = pitch_amp * omega * np.cos(omega * time)
            
            # Transform angular velocities to global frame based on ship's heading
            ship_heading_rad = np.deg2rad(all_ship_dict[ship_name]['orientation'])
            cos_h = np.cos(ship_heading_rad)
            sin_h = np.sin(ship_heading_rad)
            
            # Rotate angular velocity vector from ship frame to global frame
            # In ship frame: [roll_vel, pitch_vel, 0] (X=roll, Y=pitch)
            # Transform to global frame
            ang_x_global = roll_angular_vel * cos_h - pitch_angular_vel * sin_h
            ang_y_global = roll_angular_vel * sin_h + pitch_angular_vel * cos_h
            
            # Update ship position to float on ocean surface (estimated based on center xy location and considering draft)
            all_actors.actors[ship_name].coord_sys.pos = [xy[0], xy[1], max_z]
            
            # Set angular velocity for rocking motion
            all_actors.actors[ship_name].coord_sys.ang = [ang_x_global, ang_y_global, 0]
            
            all_actors.actors[ship_name].coord_sys.update()

            # all_actors.actors[ship_name].update_actor(time=time)
    
    # Run SAR simulation and generate image
    image = sar.run_simulation(function='complex')
    all_images.append(image)
    # Convert to dB scale for visualization
    image_db = 20 * np.log10(np.abs(image) + 1e-12)
    data_max = np.max(image_db)
    
    # Update visualization frame with 100 dB dynamic range
    modeler.update_frame(
        plot_data=image_db.T,
        plot_limits=[data_max - 100, data_max],
        write_frame=True
    )

# Close visualization and save video
modeler.close()
print(f'\nAnimation saved to: {output_movie_name}\n')

# ============================================================================
# Interactive SAR Image Viewer
# ============================================================================
# Display the final SAR image with interactive dynamic range adjustment
all_images = np.array(all_images)
# average over time frames to get a single image
image = np.mean(all_images, axis=0)

img_display = DynamicImageViewer(image=image,math_function='db',cmap='Greys_r')
img_display.save(os.path.join(sar.output_path, 'sar_image_final.png'))

