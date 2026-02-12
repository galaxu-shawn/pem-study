# -*- coding: utf-8 -*-
"""
SAR Imaging of  hips 

This script generates Synthetic Aperture Radar (SAR) images of multiple ships
. Each ship can be configured with:
- Position, heading, and velocity

The script supports:
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
which_ship = all_ships[0] # expects a list

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
# Load and Position Ships
# ============================================================================
model_path = os.path.join(paths.models, 'Ships')

# Dictionary to store ship properties for later use
all_ship_dict = {}



# Extract ship configuration parameters
ship_file = os.path.join(model_path, which_ship['filename'])
ship_velocity = which_ship['velocity']
ship_orientation = which_ship['heading']
ship_position = which_ship['position']
ship_draft = which_ship['ship_draft']

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


print(f'  Loaded: {which_ship["filename"]} at {ship_position}')




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
# Run Simulation
# ============================================================================
print(f'\n{"="*70}')
print(f'Duration: {stop_time} seconds at {fps} fps ({int(stop_time*fps)} frames)')
print(f'{"="*70}\n')
all_images = []

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

