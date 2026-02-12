"""
Tap Delay Line Generation Example

This example demonstrates:
1. Creating a point-to-point (P2P) wireless communication scenario
2. Using diversity antennas (multiple TX and RX antenna elements)
3. Applying tap delay line filtering to channel impulse responses
4. Comparing different filtering methods (power threshold vs. fixed tap count)
5. Visualizing results across multiple channels, pulses, and time steps

The simulation models a moving receiver with diversity antennas receiving
signals from a stationary transmitter in a cluttered environment with multiple
reflectors (planes, cubes, spheres).

Copyright ANSYS. All rights reserved.
"""

#######################################
# IMPORTS
#######################################
import copy
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import pyvista as pv
import scipy
from tqdm import tqdm
from scipy.interpolate import interp1d
from PIL import Image

# PEM Utilities
from pem_utilities.tap_delay import TapDelayLine
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_rx, add_single_tx, add_diversity_antenna_pair, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import Plane, Cube, Sphere
from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API

#######################################
# CONFIGURATION PARAMETERS
#######################################
# Animation settings
FPS = 10
TOTAL_TIME = 1
DT = 1 / FPS
NUM_FRAMES = int(TOTAL_TIME / DT)

# Waveform parameters
CENTER_FREQ = 3.85e9
NUM_FREQS = 512
BANDWIDTH = 300e6
CPI_DURATION = 100e-3
NUM_PULSE_CPI = 100
INPUT_POWER_WATTS = 1
TX_MULTIPLEX = 'simultaneous'  # 'simultaneous' or 'interleaved', interleave will split the pulses across TX antennas ie. pulse 0 to TX 0, pulse 1 to TX 1, etc.

# Simulation parameters
GO_BLOCKAGE = 1  # Set to -1 for no GO blockage, 0 or higher for GO blockage
MAX_NUM_REFL = 5
MAX_NUM_TRANS = 1
RAY_DENSITY = 1

# Tap Delay Filtering parameters
POWER_PERCENTAGE = 95.0 # Percentage of power to retain when filtering by power threshold
NUM_TAPS = 4 # Number of taps to retain when filtering by fixed count

# Visualization settings
DB_FLOOR = -200  # Minimum dB value for plots

# Debug and output settings
EXPORT_DEBUG = True
SAVE_RESULTS = False

#######################################
# INITIALIZATION
#######################################
# Initialize paths and API
paths = get_repo_paths()
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem
RssPy = pem_api_manager.RssPy

# Generate timestamps for animation
timestamps = np.linspace(0, TOTAL_TIME, NUM_FRAMES+1)

# Material manager - loads predefined materials from material_library.json
mat_manager = MaterialManager()

#######################################
# SCENE GEOMETRY SETUP
#######################################
# Create dictionary of all scene actors for easier organization
all_actors = Actors()

# Define primitive geometries
prim_plane = Plane(i_size=10, j_size=10, num_i=10, num_j=10, orientation=[1, 0, 0])
prim_cube = Cube(x_length=5, y_length=5, z_length=5)
prim_sphere = Sphere(radius=3, num_theta=20, num_phi=20)

# Calculate wavelength for ray spacing
wavelength = 3e8 / CENTER_FREQ

# Add back wall (plane 1)
back_wall_actor = all_actors.add_actor(
    name='back_wall',
    generator=prim_plane,
    mat_idx=mat_manager.get_index('pec'),
    target_ray_spacing=wavelength,
    dynamic_generator_updates=False
)
all_actors.actors[back_wall_actor].coord_sys.pos = [0, -20, 0]
all_actors.actors[back_wall_actor].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
all_actors.actors[back_wall_actor].coord_sys.update()

# Add front wall (plane 2)
front_wall_actor = all_actors.add_actor(
    name='front_wall',
    generator=prim_plane,
    mat_idx=mat_manager.get_index('pec'),
    target_ray_spacing=wavelength,
    dynamic_generator_updates=False
)
all_actors.actors[front_wall_actor].coord_sys.pos = [0, 30, 0]
all_actors.actors[front_wall_actor].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
all_actors.actors[front_wall_actor].coord_sys.update()

# Add cube obstacle 1
cube1_actor = all_actors.add_actor(
    name='cube_obstacle_1',
    generator=prim_cube,
    mat_idx=mat_manager.get_index('pec'),
    target_ray_spacing=wavelength,
    dynamic_generator_updates=False
)
all_actors.actors[cube1_actor].coord_sys.pos = [0, -7, 10]
all_actors.actors[cube1_actor].coord_sys.rot = euler_to_rot(phi=45, theta=35, order='zyz', deg=True)
all_actors.actors[cube1_actor].coord_sys.update()

# Add cube obstacle 2
cube2_actor = all_actors.add_actor(
    name='cube_obstacle_2',
    generator=prim_cube,
    mat_idx=mat_manager.get_index('pec'),
    target_ray_spacing=wavelength,
    dynamic_generator_updates=False
)
all_actors.actors[cube2_actor].coord_sys.pos = [0, 7, 0]
all_actors.actors[cube2_actor].coord_sys.update()

# Add sphere obstacle
sphere_actor = all_actors.add_actor(
    name='sphere_obstacle',
    generator=prim_sphere,
    mat_idx=mat_manager.get_index('pec'),
    target_ray_spacing=wavelength,
    dynamic_generator_updates=False
)
all_actors.actors[sphere_actor].coord_sys.pos = [0, -15, 0]
all_actors.actors[sphere_actor].coord_sys.update()

#######################################
# ANTENNA PLATFORM SETUP
#######################################
# Add stationary transmitter actor

# Transmitter position
tx_pos = [-10, 0, 0]
tx_actor_name = all_actors.add_actor(name='tx_platform')
all_actors.actors[tx_actor_name].velocity_mag = 0.0
all_actors.actors[tx_actor_name].coord_sys.pos = tx_pos
all_actors.actors[tx_actor_name].coord_sys.rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True)
all_actors.actors[tx_actor_name].coord_sys.update()

# Add moving receiver actor
rx_actor_name = all_actors.add_actor(name='rx_platform')
all_actors.actors[rx_actor_name].velocity_mag = 2.0
all_actors.actors[rx_actor_name].coord_sys.pos = [10, 0, 0]
all_actors.actors[rx_actor_name].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
all_actors.actors[rx_actor_name].coord_sys.update()

#######################################
# WAVEFORM DEFINITION
#######################################
mode_name = 'mode1'  # Mode identifier for post-processing

# Define waveform parameters programmatically for flexibility
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
    "center_freq": CENTER_FREQ,
    "bandwidth": BANDWIDTH,
    "num_freq_samples": NUM_FREQS,
    "cpi_duration": CPI_DURATION,
    "num_pulse_CPI": NUM_PULSE_CPI,
    "tx_multiplex": TX_MULTIPLEX,
    "mode_delay": "CENTER_CHIRP",
    "tx_incident_power": INPUT_POWER_WATTS
}

# Calculate pulse interval for reference
pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']

# Create waveform object (used for both TX and RX)
waveform = Waveform(waveform_dict)

#######################################
# ANTENNA DEVICE SETUP
#######################################
# Add diversity antenna pair to transmitter (spatial diversity)
ant_device_tx = add_diversity_antenna_pair(
    all_actors,
    waveform,
    mode_name,
    operation_mode='tx',
    parent_h_node=all_actors.actors[tx_actor_name].h_node,
    ffd_files=['dipole.ffd', 'dipole_y.ffd'],
    scale_pattern=0.75,
    spatial_diversity_offset=np.array([0.0, wavelength / 2, 0.0])
)

# Add diversity antenna pair to receiver (spatial diversity)
ant_device_rx = add_diversity_antenna_pair(
    all_actors,
    waveform,
    mode_name,
    operation_mode='rx',
    parent_h_node=all_actors.actors[rx_actor_name].h_node,
    ffd_files=['dipole.ffd', 'dipole_y.ffd'],
    scale_pattern=0.75,
    spatial_diversity_offset=np.array([0.0, wavelength / 2, 0.0])
)

# Enable coupling between TX and RX antenna pairs
enable_coupling(mode_name, ant_device_tx, ant_device_rx)

#######################################
# SIMULATION OPTIONS
#######################################
# Convert ray density to spacing
freq_center = ant_device_tx.waveforms[mode_name].center_freq
lambda_center = 2.99792458e8 / freq_center
ray_spacing = np.sqrt(2) * lambda_center / RAY_DENSITY

# List available GPUs
print(pem.listGPUs())

# Configure simulation parameters
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = MAX_NUM_REFL
sim_options.max_transmissions = MAX_NUM_TRANS
sim_options.go_blockage = GO_BLOCKAGE
sim_options.field_of_view = 360
sim_options.bounding_box = -1  # No bounding box truncation
sim_options.auto_configure_simulation()

# Extract response domains from waveform
which_mode = ant_device_tx.modes[mode_name]
ant_device_tx.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device_tx.waveforms[mode_name].vel_domain
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain
freq_domain = ant_device_tx.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx.waveforms[mode_name].pulse_domain

#######################################
# DEBUGGING AND VISUALIZATION SETUP
#######################################
# Activate radar camera view for debugging (optional, has performance cost)
if EXPORT_DEBUG:
    debug_camera = DebuggingCamera(
        hMode=ant_device_tx.modes[mode_name],
        display_mode='coating',
        output_size=(512, 512),
        background_color=255,
        frame_rate=FPS
    )
    debug_logs = DebuggingLogs(output_directory=paths.output)

# Setup 3D scene visualization
output_movie_name = os.path.join(paths.output, 'out_vis_indoor_wireless.mp4')
modeler = ModelVisualization(
    all_actors,
    show_antennas=True,
    rng_domain=None,
    vel_domain=None,
    overlay_results=False,
    fps=FPS,
    output_video_size=(800, 600),
    camera_orientation=None,
    camera_attachment=None,
    output_movie_name=output_movie_name
)
modeler.pl.show_grid()

#######################################
# TAP DELAY LINE INITIALIZATION
#######################################
# Initialize tap delay line processor
tdl = TapDelayLine(
    frequencies=freq_domain,
    tap_scale_factor=1.0,
    upsample_factor=1,
    window_type='hann'
)

#######################################
# PLOTTING SETUP
#######################################


def convert_to_db(x):
    """Convert magnitude to dB scale, avoiding log(0) issues."""
    return 20 * np.log10(np.fmax(np.abs(x), 1.e-10))


def plot_impulse_responses(axes, img, unfiltered, filtered_power, filtered_count, stats_power, stats_count):
    """Plot impulse response comparisons in a 2x2 grid."""
    # Scene visualization
    axes[0, 0].imshow(img)
    axes[0, 0].axis('off')

    # Unfiltered response
    axes[0, 1].cla()
    times, cir = tdl.get_channel_impulse_response(unfiltered, 0, 0, 0)
    axes[0, 1].plot(times * 1e9, convert_to_db(cir))
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_title('Unfiltered Response')
    axes[0, 1].grid(True, alpha=0.3)

    # Power threshold filtered response
    axes[1, 0].cla()
    times, cir = tdl.get_channel_impulse_response(filtered_power, 0, 0, 0)
    axes[1, 0].stem(times * 1e9, convert_to_db(cir), bottom=DB_FLOOR,
                    linefmt='-', markerfmt='ro', basefmt='x')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].set_title('Tap Delay: Power Threshold Response')
    axes[1, 0].set_ylim(bottom=DB_FLOOR)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add statistics text
    string_to_print = f"Keeping {stats_power['num_taps']} taps retains {stats_power['power_retained']:.2f}% power"
    axes[1, 0].text(0.05, 0.95, string_to_print, transform=axes[1, 0].transAxes,
                    fontsize=10, verticalalignment='top')

    # Fixed tap count filtered response
    axes[1, 1].cla()
    times, cir = tdl.get_channel_impulse_response(filtered_count, 0, 0, 0)
    axes[1, 1].stem(times * 1e9, convert_to_db(cir), bottom=DB_FLOOR,
                    linefmt='-', markerfmt='ro', basefmt='x')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].set_title('Tap Delay: Fixed Tap Count Response')
    axes[1, 1].set_ylim(bottom=DB_FLOOR)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    string_to_print = f"Keeping {stats_count['num_taps']} taps retains {stats_count['power_retained']:.2f}% power"
    axes[1, 1].text(0.05, 0.95, string_to_print, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top')


def plot_multi_channel_comparison(axes2, unfiltered, filtered_data_power):
    """Plot comparison across different channels and pulses."""
    # Compare different RX channels (same pulse)
    axes2[0, 0].cla()
    times, cir = tdl.get_channel_impulse_response(unfiltered, 0, 0, 0)
    axes2[0, 0].plot(times * 1e9, convert_to_db(cir), label='Tx 0 to Rx 0')
    times, cir = tdl.get_channel_impulse_response(unfiltered, 0, 1, 0)
    axes2[0, 0].plot(times * 1e9, convert_to_db(cir), label='Tx 0 to Rx 1')
    axes2[0, 0].set_xlabel('Time (ns)')
    axes2[0, 0].set_ylabel('Magnitude (dB)')
    axes2[0, 0].set_title('Unfiltered Response')
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].legend()

    # Filtered responses for both RX channels
    axes2[0, 1].cla()
    times, cir = tdl.get_channel_impulse_response(filtered_data_power, 0, 0, 0)
    axes2[0, 1].stem(times * 1e9, convert_to_db(cir), bottom=DB_FLOOR,
                     linefmt='-', markerfmt='ro', basefmt='x', label='Tx 0 to Rx 0')
    times, cir = tdl.get_channel_impulse_response(filtered_data_power, 0, 1, 0)
    axes2[0, 1].stem(times * 1e9, convert_to_db(cir), bottom=DB_FLOOR,
                     linefmt='-', markerfmt='bo', basefmt='x', label='Tx 0 to Rx 1')
    axes2[0, 1].set_xlabel('Time (ns)')
    axes2[0, 1].set_ylabel('Magnitude (dB)')
    axes2[0, 1].set_title('Tx 0 to Rx 0 and Rx 1 - Power Thresholded Response')
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].legend()

    # Compare first and last pulse for Rx 0
    axes2[1, 0].cla()
    times, cir = tdl.get_channel_impulse_response(filtered_data_power, 0, 0, 0)
    axes2[1, 0].stem(times * 1e9, convert_to_db(cir), bottom=DB_FLOOR,
                     linefmt='-', markerfmt='ro', basefmt='x', label='Tx 0 to Rx 0 - Pulse 0')
    times, cir = tdl.get_channel_impulse_response(filtered_data_power, 0, 0, -1)
    axes2[1, 0].stem(times * 1e9, convert_to_db(cir), bottom=DB_FLOOR,
                     linefmt='-', markerfmt='bo', basefmt='x', label='Tx 0 to Rx 0 - Pulse N')
    axes2[1, 0].set_xlabel('Time (ns)')
    axes2[1, 0].set_ylabel('Magnitude (dB)')
    axes2[1, 0].set_title('Tx 0 to Rx 0 Last Pulse - Power Thresholded Response')
    axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].legend()

    # Compare first and last pulse for Rx 1
    axes2[1, 1].cla()
    times, cir = tdl.get_channel_impulse_response(filtered_data_power, 0, 1, 0)
    axes2[1, 1].stem(times * 1e9, convert_to_db(cir), bottom=DB_FLOOR,
                     linefmt='-', markerfmt='ro', basefmt='x', label='Tx 0 to Rx 1 - Pulse 0')
    times, cir = tdl.get_channel_impulse_response(filtered_data_power, 0, 1, -1)
    axes2[1, 1].stem(times * 1e9, convert_to_db(cir), bottom=DB_FLOOR,
                     linefmt='-', markerfmt='bo', basefmt='x', label='Tx 0 to Rx 1 - Pulse N')
    axes2[1, 1].set_xlabel('Time (ns)')
    axes2[1, 1].set_ylabel('Magnitude (dB)')
    axes2[1, 1].set_title('Tx 0 to Rx 1 Last Pulse - Power Thresholded Response')
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].legend()


# Create matplotlib figures for visualization
plt.ion()  # Enable interactive mode

# Figure 1: Main impulse response comparison
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plt.show()
images = []

# Figure 2: Multi-channel and multi-pulse comparison
fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(14, 9))
plt.show()
images2 = []

#######################################
# MAIN SIMULATION LOOP
#######################################
# For each time step:
# 1. Update actor positions based on velocity
# 2. Compute electromagnetic response
# 3. Apply tap delay line filtering (2 methods)
# 4. Generate debug outputs
# 5. Update visualization plots

all_results = []
print('Running simulation...')



for time in tqdm(timestamps):

    # Update all actor coordinate systems
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)

    # Compute electromagnetic response for current time step
    pem_api_manager.isOK(pem.computeResponseSync())
    
    # Retrieve frequency-pulse domain response
    (ret, response) = pem.retrieveP2PResponse(
        ant_device_tx.modes[mode_name],
        ant_device_rx.modes[mode_name],
        RssPy.ResponseType.FREQ_PULSE
    )

    # Method 1: Filter by power percentage
    # channel_specific=True creates a unique mask for each channel and pulse
    # This adapts to the characteristics of each individual channel
    filtered_data_power, tap_mask = tdl.filter_by_power_threshold(
        response,
        power_percentage=POWER_PERCENTAGE,
        channel_specific=True,
        accumulate_data=True # Accumulate data for export, this will store the filtered data and mask for every time this function is called
    )
    stats_power = tdl.get_tap_statistics(tap_mask)
    print(f"Filter by Power (Average Across All Channels): Keeping {stats_power['num_taps']} taps retains {stats_power['power_retained']:.2f}% power")

    # Method 2: Filter by fixed tap count
    # Keeps exactly NUM_TAPS taps with highest power
    # channel_specific=False (default) creates single mask across all channels
    filtered_data_count, tap_mask = tdl.filter_by_tap_count(
        response,
        num_taps=NUM_TAPS,
        method='power_based',
        channel_specific=True,
        accumulate_data=True # if we want to write data to TDD later, we need to accumulate it, filter by count and power are stored separately
    )
    stats_count = tdl.get_tap_statistics(tap_mask)
    print(f"Filter by Count: Keeping {stats_count['num_taps']} taps retains {stats_count['power_retained']:.2f}% power")

    # Export debug outputs if enabled
    if EXPORT_DEBUG:
        debug_logs.write_scene_summary(file_name=f'out.json')
        debug_camera.generate_image()

    # Get 3D scene visualization as image
    img = modeler.update_frame(return_raw_image=True)

    # Convert to time domain for plotting
    unfiltered = tdl.convert_to_time_domain(response)

    # Update main comparison plot
    plot_impulse_responses(axes, img, unfiltered, filtered_data_power,
                          filtered_data_count, stats_power, stats_count)
    
    # Capture figure as image
    buf = np.asarray(fig.canvas.buffer_rgba())
    output_img = PIL.Image.fromarray(buf[:, :, :3])  # RGB only
    images.append(output_img)

    # Update multi-channel comparison plot
    plot_multi_channel_comparison(axes2, unfiltered, filtered_data_power)
    
    # Capture figure as image
    buf = np.asarray(fig2.canvas.buffer_rgba())
    output_img = PIL.Image.fromarray(buf[:, :, :3])  # RGB only
    images2.append(output_img)

# we need to know the absolute time of each pulse for TDD export. this pulse domain is usually centered around 0, so we need to shift it to be relative to the start of the CPI
pulse_domain -= pulse_domain[0] 
# when we export the data, we need to provide the absolute time of each scenario time step of each pulse as well becuase we simulate
# each frame with N pulses, so to create the absolute time of each pulse we need to add the scenario time to each pulse time. For example timestamps[0] + pulse_domain, would give use the time for each pulse at the first time step
tdl.export_to_tdd(os.path.join(paths.output, 'tap_delay_output.tdd'),
                  scenario_times = timestamps,
                  pulse_times=pulse_domain,
                  filter_type='fixed_count',# 'power_threshold' or 'fixed_count'
                  summary_as_json=True) 

# Save animations
print("Saving animations...")
if images:
    images[0].save(os.path.join(paths.output, 'tap_delay_comparison.gif'),
                   save_all=True, append_images=images[1:], optimize=False, duration=int(DT*1000), loop=0)

if images2:
    images2[0].save(os.path.join(paths.output, 'multi_channel_comparison.gif'),
                    save_all=True, append_images=images2[1:], optimize=False, duration=int(DT*1000), loop=0)

#######################################
# CLEANUP AND OUTPUT
#######################################
modeler.close()