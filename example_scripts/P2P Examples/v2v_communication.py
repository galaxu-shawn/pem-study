#
# Copyright ANSYS. All rights reserved.
#
# Example of P2P communication between two vehicles in an urban environment
# This script simulates antenna-to-antenna coupling between two cars as they
# move through a city, transitioning from line-of-sight to non-line-of-sight conditions.

import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import PIL.Image
import os
import sys
import pyvista as pv

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_rx, add_single_tx, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import Cube
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

# Simulation parameters
fps = 30
total_time = 20  # seconds to simulate
dt = 1 / fps
num_frames = int(total_time / dt)
timestamps = np.linspace(0, total_time, num_frames)

# Communication parameters
center_freq = 2.4e9  # 2.4 GHz Wi-Fi frequency
bandwidth = 40e6  # 40 MHz bandwidth
num_freqs = 128
cpi_duration = 100e-3
num_pulse_CPI = 1

# Simulation options
go_blockage = 1  # enable GO blockage to simulate shadowing effects
max_num_refl = 5  # maximum number of reflections
max_num_trans = 1  # maximum number of transmissions
ray_density = 1.0  # ray density (affects simulation accuracy and performance)

export_debug = True  # export debug information

# Material manager
mat_manager = MaterialManager()

# Create actors dictionary
all_actors = Actors()

# Add urban environment - buildings that will block signals
building_height = 15  # meters
building_width = 20   # meters
building_length = 40  # meters
street_width = 25     # meters

# Create city blocks using boxes
# First building - left side of street
building1_generator = Cube(x_length=building_width, y_length=building_length, z_length=building_height)
building1_name = all_actors.add_actor(name='building1', 
                                     generator=building1_generator, 
                                     mat_idx=mat_manager.get_index('concrete'),
                                     color='lightgray')
all_actors.actors[building1_name].coord_sys.pos = (-building_width/2 - street_width/2, 0, building_height/2)
all_actors.actors[building1_name].coord_sys.update()

# Second building - right side of street
building2_generator = Cube(x_length=building_width, y_length=building_length, z_length=building_height)
building2_name = all_actors.add_actor(name='building2', 
                                     generator=building2_generator, 
                                     mat_idx=mat_manager.get_index('concrete'),
                                     color='lightgray')
all_actors.actors[building2_name].coord_sys.pos = (building_width/2 + street_width/2, 0, building_height/2)
all_actors.actors[building2_name].coord_sys.update()

# Add the street (ground)
street_length = 150  # meters
street_generator = Cube(x_length=street_width, y_length=street_length, z_length=0.3)
street_name = all_actors.add_actor(name='street', 
                                  generator=street_generator, 
                                  mat_idx=mat_manager.get_index('asphalt'),
                                  color='darkgray')
all_actors.actors[street_name].coord_sys.pos = (0, 0, -0.15)
all_actors.actors[street_name].coord_sys.update()

# Add two vehicles
# First vehicle (transmitter) - stays in position
geo_filename = 'Audi_A1_2010/Audi_A1_2010.json'
car1_name = all_actors.add_actor(name='car1',
                                filename=os.path.join(paths.models,geo_filename),
                                target_ray_spacing=0.05)
all_actors.actors[car1_name].coord_sys.pos = (0, -40, 0)
# Change rotation to align with Y-axis (forward direction)
all_actors.actors[car1_name].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
all_actors.actors[car1_name].velocity_mag = 0.0  # stationary
all_actors.actors[car1_name].coord_sys.update()

# Second vehicle (receiver) - moves along the street
geo_filename = 'Audi_A1_2010/Audi_A1_2010.json'
car2_name = all_actors.add_actor(name='car2',
                                filename=os.path.join(paths.models, geo_filename),
                                target_ray_spacing=0.05)
all_actors.actors[car2_name].coord_sys.pos = (0, -60, 0)
# Change rotation to align with Y-axis (forward direction)
all_actors.actors[car2_name].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
all_actors.actors[car2_name].velocity_mag = 8.0  # 8 m/s (approx. 18 mph)
all_actors.actors[car2_name].coord_sys.update()

# Define movement path for car2 - it will drive straight and then turn right
# This will cause it to go from line-of-sight to non-line-of-sight
time_steps = np.linspace(0, total_time, 301)  # More points for smoother interpolation

# Define positions for car2
# Start behind car1, then drive past it and turn right behind the building
car2_positions = []
turn_time = 10  # when to start turning (in seconds)
turn_duration = 5  # how long the turn takes
turn_radius = 12  # turning radius

for t in time_steps:
    if t < turn_time:
        # Moving straight
        pos = [0, -60 + all_actors.actors[car2_name].velocity_mag * t, 0]
    else:
        # Turning right
        turn_progress = min(1.0, (t - turn_time) / turn_duration)
        turn_angle = turn_progress * np.pi/2  # 90-degree turn
        
        # Position at the start of the turn
        start_x = 0
        start_y = -60 + all_actors.actors[car2_name].velocity_mag * turn_time
        
        # Position during/after the turn (circle arc)
        x = start_x + turn_radius * np.sin(turn_angle)
        y = start_y + turn_radius * (1 - np.cos(turn_angle))
        
        # After the turn, continue straight
        if turn_progress >= 1.0:
            extra_distance = all_actors.actors[car2_name].velocity_mag * (t - (turn_time + turn_duration))
            x += extra_distance
        
        pos = [x, y, 0]
    
    car2_positions.append(pos)

car2_positions = np.array(car2_positions)

# Also calculate rotation for car2
car2_rotations = []
for i in range(len(time_steps)):
    if time_steps[i] < turn_time:
        # Driving straight ahead - aligned with Y-axis
        rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
    else:
        # Turning and after the turn
        turn_progress = min(1.0, (time_steps[i] - turn_time) / turn_duration)
        # Start from 90 degrees (y-axis aligned) and rotate to 180 degrees (x-axis aligned)
        rot_angle = 90 + turn_progress * 90  # degrees
        if turn_progress >= 1.0:
            rot_angle = 180  # Fixed angle after turn is complete (aligned with X-axis)
        rot = euler_to_rot(phi=rot_angle, theta=0, order='zyz', deg=True)
    
    car2_rotations.append(rot)

car2_rotations = np.array(car2_rotations)

# Create interpolation functions
interp_func_pos = scipy.interpolate.interp1d(time_steps, car2_positions, axis=0, assume_sorted=True)
interp_func_rot = scipy.interpolate.interp1d(time_steps, car2_rotations, axis=0, assume_sorted=True)

# Mark car2 to use position-based updates instead of velocity-based
all_actors.actors[car2_name].use_linear_velocity_equation_update = False

# Define the waveform for communication
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_pulse_CPI,
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "CENTER_CHIRP",
    "tx_incident_power": 1.0  # 1 Watt transmit power
}

waveform = Waveform(waveform_dict)
mode_name = 'mode1'  # communication mode name

# Add dipole antennas to the cars
# Transmitter antenna on car1 (roof)
ant_device_tx = add_single_tx(all_actors, waveform, mode_name,
                             ffd_file='dipole.ffd',
                             scale_pattern=0.75,
                             parent_h_node=all_actors.actors[car1_name].h_node)
ant_device_tx.coord_sys.pos = (0, 0, 1.5)  # 1.5 meters above car roof
ant_device_tx.coord_sys.update()

# Receiver antenna on car2 (roof)
ant_device_rx = add_single_rx(all_actors, waveform, mode_name,
                             ffd_file='dipole.ffd',
                             scale_pattern=0.75,
                             parent_h_node=all_actors.actors[car2_name].h_node)
ant_device_rx.coord_sys.pos = (0, 0, 1.5)  # 1.5 meters above car roof
ant_device_rx.coord_sys.update()

# Enable coupling between transmitter and receiver
enable_coupling(mode_name, ant_device_tx, ant_device_rx)

# Set up simulation options
lambda_center = 2.99792458e8 / center_freq
ray_spacing = np.sqrt(2) * lambda_center / ray_density

print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1  # no bounding box truncation
sim_options.auto_configure_simulation()

# Get response domains for post-processing
which_mode = ant_device_tx.modes[mode_name]
ant_device_tx.waveforms[mode_name].get_response_domains(which_mode)
freq_domain = ant_device_tx.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx.waveforms[mode_name].pulse_domain

# Set up debugging if enabled
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes[mode_name],
                                 display_mode='coating',
                                 output_size=(512, 512),
                                 background_color=255,
                                 frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)

# Set up visualization
output_movie_name = os.path.join(paths.output, 'v2v_communication.mp4')
modeler = ModelVisualization(all_actors,
                           show_antennas=True,
                           rng_domain=None,
                           vel_domain=None,
                           overlay_results=False,
                           fps=fps,
                           camera_orientation='scene_top',  # bird's eye view
                           camera_attachment=None,
                           output_movie_name=output_movie_name)
modeler.pl.show_grid()

# Prepare to store results
all_results = []
path_losses = []
distances = []
los_status = []  # Will store whether cars are in line-of-sight

print('Running simulation...')
for idx_frame in tqdm(range(num_frames), disable=False):
    time = idx_frame * dt
    
    # Update car2's position based on the interpolation function
    if time <= time_steps[-1]:
        all_actors.actors[car2_name].coord_sys.pos = interp_func_pos(time)
        all_actors.actors[car2_name].coord_sys.rot = interp_func_rot(time)
    
    # Update all actors
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)
    
    # Log scene summary for the first and last frames
    if export_debug and (idx_frame == 0 or idx_frame == num_frames - 1):
        debug_logs.write_scene_summary(file_name=f'v2v_{idx_frame}.json')
    
    # Run the simulation for this frame
    pem_api_manager.isOK(pem.computeResponseSync())
    
    # Retrieve the response (communication channel data)
    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes[mode_name],
                                           ant_device_rx.modes[mode_name],
                                           RssPy.ResponseType.FREQ_PULSE)
    response = np.array(response)
    all_results.append(response)
    
    # Calculate path loss
    path_loss = np.abs(response[0, 0, 0])
    path_losses.append(path_loss)
    
    # Calculate distance between antennas
    tx_global_pos = ant_device_tx.coord_sys.pos
    rx_global_pos = ant_device_rx.coord_sys.pos
    distance = np.linalg.norm(np.array(tx_global_pos) - np.array(rx_global_pos))
    distances.append(distance)
    
    # Check if cars are in line-of-sight (simplified)
    # We'll use the debugging camera's blockage information in a real implementation
    # For now, we'll use a simple heuristic: after car2 turns, it's NLOS
    is_los = time < (turn_time + turn_duration)
    los_status.append(is_los)
    
    # Generate debug camera image if enabled
    if export_debug:
        debug_camera.generate_image()
    
    # Create plot showing communication metrics
    f, ax = plt.subplots(tight_layout=True, figsize=(10, 6))
    
    # Plot frequency response
    plt.subplot(2, 1, 1)
    plt.plot(freq_domain / 1e9, 20 * np.log10(abs(response[0, 0, 0])), color='blue', label='Channel Response')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('S21 [dB]')
    plt.title('Channel Frequency Response')
    plt.grid(True)
    
    # Plot path loss over time
    plt.subplot(2, 1, 2)
    times = timestamps[:idx_frame+1]
    plt.plot(times, path_losses, 'r-', label='Path Loss')
    plt.axvline(x=turn_time, color='k', linestyle='--', label='Turn Start')
    plt.axvline(x=turn_time + turn_duration, color='g', linestyle='--', label='Turn End')
    plt.xlabel('Time [s]')
    plt.ylabel('Path Loss [dB]')
    plt.title('Communication Link Quality')
    plt.grid(True)
    if idx_frame > 0:
        los_indices = [i for i, los in enumerate(los_status[:idx_frame+1]) if los]
        nlos_indices = [i for i, los in enumerate(los_status[:idx_frame+1]) if not los]
        if los_indices:
            plt.plot(times[los_indices], [path_losses[i] for i in los_indices], 'bo', label='LOS')
        if nlos_indices:
            plt.plot(times[nlos_indices], [path_losses[i] for i in nlos_indices], 'ro', label='NLOS')
    plt.legend()
    
    # Add the plot to the 3D visualization
    h_chart = pv.ChartMPL(f, size=(0.4, 0.4), loc=(0.6, 0.6))
    modeler.pl.add_chart(h_chart)
    
    # Update the visualization frame
    modeler.update_frame()
    plt.clf()
    plt.close()
    
# End of simulation
modeler.close()

# Convert all results to numpy array
all_results = np.array(all_results)
print('Simulation complete')

# Generate output graphics if debug is enabled
if export_debug:
    debug_camera.write_camera_to_gif(file_name='v2v_camera.gif')
    
    # Create a summary plot of the entire simulation
    plt.figure(figsize=(12, 10))
    
    # Plot path loss vs time
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, path_losses, 'b-', linewidth=2)
    plt.axvline(x=turn_time, color='k', linestyle='--', label='Turn Start')
    plt.axvline(x=turn_time + turn_duration, color='g', linestyle='--', label='Turn End')
    
    los_indices = [i for i, los in enumerate(los_status) if los]
    nlos_indices = [i for i, los in enumerate(los_status) if not los]
    
    if los_indices:
        plt.plot(timestamps[los_indices], [path_losses[i] for i in los_indices], 'go', label='LOS')
    if nlos_indices:
        plt.plot(timestamps[nlos_indices], [path_losses[i] for i in nlos_indices], 'ro', label='NLOS')
    
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Path Loss [dB]', fontsize=12)
    plt.title('V2V Communication Link Path Loss Over Time', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Plot path loss vs distance, colored by LOS/NLOS
    plt.subplot(2, 1, 2)
    
    if los_indices:
        plt.scatter([distances[i] for i in los_indices], [path_losses[i] for i in los_indices], 
                  c='green', label='LOS', alpha=0.7)
    if nlos_indices:
        plt.scatter([distances[i] for i in nlos_indices], [path_losses[i] for i in nlos_indices], 
                  c='red', label='NLOS', alpha=0.7)
    
    # Theoretical free-space path loss for comparison
    d_range = np.linspace(min(distances), max(distances), 100)
    fspl = 20 * np.log10(d_range) + 20 * np.log10(center_freq/1e6) - 27.55  # Free-space path loss formula
    plt.plot(d_range, fspl, 'k--', label='Free-space path loss model')
    
    plt.xlabel('Distance [m]', fontsize=12)
    plt.ylabel('Path Loss [dB]', fontsize=12)
    plt.title('Path Loss vs Distance', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'v2v_results_summary.png'), dpi=300)
    plt.close()
    
    # Save the data for further analysis
    np.savez(os.path.join(output_path, 'v2v_simulation_data.npz'),
           timestamps=timestamps,
           distances=distances,
           path_losses=path_losses,
           los_status=los_status,
           all_results=all_results,
           freq_domain=freq_domain)

print(f"Results saved to {output_path}")