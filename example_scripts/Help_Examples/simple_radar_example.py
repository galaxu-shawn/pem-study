"""
Simple Radar Example - Minimal Setup
====================================

Simple scene, 1 perfectly conducting sphere target, 1 radar transciver antenna with a pulsed Doppler waveform.

This example demonstrates the most basic radar simulation workflow:
1. Import utilities
2. Set radar parameters  
3. Create scene with target and radar
4. Define waveform
5. Add radar antenna
6. Configure simulation
7. Run simulation
8. Retrieve results

This is the simplest possible radar example to get started.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 1) Import utilities
from pem_utilities.actor import Actors # actors for scene creation and control
from pem_utilities.materials import MaterialManager # default material library
from pem_utilities.antenna_device import Waveform, add_single_tx_rx # antenna waveform and sources
from pem_utilities.simulation_options import SimulationOptions # configure simulation options
from pem_utilities.model_visualization import ModelVisualization # for visualizing results
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 


# 2) Set radar/waveform parameters (simple 77 GHz automotive radar). This will be used to create a pulsed Doppler waveform.
# These parameters are typical for automotive radar systems.
center_freq = 77e9        # 77 GHz
bandwidth = .1e9           # 1 GHz bandwidth
num_freqs = 256           # Number of frequency samples
cpi_duration = 10e-3      # 10 ms coherent processing interval
num_pulse_CPI = 128       # Number of pulses per CPI

# Simulation parameters. These control the resolution and performance of the simulation.
# Adjust these based on your scene complexity and performance requirements.
ray_spacing = 0.1         # Ray spacing in meters, global mesh grid resolution, applied everywhere, but overridden by target ray spacing
max_reflections = 3       # Maximum number of reflections
go_blockage = 1          # Enable geometric optics blockage


print("Starting simple radar simulation...")

# 3) Create scene actors
# Initialize material manager and actors
# this will create a default material library and creates an empty actor list where we will store the scene tree
mat_manager = MaterialManager()
all_actors = Actors()

# Add a simple target (PEC sphere with 1 meter radius)
target_name = all_actors.add_actor(filename=os.path.join(paths.models,'Sphere_1meter_rad.stl'),
                                 mat_idx=mat_manager.get_index('pec'),  # Perfect conductor
                                 target_ray_spacing=0.05) # target ray spacing in meters is adaptive mesh grid that moves with the target

# Position target at 50 meters in front of radar
all_actors.actors[target_name].coord_sys.pos = [50, 0, 0]
all_actors.actors[target_name].coord_sys.update() # set the coordinate system of the target actor

# Add radar platform (empty actor to attach radar to)
radar_actor_name = all_actors.add_actor()
all_actors.actors[radar_actor_name].coord_sys.pos = [0, 0, 0]  # Radar at origin
all_actors.actors[radar_actor_name].coord_sys.update()

# 4) Define waveform, this dictionary will be used when we create the antenna device
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "RangeDoppler",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_pulse_CPI,
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "CENTER_CHIRP"
}

mode_name = 'mode1' # Name of the mode, can be anything, but may be used to reference the mode later
# Create the waveform object using the dictionary
waveform = Waveform(waveform_dict)

# 5) Add radar antenna device (simple Tx/Rx pair).
# this utility function creates a radar antenna device with a single Tx/Rx pair. This could be an ffd file or a parametric beam
ant_device = add_single_tx_rx(all_actors, # all actors is the scene tree, it is included here as we will append  this radar device to the list of actors
                             waveform, 
                             mode_name,
                             parent_h_node=all_actors.actors[radar_actor_name].h_node,
                             ffd_file='dipole.ffd',  # Use dipole antenna pattern
                             scale_pattern=1.0) # scaling used for visuzalization, not required for simulation

# 6) Configure simulation options
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_reflections
sim_options.max_transmissions = 1
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 180 # Field of view in degrees, can be 180 or 360. If the Tx will shoot in all directions, set to 360. If set to 180, the Tx will only shoot in the positive x-direction.
sim_options.auto_configure_simulation() # Automatically configure GPU based on the scene and options


# get response domain, just for reference
which_mode = ant_device.modes[mode_name]  # tell it which mode we want to get respones from
ant_device.waveforms[mode_name].get_response_domains(which_mode)

# domain in every dimension, this could be used for plotting or analysis calculated from the waveform
vel_domain = ant_device.waveforms[mode_name].vel_domain
rng_domain = ant_device.waveforms[mode_name].rng_domain
freq_domain = ant_device.waveforms[mode_name].freq_domain
pulse_domain = ant_device.waveforms[mode_name].pulse_domain
output_format = ant_device.waveforms[mode_name].output

print(f"Velocity Window: {np.ptp(vel_domain)}")
print(f"Range Window: {np.max(rng_domain)}")

# Check if simulation is ready
if not pem.isReady():
    print("Simulation not ready!")
    print(pem.getLastWarnings())
    exit()


# not required but useful to visualize the scene. Does impact performance, so disable for high performance runs
output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors, # actors to visualize
                             show_antennas=True, # show antennas along with geometry
                             fps=10, # frames per second for the output video
                             output_movie_name=output_movie_name) # output video file name


# 7) Run simulation
for idx in range(100): # lets run 100 iterations, update position of target for each iteration
    print(f"Running simulation iteration {idx+1}...")
    
    # manually update the target position and velocity or a specific actor
    all_actors.actors[target_name].coord_sys.pos = [50+idx, 0, 0] # arbitrarily move target along x-axis
    all_actors.actors[target_name].coord_sys.lin = [10, 0, 0] # assign some linear velocity
    all_actors.actors[target_name].coord_sys.update() # set the new coordinate system of the target actor

    # automatically update all actors in the scene using the update method
    # for actor in all_actors.actors:
    #     all_actors.actors[actor].update_actor(time=idx) # if a time is provided, the actor will update its position based on the time and its velocity

    pem_api_manager.isOK(pem.computeResponseSync()) # run simulation

    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER) # retrieve the response for the specified mode
    im = 20*np.log10(np.abs(response[0,0])) # index [0][0] corresponds to the first Tx/Rx pair
    modeler.update_frame(plot_data=im,plot_limits=[np.max(im)-30, np.max(im)]) # update the visualization frame, including adding a plot of the response data
    
modeler.close()
