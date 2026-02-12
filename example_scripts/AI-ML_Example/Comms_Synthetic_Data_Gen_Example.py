# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

@author: asligar
"""

import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits import mplot3d
import PIL.Image
import pyvista as pv
import os
import sys
import uuid
import shutil
import time as walltime

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import  Waveform, add_single_tx, add_single_rx, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.open_street_maps_geometry import BuildingsPrep, TerrainPrep, find_random_location
from pem_utilities.simulation_options import SimulationOptions

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


# simulation timing parameters
dt = 0.1
total_num_scenes = 20 # each scene is a new starting point for the rx antennas
num_rx = 64 # how many users to simulate per scene
include_mobility = True # if True, the rx antennas will move around the scene
# how many time steps to wait before changing the random selected position for each rx, if include_mobility is set to True
time_steps_per_position = 30


# range of values that lat/lon can be generated around
scene_lat_lon = (40.747303, -73.985701) #nyc
max_radius = 500 # meters

# Tx Antenna positions
ant_1_pos = (0 ,0 ,55)


# waveform parameters
center_freq = 3e9
num_freqs = 512
bandwidth = 3e9
cpi_duration = 100e-3
num_pulse_CPI = 101 # this will result in a pulse interval of 1ms

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .1

# export debug logs and show modeler for visualization
export_debug = False
show_modeler = True

#######################################################################################################################
# Input Parameters
#######################################################################################################################

print(f'Generating Scene: Lat/Lon: {scene_lat_lon}...')

buildings_prep = BuildingsPrep(paths.output)
# terrain is not yet created, I will create it later, using the exact same points as used for the heatmap surface
building_image_path = os.path.join(paths.output, 'buildings.png')
buildings = buildings_prep.generate_buildings(scene_lat_lon, terrain_mesh=None, max_radius=max_radius,
                                              export_image_path=building_image_path)

terrain_prep = TerrainPrep(paths.output)
terrain = terrain_prep.get_terrain(flat_surface=True)


#######################################################################################################################
# Setup and Run Perceive EM
#######################################################################################################################

debug_logs = DebuggingLogs(output_directory=paths.output)



mat_manager = MaterialManager()
all_actors = Actors()  # all actors, using the same material library for everyone


buildings_name = all_actors.add_actor(filename=buildings['file_name'],
                                 mat_idx=mat_manager.get_index('concrete'),
                                 color='grey',transparency=0.0)

terrain_name = all_actors.add_actor(filename=terrain['file_name'],
                                 mat_idx=mat_manager.get_index('asphalt'),
                                 color='black',transparency=0.5)



# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more flexibility
# and control over the parameters, without having to create/modify a json file
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_pulse_CPI,
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "CENTER_CHIRP"}

pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
mode_name = 'mode1'

# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)


#################
# Tx antenna

ant_device_tx1 = add_single_tx(all_actors,waveform,mode_name,pos=ant_1_pos,ffd_file='dipole.ffd',scale_pattern=20.0)


#################
# Rx antennas

all_rx_actors = []
all_rx_devices = []
xy = False
for rx_idx in range(num_rx):
    while not xy:
        xy = find_random_location(buildings['mesh'], outdoors=True) # only place antennas outdoors
    rx_pos = [xy[0], xy[1], 1.5] # always place the antennas at 1.5m height
    rx_actor_name = all_actors.add_actor()
    all_rx_actors.append(rx_actor_name)
    all_actors.actors[rx_actor_name].coord_sys.pos = np.array(rx_pos)
    if include_mobility:
        # randomize the velocity vector of actor,
        lin_x = np.random.uniform(low=-3, high=3)
        lin_y = np.random.uniform(low=-3, high=3)
        all_actors.actors[rx_actor_name].coord_sys.lin = np.array([lin_x, lin_y, 0])
    all_actors.actors[rx_actor_name].coord_sys.update()
    # import this antenna into an existing actor, we can just move the actor and the antenna will move with it
    ant_device_rx = add_single_rx(all_actors,waveform,mode_name,parent_h_node=all_actors.actors[rx_actor_name].h_node,ffd_file='dipole.ffd')
    all_rx_devices.append(ant_device_rx)
    # between all the existing tx, and rx antennas, which ones do we want to compute coupling between
    enable_coupling(mode_name,ant_device_tx1, ant_device_rx)


# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    freq_center = ant_device_tx1.waveforms[mode_name].center_freq
    lambda_center = 2.99792458e8 / freq_center
    ray_spacing = np.sqrt(2) * lambda_center / ray_density


sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()


# if we want to query the domain information for the solution, we can do that all for just tx1 becuase they all use
# the same waveform, they will be all the same. In this case, these domains are just used for plotting
which_mode = ant_device_tx1.modes[mode_name] # tell it which mode we want to get respones from
ant_device_tx1.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device_tx1.waveforms[mode_name].vel_domain
rng_domain = ant_device_tx1.waveforms[mode_name].rng_domain
freq_domain = ant_device_tx1.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx1.waveforms[mode_name].pulse_domain

fps = 1 / dt

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx1.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

output_movie_name = os.path.join(paths.output, 'out_vis_synt_data_gen.mp4')

# going to create a rolling plot that is num_rx * time_steps_per_position *10 long
output_size = (num_rx * num_pulse_CPI, num_freqs)
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             overlay_results=True,
                             fps=fps,
                             shape=output_size,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name,)

modeler.pl.show_grid()
print('running simulation...')



timestamps = np.linspace(0, time_steps_per_position*dt-dt, time_steps_per_position)

for idx in tqdm(range(total_num_scenes)):
    all_reponses = []
    # get new initial positions for all the rx antennas
    for rx_actor_name in all_rx_actors:
        xy = False
        while not xy:
            xy = find_random_location(buildings['mesh'], outdoors=True)  # only place antennas outdoors
        rx_pos = [xy[0], xy[1], 1.5]  # always place the antennas at 1.5m height
        all_actors.actors[rx_actor_name].coord_sys.pos = np.array(rx_pos)
        if include_mobility:
            # randomize the velocity vector of actor,
            lin_x = np.random.uniform(low=-3, high=3)
            lin_y = np.random.uniform(low=-3, high=3)
            all_actors.actors[rx_actor_name].coord_sys.lin = np.array([lin_x, lin_y, 0])
        all_actors.actors[rx_actor_name].coord_sys.update()


    for time in timestamps:
        all_reponses = []
        # update all coordinate systems
        for actor in all_actors.actors:
            all_actors.actors[actor].update_actor(time=time)

        start_sim_time = walltime.time()
        pem_api_manager.isOK(pem.computeResponseSync())
        end_sim_time = walltime.time()
        print(f'Simulation Time: {end_sim_time - start_sim_time} seconds')

        start_ret_time = walltime.time()
        for rx_device in all_rx_devices:
            (ret, response) = pem.retrieveP2PResponse(ant_device_tx1.modes[mode_name],
                                                  rx_device.modes[mode_name],
                                                  RssPy.ResponseType.FREQ_PULSE)
            all_reponses.append(response)
        end_ret_time = walltime.time()
        print(f'Retrieval Time: {end_ret_time - start_ret_time} seconds')
        all_reponses = np.array(all_reponses)
        center_pulse = num_pulse_CPI // 2
        one_sim_data = all_reponses[:,0,0]
        one_sim_data = one_sim_data.reshape((num_rx*num_pulse_CPI, num_freqs))
        im_data = 20 * np.log10(np.fmax(np.abs(one_sim_data), 1.e-30))
        modeler.mpl_ax_handle.set_data(im_data.T)  # update pyvista matplotlib plot
        max_of_data = np.max(im_data)
        modeler.mpl_ax_handle.set_clim(vmin=max_of_data-30, vmax=max_of_data)
        # exporting radar camera images
        if export_debug:
            if idx == 0 or idx == num_frames - 1:
                debug_logs.write_scene_summary(file_name=f'out_{idx}.json')
            debug_camera.generate_image()

        modeler.update_frame() # update the visualization


modeler.close()
if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')