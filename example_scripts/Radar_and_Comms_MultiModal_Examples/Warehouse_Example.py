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
from pem_utilities.antenna_device import  Waveform, add_single_tx, add_single_rx, enable_coupling, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.open_street_maps_geometry import BuildingsPrep, TerrainPrep, find_random_location
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.animation_selector_CMU import select_CMU_animation
from pem_utilities.router import Pick_Path

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy

# simulation timing parameters
dt = 0.05
total_time = 10

custom_user_path = False

type_of_simulation = 'comms' # 'radar' or 'comms'
which_radar_response_to_plot = 'forklift' # 'room' or 'forklift'
# Tx Antenna positions
ant_1_pos = (1.5 ,-1.4 ,2.55)

# room monitor radar position and rotation
radar_pos = (2 ,-1.5 ,2.5)
radar_rot = euler_to_rot(phi=-45,theta=15, order='zyz',deg=True)

# waveform parameters for commms
center_freq_comms = 3e9
num_freqs_comms = 512
bandwidth_comms = 1e9
cpi_duration_comms = 100e-3
num_pulse_CPI_comms = 101 # this will result in a pulse interval of 1ms

# waveform parameters for radar
center_freq_radar = 77e9
num_freqs_radar = 512
bandwidth_radar = 600e6
cpi_duration_radar = 8e-3
num_pulse_CPI_radar = 64 # this will result in a pulse interval of 1ms

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .1

# export debug logs and show modeler for visualization
export_debug = True
show_modeler = True

#######################################################################################################################
# Input Parameters
#######################################################################################################################


#######################################################################################################################
# Setup and Run Perceive EM
#######################################################################################################################

debug_logs = DebuggingLogs(output_directory=paths.output)

mat_manager = MaterialManager()
all_actors = Actors()  # all actors, using the same material library for everyone

warehouse_name = all_actors.add_actor(filename=os.path.join(paths.models,'warehouse','warehouse.json'),
                                      transparency=0.5,
                                      scale_mesh=1/0.0254,
                                      target_ray_spacing=0.05) # units of cad are in inches, so we need to scale to meters

forklift_name = all_actors.add_actor(filename=os.path.join(paths.models,'forklift.obj'),
                                        name='forklift',
                                        mat_idx=mat_manager.get_index('pec'),
                                        color='orange',
                                        target_ray_spacing=0.05)


all_actors.actors[warehouse_name].coord_sys.pos = (0,0,.35) # floor of warehouse is not at 0

all_actors.actors[forklift_name].velocity_mag = 1.5
all_actors.actors[forklift_name].coord_sys.pos = (6,-20,0)
all_actors.actors[forklift_name].coord_sys.rot = euler_to_rot(phi=90,theta=0, order='zyz',deg=True)


#######################################################################################################################
# Add people to warehouse
#######################################################################################################################


if custom_user_path:
    pre_picked_path = None
else:
    pre_picked_path = [[15.28256532, -1.90903491, 0],
       [16.95816197,  -3.737338 , 0],
       [14.87917521, -3.43745181, 0],
       [14.40346811, -7.23835946, 0],
       [13.66570333, -8.69189646, 0],
       [11.8505088 , -10.26153286, 0],
       [11.24930255, -9.21485172, 0],
       [11.83755543, -7.68330629, 0],
       [12.86142958, -5.45805469, 0]]
path_picked_person = Pick_Path()
path_picked_person.custom_path(mesh_list=all_actors.actors[warehouse_name].get_mesh(), pre_picked=pre_picked_path,
                               speed=1.0,
                               snap_to_surface=True)


ped1_name = all_actors.add_actor(filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'), target_ray_spacing=0.05,
                                 scale_mesh=1)
all_actors.actors[ped1_name].velocity_mag = 1.0
all_actors.actors[ped1_name].use_linear_velocity_equation_update = False
# cmu_filename, cmu_description = select_CMU_animation(activity_type='walking') # random selection within walking category
# cmu_filename, cmu_description = select_CMU_animation() # random selection
cmu_filename, cmu_description = select_CMU_animation(file_name='13_23')  # specific selection
ped2_name = all_actors.add_actor(filename=cmu_filename, target_ray_spacing=0.05, scale_mesh=1)
all_actors.actors[ped2_name].coord_sys.pos = (3.5, -5, 0.)
cmu_filename, cmu_description = select_CMU_animation(file_name='62_18')  # specific selection
ped3_name = all_actors.add_actor(filename=cmu_filename, target_ray_spacing=0.05, scale_mesh=1)
all_actors.actors[ped3_name].coord_sys.pos = (7.25, -1.75, 0.)
all_actors.actors[ped3_name].coord_sys.rot = euler_to_rot(phi=0, order='zyz',deg=True)


# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more flexibility
# and control over the parameters, without having to create/modify a json file
waveform_radar_dict = {
    "mode": "PulsedDoppler",
    "output": "RangeDoppler",
    "center_freq": center_freq_radar,
    "bandwidth": bandwidth_radar,
    "num_freq_samples": num_freqs_radar,
    "cpi_duration": cpi_duration_radar,
    "num_pulse_CPI": num_pulse_CPI_radar,
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "CENTER_CHIRP"}

pulse_interval = waveform_radar_dict['cpi_duration'] / waveform_radar_dict['num_pulse_CPI']
mode_name_radar = 'mode1_radar'

waveform_comms_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
    "center_freq": center_freq_comms,
    "bandwidth": bandwidth_comms,
    "num_freq_samples": num_freqs_comms,
    "cpi_duration": cpi_duration_comms,
    "num_pulse_CPI": num_pulse_CPI_comms,
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "CENTER_CHIRP"}

pulse_interval = waveform_comms_dict['cpi_duration'] / waveform_comms_dict['num_pulse_CPI']
mode_name_comms = 'mode1_comms'



# waveform will be used for both Tx and Rx
waveform_radar = Waveform(waveform_radar_dict)
waveform_comms = Waveform(waveform_comms_dict)

#################
# Tx antenna


if type_of_simulation == 'comms':
    ant_device_tx1 = add_single_tx(all_actors, waveform_comms, mode_name_comms, pos=ant_1_pos, ffd_file='dipole.ffd',
                                   scale_pattern=.5)

    ant_device_rx = add_single_rx(all_actors, waveform_comms, mode_name_comms, pos=(0, 0, 2.5),
                                  parent_h_node=all_actors.actors[forklift_name].h_node,
                                  ffd_file='dipole.ffd',
                                  scale_pattern=.5)
    # between all the existing tx, and rx antennas, which ones do we want to compute coupling between
    enable_coupling(mode_name_comms,ant_device_tx1, ant_device_rx)
    center_freq = ant_device_tx1.waveforms[mode_name_comms].center_freq
else:
    radar_room_actor_name = all_actors.add_actor() # easier to attach a camera to an actor rather than an antenna
    all_actors.actors[radar_room_actor_name].coord_sys.pos = radar_pos
    all_actors.actors[radar_room_actor_name].coord_sys.rot = radar_rot
    ant_device_radar_room = add_single_tx_rx(all_actors, waveform_radar, mode_name_radar,
                                        scale_pattern=.5,parent_h_node=all_actors.actors[radar_room_actor_name].h_node)
    ant_device_radar_forklift = add_single_tx_rx(all_actors, waveform_radar, mode_name_radar,
                                        pos=(1.85, 0, 3),
                                        scale_pattern=.75,
                                                 parent_h_node=all_actors.actors[forklift_name].h_node)
    center_freq = ant_device_radar_room.waveforms[mode_name_radar].center_freq


# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = 2.99792458e8 / center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density


sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

if type_of_simulation == 'comms':
    ant_device = ant_device_tx1
    mode_name = mode_name_comms
else:
    ant_device = ant_device_radar_room
    mode_name = mode_name_radar

which_mode = ant_device.modes[mode_name] # tell it which mode we want to get respones from
ant_device.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device.waveforms[mode_name].vel_domain
rng_domain = ant_device.waveforms[mode_name].rng_domain
freq_domain = ant_device.waveforms[mode_name].freq_domain
pulse_domain = ant_device.waveforms[mode_name].pulse_domain

print(f'max range: {rng_domain[-1]}')
print(f'velocity window: {vel_domain[-1]-vel_domain[0]}')
fps = 1 / dt

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=which_mode,
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

output_movie_name = os.path.join(paths.output, 'out_warehouse.mp4')


if type_of_simulation == 'comms':
    modeler = ModelVisualization(all_actors,
                                 show_antennas=True,
                                 overlay_results=True,
                                 fps=fps,
                                 freq_domain=freq_domain,
                                 pulse_domain=pulse_domain,
                                 camera_orientation='follow4',
                                 camera_attachment='rx',
                                 output_movie_name=output_movie_name,)
else:
    modeler = ModelVisualization(all_actors,
                                 show_antennas=True,
                                 overlay_results=True,
                                 fps=fps,
                                 vel_domain=vel_domain,
                                 rng_domain=rng_domain,
                                 camera_orientation='follow8',
                                 camera_attachment=radar_room_actor_name,
                                 output_movie_name=output_movie_name,)

modeler.pl.show_grid()
print('running simulation...')

timestamps = np.linspace(0, total_time, num=int(total_time/dt)+1)

for time in tqdm(timestamps):
    # update all coordinate systems
    for actor in all_actors.actors:
        if actor == ped1_name:
            all_actors.actors[actor].coord_sys.pos = path_picked_person.pos_func(time)
            all_actors.actors[actor].coord_sys.rot = path_picked_person.rot_func(time)
        all_actors.actors[actor].update_actor(time=time)


    pem_api_manager.isOK(pem.computeResponseSync())

    if type_of_simulation == 'comms':
        (ret, response) = pem.retrieveP2PResponse(ant_device_tx1.modes[mode_name_comms],
                                              ant_device_rx.modes[mode_name_comms],
                                              RssPy.ResponseType.FREQ_PULSE)
        imData = 20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30))
    else:
        (ret, response_room) = pem.retrieveResponse(ant_device_radar_room.modes[mode_name_radar],
                                              RssPy.ResponseType.RANGE_DOPPLER)
        (ret, response_fork) = pem.retrieveResponse(ant_device_radar_forklift.modes[mode_name_radar],
                                              RssPy.ResponseType.RANGE_DOPPLER)
        if which_radar_response_to_plot == 'room':
            imData = np.rot90(20 * np.log10(np.fmax(np.abs(response_room[0][0]), 1.e-30)))
        else:
            imData = np.rot90(20 * np.log10(np.fmax(np.abs(response_fork[0][0]), 1.e-30)))

    if export_debug:
        if time == 0:
            debug_logs.write_scene_summary(file_name=f'out_0.json')
        debug_camera.generate_image()

    modeler.update_frame(plot_data=imData,plot_limits=[imData.min(),imData.max()]) # update the visualization


modeler.close()
if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')