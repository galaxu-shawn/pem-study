#
# Copyright ANSYS. All rights reserved.
#

import json

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time as walltime
import os
import sys
from scipy.constants import speed_of_light

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, look_at
from pem_utilities.antenna_device import add_single_tx_rx, Waveform
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import *
from pem_utilities.barker_code import BarkerCode
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


center_freq = 10e9
transmit_power = 10e3
rx_noise_db = None
show_modeler=True

# number of frames to simulate, including time step
num_frames = 1
dt = 0.1

corner_reflector1_pos = [4500, 2000,  0]
corner_reflector2_pos = [3000, 0,    1000]
corner_reflector3_pos = [9259, 200, -200]

corner_reflector1_lin = [10, 0,  0]
corner_reflector2_lin = [20, 0,    0]
corner_reflector3_lin = [-10, 0, -0]

# I am setting these rcs values (dbsm) to result in about the same returned power for each distance
corner_reflector1_rcs = 36.825
corner_reflector2_rcs = 29.13
corner_reflector3_rcs = 47.8

barker = BarkerCode(code_length = 13,center_freq=center_freq,
             num_pulses=128,
             pri=100e-6,
             duty_cycle=0.025,
             adc_sampling_rate=6.5e6,
             number_adc_samples=512,
             blanking_period=2000,
             upsample_factor=8,
             transmit_power=transmit_power,
             rx_noise_db=rx_noise_db)
# barker.waveform_dict will be calculated that will be the required input waveform for the percive EM simulation

barker.plot_input_waveform(enhanced_plots=True)



# simulation options
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 4
max_num_trans = 1
ray_density = .01


export_debug = False  # use this to export the movie at the end of the simulation (.gif showing radar camera and response)


# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
# multiple material libraries can be loaded and the indices will be updated accordingly
# mat_manager = MaterialManager(['material_properties_ITU_3.85GHz.json','material_library.json'])
mat_manager = MaterialManager()
# output file will be stored in this directory



#######################################################################################################################
#
# Create Scene, 3 corner reflectors
#
#######################################################################################################################

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

rcs_dbsm = corner_reflector1_rcs
rcs = 10 ** (rcs_dbsm / 10)
prim = CornerReflector(rcs=rcs, wl=speed_of_light/center_freq, orientation='x',
                       is_square=False)  # if use an rcs value, it will override teh length and calculate length from rcs
rcs_dbsm = corner_reflector2_rcs
rcs = 10 ** (rcs_dbsm / 10)
prim2 = CornerReflector(rcs=rcs, wl=speed_of_light/center_freq, orientation='x',
                       is_square=False)  # if use an rcs value, it will override teh length and calculate length from rcs
rcs_dbsm = corner_reflector3_rcs
rcs = 10 ** (rcs_dbsm / 10)
prim3 = CornerReflector(rcs=rcs, wl=speed_of_light/center_freq, orientation='x',
                       is_square=False)  # if use an rcs value, it will override teh length and calculate length from rcs

wl = speed_of_light / center_freq


#dynamic_generator_updates=False will not create a new geometry at each time step, only set to True if you
# want to change the size of the reflector on each update
prim_name = all_actors.add_actor(name='prim_example', generator=prim, mat_idx=mat_manager.get_index('pec'),
                                 target_ray_spacing=wl/4,dynamic_generator_updates=False)
pos1 = corner_reflector1_pos
print(f'Magnitude of pos1: {np.linalg.norm(pos1)}')
all_actors.actors[prim_name].coord_sys.rot = look_at((pos1[0], pos1[1], pos1[2]),(0, 0, 0))
all_actors.actors[prim_name].coord_sys.pos = pos1
all_actors.actors[prim_name].coord_sys.lin = corner_reflector1_lin
all_actors.actors[prim_name].coord_sys.update()

#dynamic_generator_updates=False will not create a new geometry at each time step
prim_name2 = all_actors.add_actor(name='prim_example2', generator=prim2, mat_idx=mat_manager.get_index('pec'),
                                 target_ray_spacing=wl/4,dynamic_generator_updates=False)
pos2 = corner_reflector2_pos
print(f'Magnitude of pos2: {np.linalg.norm(pos2)}')
all_actors.actors[prim_name2].coord_sys.rot = look_at((pos2[0], pos2[1], pos2[2]),(0, 0, 0))
all_actors.actors[prim_name2].coord_sys.pos = pos2
all_actors.actors[prim_name2].coord_sys.lin = corner_reflector2_lin
all_actors.actors[prim_name2].coord_sys.update()
#
#
#dynamic_generator_updates=False will not create a new geometry at each time step
prim_name3 = all_actors.add_actor(name='prim_example3', generator=prim3, mat_idx=mat_manager.get_index('pec'),
                                 target_ray_spacing=wl/4,dynamic_generator_updates=False)

pos3 = corner_reflector3_pos
print(f'Magnitude of pos3: {np.linalg.norm(pos3)}')
all_actors.actors[prim_name3].coord_sys.rot = look_at((pos3[0], pos3[1], pos3[2]),(0, 0, 0))
all_actors.actors[prim_name3].coord_sys.pos = pos3
all_actors.actors[prim_name3].coord_sys.lin = corner_reflector3_lin
all_actors.actors[prim_name3].coord_sys.update()


waveform_dict = barker.waveform_dict
pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
mode_name = 'mode1'
# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)

#################
# Tx antenna

radar_actor_name = all_actors.add_actor() # easier to attach a camera to an actor rather than an antenna
all_actors.actors[radar_actor_name].coord_sys.pos = [0, 0, 0]
all_actors.actors[radar_actor_name].coord_sys.rot = np.eye(3)
all_actors.actors[radar_actor_name].coord_sys.update()
#use dipole antenna for easier radar range calculation
ant_device = add_single_tx_rx(all_actors, waveform, mode_name,ffd_file='dipole.ffd',
                              scale_pattern=.5,parent_h_node=all_actors.actors[radar_actor_name].h_node)
center_freq = ant_device.waveforms[mode_name].center_freq

# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = speed_of_light / center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density

sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 180
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device.modes[mode_name]  # tell it which mode we want to get respones from
ant_device.waveforms[mode_name].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device.waveforms[mode_name].vel_domain
rng_domain = ant_device.waveforms[mode_name].rng_domain
freq_domain = ant_device.waveforms[mode_name].freq_domain
pulse_domain = ant_device.waveforms[mode_name].pulse_domain
output_format = ant_device.waveforms[mode_name].output

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
fps = 10
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)


# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
if show_modeler:
    output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
    modeler = ModelVisualization(all_actors,
                                 show_antennas=True,
                                 overlay_results=False,
                                 camera_orientation=None,
                                 camera_attachment=None,
                                 output_movie_name=output_movie_name)
    modeler.pl.show_grid()


print('running simulation...')

time_stamps = np.linspace(0, num_frames*dt-dt, num_frames)
all_responses = []
for iFrame in tqdm(range(num_frames), disable=True):
    time = time_stamps[iFrame]
    for actor in all_actors.actors:
        # default will update the actors position based on intial position and lin vel
        all_actors.actors[actor].update_actor(time=time)
    start_sim_time = walltime.time()
    pem_api_manager.isOK(pem.computeResponseSync())
    end_sim_time = walltime.time()
    print(f'Simulation Time: {(end_sim_time - start_sim_time)*1e3} mseconds')
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)
    all_responses.append(response)
    if show_modeler:
        modeler.update_frame()

all_responses = np.array(all_responses)
# single frame, single tx,rx, and all pulses. this is frequency domain response
fd_response = all_responses[0,0,0] # [frame][tx][rx][pulse][freq]

# use_perceive_em = True # for debugging, will use analytic repsonse from a single corner reflector
# # Compute Simple Pulse Response of Scene
# if not use_perceive_em:
#     # analytic response for a fixed distance
#     test_distance = 5000
#     # Calculate wavenumbers at all frequencies
#     ak0_perm = 2*np.pi * (f_if_mhz / 1000 + center_freq) / c_mpns  # [1/m]
#     fd_response = np.exp(-1j * 2 * ak0_perm * test_distance)  # H(jw) of scene


barker.process_received_signal(fd_response)

barker.plot_output_waveform(plot_freq_response=True,
                             plot_time_domain_signal=True,
                             plot_adc_samples=True,
                             plot_resampled_signal=True,
                             plot_matched_filter_results=True,
                             pulse_idx=0)

barker.plot_output_waveform(plot_freq_response=True,
                             plot_time_domain_signal=True,
                             plot_adc_samples=True,
                             plot_resampled_signal=True,
                             plot_matched_filter_results=True,
                             pulse_idx=0,
                             dpi=100,
                             figsize=(10, 8),
                             style='seaborn-v0_8-whitegrid',
                             dynamic_range_db=60,
                             annotate_peaks=True,
                             enhanced_plots=True)

if show_modeler:
    modeler.close()
