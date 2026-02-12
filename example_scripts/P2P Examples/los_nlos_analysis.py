#
# Copyright ANSYS. All rights reserved.
#
#######################################


import copy
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import pyvista as pv
import os
import sys
import scipy
from scipy.interpolate import interp1d

from pem_utilities.animation_selector_CMU import select_CMU_animation
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_rx, add_single_tx, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.router import Pick_Path
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import Plane
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 


fps = 10
total_time = 10
dt = 1 / fps
num_frames = int(total_time / dt)
timestamps = np.linspace(0, total_time, num_frames)


tx_pos = [-10, 0, 0]
save_results = False

# waveform to plot
center_freq = 3.85e9
num_freqs = 512
bandwidth = 300e6
cpi_duration = 100e-3
num_pulse_CPI = 100

# simulation parameters
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = 1


export_debug = True

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()


# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()  # all actors, using the same material library for everyone
prim = Plane(i_size=10,j_size=10, num_i=10, num_j=10, orientation=[1, 0, 0])

wl = 3e8 / center_freq
prim_name = all_actors.add_actor(name='prim_example', generator=prim, mat_idx=mat_manager.get_index('pec'),
                                 target_ray_spacing=wl,dynamic_generator_updates=False) #dynamic_generator_updates=False will not create a new geometry at each time step


#add actor where Tx will be attached
tx_actor_name = all_actors.add_actor()
all_actors.actors[tx_actor_name].velocity_mag = 0.0
all_actors.actors[tx_actor_name].coord_sys.pos = tx_pos
all_actors.actors[tx_actor_name].coord_sys.rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True)
all_actors.actors[tx_actor_name].coord_sys.update()

#add actor where Rx will be attached
rx_actor_name = all_actors.add_actor()
all_actors.actors[rx_actor_name].velocity_mag = 1.0
all_actors.actors[rx_actor_name].coord_sys.pos = [10, 0, 0]
all_actors.actors[rx_actor_name].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
all_actors.actors[rx_actor_name].coord_sys.update()

######################
# Define the waveform to be used

mode_name = 'mode1' # name of mode so we can reference it in post processing
# input_power_dbm = 10.0 # dBm
# # convert to watts
# input_power_watts = 10 ** ((input_power_dbm - 30) / 10)
input_power_watts = 1
# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more
# flexibility and control over the parameters, without having to create/modify a json file. This is the same as the
# results as if the same parameters had been created in a json file.
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
    "tx_incident_power": input_power_watts}

pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)


######################
# Add 2 antenna to teh scene, one tx and one rx
####################

# add a single Tx antenna to the scene at tx_pos
ant_device_tx = add_single_tx(all_actors,waveform,mode_name,
                              ffd_file='dipole.ffd',
                              scale_pattern=0.75,
                              parent_h_node=all_actors.actors[tx_actor_name].h_node)

######################
# Rx on walking user
####################
# the antenna can be attached to a specific body part, or just relative to the root location of the actor


ant_device_rx = add_single_rx(all_actors,waveform,mode_name,
                              ffd_file='dipole.ffd',
                              scale_pattern=0.5,
                              parent_h_node=all_actors.actors[rx_actor_name].h_node)

enable_coupling(mode_name,ant_device_tx, ant_device_rx)

# rays can be defined as density or spacing, this is the conversion between the two, if ray_density is set, then
freq_center = ant_device_tx.waveforms[mode_name].center_freq
lambda_center = 2.99792458e8 / freq_center
ray_spacing = np.sqrt(2) * lambda_center / ray_density

print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
sim_options.auto_configure_simulation()

which_mode = ant_device_tx.modes[mode_name] # tell it which mode we want to get respones from
ant_device_tx.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device_tx.waveforms[mode_name].vel_domain
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain
freq_domain = ant_device_tx.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx.waveforms[mode_name].pulse_domain

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)

output_movie_name = os.path.join(paths.output, 'out_vis_indoor_wireless.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=None,
                             vel_domain=None, overlay_results=False,
                             fps=fps,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)

modeler.pl.show_grid()

all_results = []
print('running simulation...')

for idx_frame in tqdm(range(num_frames), disable=True):
    time = idx_frame * dt
    # update all coordinate systems
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes[mode_name],
                                              ant_device_rx.modes[mode_name],RssPy.ResponseType.FREQ_PULSE)


    # all_results is a 4D array with dimensions [num_tx, num_rx, num_pulses, num_freqs]
    response = np.array(response)
    # define as numpy array and accumulate in all_results
    # all_results is a 5D array with dimensions [num_time_stamps num_tx, num_rx, num_pulses, num_freqs]
    all_results.append(response)


    # exporting radar camera images
    if export_debug:
        debug_logs.write_scene_summary(file_name=f'out_{idx_frame}.json')
        debug_camera.generate_image()

    f, ax = plt.subplots(tight_layout=True)
    plt.plot(freq_domain / 1e9, 20 * np.log10(abs((response[0,0,0]))), color='red', label='Kitchen Room Rx')
    plt.xlabel('Frequency [GHz]', fontsize=15)
    plt.ylabel('S21[dB]', fontsize=15)
    plt.legend(loc='upper right')
    ax.set_ylim(-100, -30)
    h_chart = pv.ChartMPL(f, size=(0.275, 0.375), loc=(0.7, 0.6))
    modeler.pl.add_chart(h_chart)
    modeler.update_frame()
    plt.clf()
    plt.close()

modeler.close()
all_results = np.array(all_results)
# all_results is a 5D array with dimensions [time_idx, num_tx, num_rx, num_pulses, num_freqs]
print('Done')

if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')

if save_results:
    # results are generated with the dimensions [time_idx, num_tx, num_rx, num_pulses, num_freqs]
    # where each time_idx is when we call the simulator. For each simulation we get N (num_pulses) channel soundings
    # as an output. The pulse_domain will tell us the time spacing of each channel sounding,
    # if we wanted a continuous time domain output, we could set the spacing of timestamps to be the same as the CPI
    # (or total time shown in pulse_domain)

    np.save(os.path.join(paths.output, 'all_results.npy'), all_results)
    np.save(os.path.join(paths.output, 'freq_domain.npy'), freq_domain) # what frequency samples were used
    np.save(os.path.join(paths.output, 'pulse_domain.npy'), pulse_domain) # time domain of the pulses
    np.save(os.path.join(paths.output, 'simulation_timestamps.npy'), timestamps) # velocity domain
    # save results as mat file
    scipy.io.savemat(os.path.join(paths.output, 'all_results.mat'), {'all_results': all_results,
                                                                        'freq_domain': freq_domain,
                                                                        'pulse_domain': pulse_domain,
                                                                        'simulation_timestamps': timestamps})