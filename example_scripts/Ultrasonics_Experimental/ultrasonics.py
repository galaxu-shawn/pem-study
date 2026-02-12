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
from scipy.io.wavfile import read

from example_scripts.pem_utilities.primitives import Cylinder

sys.path.append("..")  # directory above this directory is where api_core exists

#######################################################################################################################
# IMPORTANT! edit api_settings.json to use the correct API, P2P API is needed for this script to run
#######################################################################################################################
import pem_utilities.pem_core as pem_core

api = pem_core.api

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import Cylinder


fps = 1
distances = [0.5,1,2,3,4,5,6]

tx_pos = [0, 0, 0]

audio_bw = 17e3
audio_center = 50e3
freq_step = 28.33

speed_of_sound = 343
speed_of_light = 299792458

# waveform to plot
center_freq = speed_of_light/speed_of_sound*audio_center
bandwidth = speed_of_light/speed_of_sound*audio_bw
num_freqs = int(bandwidth/(speed_of_light/speed_of_sound*freq_step))

cpi_duration = 100e-3 # not important for this simulation, we will only use center pulse
num_pulse_CPI = 3


# simulation parameters
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = 1


export_debug = True

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()


# output file will be stored in this directory
output_path = '../output/'
os.makedirs(output_path, exist_ok=True)


# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()  # all actors, using the same material library for everyone
prim = Cylinder(radius=0.8,height=1,num_theta=45,orientation=[0,0,1])

wl = 3e8 / center_freq
prim_name = all_actors.add_actor(name='prim_example', generator=prim, mat_idx=mat_manager.get_index('pec'),
                                 target_ray_spacing=wl,dynamic_generator_updates=False) #dynamic_generator_updates=False will not create a new geometry at each time step


all_actors.actors[prim_name].coord_sys.pos = [0.5,0,0]
all_actors.actors[prim_name].coord_sys.update()

#add actor where Tx will be attached
tx_actor_name = all_actors.add_actor()
all_actors.actors[tx_actor_name].velocity_mag = 0.0
all_actors.actors[tx_actor_name].coord_sys.pos = tx_pos
all_actors.actors[tx_actor_name].coord_sys.rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True)
all_actors.actors[tx_actor_name].coord_sys.update()



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

ant_device = add_single_tx_rx(all_actors, waveform, mode_name,ffd_file='isotropic_theta.ffd',scale_pattern=.5,parent_h_node=all_actors.actors[tx_actor_name].h_node)


# rays can be defined as density or spacing, this is the conversion between the two, if ray_density is set, then
freq_center = ant_device.waveforms[mode_name].center_freq
lambda_center = 2.99792458e8 / freq_center
ray_spacing = np.sqrt(2) * lambda_center / ray_density

print(api.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
sim_options.auto_configure_simulation()

which_mode = ant_device.modes[mode_name] # tell it which mode we want to get respones from
ant_device.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device.waveforms[mode_name].vel_domain
rng_domain = ant_device.waveforms[mode_name].rng_domain
freq_domain = ant_device.waveforms[mode_name].freq_domain
pulse_domain = ant_device.waveforms[mode_name].pulse_domain

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=output_path)

output_movie_name = os.path.join(output_path, 'out_vis_indoor_wireless.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=None,
                             vel_domain=None,
                             overlay_results=False,
                             fps=fps,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)

modeler.pl.show_grid()

all_results = []
print('running simulation...')

for d in tqdm(distances):

    all_actors.actors[prim_name].coord_sys.pos = [d, 0, 0]
    all_actors.actors[prim_name].coord_sys.update()

    pem_core.isOK(api.computeResponseSync())
    (ret, response) = api.retrieveResponse(ant_device.modes[mode_name],pem_core.RssPy.ResponseType.FREQ_PULSE)


    # all_results is a 4D array with dimensions [num_tx, num_rx, num_pulses, num_freqs]
    response = np.array(response)

    response_td = np.fft.fft(response[0,0,0])
    # define as numpy array and accumulate in all_results
    # all_results is a 5D array with dimensions [num_time_stamps num_tx, num_rx, num_pulses, num_freqs]
    all_results.append(response)


    # exporting radar camera images
    if export_debug:
        debug_logs.write_scene_summary(file_name=f'out.json')
        debug_camera.generate_image()

    f, ax = plt.subplots(tight_layout=True)
    plt.plot(freq_domain / 1e9, 20 * np.log10(abs(response_td)), color='red', label='d')
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

