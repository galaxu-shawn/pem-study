#
# Copyright ANSYS. All rights reserved.
#
#######################################
# Example of P2P coupling between a pedestrian and a building using multiple animated people

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
from pem_utilities.antenna_device import Waveform, add_single_rx, add_single_tx, enable_coupling, add_antenna_device_from_json
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.router import Pick_Path
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.ofdm import OFDMChannel
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 

# simulation speed
fps = 30
T = 10 # total time in seconds

tx_pos = (1, -7.5, 0.9)

# waveform to plot
center_freq = 2.4e9
num_freqs = 512
bandwidth = 300e6
cpi_duration = 100e-3
num_pulse_CPI = 100

# simulation parameters
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = 1

# The path the person takes walking through the building can be defined by the user or by a predefined path. If
# custom_user_path is set to True, the user will be prompted to define the path by clicking on the screen. If False,
# the path will be loaded from a predefined file. The path is saved as a numpy array in the output directory.
custom_user_path = False

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()


# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()  # all actors, using the same material library for everyone

#add a multi part actor from json
house_name = all_actors.add_actor(name='house',
                                  filename=os.path.join(paths.models,'Home_P2P/Home_P2P.json'),
                                  transparency=0.7)  #

if custom_user_path:
    pre_picked_path = None
else:
    pre_picked_path = [[8.85048141, -1.84903715, 0],
                       [8.81217011, -3.517996, 0],
                       [8.16490271, -4.08738815, 0],
                       [7.24634182, -4.24928382, 0],
                       [5.76819156, -4.5098067, 0],
                       [4.47601666, -4.90547279, 0],
                       [2.94367776, -3.90116165, 0],
                       [1.96960152, -3.35601733, 0],
                       [1.9736437, -1.69821011, 0],
                       [1.9908724, -0.68093924, 0]]
path_picked_person = Pick_Path()
path_picked_person.custom_path(mesh_list=all_actors.actors[house_name].get_mesh(), pre_picked=pre_picked_path,
                               speed=1.0,snap_to_surface=False)

#add a multi part actor from dae
ped1_name = all_actors.add_actor(filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'), target_ray_spacing=0.1,
                                 scale_mesh=1)
all_actors.actors[ped1_name].velocity_mag = 1.0
all_actors.actors[ped1_name].use_linear_velocity_equation_update = False



dt = 1 / fps
numFrames = int(T / dt)
#Original template for motion was created

timestamps = np.linspace(0, T, numFrames)

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
ant_device_tx = add_antenna_device_from_json(all_actors,
                                             'Indoor_P2P_1tx_5p5GHz_ffd.json',
                                             mode_name,
                                             pos=tx_pos,
                                             rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True),
                                             scale_pattern=0.75,
                                             parent_h_node=all_actors.actors[house_name].h_node)

######################
# Rx on walking user
####################
# the antenna can be attached to a specific body part, or just relative to the root location of the actor
node_to_attach_rx = all_actors.actors[ped1_name].parts['mixamorig_LeftHand'].h_node
# node_to_attach_rx = all_actors.actors[ped1_name].h_node
ant_device_rx1 = add_antenna_device_from_json(all_actors,'Indoor_P2P_2rx_5p5GHz_ffd.json',mode_name,
                              pos=(0.05,0,0.05),
                              rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True),
                              scale_pattern=0.5,
                              parent_h_node=node_to_attach_rx)

enable_coupling(mode_name,ant_device_tx, ant_device_rx1)

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

ofdm = OFDMChannel(SNRdb=18,K=64, P=8, mu=4)

for iFrame in tqdm(range(numFrames), disable=True):
    time = iFrame * dt
    # update all coordinate systems
    for actor in all_actors.actors:
        # Actors can have actor_type value, this will handle sub-parts of the actors. For example, if we have a vehicle
        # it will update the tires rotation and velocity, and the vehicle linear velocity based on its orientation
        # if actor is other, it will use the lin/rot/ang values to update the position of the actor
        if actor == ped1_name:
            all_actors.actors[actor].coord_sys.pos = path_picked_person.pos_func(time)
            all_actors.actors[actor].coord_sys.rot = path_picked_person.rot_func(time)
        all_actors.actors[actor].update_actor(time=time)

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes[mode_name],
                                              ant_device_rx1.modes[mode_name],
                                              RssPy.ResponseType.FREQ_PULSE)


    # all_results is a 4D array with dimensions [num_tx, num_rx, num_pulses, num_freqs]
    response = np.array(response)

    test = ofdm.ofdm_example(response[0,0,0])
    figures_dict = ofdm.generate_plots()

    # these are all the available figure, # 'qam_constellation', 'pilot_data_carriers', 'qam_constellation_2d', 'qam_constellation_3d',
    pilot_fig = figures_dict['pilot_data_carriers']
    received_constellation_fig = figures_dict['received_constellation']
    tx_rx_signals_fig = figures_dict['tx_rx_signals']
    qam_constellation_fig = figures_dict['qam_constellation']
    hard_decision_demapping_fig = figures_dict['hard_decision_demapping']
    final_received_constellation_fig = figures_dict['final_received_constellation']

    # add figure to the modeler window for visualization
    h_chart = pv.ChartMPL(hard_decision_demapping_fig, size=(0.275, 0.375), loc=(0.7, 0.6))
    modeler.pl.add_chart(h_chart)
    modeler.update_frame()
    plt.clf() # clear all figures to avoid memory issues
    plt.close()

modeler.close()
# all_results is a 5D array with dimensions [time_idx, num_tx, num_rx, num_pulses, num_freqs]
print('Done')

