# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

This uses the vehicles velocities to create the image

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

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.post_processing_radar_imaging import frequency_azimuth_to_isar_image
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


#######################################################################################################################
# Input Parameters
#######################################################################################################################

dist_from_target = 1010

num_phi_samples = 56
phi_aspect_deg = 3.327

angle_update_per_look = 1 # deg, rotate the target by this much per look
num_looks = 90 # total rotation will be angle_update_per_look*num_looks

# waveform parameters for radar
center_freq = 10e9
num_freqs = 59
bandwidth = 584e6


# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .1

export_debug = False


#######################################################################################################################
# End Input Parameters
#######################################################################################################################

# for explicit angle simulation these are not used
cpi_duration = 1 # 1 second for easy math
num_pulse_CPI = num_phi_samples
all_looks = np.linspace(0,num_looks*angle_update_per_look,num_looks)

# output file will be stored in this directory
output_path = paths.output
if not os.path.exists(output_path):
    os.makedirs(output_path)
os.path.abspath(output_path)

mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()


actor_target_name = all_actors.add_actor(filename=os.path.join(paths.models,'rv.stl'),
                                 mat_idx=mat_manager.get_index('pec'),
                                 color='grey',
                                 target_ray_spacing=0.01)

all_actors.actors[actor_target_name].coord_sys.rot = euler_to_rot(phi=45)
# cover the entire phi aspect per CPI, angular velocity is aspect/CPI (rad/s) around z-axis
all_actors.actors[actor_target_name].coord_sys.ang = [0, 0, np.deg2rad(phi_aspect_deg)]
all_actors.actors[actor_target_name].coord_sys.update()

###########Define Radar Location ########
actor_radar_name = all_actors.add_actor()
all_actors.actors[actor_radar_name].coord_sys.pos = [dist_from_target,0,0]
all_actors.actors[actor_radar_name].coord_sys.rot = euler_to_rot(phi=180) # point back at origin
all_actors.actors[actor_radar_name].coord_sys.update()

# simulation parameters
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "RangeDoppler",
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
waveform = Waveform(waveform_dict)

########################## Setup Radar Platform  ##########################

ant_device = add_single_tx_rx(all_actors, waveform, mode_name,scale_pattern=.5,
                              parent_h_node=all_actors.actors[actor_radar_name].h_node,
                              doppler_pixels=512,range_pixels=1024)
# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = 2.99792458e8 / ant_device.waveforms[mode_name].center_freq
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
# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(pem.getLastWarnings())


# get response domains

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device.modes[mode_name]  # tell it which mode we want to get respones from
ant_device.waveforms[mode_name].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device.waveforms[mode_name].vel_domain
rng_domain = ant_device.waveforms[mode_name].rng_domain
freq_domain = ant_device.waveforms[mode_name].freq_domain
pulse_domain = ant_device.waveforms[mode_name].pulse_domain

if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=10)

# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(output_path, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)




print('running simulation...')
for look in tqdm(all_looks):
    all_actors.actors[actor_target_name].coord_sys.rot = euler_to_rot(phi=look)
    all_actors.actors[actor_target_name].coord_sys.update()
    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)
    response = response[0,0]
    response = np.array(response)
    isar_image = 20 * np.log10(np.abs(response))
    modeler.update_frame(plot_data=isar_image,plot_limits=[isar_image.min(),isar_image.max()])

modeler.close()

print('starting post processing ')

response = np.array(response)
isar_image = 20*np.log10(np.abs(response))
# ToDo, not completed yet
# isar_image = isar_2d(responses, freq_domain, phi_points,function='db',window='hann') #wip
print(np.min(isar_image))
print(np.max(isar_image))
plt.close('all')

fig, ax = plt.subplots()
ax.imshow(isar_image, cmap='jet',clim=[-272,-128])
plt.show()
