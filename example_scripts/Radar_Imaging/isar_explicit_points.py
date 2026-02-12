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

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.materials import MaterialManager
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.post_processing_radar_imaging import frequency_azimuth_to_isar_image
from pem_utilities.antenna_device import AntennaDevice, Waveform
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

# simulation parameters
max_num_refl = 3
max_num_trans = 0
ray_density = 0.1

# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 10.0e9
num_freqs = 59
bandwidth = 584e6


export_debug = False


#######################################################################################################################
# End Input Parameters
#######################################################################################################################

# for explicit angle simulation these are not used
cpi_duration = 1e-3
num_pulse_CPI = 3


mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()


actor_target_name = all_actors.add_actor(filename=os.path.join(paths.models,'rv.stl'),
                                 mat_idx=mat_manager.get_index('pec'),
                                 color='grey',
                                 target_ray_spacing=0.05)

all_actors.actors[actor_target_name].coord_sys.rot = euler_to_rot(phi=0)
all_actors.actors[actor_target_name].coord_sys.update()
actor_radar_name = all_actors.add_actor()

###########Define Sample Locations ########

phi_points = np.linspace(-phi_aspect_deg/2, phi_aspect_deg/2, num=num_phi_samples)
x = dist_from_target * np.cos(np.deg2rad(phi_points))
y = dist_from_target * np.sin(np.deg2rad(phi_points))
all_radar_positions = np.vstack((x,y,np.zeros(len(x)))).T # Z at zero

# point radar back at origin
all_radar_rotations = euler_to_rot(phi=180-phi_points,deg=True)


# simulation parameters

waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "FREQPULSE",
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


########################## Setup Radar Platform  ##########################

# we are not loading an antenna device from file, so this will just create a blank antenna device,
# as a child of radar_device. Instead of loading from a file, this will be manually setup using a combination
# of direct API calls and the AntennaDevice class
ant_device = AntennaDevice(file_name=None, parent_h_node=all_actors.actors[actor_radar_name].h_node)
ant_device.initialize_device()

waveform = Waveform(waveform_dict)
ant_device.waveforms[mode_name] = waveform
# configure radar mode
h_mode = RssPy.RadarMode()
ant_device.modes[mode_name] = h_mode
pem_api_manager.isOK(pem.addRadarMode(h_mode, ant_device.h_device))

tx_dict = {
        "type": "parametric",
        "operation_mode": "tx",
        "polarization": "VERTICAL",
        "hpbwHorizDeg": 30.5,
        "hpbwVertDeg": 60.5,
        "position": [0.0, 0.0, 0.0]
    }

rx_dict = {
        "type": "parametric",
        "operation_mode": "rx",
        "polarization": "VERTICAL",
        "hpbwHorizDeg": 30.5,
        "hpbwVertDeg": 60.5,
        "position": [0.0, 0.0, 0.0]
    }

antennas_dict = {"Tx1": tx_dict, "Rx1": rx_dict}
ant_device.add_antennas( mode_name=mode_name, load_pattern_as_mesh=True, scale_pattern=100,antennas_dict=antennas_dict)
ant_device.set_mode_active(mode_name)
ant_device.add_mode(mode_name)

# add antennas to scene for visulization
for each in ant_device.all_antennas_properties:
    name = all_actors.add_actor(name=each, actor=ant_device.all_antennas_properties[each]['Actor'])

ray_spacing = 0.2

sim_options = SimulationOptions(center_freq=ant_device.waveforms[mode_name].center_freq)
sim_options.ray_spacing = ray_spacing
sim_options.ray_shoot_method = "sbr"  # use grid or sbr method for ray shooting
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = -1
sim_options.field_of_view = 180
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()
# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(pem.getLastWarnings())

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
    debug_camera = DebuggingCamera(hMode=ant_device.modes[modes[0]],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=fps)

# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)


responses = []
plateform = {}
plateform['freq'] = freq_domain
plateform['phi']=phi_points

print('running simulation...')
for idx in tqdm(range(len(all_radar_positions))):

    # we are just updating the positions, not using any velocities to move the radar
    all_actors.actors[actor_radar_name].coord_sys.pos = all_radar_positions[idx]
    all_actors.actors[actor_radar_name].coord_sys.rot = all_radar_rotations[idx]
    all_actors.actors[actor_radar_name].coord_sys.update()

    modeler.update_frame() # update visualization

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)

    # response is stored as [tx_idx,rx_idx,pulse_idx,freq_idx], we will just use center pulse (idx=1)
    # im_data = np.abs(response[0,0,1])
    responses.append(response[0,0,1])
    # modeler.mpl_ax_handle.set_data(im_data)  # update pyvista matplotlib plot

    if export_debug:
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
modeler.close()

print('starting post processing ')

responses = np.array(responses)
isar_image = isar_image = 20*np.log10(np.abs(frequency_azimuth_to_isar_image(responses, plateform, size=(512, 512), window='hann')))
# ToDo, not completed yet
# isar_image = isar_2d(responses, freq_domain, phi_points,function='db',window='hann') #wip
print(np.min(isar_image))
print(np.max(isar_image))
plt.close('all')

fig, ax = plt.subplots()
ax.imshow(isar_image, cmap='jet',clim=[-272,-128])
plt.show()
