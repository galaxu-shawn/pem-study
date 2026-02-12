#
# Copyright ANSYS. All rights reserved.
#

import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import PIL.Image
import os
import sys

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.animation_selector_CMU import select_CMU_animation
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


fps = 30
total_time = 10

radar_pos = [1.2, -0.2, .5]
radar_rot = euler_to_rot(phi=180, theta=0, psi=0)

person_init_pos = [0, 0, 0]
person_init_rot = euler_to_rot(phi=0, theta=0, psi=0)

# waveform parameters for radar
center_freq = 77e9
num_freqs = 18
bandwidth = 900e6
cpi_duration = 25
num_pulse_CPI = 128

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .1

export_debug = True  # use this to export the movie at the end of the simulation (.gif showing radar camera and response)

#######################################################################################################################
# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()


# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

# empty actor, we will use this as a parent node for the radar platform
actor_radar_name = all_actors.add_actor(name='radar_actor')

cmu_filename, cmu_description = select_CMU_animation(file_name='82_05')  # specific selection
person_name = all_actors.add_actor(filename=cmu_filename, target_ray_spacing=0.05, scale_mesh=1)
all_actors.actors[person_name].coord_sys.pos = (-1.5, 0, 0.)
all_actors.actors[person_name].coord_sys.rot = euler_to_rot(phi=0, order='zyz',deg=True)

# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more flexibility
# and control over the parameters, without having to create/modify a json file
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
mode_name = 'mode1'

# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)

#################
# Tx antenna

radar_actor_name = all_actors.add_actor() # easier to attach a camera to an actor rather than an antenna
all_actors.actors[radar_actor_name].coord_sys.pos = radar_pos
all_actors.actors[radar_actor_name].coord_sys.rot = radar_rot
all_actors.actors[radar_actor_name].coord_sys.update()
ant_device = add_single_tx_rx(all_actors, waveform, mode_name,scale_pattern=.5,parent_h_node=all_actors.actors[radar_actor_name].h_node)
center_freq = ant_device.waveforms[mode_name].center_freq

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



# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device.modes[mode_name]  # tell it which mode we want to get respones from
ant_device.waveforms[mode_name].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device.waveforms[mode_name].vel_domain
rng_domain = ant_device.waveforms[mode_name].rng_domain
freq_domain = ant_device.waveforms[mode_name].freq_domain
pulse_domain = ant_device.waveforms[mode_name].pulse_domain



responses = []
cameraImages = []
# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes['mode1'],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)
# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(paths.output, 'out_vis_breathing.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=rng_domain,
                             vel_domain=vel_domain,
                             fps=fps,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)


dt = 1 / fps
num_frames = int(total_time / dt)

print('running simulation...')
for idx in tqdm(range(num_frames), disable=True):
    time = idx * dt
    # update all coordinate systems
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)
        if actor == person_name:
            # this would add an addition theta rotation to the persons chest
            # all_actors.actors[actor].parts['Chest'].coord_sys.rot = euler_to_rot(phi=0, theta=45*np.sin(time), order='zxz', deg=True)
            pos = all_actors.actors[actor].parts['Chest'].coord_sys.pos
            all_actors.actors[actor].parts['Chest'].coord_sys.pos = [pos[0]+0,pos[1]+0,pos[2]+.01*np.sin(2*np.pi*time)]
            all_actors.actors[actor].parts['Chest'].coord_sys.update()

    # This is not required, but will write out scene summary of the simulation setup,
    # only writing out first and last frame to limit number of files, useful for debugging.
    if idx == 0:
        debug_logs.write_scene_summary(file_name=f'out_{idx}.json')

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)
    # calculate response in dB to overlay in pyvista plotter
    imData = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))
    modeler.mpl_ax_handle.set_data(imData)  # update pyvista matplotlib plot

    # exporting radar camera images
    if export_debug:
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{idx}.png') # write current image to file
        responses.append(response)

    modeler.update_frame()
    if idx==0:
        modeler.pl.show_grid()
modeler.close()

# post-process images into gif. This will take the radar camera and range doppler response and place them side by
# side in a gif
if export_debug:
    # debug_camera.write_camera_to_gif(file_name='camera.gif')
    # debug_camera.write_response_to_gif(responses,file_name='response.gif',rng_domain=rng_domain,vel_domain=vel_domain)
    debug_camera.write_camera_and_response_to_gif(responses,
                                                  file_name='camera_and_response.gif',
                                                  rng_domain=rng_domain,
                                                  vel_domain=vel_domain)
