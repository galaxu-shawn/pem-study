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

dist_from_target = 1000

num_phi_samples = 60 # must be divisible by num_batches, how many samples to take over phi_aspect_deg
phi_aspect_deg = 3 # observed aspect of the target in degrees

# number of batches to divide the phi_aspect_deg into seperate simulation, each simulation uses ADP
# num_batches=1 will be the same as using a single simulation with ADP to create an image. if num_batches=num_phi_samples
# it will be a full pulse-by-pulse simulation, where ADP is effectively not used.
num_batches = 60


angle_update_per_look = 1 # deg, rotate the target by this much per look, independent of phi_aspect_deg for creating video
num_looks = 90 # total rotation will be angle_update_per_look*num_looks,

# waveform parameters for radar
center_freq = 10e9
num_freqs = 59
bandwidth = 584e6


# simulation options
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .1

export_debug = False
show_diff = False # only show the output as the difference between results
show_freq_response=False
#######################################################################################################################
# End Input Parameters
#######################################################################################################################

# for explicit angle simulation these are not used
cpi_duration = 1 # 1 second for easy math
num_pulse_CPI = num_phi_samples
all_looks = np.linspace(0,num_looks*angle_update_per_look,num_looks+1)


all_looks_single_image = np.linspace(0,phi_aspect_deg,num_phi_samples)


mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()


# actor_target_name = all_actors.add_actor(filename=os.path.join(model_path,'/Sphere_1meter_rad.stl'),
#                                  mat_idx=mat_manager.get_index('pec'),
#                                  color='grey',
#                                  target_ray_spacing=0.01)
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
    "output": "FreqPulse",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_pulse_CPI,
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "FIRST_CHIRP"}
pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
mode_name = 'mode1'
waveform = Waveform(waveform_dict)


waveform2_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": int(num_phi_samples/num_batches),
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "FIRST_CHIRP"} #
pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
mode2_name = 'mode2'
waveform2 = Waveform(waveform2_dict)



########################## Setup Radar Platform  ##########################

ant_device = add_single_tx_rx(all_actors, waveform, mode_name,
                              range_pixels=512,
                              doppler_pixels=512,
                              scale_pattern=.5,
                              parent_h_node=all_actors.actors[actor_radar_name].h_node)

# I could use the same device and add a second mode, but this also works, adding a second radar device
ant_device2 = add_single_tx_rx(all_actors, waveform2, mode2_name,
                              range_pixels=512,
                              doppler_pixels=512,
                              scale_pattern=.5,
                              parent_h_node=all_actors.actors[actor_radar_name].h_node)



pem_api_manager.isOK(pem.setPrivateKey("RayShootGrid", "SBR," + str(0.01)))

ray_spacing = 0.2

sim_options = SimulationOptions(center_freq=ant_device.waveforms[mode_name].center_freq)
sim_options.ray_spacing = ray_spacing
sim_options.ray_shoot_method = "sbr"  # use grid or sbr method for ray shooting
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 180
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
output_movie_name = os.path.join(paths.output, 'out_vis_hann_4batches.mp4')
factor = 1
if show_diff:
    factor = 2

if show_freq_response:
    output_shape = (num_pulse_CPI, num_freqs*factor)
else:
    output_shape = (512, 512*factor)



modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=10,
                             shape=output_shape,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_video_size=(1280, 720),
                             figure_size=(1, 1),
                             output_movie_name=output_movie_name)
# figure size is percentage of window size, 1,1 fills entire modeler window (geometry will be obscured).

all_images_adp = []
all_images_pbp = []
plateform = {}
plateform['freq'] = freq_domain
plateform['phi']=all_looks_single_image

print('running simulation...')
for look in tqdm(all_looks):

    # single simulation to generate image using ADP for the entire phi_aspect_deg observation angle.
    # this produces a clean image without ray jitter.
    all_actors.actors[actor_target_name].coord_sys.rot = euler_to_rot(phi=look,theta=45)
    all_actors.actors[actor_target_name].coord_sys.ang = [0, 0, np.deg2rad(phi_aspect_deg)]
    all_actors.actors[actor_target_name].coord_sys.update()
    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)

    isar_image = 20*np.log10(np.abs(frequency_azimuth_to_isar_image(response[0,0], plateform, size=(512, 512), window='hann')))
    all_images_adp.append(isar_image)

    # This section will create an image from the second radar device, where we will use pulse by pulse simulation
    # to compare the results. This also will allow the pulse-by-pulse simulation to be split into sub-batches,
    # where each batch will use ADP. For example, if this number is 2, then the phi_aspect_deg will be divided into
    # 2 simulations, each using ADP to create an image. If this was set to 1 it would create a single image using ADP
    # for the entire range of aspect angles. If this was set to num_batches==num_phi_samples, then we would run a
    # seperate simulation for every single pulse, not using ADP at all.
    for batch in range(num_batches):
        step_size = phi_aspect_deg/(num_batches-1)
        sub_look = step_size*batch
        all_actors.actors[actor_target_name].coord_sys.rot = euler_to_rot(phi=sub_look+look)
        all_actors.actors[actor_target_name].coord_sys.ang = [0, 0, np.deg2rad(phi_aspect_deg/(num_batches-1))]
        all_actors.actors[actor_target_name].coord_sys.update()
        pem_api_manager.isOK(pem.computeResponseSync())
        (ret, response_single) = pem.retrieveResponse(ant_device2.modes[mode2_name], RssPy.ResponseType.FREQ_PULSE)
        if batch == 0:
            responses = np.array(response_single[0, 0])
        else:
            responses = np.vstack((responses, response_single[0,0]))

    isar_image2 = 20*np.log10(np.abs(frequency_azimuth_to_isar_image(response[0,0], plateform, size=(512, 512), window='hann')))
    all_images_pbp.append(isar_image2)

    # compare results by putting them in the same image to display
    if show_freq_response:
        show_freq_response_both = np.hstack((response[0,0], responses))
        show_freq_response_both = np.real(show_freq_response_both)
        modeler.update_frame(plot_data=show_freq_response_both, plot_limits=[show_freq_response_both.min(), show_freq_response_both.max()])
    else:
        if not show_diff:
            isar_both = np.hstack((isar_image, isar_image2))
            modeler.update_frame(plot_data=isar_both, plot_limits=[isar_image.max() - 80, isar_image.max()])
        else:
            isar_diff = isar_image - isar_image2
            modeler.update_frame(plot_data=isar_diff, plot_limits=[isar_diff.max() - 80, isar_diff.max()])


modeler.close()

# plot CDF of the results.

import matplotlib.animation as animation

# sort the data:
print('sorting data...')
adp_sorted = np.sort(all_images_adp[0].flatten())
pbp_sorted = np.sort(all_images_pbp[0].flatten())

# calculate the proportional values of samples
p_adp = 1. * np.arange(len(adp_sorted)) / (len(adp_sorted) - 1)
p_pbp = 1. * np.arange(len(pbp_sorted)) / (len(pbp_sorted) - 1)

f, ax = plt.subplots(tight_layout=True)
line_adp, = ax.plot(adp_sorted,p_adp,label='ADP', color='red')
line_pbp, = ax.plot(pbp_sorted,p_pbp,label='PBP', color='blue')
ax.set_xlabel('Image Intensity (dB)')
ax.set_ylabel('Cumulative Probability')
ax.set_title('CDF of ISAR Image Intensity, Rotation=0 deg')

def animate(i):
    adp_sorted = np.sort(all_images_adp[i].flatten())
    pbp_sorted = np.sort(all_images_pbp[i].flatten())

    # calculate the proportional values of samples
    p_adp = 1. * np.arange(len(adp_sorted)) / (len(adp_sorted) - 1)
    p_pbp = 1. * np.arange(len(pbp_sorted)) / (len(pbp_sorted) - 1)

    line_adp.set_xdata(adp_sorted)  # update the data.
    line_pbp.set_xdata(pbp_sorted)  # update the data.
    ax.set_title(f'CDF of ISAR Image Intensity, Rotation={all_looks[i]} deg')
    return line_adp, line_pbp

ani = animation.FuncAnimation(f, animate, interval=300, blit=True, frames=len(all_images_adp))

ani.save(os.path.join(paths.output,"cdf_hann_4batches.gif"))
plt.show()