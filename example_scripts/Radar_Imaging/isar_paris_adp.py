# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

This uses the vehicles velocities to create the image

@author: asligar
"""

from tqdm import tqdm
import numpy as np
import os
import sys
import scipy

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.materials import MaterialManager
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, vec_to_rot, look_at
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

# define a circle that are radar will orbit around (uses target geometry center as the center of the circle + orbit elevation)
orbit_radius = 2000*2
orbit_elevation = 2100*2

# update the model every n degrees, smaller updates make smoother velocity estimates. The aspect angle defined for an
# image is independent of this value, this is only used to update the image and run a new simulation
simulation_update_every_n_deg = 1.0
# number of images to create (created every simulation_update_every_n_deg), independent of the view aspect used to
# create an image. WIth simulation_update_every_n_deg = 1 and num_of_looks this will create 360 images spanning 360deg
num_of_looks = 360

# simulation parameters
max_num_refl = 3
max_num_trans = 0
ray_density = 0.1  # global ray density setting, this will be overwritten by the target_ray_spacing in the actor if defined

# radar parameters
center_freq = 10.0e9
num_freqs = 1400
bandwidth = 300e6

# how many pulses to create in a CPI, this is the number of pulses used to create a single image
num_pulse_CPI = 1400
# define the aspect angle of the image, this is the angle that the radar will look at the target and collect data for
# 1 image.
phi_aspect_deg = 1.7

# terrain_path = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Scripting\RFSX_2\scenarios\Denver_20mm_res\Denver_20mm_res'
beam_width = 90 #deg
export_debug = True

#######################################################################################################################
# End Input Parameters
#######################################################################################################################

# To create an image, we can create a cpi of any length, but using 1 second will make defining the location of the radar
# easier. The number of pulses in a CPI will define how many pulses are used to create 1 image
cpi_duration = 1  # 1 second for easy math

# output file will be stored in this directory

mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()

if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)


test = euler_to_rot(phi=45)
terrain_name = all_actors.add_actor(name='terrain1',
                                    filename=os.path.join(paths.models,'Paris/Paris.json'),
                                    include_texture=False, map_texture_to_material=False, target_ray_spacing=0.5,
                                    clip_at_bounds=[-400,400,-400,400])
all_actors.actors[terrain_name].update_actor()
mesh = all_actors.actors[terrain_name].get_mesh()
terrain_bounds = mesh[0].bounds

# center xyz location of terrain_bounds
terrain_center = [(terrain_bounds[0] + terrain_bounds[1]) / 2, (terrain_bounds[2] + terrain_bounds[3]) / 2,
                  (terrain_bounds[4] + terrain_bounds[5]) / 2]

# define spotlight sar orbital path

# empty actor
actor_radar_name = all_actors.add_actor()

# just create enough points to makes a smooth circle, we will use a spline to interpolate between these points later
num_points = 10001
# how long it take to make one orbit. This is related to the aspect angle of the image,and our known cpi length of 1 sec
# the time is defined this way becuase we are just going to estimate the velocity based on location. For example, if
# phi_aspect_deg = 8.85, then the radar will take 360/8.85 = 40.678 seconds to make a full orbit. This is just used to
# estimate the velocity of the radar, the actual simulation will be updated every simulation_update_every_n_deg. The CPI
# time is arbitrary, we could have defined it as any length of time, because we are calculating velocity.
end_time = 360 / phi_aspect_deg
time_stamps_temp = np.linspace(0, end_time, num_points)
# points that make up a circle in the xy plane at z=terrain_bounds[5]+orbit_elevation
orbit_center = np.array([terrain_center[0], terrain_center[1], terrain_bounds[5] + orbit_elevation])
all_positions_radar = np.array([orbit_center[0] + orbit_radius * np.cos(np.linspace(0, 2 * np.pi, num_points)),
                                orbit_center[1] + orbit_radius * np.sin(np.linspace(0, 2 * np.pi, num_points)),
                                np.ones(num_points) * orbit_center[2]]).T

# allow us to update simulation at any time step, not tied to the time_stamp_temp used to define this circle
interp_func_pos = scipy.interpolate.interp1d(time_stamps_temp, all_positions_radar, axis=0, assume_sorted=True)

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

########################## Setup Radar Platform  ##########################

# we are not loading an antenna device from file, so this will just create a blank antenna device,
# as a child of radar_device. Instead of loading from a file, this will be manually setup using a combination
# of direct API calls and the AntennaDevice class
ant_device = AntennaDevice(file_name=None,
                           parent_h_node=all_actors.actors[actor_radar_name].h_node,
                           all_actors=all_actors)
ant_device.initialize_device()
ant_device.range_pixels = num_freqs
ant_device.doppler_pixels = num_pulse_CPI
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
    "hpbwHorizDeg": beam_width,
    "hpbwVertDeg": beam_width,
    "position": [0.0, 0.0, 0.0]
}

rx_dict = {
    "type": "parametric",
    "operation_mode": "rx",
    "polarization": "VERTICAL",
    "hpbwHorizDeg": beam_width,
    "hpbwVertDeg": beam_width,
    "position": [0.0, 0.0, 0.0]
}

antennas_dict = {"Tx1": tx_dict, "Rx1": rx_dict}
ant_device.add_antennas(mode_name=mode_name, load_pattern_as_mesh=True, scale_pattern=100, antennas_dict=antennas_dict)
ant_device.set_mode_active(mode_name)
ant_device.add_mode(mode_name)

# assign modes to gpu devices

ray_spacing = 1.0

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

# pem_api_manager.isOK(pem.setPrivateKey("RayShootGrid", "SBR," + str(0.001)))
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
                                   display_mode='normal',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=10)

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
                             output_movie_name=output_movie_name,
                             figure_size=(1, 1),
                             shape=(len(rng_domain), len(vel_domain)),
                             cmap='Greys_r')  #shape is rotated from actual output shape

# percentage of total orbit, converted to time (based on CPI update rate)
simulation_end_time = (num_of_looks * simulation_update_every_n_deg)/360*end_time
update_times = np.linspace(0, simulation_end_time, num_of_looks)
all_max = []
print('running simulation...')
all_lin = []
all_ang = []
all_pos = []
all_rot = []
for idx in tqdm(range(num_of_looks)):
    time = update_times[idx]
    all_actors.actors[actor_radar_name].coord_sys.pos = interp_func_pos(time)
    all_actors.actors[actor_radar_name].coord_sys.rot = look_at(interp_func_pos(time), terrain_center,correct_rotation_matrix=True)
    # because I am not explicitly defining a velocity, I can use the update function to update the position and if
    # a time is provided, it will also update the velocity by estimating it based on teh change in position
    # the first 3 frames will be poor quality because the velocity is estimated based on at least 3 points
    all_actors.actors[actor_radar_name].coord_sys.update(time=time)
    all_pos.append(all_actors.actors[actor_radar_name].coord_sys.pos)
    all_rot.append(all_actors.actors[actor_radar_name].coord_sys.rot)
    all_lin.append(all_actors.actors[actor_radar_name].coord_sys.lin)
    all_ang.append(all_actors.actors[actor_radar_name].coord_sys.ang)
    modeler.update_frame()  # update visualization


    # calculate parameters required to center image in both range and doppler
    vector_to_target = all_actors.actors[actor_radar_name].coord_sys.pos - np.array(terrain_center)
    vector_to_target_n = vector_to_target / np.linalg.norm(vector_to_target)
    dist_to_center = np.sqrt(vector_to_target[0] ** 2 + vector_to_target[1] ** 2 + vector_to_target[2] ** 2)
    # print(f'distance to center: {dist_to_center} m')
    ref_pixel = RssPy.ImagePixelReference.MIDDLE
    vector_velocity = all_actors.actors[actor_radar_name].coord_sys.lin - all_actors.actors[terrain_name].coord_sys.lin # scene velocity is 0 for this example
    # calculate the radial velocity between center of scene and satellite
    radial_velocity = -1 * np.dot(vector_velocity, vector_to_target_n)
    # print(f'radial velocity: {radial_velocity} m/s')
    sideLobeLevelDb = 50.0
    rSpecs = "hann," + str(sideLobeLevelDb)
    dSpecs = "hann," + str(sideLobeLevelDb)

    # adjust the range and doppler pixels to center the image based on distance and radial velocity
    pem_api_manager.isOK(pem.activateRangeDopplerResponse(ant_device.modes[mode_name],
                                                   ant_device.range_pixels,
                                                   ant_device.doppler_pixels,
                                                   ref_pixel,
                                                   dist_to_center,
                                                   ref_pixel,
                                                   radial_velocity,
                                                   rSpecs,
                                                   dSpecs))

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)

    # response is stored as [tx_idx,rx_idx,pulse_idx,freq_idx], we will just use center pulse (idx=1)
    response = response[0, 0]
    isar_image = 20 * np.log10(np.fmax(np.abs(response), 1.e-30))
    data_max = (np.max(isar_image))
    all_max.append(data_max)
    data_max = -190#
    modeler.mpl_ax_handle.set_data(
        isar_image.T)  # update pyvista matplotlib plot, rotate to max image orientation match
    modeler.mpl_ax_handle.set_clim(vmin=data_max - 50, vmax=data_max)  # set color limits
    all_scalar_bars = list(modeler.pl.scalar_bars.keys())
    for each in all_scalar_bars:
        modeler.pl.remove_scalar_bar(each)

    if export_debug:
        debug_camera.generate_image()
print(f'MaxMax of all frames: {np.max(all_max)}')
print(f'MinMax of all frames: {np.min(all_max)}')
print(f'Avg Max of all frames: {np.median(all_max)}')

modeler.close()
gdxg = 1
print('starting post processing ')
#
# response = np.array(response)
# isar_image = 20*np.log10(np.abs(response))
# # ToDo, not completed yet
# # isar_image = isar_2d(responses, freq_domain, phi_points,function='db',window='hann') #wip
# print(np.min(isar_image))
# print(np.max(isar_image))
# plt.close('all')
#
# fig, ax = plt.subplots()
# ax.imshow(isar_image, cmap='jet',clim=[-272,-128])
# plt.show()

if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')
