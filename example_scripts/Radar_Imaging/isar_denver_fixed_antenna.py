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
import pyvista as pv

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, vec_to_rot, look_at
from pem_utilities.post_processing import isar_2d, range_crossrange
from pem_utilities.antenna_device import add_single_tx_rx, Waveform, AntennaDevice
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
orbit_radius = 4000
orbit_elevation = 4200

# if you want to pick the center of the scene manually, if this is False,
# the center of the scene will be the value entered as center_of_scene below.
center_of_scene = [0, 0, 0] # center of the scene, where the radar will point at and clip the geometry
center_image = True

# update the model every n degrees, smaller updates make smoother velocity estimates. The aspect angle defined for an
# image is independent of this value, this is only used to update the image and run a new simulation
simulation_update_every_n_deg = 1.0
# number of images to create (created every simulation_update_every_n_deg), independent of the view aspect used to
# create an image. WIth simulation_update_every_n_deg = 1 and num_of_looks this will create 360 images spanning 360deg
num_of_looks = 360

# simulation options
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 1
max_num_trans = 0
ray_density = .01

# radar parameters
center_freq = 10.0e9
num_freqs = 1200
bandwidth = 300e6

# how many pulses to create in a CPI, this is the number of pulses used to create a single image
num_pulse_CPI = 1200
# define the aspect angle of the image, this is the angle that the radar will look at the target and collect data for
# 1 image.
phi_aspect_deg = 1.7

beam_width = 120#deg
export_debug = True

#######################################################################################################################
# End Input Parameters
#######################################################################################################################

# To create an image, we can create a cpi of any length, but using 1 second will make defining the location of the radar
# easier. The number of pulses in a CPI will define how many pulses are used to create 1 image
cpi_duration = 1  # 1 second for easy math


mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()

if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)



terrain_name = all_actors.add_actor(name='terrain1',
                                    filename= r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Scripting\RFSX_2\scenarios\Denver_20mm_res\Denver_20mm_res',
                                    include_texture=False, map_texture_to_material=False, target_ray_spacing=.05,center_geometry=True)

all_actors.actors[terrain_name].update_actor()

mesh = all_actors.actors[terrain_name].get_mesh()
terrain_bounds = mesh[0].bounds
# all_actors.actors[terrain_name].coord_sys.pos = -1*np.array(center_of_scene) # move the terrain to the center of the scene
all_actors.actors[terrain_name].coord_sys.update()
# center xyz location of terrain_bounds, used to point antenna at the center of the scene
terrain_center = [(terrain_bounds[0] + terrain_bounds[1]) / 2, (terrain_bounds[2] + terrain_bounds[3]) / 2,
                  (terrain_bounds[4] + terrain_bounds[5]) / 2]
terrain_center = [0,0,0]

# define spotlight sar orbital path, this could be simplified, in this example, we are only rotating the city, so the
# radar can just stay fixed at a single point.

# empty actor
actor_radar_name = all_actors.add_actor()

# just create enough points to makes a smooth circle, we will use a spline to interpolate between these points later
num_points = 10001
# how long it take to make one orbit. This is related to the aspect angle of the image,and our known cpi length of 1 sec
# the time is defined this way because we are just going to estimate the velocity based on location. For example, if
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
phi_angles = np.linspace(0, 360, num_points)
# allow us to update simulation at any time step, not tied to the time_stamp_temp used to define this circle
interp_func_pos = scipy.interpolate.interp1d(time_stamps_temp, all_positions_radar, axis=0, assume_sorted=True)
interp_func_phis = scipy.interpolate.interp1d(time_stamps_temp, phi_angles, axis=0, assume_sorted=True)

# waveform used for radar
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

# helper function to add a simple 1 tx 1 rx radar to the scene
ant_device = add_single_tx_rx(all_actors,
                              waveform,
                              mode_name,
                              parent_h_node=all_actors.actors[actor_radar_name].h_node,
                              beamwidth_H=beam_width,
                              beamwidth_V=beam_width,
                              range_pixels=num_freqs,
                              doppler_pixels=num_pulse_CPI,
                              scale_pattern=2)

if ray_density is not None:
    freq_center = ant_device.waveforms[mode_name].center_freq
    lambda_center = 2.99792458e8 / freq_center
    ray_spacing = np.sqrt(2) * lambda_center / ray_density

sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 180
sim_options.bounding_box = -1
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

print(f'Max Range: {rng_domain[-1]} m')

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
                             show_antennas=False,
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name,
                             figure_size=(1.0, 1.0),
                             output_video_size=(3840,2160),
                             shape=(len(rng_domain), len(vel_domain)),
                             cmap='Greys_r')  #shape is rotated from actual output shape
modeler.pl.add_axes_at_origin(labels_off=True)
# initialize positions and rotations of the scene
# percentage of total orbit, converted to time (based on CPI update rate)
simulation_end_time = (num_of_looks * simulation_update_every_n_deg)/360*end_time
update_times = np.linspace(0, simulation_end_time, num_of_looks)
all_max = []

all_actors.actors[actor_radar_name].coord_sys.pos = interp_func_pos(0)
all_actors.actors[actor_radar_name].coord_sys.rot = look_at(interp_func_pos(0), terrain_center,
                                                            correct_rotation_matrix=False)
all_actors.actors[actor_radar_name].coord_sys.update()
all_actors.actors[terrain_name].coord_sys.ang = (0,0,np.deg2rad(phi_aspect_deg))
# all_actors.actors[terrain_name].update_rot_based_on_ang_vel = False
all_actors.actors[terrain_name].coord_sys.update()

print('running simulation...')
for idx in tqdm(range(num_of_looks)):
    time = update_times[idx]
    phi = np.atleast_1d(interp_func_phis(time))
    all_actors.actors[terrain_name].coord_sys.rot = euler_to_rot(phi=phi)
    all_actors.actors[terrain_name].coord_sys.update()

    if center_image:
        # center the image based on range. Could also center based on velocity if the radar or target had a linear velocity
        # but in this case we are only rotating the city, so the center will be the center of the scene in doppler
        vector_to_target = np.array(all_actors.actors[actor_radar_name].coord_sys.pos) - np.array(terrain_center)
        vector_to_target_n = vector_to_target / np.linalg.norm(vector_to_target)
        dist_to_center = np.sqrt(vector_to_target[0] ** 2 + vector_to_target[1] ** 2 + vector_to_target[2] ** 2)
        # print(f'distance to center: {dist_to_center} m')
        ref_pixel = RssPy.ImagePixelReference.MIDDLE

        # center in doppler based on velocity
        # vector_velocity = np.array(all_actors.actors[actor_radar_name].coord_sys.lin) - all_actors.actors[terrain_name].coord_sys.lin)
        # calculate the radial velocity between center of scene and satellite
        # radial_velocity = -1 * np.dot(vector_velocity, vector_to_target_n)
        radial_velocity = 0
        # print(f'radial velocity: {radial_velocity} m/s')
        pem_api_manager.isOK(pem.activateRangeDopplerResponse(ant_device.modes[mode_name],
                                                       ant_device.range_pixels,
                                                       ant_device.doppler_pixels,
                                                       ref_pixel,
                                                       dist_to_center,
                                                       ref_pixel,
                                                       radial_velocity,
                                                       ant_device.r_specs, ant_device.d_specs))

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)

    # response is stored as [tx_idx,rx_idx,pulse_idx,freq_idx], we will just use center pulse (idx=1)
    response = response[0, 0]
    isar_image = 20 * np.log10(np.fmax(np.abs(response), 1.e-30))
    data_max = np.max(isar_image)
    # print(f'Max of frame {idx}: {data_max}')
    all_max.append(data_max)
    data_max = -200 # manually adjusting plot max
    modeler.update_frame(plot_data=isar_image.T,plot_limits=[data_max-70,data_max])  # update visualization

    # all_scalar_bars = list(modeler.pl.scalar_bars.keys())
    # for each in all_scalar_bars:
    #     modeler.pl.remove_scalar_bar(each)
    if export_debug:
        debug_camera.generate_image()
        if idx == 0:
            modeler.pl.show_grid()
        debug_logs.write_scene_summary(file_name=f'out.json')
print(f'MaxMax of all frames: {np.max(all_max)}')
print(f'MinMax of all frames: {np.min(all_max)}')
print(f'Avg Max of all frames: {np.median(all_max)}')

modeler.close()

if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')
