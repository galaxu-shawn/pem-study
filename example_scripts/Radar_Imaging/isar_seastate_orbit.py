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
from pathlib import Path

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, vec_to_rot, look_at
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.seastate import OceanSurface
from pem_utilities.simulation_options import SimulationOptions

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy

#######################################################################################################################
# Input Parameters
#######################################################################################################################

remove_sea = False

# define a circle that are radar will orbit around (uses target geometry center as the center of the circle + orbit elevation)
orbit_radius = 1000*2
orbit_elevation = 1100*2

# update the model every n degrees, smaller updates make smoother velocity estimates. The aspect angle defined for an
# image is independent of this value, this is only used to update the image and run a new simulation
simulation_update_every_n_deg = 1.7
# number of images to create (created every simulation_update_every_n_deg), independent of the view aspect used to
# create an image. WIth simulation_update_every_n_deg = 1 and num_of_looks this will create 360 images spanning 360deg
num_of_looks = 90

# simulation parameters
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_spacing = 0.25  # global ray spacing in meters


# radar parameters
center_freq = 10.0e9
num_freqs = 1001
bandwidth = 300e6

# how many pulses to create in a CPI, this is the number of pulses used to create a single image
pulses_per_image = 1001
# one_pulse_per_cpi = False

# define the aspect angle of the image, this is the angle that the radar will look at the target and collect data for
# 1 image.
phi_aspect_deg = 1.7
pulse_every_n_deg = phi_aspect_deg/pulses_per_image
how_long_to_travel_one_image = 10  # seconds
rotation_speed =  2 * np.pi / how_long_to_travel_one_image * phi_aspect_deg/360  # rad/s
end_simulation_time =  how_long_to_travel_one_image * num_of_looks
time_step_per_chirp = how_long_to_travel_one_image/pulses_per_image
time_stamps_per_image = np.linspace(0, how_long_to_travel_one_image, pulses_per_image)

time_transit_360  = 2*np.pi/rotation_speed
beam_width = 120 #deg
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

# Set up the ocean parameters
num_grid = 500  # grid size
scene_length = 500  # domain size
wind_speed = 20
wave_amplitude = 3  # wave amplitude
choppiness = .05  # choppiness factor
ship_velocity = 10  # ship velocity
ship_length = 110

# ship position, z is the pos of the ship, ocean is at z=0,
# so ship waterline is 3m above ocean based on the CAD model
# ship_position = [0, -125, 3]
ship_position = [0, 0, 3]

terrain_bounds = [-100,100,-100,100,0,100]
terrain_center = [0,0,0]
if not remove_sea:
    ocean = OceanSurface(num_grid, scene_length, wind_speed, wave_amplitude, choppiness,
                            include_wake=True,
                            velocity_ship=ship_velocity,
                            length_ship=ship_length,
                            beam_ship=20.3,
                            draft_ship=6.5,
                            initial_wake_position=ship_position[:2],
                            update_wake_position=False)


    ocean_name = all_actors.add_actor(name='ocean',generator=ocean,mat_idx=mat_manager.get_index('seawater'))

    # all_actors.actors[ocean_name].update_actor()
    mesh = all_actors.actors[ocean_name].get_mesh()
    terrain_bounds = mesh[0].bounds

    # center xyz location of terrain_bounds
    terrain_center = [(terrain_bounds[0] + terrain_bounds[1]) / 2, (terrain_bounds[2] + terrain_bounds[3]) / 2,
                      (terrain_bounds[4] + terrain_bounds[5]) / 2]

ship_model_path = os.path.join(paths.models,'Ships')
geo_filename = 'explorer_ship_meter.stl'
ship_name = all_actors.add_actor(name='ship',filename=os.path.join(ship_model_path,geo_filename),color=(0.5,0.5,0.5),mat_idx=0)

all_actors.actors[ship_name].coord_sys.pos = [ship_position[0],ship_position[1],ship_position[2]]
initial_ship_rot = 90
all_actors.actors[ship_name].coord_sys.rot = euler_to_rot(phi=initial_ship_rot) # wake is only supported for +Y movement
lin_x = ship_velocity * np.cos(np.deg2rad(initial_ship_rot))
lin_y = ship_velocity * np.sin(np.deg2rad(initial_ship_rot))
all_actors.actors[ship_name].coord_sys.lin = [lin_x, lin_y, 0]
all_actors.actors[ship_name].coord_sys.update()
all_actors.actors[ship_name].use_linear_velocity_equation_update = False
# define spotlight sar orbital path

# empty actor
actor_radar_name = all_actors.add_actor()

# just create enough points to makes a smooth circle, we will use a spline to interpolate between these points later
num_points = int(pulses_per_image*360/simulation_update_every_n_deg)
# how long it take to make one orbit. This is related to the aspect angle of the image,and our known cpi length of 1 sec
# the time is defined this way becuase we are just going to estimate the velocity based on location. For example, if
# phi_aspect_deg = 8.85, then the radar will take 360/8.85 = 40.678 seconds to make a full orbit. This is just used to
# estimate the velocity of the radar, the actual simulation will be updated every simulation_update_every_n_deg. The CPI
# time is arbitrary, we could have defined it as any length of time, because we are calculating velocity.

time_stamps_temp = np.linspace(0, time_transit_360, num_points)
# points that make up a circle in the xy plane at z=terrain_bounds[5]+orbit_elevation
orbit_center = np.array([terrain_center[0], terrain_center[1], terrain_bounds[5] + orbit_elevation])
all_positions_radar = np.array([orbit_center[0] + orbit_radius * np.cos(np.linspace(0, 2 * np.pi, num_points)),
                                orbit_center[1] + orbit_radius * np.sin(np.linspace(0, 2 * np.pi, num_points)),
                                np.ones(num_points) * orbit_center[2]]).T
phi_angles = np.linspace(0, 360, num_points)
# allow us to update simulation at any time step, not tied to the time_stamp_temp used to define this circle
interp_func_pos = scipy.interpolate.interp1d(time_stamps_temp, all_positions_radar, axis=0, assume_sorted=True)
interp_func_phis = scipy.interpolate.interp1d(time_stamps_temp, phi_angles, axis=0, assume_sorted=True)

# simulation parameters


cpi_duration = time_step_per_chirp * pulses_per_image
num_pulse_CPI = pulses_per_image


waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
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

ant_device = add_single_tx_rx(all_actors,
                              waveform,
                              mode_name,
                              parent_h_node=all_actors.actors[actor_radar_name].h_node,
                              beamwidth_H=beam_width,
                              beamwidth_V=beam_width,
                              range_pixels=num_freqs,
                              doppler_pixels=num_pulse_CPI,
                              scale_pattern=200)

# assign modes to gpu devices
print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
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
                                   display_mode='coatings',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=10)


_, _,pos_tx,_,_ = pem.coordSysInGlobal(ant_device.coord_sys.h_node)
scene_center = np.array([0,0,0])
vector_to_target = np.array(pos_tx) - scene_center
dist_to_center = np.sqrt(vector_to_target[0]**2 + vector_to_target[1]**2 + vector_to_target[2]**2)
print(f'Distance to center: {dist_to_center} m')

# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
# camera_orientation = 'follow7' and camera_attachment = ship_name will follow the ship
output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=10,
                             camera_orientation='follow7',
                             camera_attachment=ship_name,
                             output_movie_name=output_movie_name,
                             figure_size=(.4, .4),
                             shape=(len(rng_domain), len(rng_domain)),
                             cmap='Greys_r')  #shape is rotated from actual output shape

modeler.pl.show_grid()
# modeler.pl.background_color = "black"
# percentage of total orbit, converted to time (based on CPI update rate)
# simulation_end_time = (num_of_looks * simulation_update_every_n_deg)/360*end_time
# update_times = np.linspace(0, simulation_end_time, num_of_looks)
all_max = []
print('running simulation...')


all_actors.actors[actor_radar_name].coord_sys.pos = interp_func_pos(0)
all_actors.actors[actor_radar_name].coord_sys.rot = look_at(interp_func_pos(0), terrain_center,correct_rotation_matrix=False)
all_actors.actors[actor_radar_name].coord_sys.update()



for idx in range(num_of_looks):
    print(f'Running simulation for look {idx+1} of {num_of_looks}')
    data_cube = []
    
    
    time = time_stamps_per_image[pulses_per_image//2] + idx * how_long_to_travel_one_image
    cur_rot = np.atleast_1d(interp_func_phis(time))
    lin_x = ship_velocity * np.cos(np.deg2rad(initial_ship_rot + cur_rot))[0]
    lin_y = ship_velocity * np.sin(np.deg2rad(initial_ship_rot + cur_rot))[0]
    if not remove_sea:
        # all_actors.actors[ocean_name].coord_sys.lin = [lin_x,lin_y, 0]
        all_actors.actors[ocean_name].coord_sys.rot = euler_to_rot(phi=cur_rot)
        all_actors.actors[ocean_name].coord_sys.ang = (0, 0, rotation_speed)
        all_actors.actors[ocean_name].update_actor(time=time)

    all_actors.actors[ship_name].coord_sys.ang = (0, 0, rotation_speed)
    lin_x = 1 * np.cos(np.deg2rad(initial_ship_rot + cur_rot))[0]
    lin_y = 1 * np.sin(np.deg2rad(initial_ship_rot + cur_rot))[0]
    all_actors.actors[ship_name].coord_sys.lin = [lin_x,lin_y, 0]
    all_actors.actors[ship_name].coord_sys.rot = euler_to_rot(phi=initial_ship_rot+cur_rot)
    # all_actors.actors[ship_name].coord_sys.update()
    all_actors.actors[ship_name].update_actor(time=None)

    # center image
    _, _, pos_tx, _, _ = pem.coordSysInGlobal(ant_device.coord_sys.h_node)
    scene_center = np.array([0, 0, 0])
    vector_to_target = np.array(pos_tx) - scene_center
    dist_to_center = np.sqrt(vector_to_target[0] ** 2 + vector_to_target[1] ** 2 + vector_to_target[2] ** 2)
    # print(f'Distance to center: {dist_to_center} m')
    range_ref_pixel = RssPy.ImagePixelReference.MIDDLE
    pem_api_manager.isOK(pem.activateRangeDopplerResponse(ant_device.modes[mode_name],
                                            ant_device.range_pixels,
                                            ant_device.doppler_pixels,
                                            range_ref_pixel,
                                            dist_to_center,
                                            range_ref_pixel,
                                            0.,
                                            ant_device.r_specs,
                                            ant_device.d_specs))

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)
    range_doppler = response[0, 0, :]
    # remove scalar bars, they start to get in the way with lots of things going on
    all_scalar_bars = list(modeler.pl.scalar_bars.keys())
    for each in all_scalar_bars:
        modeler.pl.remove_scalar_bar(each)
    # response is stored as [tx_idx,rx_idx,pulse_idx,freq_idx], we will just use center pulse (idx=1)

    sar_image = 20 * np.log10(np.fmax(np.abs(range_doppler), 1.e-30))
    data_max = (np.max(sar_image))
    print(f'Max of frame {idx + 1}: {data_max}')
    modeler.mpl_ax_handle.set_data(sar_image.T)
    # data_max = -220
    modeler.mpl_ax_handle.set_clim(vmin=data_max-120, vmax=data_max)

    if export_debug:
        debug_camera.generate_image()
        debug_logs.write_scene_summary(file_name=f'out_seastate.json')
        # we can put the debug_camera image into the modeler for debugging
        # modeler.mpl_ax_handle.set_data(debug_camera.current_image)

    modeler.update_frame()



# print(f'MaxMax of all frames: {np.max(all_max)}')
# print(f'MinMax of all frames: {np.min(all_max)}')
# print(f'Avg Max of all frames: {np.median(all_max)}')

modeler.close()

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
