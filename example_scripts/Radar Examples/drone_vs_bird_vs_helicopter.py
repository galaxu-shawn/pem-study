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

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.utils import generate_rgb_from_array
from pem_utilities.far_fields import convert_ffd_to_gltf
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

radar_pos = [45, 0, 5]

# waveform parameters
center_freq = 77e9
num_freqs = 512
bandwidth = 300e6
cpi_duration = 10e-3
num_pulse_CPI = 101 # this will result in a pulse interval of 1ms

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .4

# export debug logs and show modeler for visualization
export_debug = True
show_modeler = True

# output file will be stored in this directory
os.makedirs(paths.output, exist_ok=True)

debug_logs = DebuggingLogs(output_directory=paths.output)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

# load the road from a file, and assign a material property
# actor_terrain_name = all_actors.add_actor(filename='../models/whole-scene-static.stl',
#                                              mat_idx=mat_manager.get_index('asphalt'),
#                                              color='grey')
# actor_terrain_name2 = all_actors.add_actor(filename='../models/whole-scene-static.stl',
#                                              mat_idx=mat_manager.get_index('asphalt'),
#                                              color='grey')

actor_drone_name = all_actors.add_actor(name='drone',
                                             filename=os.path.join(paths.models,'drone1/drone1.json'),
                                             target_ray_spacing=0.005,scale_mesh=1/5)
all_actors.actors[actor_drone_name].rotor_ang = [1000,0,0] # speed of rotor, otherwise use what is in the json

actor_heli_name = all_actors.add_actor(name='heli',
                                             filename=os.path.join(paths.models,'AH64/AH64.json'),
                                             target_ray_spacing=0.5)
# if you don't define a rotor_ang (main rotor) or rear_rotr_ang (tail rotor), the default value is defined in the json
all_actors.actors[actor_heli_name].rotor_ang = [0,0,1000]
all_actors.actors[actor_heli_name].rear_rotor_ang = [0,700,0]


actor_bird_name = all_actors.add_actor(name='bird',
                                             filename=os.path.join(paths.models,'Bird3/Bird3.json'),
                                             target_ray_spacing=0.005,scale_mesh=1)
# all_actors.actors[actor_bird_name].update_rot_based_on_ang_vel = False
all_actors.actors[actor_bird_name].flap_freq = 3
all_actors.actors[actor_bird_name].flap_range = 45

# here are two different ways of updating the position of an actor in the scene.
# One is kind of a manual update and the other is using the .velocity_mag along with the orientation of the object
#
# manual update:
# all_actors.actors[actor_heli_name].coord_sys.pos = [0,0,0]
# all_actors.actors[actor_heli_name].coord_sys.update()
#
# This would be used along with all_actors.actors[actor_drone_name].use_linear_velocity_equation_update = False
# this does a manual update, and can be done for linear/angular and rotation values as well. In this example,
# we are manually setting the position based on the equation of a circle. But when we do this, we are not also setting
# the .lin property for the objects linear velocity. So the solver see the object as having 0 velocity.
# for the automatic update, we would use
# all_actors.actors[actor_drone_name].use_linear_velocity_equation_update = True
# all_actors.actors[actor_drone_name].velocity_mag = 10 #m/s
# these setting will update the position based on the velocity_mag and the orientation of the object. This is useful
# when we don't want to create an exact path, instead just provide intial position and velocity, and let the solver
# update the location based on the velocity and orientation of the object.

all_actors.actors[actor_drone_name].use_linear_velocity_equation_update = False
all_actors.actors[actor_heli_name].use_linear_velocity_equation_update = False
all_actors.actors[actor_bird_name].use_linear_velocity_equation_update = False

# all_actors.actors[actor_terrain_name2].coord_sys.pos = (50., 0.0, 0.)
# all_actors.actors[actor_terrain_name2].coord_sys.update()


# let's make this quadcopter orbit in a circle
# the pedestrian will do a full circle in 10 seconds.
# create all the points that make up the circle, and the rotation of the pedestrian so that it is facing tangent to the circle
# at each point. The total number of steps in this points won't matter that much because we will turn this into an
# interpolated function that we can later update the position of the pedestrian with.

# For this Actor, we will demonstrate how to update the velocity using an estimate value based only on the position
# and rotation of the actor. This is useful for objects that we don't know the velocity, but we know the position.
time_steps = np.linspace(0, 10, 301)
circle_radius = 6
orbit_center = np.array([60, 0, 5])
all_positions_quadcopter = np.array([orbit_center[0] + circle_radius * np.cos(np.linspace(0, 2 * np.pi, 301)),
                                     orbit_center[1] + circle_radius * np.sin(np.linspace(0, 2 * np.pi, 301)),
                                     np.ones(301)*orbit_center[2]]).T
interp_func_pos = scipy.interpolate.interp1d(time_steps, all_positions_quadcopter, axis=0, assume_sorted=True)
all_rots_quadcopter = euler_to_rot(phi=np.linspace(90, 450, 301),
                                   theta=np.linspace(0, 0, 301),
                                   psi=np.linspace(0, 0, 301),
                                   order='zyz', deg=True)
interp_func_rot = scipy.interpolate.interp1d(time_steps, all_rots_quadcopter, axis=0, assume_sorted=True)

# heli pos
time_steps = np.linspace(0, 10, 301)
circle_radius = 20
orbit_center = np.array([40, 0, 10])
all_positions_heli = np.array([orbit_center[0] + circle_radius * np.cos(np.linspace(0, 2 * np.pi, 301)),
                                     orbit_center[1] + circle_radius * np.sin(np.linspace(0, 2 * np.pi, 301)),
                                     np.ones(301)*orbit_center[2]]).T
interp_func_heli_pos = scipy.interpolate.interp1d(time_steps, all_positions_heli, axis=0, assume_sorted=True)
all_rots_heli = euler_to_rot(phi=np.linspace(90, 450, 301),
                                   theta=np.linspace(0, 0, 301),
                                   psi=np.linspace(0, 0, 301),
                                   order='zyz', deg=True)
interp_func_heli_rot = scipy.interpolate.interp1d(time_steps, all_rots_heli, axis=0, assume_sorted=True)


# bird pos
circle_radius = 5.25
orbit_center = np.array([59, 1, 5])
all_positions_bird = np.array([orbit_center[0] + circle_radius * np.cos(np.linspace(0, 2 * np.pi, 301)),
                                     orbit_center[1] + circle_radius * np.sin(np.linspace(0, 2 * np.pi, 301)),
                                     np.ones(301)*orbit_center[2]]).T
interp_func_pos_bird = scipy.interpolate.interp1d(time_steps, all_positions_bird, axis=0, assume_sorted=True)
all_rots_bird = euler_to_rot(phi=np.linspace(90, 450, 301),
                                   theta=np.linspace(0, 0, 301),
                                   psi=np.linspace(0, 0, 301),
                                   order='zyz', deg=True)
interp_func_rot_bird = scipy.interpolate.interp1d(time_steps, all_rots_bird, axis=0, assume_sorted=True)




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
wavelength = 3e8 / center_freq
mode_name = 'mode1'
# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)
ant_device = add_single_tx_rx(all_actors,waveform,mode_name,pos=radar_pos,ffd_file='dipole.ffd',scale_pattern=2)


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
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(pem.getLastWarnings())

# display setup
# print(api.reportSettings())

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device.modes[mode_name]  # tell it which mode we want to get respones from
ant_device.waveforms[mode_name].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device.waveforms[mode_name].vel_domain
rng_domain = ant_device.waveforms[mode_name].rng_domain
freq_domain = ant_device.waveforms[mode_name].freq_domain
pulse_domain = ant_device.waveforms[mode_name].pulse_domain

print(f'Velocity Window: {vel_domain[-1]-vel_domain[0]}')
print(f'Max Range: {rng_domain[-1]}')

fps = 30
dt = 1 / fps
T = 10
numFrames = int(T / dt)
responses = []
cameraImages = []

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
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
                             rng_domain=rng_domain,
                             vel_domain=vel_domain,
                             fps=fps,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)

# modeler = ModelVisualization(all_actors,
#                              show_antennas=True,
#                              rng_domain=rng_domain,
#                              vel_domain=vel_domain,
#                              fps=fps,
#                              camera_orientation=None,
#                              camera_attachment=None,
#                              output_movie_name=output_movie_name)



print('running simulation...')
for iFrame in tqdm(range(numFrames), disable=True):
    time = iFrame * dt
    # update all coordinate systems
    for actor in all_actors.actors:
        # Actors can have actor_type value, this will handle sub-parts of the actors. For example, if we have a vehicle
        # it will update the tires rotation and velocity, and the vehicle linear velocity based on its orientation
        # if actor is other, it will use the lin/rot/ang values to update the position of the actor
        if actor == actor_drone_name:
            # this is the special case we define above that is not updating the position based on the velocity vector,
            # but instead we defined a series of points (converted to interpolated function), that we will use to update
            # the position, which will then in turn estimate the velocity.
            if time > 10:  # our limited time function can just be looped since the quadcopter is orbiging in a circle
                # earlier when we defined the quadcopter orbit path, the position was only defined for 10 seconds, so we
                # will loop the time, this is not required, but just to show how we can do this
                update_time = np.mod(time, 10)
            else:
                update_time = time
            # because velocity estimates are based on finite difference, the accuracy will depend on the size of the
            # time step for non-linear motion. For linear motion, the velocity estimate will be accurate.
            all_actors.actors[actor].coord_sys.pos = interp_func_pos(update_time)
            all_actors.actors[actor].coord_sys.rot = interp_func_rot(update_time)
        elif actor == actor_heli_name:
            # this is the special case we define above that is not updating the position based on the velocity vector,
            # but instead we defined a series of points (converted to interpolated function), that we will use to update
            # the position, which will then in turn estimate the velocity.
            if time > 10:  # our limited time function can just be looped since the quadcopter is orbiging in a circle
                # earlier when we defined the quadcopter orbit path, the position was only defined for 10 seconds, so we
                # will loop the time, this is not required, but just to show how we can do this
                update_time = np.mod(time, 10)
            else:
                update_time = time
            # because velocity estimates are based on finite difference, the accuracy will depend on the size of the
            # time step for non-linear motion. For linear motion, the velocity estimate will be accurate.
            all_actors.actors[actor].coord_sys.pos = interp_func_heli_pos(update_time)
            all_actors.actors[actor].coord_sys.rot = interp_func_heli_rot(update_time)
        elif actor == actor_bird_name:
            # this is the special case we define above that is not updating the position based on the velocity vector,
            # but instead we defined a series of points (converted to interpolated function), that we will use to update
            # the position, which will then in turn estimate the velocity.
            if time > 10:  # our limited time function can just be looped since the quadcopter is orbiging in a circle
                # earlier when we defined the quadcopter orbit path, the position was only defined for 10 seconds, so we
                # will loop the time, this is not required, but just to show how we can do this
                update_time = np.mod(update_time, 10)
            else:
                update_time = time
            # because velocity estimates are based on finite difference, the accuracy will depend on the size of the
            # time step for non-linear motion. For linear motion, the velocity estimate will be accurate.
            all_actors.actors[actor].coord_sys.pos = interp_func_pos_bird(update_time)
            all_actors.actors[actor].coord_sys.rot = interp_func_rot_bird(update_time)
        all_actors.actors[actor].update_actor(time=time)

    # This is not required, but will write out scene summary of the simulation setup,
    # only writing out first and last frame to limit number of files, useful for debugging.
    if iFrame == 0 or iFrame == numFrames - 1:
        modeler.pl.show_grid()
        debug_logs.write_scene_summary(file_name=f'out_{iFrame}.json')

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes['mode1'], RssPy.ResponseType.RANGE_DOPPLER)
    # calculate response in dB to overlay in pyvista plotter
    imData = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))

    # exporting radar camera images
    if export_debug:
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)

    modeler.update_frame(plot_data=imData,plot_limits=[imData.min(),imData.max()])
    if iFrame == 0:
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