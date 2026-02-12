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
import copy

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.utils import generate_rgb_from_array
from pem_utilities.far_fields import convert_ffd_to_gltf
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.open_street_maps_geometry import BuildingsPrep, TerrainPrep, find_random_location, get_z_elevation_from_mesh
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

fps = 20
end_time = 20
time_stamps = np.linspace(0, end_time, num=int(end_time*fps)) # 10 seconds, 101 frames

# range of values that lat/lon can be generated around
scene_lat_lon = (37.800939, -122.417406)
max_radius = 1000 # meters

# Tx Antenna positions
radar_pos = [0, 0, 55]  # this will be offset from terrain  in Z
radar_rot = euler_to_rot(180, 0, 0, order='zyx', deg=True)

drone_pos = [-90, 0, 50] # this will be offset from terrain  in Z
# bird_pos = [-150, 0, 30]  # this will be offset from terrain  in Z

# waveform parameters
center_freq = 77e9
num_freqs = 512
bandwidth = 300e6
cpi_duration = 10e-3
num_pulse_CPI = 501 # this will result in a pulse interval of 1ms

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 3
max_num_trans = 1
ray_density = .04

# export debug logs and show modeler for visualization
export_debug = True
show_modeler = True

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()


print(f'Generating Scene: Lat/Lon: {scene_lat_lon}...')

terrain_prep = TerrainPrep(paths.output)
terrain = terrain_prep.get_terrain(center_lat_lon=scene_lat_lon, max_radius=max_radius, grid_size=5,)

buildings_prep = BuildingsPrep(paths.output)
# terrain is not yet created, I will create it later, using the exact same points as used for the heatmap surface
building_image_path = os.path.join(paths.output, 'buildings.png')
buildings = buildings_prep.generate_buildings(scene_lat_lon, terrain_mesh=terrain['mesh'], max_radius=max_radius)


# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

# load the road from a file, and assign a material property
buildings_name = all_actors.add_actor(filename=buildings['file_name'],
                                 mat_idx=mat_manager.get_index('concrete'),
                                 color='grey',transparency=0.0)

terrain_name = all_actors.add_actor(filename=terrain['file_name'],
                                 mat_idx=mat_manager.get_index('asphalt'),
                                 color='black',transparency=0.5)

z_intersect = get_z_elevation_from_mesh(radar_pos[:2], all_actors.actors[terrain_name].get_mesh())
radar_pos[2] += np.max(z_intersect)
z_intersect = get_z_elevation_from_mesh(drone_pos[:2], all_actors.actors[terrain_name].get_mesh())
drone_pos[2] += np.max(z_intersect)
# z_intersect = get_z_elevation_from_mesh(bird_pos[:2], all_actors.actors[terrain_name].get_mesh())
# bird_pos[2] += np.max(z_intersect)

actor_drone_name = all_actors.add_actor(name='drone',
                                             filename=os.path.join(paths.models,'Quadcopter2/Quadcopter2.json'),
                                             target_ray_spacing=0.005,scale_mesh=1)
rotor_speed = 77# in Hz
velocity = 3  # in m/s
actor_phi = 0
actor_theta = 0
actor_psi = 0
all_actors.actors[actor_drone_name].rotor_ang = [0, 0, rotor_speed * 2 * np.pi]  # in rad/s
all_actors.actors[actor_drone_name].velocity_mag = velocity
all_actors.actors[actor_drone_name].coord_sys.pos = drone_pos
all_actors.actors[actor_drone_name].coord_sys.rot = euler_to_rot(actor_phi, actor_theta, actor_psi, order='zyx', deg=True)
all_actors.actors[actor_drone_name].coord_sys.update()

for each in range(8):
    actor_bird_name = all_actors.add_actor(name='bird',
                                                 filename=os.path.join(paths.models,'Bird3/Bird3.json'),
                                                 target_ray_spacing=0.005,scale_mesh=np.random.uniform(0.9,1.1))
    # all_actors.actors[actor_bird_name].update_rot_based_on_ang_vel = False
    velocity = np.random.uniform(2.9,3.1)  # in m/s
    actor_phi = -45
    actor_theta = 0
    actor_psi = 0
    bird_pos = copy.copy(drone_pos)
    bird_pos[0] += np.random.uniform(4,5)
    bird_pos[1] -= np.random.uniform(-8,-7)
    bird_pos[2] += np.random.uniform(0.5,1.1)
    all_actors.actors[actor_bird_name].velocity_mag = velocity
    all_actors.actors[actor_bird_name].coord_sys.pos = bird_pos
    all_actors.actors[actor_bird_name].coord_sys.rot = euler_to_rot(actor_phi, actor_theta, actor_psi, order='zyx', deg=True)
    all_actors.actors[actor_bird_name].coord_sys.update()
    all_actors.actors[actor_bird_name].flap_freq = np.random.uniform(2,4)
    all_actors.actors[actor_bird_name].flap_range = np.random.uniform(40,46)


heli_pos = copy.copy(drone_pos)
heli_pos[0] += 100
heli_pos[1] += 10
heli_pos[2] += 4
actor_heli_name = all_actors.add_actor(name='heli',
                                             filename=os.path.join(paths.models,'AH64/AH64.json'),
                                             target_ray_spacing=0.01)
# if you don't define a rotor_ang (main rotor) or rear_rotor_ang (tail rotor), the default value is defined in the json
all_actors.actors[actor_heli_name].coord_sys.pos = heli_pos
all_actors.actors[actor_heli_name].coord_sys.rot = euler_to_rot(45, 5, 0, order='zyx', deg=True)
all_actors.actors[actor_heli_name].rotor_ang = [0,0,100]
all_actors.actors[actor_heli_name].rear_rotor_ang = [0,70,0]
all_actors.actors[actor_heli_name].coord_sys.update()


actor_drone2_name = all_actors.add_actor(name='drone2',
                                             filename=os.path.join(paths.models,'drone1/drone1.json'),
                                             target_ray_spacing=0.005,scale_mesh=1)

velocity = 7  # in m/s
actor_phi = 180
actor_theta = 0
actor_psi = 0
drone2_pos = copy.copy(drone_pos)
drone2_pos[0] += 100
drone2_pos[1] -= 10
drone2_pos[2] += 4
# all_actors.actors[actor_drone_name].rotor_ang = [0, 0, rotor_speed * 2 * np.pi]  # in rad/s
all_actors.actors[actor_drone2_name].velocity_mag = velocity
all_actors.actors[actor_drone2_name].coord_sys.pos = drone2_pos
all_actors.actors[actor_drone2_name].coord_sys.rot = euler_to_rot(actor_phi, actor_theta, actor_psi, order='zyx', deg=True)
all_actors.actors[actor_drone2_name].coord_sys.update()


actor_drone3_name = all_actors.add_actor(name='drone3',
                                             filename=os.path.join(paths.models,'drone1/drone1.json'),
                                             target_ray_spacing=0.005,scale_mesh=1.1)

velocity = 10  # in m/s
actor_phi = 0
actor_theta = 0
actor_psi = 0
drone3_pos = copy.copy(drone_pos)
drone3_pos[0] -= 20
drone3_pos[1] += 5
drone3_pos[2] += .5
# all_actors.actors[actor_drone_name].rotor_ang = [0, 0, rotor_speed * 2 * np.pi]  # in rad/s
all_actors.actors[actor_drone3_name].velocity_mag = velocity
all_actors.actors[actor_drone3_name].coord_sys.pos = drone3_pos
all_actors.actors[actor_drone3_name].coord_sys.rot = euler_to_rot(actor_phi, actor_theta, actor_psi, order='zyx', deg=True)
all_actors.actors[actor_drone3_name].coord_sys.update()

# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more flexibility
# and control over the parameters, without having to create/modify a json file
waveform_dict = {    "mode": "PulsedDoppler",    "output": "RANGE_DOPPLER",
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
ant_device = add_single_tx_rx(all_actors,waveform,mode_name,pos=[0.25,0,0],
                              beamwidth_H=140,
                              beamwidth_V=120,
                              scale_pattern=.4,
                              parent_h_node=all_actors.actors[actor_drone_name].h_node)


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
# print(pem.reportSettings())

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




numFrames = len(time_stamps)
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
    debug_logs = DebuggingLogs(output_directory=paths.output)

# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(paths.output, 'out_vis.mp4')


camera_orientation = {}
camera_orientation['cam_offset'] =[-2.4, 0, 1]
camera_orientation['focal_offset'] = [25, 0, .5]
camera_orientation['up'] =(0.0, 0.0, 1.0)
camera_orientation['view_angle'] =80

modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=rng_domain,
                             vel_domain=vel_domain,
                             fps=fps,
                             camera_orientation=camera_orientation,
                             camera_attachment='tx',
                             output_movie_name=output_movie_name)




print('running simulation...')
all_responses = []
for idx, time in enumerate(time_stamps):

    # update all coordinate systems
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)
    all_responses.append(response)
    # calculate response in dB to overlay in pyvista plotter
    im_data = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))

    # exporting radar camera images
    if export_debug:
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)
        if idx == 0:
            debug_logs.write_scene_summary(file_name=f'out_{idx}.json')

    modeler.update_frame(plot_data=im_data,plot_limits=[im_data.max()-100,im_data.max()])
    if idx == 0:
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