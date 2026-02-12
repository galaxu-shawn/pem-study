#
# Copyright ANSYS. All rights reserved.
#
# This example demonstrates a workflow that utilizes AI to predict the material properties of a scene, and then uses
# these predicted material properties within the simulation.
#
# The workflow is as follows:
# 1. Predict the material properties of a scene using the Semantic Urban Mesh Segmentation (SUMS) tool.
#       The work is based on this paper -
#       https://www.sciencedirect.com/science/article/pii/S0924271621001854
# 2. Convert the output of the SUMS tool to a format that can be used in the simulation.
#       convert_ply_to_btp_with_tex.py can convert the resulting ply file into a Perceive EM readable vtp format that
#       has the material properties baked into the mesh cell data. The mapping is currently based on the material index
#       defined in teh material_library.json file.

# Additional information for using the SUMS tool is available, contact arien.sligar@ansys.com for more information
# this example is using Helsinki data, which is free to use under the Creative Commons Attribution 4.0 International
# https://www.hel.fi/en/decision-making/information-on-helsinki/maps-and-geospatial-data/helsinki-3d#3d-mesh

#######################################


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
from pem_utilities.antenna_device import Waveform, add_single_rx, add_single_tx, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.router import Pick_Path
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.open_street_maps_geometry import find_random_location_by_points
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 

fps = 10
total_time = 5
dt = 1 / fps
num_frames = int(total_time / dt)
timestamps = np.linspace(0, total_time, num_frames)


# waveform to plot
center_freq = 3.85e9
num_freqs = 512
bandwidth = 300e6
cpi_duration = 100e-3
num_pulse_CPI = 100

# simulation parameters
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = 0.1

export_debug = True

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()
all_actors = Actors()

terrain_path = os.path.join(paths.models,'Helsinki')

all_terrain_actor_names = []

# add tiles in teh scene, centered around the center_tile, with a total ox tiles_x by tiles_y
center_tile = [1987,2691]
# center_tile = [1990,2695]
# center_tile = [1984,2688]
# tiles in x and y direction
tiles_x = 4
tiles_y =4
center_pos = [0,0,0]
z_max=0
all_meshes = []

# load the terrain tiles, using the center tile index, and the number of tiles in x and y direction
for x in range(tiles_x):
    for y in range(tiles_y):
        x_index = center_tile[0] - tiles_x//2 + x
        y_index = center_tile[1] - tiles_y//2 + y
        tile_name = f'Tile_+{x_index:03d}_+{y_index:03d}'
        if not os.path.exists(os.path.join(terrain_path, tile_name + '.vtp')):
            print(f'Error: file {tile_name} not found in {terrain_path}')
            print('Contact arien.sligar@ansys.com for access to the Helsinki data')
        file_name = os.path.abspath(os.path.join(terrain_path, tile_name + '.vtp'))
        actor_name = all_actors.add_actor(name=tile_name,
                                         filename=file_name,
                                         include_texture=True, map_texture_to_material=True)
        all_terrain_actor_names.append(actor_name)
        if x+1 == tiles_x//2+tiles_x%2 and y+1 == tiles_y//2+tiles_y%2 :
            center_tile_mesh = all_actors.actors[actor_name].get_mesh()
            center_pos = center_tile_mesh[0].center
            z_max = center_tile_mesh[0].bounds[5]
        all_meshes.extend(all_actors.actors[actor_name].get_mesh())


# this is a hardcoded path that we can use. If you don't want to use this path, you can
# set pre_picked=None on the custom_path function, and it will prompt you to manually pick a path
# from the cad model.
# This path is a pre-picked path that follows the terrain, and is used to place the actor along the path.
path = np.array([[6810.91142826, 4697.85762821,    9.42127645],
       [6804.18391757, 4706.53025857,    9.30636467],
       [6794.08291992, 4720.13263389,    9.13273312],
       [6740.2002233 , 4792.17190248,   9.89879114]])
path = None
pedestrian_speed = 1.4 # m/s, average walking speed of a pedestrian

# path class. This class will take the path and create a function that can be used to get the position and rotation
# of the actir along the path. The z_offset is used to offset the actor above the terrain, so it doesn't clip into the ground.
path_picked_rx = Pick_Path()
path_picked_rx.custom_path(mesh_list=all_meshes,
                           pre_picked=path, 
                           speed=pedestrian_speed,
                           upsample_selection=101,
                           z_offset=1,
                           snap_to_surface=True)

#add actor where Tx will be attached
tx_actor_name = all_actors.add_actor()
all_actors.actors[tx_actor_name].coord_sys.pos = [6791.85, 4762.6,   35.1]
# all_actors.actors[tx_actor_name].coord_sys.rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True)
all_actors.actors[tx_actor_name].coord_sys.update() # force update of the coordinate system


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
# Add 2 antenna to the scene, one tx and one rx
####################

# add a single Tx antenna to the scene at tx_pos
ant_device_tx = add_single_tx(all_actors,waveform,mode_name,
                              ffd_file='dipole.ffd',
                              scale_pattern=18,
                              parent_h_node=all_actors.actors[tx_actor_name].h_node)

######################
# Rx on walking user
####################
# the antenna can be attached to a specific body part, or just relative to the root location of the actor

#################
# Rx antennas


#add a multi part actor from dae
rx_actor_name = all_actors.add_actor(name='ped_actors',filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'), 
                                     target_ray_spacing=0.2,
                                     scale_mesh=1)
# all_actors.actors[rx_actor_name].velocity_mag = 1.0
all_actors.actors[rx_actor_name].use_linear_velocity_equation_update = False
all_actors.actors[rx_actor_name].coord_sys.update()

# i know the actor has a part called mixamorig_LeftHand, and I want to attach the antenna to that part
# if you want a list of all body parts, just look at all_actors.actors[rx_actor_name].parts
node_to_attach_rx = all_actors.actors[rx_actor_name].parts['mixamorig_LeftHand'].h_node


# import this antenna into an existing actor, we can just move the actor and the antenna will move with it
ant_device_rx = add_single_rx(all_actors,
                              waveform,
                              mode_name,
                              pos=(0.05,0,0.05), # you can add an extra offset to the antenna device from the node_to_attach_rx
                              rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True), # extra rotation of the antenna device
                              parent_h_node=node_to_attach_rx, # antenna device will attach to this node (which is a body part of the actor)
                              scale_pattern=0.5, # for visualization purposes, this will scale the antenna pattern
                              ffd_file='dipole.ffd')

# between all the existing tx, and rx antennas, which ones do we want to compute coupling between
enable_coupling(mode_name,ant_device_tx, ant_device_rx)



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

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)

output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             overlay_results=True,
                             fps=fps,
                             camera_orientation="follow",
                             camera_attachment=rx_actor_name,
                             output_movie_name=output_movie_name)

print('running simulation...')

# modeler.pl.camera_position = [(7671.233362233306, 5756.587953356666, 790.4770225294501),
#                          (6777.823051566137, 4863.177642689497, -102.93328813771922),
#                          (0.0, 0.0, 1.0)]
all_results = []
for idx_frame in tqdm(range(num_frames), disable=False):
    all_reponses = []
    # print(modeler.pl.camera_position)
    time = idx_frame * dt

    all_actors.actors[rx_actor_name].coord_sys.pos = path_picked_rx.pos_func(time)
    all_actors.actors[rx_actor_name].coord_sys.rot = path_picked_rx.rot_func(time)
    all_actors.actors[rx_actor_name].coord_sys.update()


    # update all coordinate systems
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)
    pem_api_manager.isOK(pem.computeResponseSync())


    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes[mode_name],
                                                ant_device_rx.modes[mode_name],
                                                RssPy.ResponseType.FREQ_PULSE)
    # response will be a 4D array with dimensions [num_tx, num_rx, num_pulses, num_freqs]

    # exporting radar camera images
    if export_debug:
        debug_logs.write_scene_summary(file_name=f'out.json')
        debug_camera.generate_image()

    im_data = 20 * np.log10(np.fmax(np.abs(response[0,0]), 1.e-30)) # only show pulse vs freqency for the first Tx and Rx
    max_of_data = np.max(im_data)
    modeler.update_frame(plot_data=im_data.T,plot_limits=[max_of_data-30,max_of_data])

modeler.close()
print('Done')

if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')
