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
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, enable_coupling, add_single_tx
from pem_utilities.heat_map import HeatMap
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.router import Pick_Path

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

#######################################################################################################################
#       PROBE: 2D Array of Antennas                 # Scene
#              ny, dy  -> Total Y,
#              nx, dx  -> Total x
#              [                                    |--------------------------> Scene Size Y
#                                                   [*][*][*][*][*][*][*][*][*]
#         |-----------> Y                           [*][*][*][*][*][*][*][*][*]
#       _ * * * * * * *                             [*][*][*][*][*][*][*][*][*]
#       | * * * * * * *                             [*][*][*][*][*][*][*][*][*] <-- Probe, stepped through scene
#       | * * * * * * *                                                              Spacing is Total Y and Total X
#       | * * * * * * *                                                              [*] probe size determines spacing
#       V * * * * * * *
#       X
#
#######################################################################################################################




#######################################################################################################################
# Input Parameters
#######################################################################################################################tx_pos = [8.5, 21, 27]
tx_pos = [6778.24214979, 4821.19459181,   61.82115445]


# rx postion, used to define grid
# rx_pos = [6778.24214979, 4821.19459181,   20] # center of the grid
rx_pos = np.array([6875.0, 4875.0, 8.385555684566498])

rx_offset_x = 200 # in meter, + and - from rx_pos
rx_offset_y = 200

# grid bounds where heatmap will be calculated [xmin,xmax,ymin,ymax]
grid_bounds = [rx_pos[0]-rx_offset_x, rx_pos[0]+rx_offset_x,
               rx_pos[1]-rx_offset_y, rx_pos[1]+rx_offset_y,
               0,100]
rx_zpos = rx_pos[2]
sampling_spacing_wl = 20

# simulation options
go_blockage = 0 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 7
max_num_trans = 1
ray_density = .2

# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 2.4e9
num_freqs = 256
bandwidth = 150e6
cpi_duration = 100e-3
num_pulse_CPI = 3

export_debug = False

#######################################################################################################################
# Setup and Run Perceive EM
#######################################################################################################################

debug_logs = DebuggingLogs(output_directory=paths.output)

mat_manager = MaterialManager()
all_actors = Actors()  # all actors, using the same material library for everyone
actor_scene_ref_name = all_actors.add_actor()
terrain_path = os.path.join(paths.models,'Helsinki')

all_terrain_actor_names = []

# add tiles in teh scene, centered around the center_tile, with a total ox tiles_x by tiles_y
center_tile = [1987,2691]
# center_tile = [1990,2695]
# center_tile = [1984,2688]
# tiles in x and y direction
tiles_x = 3
tiles_y =3

z_max=0
all_meshes = []
terrain_bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]
cam_actor_name = all_actors.add_actor()
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
                                             include_texture=True,
                                             map_texture_to_material=True,
                                             parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                             target_ray_spacing=.2)


        mesh = all_actors.actors[actor_name].get_mesh()
        bounds = mesh[0].bounds
        if bounds[0] < terrain_bounds[0]:
            terrain_bounds[0] = bounds[0]
        if bounds[1] > terrain_bounds[1]:
            terrain_bounds[1] = bounds[1]
        if bounds[2] < terrain_bounds[2]:
            terrain_bounds[2] = bounds[2]
        if bounds[3] > terrain_bounds[3]:
            terrain_bounds[3] = bounds[3]
        if bounds[4] < terrain_bounds[4]:
            terrain_bounds[4] = bounds[4]
        if bounds[5] > terrain_bounds[5]:
            terrain_bounds[5] = bounds[5]
        all_meshes.extend(mesh)

grid_bounds[4] = 13
grid_bounds[5] = 70
# allow user interactivity to set the camera angle before the animation starts
pl = pv.Plotter(notebook=False)
for each in all_meshes:
    pl.add_mesh(each,color='grey')

camera_position = [(6926.889614903849, 4363.312459819196, 196.5364202540863),
 (6832.486880100233, 4853.374650945023, -15.865415809700462),
 (-0.09792638158092881, 0.37982097647991814, 0.9198621905569919)]

# pl.camera_position = camera_position
def my_cpos_callback():
    pl.add_text(str(pl.camera.position), position='lower_left', name="cpos")
    print(pl.camera_position)
    return
pl.add_key_event("p", my_cpos_callback)
pl.add_text("Interactively Set Camera Angle\nPress P and close window to set", position='upper_left', font_size=12, color='black')
pl.show()


# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more flexibility
# and control over the parameters, without having to create/modify a json file
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

# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)

# initialize the antenna device, one for Tx, one for Rx

ant_device_tx = add_single_tx(all_actors,waveform,mode_name,pos=tx_pos,ffd_file='dipole.ffd',scale_pattern=15.0)

heatmap = HeatMap(all_actors=all_actors,
                  sampling_spacing_wl=sampling_spacing_wl,
                  bounds=grid_bounds,
                  z_elevation=rx_zpos,
                  waveform=waveform,
                  mode_name=mode_name,
                  num_subgrid_samples_nx=10,
                  num_subgrid_samples_ny=10,
                  polarization='Z',
                  show_patterns=False,
                  cmap='jet',
                  opacity=0.9
                  )


enable_coupling(mode_name,ant_device_tx, heatmap.probe_device)

if ray_density is not None:
    lambda_center = 2.99792458e8 / ant_device_tx.waveforms[mode_name].center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density

# assign modes to devices
print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes[mode_name],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=1)

# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(pem.getLastWarnings())

which_mode = ant_device_tx.modes[mode_name]  # tell it which mode we want to get response from
ant_device_tx.waveforms[mode_name].get_response_domains(which_mode)

# this is calculated for round trip, multiply to get one way
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain * 2
time_domain = rng_domain / 3e8

print(f"Range domain max: {rng_domain[-1]}")
print(f"Time domain max: {time_domain[-1]}")

# video output speed
fps = 20

output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=fps,
                             shape=(heatmap.total_samples_x, heatmap.total_samples_y),
                             x_domain=heatmap.x_domain,
                             y_domain=heatmap.y_domain,
                             overlay_results=False,
                             camera_attachment=None,
                             camera_orientation=None,
                             camera_position=camera_position,
                             output_movie_name=output_movie_name)

modeler.pl.background_color = [1, 1, 1]
# modeler.pl.show_grid()
all_scalar_bars = list(modeler.pl.scalar_bars.keys())
for each in all_scalar_bars:
    modeler.pl.remove_scalar_bar(each)

print("Running Perceive EM Simulation...")


print("Running Perceive EM Simulation...")
# modeler.pl.camera_position = camera_position
heatmap.update_heatmap_time_domain_3d(tx_mode=ant_device_tx.modes[mode_name],
                                      probe_mode=heatmap.probe_device.modes[mode_name],
                                      function='db',
                                      modeler=modeler,
                                      plot_min=-130,
                                      plot_max=-80,
                                      td_output_size=512,
                                      window='hamming',
                                      add_mesh_to_overlay=True,
                                      end_animation_after_time=1.65e-06,
                                      loop_animation=True,
                                      use_slider_widget=False)

if export_debug:
    debug_logs.write_scene_summary(file_name=f'out.json')
    debug_camera.generate_image()
    debug_camera.write_image_to_file('debug_camera.png')

modeler.close()



