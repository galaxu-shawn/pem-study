# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

@author: asligar
"""

import numpy as np
import os
import sys
import copy

# Perceive EM Imports and Utilities
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, enable_coupling, add_single_tx
from pem_utilities.heat_map import HeatMapArbitraryPoints
from pem_utilities.open_street_maps_geometry import TerrainPrep, get_z_elevation_from_mesh
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

#######################################################################################################################
# Input Parameters
#######################################################################################################################tx_pos = [8.5, 21, 27]
tx_pos = [20, 0, 2000]

# range of values that lat/lon can be generated around
scene_lat_lon = (45.333849, -121.708249) #timberline lodge, mt hood
max_radius = 500 # meters


rx_zpos = 20 # elevation above ground

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 3
max_num_trans = 0
ray_density = .02

# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 2.14e9
num_freqs = 512
bandwidth = 200e6
cpi_duration = 0.9e-3
num_pulse_CPI = 2

export_debug = False

#######################################################################################################################
# Setup and Run Perceive EM
#######################################################################################################################

debug_logs = DebuggingLogs(output_directory=paths.output)

mat_manager = MaterialManager()
all_actors = Actors()  # all actors, using the same material library for everyone

print(f'Generating Scene: Lat/Lon: {scene_lat_lon}...')

terrain_prep = TerrainPrep(paths.output)
terrain = terrain_prep.get_terrain(center_lat_lon=scene_lat_lon, max_radius=max_radius)

terrain_name = all_actors.add_actor(filename=terrain['file_name'],
                                 mat_idx=mat_manager.get_index('asphalt'),
                                 color='grey',transparency=0.5)

terrain_mesh = all_actors.actors[terrain_name].get_mesh()


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

ant_device_tx = add_single_tx(all_actors,waveform,mode_name,pos=tx_pos,ffd_file='dipole.ffd',scale_pattern=10.0)

all_points_of_mesh = copy.copy(terrain_mesh[0].points)
all_points_of_mesh[:, 2] = all_points_of_mesh[:, 2] + rx_zpos # offset in Z position from mesh
# minmax_pos = get_z_elevation_from_mesh([self.all_grid_positions[idx][0],self.all_grid_positions[idx][1]],self.mesh)

heatmap = HeatMapArbitraryPoints(ant_device_tx,
                  all_actors=all_actors,
                  list_of_points=all_points_of_mesh,
                  waveform=waveform,
                  mode_name=mode_name,
                  polarization='Z',
                  show_patterns=True,
                  cmap='inferno',
                  opacity=1.0
                  )


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
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes['mode1'],
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

# this is calculated for round trip, multiply by 2 to get one way
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain * 2
time_domain = rng_domain / 3e8

print(f"Range domain max: {rng_domain[-1]}")
print(f"Time domain max: {time_domain[-1]}")

# video output speed
fps = 100

output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             overlay_results=False,
                             show_antennas=True,
                             fps=fps,
                             camera_attachment=None,
                             camera_orientation=None,
                             output_movie_name=output_movie_name)

modeler.pl.show_grid()

print("Running Perceive EM Simulation...")

# heatmap.update_heatmap(function='db',
#                        modeler=modeler,
#                        plot_min=-100,
#                        plot_max=-50,)


heatmap.update_heatmap_time_domain(function='db',
                                   modeler=modeler,
                                   plot_min=-250,
                                   plot_max=-150,
                                   add_mesh_to_overlay=True,
                                   td_output_size=512,
                                   window='hann',
                                   start_animation_after_time=None,
                                   end_animation_after_time=None,
                                   use_slider_widget=False,
                                   loop_animation=True)

if export_debug:
    debug_logs.write_scene_summary(file_name=f'out.json')
    debug_camera.generate_image()
    debug_camera.write_image_to_file('debug_camera.png')


# add point picking to the modeler so you can interactively get the value of the heatmap at a specific point
# def callback(point):
#     pid = heatmap.mesh.find_closest_point(point)
#
#     point = heatmap.mesh.points[pid]
#     array_name = heatmap.mesh.active_scalars_name
#     label = ['ID: {}\n{}: {}'.format(pid,
#                                      array_name,
#                                      heatmap.mesh[array_name][pid])]
#     modeler.pl.add_point_labels(point, label)
#
# modeler.pl.enable_point_picking(callback=callback, show_message=True,
#                                color='pink', point_size=10, show_point=True)
modeler.update_frame(write_frame=False)  # if write_frame=False, no video will be created, just the modeler shown.
modeler.close()




