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
tx_pos = [20, 0, 1.5]


# rx postion, used to define grid
rx_pos = [20, 0, 1.5] # center of the grid
rx_offset_x = 80 # in meter, + and - from rx_pos
rx_offset_y = 60

# grid bounds where heatmap will be calculated [xmin,xmax,ymin,ymax]
grid_bounds = [rx_pos[0]-rx_offset_x, rx_pos[0]+rx_offset_x,
               rx_pos[1]-rx_offset_y, rx_pos[1]+rx_offset_y]
rx_zpos = rx_pos[2]
sampling_spacing_wl = 10

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .2

# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 2.14e9
num_freqs = 2
bandwidth = 200e6
cpi_duration = 0.9e-3
num_pulse_CPI = 2

export_debug = False
save_results = True
#######################################################################################################################
# Setup and Run Perceive EM
#######################################################################################################################

debug_logs = DebuggingLogs(output_directory=paths.output)

mat_manager = MaterialManager()
all_actors = Actors()  # all actors, using the same material library for everyone
file_path = os.path.join(paths.models, 'whole-scene-static.stl')
city_name = all_actors.add_actor(filename=file_path,
                                 mat_idx=mat_manager.get_index('asphalt'),
                                 color='grey',transparency=0.5)

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
                  cmap='inferno',
                  opacity=1.0
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
    debug_camera = DebuggingCamera(hMode=antenna_device_tx.modes['mode1'],
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
fps = 100

output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=fps,
                             shape=(heatmap.total_samples_x, heatmap.total_samples_y),
                             x_domain=heatmap.x_domain,
                             y_domain=heatmap.y_domain,
                             camera_attachment=None,
                             camera_orientation=None,
                             output_movie_name=output_movie_name)

modeler.pl.show_grid()

print("Running Perceive EM Simulation...")

heatmap.update_heatmap(tx_mode=ant_device_tx.modes[mode_name],
                       probe_mode=heatmap.probe_device.modes[mode_name],
                       function='db',
                       modeler=modeler,
                       plot_min=-100,
                       plot_max=-50,)

if export_debug:
    debug_logs.write_scene_summary(file_name=f'out.json')
    debug_camera.generate_image()
    debug_camera.write_image_to_file('debug_camera.png')


# add point picking to the modeler so you can interactively get the value of the heatmap at a specific point
def callback(point):
    pid = heatmap.mesh.find_closest_point(point)

    point = heatmap.mesh.points[pid]
    array_name = heatmap.mesh.active_scalars_name
    label = ['ID: {}\n{}: {}'.format(pid,
                                     array_name,
                                     heatmap.mesh[array_name][pid])]
    modeler.pl.add_point_labels(point, label)

modeler.pl.enable_point_picking(callback=callback, show_message=True,
                               color='pink', point_size=10, show_point=True)
modeler.update_frame(write_frame=False)  # if write_frame=False, no video will be created, just the modeler shown.
modeler.close()

if save_results:
    # save results, you can use Results_View_2D_Heamap.py to plot these results again so you don't need to rerun the simulation
    np.save(os.path.join(paths.output,'output.npy'),heatmap.image)
    np.save(os.path.join(paths.output,'grid_x.npy'),heatmap.x_domain)
    np.save(os.path.join(paths.output,'grid_y.npy'),heatmap.y_domain)


