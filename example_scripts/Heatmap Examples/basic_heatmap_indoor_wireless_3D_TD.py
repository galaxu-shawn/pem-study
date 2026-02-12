#
# Copyright ANSYS. All rights reserved.
#
#######################################
# Example of P2P coupling between a pedestrian and a building using multiple animated people

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

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.animation_selector_CMU import select_CMU_animation
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx, add_single_rx, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.heat_map import HeatMap
from pem_utilities.simulation_options import SimulationOptions

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

########################################################################################################################
# USER INPUTS
########################################################################################################################

center_pos = [5,-4,0] # about the center of the house
bounding_box_offset_x = 5 # in meter, + and - from center_pos
bounding_box_offset_y = 4
bounding_box_offset_z = 3


# grid bounds where heatmap will be calculated [xmin,xmax,ymin,ymax,zmin,zmax]
grid_bounds = [center_pos[0]-bounding_box_offset_x, center_pos[0]+bounding_box_offset_x,
               center_pos[1]-bounding_box_offset_y, center_pos[1]+bounding_box_offset_y,
               0,bounding_box_offset_z] #

sampling_spacing_wl = 4

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 2
ray_density = 0.2


# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 6e9
num_freqs = 101
bandwidth = 1500e6
cpi_duration = 0.9e-3
num_pulse_CPI = 2

export_debug = True


# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

debug_logs = DebuggingLogs(output_directory=paths.output)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()  # all actors, using the same material library for everyone


#add a multi part actor from json
house_name = all_actors.add_actor(name='house',
                                  filename=os.path.join(paths.models,'Home_P2P/Home_P2P.json'),
                                  transparency=0.7, scale_mesh=1.0) #
#add a multi part actor from dae
ped1_name = all_actors.add_actor(filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'), target_ray_spacing=0.1, scale_mesh=1)
all_actors.actors[ped1_name].velocity_mag = 1.0
all_actors.actors[ped1_name].coord_sys.pos = (3, -7, 0.)
all_actors.actors[ped1_name].coord_sys.update() # this updates the root cs where the actor will be placed
# becuase it is an animation, we need to make sure we call update_actor() to update all the joint positions
# normally this would be updated during the simulation loop, but becuase this is a static heatmap, we will not be
# calling an updates to the scene
all_actors.actors[ped1_name].update_actor(time=0)

# cmu_filename, cmu_description = select_CMU_animation(activity_type='walking') # random selection within walking category
# cmu_filename, cmu_description = select_CMU_animation() # random selection
# cmu_filename, cmu_description = select_CMU_animation(file_name='05_11') # specific selection
# ped2_name = all_actors.add_actor(filename=cmu_filename, target_ray_spacing=0.1, scale_mesh=1)
# all_actors.actors[ped2_name].coord_sys.pos = (3.5, -5, 0.)
# all_actors.actors[ped2_name].coord_sys.update()
# cmu_filename, cmu_description = select_CMU_animation(file_name='19_10') # specific selection
# ped3_name = all_actors.add_actor(filename=cmu_filename, target_ray_spacing=0.1, scale_mesh=1)
# all_actors.actors[ped3_name].coord_sys.pos = (8, -5.5, 0.)
# all_actors.actors[ped3_name].coord_sys.update()


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

node_to_attach_tx = all_actors.actors[ped1_name].parts['mixamorig_LeftHand'].h_node

antenna_device_tx = add_single_tx(all_actors, waveform, mode_name, pos=[0,.1,0], ffd_file='dipole.ffd',
                               scale_pattern=.1,parent_h_node=node_to_attach_tx)

#empty probe that will be what gets moved around to sample fields
probe_name = all_actors.add_actor(name='probe')
# show_results_in_modeler=False, this is to not show the results in the modeler we will use the resulting heatmap
# to plot the animated time series after the frequency domain heatmap is calculated
heatmap = HeatMap(all_actors=all_actors,
                  sampling_spacing_wl=sampling_spacing_wl,
                  bounds=grid_bounds,
                  waveform=waveform,
                  mode_name=mode_name,
                  num_subgrid_samples_nx=10,
                  num_subgrid_samples_ny=10,
                  polarization='Z',
                  show_patterns=False,
                  show_results_in_modeler=False,
                  cmap='jet',
                  )

if export_debug:
    debug_camera = DebuggingCamera(hMode=antenna_device_tx.modes[mode_name],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=1)

# set up response composition, and which modes to capture
enable_coupling(mode_name,antenna_device_tx, heatmap.probe_device)

# assign modes to devices
print(pem.listGPUs())

# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = 2.99792458e8 / center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density

sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(pem.getLastWarnings())

which_mode = antenna_device_tx.modes[mode_name]  # tell it which mode we want to get response from
antenna_device_tx.waveforms[mode_name].get_response_domains(which_mode)

# this is calculated for round trip, multiply by 2 to get one way
rng_domain = antenna_device_tx.waveforms[mode_name].rng_domain * 2
time_domain = rng_domain / 3e8



# video output speed
fps = 20

output_movie_name = os.path.join(paths.output, 'out_vis_indoor_wireless_heatmap_TD.mp4')
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

heatmap.update_heatmap_time_domain_3d(tx_mode=antenna_device_tx.modes[mode_name],
                                       probe_mode=heatmap.probe_device.modes[mode_name],
                                       function='db',
                                       modeler=modeler,
                                       plot_min=-81,
                                       plot_max=-60,
                                       td_output_size=256,
                                       window='hamming',
                                       add_mesh_to_overlay=True,
                                       end_animation_after_time=0.3e-6,
                                       use_slider_widget=False)
# set use_slider_widget to make a movie or use a slider bar to control animation


if export_debug:
    debug_logs.write_scene_summary(file_name=f'out.json')
    debug_camera.generate_image()
    debug_camera.write_image_to_file('debug_camera.png')
modeler.close()

