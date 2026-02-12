#
# Copyright ANSYS. All rights reserved.
#

#######################################
#######################################
# IMPORTANT! edit api_core.py to use the correct API, P2P API is needed for this script to run
#######################################
#######################################


import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import os
import sys
import time as walltime
import pyvista as pv

from pathlib import Path
# Dynamically resolve the path to the api_core module
script_path = Path(__file__).resolve()
api_core_path = script_path.parent.parent  # Adjusted to point to the correct parent directory
model_path = os.path.join(script_path.parent.parent, 'models')
output_path = os.path.join(script_path.parent.parent, 'output')
# output file will be stored in this directory
os.makedirs(output_path, exist_ok=True)
if api_core_path not in sys.path:
    sys.path.insert(0, str(api_core_path))
import pem_utilities.pem_core as pem_core
RssPy = pem_core.RssPy
api = pem_core.api

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaDevice, Waveform, AntennaArray
from pem_utilities.router import Pick_Path
from pem_utilities.heat_map import HeatMap
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

sampling_spacing_wl = 2000
# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .02

# Perceive EM waveform for Comms
center_freq = 30e9
num_freqs = 512
bandwidth = 150e6
cpi_duration = 100e-3
num_pulse_CPI = 3

export_debug = False  #use this to export the movie at the end of the simulation (.gif showing radar camera and response)

is_tx_on = {'bs1': True, 'bs2': False, 'bs3': False}

# what to display

debug_logs = DebuggingLogs(output_directory=output_path)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

mat_manager = MaterialManager()
all_actors = Actors()  # all actors, using the same material library for everyone

# these are all the objects that are part of the scene


all_terrain = ['Tile_+026_+020', 'Tile_+026_+021',
               'Tile_+027_+019', 'Tile_+027_+020', 'Tile_+027_+021', 'Tile_+027_+022',
               'Tile_+028_+020', 'Tile_+028_+021', 'Tile_+028_+022', 'Tile_+028_+023',
               'Tile_+029_+021', 'Tile_+029_+022']
terrain_path = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Scripting\RFSX_2\scenarios\Denver_20mm_res\Denver_20mm_res'

all_terrain_meshes = []
terrain_bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]
for each in all_terrain:
    if not os.path.exists(os.path.join(terrain_path, each, each + '.obj')):
        print(f'Error: {os.path.join(terrain_path, each, each + ".obj")} does not exist')
        print('Aerometrix Dataset required to run this script')
        # issue an exeception and exit
        raise FileNotFoundError
    import_file_name = os.path.join(terrain_path, each, each + '.obj')
    terrain_name = all_actors.add_actor(name='terrain1',
                                        filename=import_file_name,
                                        include_texture=True, map_texture_to_material=False)
    mesh = all_actors.actors[terrain_name].get_mesh()
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

    all_terrain_meshes.extend(mesh)


# allow user interactivity to set the camera angle before the animation starts
pl = pv.Plotter(notebook=False)
for each in all_terrain_meshes:
    pl.add_mesh(each,color='grey')

camera_position = [(230.89597325368192, -533.8079601836918, 1904.3552629579922),
                     (382.667938112316, -51.75421959100261, 1618.1326682340084),
                     (0.19749595719218016, 0.45376930206556765, 0.8689584382441269)]
# top view
# camera_position = [(363.2687945557797, -43.5868886597776, 2097.951619982046),
#                  (394.1468665107654, -39.76169065947053, 1618.9573472723632),
#                  (0.7033490986782036, -0.7097366354112413, 0.03967308588563839)]

pl.camera_position = camera_position
def my_cpos_callback():
    pl.add_text(str(pl.camera.position), position='lower_left', name="cpos")
    print(pl.camera_position)
    return
pl.add_key_event("p", my_cpos_callback)
pl.add_text("Interactively Set Camera Angle\nPress P and close window to set", position='upper_left', font_size=12, color='black')
camera_position= pl.camera_position
pl.show()

# terrain_bounds[5] = terrain_bounds[5] - 100  #

# grid bounds where heatmap will be calculated [xmin,xmax,ymin,ymax,zmin,zmax]
grid_bounds = terrain_bounds

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
    "tx_multiplex": "SIMULTANEOUS",
    "mode_delay": "CENTER_CHIRP"}
pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
mode_name_comms = 'mode_comms'
waveform = Waveform(waveform_dict=waveform_dict)

#######################################################################################################################
# Setup Comms Antennas
#######################################################################################################################


######################
# SETUP  BASE STATIONS (32 element array acting as Tx)
######################
if is_tx_on['bs1']:
    bs1_root = all_actors.add_actor(name='bs1')
    bs1 = AntennaArray(name='bs1_array', waveform=waveform, mode_name=mode_name_comms,
                       beamwidth_H=140, beamwidth_V=120, polarization='V',
                       rx_shape=0, tx_shape=[1, 1], spacing_wl_x=0.5, spacing_wl_y=0.5, normal='x',
                       load_pattern_as_mesh=False, scale_pattern=5,
                       parent_h_node=all_actors.actors[bs1_root].h_node)
    bs1_pos = [412.6782, -79, 1624.]
    bs1_rot = euler_to_rot(phi=90, theta=15, order='zyz', deg=True)
    all_actors.actors[bs1_root].coord_sys.pos = bs1_pos
    all_actors.actors[bs1_root].coord_sys.rot = bs1_rot
    all_actors.actors[bs1_root].coord_sys.update()
    for each in bs1.antenna_device.all_antennas_properties:
        name = all_actors.add_actor(name=each, actor=bs1.antenna_device.all_antennas_properties[each]['Actor'])
    rComp = pem_core.RssPy.ResponseComposition.INDIVIDUAL
    pem_core.isOK(api.setTxResponseComposition(bs1.antenna_device.modes[mode_name_comms], rComp))
    ant_device_tx = bs1.antenna_device
if is_tx_on['bs2']:
    bs2_root = all_actors.add_actor(name='bs2')
    bs2 = AntennaArray(name='bs2_array', waveform=waveform, mode_name=mode_name_comms,
                       beamwidth_H=140, beamwidth_V=120, polarization='V',
                       rx_shape=0, tx_shape=[1, 1], spacing_wl_x=0.5, spacing_wl_y=0.5, normal='x',
                       load_pattern_as_mesh=False, scale_pattern=5,
                       parent_h_node=all_actors.actors[bs2_root].h_node)
    bs2_pos = [448.6, 31.3, 1612.693]
    bs2_rot = euler_to_rot(phi=-10, theta=15, order='zyz', deg=True)
    all_actors.actors[bs2_root].coord_sys.pos = bs2_pos
    all_actors.actors[bs2_root].coord_sys.rot = bs2_rot
    all_actors.actors[bs2_root].coord_sys.update()
    for each in bs2.antenna_device.all_antennas_properties:
        name = all_actors.add_actor(name=each, actor=bs2.antenna_device.all_antennas_properties[each]['Actor'])
    rComp = pem_core.RssPy.ResponseComposition.INDIVIDUAL
    pem_core.isOK(api.setTxResponseComposition(bs2.antenna_device.modes[mode_name_comms], rComp))
    ant_device_tx = bs2.antenna_device
if is_tx_on['bs3']:
    bs3_root = all_actors.add_actor(name='bs3')
    bs3 = AntennaArray(name='bs3_array', waveform=waveform, mode_name=mode_name_comms,
                       beamwidth_H=140, beamwidth_V=120, polarization='V',
                       rx_shape=0, tx_shape=[1, 1], spacing_wl_x=0.5, spacing_wl_y=0.5, normal='x',
                       load_pattern_as_mesh=True, scale_pattern=5,
                       parent_h_node=all_actors.actors[bs3_root].h_node)
    bs3_pos = [305, -11, 1604]
    bs3_rot = euler_to_rot(phi=-10, theta=15, order='zyz', deg=True)
    all_actors.actors[bs3_root].coord_sys.pos = bs3_pos
    all_actors.actors[bs3_root].coord_sys.rot = bs3_rot
    all_actors.actors[bs3_root].coord_sys.update()
    for each in bs3.antenna_device.all_antennas_properties:
        name = all_actors.add_actor(name=each, actor=bs3.antenna_device.all_antennas_properties[each]['Actor'])
    rComp = pem_core.RssPy.ResponseComposition.INDIVIDUAL
    pem_core.isOK(api.setTxResponseComposition(bs3.antenna_device.modes[mode_name_comms], rComp))
    ant_device_tx = bs3.antenna_device

#empty probe that will be what gets moved around to sample fields
probe_name = all_actors.add_actor(name='probe')
# show_results_in_modeler=False, this is to not show the results in the modeler we will use the resulting heatmap
# to plot the animated time series after the frequency domain heatmap is calculated
heatmap = HeatMap(all_actors=all_actors,
                  sampling_spacing_wl=sampling_spacing_wl,
                  bounds=grid_bounds,
                  waveform=waveform,
                  mode_name=mode_name_comms,
                  num_subgrid_samples_nx=10,
                  num_subgrid_samples_ny=10,
                  polarization='Z',
                  show_patterns=False,
                  show_results_in_modeler=False,
                  cmap='jet',
                  opacity=0.9
                  )

if is_tx_on['bs1']:
    pem_core.isOK(api.setDoP2PCoupling(bs1.antenna_device.h_node_platform, heatmap.probe_device.h_node_platform, True))
if is_tx_on['bs2']:
    pem_core.isOK(api.setDoP2PCoupling(bs2.antenna_device.h_node_platform, heatmap.probe_device.h_node_platform, True))
if is_tx_on['bs3']:
    pem_core.isOK(api.setDoP2PCoupling(bs3.antenna_device.h_node_platform, heatmap.probe_device.h_node_platform, True))


if ray_density is not None:
    lambda_center = 2.99792458e8 / ant_device_tx.waveforms[mode_name_comms].center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density

# assign modes to devices
print(api.listGPUs())
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
if not api.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(api.getLastWarnings())

# display setup
#print(api.reportSettings())

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes


if is_tx_on['bs1']:
    which_mode = bs1.antenna_device.modes[mode_name_comms]  # tell it which mode we want to get respones from
    which_waveform = bs1.antenna_device.waveforms[
        mode_name_comms]  # tell it which waveform we want to get respones from
    ant_device_tx = bs1.antenna_device
elif is_tx_on['bs2']:
    which_mode = bs2.antenna_device.modes[mode_name_comms]
    which_waveform = bs2.antenna_device.waveforms[mode_name_comms]
    ant_device_tx = bs2.antenna_device
elif is_tx_on['bs3']:
    which_mode = bs3.antenna_device.modes[mode_name_comms]
    which_waveform = bs3.antenna_device.waveforms[mode_name_comms]
    ant_device_tx = bs3.antenna_device

which_waveform.get_response_domains(which_mode)

rng_domain = which_waveform.rng_domain * 2
time_domain = rng_domain / 3e8

fps = 20

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=which_mode,
                                   display_mode='normal',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

responses = []
output_movie_name = os.path.join(output_path, 'out_vis_heatmap.mp4')
# bs1_root
# 'radar'
# actor_quadcopter_name
# camera_orientation = 'follow4',

if is_tx_on['bs1']:
    camera_attachment = bs1_root
elif is_tx_on['bs2']:
    camera_attachment = bs2_root
elif is_tx_on['bs3']:
    camera_attachment = bs3_root
else:
    camera_attachment = None
camera_attachment = None
camera_orientation = None
# # create a pyvista plotter

modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=fps,
                             shape=(heatmap.total_samples_x, heatmap.total_samples_y),
                             x_domain=heatmap.x_domain,
                             y_domain=heatmap.y_domain,
                             camera_attachment=camera_attachment,
                             camera_orientation=camera_orientation,
                             camera_position=camera_position,
                             output_movie_name=output_movie_name)

# modeler.pl.show_grid()

print("Running Perceive EM Simulation...")
print(camera_position)
heatmap.update_heatmap_time_domain_3d(tx_mode=ant_device_tx.modes[mode_name_comms],
                                      probe_mode=heatmap.probe_device.modes[mode_name_comms],
                                      function='db',
                                      modeler=modeler,
                                      plot_min=-120,
                                      plot_max=-75,
                                      td_output_size=512,
                                      window='hamming',
                                      add_mesh_to_overlay=True,
                                      end_animation_after_time=1.5e-6,
                                      use_slider_widget=False)
# set use_slider_widget to make a movie or use a slider bar to control animation


if export_debug:
    debug_logs.write_scene_summary(file_name=f'out.json')
    debug_camera.generate_image()
    debug_camera.write_image_to_file('debug_camera.png')
modeler.close()
