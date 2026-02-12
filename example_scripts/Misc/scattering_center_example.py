# The exmaple demonstrates the output type of Range Doppler for a corner reflector.
# Copyright ANSYS. All rights reserved.
#

import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import PIL.Image
import time as walltime
import os
import sys
import pandas as pd

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, look_at
from pem_utilities.antenna_device import AntennaArray, Waveform, add_single_tx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import *
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.post_processing_radar_imaging import isar_3d

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


# waveform parameters for radar
center_freq = 76.5e9
num_freqs = 128
bandwidth = 512e6
cpi_duration = 0.010
num_pulse_CPI = 3

# simulation options
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 4
max_num_trans = 1
ray_density = .1

export_debug = True  # use this to export the movie at the end of the simulation (.gif showing radar camera and response)



# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
# multiple material libraries can be loaded and the indices will be updated accordingly
# mat_manager = MaterialManager(['material_properties_ITU_3.85GHz.json','material_library.json'])
mat_manager = MaterialManager()
# output file will be stored in this directory

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

# if a filename is not provided in the previous line, this commented out code can be used to add part to the actor
# all_scene_actors['road'].add_part(filename='../models/intersection.stl')

rcs_dbsm = 10
rcs = 10 ** (rcs_dbsm / 10)


wl = 3e8 / center_freq
# prim = CornerReflector(rcs=rcs, wl=3e8/center_freq, orientation='x',
#                        is_square=False)  # if use an rcs value, it will override teh length and calculate length from rcs
# target_name = all_actors.add_actor(name='prim_example', generator=prim, mat_idx=mat_manager.get_index('pec'),
                                #  target_ray_spacing=wl/4,dynamic_generator_updates=False) #dynamic_generator_updates=False will not create a new geometry at each time step

target_name = all_actors.add_actor(name='target_truck',
                                   filename=os.path.join(paths.models, 'Chevy_Silverado','Chevy_Silverado.json'),
                                   target_ray_spacing=wl/4) 



all_actors.actors[target_name].coord_sys.pos = (0, 0, 0.)
all_actors.actors[target_name].coord_sys.lin = (0, 0, 0.)
all_actors.actors[target_name].coord_sys.rot = euler_to_rot(phi=45, theta=0, psi=0)  # rotate the radar to face the target
all_actors.actors[target_name].coord_sys.update()


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
mode_name = 'mode1'

# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)

#################
# Tx antenna

dist = 100
radar_actor_name = all_actors.add_actor() # easier to attach a camera to an actor rather than an antenna
all_actors.actors[radar_actor_name].coord_sys.pos = [dist, 0, 0]
all_actors.actors[radar_actor_name].coord_sys.rot = euler_to_rot(phi=180, theta=0, psi=0)  # rotate the radar to face the target
all_actors.actors[radar_actor_name].coord_sys.update()

num_pixels_az = 128
num_pixels_el = 128
observation_angle_az = 2.23 # deg
observation_angle_el = 2.23 # deg

# calculate the cone angle at distance dist
cone_angle_az = np.deg2rad(observation_angle_az/2)
cone_angle_el = np.deg2rad(observation_angle_el/2)
# calculate the width and height of the cone at distance dist
width = dist * np.tan(cone_angle_az)
height = dist
# create a cone primitive to visualize the radar field of view
# cone_prim = Cone(radius=width, height=height)

# # add the cone primitive to the radar actor
# fov_actor = all_actors.add_actor(generator=cone_prim,parent_h_node=all_actors.actors[radar_actor_name].h_node,)
# all_actors.actors[fov_actor].coord_sys.pos = [dist/2, 0, 0]
# all_actors.actors[fov_actor].coord_sys.update()

spacing_x = 2*width / num_pixels_az
spacing_y = 2*width / num_pixels_el
spacing_wl_x = spacing_x / wl
spacing_wl_y = spacing_y / wl

#use dipole antenna for easier radar range calculation
ant_array = AntennaArray(name='array',
                 waveform=waveform,
                 mode_name=mode_name,
                 polarization='VV',
                 beamwidth_H=140,
                 beamwidth_V=120,
                 planewave=False,
                 rx_shape=[1,num_pixels_az],
                 tx_shape=[1,1],
                 spacing_wl_x=spacing_wl_x,
                 spacing_wl_y=spacing_wl_y,
                 load_pattern_as_mesh=True,
                 scale_pattern=.1,
                 parent_h_node=all_actors.actors[radar_actor_name].h_node,
                 array_elements_centered=True, # center element is at origin, -xpos/2, -ypos/2 to center the array
                 normal='x',
                 all_actors=all_actors)

ant_device = ant_array.antenna_device
center_freq = ant_device.waveforms[mode_name].center_freq


sim_options = SimulationOptions(center_freq=center_freq,)

sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 180
sim_options.ray_shoot_method='sbr'
sim_options.ray_density = ray_density
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
output_format = ant_device.waveforms[mode_name].output

print(f'Max Range {rng_domain[-1]}')


# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
fps = 10
if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)


# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             x_domain=np.linspace(-observation_angle_az/2, observation_angle_az/2, num_pixels_az),
                             y_domain=np.linspace(-observation_angle_el/2, observation_angle_el/2, num_pixels_el),
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)
modeler.pl.show_grid()

all_p_received = []
all_distances = []
print('running simulation...')
responses = []
data_cube = np.empty((num_pixels_el, num_pixels_az, num_freqs), dtype=np.complex64)
for row in range(num_pixels_el):
    row_pos_z = spacing_y * (row - num_pixels_el // 2)
    ant_array.shift_antenna_positions(antenna_operation_mode_to_update='rx',pos_z=row_pos_z,all_actors=all_actors)
    all_actors.actors[target_name].coord_sys.update()

    start_sim_time = walltime.time()
    pem_api_manager.isOK(pem.computeResponseSync())
    end_sim_time = walltime.time()
    print(f'Simulation Time: {(end_sim_time - start_sim_time)*1e3} mseconds')
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)
    data_cube[row] = np.array(response, dtype=np.complex64)[0,:,1] # 1 Tx, all rx, only 1 pulse, all freq
    max_val = np.max(20*np.log10(np.abs(data_cube[row])))
    min_val = np.min(20*np.log10(np.abs(data_cube[row])))
    max_val = np.max(np.angle(data_cube[row]))
    min_val = np.min(np.angle(data_cube[row]))
    print(f'Row {row}, Max value: {max_val}, Min value: {min_val}')


    # modeler.update_frame(plot_data=20*np.log10(np.abs(data_cube[:,:,0])),plot_limits=[min_val,max_val],write_frame=True) #only update 1 freq
    modeler.update_frame(plot_data=np.angle(data_cube[:,:,0]),plot_limits=[-3,3],write_frame=True) #only update 1 freq

# convert numpy 3d array into pandas data frame labeld with 'Freq' 'IWavePhi' and 'IWaveTheta'
phi_domain = np.linspace(-observation_angle_az/2, observation_angle_az/2, num_pixels_az)
theta_domain = np.linspace(-observation_angle_el/2, observation_angle_el/2, num_pixels_el)
Freq = freq_domain
#
# Create coordinate meshgrids for all three dimensions
# theta_grid, phi_grid, freq_grid = np.meshgrid(IWaveTheta, IWavePhi, Freq, indexing='ij')

# Flatten all arrays to create 1D arrays for DataFrame
# theta_flat = theta_grid.flatten()
# phi_flat = phi_grid.flatten()
# freq_flat = freq_grid.flatten()
# data_flat = data_cube.flatten()

# # Create the DataFrame
# df = pd.DataFrame({
#     'IWaveTheta': theta_flat,
#     'IWavePhi': phi_flat, 
#     'Freq': freq_flat,
#     'data': data_cube.flatten()
# })

# print(f"DataFrame created with shape: {df.shape}")
# print(f"DataFrame columns: {df.columns.tolist()}")
# print(f"First few rows:")
# print(df.head())

# save the data cube, phi_domain, theta_domain and freq_domain to numpy arrays
np.save(os.path.join(paths.output, 'data_cube.npy'),data_cube)
np.save(os.path.join(paths.output, 'phi_domain.npy'),phi_domain)
np.save(os.path.join(paths.output, 'theta_domain.npy'),theta_domain)
np.save(os.path.join(paths.output, 'freq_domain.npy'),Freq)


img = isar_3d(data_cube, freq_domain=Freq, phi_domain=phi_domain, theta_domain=theta_domain, function='abs', size=(256,256,256), window=None)

if export_debug:
    debug_logs.write_scene_summary(file_name=f'out.json')



modeler.close()


