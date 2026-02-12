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
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.load_mesh import MeshLoader
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import *

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

# radar parameters
center_freq = 25.0e9
num_freqs = 3 # number of frequency samples, this is the number of samples in the frequency domain, using 3 so middle sample is center freq
bandwidth = 1000e6
num_pulse_CPI = 3 # static simulation so all pulses we will be the same (no moving targets), simulation setup requires >1 pulse
cpi_duration =1 # simulation is static so this parameter doesn't really matter, but set to 1 second for easy math

# simulation parameters
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 2
ray_spacing = 0.25  # global ray spacing in meters, each actor can have a different ray spacing if you want

export_debug = False
# output file will be stored in this directory
os.makedirs(paths.output, exist_ok=True)

mat_manager = MaterialManager()

# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()

actor_target_name1 = all_actors.add_actor(filename=os.path.join(paths.models,'F.stl'),
                                 mat_idx=mat_manager.get_index('pec'),
                                 color='black',
                                 target_ray_spacing=0.01)
actor_target_name2 = all_actors.add_actor(filename=os.path.join(paths.models,'F.stl'),
                                 mat_idx=mat_manager.get_index('glass'),
                                 color='red',
                                 target_ray_spacing=0.01)
actor_target_name3 = all_actors.add_actor(filename=os.path.join(paths.models,'F.stl'),
                                 mat_idx=mat_manager.get_index('absorber'),
                                 color='green',
                                 target_ray_spacing=0.01)

all_actors.actors[actor_target_name1].coord_sys.pos = [-.03, 0, -.1]
all_actors.actors[actor_target_name1].coord_sys.update()
all_actors.actors[actor_target_name2].coord_sys.pos = [-.03, 0, 0]
all_actors.actors[actor_target_name2].coord_sys.update()
all_actors.actors[actor_target_name3].coord_sys.pos = [-0.03, 0, +.1]
all_actors.actors[actor_target_name3].coord_sys.update()

prim = Plane(i_size=1,j_size=1,num_i=10,num_j=10,orientation=[1,0,0]) # if use an rcs value, it will override teh length and calculate length from rcs

prim_name = all_actors.add_actor(name='prim_example', generator=prim, mat_idx=mat_manager.get_index('pec'),
                                 target_ray_spacing=0.01,dynamic_generator_updates=False) #dynamic_generator_updates=False will not create a new geometry at each time step


actor_radar_name = all_actors.add_actor()


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
mode_name = 'mode1'
waveform = Waveform(waveform_dict)

###########Load Scenario Configuration Files ########
apertureY = .4
apertureZ = .4
dy = 0.00625
dz = 0.00625

antenna_offset = -.4  # only using x position for now

###############################

ny = int(apertureY / dy) + 1
nz = int(apertureZ / dz) + 1

# x = np.zeros((ny))+antenna_offset
# y = np.linspace(-apertureY / 2, apertureY / 2, num=ny)
# z = np.zeros((ny))+apertureZ/2
#
# all_positions = np.array([x, y, z]).T
positions_y = np.linspace(-apertureY / 2, apertureY / 2, num=ny)
positions_z = np.linspace(-apertureZ / 2, apertureZ / 2, num=nz)
total_num_points = nz * ny
all_positions = np.zeros((total_num_points, 3))
all_velocities = np.zeros((total_num_points, 3))
all_y, all_z = np.meshgrid(positions_y, positions_z)
idx = 0
all_idx = []
idx_y = 0
idx_z = 0
for y in positions_y:
    idx_z = 0
    for z in positions_z:
        all_positions[idx][0] = antenna_offset
        all_positions[idx][1] = y
        all_positions[idx][2] = z
        all_idx.append([idx_y, idx_z])
        idx += 1
        idx_z += 1
    idx_y += 1
all_idx = np.array(all_idx)
all_actors.actors[actor_radar_name].coord_sys.lin = [0, 0, 0]

########################## Setup Radar Platform  ##########################
# helpful utility function to easily create a single tx/rx device, where tx and rx are colocated
# range_pixels and doppler_pixels are not used if output is FREQPULSE, if RANGE_DOPPLER, then they are used to upsample
# attach the device to the radar actor (the empty one we created above)
ant_device = add_single_tx_rx(all_actors,
                              waveform,
                              mode_name,
                              parent_h_node=all_actors.actors[actor_radar_name].h_node,
                              ffd_file='dipole.ffd',
                              scale_pattern=.1)

# assign modes to devices

sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360 # if 180 it will only show in the +X direction
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
sim_options.auto_configure_simulation()




# display setup
# print(api.reportSettings())

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



fps = 10
dt = 1 / fps
T = 5
numFrames = int(T / dt)
responses = []
cameraImages = []

if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)
# # create a pyvista plotter
modeler = ModelVisualization(all_actors, fps=fps, shape=(ny, nz))
# modeler.pl.add_axes_at_origin()
imData = np.zeros((ny, nz))
all_results = np.zeros((ny, nz), dtype='complex')
print('running simulation...')
for idx in tqdm(range(len(all_positions))):

    idx_ny = all_idx[idx][0]
    idx_nz = all_idx[idx][1]
    # update all coordinate systems
    all_actors.actors[actor_radar_name].coord_sys.pos = all_positions[idx]
    all_actors.actors[actor_radar_name].coord_sys.update()

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)
    # imData[idx_ny, idx_nz] = np.angle(response[0, 0, 1, 1])
    imData[idx_ny, idx_nz] = np.abs(response[0, 0, 1, 1])
    response = response[0, 0, 1, 1]  # [tx,rx,chirp#]
    all_results[idx_ny, idx_nz] = response
    # calculate response in dB to overlay in pyvista plotter

    # modeler.mpl_ax_handle.set_data(imData)  # update pyvista matplotlib plot

    if export_debug:
        debug_logs.write_scene_summary(file_name=f'out.json')
        debug_camera.generate_image()
    for_plot = 20 * np.log10(np.abs(imData))
    max_data = np.max(for_plot)
    modeler.update_frame(plot_data=for_plot,plot_limits=[max_data-40,max_data])

modeler.close()
if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')
print('starting post processing ')

# only want one channel


# fig, ax = plt.subplots()
# ax.imshow(np.flipud(np.real(all_results_freq_pos[0]).T))
# plt.show()


positions_y = np.linspace(-apertureY / 2, apertureY / 2, num=ny)
positions_z = np.linspace(-apertureZ / 2, apertureZ / 2, num=nz)
all_y, all_z = np.meshgrid(positions_y, positions_z)

ky_const = (2 * np.pi / apertureY)
ky_num = np.linspace(-ny / 2, ny / 2 - 1, num=ny)
ky = ky_const * ky_num

kz_const = (2 * np.pi / apertureZ)
kz_num = np.linspace(-nz / 2, nz / 2 - 1, num=nz)
kz = kz_const * kz_num

ky_grid, kz_grid = np.meshgrid(ky, kz)

focus = 0.37

freq = 25e9

k = 2 * np.pi * freq / 3e8

k1 = 4.0 * np.power(k, 2) - np.power(ky_grid, 2) - np.power(kz_grid, 2) + 0j
K_all = np.exp(complex(0, 1) * np.sqrt(k1) * focus)

data_one_freq = all_results
step1 = np.fft.fft2(data_one_freq)
step1 = np.fft.fftshift(step1)
step1 = np.rot90(step1)
step2 = K_all * step1
step3 = np.rot90(np.fliplr(np.fft.ifft2(step2)))  # get orientaiton correct

######################## PLOT ################################################


plt.close('all')
max_data = np.max(np.abs(step3))
print(max_data)
print(np.min(np.abs(step3)))
fig, ax = plt.subplots()
ax.imshow(np.flipud(np.abs(step3)),cmap='jet',vmin=1.5e-7,vmax=0.0015825638780746147)
plt.show()
