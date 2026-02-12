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

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, look_at
from pem_utilities.antenna_device import add_single_tx_rx, Waveform
from pem_utilities.utils import generate_rgb_from_array
from pem_utilities.far_fields import convert_ffd_to_gltf
from pem_utilities.post_processing import range_angle_map, range_profile
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import *
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


range_start= 500
range_stop = 700
range_dist = range_stop-range_start

# waveform parameters for radar
center_freq = 76.5e9
num_freqs = 451
bandwidth = 300e6
cpi_duration = 0.010
num_pulse_CPI = 3

# simulation options
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 12
max_num_trans = 1
ray_density = .01

export_debug = False  # use this to export the movie at the end of the simulation (.gif showing radar camera and response)



# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
# multiple material libraries can be loaded and the indices will be updated accordingly
# mat_manager = MaterialManager(['material_properties_ITU_3.85GHz.json','material_library.json'])
mat_manager = MaterialManager(generate_itu_materials=True, itu_freq_ghz=center_freq*1e-9)
# output file will be stored in this directory

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

# if a filename is not provided in the previous line, this commented out code can be used to add part to the actor
# all_scene_actors['road'].add_part(filename='../models/intersection.stl')

rcs_dbsm = 10
rcs = 10 ** (rcs_dbsm / 10)

# prim2 = Sphere(rcs=rcs*5, wl=3e8/center_freq, num_theta=45, num_phi=45)
# prim = Cube(x_length=1, y_length=1, z_length=1)
# prim = Cylinder(radius=1, height=10, num_theta=45, orientation=[0, 0, 1])
# prim = Capsule(radius=1, height=10, num_theta=45, orientation=[0, 0, 1])
# prim2 = Plane(rcs=rcs, wl=3e8/center_freq, num_i=10, num_j=10, orientation=[0, 1, 0])
prim = CornerReflector(rcs=rcs, wl=3e8/center_freq, orientation='x',
                       is_square=False)  # if use an rcs value, it will override teh length and calculate length from rcs


wl = 3e8 / center_freq
prim_name = all_actors.add_actor(name='prim_example', generator=prim, mat_idx=mat_manager.get_index('pec'),
                                 target_ray_spacing=wl/4,dynamic_generator_updates=False) #dynamic_generator_updates=False will not create a new geometry at each time step

# prim_name2 = all_actors.add_actor(name='prim_example2', generator=prim2, mat_idx=mat_manager.get_index('pec'),
#                                  target_ray_spacing=wl/4,dynamic_generator_updates=False) #dynamic_generator_updates=False will not create a new geometry at each time step
#


num_frames = 45

xpos = np.linspace(range_start,range_stop, num_frames)
ypos = np.zeros(num_frames)
all_actors.actors[prim_name].coord_sys.pos = (xpos[0], ypos[0], 0.)
all_actors.actors[prim_name].coord_sys.lin = (0, 0, 0.)

# all_scene_actors['vehicle2'].velocity_mag = 10.
# The 3x3 rotation matrix is set using a more intuitive input of euler angles.
all_actors.actors[prim_name].coord_sys.rot = look_at((xpos[0], ypos[0], 0.),(0, 0, 0))
all_actors.actors[prim_name].coord_sys.update()

# all_actors.actors[prim_name2].coord_sys.pos = (235/2, 50, 0.)
# # all_actors.actors[prim_name2].coord_sys.rot = look_at((100, 5, 0.),(0, 0, 0))
# all_actors.actors[prim_name2].coord_sys.update()
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
mode_name = 'mode1'

# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)

#################
# Tx antenna

radar_actor_name = all_actors.add_actor() # easier to attach a camera to an actor rather than an antenna
all_actors.actors[radar_actor_name].coord_sys.pos = [0, 0, 0]
all_actors.actors[radar_actor_name].coord_sys.rot = np.eye(3)
all_actors.actors[radar_actor_name].coord_sys.update()
#use dipole antenna for easier radar range calculation
ant_device = add_single_tx_rx(all_actors, waveform, mode_name,ffd_file='dipole.ffd',scale_pattern=.5,parent_h_node=all_actors.actors[radar_actor_name].h_node)
center_freq = ant_device.waveforms[mode_name].center_freq

# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = 2.99792458e8 / center_freq
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


# Below is an example of how to apply a range filter to the radar. This is not required, but can be used to remove
# effects beyond or before an expected range
rngFilter_high = RssPy.RangeFilter()
rngFilter_low = RssPy.RangeFilter()
rangeCutoff = range_start
rngFilter_high.setIdealAbrupt(RssPy.FilterPassType.HIGH_PASS, rangeCutoff)
rangeCutoff = range_stop
rngFilter_low.setIdealAbrupt(RssPy.FilterPassType.LOW_PASS, rangeCutoff)
# rngFilter.isValid()
pem.addRangeFilter(ant_device.modes[mode_name], rngFilter_high)
pem.addRangeFilter(ant_device.modes[mode_name], rngFilter_low)
#
#
ref_pixel_mid = RssPy.ImagePixelReference.MIDDLE
ref_pixel_begin = RssPy.ImagePixelReference.BEGIN
radial_velocity = 0

pem_api_manager.isOK(pem.activateRangeDopplerResponse(ant_device.modes[mode_name],
                                               ant_device.range_pixels,
                                               ant_device.doppler_pixels,
                                               ref_pixel_begin,
                                               range_start,
                                               ref_pixel_mid,
                                               radial_velocity,
                                               ant_device.r_specs,
                                               ant_device.d_specs))




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
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=rng_domain,
                             vel_domain=vel_domain,
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)
# modeler.pl.show_grid()

all_p_received = []
all_distances = []
print('running simulation...')
responses = []
for iFrame in tqdm(range(num_frames), disable=True):



    # update all coordinate systems
    all_actors.actors[prim_name].coord_sys.pos = (xpos[iFrame], ypos[iFrame], 0.)
    # print(f'Corner Reflector at frame {iFrame}')
    # print(f'Position {xpos[iFrame], ypos[iFrame]}')
    # print(f'Velocity {all_actors.actors[prim_name].coord_sys.lin}')
    all_actors.actors[prim_name].coord_sys.rot = look_at((xpos[iFrame], ypos[iFrame], 0.),(0, 0, 0))
    all_actors.actors[prim_name].coord_sys.update()

    start_sim_time = walltime.time()
    pem_api_manager.isOK(pem.computeResponseSync())
    end_sim_time = walltime.time()
    print(f'Simulation Time: {(end_sim_time - start_sim_time)*1e3} mseconds')
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)


    if export_debug:
        debug_logs.write_scene_summary(file_name=f'out_{iFrame}.json')
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file


    Gt = 2.15
    Gr = 2.15
    Pt_dB = 0  # default in Perceive EM
    # Pr = np.max(np.abs(response[0][0]))
    Pr = np.sum(np.abs(response[0][0][1]) ** 2)
    Pr_dB = 10 * np.log10(Pr)
    all_p_received.append(Pr)

    # index where Pr_dB is max
    all_distances.append(rng_domain[np.argmax(np.abs(response[0][0][1]))])
    # Pr_dB = 20 * np.log10(Pr)
    print(f'Pr_dB {Pr_dB}')
    R_dist = np.sqrt(xpos[iFrame] ** 2 + ypos[iFrame] ** 2) # radar is always at origin
    # radar range equation in dB
    # Pr_dB = Pt_dB + Gt + Gr 30 * np.log10(1/4 / np.pi) + 20 * np.log10(3e8))+ rcs_dB40 * np.log10(R_dist)

    # Pr_dB = 30 * np.log10(1 / (4 * np.pi)) + Pt_dB + Gt + Gr + 20 * np.log10(300000000) + rcs_target_db - 20 * np.log10(center_freq) - 40 * np.log10(R_dist)
    rcs_sim_dB = -30 * np.log10(1 / (4 * np.pi)) - Pt_dB - Gt - Gr - 20 * np.log10(300000000) + 20 * np.log10(center_freq) + 40 * np.log10(R_dist)+Pr_dB

    print(f'Analytic RCS {rcs_dbsm}, Simulated {rcs_sim_dB}')
    responses.append(response)
    plot_data = np.rot90(20*np.log10(np.fmax(np.abs(response[0,0]),1e-15)))
    modeler.update_frame(plot_data=plot_data,write_frame=True)

modeler.close()
responses = np.array(responses)
# create a plot of the received power
plt.figure()
plt.plot(20*np.log10(all_p_received))
plt.xlabel('Frame')
plt.ylabel('Received Power')
plt.title('Received Power vs Frame')
plt.grid()
plt.show()


# responses
plot_data = responses[0,0,0,0]

plt.figure()
plt.plot(rng_domain,20*np.log10(np.abs(plot_data)))
plt.xlabel('Range (m)')
plt.ylabel('Received Power')
plt.title('Received Power vs Range')
plt.grid()
plt.show()

# plot all distances
plt.figure()
plt.plot(all_distances,label='Simulated Distance')
plt.plot(xpos,label='Actual Distance')
plt.xlabel('Frame')
plt.ylabel('Distance (m)')
plt.title('Distance vs Frame')
plt.legend()
plt.grid()
plt.show()

# create an animation the first index of responses[:,0,0,0,:]. This will show the range profile of the corner reflector versus position
# in the scene

import matplotlib.animation as animation

# create an animated plot of the range profile
plt.close('all')
fig, ax = plt.subplots()
frame_idx = range(num_frames)

plot_data = responses[0,0,0,0]
rp = 20*np.log10(np.abs(plot_data))

line = ax.plot(rng_domain, rp)[0]
ax.set(xlim=[rng_domain[0], rng_domain[-1]], ylim=[-300, -100], xlabel='Range [m]', ylabel='Prcv [dB]')
# ax.xlabel('Frame')
# ax.ylabel('Distance (m)')
ax.set_title('Range Profile')

def update(frame):
    plot_data = responses[frame, 0, 0, 0]
    rp = 20 * np.log10(np.abs(plot_data))

    # update the line plot:
    line.set_xdata(rng_domain)
    line.set_ydata(rp)
    return line


ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=30)
# save the animation to a file
ani.save('range_profile.gif', writer='imagemagick', fps=6)
plt.show()


