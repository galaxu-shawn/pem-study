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
import pyvista as pv
import sys

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx, add_single_rx, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.post_processing import range_profile
from pem_utilities.open_street_maps_geometry import get_z_elevation_from_mesh
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 
#
# z_offset = 3
# rx_pos = [-1.16791992e+01, 1.76538223e+02, 7.08180014e-06+z_offset]
# z_offset = 2.5
# tx_pos = [-419.89949219, -5.16172302, 134.07287109+z_offset]
rx_pos = [-60.24069336,177.63636719  ,6.10316833]
tx_pos = [-432.09566406,2.1259462 ,151.02806641]
fps = 1
total_time = 1

# waveform to plot
center_freq = 3.6e9
num_freqs = 4096
bandwidth = 122880000
cpi_duration = 1000e-3
num_pulse_CPI = 1

# simulation parameters
go_blockage = 0 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 3
max_num_trans = 0
ray_spacing = .5

export_debug = True #use this to export the movie at the end of the simulation (.gif showing radar camera and response)
save_results = True

# generate material from ITU library
mat_manager = MaterialManager(generate_itu_materials=True, itu_freq_ghz=3.6)
time_stamps = np.linspace(0,total_time,int(total_time*fps))

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors(material_manager=mat_manager)

# usd_file_example = '../models/tokyo.usd'
usd_file_example = os.path.join(paths.models,'tokyo_flat.usd')
if not os.path.exists(usd_file_example):
    raise FileNotFoundError(f'USD file not found: {usd_file_example}, please download usd file')

#                                 target_ray_spacing=0.5,
# geom_name = all_actors.add_actor(filename=usd_file_example,
#                                  scale_mesh=1e-2,
#                                  mat_idx = mat_manager.get_index('pec'),
#                                  target_ray_spacing=0.01) # tokyo.usd is in cm, so scale it to meter

# geom_name = all_actors.add_actor(filename=r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\test2.stl',
#                                  scale_mesh=1,
#                                  mat_idx = mat_manager.get_index('concrete'),
#                                  target_ray_spacing=.02) # tokyo.usd is in cm, so scale it to meter
# for quicker testing, use a simple stl file, comment out the line above and uncomment the line below
# geom_name = all_actors.add_actor(filename='../models/rv.stl')


# get the z elevation of the intersection of the ray with the mesh
# I am commenting out for now, because I know the z elevation of the mesh is 0 in this case.
# z_intersect = get_z_elevation_from_mesh(rx_pos[:2], all_actors.actors[geom_name].get_mesh())
# rx_pos[2] += np.max(z_intersect)


# empty actor where we will place the antennas
actor_tx_name = all_actors.add_actor()
actor_rx_name = all_actors.add_actor()

# all_actors.actors[agi_hq_name].coord_sys.rot = euler_to_rot(phi=0,theta=180,psi=180)
# all_actors.actors[agi_hq_name].coord_sys.update()


# Antenna Device
mode_name = 'mode1' # name of mode so we can reference it in post processing
input_power_dbm = 43 # dBm
# # convert to watts
input_power_watts = 10 ** ((input_power_dbm - 30) / 10)
# input_power_watts =1
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
    "tx_multiplex": "SIMULTANEOUS",
    "mode_delay": "FIRST_CHIRP",
    "tx_incident_power": input_power_watts}

pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)


######################
# Add 2 antenna to teh scene, one tx and one rx
####################


# add a single Tx antenna to the scene at tx_pos
ant_device_tx = add_single_tx(all_actors,waveform,mode_name,
                              pos=[0,0,0],
                              ffd_file='dipole.ffd',
                              scale_pattern=20,
                              parent_h_node=all_actors.actors[actor_tx_name].h_node)

######################
# Rx on walking user
####################

ant_device_rx = add_single_rx(all_actors,waveform,mode_name,
                              pos=[0,0,0],
                              ffd_file='dipole.ffd',
                              scale_pattern=10,
                              parent_h_node=all_actors.actors[actor_rx_name].h_node)

enable_coupling(mode_name,ant_device_tx, ant_device_rx)

# rays can be defined as density or spacing, this is the conversion between the two, if ray_density is set, then

freq_center = ant_device_tx.waveforms[mode_name].center_freq
lambda_center = 2.99792458e8 / freq_center
# ray_spacing = np.sqrt(2) * lambda_center / ray_density
# pem_api_manager.isOK(pem.setPrivateKey("RayShootGrid", "SBR," + str(1.0)))
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
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain*2 # multiply by 2 to get total distance, rng_domain is roundtrip
freq_domain = ant_device_tx.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx.waveforms[mode_name].pulse_domain
print(f'Maximum range: {rng_domain[-1]}m')
print(f'range resolution: {rng_domain[1]-rng_domain[0]}m')


# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

dt = 1/fps
num_frames = int(total_time/dt)
responses = []

# max sure range domain is the same length as num_freqs because we will not be up/down sampling the range profile
range_domain_new = np.linspace(rng_domain[0],rng_domain[-1],num_freqs)
#                              figure_size = (1.0, 1.0),
# # create a pyvista plotter
modeler = ModelVisualization(all_actors,
                                 show_antennas=True,
                                 x_domain=range_domain_new,
                                 fps=fps,
                                 output_video_size = (1280,768),
                                 figure_size = (0.5, 0.5),
                                 camera_attachment=None,
                                 camera_orientation=None)
modeler.pl.show_grid()
# modeler.pl.add_mesh(pv.Sphere(radius=514.8, center=tx_pos), color='red',opacity=0.5)
points = np.array([[-43209.56640625, 212.59461975097656, 15102.806640625], [-10960.1826171875, 24924.845703125, 3166.357177734375], [-6024.0693359375, 17763.63671875, 610.3168334960938]])
actor = modeler.pl.add_lines(
    points*1e-2, color='purple', width=3, connected=True)

all_range_profiles = []
print('running simulation...')
timestamps = np.linspace(0,total_time,num_frames)
for frame_idx in tqdm(range(num_frames)):
    time = frame_idx*dt
    # update all coordinate systems
    all_actors.actors[actor_rx_name].coord_sys.pos = rx_pos
    all_actors.actors[actor_tx_name].coord_sys.pos = tx_pos
    all_actors.actors[actor_tx_name].coord_sys.update()
    all_actors.actors[actor_rx_name].coord_sys.update()

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes[mode_name],ant_device_rx.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)

    if export_debug:
        if frame_idx == 0:
            debug_logs.write_scene_summary(file_name=f'out_{frame_idx}_from_pem.json')
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
    responses.append(response)

    # free space path loss using Friss formula
    pos1 = all_actors.actors[actor_tx_name].coord_sys.pos
    pos2 = all_actors.actors[actor_rx_name].coord_sys.pos
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))

    # Approximate antenna gain for tx/rx is 2.15 (dipole)
    free_space_loss = 20 * np.log10(distance) + 20 * np.log10(freq_center) - 147.55 - 2.15 * 2
    input_power_watts_db = 10*np.log10(input_power_watts)

    plot_data_freq = 20*np.log10(np.abs(response[0,0,0])) - input_power_watts_db
    plot_data_td = 20*np.log10(np.abs(range_profile(response[0,0], window= True, size=num_freqs)))- input_power_watts_db
    range_value = range_domain_new[np.argmax(plot_data_td)]
    all_range_profiles.append(plot_data_td)
    print(f'\nPhysical Distance = {distance}')
    print(f'Range value at peak of range profile: {range_value}m')

    print(f'Friis (dB): {free_space_loss}')
    print(f'Perceive EM (peak from range profile, dB): {np.abs(np.max(plot_data_td))}')
    print(f'Perceive EM (mean freq response, dB): {np.abs(20 * np.log10(np.abs(response[0, 0, 0])).mean() - input_power_watts_db)}')
    # swap out plot_data_freq or plot_data_td to plot different data
    modeler.h_chart.background_color = (1.0, 1.0, 1.0)
    modeler.update_frame(plot_data=plot_data_freq,write_frame=False) # write frame false will not write the frame to disk, and just pause the animation
modeler.close()

# post-process images into gif

if export_debug:
    # debug_camera.write_camera_to_gif(file_name='camera.gif')
    # debug_camera.write_response_to_gif(responses,file_name='response.gif',rng_domain=rng_domain,vel_domain=vel_domain)
    debug_camera.write_camera_and_response_to_gif(responses,
                                                  file_name='camera_and_response.gif',
                                                  rng_domain=rng_domain,
                                                  vel_domain=vel_domain)

responses = np.array(responses)
all_range_profiles = np.array(all_range_profiles)
plt.close('all')


if save_results:
    # results are generated with the dimensions [time_idx, num_tx, num_rx, num_pulses, num_freqs]
    # where each time_idx is when we call the simulator. For each simulation we get N (num_pulses) channel soundings
    # as an output. The pulse_domain will tell us the time spacing of each channel sounding,
    # if we wanted a continuous time domain output, we could set the spacing of timestamps to be the same as the CPI
    # (or total time shown in pulse_domain)

    np.save(os.path.join(paths.output, 'all_results_usd.npy'), responses)
    np.save(os.path.join(paths.output, 'range_profiles_usd.npy'), all_range_profiles)
    np.save(os.path.join(paths.output, 'freq_domain.npy'), freq_domain) # what frequency samples were used
    np.save(os.path.join(paths.output, 'range_domain.npy'), rng_domain) # time domain of the pulses
    np.save(os.path.join(paths.output, 'simulation_timestamps.npy'), timestamps) # velocity domain
    np.save(os.path.join(paths.output, 'incident_power_dbm.npy'), input_power_dbm)



from_aodt_solver = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\cfrs_new.txt'
# read this file line by line
with open(from_aodt_solver, 'r') as f:
    lines = f.readlines()
    all_results_aodt = []
    for line in lines:
        asdf = line.replace('(',"").replace(')',"").replace("\n","")

        all_results_aodt.append(complex(asdf))
all_results_aodt = np.array(all_results_aodt,dtype=complex)



# Plot the CIR for a single channel sounding (using pulse index = 0)
len_fft = num_freqs
results_td_db_for_plot_pem_usd= 20*np.log10(np.abs(range_profile(responses[0,0,0,0], window= True, size=len_fft)))#- input_power_watts_db
results_td_db_for_plot_aodt= 20*np.log10(np.abs(range_profile(all_results_aodt, window= True, size=len_fft)))#- input_power_watts_db
# results_td_db_for_plot_aodt = 20*np.log10(np.abs(np.fft.ifft(all_results_aodt,norm = 'ortho')))

rng_domain_aodt = np.linspace(0,9993,4096)
fig, ax = plt.subplots()
ax.plot(rng_domain, results_td_db_for_plot_pem_usd,label='Perceive EM - USD')
ax.plot(rng_domain_aodt,results_td_db_for_plot_aodt,label='AODT')
ax.set(xlabel='Range (m)', ylabel='Results (dB20)',title='CIR for a PEM from USD and from Proto')
ax.set_xlim([400, 700])
plt.legend()
ax.grid()
plt.show()
