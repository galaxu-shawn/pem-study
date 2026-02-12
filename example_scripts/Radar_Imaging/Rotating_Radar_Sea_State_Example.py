# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

This uses the vehicles velocities to create the image

@author: asligar
"""

from tqdm import tqdm
import numpy as np
import os
import sys
import scipy
import matplotlib.pyplot as plt

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

from pem_utilities.materials import MaterialManager
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, vec_to_rot, look_at
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.seastate import OceanSurface
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.domain_transforms import DomainTransforms

#######################################################################################################################
# Input Parameters
#######################################################################################################################

remove_sea = False
time_stamps = np.linspace(0, 1, 201)
initial_radar_pos = [-14, 0, 27]  # radar position in meters, Moving x position so that the radar is not at the origin, and is looking towards the [0,0,0] where the sea geometry is centered
initial_radar_lin = [0, 0, 0]  # radar linear velocity in m/s
intial_radar_rot = euler_to_rot(phi=0, theta=-20, psi=0,order='zyz')  # radar rotation in deg
scan_rate = 60 #rpm
initial_radar_ang = [0, 0, scan_rate*2*np.pi/60]  # radar angular velocity in rad/s

antenna_beamwidth_azimuth = 10  # radar antenna beamwidth in degrees
antenna_beamwidth_elevation = 25  # radar antenna beamwidth in degrees
polarization = 'HH'
# simulation parameters
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 3
max_num_trans = 1
ray_spacing = 0.25  # global ray spacing in meters

center_freq = 10e9

desired_max_range = 700
resolution = 0.5
num_range_bins = int(desired_max_range/resolution) # number of samples in the range domain

range_domain = np.linspace(0, resolution * (num_range_bins - 1), num=num_range_bins)
dt = DomainTransforms(range_domain=range_domain, center_freq=center_freq)

bandwidth = dt.bandwidth
num_freqs = dt.num_freq

# radar parameters
 
prf = 10e3  # pulse repetition frequency in Hz
num_pulse_CPI  = 3
cpi_duration = 1  # CPI duration in seconds


dt = time_stamps[1] - time_stamps[0]  # time step in seconds
export_debug = True

#######################################################################################################################
# End Input Parameters
#######################################################################################################################

# output file will be stored in this directory

mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()

# Set up the ocean parameters
num_grid = 500  # grid size
scene_length = 1000  # domain size
wind_speed = 20
wave_amplitude = 3  # wave amplitude
choppiness = .05  # choppiness factor


if not remove_sea:
    ocean = OceanSurface(num_grid, scene_length, wind_speed, wave_amplitude, choppiness,include_wake=False)
    ocean_name = all_actors.add_actor(name='ocean',generator=ocean,mat_idx=mat_manager.get_index('seawater'))

ship_model_path = os.path.join(paths.models,'Ships')
geo_filename = 'explorer_ship_meter.stl'
ship_name = all_actors.add_actor(name='ship',filename=os.path.join(ship_model_path,geo_filename),color=(0.5,0.5,0.5),mat_idx=0)

all_actors.actors[ship_name].coord_sys.pos = [0,0,0]
all_actors.actors[ship_name].coord_sys.update()


ship_name2 = all_actors.add_actor(name='ship2',filename=os.path.join(paths.models,geo_filename),color=(0.5,0.5,0.5),mat_idx=0)

all_actors.actors[ship_name2].coord_sys.pos = [200,100,0]
all_actors.actors[ship_name2].coord_sys.update()

# empty actor
actor_radar_name = all_actors.add_actor(parent_h_node=all_actors.actors[ship_name].h_node,
                                         name='radar')
all_actors.actors[actor_radar_name].coord_sys.pos = initial_radar_pos
all_actors.actors[actor_radar_name].coord_sys.lin = initial_radar_lin
all_actors.actors[actor_radar_name].coord_sys.ang = initial_radar_ang
all_actors.actors[actor_radar_name].coord_sys.update()


# geo_filename = 'Sphere_1meter_rad.stl'
# sphere_name = all_actors.add_actor(name='sphere',filename=os.path.join(paths.models,geo_filename),color=(0.5,0.5,0.5),target_ray_spacing=0.1,mat_idx=0)

# all_actors.actors[sphere_name].coord_sys.pos = [250,0,0]
# all_actors.actors[sphere_name].coord_sys.update()


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
wavelength = 3e8 / center_freq
mode_name = 'mode1'

waveform = Waveform(waveform_dict)

########################## Setup Radar Platform  ##########################

ant_device = add_single_tx_rx(all_actors,
                              waveform,
                              mode_name,
                              parent_h_node=all_actors.actors[actor_radar_name].h_node,
                              beamwidth_H=antenna_beamwidth_azimuth,
                              beamwidth_V=antenna_beamwidth_elevation,
                              polarization='HH',
                              range_pixels=512,
                              doppler_pixels=32,
                              scale_pattern=100)

# assign modes to gpu devices
print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 180
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(pem.getLastWarnings())

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

if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)
    debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
                                   display_mode='coatings',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=10)


_, _,pos_tx,_,_ = pem.coordSysInGlobal(ant_device.coord_sys.h_node)
scene_center = np.array([0,0,0])
vector_to_target = np.array(pos_tx) - scene_center
dist_to_center = np.sqrt(vector_to_target[0]**2 + vector_to_target[1]**2 + vector_to_target[2]**2)
print(f'Distance to center: {dist_to_center} m')

# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
# camera_orientation = 'follow7' and camera_attachment = ship_name will follow the ship
output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name,
                             figure_size=(.4, .4),
                             shape=(len(rng_domain), len(vel_domain)),
                             cmap='Greys_r')  #shape is rotated from actual output shape

modeler.pl.show_grid()
# modeler.pl.background_color = "black"
# percentage of total orbit, converted to time (based on CPI update rate)
# simulation_end_time = (num_of_looks * simulation_update_every_n_deg)/360*end_time
# update_times = np.linspace(0, simulation_end_time, num_of_looks)
all_max = []
print('running simulation...')

# create a matplotlib polar plot for the radar response. This rotating radar will be centered at the origin
# polar plot will be created with the range domain on the radial axis and the samples on the angular axis
# The total number of samples will for all angles (360deg) will be dt*scan_rate/60seconds 
scan_rate_rps = scan_rate/ 60 
delta_ang_deg = 360/ (scan_rate_rps/ dt)
num_angles = int(360 / delta_ang_deg)
angles = np.linspace(0, 360, num_angles, endpoint=False)

# Initialize the rotating radar data array outside the loop
rotating_radar_data = np.zeros((num_angles, len(rng_domain)))

# Create the polar plot once outside the loop
fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
ax_polar.set_theta_zero_location('N')  # set the zero location to the top
ax_polar.set_theta_direction(-1)  # set the direction to clockwise
ax_polar.set_rlabel_position(90)  # set the radial label position to the top

# Create initial pcolormesh (will be updated in the loop)
theta_mesh, r_mesh = np.meshgrid(np.deg2rad(angles), rng_domain)
# Set initial color limits
vmin, vmax = -180, -100
c = ax_polar.pcolormesh(theta_mesh, r_mesh, rotating_radar_data.T, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
cbar = plt.colorbar(c, ax=ax_polar, label='Response (dB)')
# Set radial limits to show reasonable range (adjust as needed)
ax_polar.set_ylim(0, np.max(rng_domain))

plt.ion()  # Turn on interactive mode for updating

for idx, time in enumerate(time_stamps):
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)

    # response is stored as [tx_idx,rx_idx,vel_idx,range_idx]
    range_doppler = np.fmax(np.abs(response[0][0]), 1.e-30)
    center_pulse = response.shape[2] // 2  # center pulse index
    range_profile = np.fmax(np.abs(response[0][0][center_pulse]), 1.e-30)
    sar_image = 20 * np.log10(range_doppler)
    range_profile_db = 20 * np.log10(range_profile)
    data_max = np.max(sar_image)

    if export_debug:
        debug_camera.generate_image()
        debug_logs.write_scene_summary(file_name=f'out_seastate.json')
        # we can put the debug_camera image into the modeler for debugging
        # modeler.mpl_ax_handle.set_data(debug_camera.current_image)

    modeler.update_frame(plot_data=sar_image.T, plot_limits=[data_max - 120, data_max])  # update visualization

    # Update the rotating radar data array
    idx_loop = idx % num_angles
    rotating_radar_data[idx_loop, :] = range_profile_db
    
    # Update the polar plot
    ax_polar.clear()
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rlabel_position(90)
    ax_polar.set_title(f'Rotating Radar Response at {time:.2f} s', va='bottom')
    
    # Only plot data where we have collected it
    valid_data = rotating_radar_data.copy()
    valid_data[valid_data == 0] = np.nan  # Set zeros to NaN so they don't appear in the plot
    
    # Use vmin/vmax to control the color scale limits
    c = ax_polar.pcolormesh(theta_mesh, r_mesh, valid_data.T, shading='nearest', cmap='jet', vmin=-180, vmax=-100)
    print(f' plot max: {np.max(range_profile_db)}')
    print(f' plot min: {np.min(range_profile_db)}')
    # Set radial limits for distance, not color scale
    ax_polar.set_ylim(0, np.max(rng_domain))
    
    plt.pause(0.01)  # Small pause to allow plot to update

# Save the final polar plot to file
polar_plot_filename = os.path.join(paths.output, 'rotating_radar_polar_plot.png')
plt.savefig(polar_plot_filename, dpi=300, bbox_inches='tight')
print(f'Polar plot saved to: {polar_plot_filename}')

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final plot open

modeler.close()

if export_debug:
    # export the debug camera images to a video file
    debug_camera.export_video(output_file=os.path.join(paths.output, 'debug_camera.mp4'))
