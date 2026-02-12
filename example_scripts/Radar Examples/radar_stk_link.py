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
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, rot_to_euler
from pem_utilities.antenna_device import Waveform, AntennaArray
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.stk_utils import STK_Utils, STK_Results_Reader
from pem_utilities.post_processing import create_target_list

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy

fps = 10
total_time = 80

# waveform to plot
center_freq = 10e9
num_freqs = 512
bandwidth = 200e6
cpi_duration = 150e-3
num_pulse_CPI = 201

rng_pixels = 512
doppler_pixels = 256

num_rx_az = 8
num_rx_el = 4

# simulation parameters
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 3
max_num_trans = 1
ray_density = .02

export_debug = True #use this to export the movie at the end of the simulation (.gif showing radar camera and response)

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

time_stamps = np.linspace(0,total_time,int(total_time*fps))

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()


actor_uas1_name = all_actors.add_actor(filename =os.path.join(paths.models,'drone1/drone1.json'),scale_mesh=2,target_ray_spacing=.02)
actor_uas2_name = all_actors.add_actor(filename =os.path.join(paths.models,'drone1/drone1.json'),scale_mesh=2,target_ray_spacing=.02)
actor_uas3_name = all_actors.add_actor(filename =os.path.join(paths.models,'drone1/drone1.json'),scale_mesh=2,target_ray_spacing=.02)

actor_quad1_name = all_actors.add_actor(filename =os.path.join(paths.models,'Quadcopter/Quadcopter.json'),scale_mesh=5,target_ray_spacing=.02)
actor_quad2_name = all_actors.add_actor(filename =os.path.join(paths.models,'Quadcopter/Quadcopter.json'),scale_mesh=5,target_ray_spacing=.02)
actor_quad3_name = all_actors.add_actor(filename =os.path.join(paths.models,'Quadcopter/Quadcopter.json'),scale_mesh=5,target_ray_spacing=.02)
actor_quad4_name = all_actors.add_actor(filename =os.path.join(paths.models,'Quadcopter/Quadcopter.json'),scale_mesh=5,target_ray_spacing=.02)

actor_radar_name = all_actors.add_actor()

agi_hq_name = all_actors.add_actor(filename =os.path.join(paths.models,'agi_hq/agi_hq.json'))
all_actors.actors[agi_hq_name].coord_sys.rot = euler_to_rot(phi=0,theta=0,psi=0)
all_actors.actors[agi_hq_name].coord_sys.update()

# IDENTIFY THE OBJECTS IN THE STK SCENARIO
#
# - We first define the Global Coordinate System
#
global_object_path = 'Facility/Global_CS'     # Global Coordinate STK Object
global_cs = "Global_CS"                       # Global Coordinate Axes from AWB
#
#   We then assign PY script variables for the intended STK Objects defined below
#
#   THE OBSERVER NAME
#
stk_scaling=1e3
stk_radar = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)

stk_quad1 = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)

stk_quad2 = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)

stk_quad3 = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)


stk_quad4 = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)



stk_uas1 = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)


stk_uas2 = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)


stk_uas3 = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)


# IDENTIFY THE CORRESPONDING STK OBJECT NAME AND AXES
#
#        This is the STK OBSERVER Platform and Coordinate System

local_object_path = 'Aircraft/QUAD_001_Main_Body'
local_cs = 'Body'
stk_quad1.getDataFromSTK(local_object_path, local_cs)

local_object_path = 'Aircraft/QUAD_002_Main_Body'
local_cs = 'Body'
stk_quad2.getDataFromSTK(local_object_path, local_cs)

local_object_path = 'Aircraft/QUAD_003_Main_Body'
local_cs = 'Body'
stk_quad3.getDataFromSTK(local_object_path, local_cs)

local_object_path = 'Aircraft/QUAD_004_Main_Body'
local_cs = 'Body'
stk_quad4.getDataFromSTK(local_object_path, local_cs)

local_object_path = 'Aircraft/UAS_001'
local_cs = 'Body'
stk_uas1.getDataFromSTK(local_object_path, local_cs)

local_object_path = 'Aircraft/UAS_002'
local_cs = 'Body'
stk_uas2.getDataFromSTK(local_object_path, local_cs)

local_object_path = 'Aircraft/UAS_003'
local_cs = 'Body'
stk_uas3.getDataFromSTK(local_object_path, local_cs)


local_object_path_radar = 'Facility/MTI_Sensor/Sensor/MTI_TX_Center'
local_cs_radar = 'Sensor_Reference_Axis'
stk_radar.getDataFromSTK(local_object_path_radar, local_cs_radar)


# Antenna Device

mode_name = 'mode1' # name of mode so we can reference it in post processing
# input_power_dbm = 10.0 # dBm
# # convert to watts
# input_power_watts = 10 ** ((input_power_dbm - 30) / 10)
input_power_watts = 1
# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more
# flexibility and control over the parameters, without having to create/modify a json file. This is the same as the
# results as if the same parameters had been created in a json file.
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "RangeDoppler",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_pulse_CPI,
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "CENTER_CHIRP",
    "tx_incident_power": input_power_watts}

pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)


######################
# Add 2 antenna to teh scene, one tx and one rx
####################


# ant_device_tx1 = add_single_tx(all_actors,waveform,mode_name,pos=ant_1_pos,ffd_file='dipole.ffd',scale_pattern=20.0)
ant_array_az = AntennaArray(all_actors=all_actors,
                            name = 'array',
                            waveform = waveform,
                            mode_name = mode_name,
                            file_name = None,
                            beamwidth_H = 140,
                            beamwidth_V = 120,
                            polarization = 'H',
                            rx_shape = [1, num_rx_az],
                            tx_shape = [1, 1],
                            spacing_wl_x = 0.5,
                            spacing_wl_y = 0.5,
                            load_pattern_as_mesh = True,
                            scale_pattern = 10,
                            parent_h_node = all_actors.actors[actor_radar_name].h_node,
                            normal = 'x',
                            range_pixels=rng_pixels,
                            doppler_pixels=doppler_pixels)

ant_device_az = ant_array_az.antenna_device

ant_array_el = AntennaArray(all_actors=all_actors,
                            name = 'array',
                            waveform = waveform,
                            mode_name = mode_name,
                            file_name = None,
                            beamwidth_H = 140,
                            beamwidth_V = 120,
                            polarization = 'H',
                            rx_shape = [num_rx_el, 1],
                            tx_shape = [1, 1],
                            spacing_wl_x = 0.5,
                            spacing_wl_y = 0.5,
                            load_pattern_as_mesh = True,
                            scale_pattern = 10,
                            parent_h_node = all_actors.actors[actor_radar_name].h_node,
                            normal = 'x',
                            range_pixels=rng_pixels,
                            doppler_pixels=doppler_pixels)

ant_device_el = ant_array_el.antenna_device



# rays can be defined as density or spacing, this is the conversion between the two, if ray_density is set, then
freq_center = ant_device_az.waveforms[mode_name].center_freq
lambda_center = 2.99792458e8 / freq_center
ray_spacing = np.sqrt(2) * lambda_center / ray_density


rngFilter_high = RssPy.RangeFilter()
rngFilter_low = RssPy.RangeFilter()
rangeCutoff = 20
rngFilter_high.setIdealAbrupt(RssPy.FilterPassType.HIGH_PASS, rangeCutoff)
rangeCutoff = 380
rngFilter_low.setIdealAbrupt(RssPy.FilterPassType.LOW_PASS, rangeCutoff)
# rngFilter.isValid()
pem.addRangeFilter(ant_device_az.modes[mode_name], rngFilter_high)
pem.addRangeFilter(ant_device_az.modes[mode_name], rngFilter_low)
pem.addRangeFilter(ant_device_el.modes[mode_name], rngFilter_high)
pem.addRangeFilter(ant_device_el.modes[mode_name], rngFilter_low)

print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
sim_options.auto_configure_simulation()

which_mode = ant_device_az.modes[mode_name] # tell it which mode we want to get respones from
ant_device_az.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device_az.waveforms[mode_name].vel_domain
rng_domain = ant_device_az.waveforms[mode_name].rng_domain
freq_domain = ant_device_az.waveforms[mode_name].freq_domain
pulse_domain = ant_device_az.waveforms[mode_name].pulse_domain

# maximum range and velocity for the radar
max_range = rng_domain[-1]
velocity_amg = vel_domain[-1]-vel_domain[0]
print(f'max range: {max_range} m')
print(f'max velocity: {velocity_amg} m/s')

# output file will be stored in this directory
output_path = paths.output
os.makedirs(output_path, exist_ok=True)

stk_radar.save_values_as_numpy(os.path.join(output_path, 'stk_radar'))
stk_quad1.save_values_as_numpy(os.path.join(output_path, 'stk_quad1'))
stk_quad2.save_values_as_numpy(os.path.join(output_path, 'stk_quad2'))
stk_quad3.save_values_as_numpy(os.path.join(output_path, 'stk_quad3'))
stk_quad4.save_values_as_numpy(os.path.join(output_path, 'stk_quad4'))
stk_uas1.save_values_as_numpy(os.path.join(output_path, 'stk_uas1'))
stk_uas2.save_values_as_numpy(os.path.join(output_path, 'stk_uas2'))
stk_uas3.save_values_as_numpy(os.path.join(output_path, 'stk_uas3'))


# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_logs = DebuggingLogs(output_directory=output_path)
    debug_camera = DebuggingCamera(hMode=ant_device_az.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

dt = 1/fps
num_frames = int(total_time/dt)

responses = []
output_movie_name = os.path.join(output_path, 'out_vis_quad2.mp4')
# # create a pyvista plotter
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             overlay_results=False,
                             rng_domain=rng_domain,vel_domain=vel_domain,
                             fps=fps,
                             camera_attachment=None,
                             camera_orientation=None,
                             output_movie_name=output_movie_name
                             )

print('running simulation...')
all_target_all_time = {}
for frame_idx in tqdm(range(num_frames)):
    time = frame_idx*dt
    # update all coordinate systems


    all_actors.actors[actor_radar_name].coord_sys.pos = stk_radar.pos[frame_idx]
    all_actors.actors[actor_radar_name].coord_sys.rot = stk_radar.rot[frame_idx]
    all_actors.actors[actor_radar_name].coord_sys.lin = stk_radar.lin[frame_idx]
    all_actors.actors[actor_radar_name].coord_sys.update()

    all_actors.actors[actor_quad1_name].coord_sys.pos = stk_quad1.pos[frame_idx]
    eul = rot_to_euler(stk_quad1.rot[frame_idx], order='zyz', deg=True) # stk body axies is Zpointing down
    rot = euler_to_rot(phi=eul[0],theta=eul[1]+180,psi=eul[2]+180)
    all_actors.actors[actor_quad1_name].coord_sys.rot = rot
    all_actors.actors[actor_quad1_name].coord_sys.lin = stk_quad1.lin[frame_idx]

    all_actors.actors[actor_quad1_name].coord_sys.update()

    all_actors.actors[actor_quad2_name].coord_sys.pos = stk_quad2.pos[frame_idx]
    eul = rot_to_euler(stk_quad2.rot[frame_idx], order='zyz', deg=True)
    rot = euler_to_rot(phi=eul[0],theta=eul[1]+180,psi=eul[2]+180)
    all_actors.actors[actor_quad2_name].coord_sys.rot = rot
    all_actors.actors[actor_quad2_name].coord_sys.lin = stk_quad2.lin[frame_idx]
    all_actors.actors[actor_quad2_name].coord_sys.update()

    all_actors.actors[actor_quad3_name].coord_sys.pos = stk_quad3.pos[frame_idx]
    eul = rot_to_euler(stk_quad3.rot[frame_idx], order='zyz', deg=True)
    rot = euler_to_rot(phi=eul[0],theta=eul[1]+180,psi=eul[2]+180)
    all_actors.actors[actor_quad3_name].coord_sys.rot = rot
    all_actors.actors[actor_quad3_name].coord_sys.lin = stk_quad3.lin[frame_idx]
    all_actors.actors[actor_quad3_name].coord_sys.update()

    all_actors.actors[actor_quad4_name].coord_sys.pos = stk_quad4.pos[frame_idx]
    eul = rot_to_euler(stk_quad4.rot[frame_idx], order='zyz', deg=True)
    rot = euler_to_rot(phi=eul[0],theta=eul[1]+180,psi=eul[2]+180)
    all_actors.actors[actor_quad4_name].coord_sys.rot = rot
    all_actors.actors[actor_quad4_name].coord_sys.lin = stk_quad4.lin[frame_idx]
    all_actors.actors[actor_quad4_name].coord_sys.update()

    all_actors.actors[actor_uas1_name].coord_sys.pos = stk_uas1.pos[frame_idx]
    eul = rot_to_euler(stk_uas1.rot[frame_idx], order='zyz', deg=True)
    rot = euler_to_rot(phi=eul[0],theta=eul[1]+180,psi=eul[2]+180)
    all_actors.actors[actor_uas1_name].coord_sys.rot = rot
    all_actors.actors[actor_uas1_name].coord_sys.lin = stk_uas1.lin[frame_idx]
    all_actors.actors[actor_uas1_name].coord_sys.update()

    all_actors.actors[actor_uas2_name].coord_sys.pos = stk_uas2.pos[frame_idx]
    eul = rot_to_euler(stk_uas2.rot[frame_idx], order='zyz', deg=True)
    rot = euler_to_rot(phi=eul[0],theta=eul[1]+180,psi=eul[2]+180)
    all_actors.actors[actor_uas2_name].coord_sys.rot = rot
    all_actors.actors[actor_uas2_name].coord_sys.lin = stk_uas2.lin[frame_idx]
    all_actors.actors[actor_uas2_name].coord_sys.update()

    all_actors.actors[actor_uas3_name].coord_sys.pos = stk_uas3.pos[frame_idx]
    eul = rot_to_euler(stk_uas3.rot[frame_idx], order='zyz', deg=True)
    rot = euler_to_rot(phi=eul[0],theta=eul[1]+180,psi=eul[2]+180)
    all_actors.actors[actor_uas3_name].coord_sys.rot = rot
    all_actors.actors[actor_uas3_name].coord_sys.lin = stk_uas3.lin[frame_idx]
    all_actors.actors[actor_uas3_name].coord_sys.update()

    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)


    # write out scene summary, only writing out first and last frame, useful for debugging

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response_az) = pem.retrieveResponse(ant_device_az.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)
    (ret, response_el) = pem.retrieveResponse(ant_device_el.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)


    # elevation field of view is minus angles towards sky (becuase antenna in STK is -Z pointing up)
    target_list, fps_create_target = create_target_list(rd_all_channels_az=response_az[0],
                                                        rd_all_channels_el=response_el[0],
                                                        rngDomain=rng_domain,
                                                        velDomain=vel_domain,
                                                        azPixels=512, elPixels=512,
                                                        antenna_spacing_wl=0.5,
                                                        radar_fov_az=[-90, 90],
                                                        radar_fov_el=[-30, -2],
                                                        centerFreq=center_freq,
                                                        rcs_min_detect=-100,
                                                        min_detect_range=50,
                                                        rel_peak_threshold=1e-3,
                                                        max_detections=200,
                                                        exclude_velocities=[-1,1],
                                                        return_cfar=False)
    all_target_all_time[time] = target_list
    # velocity_min = -10 and max is just for visualization purposes, to scale color of the point cloud
    modeler.add_point_cloud_to_scene(target_list,
                                     tx_pos=all_actors.actors[actor_radar_name].coord_sys.pos,
                                     tx_rot=all_actors.actors[actor_radar_name].coord_sys.rot,
                                     color_min=-250,
                                     color_max=-120,
                                     color_mode='p_received',
                                     size_mode='p_received',
                                     max_radius=.5,)
    #
    # if frame_idx==0:
    #     all_scalar_bars = list(modeler.pl.scalar_bars.keys())
    #     for each in all_scalar_bars:
    #         modeler.pl.remove_scalar_bar(each)
    # x = []
    # y = []
    # all_mag = []
    # max_size = 2
    # for target in target_list:
    #     x.append(target_list[target]['xpos'])
    #     y.append(target_list[target]['ypos'])
    #     p_rec = target_list[target]['p_received']
    #     p_rec_db = 10 * np.log10(p_rec)
    #     all_mag.append(p_rec_db)
    # p_rec_db = np.array(all_mag)
    # p_rec_db_norm = (p_rec_db - np.min(p_rec_db)) / (np.max(p_rec_db) - np.min(p_rec_db))
    #
    # p_rec_db_norm = -100 / p_rec_db * 10  # set a baseline size for all points relative to the numerator
    #
    # if time == 0:
    #     f, ax = plt.subplots(tight_layout=True)
    #     scat = ax.scatter(x, y, s=p_rec_db_norm * 7, c=p_rec_db, vmin=-100, vmax=-40, cmap='jet')
    #     ax.set_xlim(-600, 600)
    #     ax.set_ylim(-600, 600)
    #     h_chart = pv.ChartMPL(f, size=(0.275, 0.375), loc=(0.7, 0.6))
    #     h_chart.background_color = (0.5, 0.5, 0.5)
    #     modeler.pl.add_chart(h_chart)
    # else:
    #     scat.set_offsets(np.c_[x, y])
    #     scat.set_sizes(p_rec_db_norm * 7)
    #     scat.set_array(p_rec_db)

    #
    # # calculate response in dB to overlay in pyvista plotter
    im_data = np.rot90(20 * np.log10(np.fmax(np.abs(response_az[0][0]), 1.e-30)))
    if export_debug:
        debug_logs.write_scene_summary(file_name=f'out.json')
        debug_camera.generate_image()
    responses.append(response_az)
    modeler.update_frame()
modeler.close()

# post-process images into gif
import json
# save all_target_all_time to a json file, pretty print it
with open(os.path.join(output_path, 'all_target_all_time.json'), 'w') as f:
    json.dump(all_target_all_time, f, indent=4)



if export_debug:
    # debug_camera.write_camera_to_gif(file_name='camera.gif')
    # debug_camera.write_response_to_gif(responses,file_name='response.gif',rng_domain=rng_domain,vel_domain=vel_domain)
    debug_camera.write_camera_and_response_to_gif(responses,
                                                  file_name='camera_and_response.gif',
                                                  rng_domain=rng_domain,
                                                  vel_domain=vel_domain)

responses = np.array(responses)
# plt.close('all')




