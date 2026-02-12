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
import pyvista as pv

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx, add_single_rx, enable_coupling, AntennaArray
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.post_processing import range_profile, create_target_list
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
num_pulse_CPI = 2

# simulation parameters
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 3
max_num_trans = 3
ray_spacing = .5

export_debug = True #use this to export the movie at the end of the simulation (.gif showing radar camera and response)


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
geom_name = all_actors.add_actor(filename=usd_file_example,
                                 scale_mesh=1e-2,
                                 mat_idx = mat_manager.get_index('concrete')) # tokyo.usd is in cm, so scale it to meter
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
    "output": "RangeDoppler",
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
range_pixels = 512
doppler_pixels = 256


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
                            polarization = 'V',
                            rx_shape = [0, 0],
                            tx_shape = [1, 8],
                            spacing_wl_x = 0.5,
                            spacing_wl_y = 0.5,
                            load_pattern_as_mesh = True,
                            scale_pattern = 10,
                            parent_h_node = None,
                            normal = 'x',
                            range_pixels=range_pixels,
                            doppler_pixels=doppler_pixels)

ant_device_tx_az = ant_array_az.antenna_device
ant_device_tx_az.coord_sys.pos = np.array(tx_pos)
# ant_device_tx_az.coord_sys.rot = euler_to_rot(phi=135,theta=-15)
ant_device_tx_az.coord_sys.update()

ant_array_el = AntennaArray(all_actors=all_actors,
                            name = 'array',
                            waveform = waveform,
                            mode_name = mode_name,
                            file_name = None,
                            beamwidth_H = 140,
                            beamwidth_V = 120,
                            polarization = 'V',
                            rx_shape = [0, 0],
                            tx_shape = [8, 1],
                            spacing_wl_x = 0.5,
                            spacing_wl_y = 0.5,
                            load_pattern_as_mesh = True,
                            scale_pattern = 10,
                            parent_h_node = None,
                            normal = 'x',
                            range_pixels=range_pixels,
                            doppler_pixels=doppler_pixels)

ant_device_tx_el = ant_array_el.antenna_device


ant_device_tx_el.coord_sys.pos = np.array(tx_pos)
# ant_device_tx_el.coord_sys.rot = euler_to_rot(phi=135,theta=-15)
ant_device_tx_el.coord_sys.update()


######################
# Rx on walking user
####################

ant_device_rx = add_single_rx(all_actors,waveform,mode_name,
                              pos=[0,0,0],
                              ffd_file='dipole.ffd',
                              scale_pattern=10,
                              parent_h_node=all_actors.actors[actor_rx_name].h_node)

enable_coupling(mode_name,ant_device_tx_az, ant_device_rx)
enable_coupling(mode_name,ant_device_tx_el, ant_device_rx)

# rays can be defined as density or spacing, this is the conversion between the two, if ray_density is set, then

freq_center = ant_device_tx_az.waveforms[mode_name].center_freq
lambda_center = 2.99792458e8 / freq_center
# ray_spacing = np.sqrt(2) * lambda_center / ray_density

print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
sim_options.auto_configure_simulation()

which_mode = ant_device_tx_az.modes[mode_name] # tell it which mode we want to get respones from
ant_device_tx_az.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device_tx_az.waveforms[mode_name].vel_domain
rng_domain = ant_device_tx_az.waveforms[mode_name].rng_domain*2 # multiply by 2 to get total distance, rng_domain is roundtrip
freq_domain = ant_device_tx_az.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx_az.waveforms[mode_name].pulse_domain


# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)
    debug_camera = DebuggingCamera(hMode=ant_device_tx_az.modes[mode_name],
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
                                 x_domain=np.linspace(0, rng_domain[-1], num=512),
                                 y_domain=np.linspace(0, rng_domain[-1], num=512),
                                 fps=fps,
                                 output_video_size = (1280,768),
                                 figure_size = (0.5, 0.5),
                                 camera_attachment=None,
                                 camera_orientation=None)
modeler.pl.show_grid()

all_range_profiles = []
print('running simulation...')
timestamps = np.linspace(0,total_time,num_frames)
for frame_idx in tqdm(range(num_frames)):
    time = frame_idx*dt
    # update all coordinate systems
    all_actors.actors[actor_rx_name].coord_sys.pos = rx_pos
    all_actors.actors[actor_rx_name].coord_sys.update()

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response_az) = pem.retrieveP2PResponse(ant_device_tx_az.modes[mode_name],ant_device_rx.modes[mode_name],
                                              RssPy.ResponseType.RANGE_DOPPLER)
    (ret, response_el) = pem.retrieveP2PResponse(ant_device_tx_el.modes[mode_name], ant_device_rx.modes[mode_name],
                                              RssPy.ResponseType.RANGE_DOPPLER)

    if export_debug:
        if frame_idx == 0:
            debug_logs.write_scene_summary(file_name=f'out_{frame_idx}_from_pem.json')
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file


    target_list, fps_create_target = create_target_list(rd_all_channels_az=response_az[:,0],
                                                        rd_all_channels_el=response_el[:,0],
                                                        rngDomain=rng_domain,
                                                        velDomain=vel_domain,
                                                        azPixels=512, elPixels=512,
                                                        antenna_spacing_wl=0.5,
                                                        radar_fov_az=[-90, 90],
                                                        radar_fov_el=[-90, 90],
                                                        centerFreq=center_freq,
                                                        rcs_min_detect=0,
                                                        min_detect_range=7.5,
                                                        rel_peak_threshold=1e-2,
                                                        max_detections=200,
                                                        return_cfar=False)

    # modeler.mpl_ax_handle.set_data(range_angle_response.T)  # update pyvista matplotlib plot
    # max_of_data = np.max(range_angle_response)
    # modeler.mpl_ax_handle.set_clim(vmin=max_of_data-30, vmax=max_of_data)

    # velocity_min = -10 and max is just for visualization purposes, to scale color of the point cloud
    modeler.add_point_cloud_to_scene(target_list,
                                     tx_pos=tx_pos,
                                     tx_rot=ant_device_tx_az.coord_sys.rot,
                                     color_min=-100,
                                     color_max=-40,
                                     color_mode='p_received',
                                     size_mode='p_received', )
    x = []
    y = []
    all_mag = []
    max_size = 2
    for target in target_list:
        x.append(target_list[target]['xpos'])
        y.append(target_list[target]['ypos'])
        p_rec = target_list[target]['p_received']
        p_rec_db = 10 * np.log10(p_rec)
        all_mag.append(p_rec_db)
    p_rec_db = np.array(all_mag)
    p_rec_db_norm = (p_rec_db - np.min(p_rec_db)) / (np.max(p_rec_db) - np.min(p_rec_db))

    p_rec_db_norm = -100 / p_rec_db * 10  # set a baseline size for all points relative to the numerator

    if time == 0:
        f, ax = plt.subplots(tight_layout=True)
        scat = ax.scatter(x, y, s=p_rec_db_norm * 7, c=p_rec_db, vmin=-100, vmax=-40, cmap='jet')
        ax.set_xlim(0, 1600)
        ax.set_ylim(-800, 800)
        h_chart = pv.ChartMPL(f, size=(0.275, 0.375), loc=(0.7, 0.6))
        h_chart.background_color = (0.5, 0.5, 0.5)
        modeler.pl.add_chart(h_chart)
    else:
        scat.set_offsets(np.c_[x, y])
        scat.set_sizes(p_rec_db_norm * 10)
        scat.set_array(p_rec_db)

    modeler.update_frame(write_frame=False)  # update the visualization and save the image as part of a video

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






