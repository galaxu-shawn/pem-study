#
# Copyright ANSYS. All rights reserved.
#

import json

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import speed_of_light
import time as walltime
import os
import sys

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot, look_at
from pem_utilities.antenna_device import Waveform, AntennaArray
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import *
from pem_utilities.post_processing import create_target_list, plot_target_list, range_doppler_map, animate_target_list
from pem_utilities.barker_code import BarkerCode
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

use_barker_code_and_matched_filter = True
replace_targets_with_corner_reflectors = True
center_freq = 10e9
transmit_power = 10e3
rx_noise_db = None
show_modeler=True

# number of frames to simulate, including time step
num_frames = 41
dt = 2

target1_pos = [10000, 2000,  0]
target2_pos = [3000, 0,    1000]
target3_pos = [9259, 200, -200]
target4_pos = [10003, -2000, 0]

target1_lin = [-100, 0,  0]
target2_lin = [200, 0,    0]
target3_lin = [-100, 0, -0]
target4_lin = [-100, 0,  0]

# if corner reflectors are used, the rcs value will be used to calculate the size of the reflector
target1_rcs = 36.825
target2_rcs = 29.13
target3_rcs = 47.8
target4_rcs = 36.825

# all_target_pos = [target1_pos, target2_pos, target3_pos, target4_pos]
# all_target_lin = [target1_lin, target2_lin, target3_lin, target4_lin]
# all_target_rcs = [target1_rcs, target2_rcs, target3_rcs, target4_rcs]

all_target_pos = [target1_pos, target4_pos]
all_target_lin = [target1_lin, target4_lin]
all_target_rcs = [target1_rcs, target4_rcs]

all_targets = zip(all_target_pos, all_target_lin,all_target_rcs)

barker = BarkerCode(code_length = 13,center_freq=center_freq,
             num_pulses=512,
             pri=100e-6,
             duty_cycle=0.025,
             adc_sampling_rate=6.5e6,
             number_adc_samples=512,
             blanking_period=2000,
             upsample_factor=8,
             transmit_power=transmit_power,
             rx_noise_db=rx_noise_db)

# barker.waveform_dict will be calculated that will be the required input waveform for the perceive EM simulation
# barker.plot_input_waveform(enhanced_plots=True)


# simulation options
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 4
max_num_trans = 1
ray_density = .01


export_debug = False  # use this to export the movie at the end of the simulation (.gif showing radar camera and response)


# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
# multiple material libraries can be loaded and the indices will be updated accordingly
# mat_manager = MaterialManager(['material_properties_ITU_3.85GHz.json','material_library.json'])
mat_manager = MaterialManager()
# output file will be stored in this directory
os.makedirs(paths.output, exist_ok=True)


#######################################################################################################################
#
# Create Scene, 3 corner reflectors
#
#######################################################################################################################

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()


wl = speed_of_light / center_freq


#dynamic_generator_updates=False will not create a new geometry at each time step, only set to True if you
# want to change the size of the reflector on each update
target_names = []

for pos, lin, rcs in all_targets:
    if not replace_targets_with_corner_reflectors:
        target_name = all_actors.add_actor(name='target',
                                           filename=os.path.join(model_path,'/Quadcopter/Quadcopter.json'),
                                           target_ray_spacing=0.05,scale_mesh=4)
    else:
        rcs = 10 ** (rcs / 10)
        prim = CornerReflector(rcs=rcs, wl=speed_of_light / center_freq, orientation='x',
                                       is_square=False)  # if use an rcs value, it will override teh length and calculate length from rcs
        target_name = all_actors.add_actor(name='prim_example', generator=prim, mat_idx=mat_manager.get_index('pec'),
                                         target_ray_spacing=wl / 4, dynamic_generator_updates=False)
        print(f'Magnitude of pos1: {np.linalg.norm(pos)}')
        all_actors.actors[target_name].coord_sys.rot = look_at((pos[0], pos[1], pos[2]), (0, 0, 0))

    all_actors.actors[target_name].coord_sys.pos = pos
    all_actors.actors[target_name].coord_sys.lin = lin
    all_actors.actors[target_name].coord_sys.update()
    target_names.append(target_name)

waveform_dict = barker.waveform_dict
pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
mode_name = 'mode1'
if not use_barker_code_and_matched_filter:
    waveform_dict["output"] = "RangeDoppler" # this will output the range doppler map, and not use any of the barker matched filter
# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)

#################
# Tx antenna

radar_actor_name = all_actors.add_actor() # easier to attach a camera to an actor rather than an antenna
all_actors.actors[radar_actor_name].coord_sys.pos = [0, 0, 0]
all_actors.actors[radar_actor_name].coord_sys.rot = np.eye(3)
all_actors.actors[radar_actor_name].coord_sys.update()
#use dipole antenna for easier radar range calculation

num_rx_az = 32
num_rx_el = 16

ant_array_az = AntennaArray(all_actors=all_actors,
                            name = 'array',
                            waveform = waveform,
                            mode_name = mode_name,
                            beamwidth_H = 140,
                            beamwidth_V = 120,
                            polarization = 'V',
                            rx_shape = [1, num_rx_az],
                            tx_shape = [1, 1],
                            spacing_wl_x = 0.5,
                            spacing_wl_y = 0.5,
                            load_pattern_as_mesh = True,
                            scale_pattern = 10,
                            parent_h_node = all_actors.actors[radar_actor_name].h_node,
                            normal = 'x',
                            range_pixels=512,
                            doppler_pixels=256
                            )

ant_device_az = ant_array_az.antenna_device

ant_array_el = AntennaArray(all_actors=all_actors,
                            name = 'array',
                            waveform = waveform,
                            mode_name = mode_name,
                            file_name = None,
                            beamwidth_H = 140,
                            beamwidth_V = 120,
                            polarization = 'V',
                            rx_shape = [num_rx_el, 1],
                            tx_shape = [1, 1],
                            spacing_wl_x = 0.5,
                            spacing_wl_y = 0.5,
                            load_pattern_as_mesh = True,
                            scale_pattern = 1000,
                            parent_h_node = all_actors.actors[radar_actor_name].h_node,
                            normal = 'x',
                            range_pixels=512,
                            doppler_pixels=256
                            )

ant_device_el = ant_array_el.antenna_device

center_freq = ant_device_az.waveforms[mode_name].center_freq

# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = speed_of_light / center_freq
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

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device_az.modes[mode_name]  # tell it which mode we want to get respones from
ant_device_az.waveforms[mode_name].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device_az.waveforms[mode_name].vel_domain
rng_domain = ant_device_az.waveforms[mode_name].rng_domain
freq_domain = ant_device_az.waveforms[mode_name].freq_domain
pulse_domain = ant_device_az.waveforms[mode_name].pulse_domain
output_format = ant_device_az.waveforms[mode_name].output

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
fps = 10
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_az.modes[mode_name],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)


# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
if show_modeler:
    output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
    modeler = ModelVisualization(all_actors,
                                 show_antennas=True,
                                 overlay_results=True,
                                 rng_domain=rng_domain,
                                 vel_domain=vel_domain,
                                 shape=(512, 512),
                                 camera_orientation='follow',
                                 camera_attachment=target_names[0],
                                 output_movie_name=output_movie_name)
    # modeler.pl.show_grid()


print('running simulation...')

time_stamps = np.linspace(0, num_frames*dt-dt, num_frames)
single_response_all_frames = []
all_ref_points = []
target_lists = []
for iFrame in tqdm(range(num_frames), disable=False):
    time = time_stamps[iFrame]
    for actor in all_actors.actors:
        # default will update the actors position based on intial position and lin vel
        all_actors.actors[actor].update_actor(time=time)
    start_sim_time = walltime.time()
    pem_api_manager.isOK(pem.computeResponseSync())
    end_sim_time = walltime.time()
    # print(f'Simulation Time: {(end_sim_time - start_sim_time)*1e3} mseconds')

    response_type = RssPy.ResponseType.FREQ_PULSE
    if not use_barker_code_and_matched_filter:
        response_type = RssPy.ResponseType.RANGE_DOPPLER

    (ret, response_az) = pem.retrieveResponse(ant_device_az.modes[mode_name], response_type)
    (ret, response_el) = pem.retrieveResponse(ant_device_el.modes[mode_name], response_type)
    response_az = np.array(response_az)
    response_el = np.array(response_el)

    ref_points = {}
    for target_name in target_names:
        ref_points[target_name] = all_actors.actors[target_name].coord_sys.pos

    all_ref_points.append(ref_points)

    if use_barker_code_and_matched_filter:
        start_sim_time = walltime.time()
        rd_all_channels_az = []
        for ch in range(num_rx_az):
            barker.process_received_signal(response_az[0][ch])
            pulse_vs_range = barker.output_waveform_matched_filter_td
            doppler_vs_range = np.fft.fftshift(np.fft.fft(pulse_vs_range, axis=0),axes=0)
            rd_all_channels_az.append(doppler_vs_range)

        single_response_all_frames.append(doppler_vs_range)
        img = 20*np.log10(np.fmax(np.abs(doppler_vs_range),1e-30))
        print(f'Max: {np.max(img)}')

        rd_all_channels_el = []
        for ch in range(num_rx_el):
            barker.process_received_signal(response_el[0][ch])
            pulse_vs_range = barker.output_waveform_matched_filter_td
            doppler_vs_range = np.fft.fftshift(np.fft.fft(pulse_vs_range, axis=0),axes=0)
            rd_all_channels_el.append(doppler_vs_range)
        rng_domain = barker.output_range_domain_matched_filter
        end_sim_time = walltime.time()
        print(f'Barker Processing Time: {(end_sim_time - start_sim_time)*1e3} mseconds')
    else:
        # this is if the output is already range doppler
        rd_all_channels_az = response_az[0]
        rd_all_channels_el = response_el[0]

        single_response_all_frames.append(rd_all_channels_az[0])
        img = 20*np.log10(np.fmax(np.abs(rd_all_channels_az[0]),1e-30))
        print(f'Max: {np.max(img)}')

    target_list, fps_create_target, cfar = create_target_list(rd_all_channels_az=rd_all_channels_az,
                                                              rd_all_channels_el=rd_all_channels_el,
                                                              rngDomain=rng_domain,
                                                              velDomain=vel_domain,
                                                              azPixels=512, elPixels=512,
                                                              antenna_spacing_wl=0.5,
                                                              radar_fov_az=[-90, 90],
                                                              radar_fov_el=[-90, 90],
                                                              centerFreq=center_freq,
                                                              rcs_min_detect=5,
                                                              min_detect_range=2000,
                                                              rel_peak_threshold=1e-2,
                                                              max_detections=100,
                                                              incident_power=transmit_power,
                                                              return_cfar=True)
    target_lists.append(target_list)
    print(f'Number of Targets: {len(target_list)}')

    if show_modeler:
        modeler.update_frame(plot_data=img.T,plot_limits=[np.max(img)-80,np.max(img)])


# create scatter plots of the target list (intial position)
plot_target_list(target_lists[0],
                 plot_2d=True,plot_3d=True,
                 maximum_range=rng_domain[-1],
                 reference_points=all_ref_points[0],
                 figure_size=(14,7))


# provide all range doppler data for a single channel, so plot compare is easier
animate_target_list(target_lists,
                    range_doppler=single_response_all_frames,
                    maximum_range=rng_domain[-1],
                    reference_points=all_ref_points,
                    output_file=os.path.join(output_path,'target_animation.gif'),
                    interval=200, fps=1/dt*5,figure_size=(16,12))

if show_modeler:
    modeler.close()
