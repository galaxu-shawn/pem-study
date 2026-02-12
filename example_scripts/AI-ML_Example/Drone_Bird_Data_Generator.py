#
# Copyright ANSYS. All rights reserved.
#

import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import PIL.Image
import os
import uuid
import shutil
import sys
import scipy.io
import pyvista as pv

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx, add_multi_channel_radar_az_el
from pem_utilities.utils import generate_rgb_from_array
from pem_utilities.far_fields import convert_ffd_to_gltf
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.video_generation import generate_video_from_multiple_videos


num_scenes = 20
list_of_actors = [os.path.join(paths.models,'Quadcopter/Quadcopter.json'),os.path.join(paths.models,'Quadcopter2/Quadcopter2.json'),
                  os.path.join(paths.models,'Bird1/Bird1.json'),os.path.join(paths.models,'Bird2/Bird2.json')]

radar_pos = [0, 0, 0]
radar_rot = euler_to_rot(0, -90, 0, order='zyx', deg=True) # looking up
num_rx_azimuth = 1
num_rx_elevation = 0
polarization = 'VV'
ant_beamwidth_h = 160
ant_beamwidth_v = 160
dwell_time = 2 # seconds, will be corrected to N number of frames int(dwell_time/cpi_duration)


placement_dome_radius_min_max = [5, 25] # meters
fov_min_max = [-90, 90] # degrees
include_mobility = True
velocity_min_max = [0, 2] # m/s, only used if mobility is True
scale_min_max = [0.5, 2] # scale of the drone

# drone parameters if a drone is chosen
rotor_speed_min_max = [50, 80] # Hz

# bird parameters if a bird is chosen
flap_freq_min_max = [0.01,5]    # Hz
flap_ang_min_max = [10, 45]      # degrees

# waveform parameters
center_freq = 10e9
num_freqs = 512
bandwidth = 300e6
cpi_duration = 100e-3
num_pulse_CPI = 1001 # this will result in a pulse interval of 1ms

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .2

# export debug logs and show modeler for visualization
export_debug = False
show_modeler = True


#######################################################################################################################
# End of user defined parameters
#######################################################################################################################

# output file will be stored in this directory
output_path = os.path.join(paths.output, 'drone_syn_data/')
os.makedirs(output_path, exist_ok=True)

if int(dwell_time/cpi_duration) <= 1:
    # create exception for dwell time less than 2 * CPI duration
    raise ValueError('Dwell time must be greater than 2 * CPI duration')

actual_dwell_time = cpi_duration*int(dwell_time/cpi_duration)

output_id = 'Gen_' + str(uuid.uuid4().hex[:6])
output_path = os.path.join(output_path, output_id)
os.makedirs(output_path, exist_ok=True)

# convert numpy arrays to lists for json serialization
def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return x.tolist()
    raise TypeError(x)

time_stamps = np.linspace(0, actual_dwell_time, int(dwell_time/cpi_duration))

summary_dict = {'output_id': output_id}
summary_dict['scene_parameter_inputs'] = {'num_scenes': num_scenes,
                                          'list_of_actors': list_of_actors,
                                          'radar_pos': radar_pos,
                                          'radar_rot': radar_rot,
                                          'num_rx_azimuth': num_rx_azimuth,
                                          'num_rx_elevation': num_rx_elevation,
                                          'polarization': polarization,
                                          'ant_beamwidth_h': ant_beamwidth_h,
                                          'ant_beamwidth_v': ant_beamwidth_v,
                                          'dwell_time': actual_dwell_time,
                                          'placement_dome_radius_min_max': placement_dome_radius_min_max,
                                          'fov_min_max': fov_min_max,
                                          'include_mobility': include_mobility,
                                          'velocity_min_max': velocity_min_max,
                                          'scale_min_max': scale_min_max,
                                          'rotor_speed_min_max': rotor_speed_min_max,
                                          'flap_freq_min_max': flap_freq_min_max,
                                          'flap_ang_min_max': flap_ang_min_max,
                                          'center_freq': center_freq,
                                          'num_freqs': num_freqs,
                                          'bandwidth': bandwidth,
                                          'cpi_duration': cpi_duration,
                                          'num_pulse_CPI': num_pulse_CPI,
                                          'go_blockage': go_blockage,
                                          'max_num_refl': max_num_refl}

summary_filename = output_id + '_summary.json'
summary_output_filename = os.path.join(output_path, summary_filename)
scene_idx_creation = False # intial state, only changes to True it success

if show_modeler:
    list_of_all_output_videos = []

for scene_idx in range(num_scenes):

    # define paths to save data
    print(f'Generating Variation {scene_idx}...')
    scene_idx_str = str(scene_idx).zfill(5)
    scene_file_name = f'Result_{output_id}_{scene_idx_str}'

    individual_scene_path = os.path.join(output_path, f'{scene_idx_str}')
    if not os.path.exists(individual_scene_path):
        os.makedirs(individual_scene_path)

    # define scene parameters
    choose_actor = np.random.choice(list_of_actors)
    # get filename from choose_drone, this is the path to the drone json file
    actor_filename = os.path.basename(choose_actor)
    actor_name = os.path.splitext(actor_filename)[0]

    distance = np.random.uniform(placement_dome_radius_min_max[0], placement_dome_radius_min_max[1])
    angle_theta = np.random.uniform(fov_min_max[0], fov_min_max[1])
    angle_phi = np.random.uniform(0, 360)
    rotor_speed = np.random.uniform(rotor_speed_min_max[0], rotor_speed_min_max[1])
    actor_phi = np.random.uniform(0, 360)
    actor_theta = np.random.uniform(-30, 30)
    actor_psi = np.random.uniform(-30, 30)
    velocity = np.random.uniform(velocity_min_max[0], velocity_min_max[1])
    scale = np.random.uniform(scale_min_max[0], scale_min_max[1])

    flap_freq = np.random.uniform(flap_freq_min_max[0], flap_freq_min_max[1])
    flap_ang = np.random.uniform(flap_ang_min_max[0], flap_ang_min_max[1])

    simulation_setup_params = {'num_reflections': max_num_refl,
                               'num_transmissions': max_num_trans,
                               'ray_density': ray_density,
                               'go_blockage': go_blockage,
                               'waveform': None,
                               'time_start': float(time_stamps[0]),
                               'time_end': float(time_stamps[-1]),
                               'time_n': len(time_stamps)}

    # converte distance anggle_theta and angle_phi to x,y,z
    x = distance * np.sin(np.deg2rad(angle_theta)) * np.cos(np.deg2rad(angle_phi))
    y = distance * np.sin(np.deg2rad(angle_theta)) * np.sin(np.deg2rad(angle_phi))
    z = distance * np.cos(np.deg2rad(angle_theta))
    pos = [x, y, z]

    radar_instance = {'pos': list(radar_pos),
                      'rot': list(radar_rot),
                      'polarization': polarization,
                      'ant_beamwidth_h': ant_beamwidth_h,
                      'ant_beamwidth_v': ant_beamwidth_v,
                      'num_rx_azimuth': num_rx_azimuth,
                      'num_rx_elevation': num_rx_elevation,
                      'dwell_time': actual_dwell_time}

    output_npy_files = {'range_domain': None,
                        'vel_domain': None,
                        'freq_domain': None,
                        'pulse_domain': None,
                        'responses': None}




    # begin simulation
    # try:


    # material manager used to load predefined materials, defined in material_library.json. This will load all the materials
    # and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
    mat_manager = MaterialManager()
    # create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
    all_actors = Actors()

    actor_name = all_actors.add_actor(name='actor',filename=choose_actor,target_ray_spacing=0.05,scale_mesh=scale)
    # all_actors.actors[actor_name].update_rot_based_on_ang_vel = False
    actor_type = all_actors.actors[actor_name].actor_type

    if include_mobility:
        all_actors.actors[actor_name].velocity_mag = velocity

    all_actors.actors[actor_name].coord_sys.pos = pos
    all_actors.actors[actor_name].coord_sys.rot = euler_to_rot(actor_phi, actor_theta, actor_psi, order='zyx', deg=True)
    all_actors.actors[actor_name].coord_sys.update()

    if actor_type == 'quadcopter':
        all_actors.actors[actor_name].rotor_ang = [0, 0, rotor_speed*2*np.pi] # in rad/s
    elif actor_type == 'bird':
        all_actors.actors[actor_name].flap_ang = flap_ang
        all_actors.actors[actor_name].flap_freq = flap_freq


    actor_instance = {'actor':  actor_name,
                      'actor_type':actor_type,
                      'distance': distance,
                      'angle_theta': angle_theta,
                      'angle_phi': angle_phi,
                      'cartesian_pos': pos,
                      'actor_phi': actor_phi,
                      'actor_theta': actor_theta,
                      'actor_psi': actor_psi,
                      'rotation_order': 'zyx',
                      'include_mobility': include_mobility,
                      'velocity': velocity,
                      'scale_model': scale}

    if actor_type == 'quadcopter':
        actor_instance['rotor_speed'] = rotor_speed,
    elif actor_type == 'bird':
        actor_instance['flap_freq'] = flap_freq
        actor_instance['flap_ang'] = flap_ang

    meta_data_dict = {'scene_idx': scene_idx,
                      'scene_file_name': scene_file_name,
                      'output_id': scene_idx_str,
                      'output_path': individual_scene_path,
                      'radar_parameters': radar_instance,
                      'actor_parameters': actor_instance,
                      'simulation_parameters': simulation_setup_params,
                      'output_npy': output_npy_files,
                      'output_mat': None
                      }



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

    simulation_setup_params['waveform'] = waveform_dict

    pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
    wavelength = 3e8 / center_freq
    mode_name = 'mode1'
    # waveform will be used for both Tx and Rx
    waveform = Waveform(waveform_dict)

    ant_device = add_multi_channel_radar_az_el(all_actors,
                                                  waveform,
                                                  mode_name,
                                                  num_rx_az=num_rx_azimuth,
                                                  num_rx_el=num_rx_elevation,
                                                  spacing_wl=0.5,
                                                  pos=radar_pos,
                                                  rot=radar_rot,
                                                  normal='x',
                                                  beamwidth_H=ant_beamwidth_h,
                                                  beamwidth_V=ant_beamwidth_v,
                                                  polarization=polarization,
                                                  scale_pattern=2)


    if ray_density is not None:
        freq_center = ant_device.waveforms[mode_name].center_freq
        lambda_center = 2.99792458e8 / freq_center
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

    # optional check if RSS is configured
    # this will also be checked before response computation
    if not pem.isReady():
        print("RSS is not ready to execute a simulation:\n")
        print(pem.getLastWarnings())

    # display setup
    # print(pem.reportSettings())

    # get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
    # for post-processing and scaling axes
    which_mode = ant_device.modes[mode_name]  # tell it which mode we want to get respones from
    ant_device.waveforms[mode_name].get_response_domains(which_mode)

    # domain in every dimension
    vel_domain = ant_device.waveforms[mode_name].vel_domain
    rng_domain = ant_device.waveforms[mode_name].rng_domain
    freq_domain = ant_device.waveforms[mode_name].freq_domain
    pulse_domain = ant_device.waveforms[mode_name].pulse_domain

    print(f'Velocity Window: {vel_domain[-1]-vel_domain[0]}')
    print(f'Max Range: {rng_domain[-1]}')


    responses = []
    fps = 1 / (time_stamps[1] - time_stamps[0])
    # activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
    # if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
    if export_debug:
        debug_logs = DebuggingLogs(output_directory=individual_scene_path)
        debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
                                        display_mode='coating',
                                        output_size=(512,512),
                                        background_color=255,
                                        frame_rate=fps,
                                        output_directory=individual_scene_path)


    if show_modeler:
        output_movie_name = os.path.join(individual_scene_path, 'out_vis.mp4')
        modeler = ModelVisualization(all_actors,
                                     show_antennas=True,
                                     freq_domain=freq_domain,
                                     pulse_domain=pulse_domain,
                                     fps=fps,
                                     camera_orientation=None,
                                     camera_attachment=None,
                                     output_movie_name=output_movie_name)
        list_of_all_output_videos.append(output_movie_name)

        modeler.pl.add_mesh(pv.Sphere(radius=distance,end_phi=90), color='blue',opacity=0.15)


        modeler.pl.show_grid()
    print('running simulation...')
    for time in time_stamps:
        # update all coordinate systems
        for actor in all_actors.actors:
            all_actors.actors[actor].update_actor(time=time)

        pem_api_manager.isOK(pem.computeResponseSync())
        (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)

        # exporting radar camera images
        if export_debug:
            if time == 0:
                debug_logs.write_scene_summary(file_name=f'out_0.json')
            debug_camera.generate_image()
            # generate radar camera image debug_camera.current_image now has this image frame,
            # debug_camera.camera_images has all the images from all frames
            # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)

        if show_modeler:
            im_data = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))

            if time == 0:
                str_n = 2
                # create a string that list all of the scene parameters
                string_to_print = f'SceneIDX:{scene_idx_str}  Model:{actor_filename}'
                string_to_print += f'\nActor Orientation - Phi:{actor_phi:.{str_n}f} deg, Theta:{actor_theta:.{str_n}f} deg, Psi:{actor_psi:.{str_n}f} deg'
                string_to_print += f'\nActor Position - XYZ: {pos[0]:.{str_n}f},{pos[1]:.{str_n}f},{pos[2]:.{str_n}f}'
                string_to_print += f'\nDistance: {distance:.{str_n}f} m, Theta: {angle_theta:.{str_n}f} deg, Phi: {angle_phi:.{str_n}f} deg'
                string_to_print += f'\nVelocity:{velocity:.{str_n}f} m/s\nScale{scale:.{str_n}f}'
                if actor_type == 'quadcopter':
                    string_to_print += f'\n{rotor_speed:.{str_n}f} Hz'
                elif actor_type == 'bird':
                    string_to_print += f'\nFlapFreq{flap_freq:.{str_n}f} Hz\nFlapAngle{flap_ang:.{str_n}f} deg'
            # string_to_print += f'\n{time:.2f} s'


            modeler.pl.add_text(string_to_print, position='upper_left', font_size=12, color='black')
            modeler.update_frame(plot_data=im_data,plot_limits=[im_data.min(),im_data.max()])

    if show_modeler:
        modeler.close()

    # post-process images into gif. This will take the radar camera and range doppler response and place them side by
    # side in a gif
    if export_debug:
        # debug_camera.write_camera_to_gif(file_name='camera.gif')
        # debug_camera.write_response_to_gif(responses,file_name='response.gif',rng_domain=rng_domain,vel_domain=vel_domain)
        debug_camera.write_camera_and_response_to_gif(responses,
                                                      file_name='camera_and_response.gif',
                                                      rng_domain=rng_domain,
                                                      vel_domain=vel_domain)

    responses = np.array(responses)
    # save data
    np.save(os.path.abspath(os.path.join(individual_scene_path, f'range_domain.npy')), rng_domain)
    np.save(os.path.abspath(os.path.join(individual_scene_path, f'vel_domain.npy')), vel_domain)
    np.save(os.path.abspath(os.path.join(individual_scene_path, f'freq_domain.npy')), freq_domain)
    np.save(os.path.abspath(os.path.join(individual_scene_path, f'pulse_domain.npy')), pulse_domain)
    np.save(os.path.abspath(os.path.join(individual_scene_path, f'responses.npy')), responses)
    np.save(os.path.abspath(os.path.join(individual_scene_path, f'time_stamps.npy')), time_stamps)
    meta_data_dict['output_npy']['range_domain'] = 'range_domain.npy'
    meta_data_dict['output_npy']['vel_domain'] = 'vel_domain.npy'
    meta_data_dict['output_npy']['freq_domain'] = 'freq_domain.npy'
    meta_data_dict['output_npy']['pulse_domain'] = 'pulse_domain.npy'
    meta_data_dict['output_npy']['responses'] = 'responses.npy'
    meta_data_dict['output_npy']['time_stamps'] = 'time_stamps.npy'

    scipy.io.savemat(os.path.join(individual_scene_path,'results.mat'), {"range_domain": rng_domain,
                                                      "vel_domain": vel_domain,
                                                      "freq_domain": freq_domain,
                                                      "pulse_domain": pulse_domain,
                                                      "responses": responses,
                                                      "time_stamps": time_stamps})
    meta_data_dict['output_mat'] = 'results.mat'

    meta_data_output_filename = os.path.abspath(os.path.join(individual_scene_path, f'meta_data_{scene_idx_str}.json'))
    json_object = json.dumps(meta_data_dict, default=convert,indent=4)
    with open(meta_data_output_filename, "w") as outfile:
        outfile.write(json_object)

    summary_dict[scene_idx] = 'Success'
    pem.reset()


    json_object = json.dumps(summary_dict, default=convert,indent=4)
    with open(summary_output_filename, "w") as outfile:
        outfile.write(json_object)

# create a video showing all the videos concatenated together
if show_modeler:
    generate_video_from_multiple_videos(list_of_all_output_videos,
                                        output_dir = output_path,
                                        output_name =  'all_output_videos.mp4')
