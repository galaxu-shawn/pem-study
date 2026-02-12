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
import sys
import pyvista as pv

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaArray, Waveform
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.post_processing import create_target_list

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy

fps = 30
total_time = 10

num_rx_az = 8
num_rx_el = 8

# waveform parameters for radar
center_freq = 77e9
num_freqs = 512
bandwidth = 300e6
cpi_duration = 0.1
num_pulse_CPI = 128

# simulation options
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .1

export_debug = True  # use this to export the movie at the end of the simulation (.gif showing radar camera and response)

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

# load the road from a file, and assign a material property
actor_road_name = all_actors.add_actor(filename=os.path.join(paths.models,'Simple_Road.stl'),
                                         mat_idx=mat_manager.get_index('asphalt'),
                                         color='black')

actor_road2_name = all_actors.add_actor(filename=os.path.join(paths.models,'Simple_Road.stl'),
                                         mat_idx=mat_manager.get_index('asphalt'),
                                         color='black')

actor_overpass_name = all_actors.add_actor(filename=os.path.join(paths.models,'Overpass.stl'),
                                         mat_idx=mat_manager.get_index('pec'),
                                         color='grey')

# add a multipart actor from json, this preserves the hierarchy of the actor, defined in the json file. There are
# some special types of files that can be loaded. For example, if we have a vehicle, we can load the vehicle from a json
# which will automatically load tires and rotate them as the car drives

actor_vehicle1_name = all_actors.add_actor(filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json'))

actor_vehicle2_name = all_actors.add_actor(filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json'))
actor_vehicle3_name = all_actors.add_actor(filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json'))
actor_vehicle4_name = all_actors.add_actor(filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json'))
actor_vehicle5_name = all_actors.add_actor(filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json'))

# Another example of a multi-part actor, this is a wind turbine, which has a tower, and blades that rotate. The rate
# of rotation is defined in the json file, and the rotation is updated in the update_actor function
actor_windturbine_name = all_actors.add_actor(filename=os.path.join(paths.models,'Wind_Turbine/Wind_Turbine.json'))

# Another special type of multi-part actor is a pedestrian defined as a collada file. This will load the pedestrian and
# animate the body to walk. A an argument 'target_ray_spacing' can be used to create an adaptive shoot grid that will
# use a fixed ray spacing, in meters, no matter where the actor is in the scene
actor_pedestrian_name = all_actors.add_actor(filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'), target_ray_spacing=0.1)

actor_semi_name = all_actors.add_actor(filename=os.path.join(paths.models,'tractor-trailor.stl'), mat_idx=mat_manager.get_index('pec'))

# initialize coordinate systems for all actors, and then set the
# initial positions of each actor. Any value that is not set, will be set to zero. We can set pos/lin/rot/ang for each

# Initialize the position of the actors.

# The vehicle1_ego is a single part actor, so we can set all the parameters directly. We need to take special
# consideration of the velocity vector and the rotation. Because this is a general actor, we would need to rotate
# the actor and the velocity vector to be consistent with how we want the actor to behave
all_actors.actors[actor_vehicle1_name].coord_sys.pos = (0., -8, 0.)
all_actors.actors[actor_vehicle1_name].velocity_mag = 10.
# all_scene_actors['vehicle1_ego'].coord_sys.lin = (5., 0., 0.)

# The vehicle2 is a multipart actor assigned to be a actor_type='vehicle'. This type of actor has some special
# considerations already embedded into the actor. For example, 4 tires will be placed in the correct location, and
# each tire will rotate depending on the velocity of the vehicle. The velocity only needs to be defined with a special
# property .velocity_mag, and the linear velocity vector will be directly calculated from the rotation of the vehicle
all_actors.actors[actor_vehicle2_name].coord_sys.pos = (60, 8, 0.)
all_actors.actors[actor_vehicle2_name].velocity_mag = 10.
all_actors.actors[actor_vehicle2_name].coord_sys.rot = euler_to_rot(phi=-180, theta=0, order='zyz', deg=True)
#
all_actors.actors[actor_vehicle3_name].coord_sys.pos = (120, 8, 0.)
all_actors.actors[actor_vehicle3_name].velocity_mag = 10.
all_actors.actors[actor_vehicle3_name].coord_sys.rot = euler_to_rot(phi=-180, theta=0, order='zyz', deg=True)

all_actors.actors[actor_vehicle4_name].coord_sys.pos = (105, -70, 0.)
all_actors.actors[actor_vehicle4_name].velocity_mag = 7.
all_actors.actors[actor_vehicle4_name].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)

all_actors.actors[actor_vehicle5_name].coord_sys.pos = (110, 70, 0.)
all_actors.actors[actor_vehicle5_name].velocity_mag = 12.
all_actors.actors[actor_vehicle5_name].coord_sys.rot = euler_to_rot(phi=-90, theta=0, order='zyz', deg=True)

all_actors.actors[actor_semi_name].coord_sys.pos = (40, -8, 0.)
all_actors.actors[actor_semi_name].coord_sys.lin = (15., 0., 0.)


# this is an actor_type='other', demonstrates how a multipart actor can be defined that incorporates a rotating part.
# this is a static object, so we will just assign a position within the scene. The rotation of the blades is defined
# in the json file.
all_actors.actors[actor_windturbine_name].coord_sys.pos = (150., 50., 0.)
all_actors.actors[actor_overpass_name].coord_sys.pos = (40.,0.,0.)
all_actors.actors[actor_road2_name].coord_sys.pos = (110., -80, 0.)
all_actors.actors[actor_road2_name].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)

# similiar velocity magnitude can be used with pedestrian loaded from dae, where rotation modifies the linear velocity vector
all_actors.actors[actor_pedestrian_name].coord_sys.pos = (70., 10, 0.)
all_actors.actors[actor_pedestrian_name].velocity_mag = 1.0
all_actors.actors[actor_pedestrian_name].coord_sys.rot = euler_to_rot(phi=180, theta=0, order='zyz', deg=True)
# Let's add a pedestrian with a more complex path to follow instead of a simple linear path. This will also demonstrate
# how to estimate the bulk velocity of the pedestrian using the velocity estimate functionality of the api. This time
# we don't want to update the position based on use_linear_velocity_equation_update, so we will set it to False. This
# means each update of the simulation we must set the position of the pedestrian.
actor_pedestrian2_name = all_actors.add_actor(filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'),
                                                target_ray_spacing=0.1,
                                                use_linear_velocity_equation_update=False)

all_actors.actors[actor_pedestrian2_name].coord_sys.pos = (80., -20, 0.)
all_actors.actors[actor_pedestrian2_name].velocity_mag = 1.0
all_actors.actors[actor_pedestrian2_name].coord_sys.rot = euler_to_rot(phi=180, theta=0, order='zyz', deg=True)


# We are going to create a radar that is attached the vehicle, the hierarchy will look like this:
#
#  Ego Vehicle 1 --> Radar Platform --> Radar Device --> Radar Antenna (Tx/Rx)
#


# simulation parameters
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

range_pixels = 512
doppler_pixels = 512

# Radar Device
# create radar device and antennas, the radar platform is loaded from json file. It is then created in reference to the
# ego vehicle node. The position of the device is place 2.5 meters in front of the vehicle, and 1 meter above the ground
# the device itself does not have any meshes attached to it.
ant_array = AntennaArray(all_actors=all_actors,
                            name = 'array',
                            waveform = waveform,
                            mode_name = mode_name,
                            file_name = None,
                            beamwidth_H = 140,
                            beamwidth_V = 120,
                            polarization = 'V',
                            rx_shape = [num_rx_az, num_rx_el],
                            tx_shape = [1, 1],
                            spacing_wl_x = 0.5,
                            spacing_wl_y = 0.5,
                            load_pattern_as_mesh = True,
                            scale_pattern = 1,
                            parent_h_node = all_actors.actors[actor_vehicle1_name].h_node,
                            normal = 'x',
                            range_pixels=range_pixels,
                            doppler_pixels=doppler_pixels)

ant_device = ant_array.antenna_device
ant_device.coord_sys.pos = (3.1, 0, 1)
ant_device.coord_sys.update()

# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    freq_center = ant_device.waveforms[mode_name].center_freq
    lambda_center = 2.99792458e8 / freq_center
    ray_spacing = np.sqrt(2) * lambda_center / ray_density


# assign modes to devices
print(pem.listGPUs())

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



# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device.modes[mode_name]  # tell it which mode we want to get respones from
ant_device.waveforms[mode_name].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device.waveforms[mode_name].vel_domain
rng_domain = ant_device.waveforms[mode_name].rng_domain
freq_domain = ant_device.waveforms[mode_name].freq_domain
pulse_domain = ant_device.waveforms[mode_name].pulse_domain

print(f'Max Range Domain: {rng_domain[-1]}')




dt = 1 / fps
num_frames = int(total_time / dt)
responses = []
cameraImages = []

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=fps)

# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'

# Start with camera orientation = 'radar' and camera_attachment = 'vehicle1_ego' to get an idea of where the ego vehicle is going.
# Using None for camera_orientation and camera_attachment will free up the camera so that user can pan, zoom,tilt the animation
# as it is being run in pyvista. Note that any manipulations you make will be recorded in the final video since they are being recorded
# as they are being displayed.

output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             x_domain=np.linspace(0,rng_domain[-1],num=512),
                             y_domain=np.linspace(0,rng_domain[-1],num=512),
                             fps=fps,
                             camera_orientation='follow',
                             camera_attachment=actor_vehicle1_name,
                             output_movie_name=output_movie_name)
#modeler.pl.show_grid()

print('running simulation...')
for idx in tqdm(range(num_frames), disable=True):
    time = idx * dt
    # update all coordinate systems
    for actor in all_actors.actors:
        # Actors can have actor_type value, this will handle sub-parts of the actors. For example, if we have a vehicle
        # it will update the tires rotation and velocity, and the vehicle linear velocity based on its orientation
        # if actor is other, it will use the lin/rot/ang values to update the position of the actor
        if actor == 'pedestrian2':
            # this is the special case we define above that is not updating the position based on the velocity vector,
            # but instead we defined a series of points (converted to interpolated function), that we will use to update
            # the position, which will then in turn estimate the velocity.
            if time > 10:  # our limited time function can just be looped since the person is walking in a circle
                # earlier when we defined the pedestrian2, the position was only defined for 10 seconds, so we will
                # loop the time
                update_time = np.mod(time, 10)
            else:
                update_time = time
            # because velocity estimates are based on finite difference, the accuracy will depend on the size of the
            # time step for non-linear motion. For linear motion, the velocity estimate will be accurate.
        all_actors.actors[actor].update_actor(time=time)

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)

    # ToDo, not sure arrangement of elements is correct, will revisit later, point cloud might be flipped
    rd_all_channels_az = response[0,0:num_rx_az]
    rd_all_channels_el = response[0,::num_rx_az]
    target_list, fps_create_target = create_target_list(rd_all_channels_az=rd_all_channels_az,
                                                        rd_all_channels_el=rd_all_channels_el,
                                                        rngDomain=rng_domain,
                                                        velDomain=vel_domain,
                                                        azPixels=512, elPixels=512,
                                                        antenna_spacing_wl=0.5,
                                                        radar_fov_az=[-90, 90],
                                                        radar_fov_el=[-90, 90],
                                                        centerFreq=center_freq,
                                                        rcs_min_detect=0,
                                                        min_detect_range=20,
                                                        rel_peak_threshold=1e-2,
                                                        max_detections=20,
                                                        return_cfar=False)

    # modeler.mpl_ax_handle.set_data(range_angle_response.T)  # update pyvista matplotlib plot
    # max_of_data = np.max(range_angle_response)
    # modeler.mpl_ax_handle.set_clim(vmin=max_of_data-30, vmax=max_of_data)

    # velocity_min = -10 and max is just for visualization purposes, to scale color of the point cloud
    modeler.add_point_cloud_to_scene(target_list,
                                     tx_pos=ant_device.coord_sys.pos,
                                     tx_rot=ant_device.coord_sys.rot,
                                     color_min=-100,
                                     color_max=-40,
                                     color_mode='p_received',
                                     size_mode='p_received',
                                     max_radius=2,)
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
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)
        h_chart = pv.ChartMPL(f, size=(0.275, 0.375), loc=(0.7, 0.6))
        h_chart.background_color = (0.5, 0.5, 0.5)
        modeler.pl.add_chart(h_chart)
    else:
        scat.set_offsets(np.c_[x, y])
        scat.set_sizes(p_rec_db_norm * 7)
        scat.set_array(p_rec_db)


    # exporting radar camera images
    if export_debug:
        if idx == 0:
            debug_logs.write_scene_summary(file_name=f'out_{idx}.json')
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)

    modeler.update_frame()
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
