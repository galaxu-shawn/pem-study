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

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaDevice, add_antenna_device_from_json
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.cfar import CFAR
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

export_debug = True  # use this to export the movie at the end of the simulation (.gif showing radar camera and response)


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


# Radar Device
# create radar device and antennas, the radar platform is loaded from json file. It is then created in reference to the
# ego vehicle node. The position of the device is place 2.5 meters in front of the vehicle, and 1 meter above the ground
# the device itself does not have any meshes attached to it.

# When loading an antenna from json, we need to first initialize the mode defined in teh json file, if multiple modes are
# defined we need to select which one. Once initialized, we can add the antennas to the device, select which mode we
# and add the antennas to that node. The antennas are defined inside the json file. We have an additional option
# to load the far-field pattern data as a mesh, and create a pyvista actor that we can later visualize
# once the antennas are added, we can add the mode to the device


# helper function to add antenan from json
ant_device = add_antenna_device_from_json(all_actors,
                                             'example_1tx_1rx.json',
                                             'mode1',
                                             pos=(3.5, 0., .5),
                                             rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True),
                                             scale_pattern=0.75,
                                             parent_h_node=all_actors.actors[actor_vehicle1_name].h_node)

# manually add antenna from json, showing each step, commented out in favor of the helper function above
# ant_device = AntennaDevice('example_1tx_1rx.json',
#                            parent_h_node=all_actors.actors[actor_vehicle1_name].h_node,
#                            all_actors=all_actors)
# ant_device.initialize_mode(mode_name='mode1')
# ant_device.coord_sys.pos = (3.5, 0., .5)
# ant_device.coord_sys.update()
# ant_device.add_antennas(mode_name='mode1', load_pattern_as_mesh=True, scale_pattern=1)
# ant_device.add_mode(mode_name='mode1')

print(pem.listGPUs())
ray_density = 0.01
# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = 2.99792458e8 / ant_device.waveforms['mode1'].center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density


sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = 3
sim_options.max_transmissions = 2
sim_options.go_blockage = -1
sim_options.field_of_view = 180
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

# display setup
# print(pem.reportSettings())

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device.modes['mode1']  # tell it which mode we want to get respones from
ant_device.waveforms['mode1'].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device.waveforms['mode1'].vel_domain
rng_domain = ant_device.waveforms['mode1'].rng_domain
freq_domain = ant_device.waveforms['mode1'].freq_domain
pulse_domain = ant_device.waveforms['mode1'].pulse_domain

fps = 30;
dt = 1 / fps
T = 11;
numFrames = int(T / dt)
responses = []
cameraImages = []

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes['mode1'],
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
                             rng_domain=rng_domain,
                             vel_domain=vel_domain,
                             fps=fps,
                             camera_orientation='follow',
                             camera_attachment=actor_vehicle1_name,
                             output_movie_name=output_movie_name)
#modeler.pl.show_grid()

print('running simulation...')
for iFrame in tqdm(range(numFrames), disable=True):
    time = iFrame * dt
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
    (ret, response) = pem.retrieveResponse(ant_device.modes['mode1'], RssPy.ResponseType.RANGE_DOPPLER)
    # calculate response in dB to overlay in pyvista plotter
    imData = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))


    # cfar_types = ['CA', 'GO', 'SO', 'OS']
    cfar_types = ['GO']
    
    print("CFAR Detection Demo")
    print("=" * 50)
    data = np.abs(response[0][0])
    for cfar_type in cfar_types:
        detector = CFAR(
            training_cells=12,
            guard_cells=3,
            threshold_factor=10.0,
            cfar_type=cfar_type,
            normalized_data=True
        )
        
        detections, threshold = detector.detect(data, return_threshold=True)
        n_detections = np.sum(detections)
        
        print(f"\n{cfar_type}-CFAR Results:")
        print(f"  Total detections: {n_detections}")
        print(f"  Detection rate: {n_detections / data.size * 100:.3f}%")
        print(f"  Theoretical PFA: {detector.get_pfa():.2e}")
        

    # exporting radar camera images
    if export_debug:
        # This is not required, but will write out scene summary of the simulation setup,
        # only writing out first and last frame to limit number of files, useful for debugging.
        if iFrame == 0 or iFrame == numFrames - 1:
            debug_logs.write_scene_summary(file_name=f'out_{iFrame}.json')
            debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)

    modeler.update_frame(plot_data=np.rot90(detections),plot_limits=[0,1])
    # modeler.update_frame(plot_data=imData)
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
