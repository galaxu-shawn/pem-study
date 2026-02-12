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
from pem_utilities.antenna_device import add_antenna_device_from_json

from pem_utilities.utils import generate_rgb_from_array
from pem_utilities.far_fields import convert_ffd_to_gltf
from pem_utilities.animation_selector_CMU import select_CMU_animation
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
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

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .1


debug_logs = DebuggingLogs(output_directory=paths.output)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()


# empty actor, we will use this as a parent node for the radar platform
actor_radar_name = all_actors.add_actor(name='radar_actor')


# Another special type of multi-part actor is a pedestrian defined as a collada file. This will load the pedestrian and
# animate the body to walk. A an argument 'target_ray_spacing' can be used to create an adaptive shoot grid that will
# use a fixed ray spacing, in meters, no matter where the actor is in the scene

# cmu_filename, cmu_description = select_CMU_animation(activity_type='walking') # random selection within walking category
cmu_filename, cmu_description = select_CMU_animation() # random selection
# cmu_filename, cmu_description = select_CMU_animation(file_name='140_07') # specific selection
# cmu_filename = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\Texting and Walking_Blender.dae'
# cmu_filename = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\Texting and Walking.dae'
actor_pedestrian_name = all_actors.add_actor(name='pedestrian', filename=cmu_filename, target_ray_spacing=0.1,use_experimental_dae_load=False)
mesh = all_actors.actors[actor_pedestrian_name].get_mesh()
# initialize coordinate systems for all actors, and then set the
# initial positions of each actor. Any value that is not set, will be set to zero. We can set pos/lin/rot/ang for each

# Initialize the position of the actors.

# The vehicle1_ego is a single part actor, so we can set all the parameters directly. We need to take special
# consideration of the velocity vector and the rotation. Because this is a general actor, we would need to rotate
# the actor and the velocity vector to be consistent with how we want the actor to behave
all_actors.actors[actor_radar_name].coord_sys.pos = (0., 0.0, 0.)


# similiar velocity magnitude can be used with pedestrian loaded from dae, where rotation modifies the linear velocity vector
all_actors.actors[actor_pedestrian_name].coord_sys.pos = (5, 0,  0.)


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

mode_name = 'mode1'
# ant_device = add_antenna_device_from_json(all_actors,'example_1tx_1rx.json',
#                                           mode_name=mode_name,
#                                           parent_h_node=all_actors.actors[actor_radar_name].h_node,
#                                           scale_pattern=1.0)
ant_device = add_antenna_device_from_json(all_actors,'AWR_1642/awr_1642.json',
                                          mode_name=mode_name,
                                          parent_h_node=all_actors.actors[actor_radar_name].h_node,
                                          scale_pattern=1.0)
rComp = RssPy.ResponseComposition.INDIVIDUAL
pem_api_manager.isOK(pem.setTxResponseComposition(ant_device.modes[mode_name], rComp))

center_freq = ant_device.waveforms[mode_name].center_freq

# assign modes to devices
print(pem.listGPUs())

# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = 2.99792458e8 / center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density

sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.ray_shoot_method = 'sbr'
sim_options.gpu_quota = 0.9  # set the GPU memory quota to use, between 0 and 1.0
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
which_mode = ant_device.modes['mode1']  # tell it which mode we want to get respones from
ant_device.waveforms['mode1'].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device.waveforms['mode1'].vel_domain
rng_domain = ant_device.waveforms['mode1'].rng_domain
freq_domain = ant_device.waveforms['mode1'].freq_domain
pulse_domain = ant_device.waveforms['mode1'].pulse_domain


fps = 30;
dt = 1 / fps
T = 10;
numFrames = int(T / dt)
responses = []
cameraImages = []
# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate

debug_camera = DebuggingCamera(hMode=ant_device.modes['mode1'],
                                display_mode='coating',
                                output_size=(512,512),
                                background_color=255,
                                frame_rate=fps)
# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=fps,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)


print('running simulation...')
for iFrame in tqdm(range(numFrames), disable=True):
    time = iFrame * dt
    # update all coordinate systems
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes['mode1'], RssPy.ResponseType.RANGE_DOPPLER)
    # calculate response in dB to overlay in pyvista plotter
    imData = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))
    # modeler.mpl_ax_handle.set_data(imData)  # update pyvista matplotlib plot

    # exporting radar camera images
    if export_debug:
        debug_logs.write_scene_summary(file_name=f'out.json')
    debug_camera.generate_image(store_images=False,up_direction='-Y')

    
    modeler.update_frame(plot_data=debug_camera.raw_image)
    if iFrame==0:
        modeler.pl.show_grid()
modeler.close()


