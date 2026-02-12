#
# Copyright ANSYS. All rights reserved.
#
#######################################
# Example of P2P coupling between a pedestrian and a building using multiple animated people

import copy
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import pyvista as pv
import os
import sys
import scipy
from scipy.interpolate import interp1d

from pem_utilities.animation_selector_CMU import select_CMU_animation
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaDevice
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.router import Pick_Path
from pem_utilities.simulation_options import SimulationOptions


from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_density = .2

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

# The path the person takes walking through the building can be defined by the user or by a predefined path. If
# custom_user_path is set to True, the user will be prompted to define the path by clicking on the screen. If False,
# the path will be loaded from a predefined file. The path is saved as a numpy array in the output directory.
custom_user_path = False

export_debug = True

debug_logs = DebuggingLogs(output_directory=paths.output)

# channel_1_path ='../output/channel_1.npy'
# channel_2_path ='../output/channel_2.npy'
# path to predefined path file, where the person walks

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()  # all actors, using the same material library for everyone

#add a multi part actor from json
usd_file_example = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\Kitchen_set\Kitchen_set\Kitchen_set.usd'
# usd_file_example = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\UsdSkelExamples\HumanFemale\HumanFemale.walk.usd'
# usd_file_example = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\tokyo.usd'
if not os.path.exists(usd_file_example):
    path_to_example = 'https://openusd.org/release/dl_kitchen_set.html'
    raise ValueError(f'Please provide a valid USD file, example available at {path_to_example}')

house_name = all_actors.add_actor(name='house',
                                  filename=usd_file_example,
                                  mat_idx=mat_manager.get_index('concrete'),
                                  transparency=0.7, scale_mesh=0.01)  #

# debug_logs.write_scene_to_proto(file_name='scene_tokyo_only.proto')
# debug_logs.read_scene_from_proto(file_name=os.path.join(output_path,'scene_tokyo_only.proto'))
if custom_user_path:
    pre_picked_path = None
else:
    pre_picked_path = [[3.48324520e+00, -2.40016279e-01, 0],
                       [2.26093793e+00, -6.10527518e-01, 0],
                       [2.67025077e-01, -8.40076875e-01, 0],
                       [-4.33251947e-01, -2.04161691e+00, 0],
                       [4.31067728e-02, -2.27462083e+00, 0]]
path_picked_person = Pick_Path()
path_picked_person.custom_path(mesh_list=all_actors.actors[house_name].get_mesh(), pre_picked=pre_picked_path,
                               speed=1.0,
                               snap_to_surface=True)

#add a multi part actor from dae
ped1_name = all_actors.add_actor(filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'), target_ray_spacing=0.1,
                                 scale_mesh=1)
all_actors.actors[ped1_name].velocity_mag = 1.0
all_actors.actors[ped1_name].use_linear_velocity_equation_update = False  #update position based on where the person is

fps = 15;
dt = 1 / fps
T = 3;
numFrames = int(T / dt)
#Original template for motion was created

timestamps = np.linspace(0, T, numFrames)

#interp_func_pos = scipy.interpolate.interp1d(numFrames, all_positions_pedestrian, axis=0, assume_sorted=True)


######################
# Tx on walking user
####################
#node_to_attach_tx = all_scene_actors['pedestrian'].parts['mixamorig_LeftHand'].h_node
node_to_attach_tx = all_actors.actors[ped1_name].h_node

ant_device_tx = AntennaDevice(file_name='Indoor_P2P_1tx_6p0GHZ.json',
                              parent_h_node=node_to_attach_tx,
                              all_actors=all_actors)
ant_device_tx.initialize_mode(mode_name='mode1')
ant_device_tx.coord_sys.rot = euler_to_rot(phi=0, theta=0, order='zyz', deg=True)
#ant_device_tx.coord_sys.pos = (-0.2,0.0,0.2) for hand attachment
#for body attachment:
ant_device_tx.coord_sys.pos = (0, 0.0, 1)
ant_device_tx.coord_sys.update()
ant_device_tx.add_antennas(mode_name='mode1', load_pattern_as_mesh=True, scale_pattern=0.3)
ant_device_tx.add_mode(mode_name='mode1')

######################
# Rx1 and Rx2
#####################
# we can set the private key for the device, this is not required, but it is used to set the field of view for the
# antenna. without this key, the shoot mode is only the hemisphere in the +X direction
pem_api_manager.isOK(pem.setPrivateKey("FieldOfView", "360"))

ant_device_rx1 = AntennaDevice(file_name='Indoor_P2P_1rx_6p0GHZ.json',
                               parent_h_node=all_actors.actors[house_name].h_node,
                               all_actors=all_actors)
ant_device_rx1.initialize_mode(mode_name='mode1')
ant_device_rx1.coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
# put this antenna on the roof
ant_device_rx1.coord_sys.pos = (3.6, -2.5, 0.9)
ant_device_rx1.coord_sys.update()
ant_device_rx1.add_antennas(mode_name='mode1', load_pattern_as_mesh=True, scale_pattern=0.4)
ant_device_rx1.add_mode(mode_name='mode1')


ant_device_rx2 = AntennaDevice(file_name='Indoor_P2P_2rx_6p0GHZ.json',
                               parent_h_node=all_actors.actors[house_name].h_node,
                               all_actors=all_actors)
ant_device_rx2.initialize_mode(mode_name='mode1')
ant_device_rx2.coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)
# put this antenna on the roof
ant_device_rx2.coord_sys.pos = (-2.0, -2.5, 0.9)
ant_device_rx2.coord_sys.update()
ant_device_rx2.add_antennas(mode_name='mode1', load_pattern_as_mesh=True, scale_pattern=0.4)
ant_device_rx2.add_mode(mode_name='mode1')

rComp = RssPy.ResponseComposition.INDIVIDUAL
pem_api_manager.isOK(pem.setTxResponseComposition(ant_device_tx.modes['mode1'], rComp))

pem_api_manager.isOK(pem.setDoP2PCoupling(ant_device_tx.h_node_platform, ant_device_rx1.h_node_platform, True))
pem_api_manager.isOK(pem.setDoP2PCoupling(ant_device_tx.h_node_platform, ant_device_rx2.h_node_platform, True))

print(pem.listGPUs())
# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    lambda_center = 2.99792458e8 / ant_device_tx.waveforms['mode1'].center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density


sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()


f_low = ant_device_tx.waveforms['mode1'].center_freq - (ant_device_tx.waveforms['mode1'].bandwidth) / 2
f_high = ant_device_tx.waveforms['mode1'].center_freq + (ant_device_tx.waveforms['mode1'].bandwidth) / 2
num_freq_samples = ant_device_tx.waveforms['mode1'].num_freq_samples

freq_domain = np.linspace(f_low, f_high, num=num_freq_samples)

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes['mode1'],
                                   display_mode='normal',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=None,
                             vel_domain=None, overlay_results=False,
                             fps=fps,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)

modeler.pl.show_grid()
#modeler.pl.show()
meshes = modeler.pl.meshes

all_results = []
all_results2 = []
print('running simulation...')

for iFrame in tqdm(range(numFrames), disable=True):
    time = iFrame * dt
    # update all coordinate systems
    for actor in all_actors.actors:
        # Actors can have actor_type value, this will handle sub-parts of the actors. For example, if we have a vehicle
        # it will update the tires rotation and velocity, and the vehicle linear velocity based on its orientation
        # if actor is other, it will use the lin/rot/ang values to update the position of the actor
        if actor == ped1_name:
            all_actors.actors[actor].coord_sys.pos = path_picked_person.pos_func(time)
            all_actors.actors[actor].coord_sys.rot = path_picked_person.rot_func(time)
        all_actors.actors[actor].update_actor(time=time)

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes['mode1'],
                                              ant_device_rx1.modes['mode1'],
                                              RssPy.ResponseType.FREQ_PULSE)
    (ret2, response2) = pem.retrieveP2PResponse(ant_device_tx.modes['mode1'],
                                                ant_device_rx2.modes['mode1'],
                                                RssPy.ResponseType.FREQ_PULSE)
    response = np.array(response)[0][0]
    response2 = np.array(response2)[0][0]

    all_results.append(response)
    all_results2.append(response2)

    # exporting radar camera images
    if export_debug:
        if iFrame == 0 or iFrame == numFrames - 1:
            debug_logs.write_scene_summary(file_name=f'out_{iFrame}.json')
        debug_camera.generate_image()

    f, ax = plt.subplots(tight_layout=True)
    plt.plot(freq_domain / 1e9, 20 * np.log10(abs((response[0]))), color='red', label='Kitchen Room Rx')
    plt.plot(freq_domain / 1e9, 20 * np.log10(np.abs((response2[0]))), color='blue', label='Office Rx')
    plt.xlabel('Frequency [GHz]', fontsize=15)
    plt.ylabel('S21[dB]', fontsize=15)
    plt.legend(loc='upper right')
    ax.set_ylim(-100, -30)
    h_chart = pv.ChartMPL(f, size=(0.275, 0.375), loc=(0.7, 0.6))
    modeler.pl.add_chart(h_chart)
    modeler.update_frame()
    plt.clf()
    plt.close()

modeler.close()
# np.save(channel_1_path,all_results[0])
# np.save(channel_2_path,all_results2[0])

print('Done')

#add the code above and run the scene first. Zoom to where you want to go to manually then exit the scene.
# Copy and paste the array shown so that you don't have to manually zoom everytime
#modeler.pl.camera_position = [(-6.628138388036804, -13.533762431546673, 4.416518949264247),
#(4.839783862233162, -4.109425738453865, 1.3774030953645706),
#(0.14661084374231628, 0.13749220738163118, 0.9795923404184482)]
#modeler.pl.show()

if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')
