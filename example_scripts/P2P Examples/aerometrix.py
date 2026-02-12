#
# Copyright ANSYS. All rights reserved.
#


import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import os
import sys
import time as walltime

from aerometrix_paths import Paths
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaDevice, Waveform, AntennaArray
from pem_utilities.router import Pick_Path
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.utils import create_pack_of_peds, create_traffic, create_traffic_grid, create_crowd
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 



# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()
all_example_paths = Paths()

# simulation parameters
max_num_refl = 3
max_num_trans = 2
ray_density = 0.01
ray_batches = 25

# Perceive EM waveform for Comms
center_freq = 30e9
num_freqs = 256
bandwidth = 300e6
cpi_duration = 0.9e-3
num_pulse_CPI = 80

export_debug = True  #use this to export the movie at the end of the simulation (.gif showing radar camera and response)

is_tx_on = {'bs1':False, 'bs2': True, 'bs3': False, 'quadcopter': False}
is_traffic_on = False
is_crowd_on = True
is_pedestrian_on = False
# what to display


debug_logs = DebuggingLogs(output_directory=paths.output)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

mat_manager = MaterialManager()
all_actors = Actors()  # all actors, using the same material library for everyone

# these are all the objects that are part of the scene


all_terrain = ['Tile_+026_+020', 'Tile_+026_+021',
               'Tile_+027_+019', 'Tile_+027_+020', 'Tile_+027_+021', 'Tile_+027_+022',
               'Tile_+028_+020', 'Tile_+028_+021', 'Tile_+028_+022', 'Tile_+028_+023',
               'Tile_+029_+021', 'Tile_+029_+022']
terrain_path = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Scripting\RFSX_2\scenarios\Denver_20mm_res\Denver_20mm_res'

all_terrain_meshes = []
for each in all_terrain:
    if not os.path.exists(os.path.join(terrain_path, each, each + '.obj')):
        print(f'Error: {os.path.join(terrain_path, each, each + ".obj")} does not exist')
        print('Aerometrix Dataset required to run this script')
        # issue an exeception and exit
        raise FileNotFoundError
    import_file_name = os.path.join(terrain_path, each, each + '.obj')
    terrain_name = all_actors.add_actor(name='terrain1',
                                        filename=import_file_name,
                                        include_texture=True, map_texture_to_material=False)
    all_terrain_meshes.extend(all_actors.actors[terrain_name].get_mesh())


# uncomment this to pick a new path
# for_creating_new = Pick_Path()
# for_creating_new.custom_path(mesh_list=all_terrain_meshes, pre_picked=None, speed=1, upsample_selection=101, z_offset=0,
#                              snap_to_surface=True)



# drone
path_picked_drone = Pick_Path()
path_picked_drone.custom_path(mesh_list=all_terrain_meshes, pre_picked=all_example_paths.pre_picked_drone, speed=8,
                              upsample_selection=101,
                              z_offset=9, snap_to_surface=False)
actor_quadcopter_name = all_actors.add_actor(name='quadcopter',
                                             filename=os.path.join(paths.models,'Quadcopter/Quadcopter.json'),
                                             target_ray_spacing=0.01)
all_actors.actors[actor_quadcopter_name].use_linear_velocity_equation_update = False

# cars turning corner
traffic_dict = {}
if is_traffic_on:
    traffic_dict.update(create_traffic(all_actors,
                                       traffic_dict=traffic_dict,
                                       speed=10,
                                       num_cars=3,
                                       vehicle_separation=10,
                                       add_randomness=True,
                                       pre_picked=all_example_paths.pre_picked_veh1,
                                       surface_meshes=all_terrain_meshes,
                                       name='traffic_group1'))
    # cars coming straight down main street
    traffic_dict.update(create_traffic_grid(all_actors,traffic_dict=traffic_dict,
                                            num_cars_per_column=3,
                                            num_car_to_driver_side=2,
                                            num_cars_to_passenger_side=3,
                                            speed=9,
                                            vehicle_separation=10,
                                            horizontal_seperation=3.5,
                                            add_randomness=True,
                                            pre_picked=all_example_paths.pre_picked_veh2,
                                            surface_meshes=all_terrain_meshes,
                                            name='traffic_group2'))

    # # cars going on horizontal road
    traffic_dict.update(create_traffic_grid(all_actors,traffic_dict=traffic_dict,
                                            num_cars_per_column=3,
                                            num_car_to_driver_side=0,
                                            num_cars_to_passenger_side=2,
                                            speed=5,
                                            vehicle_separation=10,
                                            horizontal_seperation=3.5,
                                            add_randomness=True,
                                            pre_picked=all_example_paths.pre_picked_veh3,
                                            surface_meshes=all_terrain_meshes,
                                            name='traffic_group3'))

    # cars starting in main instersection, then going down main street
    traffic_dict.update(create_traffic_grid(all_actors,traffic_dict=traffic_dict,
                                            num_cars_per_column=3,
                                            num_car_to_driver_side=4,
                                            num_cars_to_passenger_side=5,
                                            speed=7,
                                            vehicle_separation=10,
                                            horizontal_seperation=3.5,
                                            add_randomness=True,
                                            pre_picked=all_example_paths.pre_picked_veh4,
                                            surface_meshes=all_terrain_meshes,
                                            name='traffic_group4'))


# ########################################################################################################################
# #
# # PEDESTRIANS
# #
# #####################################################################################################################
#
crowd_names = []
if is_crowd_on:
    crowd_names += create_crowd(all_actors,num_peds=30,xyz=[510.06649326, 7.02949524, 1595.01401898],radius=20,
                                 name='crowd_group1')
    crowd_names += create_crowd(all_actors,num_peds=20,xyz=[443.36332947,   69.32958048, 1595.74324645],radius=10,
                                 name='crowd_group2')
    crowd_names += create_crowd(all_actors,num_peds=10,xyz=[408, 7.02949524,1589],radius=10,
                                 name='crowd_group3')
    crowd_names += create_crowd(all_actors,num_peds=10,xyz=[339.57702861,  -42.7196917 , 1589.81064917],radius=5,
                                 name='crowd_group4')
    crowd_names += create_crowd(all_actors,num_peds=5,xyz=[468.41765077,  -15.80263521, 1595.02665343],radius=7,
                                 name='crowd_group5')
    crowd_names += create_crowd(all_actors,num_peds=5,xyz=[474.58,  52.5681, 1596.1],radius=4,
                                 name='crowd_group6')
peds_dict = {}

if is_pedestrian_on:
    # # create a group of pedestrians, all moving in the same direction with a random distribution around the path above
    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=8, pre_picked=all_example_paths.pre_picked_ped1,
                                         surface_meshes=all_terrain_meshes,name='ped_group1'))

    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=9, pre_picked=all_example_paths.pre_picked_ped2,
                                         surface_meshes=all_terrain_meshes,name='ped_group2'))

    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=5, pre_picked=all_example_paths.pre_picked_ped3,
                                         surface_meshes=all_terrain_meshes,name='ped_group3'))

    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=6, pre_picked=all_example_paths.pre_picked_ped4,
                                         surface_meshes=all_terrain_meshes,name='ped_group4'))

    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=7, pre_picked=all_example_paths.pre_picked_ped5,
                                         surface_meshes=all_terrain_meshes,name='ped_group5'))

    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=8, pre_picked=all_example_paths.pre_picked_ped6,
                                         surface_meshes=all_terrain_meshes,name='ped_group6'))

    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=6, pre_picked=all_example_paths.pre_picked_ped7,
                                         surface_meshes=all_terrain_meshes,name='ped_group7'))

    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=5, pre_picked=all_example_paths.pre_picked_ped8,
                                         surface_meshes=all_terrain_meshes,name='ped_group8'))

    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=5, pre_picked=all_example_paths.pre_picked_ped9,
                                         surface_meshes=all_terrain_meshes,name='ped_group9'))
    peds_dict.update(create_pack_of_peds(all_actors, peds_dict, num_peds=5, pre_picked=all_example_paths.pre_picked_ped10,
                                         surface_meshes=all_terrain_meshes,name='ped_group10'))

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
# to load the farfield pattern data as a mesh, and create a pyvista actor that we can later visualize
# once the antennas are added, we can add the mode to the device

# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more flexibility
# and control over the parameters, without having to create/modify a json file
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "RangeDoppler",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_pulse_CPI,
    "tx_multiplex": "SIMULTANEOUS",
    "mode_delay": "CENTER_CHIRP"}
pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
mode_name_comms = 'mode_comms'
waveform = Waveform(waveform_dict=waveform_dict)

#######################################################################################################################
# Setup Radar Antennas
#######################################################################################################################
mode_name_radar = 'mode1'
ant_device_radar = AntennaDevice('example_1tx_1rx.json', parent_h_node=all_actors.actors[actor_quadcopter_name].h_node)
ant_device_radar.initialize_mode(mode_name=mode_name_radar)
ant_device_radar.coord_sys.pos = (.1, 0., 0)
ant_device_radar.coord_sys.rot = euler_to_rot(phi=10, theta=30, order='zyz', deg=True)
ant_device_radar.coord_sys.update()
ant_device_radar.add_antennas(mode_name=mode_name_radar, load_pattern_as_mesh=True, scale_pattern=.2)
ant_device_radar.add_mode(mode_name=mode_name_radar)
# If we want to visualize the antennas, we can add them to the scene as actors. This is not required for the simulation
# but is used only for visualization purposes.
for each in ant_device_radar.all_antennas_properties:
    name = all_actors.add_actor(name=each,actor=ant_device_radar.all_antennas_properties[each]['Actor'])

#######################################################################################################################
# Setup Comms Antennas
#######################################################################################################################


######################
# SETUP  BASE STATIONS (32 element array acting as Tx)
######################
if is_tx_on['bs1']:
    bs1_root = all_actors.add_actor(name='bs1')
    bs1 = AntennaArray(name='bs1_array',waveform=waveform,mode_name=mode_name_comms,
                        beamwidth_H=140,beamwidth_V=120,polarization='V',
                        rx_shape=0,tx_shape=[8, 4],spacing_wl_x=0.5,spacing_wl_y=0.5,normal='x',
                        load_pattern_as_mesh=True,scale_pattern=5,
                        parent_h_node=all_actors.actors[bs1_root].h_node)
    bs1_pos = [412.6782,-79,1624.]
    bs1_rot = euler_to_rot(phi=90, theta=15, order='zyz', deg=True)
    all_actors.actors[bs1_root].coord_sys.pos = bs1_pos
    all_actors.actors[bs1_root].coord_sys.rot = bs1_rot
    all_actors.actors[bs1_root].coord_sys.update()
    for each in bs1.antenna_device.all_antennas_properties:
        name = all_actors.add_actor(name=each, actor=bs1.antenna_device.all_antennas_properties[each]['Actor'])
    rComp = RssPy.ResponseComposition.INDIVIDUAL
    pem_api_manager.isOK(pem.setTxResponseComposition(bs1.antenna_device.modes[mode_name_comms], rComp))
if is_tx_on['bs2']:
    bs2_root = all_actors.add_actor(name='bs2')
    bs2 = AntennaArray(name='bs2_array',waveform=waveform,mode_name=mode_name_comms,
                        beamwidth_H=140,beamwidth_V=120,polarization='V',
                        rx_shape=0,tx_shape=[8, 4],spacing_wl_x=0.5,spacing_wl_y=0.5,normal='x',
                        load_pattern_as_mesh=False,scale_pattern=5,
                        parent_h_node=all_actors.actors[bs2_root].h_node)
    bs2_pos = [448.6,31.3,1612.693]
    bs2_rot = euler_to_rot(phi=-10, theta=15, order='zyz', deg=True)
    all_actors.actors[bs2_root].coord_sys.pos = bs2_pos
    all_actors.actors[bs2_root].coord_sys.rot = bs2_rot
    all_actors.actors[bs2_root].coord_sys.update()
    for each in bs2.antenna_device.all_antennas_properties:
        name = all_actors.add_actor(name=each, actor=bs2.antenna_device.all_antennas_properties[each]['Actor'])
    rComp = RssPy.ResponseComposition.INDIVIDUAL
    pem_api_manager.isOK(pem.setTxResponseComposition(bs2.antenna_device.modes[mode_name_comms], rComp))
if is_tx_on['bs3']:
    bs3_root = all_actors.add_actor(name='bs3')
    bs3= AntennaArray(name='bs3_array',waveform=waveform,mode_name=mode_name_comms,
                        beamwidth_H=140,beamwidth_V=120,polarization='V',
                        rx_shape=0,tx_shape=[8, 4],spacing_wl_x=0.5,spacing_wl_y=0.5,normal='x',
                        load_pattern_as_mesh=True,scale_pattern=5,
                        parent_h_node=all_actors.actors[bs3_root].h_node)
    bs3_pos = [305,-11,1604]
    bs3_rot = euler_to_rot(phi=-10, theta=15, order='zyz', deg=True)
    all_actors.actors[bs3_root].coord_sys.pos = bs3_pos
    all_actors.actors[bs3_root].coord_sys.rot = bs3_rot
    all_actors.actors[bs3_root].coord_sys.update()
    for each in bs3.antenna_device.all_antennas_properties:
        name = all_actors.add_actor(name=each, actor=bs3.antenna_device.all_antennas_properties[each]['Actor'])
    rComp = RssPy.ResponseComposition.INDIVIDUAL
    pem_api_manager.isOK(pem.setTxResponseComposition(bs3.antenna_device.modes[mode_name_comms], rComp))
######################
# Tx
#####################
if is_tx_on['quadcopter']:
    # initialize the antenna device, one for Tx, one for Rx
    ant_device_tx = AntennaDevice(parent_h_node=all_actors.actors[actor_quadcopter_name].h_node)
    ant_device_tx.initialize_device()
    ant_device_tx.waveforms[mode_name_comms] = waveform

    h_mode = RssPy.RadarMode()
    ant_device_tx.modes[mode_name_comms] = h_mode
    pem_core.isOK(pem.addRadarMode(h_mode, ant_device_tx.h_device))
    antennas_dict = {}
    ant_type_tx = {
        "type": "ffd",
        "file_path": "dipole.ffd",
        "operation_mode": "tx",
        "position": [0, 0, .1]
    } # position is offset location from where antenna device is placed

    antennas_dict["Tx"] = ant_type_tx
    ant_device_tx.add_antennas(mode_name=mode_name_comms,
                                   load_pattern_as_mesh=True,
                                   scale_pattern=.1,
                                   antennas_dict=antennas_dict)
    ant_device_tx.set_mode_active(mode_name_comms)
    ant_device_tx.add_mode(mode_name_comms)

    # position of each antenna device
    ant_device_tx.coord_sys.update()


    # If we want to visualize the antennas, we can add them to the scene as actors. This is not required for the simulation
    # but is used only for visualization purposes.
    for each in ant_device_tx.all_antennas_properties:
        name = all_actors.add_actor(name=each, actor=ant_device_tx.all_antennas_properties[each]['Actor'])
    rComp = RssPy.ResponseComposition.INDIVIDUAL
    pem_core.isOK(pem.setTxResponseComposition(ant_device_tx.modes[mode_name_comms], rComp))
######################
# Rx, add one to every dynamic object in the scene
####################

all_antenna_rx_actors = {}

def add_antenna_to_actors(actor_h_node,pos=[0,0,1.5],scale_pattern=0.5,name='Rx'):
    antennas_dict = {}
    ant_type_rx = {
        "type": "ffd",
        "file_path": "dipole.ffd",
        "operation_mode": "rx",
        "position": pos
    }  # position is offset location from where antenna device is placed


    antennas_dict["Rx"] = ant_type_rx
    ant_device_rx = AntennaDevice(parent_h_node=actor_h_node)
    ant_device_rx.initialize_device()
    ant_device_rx.waveforms[mode_name_comms] = waveform

    h_mode = RssPy.RadarMode()
    ant_device_rx.modes[mode_name_comms] = h_mode
    pem_core.isOK(pem.addRadarMode(h_mode, ant_device_rx.h_device))

    ant_device_rx.add_antennas(mode_name=mode_name_comms,
                                   load_pattern_as_mesh=True,
                                   scale_pattern=scale_pattern,
                                   antennas_dict=antennas_dict)
    ant_device_rx.set_mode_active(mode_name_comms)
    ant_device_rx.add_mode(mode_name_comms)

    # position of each antenna device
    ant_device_rx.coord_sys.update()

    # If we want to visualize the antennas, we can add them to the scene as actors. This is not required for the simulation
    # but is used only for visualization purposes.
    for each in ant_device_rx.all_antennas_properties:
        rx_name = all_actors.add_actor(name=each, actor=ant_device_rx.all_antennas_properties[each]['Actor'])

    if is_tx_on['quadcopter']:
        pem_core.isOK(pem.setDoP2PCoupling(ant_device_tx.h_node_platform, ant_device_rx.h_node_platform, True))
    if is_tx_on['bs1']:
        pem_core.isOK(pem.setDoP2PCoupling(bs1.antenna_device.h_node_platform, ant_device_rx.h_node_platform, True))
    if is_tx_on['bs2']:
        pem_core.isOK(pem.setDoP2PCoupling(bs2.antenna_device.h_node_platform, ant_device_rx.h_node_platform, True))
    if is_tx_on['bs3']:
        pem_core.isOK(pem.setDoP2PCoupling(bs3.antenna_device.h_node_platform, ant_device_rx.h_node_platform, True))

    # check if name is in all_antenna_rx_actors, if not add it. if it is increment the string used for the key
    orig_name = name
    while name in all_antenna_rx_actors:
        name = orig_name + '1'
    all_antenna_rx_actors[name] = ant_device_rx

# initialize the antenna device, one for Rx for every actor in the scene

add_antenna_to_actors(all_actors.actors[actor_quadcopter_name].h_node,pos=[0,0,0.1],scale_pattern=.1,name='quadcopter')

for vehicle in traffic_dict:
    add_antenna_to_actors(all_actors.actors[vehicle].h_node,pos=[0,0,1.5],name=vehicle)

for ped in peds_dict:
    which_hand = np.random.choice(['mixamorig_LeftHand', 'mixamorig_RightHand'])
    add_antenna_to_actors(all_actors.actors[ped].parts[which_hand].h_node,pos=[.1,0.1,.1],scale_pattern=0.1,name=ped)

for ped in crowd_names:
    which_hand = np.random.choice(['lhand','rhand'])
    add_antenna_to_actors(all_actors.actors[ped].parts[which_hand].h_node, pos=[.1, 0.15, .1],scale_pattern=0.1,name=ped)

# assign modes to devices
print(pem.listGPUs())
devIDs = [0];
devQuotas = [0.8];  # limit RTR to use 80% of available gpu memory
pem_core.isOK(pem.setGPUDevices(devIDs, devQuotas))
maxNumRayBatches = 25
pem_core.isOK(pem.autoConfigureSimulation(maxNumRayBatches))
pem_core.isOK(pem.setPrivateKey("FieldOfView", "360"))

# initialize solver settings
pem_core.isOK(pem.setMaxNumRefl(max_num_refl))
pem_core.isOK(pem.setMaxNumTrans(max_num_trans))
pem_core.isOK(pem.setTargetRayDensity(ray_density))  # global ray density setting

# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(pem.getLastWarnings())

# display setup
#print(pem.reportSettings())

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes

if is_tx_on['quadcopter']:
    which_mode = ant_device_tx.modes[mode_name_comms]  # tell it which mode we want to get respones from
    which_waveform = ant_device_tx.waveforms[mode_name_comms]  # tell it which waveform we want to get respones from
elif is_tx_on['bs1']:
    which_mode = bs1.antenna_device.modes[mode_name_comms]  # tell it which mode we want to get respones from
    which_waveform = bs1.antenna_device.waveforms[mode_name_comms]  # tell it which waveform we want to get respones from
elif is_tx_on['bs2']:
    which_mode = bs2.antenna_device.modes[mode_name_comms]
    which_waveform = bs2.antenna_device.waveforms[mode_name_comms]
elif is_tx_on['bs3']:
    which_mode = bs3.antenna_device.modes[mode_name_comms]
    which_waveform = bs3.antenna_device.waveforms[mode_name_comms]
else:
    which_mode = ant_device_radar.modes[mode_name_comms]  # tell it which mode we want to get respones from
    which_waveform = ant_device_radar.waveforms[mode_name_comms]

which_waveform.get_response_domains(which_mode)
vel_domain = which_waveform.vel_domain
rng_domain = which_waveform.rng_domain
freq_domain = which_waveform.freq_domain
pulse_domain = which_waveform.pulse_domain

fps = 20;
dt = 1 / fps
T = 25
numFrames = int(T / dt)

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=which_mode,
                                   display_mode='normal',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

responses = []
output_movie_name = os.path.join(output_path, 'out_vis.mp4')
# bs1_root
# 'radar'
# actor_quadcopter_name
# camera_orientation = 'follow4',
if is_tx_on['quadcopter']:
    camera_attachment = actor_quadcopter_name
elif is_tx_on['bs1']:
    camera_attachment = bs1_root
elif is_tx_on['bs2']:
    camera_attachment = bs2_root
elif is_tx_on['bs3']:
    camera_attachment = bs3_root
else:
    camera_attachment = None

camera_orientation = 'radar'
# # create a pyvista plotter
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=rng_domain, vel_domain=vel_domain,
                             fps=fps,
                             output_video_size = (2*1024,768*2),
                             camera_attachment=camera_attachment,
                             camera_orientation=camera_orientation,
                             output_movie_name=output_movie_name)


print('running simulation...')
total_time_solve = 0
counter=0
average_solve_time_per_frame =0
for iFrame in tqdm(range(numFrames)):
    time = iFrame * dt
    # update all coordinate systems
    for actor in all_actors.actors:
        # Actors can have actor_type value, this will handle sub-parts of the actors. For example, if we have a vehicle
        # it will update the tires rotation and velocity, and the vehicle linear velocity based on its orientation
        # if actor is other, it will use the lin/rot/ang values to update the position of the actor
        if actor in traffic_dict:
            all_actors.actors[actor].coord_sys.pos = traffic_dict[actor].pos_func(time)
            all_actors.actors[actor].coord_sys.rot = traffic_dict[actor].rot_func(time)
        if actor == actor_quadcopter_name:
            all_actors.actors[actor].coord_sys.pos = path_picked_drone.pos_func(time)
            all_actors.actors[actor].coord_sys.rot = path_picked_drone.rot_func(time)
        if actor in peds_dict:
            all_actors.actors[actor].coord_sys.pos = peds_dict[actor].pos_func(time)
            all_actors.actors[actor].coord_sys.rot = peds_dict[actor].rot_func(time)
        all_actors.actors[actor].update_actor(time=time)








    # remove scalar bars, they start to get in the way with lots of things going on
    all_scalar_bars = list(modeler.pl.scalar_bars.keys())
    for each in all_scalar_bars:
        modeler.pl.remove_scalar_bar(each)

    modeler.update_frame()
modeler.close()

# post-process images into gif

if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')
    debug_camera.write_response_to_gif(responses,file_name='response.gif',rng_domain=rng_domain,vel_domain=vel_domain)
    debug_camera.write_camera_and_response_to_gif(responses,
                                                  file_name='camera_and_response.gif',
                                                  rng_domain=rng_domain,
                                                  vel_domain=vel_domain)

print(f'Average solve time per frame: {average_solve_time_per_frame} seconds')
print(f'Total solve time: {total_time_solve} seconds for {numFrames} frames')
