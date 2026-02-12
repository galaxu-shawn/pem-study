#
# Copyright ANSYS. All rights reserved.
#
import copy
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import os
import sys

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaDevice, enable_coupling
from pem_utilities.beamformer import FF_Fields_Beamformer
from pem_utilities.post_processing import channel_capacity
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 

# simulation parameters
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_spacing = 0.25  # global ray spacing in meters

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

export_debug = True #use this to export the movie at the end of the simulation (.gif showing radar camera and response)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()



vehicle1_name = all_actors.add_actor(name='vehicle1',
                                     filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json'),
                                     transparency=0.5)
# add a multi part actor from json
vehicle2_name = all_actors.add_actor(name='vehicle2',filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json'))

# Another special type of multi-part actor is a pedestrian defined as a collada file. This will load the pedestrian and
# animate the body to walk. A an argument 'target_ray_spacing' can be used to create an adaptive shoot grid that will
# use a fixed ray spacing, in meters, no matter where the actor is in the scene
pedestrian_name = all_actors.add_actor(name='pedestrian',
                                       filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'),
                                       target_ray_spacing=0.1)


# initialize coordinate systems for all actors, for general actors we can use the intialize_cs(), and then set the
# initial positions of each actor. Any value that is not set, will be set to zero. We can set pos/lin/rot/ang for each
# when we initialize the cs, it will create a node, coord_sys.hNode. If we want to create a child node, we can set the
# parent_h_node to the parent node,


all_actors.actors[vehicle1_name].coord_sys.pos = (100.,-50, 0.)
all_actors.actors[vehicle1_name].velocity_mag = 10.
# for the multi part actor of actor_type='vehicle' we will use a general velocity, where the magnitude is only defined
# the rotation will be to define the linear velocity vector
all_actors.actors[vehicle1_name].coord_sys.rot = euler_to_rot(phi=90,theta=0, order='zyz',deg=True)


all_actors.actors[vehicle2_name].coord_sys.pos = (0,0,0.)
all_actors.actors[vehicle2_name].velocity_mag = 2.
# for the multi part actor of actor_type='vehicle' we will use a general velocity, where the magnitude is only defined
# the rotation will be to define the linear velocity vector
all_actors.actors[vehicle2_name].coord_sys.rot = euler_to_rot(phi=0,theta=0, order='zyz',deg=True)


# velocity magnitude can be used with pedestrian loaded from dae, where rotation modifies the linear velocity vector
all_actors.actors[pedestrian_name].coord_sys.pos = (10.,5, 0.)
all_actors.actors[pedestrian_name].velocity_mag = 1.0
all_actors.actors[pedestrian_name].coord_sys.rot = euler_to_rot(phi=180,theta=0, order='zyz',deg=True)

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


######################
# Tx
#####################

mode_name = 'mode1'
ant_device_tx = AntennaDevice('example_1tx_30GHz_dipole.json',
                              parent_h_node=all_actors.actors[vehicle1_name].h_node,
                              all_actors=all_actors)

ant_device_tx.initialize_mode(mode_name=mode_name)
# put this antenna on the roof
ant_device_tx.coord_sys.pos = (3,0.,1)
ant_device_tx.coord_sys.update()
ant_device_tx.add_antennas(mode_name=mode_name,load_pattern_as_mesh=True,scale_pattern=10)
ant_device_tx.add_mode(mode_name=mode_name)

######################
# Rx
####################
ant_device_rx = AntennaDevice('example_16rx_30GHz.json',
                              parent_h_node=all_actors.actors[vehicle2_name].coord_sys.h_node,
                              all_actors=all_actors)
ant_device_rx.initialize_mode(mode_name=mode_name)
ant_device_rx.coord_sys.pos = (5,0.,1)
ant_device_rx.coord_sys.update()
ant_device_rx.add_antennas(mode_name=mode_name,load_pattern_as_mesh=True,scale_pattern=10)
ant_device_rx.add_mode(mode_name=mode_name)

# If we want to visualize the beam formed result, we can add them to the scene as actors.
# This is not required for the simulation
# but is used only for visualization purposes. Requried that load_pattern_as_mesh is set to True
fields_dict = {}
for each in ant_device_rx.all_antennas_properties:
    fields_dict[each] = ant_device_rx.all_antennas_properties[each]['Fields'] # used for beamforming

# If we want to beamform the fields, we need to load the fields into this beamformer object. It will load all the Rx
# antennas and their fields. When we run a simulation we will determine what beam weights are best, and the apply them
# to all these fields to get a new far field pattern for visualization. Not required to run the simulation
ff_for_beamforming = FF_Fields_Beamformer(fields_dict)

enable_coupling(mode_name,ant_device_tx, ant_device_rx)

# assign modes to devices
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


# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device_tx.modes[mode_name] # tell it which mode we want to get respones from
ant_device_tx.waveforms[mode_name].get_response_domains(which_mode)

vel_domain = ant_device_tx.waveforms[mode_name].vel_domain
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain
freq_domain = ant_device_tx.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx.waveforms[mode_name].pulse_domain


fps = 10; dt = 1/fps
T = 10; numFrames = int(T/dt)
responses = []

if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)

# # create a pyvista plotter
# I am not showing the antennas here, because I only want to show the result of the beamforming, not each indivdual
# antenna. If you want to see the antennas, you can set show_antennas=True
modeler = ModelVisualization(all_actors,
                             show_antennas=False,
                             freq_domain=freq_domain,pulse_domain=pulse_domain,
                             fps=fps,
                             overlay_results=False,
                             camera_attachment=None,
                             camera_orientation=None)

all_results = []
print('running simulation...')
for iFrame in tqdm(range(numFrames)):

    time = iFrame*dt
    # update all coordinate systems
    for actor in all_actors.actors:
        # Actors can have actor_type value, this will handle sub-parts of the actors. For example, if we have a vehicle
        # it will update the tires rotation and velocity, and the vehicle linear velocity based on its orientation
        # if actor is other, it will use the lin/rot/ang values to update the position of the actor
        all_actors.actors[actor].update_actor(time=time)

    # exporting radar camera images
    if export_debug:
        if iFrame == 0 or iFrame == numFrames - 1:
            debug_logs.write_scene_summary(file_name=f'out_{iFrame}.json')
        debug_camera.generate_image()

    # for beamforming and channel capacity, we don't need to do it for every frequency point or pulse. We can just
    # choose one frequency and pulse to do it for. Here I am choosing the first frequency and pulse, but you can choose any
    freq_idx = 0
    pulse_idx = 0

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes[mode_name],ant_device_rx.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)
    response = np.array(response)

    # beamforming is determined by applying the Maximum Ratio Transmission (MRT) weights to the received fields.
    # these weights are effectively the conjugate of the response (1 Tx -- > 16 RX). There are few other outputs
    # we can get, but we really only need teh composite weights
    weights, s_param, composite_weights = ff_for_beamforming.weighting_multi_chirp_freq(response,
                                                                                            chirp_idx=pulse_idx,
                                                                                            freq_idx=freq_idx,
                                                                                            method='MRT')

    # MRT determined what the weights should be, to beamform we just need conjugate
    source_weights = np.conj(composite_weights[:,pulse_idx,freq_idx])
    # use the weights and apply these values to each Rx antenna pattern. The result is a new far field pattern that
    # is the superpostion of all the Rx antennas with the mag/pahse applied to each antenna.
    # generate_mesh=True will create a pyvista mesh that we can visualize
    ff_for_beamforming.beamform(source_weights, generate_mesh=True, quantity='RealizedGainTotal', function='abs', freq_idx=0,scale_pattern=10)

    # probably not the most efficient way to do this, but this will copy the mesh from the beamformer
    # create a new pyvista mesh, and then transform it to the current position of the radar device
    # because the mesh is changing at each time step, we need to remove the old mesh and add the new one, I think
    # it is possible with pyvista just to update the mesh position, but this is easier and less efficient
    total_ff_mesh_copy = copy.deepcopy(ff_for_beamforming.mesh)
    if iFrame!=0:
        modeler.pl.remove_actor(ff_actor)
    which_rx_antenna = 'rx1' # put the resulting beam at one of the rx antenna locations
    T = all_actors.actors[which_rx_antenna].coord_sys.transform4x4  # current 4x4 transform
    total_ff_mesh_copy.transform(T, inplace=True)  # update positions
    ff_actor = modeler.pl.add_mesh(total_ff_mesh_copy)

    all_results.append(response)
    # channel capacity is expecting data in [num_frames, num_tx_antennas, num_rx_antennas]
    # I am not uses this channel capacity, but it is here for reference if you wanted to calculate it on the fly
    # at each time step. Lower down I calculate it for all frames at once
    chan_capacity = channel_capacity(response[:, :, pulse_idx, freq_idx], BW_Hz=10e6, temperture=290) / 1e6

    if export_debug:
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)

    modeler.update_frame()
modeler.close()

all_results = np.array(all_results)
# all_results will be in format [num_frames, num_tx_antennas, num_rx_antennas, num_chirps, num_freq_samples]
# channel capacity is expectin data in [num_frames, num_tx_antennas, num_rx_antennas]
chan_capacity = channel_capacity(all_results[:,:,:,pulse_idx,freq_idx], BW_Hz=10e6, temperture=290) / 1e6

fig, ax = plt.subplots()
ax.plot(range(numFrames), chan_capacity) # plot the first frequency sample
ax.set(xlabel='Time Index', ylabel='Capacity (Mbps)',title='Channel Capacity vs time')
ax.grid()
plt.show()

# post-process images into gif
if export_debug:
    debug_camera.write_camera_to_gif(file_name='camera.gif')






