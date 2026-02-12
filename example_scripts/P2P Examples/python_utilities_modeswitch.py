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

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaDevice
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
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

export_debug = True #use this to export the movie at the end of the simulation (.gif showing radar camera and response)

debug_logs = DebuggingLogs(output_directory=paths.output)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()


actor_overpass_name = all_actors.add_actor(filename =os.path.join(paths.models,'Overpass.stl'),mat_idx = mat_manager.get_index('asphalt'))
actor_vehicle1_name = all_actors.add_actor(filename =os.path.join(paths.models,'mustang-no-wheels.stl'),mat_idx = mat_manager.get_index('aluminum'))
actor_vehicle2_name = all_actors.add_actor(filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json'))
actor_vehicle3_name = all_actors.add_actor(filename =os.path.join(paths.models,'tractor-trailor.stl'))


# initial positions of each actor. Any value that is not set, will be set to zero. We can set pos/lin/rot/ang for each

# Initialize the position of the actors.

# The vehicle1_ego is a single part actor, so we can set all the parameters directly. We need to take special
# consideration of the velocity vector and the rotation. Becuase this is a general actor, we would need to rotate
# the actor and the velocity vector to be consistent with how we want the actor to behave
all_actors.actors[actor_overpass_name].coord_sys.pos = (50., 0., 0.)

all_actors.actors[actor_vehicle1_name].coord_sys.pos = (25.,-5.0, 0.)
all_actors.actors[actor_vehicle1_name].coord_sys.lin = (13., 0., 0.)

all_actors.actors[actor_vehicle2_name].coord_sys.pos = (130,5,0.)
all_actors.actors[actor_vehicle2_name].coord_sys.lin = (-15.,0.,0.)
# truck is coming towards the ego vehicle, so we need to rotate it 180 degrees
all_actors.actors[actor_vehicle2_name].coord_sys.rot = euler_to_rot(phi=180, order='zyz' ,deg=True)

all_actors.actors[actor_vehicle3_name].coord_sys.pos = (50,-5,0.)
all_actors.actors[actor_vehicle3_name].coord_sys.lin = (16.,0.,0.)


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
ant_device_tx = AntennaDevice('example_1tx_2modes.json',parent_h_node=all_actors.actors[actor_vehicle1_name].h_node)
ant_device_tx.initialize_mode(mode_name='mode1')
ant_device_tx.initialize_mode(mode_name='mode2')
ant_device_tx.coord_sys.pos = (2.5,0.,1)
ant_device_tx.coord_sys.update()
ant_device_tx.add_antennas(mode_name='mode1',load_pattern_as_mesh=True,scale_pattern=10)
ant_device_tx.add_mode(mode_name='mode1')
ant_device_tx.add_antennas(mode_name='mode2',load_pattern_as_mesh=True,scale_pattern=10)
ant_device_tx.add_mode(mode_name='mode2')
ant_device_tx.set_mode_active('mode1',status=True)
ant_device_tx.set_mode_active('mode2',status=False)
# If we want to visualize the antennas, we can add them to the scene as actors. This is not required for the simulation
# but is used only for visualization purposes.
for each in ant_device_tx.all_antennas_properties:
    name = all_actors.add_actor(name=each, actor=ant_device_tx.all_antennas_properties[each]['Actor'])


######################
# Rx
####################
ant_device_rx = AntennaDevice('example_1rx_2modes.json',parent_h_node=all_actors.actors[actor_vehicle2_name].h_node)
ant_device_rx.initialize_mode(mode_name='mode1')
ant_device_rx.initialize_mode(mode_name='mode2')
ant_device_rx.coord_sys.pos = (2.5,0.,1)
ant_device_rx.coord_sys.update()
ant_device_rx.add_antennas(mode_name='mode1',load_pattern_as_mesh=True,scale_pattern=10)
ant_device_rx.add_antennas(mode_name='mode2',load_pattern_as_mesh=True,scale_pattern=10)
ant_device_rx.add_mode(mode_name='mode1')
ant_device_rx.add_mode(mode_name='mode2')
ant_device_rx.set_mode_active('mode1',status=True)
ant_device_rx.set_mode_active('mode2',status=False)
# If we want to visualize the antennas, we can add them to the scene as actors. This is not required for the simulation
# but is used only for visualization purposes.
for each in ant_device_rx.all_antennas_properties:
    name = all_actors.add_actor(name=each, actor=ant_device_rx.all_antennas_properties[each]['Actor'])


rComp = RssPy.ResponseComposition.INDIVIDUAL
pem_api_manager.isOK(pem.setTxResponseComposition(ant_device_tx.modes['mode1'],rComp))


pem_api_manager.isOK(pem.setDoP2PCoupling(ant_device_tx.h_node_platform ,ant_device_rx.h_node_platform ,True))


# assign modes to devices
print(pem.listGPUs())
devIDs = [0]; devQuotas = [0.8]; # limit RTR to use 80% of available gpu memory
pem_api_manager.isOK(pem.setGPUDevices(devIDs,devQuotas))
maxNumRayBatches = 25
pem_api_manager.isOK(pem.autoConfigureSimulation(maxNumRayBatches))

# initialize solver settings
pem_api_manager.isOK(pem.setMaxNumRefl(3))
pem_api_manager.isOK(pem.setMaxNumTrans(3))


# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
  print("RSS is not ready to execute a simulation:\n")
  print(pem.getLastWarnings())

# display setup
#print(pem.reportSettings())

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device_tx.modes['mode1'] # tell it which mode we want to get respones from
ant_device_tx.waveforms['mode1'].get_response_domains(which_mode)

vel_domain = ant_device_tx.waveforms['mode1'].vel_domain
rng_domain = ant_device_tx.waveforms['mode1'].rng_domain
freq_domain = ant_device_tx.waveforms['mode1'].freq_domain
pulse_domain = ant_device_tx.waveforms['mode1'].pulse_domain


fps = 10; dt = 1/fps
T = 5; numFrames = int(T/dt)
responses = []
cameraImages = []

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes['mode1'],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=fps)

# # create a pyvista plotter
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=rng_domain,vel_domain=vel_domain,
                             fps=fps,
                             camera_attachment=actor_vehicle1_name,
                             camera_orientation='follow')

print('running simulation...')
# which mode is our active mode, we will toggle this midway through the simulation
active_mode = 'mode1'
for iFrame in tqdm(range(numFrames)):
    time = iFrame*dt
    # update all coordinate systems
    for actor in all_actors.actors:
        # Actors can have actor_type value, this will handle sub-parts of the actors. For example, if we have a vehicle
        # it will update the tires rotation and velocity, and the vehicle linear velocity based on its orientation
        # if actor is other, it will use the lin/rot/ang values to update the position of the actor
        all_actors.actors[actor].update_actor(time=time)

    # at frame 25 switch active modes

    if iFrame == 25:
        active_mode = 'mode2'
        ant_device_rx.set_mode_active('mode1', status=False)
        ant_device_rx.set_mode_active('mode2', status=True)
        ant_device_tx.set_mode_active('mode1', status=False)
        ant_device_tx.set_mode_active('mode2', status=True)

    # write out scene summary, only writing out first and last frame, useful for debugging
    if iFrame == 0 or iFrame == numFrames-1:
        debug_logs.write_scene_summary(file_name=f'out_{iFrame}.json')

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes[active_mode],ant_device_rx.modes[active_mode], RssPy.ResponseType.RANGE_DOPPLER)
    # calculate response in dB to overlay in pyvista plotter
    imData = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))
    modeler.mpl_ax_handle.set_data(imData) # update pyvista matplotlib plot

    if export_debug:
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)

    modeler.update_frame()
modeler.close()

# post-process images into gif
if export_debug:
    # debug_camera.write_camera_to_gif(file_name='camera.gif')
    # debug_camera.write_response_to_gif(responses,file_name='response.gif',rng_domain=rng_domain,vel_domain=vel_domain)
    debug_camera.write_camera_and_response_to_gif(responses,
                                                  file_name='camera_and_response.gif',
                                                  rng_domain=rng_domain,
                                                  vel_domain=vel_domain)
