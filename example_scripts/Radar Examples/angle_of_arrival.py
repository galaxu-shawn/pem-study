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
from pem_utilities.antenna_device import AntennaDevice
from pem_utilities.utils import generate_rgb_from_array
from pem_utilities.far_fields import convert_ffd_to_gltf
from pem_utilities.post_processing import range_angle_map
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
# multiple material libraries can be loaded and the indices will be updated accordingly
# mat_manager = MaterialManager(['material_properties_ITU_3.85GHz.json','material_library.json'])
mat_manager = MaterialManager(generate_itu_materials=True, itu_freq_ghz=3.85)


debug_logs = DebuggingLogs(output_directory=paths.output)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()


# if a filename is not provided in the previous line, this commented out code can be used to add part to the actor
# all_scene_actors['road'].add_part(filename='../models/intersection.stl')

# add a multipart actor from json, this preserves the hierarchy of the actor, defined in the json file. There are
# some special types of files that can be loaded. For example, if we have a vehicle, we can load the vehicle from a json
# which will automatically load tires and rotate them as the car drives
actor_vehicle1_name = all_actors.add_actor()

actor_vehicle2_name = all_actors.add_actor(filename=os.path.join(paths.models,'Sphere_1meter_rad.stl'),
                                             target_ray_spacing=.02,
                                             mat_idx=mat_manager.get_index('plywood'))

# initialize coordinate systems for all actors, and then set the
# initial positions of each actor. Any value that is not set, will be set to zero. We can set pos/lin/rot/ang for each

# Initialize the position of the actors.

# The vehicle1_ego is a single part actor, so we can set all the parameters directly. We need to take special
# consideration of the velocity vector and the rotation. Because this is a general actor, we would need to rotate
# the actor and the velocity vector to be consistent with how we want the actor to behave
all_actors.actors[actor_vehicle1_name].coord_sys.pos = (0., 0.0, 0.)

# The vehicle2 is a multipart actor assigned to be a actor_type='vehicle'. This type of actor has some special
# considerations already embedded into the actor. For example, 4 tires will be placed in the correct location, and
# each tire will rotate depending on the velocity of the vehicle. The velocity only needs to be defined with a special
# property .velocity_mag, and the linear velocity vector will be directly calculated from the rotation of the vehicle

num_frames = 45
range_dist = 40
start_ang = -45
end_ang = 45
angs = np.linspace(start_ang, end_ang, num_frames)
xpos = np.cos(np.deg2rad(angs)) * range_dist
ypos = np.sin(np.deg2rad(angs)) * range_dist
all_actors.actors[actor_vehicle2_name].coord_sys.pos = (xpos[0], ypos[0], 0.)
all_actors.actors[actor_vehicle2_name].coord_sys.lin = (0, 0, 0.)
# all_scene_actors['vehicle2'].velocity_mag = 10.
# The 3x3 rotation matrix is set using a more intuitive input of euler angles.
all_actors.actors[actor_vehicle2_name].coord_sys.rot = euler_to_rot(phi=90, theta=0, order='zyz', deg=True)


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

ant_device = AntennaDevice('example_1tx_8rx.json',
                           parent_h_node=all_actors.actors[actor_vehicle1_name].h_node,
                           all_actors=all_actors)
ant_device.initialize_mode(mode_name='mode1')
ant_device.coord_sys.pos = (0., 0., 0.)
ant_device.coord_sys.update()
ant_device.add_antennas(mode_name='mode1', load_pattern_as_mesh=True, scale_pattern=1)
ant_device.add_mode(mode_name='mode1')


sim_options = SimulationOptions()
sim_options.ray_spacing = .1
sim_options.max_reflections = 3
sim_options.max_transmissions = 1
sim_options.go_blockage = -1
sim_options.field_of_view = 180
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()


# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device.modes['mode1']  # tell it which mode we want to get respones from
ant_device.waveforms['mode1'].get_response_domains(which_mode)

# domain in every dimension
vel_domain = ant_device.waveforms['mode1'].vel_domain
rng_domain = ant_device.waveforms['mode1'].rng_domain
freq_domain = ant_device.waveforms['mode1'].freq_domain
pulse_domain = ant_device.waveforms['mode1'].pulse_domain
output_format = ant_device.waveforms['mode1'].output

if output_format.lower() == 'rangedoppler':
    output_format_api = RssPy.ResponseType.RANGE_DOPPLER
else:
    output_format_api = RssPy.ResponseType.FREQ_PULSE
fov = [-45, 45]
num_angles = 128
angle_domain = np.linspace(fov[0], fov[1], num_angles)
# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate


# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(paths.output, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=rng_domain,
                             angle_domain=angle_domain,
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name,
                             output_video_size=(1280,720))
# modeler.pl.show_grid()
responses = []
cameraImages = []
pem_api_manager.isOK(pem.initRadarCamera(RssPy.CameraProjection.FISHEYE,RssPy.CameraColorMode.COATING,512,512,255,True,1))
pem_api_manager.isOK(pem.activateRadarCamera())
print('running simulation...')
all_frames = []
for iFrame in tqdm(range(num_frames), disable=True):
    one_frame = []
    # update all coordinate systems
    all_actors.actors[actor_vehicle2_name].coord_sys.pos = (xpos[iFrame], ypos[iFrame], 0.)
    all_actors.actors[actor_vehicle2_name].update_actor()
    if iFrame == 0:
        debug_logs.write_scene_summary(file_name=f'out_{iFrame}.json')
    pem_api_manager.isOK(pem.computeResponseSync())

    (ret, response) = pem.retrieveResponse(ant_device.modes['mode1'], output_format_api)

    response = np.array(response)[0] # [tx][rx][vel][range] or [tx][rx][chirp][freq]


    range_angle_response = range_angle_map(response, antenna_spacing_wl=0.5,
                                                source_data=output_format,
                                                DoA_method='bartlett',
                                                fov=fov,
                                                out_size=(len(rng_domain), 512),
                                                chirp_index=len(pulse_domain)//2,
                                                window=False)

    # im_data = 10 * np.log10(np.fmax(np.abs(range_angle_response), 1.e-30))
    im_data = np.fmax(np.abs(range_angle_response), 1.e-30)
    modeler.mpl_ax_handle.set_data(im_data)
    modeler.mpl_ax_handle.set_clim(vmin=np.min(im_data), vmax=np.max(im_data))

    responses.append(response)
    modeler.update_frame()
modeler.close()
all_evaluated = []
all_expected = []
plt.close('all')

# Create figure with improved size
fig, ax = plt.subplots(figsize=(10, 6))

for ang_idx in range(len(angs)):
    responses = np.array(responses)
    diff = np.diff(np.rad2deg(np.unwrap(np.angle(responses[ang_idx,:,30,50]))))
    one_ang = np.deg2rad(diff[0])
    expected_angle = np.rad2deg(np.sin(one_ang*.004/2/np.pi/.002))
    all_evaluated.append(expected_angle)
    all_expected.append(angs[ang_idx])
    print(f'Angle is: {angs[ang_idx]}, Calculated angle is: {expected_angle}')

# Plot with improved styling
ax.plot(all_evaluated, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Calculated')
ax.plot(all_expected, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='Expected')

# Add descriptive labels and title
ax.set_xlabel('Frame Index', fontsize=12)
ax.set_ylabel('Angle (degrees)', fontsize=12)
ax.set_title('Angle of Arrival (AoA) Estimation Comparison', fontsize=14, fontweight='bold')

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Add error measurement
rmse = np.sqrt(np.mean((np.array(all_evaluated) - np.array(all_expected))**2))
ax.text(0.05, 0.05, f'RMSE: {rmse:.2f}°', transform=ax.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), fontsize=10)

# Improve legend
ax.legend(loc='best', frameon=True, fontsize=11, framealpha=0.9, edgecolor='gray')

# Set y-axis limits slightly beyond the min/max values
y_min = min(min(all_evaluated), min(all_expected)) - 5
y_max = max(max(all_evaluated), max(all_expected)) + 5
ax.set_ylim([y_min, y_max])

# Add x-axis ticks for every 5 frames
x_ticks = np.arange(0, len(angs), 5)
ax.set_xticks(x_ticks)

plt.tight_layout()
fig.savefig(os.path.join(paths.output, 'aoa_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
