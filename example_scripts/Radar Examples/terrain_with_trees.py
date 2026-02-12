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
import copy
import glob
import time as walltime
from pem_utilities.create_forest import ForestGenerator
from pem_utilities.materials import MaterialManager, MatData
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.utils import generate_rgb_from_array
from pem_utilities.far_fields import convert_ffd_to_gltf
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.open_street_maps_geometry import BuildingsPrep, TerrainPrep, find_random_location, get_z_elevation_from_mesh
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.cfar import CFAR

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

ray_spacing = 3
max_num_refl = 3 # number of reflections to compute
max_num_trans = 4 # number of transmissions to compute
target_ray_spacing = 1# meters, this is the ray spacing for the actors, this will be used to set the ray spacing for the antenna device
bounding_box = 'default' #disabled or default
ray_shoot_method = 'grid'  # 'sbr' or 'grid'
curved_physics = False
material_name = 'my_ground_medium_dry' # my_ground_very_dry, my_ground_medium_dry, my_ground_wet
num_trees = 1000 # use tree model from vegetation folder, if not found will use procedural trees
fps = 10
end_time = 10


# add material that can be used for the terrain
mat_manager = MaterialManager()
height_standard_dev = .5 # mm
corr_length = .05 # meter
roughness = height_standard_dev*1e-3/corr_length # rougness (unitless)
# example of some other material that could be used, swap out material on line below to use for the ground plane
my_material_very_dry_ground = MatData.from_dict({
            "thickness": -1,
            "relEpsReal": 3.0,
            "relEpsImag": -1.98768,
            "relMuReal": 1.0,
            "relMuImag": 0.0,
            "conductivity": 8.5123,
            "height_standard_dev": height_standard_dev,
            "roughness": roughness})
my_material_medium_dry_ground = MatData.from_dict({
            "thickness": -1,
            "relEpsReal": 9.714986166699312,
            "relEpsImag": -9.712854625629863,
            "relMuReal": 1.0,
            "relMuImag": 0.0,
            "conductivity": 41.595651066379276,
            "height_standard_dev": height_standard_dev,
            "roughness": roughness})
my_material_wet_ground = MatData.from_dict({
            "thickness": -1,
            "relEpsReal": 5.278675494425365,
            "relEpsImag": -9.927231828991342,
            "relMuReal": 1.0,
            "relMuImag": 0.0,
            "conductivity": 42.51372918978494,
            "height_standard_dev": height_standard_dev,
            "roughness": roughness})

height_standard_dev = 1 # mm
corr_length = 0.05 # meter
roughness = height_standard_dev*1e-3/corr_length # rougness (unitless)
my_material_vegetation = MatData.from_dict({
            "thickness": -1,
            "relEpsReal": 5.278675494425365,
            "relEpsImag": -9.927231828991342,
            "relMuReal": 1.0,
            "relMuImag": 0.0,
            "conductivity": 42.51372918978494})

materials = {
    'my_ground_very_dry': my_material_very_dry_ground,
    'my_ground_medium_dry': my_material_medium_dry_ground,
    'my_ground_wet': my_material_wet_ground
}


mat_manager.create_material('my_ground',materials[material_name])
mat_manager.create_material('my_veg',my_material_vegetation)

# create a name that captures input parameters
out_name = f'numrefl_{max_num_refl}_numtrans_{max_num_trans}_bounding_box_{bounding_box}_shoot_{ray_shoot_method}_curved_physics_{curved_physics}_target_ray_spacing_{target_ray_spacing}_global_ray_spacing_{ray_spacing}_material_{material_name}_num_trees_{num_trees}'
out_name = f'surface_roughness_{height_standard_dev}_mm_{corr_length}_m_corr_length_{roughness}_roughness_num_trees_{num_trees}'


time_stamps = np.linspace(0, end_time, num=int(end_time*fps)) # 10 seconds, 101 frames

# range of values that lat/lon can be generated around
# scene_lat_lon = (37.800939, -122.417406)
scene_lat_lon = (34.277821703449405, -117.78671032068603)
scene_lat_lon = (34.14585215242956, -117.90408811515957)
max_radius = 5000 # meters

# Tx Antenna positions
radar_pos = [.5, 0, -0.25]  # this will be offset from terrain  in Z
radar_rot = euler_to_rot(0, 0, 0, order='zyx', deg=True)

drone_pos = [-0, 0, 100] # this will be offset from terrain  in Z
# bird_pos = [-150, 0, 30]  # this will be offset from terrain  in Z

# waveform parameters
center_freq = 77e9
num_freqs = 1400
bandwidth = 750e6
cpi_duration = 4.87e-3
num_pulse_CPI = 200 # this will result in a pulse interval of 1ms

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage

# export debug logs and show modeler for visualization
export_debug = True
show_modeler = True


print(f'Generating Scene: Lat/Lon: {scene_lat_lon}...')

terrain_file_name = os.path.join(paths.models, 'terrain2.stl')

if not os.path.exists(terrain_file_name):
    terrain_prep = TerrainPrep(paths.output)
    terrain = terrain_prep.get_terrain(center_lat_lon=scene_lat_lon, max_radius=max_radius, grid_size=10,)
    terrain_file_name = terrain['file_name']


# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()


terrain_name = all_actors.add_actor(filename=terrain_file_name,
                                 mat_idx=mat_manager.get_index('my_ground'),
                                 color='brown',transparency=0,use_curved_physics=curved_physics,target_ray_spacing=target_ray_spacing)


z_intersect = get_z_elevation_from_mesh(drone_pos[:2], all_actors.actors[terrain_name].get_mesh())
if z_intersect[0] is not None:
    drone_pos[2] += np.max(z_intersect)


actor_drone_name = all_actors.add_actor(name='drone',
                                             filename=os.path.join(paths.models,'Quadcopter2/Quadcopter2.json'),
                                             target_ray_spacing=0.05,scale_mesh=1)

rotor_speed = 77# in Hz
velocity = 20  # in m/s
actor_phi = 0
actor_theta = 0
actor_psi = 0
all_actors.actors[actor_drone_name].rotor_ang = [0, 0, rotor_speed * 2 * np.pi]  # in rad/s
all_actors.actors[actor_drone_name].velocity_mag = velocity
all_actors.actors[actor_drone_name].coord_sys.pos = drone_pos
all_actors.actors[actor_drone_name].coord_sys.rot = euler_to_rot(actor_phi, actor_theta, actor_psi, order='zyx', deg=True)
all_actors.actors[actor_drone_name].coord_sys.update()

actor_drone_name2 = all_actors.add_actor(name='drone2',
                                             filename=os.path.join(paths.models,'Quadcopter2/Quadcopter2.json'),
                                             target_ray_spacing=0.05,scale_mesh=1)

rotor_speed = 77# in Hz
velocity = 25  # in m/s
actor_phi = 0
actor_theta = 0
actor_psi = 0
all_actors.actors[actor_drone_name2].rotor_ang = [0, 0, rotor_speed * 2 * np.pi]  # in rad/s
all_actors.actors[actor_drone_name2].velocity_mag = velocity
all_actors.actors[actor_drone_name2].coord_sys.pos = [drone_pos[0]+200,drone_pos[1]+5, drone_pos[2] + 5]
all_actors.actors[actor_drone_name2].coord_sys.rot = euler_to_rot(0, 0, 0, order='zyx', deg=True)
all_actors.actors[actor_drone_name2].coord_sys.update()
actor_drone_name3 = all_actors.add_actor(name='drone3',
                                             filename=os.path.join(paths.models,'Quadcopter2/Quadcopter2.json'),
                                             target_ray_spacing=0.05,scale_mesh=1)

rotor_speed = 77# in Hz
velocity = 10  # in m/s
actor_phi = 0
actor_theta = 0
actor_psi = 0
all_actors.actors[actor_drone_name3].rotor_ang = [0, 0, rotor_speed * 2 * np.pi]  # in rad/s
all_actors.actors[actor_drone_name3].velocity_mag = velocity
all_actors.actors[actor_drone_name3].coord_sys.pos = [drone_pos[0]+200,drone_pos[1]+5, drone_pos[2] + 5]
all_actors.actors[actor_drone_name3].coord_sys.rot = euler_to_rot(180, 0, 0, order='zyx', deg=True)
all_actors.actors[actor_drone_name3].coord_sys.update()


forest_file_name = os.path.join(paths.cache, f'forest_{num_trees}_output2.stl')
if not os.path.exists(forest_file_name):
    ForestGenerator = ForestGenerator(terrain_file=all_actors.actors[terrain_name].get_mesh()[0],
                                    num_trees=num_trees,
                                    output_path=paths.cache)
    ForestGenerator.generate_forest_in_batches(batch_size=100)
    forest_file_name = ForestGenerator.save_forest(filename=f'forest_{num_trees}_output.stl')

actor_forest_name = all_actors.add_actor(name='forest',
                                             filename=forest_file_name,
                                             target_ray_spacing=0.5,
                                             mat_idx=mat_manager.get_index('my_veg'))

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
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "CENTER_CHIRP"}

pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
mode_name = 'mode1'
# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)
ant_device = add_single_tx_rx(all_actors,waveform,mode_name,pos=radar_pos,rot=radar_rot,
                              beamwidth_H=140,
                              beamwidth_V=120,
                              scale_pattern=0.6,
                              range_pixels=512,
                              doppler_pixels=256,
                              parent_h_node=all_actors.actors[actor_drone_name].h_node)

sim_options = SimulationOptions(center_freq=center_freq)
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 180
if bounding_box=='disabled':
    sim_options.bounding_box=-1
if ray_shoot_method == 'sbr':
    sim_options.ray_shoot_method = 'sbr'
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

# pem_api_manager.isOK(pem.setPrivateKey("SkipTerminalBncPOBlockage", "true")) # we can skip terminal blockage for PO if we want

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

print(f'Velocity Window: {vel_domain[-1]-vel_domain[0]}')
print(f'Max Range: {rng_domain[-1]}')


numFrames = len(time_stamps)
responses = []
cameraImages = []

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device.modes[mode_name],
                                    display_mode='coating',
                                    output_size=(512,512),
                                    background_color=255,
                                    frame_rate=fps)
    debug_logs = DebuggingLogs(output_directory=paths.output)

# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(paths.output, f'{out_name}_out_vis.mp4')


camera_orientation = {}
camera_orientation['cam_offset'] =[-5, 0, 0.0]
camera_orientation['focal_offset'] = [-4, 0, 0]
camera_orientation['up'] =(0.0, 0.0, 1.0)
camera_orientation['view_angle'] =80

modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             rng_domain=rng_domain,
                             vel_domain=vel_domain,
                             fps=fps,
                             figure_size=(.5,.5),
                             camera_orientation='follow4',
                             camera_attachment=actor_drone_name,
                             output_movie_name=output_movie_name,
                             output_video_size=(1280, 720))


modeler.pl.camera.disable_parallel_projection()
# modeler.pl.camera.reset_clipping_range()
# modeler.pl.camera.clipping_range = (0, 1000)
print('running simulation...')
all_responses = []

# Initialize lists to store statistics over time
max_values = []
mean_values = []
median_values = []
percentile_90_values = []
time_list = []

# Initialize text actor for statistics display
text_actor = None

for idx, time in enumerate(tqdm(time_stamps)):

    # update all coordinate systems
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.RANGE_DOPPLER)
    all_responses.append(response)
    # calculate response in dB to overlay in pyvista plotter
    im_data = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))

    # Calculate statistics for current frame
    max_val = np.max(im_data)
    mean_val = np.mean(im_data)
    median_val = np.median(im_data)
    percentile_90_val = np.percentile(im_data, 90)
    
    # Store statistics for plotting later
    max_values.append(max_val)
    mean_values.append(mean_val)
    median_values.append(median_val)
    percentile_90_values.append(percentile_90_val)
    time_list.append(time)

    # exporting radar camera images
    if export_debug:
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)
        if idx == 0:
            debug_logs.write_scene_summary(file_name=f'out.json')
    
    # Remove previous text actor if it exists
    if text_actor is not None:
        modeler.pl.remove_actor(text_actor)
    
    # Add new text with current statistics
    string_to_print = f"Frame: {idx+1}/{numFrames}\nTime: {time:.2f}s\nMax: {max_val:.1f} dB\nMean: {mean_val:.1f} dB\nMedian: {median_val:.1f} dB\n90th %ile: {percentile_90_val:.1f} dB"
    text_actor = modeler.pl.add_text(string_to_print, position='upper_left', font_size=12, color='black')


    detector = CFAR(
        training_cells=30,
        guard_cells=10,
        cfar_type='OS',
        normalized_data=True
    )
    

    detections, threshold = detector.detect(np.abs(response[0][0])/np.max(np.abs(response[0][0])), return_threshold=True)
    n_detections = np.sum(detections)

    # modeler.update_frame(plot_data=np.rot90(detections),plot_limits=[0,1],update_camera_view=True)
    # pause script for 1 second to visualize the detection
    # walltime.sleep(1)
    modeler.update_frame(plot_data=im_data,plot_limits=[np.max(im_data)-80,np.max(im_data)],update_camera_view=True)
    if idx == 0:
        modeler.pl.show_grid()

    
modeler.close()

# Create statistics plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(time_list, max_values, label='Max', linewidth=2, marker='o', markersize=4)
ax.plot(time_list, mean_values, label='Mean', linewidth=2, marker='s', markersize=4)
ax.plot(time_list, median_values, label='Median', linewidth=2, marker='^', markersize=4)
ax.plot(time_list, percentile_90_values, label='90th Percentile', linewidth=2, marker='d', markersize=4)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Amplitude (dB)', fontsize=12)
ax.set_title('Radar Response Statistics Over Time', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add summary statistics as text box
stats_text = f'Overall Statistics:\n'
stats_text += f'Max Range: {np.max(max_values):.1f} - {np.min(max_values):.1f} dB\n'
stats_text += f'Mean Range: {np.max(mean_values):.1f} - {np.min(mean_values):.1f} dB\n'
stats_text += f'Median Range: {np.max(median_values):.1f} - {np.min(median_values):.1f} dB\n'
# stats_text += f'90th %ile Range: {np.max(percentile_90_values)::.1f} - {np.min(percentile_90_values):.1f} dB'

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
statistics_plot_name = os.path.join(paths.output, f'{out_name}_statistics_plot.png')
plt.savefig(statistics_plot_name, dpi=300, bbox_inches='tight')
plt.close()

print(f'Statistics plot saved as: {statistics_plot_name}')

# post-process images into gif. This will take the radar camera and range doppler response and place them side by
# side in a gif
if export_debug:
    # debug_camera.write_camera_to_gif(file_name='camera.gif')
    # debug_camera.write_response_to_gif(responses,file_name='response.gif',rng_domain=rng_domain,vel_domain=vel_domain)
    debug_camera.write_camera_and_response_to_gif(responses,
                                                  file_name=f'{out_name}_camera_and_response.gif',
                                                  clim_db=(np.max(max_values)-80,np.max(max_values)),
                                                  rng_domain=rng_domain,
                                                  vel_domain=vel_domain)