# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

This uses the vehicles velocities to create the image

@author: asligar
"""

from tqdm import tqdm
import numpy as np
import os
import scipy.signal.windows as windows
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.domain_transforms import DomainTransforms # convert from freq/pulse to range/doppler/crossrange domains
from pem_utilities.sar_setup_backend import SAR_Setup_Backend
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.post_processing import DynamicImageViewer

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


# INPUT PARAMETERS

ray_density = .1  # rays per wavelength
ray_shoot_method = 'sbr'  # use 'grid' or 'sbr' method for ray shooting

# add tiles in teh scene, centered around the center_tile, with a total ox tiles_x by tiles_y
center_tile = [1987,2691] # name in folder is Tile_+1987_+2691.vtp, column index, row index

# tiles in x and y direction, total tiles will be tiles_x * tiles_y
tiles_x = 2 
tiles_y = 2

num_of_looks = 1 # multiple looks, this will rotate the model at every 1 deg and create a new image (i.e. 10 looks will rotate 10 deg, at 1 deg steps)
center_freq = 10e9

desired_max_range = 2500 # this will compute the required frequency sweep setup to acheive this max range and resolution
desired_max_cross_range = 2500
resolution = 0.595  # meters, common value for down/cross range

num_range_bins = int(desired_max_range/resolution) # number of samples in the range domain
num_cross_range_bins = int(desired_max_cross_range/resolution) #

output_size_num_pixels_range = 4096
output_size_num_pixels_cross_range = 4096 # number of pixels in the output image, this is the size of the output image we want. 
# END INPUT PARAMETERS

# compute the range and cross-range domains based on desired max range/cross-range and resolution
# this will determine the waveform needed for image generation
range_domain = np.linspace(0, resolution * (num_range_bins - 1), num=num_range_bins)
crossrange_domain = np.linspace(0, resolution * (num_cross_range_bins - 1), num=num_cross_range_bins)

# Initialize domain transform objects
dt_down_range = DomainTransforms(range_domain=range_domain,center_freq=center_freq)
dt_cross_range = DomainTransforms(range_domain=crossrange_domain,center_freq=center_freq)

bandwidth = dt_down_range.bandwidth
num_freqs = dt_down_range.num_freq
aspect_ang_phi = dt_cross_range.aspect_angle
num_aspect_angle = dt_cross_range.num_aspect_angle

# print the computed parameters for waveform setup
print(f'Computed SAR parameters for desired max range {desired_max_range} m, max cross-range {desired_max_cross_range} m, resolution {resolution} m:')
print(f'  Bandwidth: {bandwidth/1e6:.2f} MHz, Number of Frequencies: {num_freqs}')
print(f'  Aspect Angle: {aspect_ang_phi:.2f} deg, Number of Aspect Angles: {num_aspect_angle}')


# all tiles are located in this sub folder
terrain_path = os.path.join(paths.models,'Helsinki')

all_terrain_actor_names = []

mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()

actor_scene_ref_name = all_actors.add_actor()

# manually setting the center position of the tiles, not required, but makes it easier to center the scene
# image within the output results
center_pos = np.array([6750.00, 4750.00,0]) 

# # load the terrain tiles, using the center tile index, and the number of tiles in x and y direction
for x in range(tiles_x):
    for y in range(tiles_y):
        x_index = center_tile[0] - tiles_x//2 + x
        y_index = center_tile[1] - tiles_y//2 + y
        tile_name = f'Tile_+{x_index:03d}_+{y_index:03d}'
        if not os.path.exists(os.path.join(terrain_path, tile_name + '.vtp')):
            print(f'Error: file {tile_name} not found in {terrain_path}')
            print('Contact arien.sligar@ansys.com for access to the Helsinki data')
        file_name = os.path.abspath(os.path.join(terrain_path, tile_name + '.vtp'))
        actor_name = all_actors.add_actor(name=tile_name,
                                             filename=file_name,
                                             include_texture=True,
                                             map_texture_to_material=True, # materials are embeded in the vtp file
                                             parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                             add_mesh_offset=-1*center_pos) # this will move the input mesh to center of scene


# sar setup helper. includes scene setup and ability to chnage sar observation angles easily
# default image is generated using range doppler output from Perceive EM. If image_method=None
# the output returned image will be freq-pulse format
sar = SAR_Setup_Backend(azimuth_observer_deg=0, # this is intial position, will be updated in loop if sar.azimuth_observer_deg is changed
                        elevation_observer_deg=30, # elevation angle of the radar, can be updated by sar.elevation_observer_deg
                        use_planewave=True,
                        azimuth_aspect_deg=aspect_ang_phi)

sar.create_scene(all_actors,target_ref_actor_name=actor_scene_ref_name)

sar.output_path = paths.output
sar.go_blockage = -1  # set to -1 if no GO blockage, set to 0 or higher for GO blockage
sar.max_num_refl = 1
sar.max_num_trans = 1
sar.ray_density = ray_density
sar.ray_shoot_method = ray_shoot_method  # use grid or sbr method for ray shooting

# radar parameters
sar.center_freq = center_freq
sar.num_freqs = num_freqs
sar.bandwidth = bandwidth
# how many pulses to create in a CPI, this is the number of pulses used to create a single image
sar.num_pulse_CPI = num_aspect_angle

sar.range_pixels = output_size_num_pixels_range
sar.doppler_pixels = output_size_num_pixels_cross_range

sar.intialize_solver()

# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(sar.output_path, 'out_vis_helsinki_sar.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=False,
                             fps=10,
                             output_movie_name=output_movie_name,
                             figure_size=(0.5, 0.5),
                             shape=(len(sar.rng_domain), len(sar.vel_domain)),
                             cmap='Greys_r')  # shape is rotated from actual output shape
modeler.pl.add_axes_at_origin(labels_off=True)
modeler.pl.show_grid()


print('running simulation...')
for idx in tqdm(range(num_of_looks)):
    sar.azimuth_observer_deg = idx # phi angles (0, 1, 2, ..., 359) # set the azimuth angle of the radar for each look
    image = sar.run_simulation(function='complex') # return data in this format, can also be 'db', 'complex' 'abs' etc.
    image_db = 20*np.log10(np.abs(image)+1e-12)
    data_max = np.max(image_db)
    modeler.update_frame(plot_data=image_db.T, plot_limits=[data_max - 100, data_max],write_frame=False)  # update visualization, write_frame=False will pause the animation an not auto close

modeler.close()


##### INTERACTIVE PLOT WITH DYNAMIC MIN/MAX ADJUSTMENT #####
img_display = DynamicImageViewer(image=image,math_function='db',cmap='Greys_r')
img_display.save(os.path.join(sar.output_path, 'sar_image_final.png'))
