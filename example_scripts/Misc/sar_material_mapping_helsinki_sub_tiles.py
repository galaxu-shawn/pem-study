# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

This uses the vehicles velocities to create the image

@author: asligar
"""

from tqdm import tqdm
import numpy as np
import os
import sys
import time as walltime
import scipy.signal.windows as windows
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.domain_transforms import DomainTransforms
from pem_utilities.sar_setup_backend import SAR_Setup_Backend
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.post_processing import crop_around_center, pulse_freq_to_doppler_range

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

terrain_path = os.path.join(paths.models,'Helsinki')

output_size_num_pixels_range = 512
output_size_num_pixels_cross_range = 512 # number of pixels in the output image, this is the size of the output image we want. 
window_func = windows.taylor 
sidelobe_level = 200 # Sidelobe level in dB, this is used for the Taylor windowing of the image


all_terrain_actor_names = []

# add tiles in teh scene, centered around the center_tile, with a total ox tiles_x by tiles_y
center_tile = [1987,2691]

# tiles in x and y direction
tiles_x = 1
tiles_y =1
center_pos = [0,0,0]
z_max=0
all_meshes = []


mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()

actor_scene_ref_name = all_actors.add_actor()

center_pos = np.array([6875.0, 4875.0, 30.385555684566498])
center_pos = np.array([6625.0, 4625.0, 27.09367561340332])
center_pos = np.array([6750.00, 4750.00,0])
#
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
                                             map_texture_to_material=True,
                                             parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                             add_mesh_offset=-1*center_pos)
        all_terrain_actor_names.append(actor_name)
        # if x+1 == tiles_x//2+tiles_x%2 and y+1 == tiles_y//2+tiles_y%2:
        #     center_tile_mesh = all_actors.actors[actor_name].get_mesh()
        #     center_pos_to_print = center_tile_mesh[0].center
        #     z_max = center_tile_mesh[0].bounds[5]
        #     print(center_pos_to_print)
        # all_meshes.extend(all_actors.actors[actor_name].get_mesh())

# import pyvista as pv
# mesh = pv.PolyData()
# for each in all_meshes:
#     mesh+=each
# x_center = (mesh.bounds[1]-mesh.bounds[0])/2+mesh.bounds[0]
# y_center = (mesh.bounds[3]-mesh.bounds[2])/2+mesh.bounds[2]
# print(f'Overall scene bounds: X: {mesh.bounds[0]:.2f} to {mesh.bounds[1]:.2f}, Y: {mesh.bounds[2]:.2f} to {mesh.bounds[3]:.2f}, Z: {mesh.bounds[4]:.2f} to {mesh.bounds[5]:.2f}')
# print(f'Overall scene size: X: {x_center:.2f} m, Y: {y_center:.2f} m')

num_of_looks = 1

center_freq = 10e9
desired_max_range = 1000
resolution = 0.5  # meters
upsample_factor = 2
num_range_bins = upsample_factor*int(desired_max_range/resolution) # number of samples in the range domain

range_domain = np.linspace(0, resolution * (num_range_bins - 1), num=num_range_bins)
dt = DomainTransforms(range_domain=range_domain, center_freq=center_freq)
aspect_ang_phi = dt.aspect_angle
num_phi = dt.num_aspect_angle
bandwidth = dt.bandwidth
num_freqs = dt.num_freq
image_method='fft'
sar = SAR_Setup_Backend(
                        azimuth_observer_deg=0,
                        elevation_observer_deg=30,
                        use_planewave=True,
                        azimuth_aspect_deg=aspect_ang_phi,
                        image_method=image_method)

sar.create_scene(all_actors,target_ref_actor_name=actor_scene_ref_name)


sar.output_path = paths.output
sar.go_blockage = -1  # set to -1 if no GO blockage, set to 0 or higher for GO blockage
sar.max_num_refl = 1
sar.max_num_trans = 1
sar.ray_density = 0.0001 # ['0.001','0.005','.01','0.025','0.05','0.1','0.25','0.5']
sar.ray_shoot_method = 'grid'  # use grid or sbr method for ray shooting
# radar parameters
sar.center_freq = center_freq
sar.num_freqs = num_freqs
sar.bandwidth = bandwidth
# how many pulses to create in a CPI, this is the number of pulses used to create a single image
sar.num_pulse_CPI = num_phi
sar.max_batches = 10000
sar.polarization = 'HH'
sar.range_pixels = output_size_num_pixels_range *upsample_factor
sar.doppler_pixels = output_size_num_pixels_cross_range*upsample_factor
# sar.enhanced_ray_processing=True
# this can be a custom value 50 or 'range' uses unambiguous range, this value is around center,
# so center_pos-range_filter/2 --> cetner_pos+range_filter/2
# sar.range_filter = 'range'

sar.intialize_solver()


# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(sar.output_path, 'out_vis_helsinki_sar.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=False,
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name,
                             figure_size=(0.5, 0.5),
                             shape=(len(sar.rng_domain), len(sar.vel_domain)),
                             cmap='Greys_r')  # shape is rotated from actual output shape
modeler.pl.add_axes_at_origin(labels_off=True)
modeler.pl.show_grid()

print('running simulation...')
for idx in tqdm(range(num_of_looks)):
    sar.azimuth_observer_deg = idx # phi angles (0, 1, 2, ..., 359)

    data = sar.run_simulation(function='complex')


    if image_method is None:
        full_image_window = pulse_freq_to_doppler_range(data, 
                                            window_function = window_func, 
                                            sidelobe_level=sidelobe_level, 
                                            output_size_doppler=upsample_factor * output_size_num_pixels_cross_range, 
                                            output_size_range=upsample_factor * output_size_num_pixels_range)
    else:
        full_image_window = data
    
    # Crop the upsampled image to the desired output size around the center pixel
    if upsample_factor > 1: 
        image_final_window = crop_around_center(full_image_window, output_size=(output_size_num_pixels_cross_range, output_size_num_pixels_range))
    else:
        # No upsampling was used, use the full image as-is
        image_final_window = full_image_window

    print(f"Cropped image shape: {image_final_window.shape} from original shape: {full_image_window.shape}")
    to_plot = 10*np.log10(np.abs(image_final_window)).T
    # to_plot = image_final_window.T
    print(f'min: {np.min(to_plot):.2f}, max: {np.max(to_plot):.2f}')
    modeler.update_frame(plot_data=to_plot,plot_limits=[np.max(to_plot)-30,np.max(to_plot)])  # update visualization

    # str_to_append = f'{sar.ray_density}_{sar.ray_shoot_method}_{idx:03d}_enhanced.npy'
    # np.save(os.path.join(sar.output_path, f'image_{str_to_append}'), image_final_window)
    # np.save(os.path.join(sar.output_path, f'time_{str_to_append}'), sar.sim_performance_time)

    # all_scalar_bars = list(modeler.pl.scalar_bars.keys())
    # for each in all_scalar_bars:
    #     modeler.pl.remove_scalar_bar(each)



modeler.close()



