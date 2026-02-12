# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

This uses the vehicles velocities to create the image

@author: asligar
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import sys

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.sar_setup_backend import SAR_Setup_Backend
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.domain_transforms import DomainTransforms

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

mat_manager = MaterialManager()
# load meshes and create scene elements

# create dictionary of all scene actors for easier organization
all_actors = Actors()
actor_scene_ref_name = all_actors.add_actor()
actor_target_name1 = all_actors.add_actor(filename=os.path.join(paths.models,'F_xy.stl'),
                                          mat_idx=mat_manager.get_index('pec'),
                                          color='black',
                                          parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                          target_ray_spacing=0.05,
                                          scale_mesh=100)
actor_target_name2 = all_actors.add_actor(filename=os.path.join(paths.models,'F_xy.stl'),
                                          mat_idx=mat_manager.get_index('glass'),
                                          color='red',
                                          parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                          target_ray_spacing=0.05,
                                          scale_mesh=100)
actor_target_name3 = all_actors.add_actor(filename=os.path.join(paths.models,'F_xy.stl'),
                                          mat_idx=mat_manager.get_index('asphalt_high_rough'),
                                          color='green',
                                          parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                          target_ray_spacing=0.05,
                                          scale_mesh=100)

all_actors.actors[actor_target_name1].coord_sys.pos = [-10, 0, 0]
all_actors.actors[actor_target_name1].coord_sys.update()
all_actors.actors[actor_target_name2].coord_sys.pos = [0, 0, 0]
all_actors.actors[actor_target_name2].coord_sys.update()
all_actors.actors[actor_target_name3].coord_sys.pos = [10, 0, +0]
all_actors.actors[actor_target_name3].coord_sys.update()

# imaging methods
# None, will use simple fft (range-doppler) imaging
# isar --> Recommended
# polar_format --> Experimental, not recommended


sar = SAR_Setup_Backend(azimuth_observer_deg=0,
                        elevation_observer_deg=45,
                        distance_observer_m=1000,
                        use_planewave=True,
                        azimuth_aspect_deg=1.7,
                        image_method='fft') # 'isar', 'polar_format', 'fft' or None

sar.create_scene(all_actors,target_ref_actor_name=actor_scene_ref_name)

sar.output_path = paths.output
sar.go_blockage = -1  # set to -1 if no GO blockage, set to 0 or higher for GO blockage
sar.max_num_refl = 3
sar.max_num_trans = 0
sar.ray_density = .001
# radar parameters
sar.center_freq = 25.0e9
sar.num_freqs = 400
sar.bandwidth = 1000e6
# how many pulses to create in a CPI, this is the number of pulses used to create a single image
sar.num_pulse_CPI = 300
sar.beam_width = 60 # unused with planewave

sar.intialize_solver()




# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(sar.output_path, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=False,
                             fps=10,
                             overlay_results=False)  # shape is rotated from actual output shape
modeler.pl.add_axes_at_origin(labels_off=True)
modeler.pl.show_grid()
sar.azimuth_observer_deg = 0
sar.elevation_observer_deg = 45
sar.azimuth_aspect_deg = 1.7
# just show the plotter so we can see what is going on with the scene. once closed we will not use it again
modeler.update_frame(write_frame=False) 
modeler.close()

# simple plot function to see teh results
def create_plot(image,min_pos=None, max_pos=None):
    if min_pos is None:
        min_pos = [0, 0]
    if max_pos is None:
        max_pos = [np.max(image.shape[0]), np.max(image.shape[1])]

    if isinstance(image, list):
        # if a list, create a subplot for each image
        num_images = len(image)
        fig, axs = plt.subplots(1, num_images, figsize=(8 * num_images, 10))
        for i, img in enumerate(image):
            axs[i].imshow(img, extent=(min_pos[0], max_pos[0], min_pos[1], max_pos[1]), cmap='jet', aspect='auto')
            axs[i].set_title(f'Image {i+1}')
            axs[i].set_xlabel('X Position (m)')
            axs[i].set_ylabel('Y Position (m)')
            axs[i].axis('equal')
            plt.colorbar(axs[i].images[0], ax=axs[i], label='Intensity')
    else:
        # if a single image, create a single plot
        plt.figure(figsize=(10, 10))
        plt.imshow(image, extent=(min_pos[0], max_pos[0], min_pos[1], max_pos[1]), cmap='jet', aspect='auto')
        plt.colorbar(label='Intensity')
        plt.title('SAR Image')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.axis('equal')
        plt.tight_layout()
    plt.tight_layout()
    plt.show()


# update observer position
sar.azimuth_observer_deg = 0
sar.elevation_observer_deg = 1
sar.azimuth_aspect_deg = 1.7
image1 = sar.run_simulation()
# pos = sar.image_plane_dict['pixel_locs']

create_plot(image1, 
            min_pos=[0, 0], 
            max_pos=[sar.domain_transforms_down_range.range_period, sar.domain_transforms_cross_range.range_period])

sar.azimuth_observer_deg = 0
sar.elevation_observer_deg = 70
sar.azimuth_aspect_deg = 1.7
image2 = sar.run_simulation()
# pos = sar.image_plane_dict['pixel_locs']

create_plot(image2, 
            min_pos=[0, 0], 
            max_pos=[sar.domain_transforms_down_range.range_period, sar.domain_transforms_cross_range.range_period])


create_plot([image1, image2],
            min_pos=[0, 0], 
            max_pos=[sar.domain_transforms_down_range.range_period, sar.domain_transforms_cross_range.range_period])





# update observer position
sar.azimuth_observer_deg = 30
sar.elevation_observer_deg = 45
sar.azimuth_aspect_deg = 2.7
image = sar.run_simulation()
# pos = sar.image_plane_dict['pixel_locs']

create_plot(image, 
            min_pos=[0, 0], 
            max_pos=[sar.domain_transforms_down_range.range_period, sar.domain_transforms_cross_range.range_period])










