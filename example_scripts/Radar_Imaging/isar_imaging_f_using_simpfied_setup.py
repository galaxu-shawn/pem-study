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

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.sar_setup_backend import SAR_Setup_Backend
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization

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
                                          mat_idx=mat_manager.get_index('absorber'),
                                          color='red',
                                          parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                          target_ray_spacing=0.05,
                                          scale_mesh=100)
actor_target_name3 = all_actors.add_actor(filename=os.path.join(paths.models,'F_xy.stl'),
                                          mat_idx=mat_manager.get_index('glass'),
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
# None, isar,  polar_format,
# None will use simple fft (range-doppler) imaging


sar = SAR_Setup_Backend(azimuth_observer_deg=0,
                        elevation_observer_deg=45,
                        distance_observer_m=1000,
                        use_planewave=True,
                        azimuth_aspect_deg=1.7,
                        image_method='fft',)

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
sar.num_pulse_CPI = 200
sar.beam_width_h = 60
sar.beam_width_v = 60

sar.intialize_solver()


# # create a pyvista plotter
# various arguments to control the visualization of the scene.
# camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
# camera_attachment is the name of the actor that the camera will be attached to, ie. 'vehicle1_ego'
output_movie_name = os.path.join(sar.output_path, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=10,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name,
                             figure_size=(0.5, .5),
                             shape=(len(sar.rng_domain), len(sar.vel_domain)),
                             cmap='Greys_r')  # shape is rotated from actual output shape
modeler.pl.add_axes_at_origin(labels_off=True)
modeler.pl.show_grid()
all_max = []

num_of_looks = 360 # phi angles (0, 1, 2, ..., 359)
print('running simulation...')
for idx in tqdm(range(num_of_looks)):

    # update observer position
    sar.azimuth_observer_deg = idx
    image = sar.run_simulation()
    data_max = np.max(image)
    all_max.append(data_max)
    # data_max = -200 # manually adjusting plot max
    modeler.update_frame(plot_data=image.T, plot_limits=[data_max - 30, data_max])  # update visualization

    # all_scalar_bars = list(modeler.pl.scalar_bars.keys())
    # for each in all_scalar_bars:
    #     modeler.pl.remove_scalar_bar(each)

print(f'MaxMax of all frames: {np.max(all_max)}')
print(f'MinMax of all frames: {np.min(all_max)}')
print(f'Avg Max of all frames: {np.median(all_max)}')

modeler.close()


