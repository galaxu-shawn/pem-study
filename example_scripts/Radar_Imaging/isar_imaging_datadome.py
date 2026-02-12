# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

Data Dome

@author: asligar
"""
import json
from tqdm import tqdm
import numpy as np
import os
import sys
import shutil
import pyvista as pv
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.sar_setup_backend import SAR_Setup_Backend
from pem_utilities.materials import MaterialManager, MatData
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.primitives import *
from pem_utilities.rotation import euler_to_rot, vec_to_rot, look_at, rot_to_euler
from pem_utilities.domain_transforms import DomainTransforms

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy

##################################################################################################
# INPUT PARAMETERS
###################################################################################################

center_freq = 9.6e9 # center frequency in Hz
desired_max_range = 25 # meters, this is the size of the output image we want. We will simulate a larger range to remove any aliasing in the larger ground plane
resolution = 0.2 # for bpth range and cross range

show_modeler = True # it is faster to not show this modeler

output_path = os.path.join(paths.output,'F')
export_debug = True


# upsample factor, this creates an image that is twice as large as the desired max range size. That way we can create ground
# plane that is large and avoid any aliasing of the plane. For example, If our desired max range is 25m, 2x upsample, will create 
# an image that is 50m x 50m, with a ground plane that is also this 50x50. then when we create an image, we just downsample it to 
# the desired size of 25m x 25m image. Throwing away pixels that are in outside the desired range window.
# this will create more frequency samples and a larger bandwidth than the desired max range, but this is ok, as we will downsample the image to the desired size
upsample_factor = 2 

##################################################################################################
# MATERIALS
###################################################################################################

mat_manager = MaterialManager()

# material used for the tank, some roughness, otherwise we only see edges of triangles if not perpendicular to the radar line of sight
height_standard_dev = 1 # mm
corr_length = 0.005 # meter
roughness = height_standard_dev*1e-3/corr_length # rougness (unitless)
my_material = MatData.from_dict({
				"thickness": -1,
				"relEpsReal": 1.0,
				"relEpsImag":0.0,
				"conductivity":1e8, # conductivity in S/m
                "height_standard_dev": height_standard_dev,
                "roughness": roughness})
mat_manager.create_material('my_metal',my_material)

##################################################################################################
# GEOMETRY SETUP
###################################################################################################
wl=3e8/center_freq

# load meshes and create scene elements
all_actors = Actors()
actor_scene_ref_name = all_actors.add_actor()
geo_filename = 'F_xy.stl'
actor_target_name1 = all_actors.add_actor(filename=os.path.join(paths.models,geo_filename),
                                          mat_idx=mat_manager.get_index('my_metal'),
                                          parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                          target_ray_spacing=wl,
                                          scale_mesh=100)

target_bounds = all_actors.actors[actor_target_name1].get_bounds() # get the mesh bounds (across all possible meshes in actor)
target_center = all_actors.actors[actor_target_name1].get_center()

# center target to [0,0,0]
all_actors.actors[actor_target_name1].coord_sys.pos = [-target_center[0], -target_center[1], -target_bounds[4]+0]
# all_actors.actors[actor_target_name1].coord_sys.rot  = euler_to_rot(theta=180, order='zyz', deg=True) # set rotation to [0,0,0] in the local coordinate system of the target
all_actors.actors[actor_target_name1].coord_sys.update()
# all_actors.actors[actor_target_name1].update_rot_based_on_ang_vel = False
# all_actors.actors[actor_target_name1].use_linear_velocity_equation_update =False


##################################################################################################
# SIMULATION SETTINGS
###################################################################################################

# some calculation to determine freq/bw/samples related to range.  We will have a square pixel, cross range and range will be the same
num_range_bins = upsample_factor*int(desired_max_range/resolution) # number of samples in the range domain

range_domain = np.linspace(0, resolution * (num_range_bins - 1), num=num_range_bins)
dt = DomainTransforms(range_domain=range_domain, center_freq=center_freq)
aspect_ang_phi = dt.aspect_angle
num_phi = dt.num_aspect_angle
bandwidth = dt.bandwidth
num_freqs = dt.num_freq

sar = SAR_Setup_Backend(distance_observer_m=10, # for visualization of observation points, does not impace simulation
                        use_planewave=True,
                        azimuth_aspect_deg=aspect_ang_phi)

sar.create_scene(all_actors,target_ref_actor_name=actor_scene_ref_name)
sar.export_debug = export_debug
sar.output_path = output_path
sar.go_blockage = 0  # set to -1 if no GO blockage, set to 0 or higher for GO blockage applied at bounce number defined
sar.max_num_refl = 1
sar.max_num_trans = 0
sar.ray_density = .1
sar.ray_shoot_method = 'sbr'  # 'grid' or 'sbr' 
# radar parameters
sar.center_freq = center_freq
sar.num_freqs = num_freqs
sar.bandwidth = bandwidth
# how many pulses to create in a CPI, this is the number of pulses used to create a single image
sar.num_pulse_CPI = num_phi
sar.polarization = 'HH'
sar.range_pixels = num_freqs # up or downsampling number of range pixels, this is the number of pixels in the range domain
sar.doppler_pixels = num_phi # up or downsampling number of doppler pixels, this is the number of pixels in the doppler domain
sar.r_specs = 'hann' # [type,db_down] window in range domain, use None or "flat" for no windowing
sar.d_specs = 'hann' # window in doppler domain, use None or "flat" for no windowing

sar.intialize_solver()
pem_api_manager.isOK(pem.setPrivateKey("SkipTerminalBncPOBlockage", "true")) # we can skip terminal blockage for PO if we want

##################################################################################################
# OBSERVER POINTS
###################################################################################################

# Generate points , this will create a uniform sampling of the hemisphere
# points_xyz, points_az_el = sar.generate_equal_space_samples_spherical(256, theta_start=10, theta_stop=45,show_plot=True)

# this would generate uniform sampling in theta and phi spacing values
points_xyz, points_az_el = sar.generate_equal_space_samples_uv(phi_spacing_deg=5, theta_spacing_deg=10,
                                                               theta_start=70, theta_stop=80,show_plot=True)


##################################################################################################
# VISUALIZATION - Does not impact simulation, but useful for debugging. Turn off for faster simulation
###################################################################################################

if show_modeler:
    # # create a pyvista plotter
    # various arguments to control the visualization of the scene.
    # camera_orientation is a string that can be 'scene_top', 'follow','follow2','follow3', 'side', 'top', 'front', 'radar'
    # camera_attachment is the name of the actor that the camera will be attached to, ie. 'actor_target_name1'
    output_movie_name = os.path.join(sar.output_path, 'out_vis.mp4') # movie will output to this path
    modeler = ModelVisualization(all_actors,
                                show_antennas=False,
                                fps=10,
                                camera_orientation=None,
                                camera_attachment=None,
                                output_movie_name=output_movie_name,
                                figure_size=(0.5, .5),
                                shape=(int(len(sar.rng_domain)/upsample_factor), int(len(sar.vel_domain)/upsample_factor)),
                                cmap='Greys_r')  # shape is rotated from actual output shape
    modeler.pl.add_axes_at_origin(labels_off=True)

all_max = []

##################################################################################################
# RUN SIMULATION
###################################################################################################

print('running simulation...')
observer_geo = pv.PolyData() # this is used to plot the observation points in the visualization (does not impact simulation)

all_data = {"geometry":geo_filename,"format":'complex',"results":{}} # data structure to save simulation results
n=0 # counter for the number of observations
for az_el in tqdm(points_az_el):
    # update observation point in the simulation
    sar.azimuth_observer_deg = az_el[0]
    sar.elevation_observer_deg = az_el[1]
    # if a generator for a ground plane is used, and dynamic_generator_updates=True we can update it its random surface roughness dynamically
    # this is useful for simulating a rough surface, where we don't want the same roughness to be used for every observation
    # we use the time argument, but for the RoughPlane generator, this argument just changes the random seed for the generator
    # if dynamic_generator_updates=False, then the generator will not be updated in the simulation loop, even if the time argument is passed
    # if use_explicit_rough_surface:
    #     all_actors.actors[ground_name].update_actor(time=n) 

    # run teh simulation, return complex data for 1 tx 1 rx
    image = sar.run_simulation(function='complex')

    # ToDo, account for even and odd sized arrays
    # downsample based on the upsample factor
    if upsample_factor > 1: 
        center = int(image.shape[0] // 2) # array should always be square

        # Calculate half of M
        half_M = int(image.shape[0]/upsample_factor // 2)

        # Slice the centered MxM region
        subarray = image[center - half_M:center + half_M,
                        center - half_M:center + half_M]

    # save image to a numpy array in output path
    image_filename = os.path.abspath(os.path.join(output_path, f'img_{str(n).zfill(4)}.npy'))
    np.save(image_filename, subarray)

    temp_dict = {
        'azimuth': az_el[0],
        'elevation': az_el[1],
        'filename': f'img_{str(n).zfill(4)}.npy'
    }

    all_data["results"][f'{str(n).zfill(4)}'] = temp_dict

    if show_modeler:

        # data for display in modeler visulation that will dynamically update as the simulation runs
        image = 10*np.log10(np.abs(subarray))
        # add an object to plot for observation angle (just visualization)
        sphere_pos = sar.convert_az_el_dist_to_xyz()
        size_val = .2
        observer_geo += pv.Cube(center=sphere_pos,x_length=size_val, y_length=size_val, z_length=size_val)
        modeler.pl.add_mesh(observer_geo, color='red', show_scalar_bar=False, reset_camera=False)

        data_max = np.max(image)
        
        dynamic_range = 60 # dynamic range for the plot, this is the range of values that will be displayed in the plot
        modeler.update_frame(plot_data=image.T, plot_limits=[data_max - dynamic_range, data_max])  # update visualization

    n += 1
if show_modeler:
    modeler.close()
sar.export_debug_camera()
# print(f'MaxMax of all frames: {np.max(all_max)}')
# print(f'MinMax of all frames: {np.min(all_max)}')
# print(f'Avg Max of all frames: {np.median(all_max)}')

# convert numpy arrays to lists for json serialization
def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return x.tolist()
    raise TypeError(x)

# save results
meta_data_output_filename = os.path.abspath(os.path.join(output_path, f'results.json'))
json_object = json.dumps(all_data, default=convert, indent=4)
with open(meta_data_output_filename, "w") as outfile:
    outfile.write(json_object)

# copy import geometry to output path
import_geo_filename = os.path.join(paths.models, geo_filename)
import_geo_filename = os.path.abspath(import_geo_filename)
output_geo_filename = os.path.join(output_path, geo_filename)
shutil.copy(import_geo_filename, output_geo_filename)


# if we want to visualize the data, we can automatically open the visualization tool
# this will open the visualization tool and load the data from the json file (experimental)
from pem_toolkits.Radar_Data_Dome_Visualization.Data_Dome_Visualizer import SARVisualizer
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)
window = SARVisualizer(json_path=meta_data_output_filename)
window.show()
sys.exit(app.exec())

