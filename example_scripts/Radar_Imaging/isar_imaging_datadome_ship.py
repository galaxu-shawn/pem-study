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

# working on variations of seastate, I have two different implementations, seastate and seastate2
from pem_utilities.seastate import OceanSurface

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

##################################################################################################
# INPUT PARAMETERS
###################################################################################################

geo_filename = 'explorer_ship_meter.stl' # name of the geometry file to load, this should be in the models directory

generate_raw_iq = False # if False, the Range vs Doppler data will be saved to a file, if True, the freq vs pulse data will be saved to a file

center_freq = 9.6e9 # center frequency in Hz
desired_max_range = 500 # meters, this is the size of the output image we want. We will simulate a larger range to remove any aliasing caused by the larger ground plane
resolution = 0.5 # for both range and cross range

# set to True if you want to use the RoughPlane generator for the ground plane, this will create a rough surface at the mesh level
# regenerated at every observation point. In addition we can apply a rough surface to this using material properties.
# set to False if you want to use a smooth surface with only surface roughness material property applied
include_ship_wake = True 
include_sea =  True

ray_density = 0.01

go_blockage = -1  # set to -1 if no GO blockage, set to 0 or higher for GO blockage applied at bounce number defined
max_num_refl = 3
max_num_trans = 0 # current model has no transparent materials, so values higher than 0 won't matter
skip_terminal_bnc = False # set to True if you want to skip terminal blockage for PO, this is a beta feature and not available in all versions of the API
polarization = 'HH' # polarization of the radar

# values of data dome in spherical coordinates, this is the spacing of the points in the spherical coordinate system
phi_spacing_deg=5
theta_spacing_deg=10
theta_start=70
theta_stop=80

show_modeler = True # it is faster to not show this modeler
export_debug = True

# configure output file naming
file_save_str = f'{geo_filename}_wake{include_ship_wake}_pol{polarization}_rd{ray_density}_refl{max_num_refl}_go{go_blockage}_skipterminalbnc{skip_terminal_bnc}_ver{pem_api_manager.version}'

output_path = os.path.join(paths.output,file_save_str)



# upsample factor, this creates an image that is twice as large as the desired max range size. That way we can create ground
# plane that is large and avoid any aliasing of the plane. For example, If our desired max range is 25m, 2x upsample, will create 
# an image that is 50m x 50m, with a ground plane that has max length (diagonal) less than 50m. Then when we create an image, we just truncate it to 
# the desired size of 25m x 25m image. Throwing away pixels that are outside the desired range window (+/- from centered pixel).
# this will create more frequency samples and a larger bandwidth than the desired max range
upsample_factor = 2 

##################################################################################################
# MATERIALS
###################################################################################################

# Rough Surface Used for the Ground Plane
mat_manager = MaterialManager()

# material used for the ship, some roughness, otherwise we only see edges of triangles if not perpendicular to the radar line of sight
height_standard_dev = 1 # mm
corr_length = 0.005 # meter
roughness = height_standard_dev*1e-3/corr_length # rougness (unitless)
my_material = MatData.from_dict({
				"thickness": -1,
				"relEpsReal": 1.0,
				"relEpsImag":0.0,
				"conductivity":1e8,
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
actor_target_name1 = all_actors.add_actor(filename=os.path.join(paths.models,geo_filename),
                                          mat_idx=mat_manager.get_index('my_metal'),
                                          parent_h_node=all_actors.actors[actor_scene_ref_name].h_node,
                                          target_ray_spacing=wl,
                                          scale_mesh=1)

target_bounds = all_actors.actors[actor_target_name1].get_bounds() # get the mesh bounds (across all possible meshes in actor)
target_center = all_actors.actors[actor_target_name1].get_center()

# center target to [0,0,0]
all_actors.actors[actor_target_name1].coord_sys.pos = [-target_center[0], -target_center[1], -target_bounds[4]+0]
all_actors.actors[actor_target_name1].coord_sys.rot  = euler_to_rot(phi=90, order='zyz', deg=True) # set rotation to [0,0,0] in the local coordinate system of the target
all_actors.actors[actor_target_name1].coord_sys.update()
# all_actors.actors[actor_target_name1].update_rot_based_on_ang_vel = False
# all_actors.actors[actor_target_name1].use_linear_velocity_equation_update =False

# assuming target is centered at [0,0,0], set the ground plane to also be centered at [0,0,0]


edge_length = desired_max_range*upsample_factor/2*np.sqrt(2) # length of the ground plane, this is the size of the ground plane in meters.

if include_sea:
    # Set up the ocean parameters
    num_grid = 500  # grid size
    scene_length = edge_length  # domain size
    wind_speed = 10
    wave_amplitude = 2  # wave amplitude
    choppiness = .5  # choppiness factor
    ship_velocity = 10  # ship velocity
    ship_length = 110

    # ship position, z is the pos of the ship, ocean is at z=0,
    # so ship waterline is 3m above ocean based on the CAD model
    ship_position = [0, 0, 3]

    ocean = OceanSurface(num_grid, 
                            scene_length,
                            wind_speed, 
                            wave_amplitude, 
                            choppiness,
                            include_wake=include_ship_wake,
                            velocity_ship=ship_velocity,
                            length_ship=ship_length,
                            beam_ship=20.3,
                            draft_ship=6.5,
                            initial_wake_position=ship_position[:2],
                            update_wake_position=False)
    
    # ocean = OceanSurface(num_grid, 
    #                         scene_length,
    #                         wind_speed, 
    #                         wave_amplitude, 
    #                         choppiness,
    #                         include_wake=include_ship_wake,
    #                         velocity_ship=ship_velocity,
    #                         length_ship=ship_length,
    #                         beam_ship=20.3,
    #                         draft_ship=6.5,
    #                         initial_wake_position=ship_position[:2],
    #                         enable_swell=True,
    #                         swell_amplitude=0.25,
    #                         swell_wavelength=10.0,
    #                         swell_direction=[1,0],
    #                         smooth=True)

    # dynamic_generator_updates=False means that the actor will not be updated in the simulation loop 
    # (set to true if you want to the geometry to be dynamically updated, useful with RoughPlane generator to create new surfaces for each observation)
    ocean_name = all_actors.add_actor(name='ocean',
                                      generator=ocean,
                                      mat_idx=mat_manager.get_index('seawater_rough'),
                                      dynamic_generator_updates=True,
                                      target_ray_spacing=wl*4,
                                      parent_h_node=all_actors.actors[actor_scene_ref_name].h_node)


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

if generate_raw_iq:
    image_method = 'raw_iq' # this will save the raw iq data to a file
else:
    image_method = 'range_doppler'

sar = SAR_Setup_Backend(distance_observer_m=10, # for visualization of observation points, does not impace simulation
                        use_planewave=True,
                        azimuth_aspect_deg=aspect_ang_phi,
                        image_method=image_method)

sar.create_scene(all_actors,target_ref_actor_name=actor_scene_ref_name)
sar.export_debug = export_debug
sar.output_path = output_path
sar.go_blockage = go_blockage
sar.max_num_refl = max_num_refl
sar.max_num_trans = max_num_trans
# sar.ray_density = .1 # this is overridden by the privatekey setting (beta feature 25.1)
# radar parameters
sar.center_freq = center_freq
sar.num_freqs = num_freqs
sar.bandwidth = bandwidth
# how many pulses to create in a CPI, this is the number of pulses used to create a single image
sar.num_pulse_CPI = num_phi
sar.polarization = polarization
sar.range_pixels = num_freqs # up or downsampling number of range pixels, this is the number of pixels in the range domain
sar.doppler_pixels = num_phi # up or downsampling number of doppler pixels, this is the number of pixels in the doppler domain
sar.r_specs = 'hann' # [type,db_down] window in range domain, use None or "flat" for no windowing
sar.d_specs = 'hann' # window in doppler domain, use None or "flat" for no windowing
sar.ray_density = ray_density # this is the density of rays in the scene, this is used to determine the ray spacing

sar.ray_shoot_method = "sbr"  # use grid or sbr method for ray shooting

sar.intialize_solver()
if skip_terminal_bnc:
    pem_api_manager.isOK(pem.setPrivateKey("SkipTerminalBncPOBlockage", "true")) # we can skip terminal blockage for PO if we want

##################################################################################################
# OBSERVER POINTS
###################################################################################################

# Generate points , this will create a uniform sampling of the hemisphere
# points_xyz, points_az_el = sar.generate_equal_space_samples_spherical(256, theta_start=10, theta_stop=45,show_plot=True)

# this would generate uniform sampling in theta and phi spacing values
points_xyz, points_az_el = sar.generate_equal_space_samples_uv(phi_spacing_deg=phi_spacing_deg, theta_spacing_deg=theta_spacing_deg,
                                                               theta_start=theta_start, theta_stop=theta_stop,show_plot=True)


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
    
    # get a new ocean state (the input time can be used as essentially a seed for the random number generator)
    if include_sea:
        all_actors.actors[ocean_name].update_actor(time=np.random.uniform(0,10)) # update the ocean surface for the current observation point

    # update observation point in the simulation
    sar.azimuth_observer_deg = az_el[0]
    sar.elevation_observer_deg = az_el[1]
    # if a generator for a ground plane is used, and dynamic_generator_updates=True we can update it its random surface roughness dynamically
    # this is useful for simulating a rough surface, where we don't want the same roughness to be used for every observation
    # we use the time argument, but for the RoughPlane generator, this argument just changes the random seed for the generator
    # if dynamic_generator_updates=False, then the generator will not be updated in the simulation loop, even if the time argument is passed
    # if use_explicit_rough_surface:
    #     all_actors.actors[ground_name].update_actor(time=n) 

    # run the simulation, return complex data for 1 tx 1 rx
    image = sar.run_simulation(function='complex')


    # the imaging method is range doppler, we can truncate the image to half original size, getting rid of the edges of the sea model
    # this is used to make sure our image window includes the sea surface filling the entire image and not the edges of the model
    
    # ToDo, account for even and odd sized arrays
    # downsample based on the upsample factor
    if upsample_factor > 1: 
        center = int(image.shape[0] // 2) # array should always be square

        # Calculate half of M
        half_M = int(image.shape[0]/upsample_factor // 2)

        # Slice the centered MxM region
        image = image[center - half_M:center + half_M,
                        center - half_M:center + half_M]

    # save image to a numpy array in output path
    image_filename = os.path.abspath(os.path.join(output_path, f'img_{str(n).zfill(4)}.npy'))

    np.save(image_filename, image)


    temp_dict = {
        'azimuth': az_el[0],
        'elevation': az_el[1],
        'filename': f'img_{str(n).zfill(4)}.npy'
    }

    all_data["results"][f'{str(n).zfill(4)}'] = temp_dict

    if show_modeler:

        # data for display in modeler visulation that will dynamically update as the simulation runs
        image = 10*np.log10(np.abs(image))
        # add an object to plot for observation angle (just visualization)
        sphere_pos = sar.convert_az_el_dist_to_xyz()
        size_val = .2
        observer_geo += pv.Cube(center=sphere_pos,x_length=size_val, y_length=size_val, z_length=size_val)
        modeler.pl.add_mesh(observer_geo, color='red', show_scalar_bar=False, reset_camera=False)

        data_max = np.max(image)
        
        dynamic_range = 60 # dynamic range for the plot, this is the range of values that will be displayed in the plot
        if generate_raw_iq:
            modeler.update_frame()  # update visualization
        else:
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

print(f"Results saved to {meta_data_output_filename}")

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

