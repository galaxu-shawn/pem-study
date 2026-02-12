#
# Copyright ANSYS. All rights reserved.
#
#######################################
# Create a heatmap using the arbitrary heatmap class. This loads an list of points, and updates the heatmap in batches

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization

from pem_utilities.antenna_device import Waveform, add_single_tx
from pem_utilities.debugging_utils import DebuggingLogs
from pem_utilities.heat_map import HeatMapArbitraryPoints
from pem_utilities.simulation_options import SimulationOptions

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

########################################################################################################################
# START USER INPUTS
########################################################################################################################

tx_pos = [5, -1.4, 1.5] # location of the transmitter

center_pos = [5,-4,0] # approximatly the center of the house
bounding_box_offset_x = 6 # in meter, + and - from center_pos
bounding_box_offset_y = 5
bounding_box_offset_z = 3

# grid bounds where heatmap will be calculated [xmin,xmax,ymin,ymax,zmin,zmax]
grid_bounds = [center_pos[0]-bounding_box_offset_x, center_pos[0]+bounding_box_offset_x,
               center_pos[1]-bounding_box_offset_y, center_pos[1]+bounding_box_offset_y,tx_pos[2],tx_pos[2]] #

sampling_spacing_wl = 2

# simulation options
go_blockage = 0 # set to -1 if no GO blockage, set to 0 or higher for GO blockage. This number represents at which bounce to apply the GO blockage model. If set to 0, GO blockage is applied immediatly to the incident field. if 1, that means the GO blockage is only applied after the first bounce.
max_num_refl = 1
max_num_trans = 0
ray_density = 0.2

# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 6e9
num_freqs = 101
bandwidth = 1500e6
cpi_duration = 0.9e-3
num_pulse_CPI = 2
export_debug = True

########################################################################################################################
# END USER INPUTS
########################################################################################################################


# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()  # all actors, using the same material library for everyone

#add a multi part actor from json
house_name = all_actors.add_actor(name='house',
                                  filename=os.path.join(paths.models,'Home_P2P/Home_P2P.json'),
                                  transparency=0.7, 
                                  scale_mesh=1.0,# scale_mesh if original geometry is incorrectly scaled
                                  target_ray_spacing=0.2) # in addition to ray shoot option sim_options below, you could add a target_ray_spacing which would be applied to invidiual objects

# NOTE: target_ray_spacing, is the ray spacing on the mesh in meters. This will be constant no matter where the Mesh is in the scene
# with grid based ray shoot, the global ray_spacing or ray_density is determined at a predefined distance
# with sbr based ray shoot, the global ray_spacing or ray_density is determined based on the triangle size of the geometry
# target_ray_spacing is only applied whtn grid based ray shoot is used, with sbr ray shoot it is ignored becuase the global ray spacing/density 
# becomes relative to the triangle size and basically overrides the target_ray_spacing

# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more flexibility
# and control over the parameters, without having to create/modify a json file
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
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

ant_device_tx = add_single_tx(all_actors,waveform,mode_name,
                              pos=tx_pos,ffd_file='dipole.ffd',scale_pattern=1.0)


# create a grid of points to evaluate the heatmap, these don't have to be a grid, they can be arbitrary points
# but for visualization a grid is easier to see
grid_of_points_x = np.arange(grid_bounds[0],grid_bounds[1],sampling_spacing_wl*wavelength)
grid_of_points_y = np.arange(grid_bounds[2],grid_bounds[3],sampling_spacing_wl*wavelength)
grid_of_points_z = np.array([tx_pos[2]])
all_points_of_mesh = np.array(np.meshgrid(grid_of_points_x,grid_of_points_y,grid_of_points_z)).T.reshape(-1,3)



# heatmap will evaluate at arbitrary points instead of a grid, this allows more flexibility in point placement
# but also allows for us to do progressive refinement, where we start with a low resolution image and progressively refine it
heatmap = HeatMapArbitraryPoints(ant_device_tx,
                  all_actors=all_actors,
                  progressive_refinement=True, # this updates the full image in low resoluiton, then progressively refines it
                  list_of_points=all_points_of_mesh, # arbitrary list of points to solve, will be reshaped if progressive refinement is used
                  probe_antenna_file='dipole.ffd', # we can use dipole, isotropic of json file to define antenna used in evaluation
                  waveform=waveform,
                  mode_name=mode_name, # which mode name from the waveform
                  polarization='Z', # origination of antenna (this would be dipole along Z axis for this example)
                  show_patterns=False, # we can show the antenna patterns (slower, only use for debugging)
                  cmap='inferno',
                  num_parallel_points=20, # how many points to compute in parallel, increase if you have more GPU memory
                  opacity=1.0
                  )

print(pem.listGPUs())

# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
sim_options = SimulationOptions(center_freq=center_freq) # must set center frequency if ray density is defined (ray spacing can be set without center frequency)
sim_options.ray_density = ray_density # global ray density in rays per wavelength, determined at a predefined distance for grid based ray shoot, and per object for sbr ray shoot. This will be overridden on an object grid based is used and a target_ray_spacing is assigned.
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.ray_shoot_method='grid' # grid or sbr (grid uses a uniform ray shoot determined by ray density/spacing at 200m. sbr uses triangle size of each object, like HFSS-SBR+)
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

# Note: ray_shoot_method = 'sbr' is beta feature in 25.2 with some performance limits to be updated in 26.1.

# optional check if RSS is configured
# this will also be checked before response computation
if not pem.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(pem.getLastWarnings())

# we can skip terminal blockage for PO if we want. By default, a single bouce we will generate a PO current for the first bounce and
# the terminal bounce of a ray. So you techincaly get more bounces even when number of reflections is 1.
# pem_api_manager.isOK(pem.setPrivateKey("SkipTerminalBncPOBlockage", "true"))

# video output speed
fps = 20
output_movie_name = os.path.join(paths.output, 'out_vis_indoor_wireless_heatmap_TD.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=fps,
                             overlay_results=False,
                             camera_attachment=None,
                             camera_orientation=None,
                             output_movie_name=output_movie_name)

modeler.pl.show_grid() # add grid to visualization

print("Running Perceive EM Simulation...")

# time the update_heatmap function
import time as walltime
start_time = walltime.time()
heatmap.update_heatmap(function='db',
                       modeler=modeler,
                       sub_grid_updates=True, # this will update as each sub grid is calculated (can be slower)
                       plot_min=-100,
                       plot_max=-10,
                       add_mesh_to_overlay=True, # overlay the heatmap on the model
                       pulse_idx=0,
                       freq_idx=0, # if multiple frequencies solved, use this index to select which one is displayed
                       tx_idx=None, 
                       rx_idx=None, # if multiple rx used, use this index to select which one is displayed
                       save_all_data=True) # save_sall_data will save the full complex data for all points, all pulses, all freq, this can be very large
end_time = walltime.time()

print(f"Heatmap calculation took {end_time-start_time} seconds, including Visualization")
print(f"Simulation (actual solve time) time inside heatmap calculation: {heatmap.simulation_time} seconds")

modeler.update_frame(write_frame=False) #if write_frame is false, it will not write the frame to the movie and freeze the last frame so we can interact with it
# when modeler is closed, the script will continue


if export_debug:
    debug_logs.write_scene_summary(file_name=f'out.json')


#################################################################
#
# PLOT RESPONSE AT A SPECIFIC POINT
# this uses the all_data variable saved from the heatmap calculation above
# this data is not saved by default becuase it can be larger. If you want to save it, set save_all_data=True in the heatmap.update_heatmap function above
#
###############################################################

# the resulting shape of the complex data will be (num_points, num_tx, num_rx, num_pulses, num_freqs)
print(f"shape of output {np.shape(heatmap.all_data)}")

# create a plot of specific point showing magnitude and phase over all frequencies
point_idx = int(len(all_points_of_mesh)/2) # middle point
data_at_point = heatmap.all_data[point_idx] # shape (num_tx, num_rx, num_pulses, num_freqs)
data_at_point_db = 20*np.log10(np.abs(data_at_point)+1e-12)
data_at_point_phase = np.angle(data_at_point)   # in radians
freqs = np.linspace(center_freq-bandwidth/2,center_freq+bandwidth/2,num_freqs)/1e9

# Create a visually engaging plot with professional styling
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.patch.set_facecolor('white')

# Magnitude plot
ax1.plot(freqs, data_at_point_db[0,0,0,:], linewidth=2.5, color='#E63946', 
         marker='o', markersize=3, markevery=10, alpha=0.9, label='Magnitude')
ax1.set_title(f'Frequency Response at Point {point_idx}\nLocation: [{all_points_of_mesh[point_idx][0]:.2f}, {all_points_of_mesh[point_idx][1]:.2f}, {all_points_of_mesh[point_idx][2]:.2f}] m', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Magnitude (dB)', fontsize=14, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax1.legend(fontsize=12, loc='best', framealpha=0.9)
ax1.set_xlim([freqs[0], freqs[-1]])

# Phase plot
ax2.plot(freqs, data_at_point_phase[0,0,0,:], linewidth=2.5, color='#457B9D', 
         marker='s', markersize=3, markevery=10, alpha=0.9, label='Phase')
ax2.set_xlabel('Frequency (GHz)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Phase (radians)', fontsize=14, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax2.legend(fontsize=12, loc='best', framealpha=0.9)
ax2.set_xlim([freqs[0], freqs[-1]])
ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

plt.tight_layout()
# plt.savefig(os.path.join(paths.output,'point_response.png'), dpi=300, bbox_inches='tight')
plt.show()


