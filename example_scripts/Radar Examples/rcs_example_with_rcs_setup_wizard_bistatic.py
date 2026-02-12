#
# Copyright ANSYS. All rights reserved.
# This example will only run in 25.1 or later

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Now import utilities - they will work because path is set up
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.rotation import euler_to_rot
from pem_utilities.rcs import RCS
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy

if '2024' in pem.version():
    print('ERROR: This example requires 25.1 or later')
    sys.exit()

# results from HFSS-SBR+  if you want to overlay and compare results
hfss_csv = os.path.join(os.path.dirname(paths.example_scripts / 'Radar Examples'),'cessna_rcs_10GHz.csv')
# import csv file from hfss_csv path
compare_results = os.path.isfile(hfss_csv)
if compare_results:
    hfss_data = np.genfromtxt(hfss_csv, delimiter=',', skip_header=1)
    mono_rcs_db_hfss= hfss_data.T[3]
    phi_hfss= hfss_data.T[2]

rcs = RCS(rcs_mode='bistatic',rayshoot_method='sbr') # rayshoot_method can be 'sbr' or 'grid'. only monostatic is supported for now, bistatic is a WiP.

# simulation parameters
rcs.go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
rcs.max_num_refl = 5
rcs.max_num_trans = 1
rcs.ray_density= 0.5  # global ray spacing in meters, each actor can have a different ray spacing if you want

# radar parameters
rcs.center_freq = 10.0e9
rcs.num_freqs = 128 # number of frequency samples, this is the number of samples in the frequency domain, using 3 so middle sample is center freq
rcs.bandwidth = 400e6
rcs.polarization='VV'
rcs.num_waves_to_solve_in_parallel=1
dynamic_modeler_update = False # True, show modeler on each update (Slower), False, show modeler after simulation completes (Faster)
rcs.output_path = '../output/'
rcs.show_antenna_pattern =False

# rcs output parameters
rcs.incident_wave.phi_start = 0
rcs.incident_wave.phi_stop = 0
rcs.incident_wave.phi_step_deg = 1 # step in degrees, this is the angle around the
rcs.incident_wave.theta_start = 90
rcs.incident_wave.theta_stop = 90 # theta is 0 because we are rotating around the target
rcs.incident_wave.theta_step_deg = 1 # 

rcs.observer_wave.phi_start = -1.0
rcs.observer_wave.phi_stop = +1.0
rcs.observer_wave.phi_step_num = 64 # step in degrees, this is the angle around the
rcs.observer_wave.theta_start = 90-1.0
rcs.observer_wave.theta_stop =  90+1.0# theta is 0 because we are rotating around the target
rcs.observer_wave.theta_step_num =64# 



# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()
all_actors = Actors()

# add Actors to the scene
actor_target_name = all_actors.add_actor() # root actor. everything can reference this actor if we want to easily rotate the entir scene

# target is going to just sit at the center of the scene,
# first actor is cessna model, we will reference this actor to the root actor_targe_name. 
# this would be useful if wanted to rotate/move all actors at once by just moving actor_targe_name
# not required for this example, but included for demonstration purposes
actor_plane_name = all_actors.add_actor(filename=os.path.join(paths.models,'cessna.stl'),
                                        parent_h_node=all_actors.actors[actor_target_name].h_node,
                                        target_ray_spacing=0.04) # ray_spacing in meters on target

all_actors.actors[actor_plane_name].coord_sys.pos = (0.,0.,0.)
all_actors.actors[actor_plane_name].coord_sys.lin = (0.,0.,0.)
all_actors.actors[actor_plane_name].coord_sys.update() # sets the values entered above

# add a corner reflector to the scene to make RCS asymetric, verify phi direction is correct
actor_cr_name = all_actors.add_actor(filename=str(paths.models / 'Corner_Reflector.stl'),
                                     target_ray_spacing=0.1,
                                     parent_h_node=all_actors.actors[actor_target_name].h_node,
                                     scale_mesh=10) # ray_spacing in meters on target
all_actors.actors[actor_cr_name].coord_sys.pos = (-4 ,8.5 ,-1)
all_actors.actors[actor_cr_name].coord_sys.rot = euler_to_rot(phi=180, order='zyz') # rotation in radians
all_actors.actors[actor_cr_name].coord_sys.update() # sets the values entered above

# add the actors to the rcs object so that we can use them in the simulation
rcs.all_actors = all_actors

print('running simulation...')
rcs.run_simulation(show_modeler=dynamic_modeler_update)
print('simulation complete')

rcs.print_timing_summary()
rcs.show_modeler()

fig, ax, x_values, y_values = rcs.plot_rectangular(theta=90, freq_idx=0, function='dB')
if compare_results: # only add trace if we have results to compare
    ax.plot(phi_hfss, mono_rcs_db_hfss, 'r', linewidth=2, label='HFSS-SBR+')

# value for plotting the RCS overlay, it will be on a circile around the target

radius = 50 # distance away from the target origin
x = radius * np.cos(np.deg2rad(rcs.incident_wave.phi_domain))
y = radius * np.sin(np.deg2rad(rcs.observer_wave.phi_domain))
all_points = np.vstack((x,y,np.zeros(len(x)))).T

# pause script to allow user to see the plot until they close it
pause = input('Press Enter to continue...')
# data = rcs.get_rcs_data(theta_range=90, phi_range=None, freq_idx=1)
# to_plot = np.concatenate(data['rcs_data'].values)
# rcs_cartesian_points = all_points # intialize list to hold all points that we will plot within the modeler window
# rcs_viz = pv.lines_from_points(rcs_cartesian_points)
# rcs_viz['rcs_abs'] = to_plot
# rcs_actor = modeler.pl.add_mesh(rcs_viz, cmap='jet', line_width=6)

# modeler.pl.update_scalar_bar_range([np.min(all_results_rcs_dB), np.max(all_results_rcs_dB)])