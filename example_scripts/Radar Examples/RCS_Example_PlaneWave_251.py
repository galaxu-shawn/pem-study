#
# Copyright ANSYS. All rights reserved.
# This example will only run in 25.1 or later
import time as perftime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pyvista as pv

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.materials import MaterialManager
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.load_mesh import MeshLoader
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy

if '2024' in pem.version():
    print('ERROR: This example requires 25.1 or later')
    sys.exit()

# results from HFSS-SBR+  if you want to overlay and compare results
hfss_csv = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\hfss.csv'
# import csv file from hfss_csv path
compare_results = os.path.isfile(hfss_csv)
if compare_results:
    hfss_data = np.genfromtxt(hfss_csv, delimiter=',', skip_header=1)
    mono_rcs_db_hfss= hfss_data.T[3]
    phi_hfss= hfss_data.T[2]


# simulation parameters
go_blockage = -1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 1
ray_spacing = 0.25  # global ray spacing in meters, each actor can have a different ray spacing if you want

# radar parameters
center_freq = 10.0e9
num_freqs = 3 # number of frequency samples, this is the number of samples in the frequency domain, using 3 so middle sample is center freq
bandwidth = 300e6
num_pulse_CPI = 3 # static simulation so all pulses we will be the same (no moving targets), simulation setup requires >1 pulse
cpi_duration =1 # simulation is static so this parameter doesn't really matter, but set to 1 second for easy math
# True, show modeler on each update (Slower), False, show modeler after simulation completes (Faster)
dynamic_modeler_update = False

os.makedirs(paths.output, exist_ok=True)

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()
all_actors = Actors()

# target is going to just sit at the center of the scene,
actor_target_name = all_actors.add_actor(filename =os.path.join(paths.models,'Ships','explorer_ship_meter.stl'),target_ray_spacing=0.04) # ray_spacing in meters on target
all_actors.actors[actor_target_name].coord_sys.pos = (0.,0.,0.)
all_actors.actors[actor_target_name].coord_sys.lin = (0.,0.,0.)
all_actors.actors[actor_target_name].coord_sys.update() # sets the values entered above


# Radar Platform > Radar Device > Radar Mode(s) > Antenna(s)
# this is a blank actor, just so we can attach the radar device to it and more easily reference it later
actor_radar_name = all_actors.add_actor()

# waveform parameters, note the output type will be in FreqPulse, so we will get
# an array of shape [num_tx,num_rx,num_pulse_CPI,num_freq_samples]
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
waveform = Waveform(waveform_dict)

########################## Setup Radar Platform  ##########################

# helpful utility function to easily create a single tx/rx device, where tx and rx are colocated
# range_pixels and doppler_pixels are not used if output is FREQPULSE, if RANGE_DOPPLER, then they are used to upsample
# attach the device to the radar actor (the empty one we created above)
ant_device = add_single_tx_rx(all_actors,
                              waveform,
                              mode_name,
                              parent_h_node=all_actors.actors[actor_radar_name].h_node,
                              planewave=True,
                              polarization='HH',
                              range_pixels=128,
                              doppler_pixels=256,
                              scale_pattern=1)
rot = euler_to_rot(phi=180, theta=0, psi=0)  # rotate antenna so it is always pointing back at the target
all_actors.actors[actor_radar_name].coord_sys.rot = rot
# radar position phase reference, in 25.1 this should be outside of the target, this will change in 25.2
all_actors.actors[actor_radar_name].coord_sys.pos = [0,0,0]
all_actors.actors[actor_radar_name].coord_sys.update()

# beta feature for sbr style ray shoot
pem_api_manager.isOK(pem.setPrivateKey("RayShootGrid", "SBR," + str(0.5)))
print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360 # if 180 it will only show in the +X direction
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
sim_options.auto_configure_simulation()


num_phi_samples = 361

# target will be rotated around the origin, so we will get RCS at all angles around the target
phi_points = np.linspace(0, 360, num=num_phi_samples)

# value for plotting the RCS overlay, it will be on a circile around the target
radius = 200 # distance away from the target origin
x = radius * np.cos(np.deg2rad(phi_points))
y = radius * np.sin(np.deg2rad(phi_points))
all_points = np.vstack((x,y,np.zeros(len(x)))).T


all_results = []
all_results_rcs_dB = []
# # create a pyvista plotter
modeler = ModelVisualization(all_actors,overlay_results=False,show_antennas=True)

rcs_cartesian_points = all_points # intialize list to hold all points that we will plot within the modeler window
rcs_viz = pv.lines_from_points(rcs_cartesian_points)
rcs_viz['rcs_abs'] = np.ones(len(rcs_cartesian_points))*-300 # initialize the rcs values to -300
rcs_actor = modeler.pl.add_mesh(rcs_viz, cmap='jet', line_width=6)

sim_performance_timer = []

print('running simulation...')

for idx in tqdm(range(num_phi_samples)):

    #only updating the orientation of the target based on the phi_points
    rot = euler_to_rot(phi=phi_points[idx],theta=0,psi=0) # rotate target
    all_actors.actors[actor_target_name].coord_sys.rot = rot
    all_actors.actors[actor_target_name].coord_sys.update()

    # run the simulaiton using pem_api_manager.isOK(pem.computeResponseSync()), capture, the simulation time
    start = perftime.perf_counter()
    pem_api_manager.isOK(pem.computeResponseSync())
    end = perftime.perf_counter()
    sim_performance_timer.append(end-start)

    (ret, response) = pem.retrieveResponse(ant_device.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)
    # response is in format (num_tx, num_rx, num_pulse_CPI, num_freq_samples)
    # we also have 3 chirps, but since nothing is moving, we only need 1 chirp, I am choosing middle chirp (3 total)
    # also we only want 1 freq point so choose the middle one which will be equal to the center freq
    response = response[0,0,1,1]

    # scale power to get RCS
    Pr = np.sqrt(4*np.pi)*np.abs(response)
    Pr_dB = 20 * np.log10(Pr) #dBsm

    all_results_rcs_dB.insert(0,Pr_dB)
    all_results.append(Pr) # all_results will be in format (num_phi_points, num_freq_samples)


    # as visualization example, lets create a graphic that shows the RCS response for each phi point.
    # update RCS overlay plot
    if dynamic_modeler_update:
        modeler.pl.actors[rcs_actor.name].mapper.dataset.active_scalars[:idx + 1] = all_results_rcs_dB
        modeler.pl.update_scalar_bar_range([np.min(all_results_rcs_dB), np.max(all_results_rcs_dB)])
        modeler.update_frame()
    elif idx == num_phi_samples-1: #only update the plot on the last frame
        modeler.pl.actors[rcs_actor.name].mapper.dataset.active_scalars[:idx + 1] = all_results_rcs_dB
        modeler.pl.update_scalar_bar_range([np.min(all_results_rcs_dB), np.max(all_results_rcs_dB)])


modeler.update_frame(write_frame=False) # if we don't write a frame, the animation will pause so we can see it
modeler.close()
sim_performance_timer = np.array(sim_performance_timer)[1:] # get rid of first sim because the includes license checkout
total_sim_time = np.sum(sim_performance_timer)
avg_sim_time = np.mean(sim_performance_timer)
print("Total Simulation Time [s]: {:.4e}".format(total_sim_time))
print("Average Time Per Observation [s]: {:.4e}".format(avg_sim_time))
all_results = np.array(all_results)
all_results_rcs_dB = np.array(all_results_rcs_dB)

fig, ax = plt.subplots()
ax.plot(phi_points, all_results_rcs_dB,label='PEM') # plot the first frequency sample
if compare_results:
    ax.plot(phi_hfss, mono_rcs_db_hfss,label='HFSS',linestyle='--')
ax.set(xlabel='Phi (deg)', ylabel='RCS dBsm',title='RCS vs. phi')
ax.grid()
ax.legend()
plt.show()







