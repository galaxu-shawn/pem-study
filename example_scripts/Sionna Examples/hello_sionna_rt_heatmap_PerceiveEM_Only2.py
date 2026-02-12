"""
Based on Sionna RT example https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html

This script demonstrates how to use Sionna RT to compute channel impulse responses (CIRs) and compare them with the
results from Perceive EM. The script uses a simple scene with a transmitter and receiver placed in a city environment.
The transmitter and receiver are placed at the same locations in both Sionna RT and Perceive EM simulations. The city
environment is loaded from a predefined scene in Sionna RT, and converted to STL files for Perceive EM. Th


"""

import os
import matplotlib.pyplot as plt
import numpy as np
import time


import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import os
import sys

# Perceive EM Imports and Utilities
sys.path.append("..")  # directory above this directory is where api_core exists
import pem_utilities.pem_core as pem_core
api = pem_core.api
from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, enable_coupling, add_single_tx
from pem_utilities.heat_map import HeatMap
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions


#######################################################################################################################
# Input Parameters
#######################################################################################################################
tx_pos = [0.,0.,20.]
tx_lin = [3., 0., 0]

# rx postion, used to define grid
rx_zpos = 1.5
rx_pos = [0, 0, rx_zpos] # center of the grid
rx_offset_x = 250
rx_offset_y = 250

# grid bounds where heatmap will be calculated [xmin,xmax,ymin,ymax]
grid_bounds = [rx_pos[0]-rx_offset_x, rx_pos[0]+rx_offset_x,
               rx_pos[1]-rx_offset_y, rx_pos[1]+rx_offset_y]
rx_zpos = rx_pos[2]
sampling_spacing_wl = 11.69825
# simulation parameters
max_num_refl = 5
max_num_trans = 1
ray_density = 0.2
ray_batches = 25
go_blockage = -1

# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 3.5e9
num_freqs = 2
bandwidth = 200e6
cpi_duration = 0.9e-3
num_pulse_CPI = 2

# sionna settings
subcarrier_spacing = 15e3
fft_size = 48


save_results = True
#######################################################################################################################
# Setup and Run Perceive EM
#######################################################################################################################

# output file will be stored in this directory
output_path = '../output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_path = os.path.abspath(output_path)

mat_manager = MaterialManager(generate_itu_materials=True, itu_freq_ghz=center_freq * 1e-9)
all_actors = Actors(material_manager=mat_manager)  # all actors, using the same material library for everyone

# Load integrated scene, this is a scene that is provided with Sionna can be convereted from xml to json format
# used by the Perceive EM API library utility. Once converted, you can just call the json file and not need to convert
# the xml file each time.
# Location where XML scene is stored
# env_filename = 'C:\\Users\\asligar\\OneDrive - ANSYS, Inc\\Documents\\Scripting\\github\\instant-rm\\scenes\\etoile\\etoile.xml'
# env_filename = r'C:\ProgramData\miniforge3\envs\sionna\Lib\site-packages\sionna\rt\scenes\etoile\etoile.xml'
# Location where converted XML to JSON file is stored
env_filename = '../output/tmp_cache/scene.json'
# env_filename = '../models/terrain.stl'
city_name = all_actors.add_actor(name='city', filename=env_filename)


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



ant_device_tx = add_single_tx(all_actors,waveform,mode_name,pos=tx_pos,ffd_file='dipole.ffd',scale_pattern=10.0)

heatmap = HeatMap(all_actors=all_actors,
                  sampling_spacing_wl=sampling_spacing_wl,
                  bounds=grid_bounds,
                  z_elevation=rx_zpos,
                  waveform=waveform,
                  mode_name=mode_name,
                  num_subgrid_samples_nx=10,
                  num_subgrid_samples_ny=10,
                  polarization='Z',
                  show_patterns=False,
                  cmap='jet',
                  opacity=1.0
                  )


enable_coupling(mode_name,ant_device_tx, heatmap.probe_device)

if ray_density is not None:
    lambda_center = 2.99792458e8 / ant_device_tx.waveforms[mode_name].center_freq
    ray_spacing = np.sqrt(2) * lambda_center / ray_density

# assign modes to devices
print(api.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()

# optional check if RSS is configured
# this will also be checked before response computation
if not api.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(api.getLastWarnings())

which_mode = ant_device_tx.modes[mode_name]  # tell it which mode we want to get response from
ant_device_tx.waveforms[mode_name].get_response_domains(which_mode)

# this is calculated for round trip, multiply by 2 to get one way
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain * 2
time_domain = rng_domain / 3e8

print(f"Range domain max: {rng_domain[-1]}")
print(f"Time domain max: {time_domain[-1]}")


# video output speed
fps = 100
x_domain = np.linspace(heatmap.grid_positions_x[0],
                       heatmap.grid_positions_x[-1]+heatmap.subgrid_rx_positions_x[-1],
                       num=heatmap.total_samples_x)
y_domain = np.linspace(heatmap.grid_positions_y[0],
                       heatmap.grid_positions_y[-1]+heatmap.subgrid_rx_positions_y[-1],
                       num=heatmap.total_samples_y)

output_movie_name = os.path.join(output_path, 'out_vis.mp4')
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=fps,
                             shape=(heatmap.total_samples_x, heatmap.total_samples_y),
                             x_domain=x_domain,
                             y_domain=y_domain,
                             camera_attachment=None,
                             camera_orientation=None,
                             output_video_size=(1920,1080))

modeler.pl.show_grid()

print("Running Perceive EM Simulation...")

heatmap.update_heatmap(tx_mode=ant_device_tx.modes[mode_name],
                       probe_mode=heatmap.probe_device.modes[mode_name],
                       function='db',
                       modeler=modeler,
                       plot_min=-150,
                       plot_max=-40,)

modeler.update_frame(write_frame=False) # if write_frame=False, no video will be created, just the modeler shown.
modeler.close()

#######################################################################################################################
# Perceive EM Simulation Completed
#######################################################################################################################

if save_results:
    # save results, you can use Results_View_2D_Heamap.py to plot these results again so you don't need to rerun the simulation
    np.save(os.path.join(output_path,'output2.npy'),heatmap.image)
    np.save(os.path.join(output_path,'grid_x2.npy'),heatmap.x_domain)
    np.save(os.path.join(output_path,'grid_y2.npy'),heatmap.y_domain)