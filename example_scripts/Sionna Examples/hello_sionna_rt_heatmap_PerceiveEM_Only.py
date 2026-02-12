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
from pem_utilities.antenna_device import AntennaDevice, Waveform
from pem_utilities.heat_map import HeatMap

#######################################################################################################################
# Input Parameters
#######################################################################################################################
tx_pos = [8.5, 21, 27]
tx_lin = [3., 0., 0]

# rx postion, used to define grid
rx_pos = [45, 90, 1.5] # center of the grid
rx_offset_x = 100
rx_offset_y = 100

# grid bounds where heatmap will be calculated [xmin,xmax,ymin,ymax]
grid_bounds = [rx_pos[0]-rx_offset_x, rx_pos[0]+rx_offset_x,
               rx_pos[1]-rx_offset_y, rx_pos[1]+rx_offset_y]
rx_zpos = rx_pos[2]
sampling_spacing_wl = 10
# simulation parameters
max_num_refl = 5
max_num_trans = 1
ray_density = 0.2
ray_batches = 25

# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 2.14e9
num_freqs = 2
bandwidth = 200e6
cpi_duration = 0.9e-3
num_pulse_CPI = 2

# sionna settings
subcarrier_spacing = 15e3
fft_size = 48

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
env_filename = 'C:\\ProgramData\\miniforge3\\envs\\sionna\\lib\\site-packages\\sionna\\rt\\scenes\\munich\\munich.xml'
# Location where converted XML to JSON file is stored
# env_filename = '../output/tmp_cache/scene.json'
# env_filename = '../models/terrain.stl'
city_name = all_actors.add_actor(name='city', filename=env_filename)

# ray shoot is +X without this, with this it is all directions
pem_core.isOK(api.setPrivateKey("FieldOfView", "360"))

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

# initzlize the antenna device, one for Tx, one for Rx
antenna_device_tx = AntennaDevice()
antenna_device_tx.initialize_device()
antenna_device_tx.waveforms[mode_name] = waveform


h_mode = pem_core.RssPy.RadarMode()
antenna_device_tx.modes[mode_name] = h_mode
pem_core.isOK(api.addRadarMode(h_mode, antenna_device_tx.h_device))
antennas_dict = {}
ant_type_tx = {
    "type": "ffd",
    "file_path": "dipole.ffd",
    "operation_mode": "tx",
    "position": [0, 0, 0]
} # position is offset location from where antenna device is placed

antennas_dict["Tx"] = ant_type_tx
antenna_device_tx.add_antennas(mode_name=mode_name,
                               load_pattern_as_mesh=True,
                               scale_pattern=10,
                               antennas_dict=antennas_dict)
antenna_device_tx.set_mode_active(mode_name)
antenna_device_tx.add_mode(mode_name)

# position of each antenna device
antenna_device_tx.coord_sys.pos = tx_pos
antenna_device_tx.coord_sys.lin = tx_lin
antenna_device_tx.coord_sys.update()

#empty probe that will be what gets moved around to sample fields
probe_name = all_actors.add_actor(name='probe')
heatmap = HeatMap(reference_actor=all_actors.actors[probe_name],
                    sampling_spacing_wl=sampling_spacing_wl,
                    bounds=grid_bounds,
                    z_elevation=rx_zpos,
                    waveform=waveform,
                    mode_name=mode_name)


# If we want to visualize the antennas, we can add them to the scene as actors. This is not required for the simulation
# but is used only for visualization purposes. Also required that load_pattern_as_mesh is set to True
for each in antenna_device_tx.all_antennas_properties: # this is adding an existing actor to the list
    name = all_actors.add_actor(name=each,actor=antenna_device_tx.all_antennas_properties[each]['Actor'])
for each in heatmap.probe_device.all_antennas_properties:
    name = all_actors.add_actor(name=each,actor = heatmap.probe_device.all_antennas_properties[each]['Actor'])

# set up response composition, and which modes to capture
rComp = pem_core.RssPy.ResponseComposition.INDIVIDUAL
pem_core.isOK(api.setTxResponseComposition(antenna_device_tx.modes[mode_name], rComp))
pem_core.isOK(api.setDoP2PCoupling(antenna_device_tx.h_node_platform, heatmap.probe_device.h_node_platform, True))

# assign modes to devices
print(api.listGPUs())
devIDs = [0];
devQuotas = [0.8];  # limit RTR to use 80% of available gpu memory
pem_core.isOK(api.setGPUDevices(devIDs, devQuotas))
pem_core.isOK(api.autoConfigureSimulation(ray_batches)) # determine max amount of batches needed to efficiently solve

# initialize solver settings
pem_core.isOK(api.setMaxNumRefl(max_num_refl))
pem_core.isOK(api.setMaxNumTrans(max_num_trans))
pem_core.isOK(api.setTargetRayDensity(ray_density))  # global ray density setting

# optional check if RSS is configured
# this will also be checked before response computation
if not api.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(api.getLastWarnings())

which_mode = antenna_device_tx.modes[mode_name]  # tell it which mode we want to get response from
antenna_device_tx.waveforms[mode_name].get_response_domains(which_mode)

# this is calculated for round trip, multiply by 2 to get one way
rng_domain = antenna_device_tx.waveforms[mode_name].rng_domain * 2
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
                             camera_orientation=None)

modeler.pl.show_grid()

print("Running Perceive EM Simulation...")

heatmap.update_heatmap(tx_mode=antenna_device_tx.modes[mode_name],
                       probe_mode=heatmap.probe_device.modes[mode_name],
                       function='db',
                       modeler=modeler,
                       plot_min=-100,
                       plot_max=-50,)

modeler.update_frame(write_frame=False) # if write_frame=False, no video will be created, just the modeler shown.
modeler.close()

#######################################################################################################################
# Perceive EM Simulation Completed
#######################################################################################################################

