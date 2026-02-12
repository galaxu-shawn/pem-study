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

gpu_num = 0  # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DRJIT_LIBLLVM_PATH'] = 'C:/Program Files/LLVM/bin/LLVM-C.dll'
# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os

    os.system("pip install sionna")
    import sionna

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

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


# Sionna Imports
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1)  # Set global random seed for reproducibility
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
ray_density = 0.1
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
os.makedirs(output_path, exist_ok=True)
output_path = os.path.abspath(output_path)

mat_manager = MaterialManager(generate_itu_materials=True, itu_freq_ghz=center_freq * 1e-9)
all_actors = Actors(material_manager=mat_manager)  # all actors, using the same material library for everyone

# Load integrated scene, this is a scene that is provided with Sionna can be convereted from xml to json format
# used by the Perceive EM API library utility. Once converted, you can just call the json file and not need to convert
# the xml file each time.
# Location where XML scene is stored
# env_filename = 'C:\\ProgramData\\miniforge3\\envs\\sionna\\lib\\site-packages\\sionna\\rt\\scenes\\munich\\munich.xml'
# Location where converted XML to JSON file is stored
env_filename = '../output/munich_scene/scene.json'
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


#######################################################################################################################
# Setup and Run Sionna RT
#######################################################################################################################

scene = load_scene(sionna.rt.scene.munich)  # Try also sionna.rt.scene.etoile

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="V")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="V")

# Create transmitter
tx = Transmitter(name="tx",position=tx_pos)

# Add transmitter instance to scene
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx",
              position=rx_pos,
              orientation=[0, 0, 0])

# Add receiver instance to scene
scene.add(rx)

tx.look_at(rx)  # Transmitter points towards receiver

scene.frequency = center_freq # in Hz; implicitly updates RadioMaterials

scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)

# Select an example object from the scene
so = scene.get("Altes_Rathaus-itu_marble")

# Print name of assigned radio material for different frequenies
for f in [3.5e9, 2.14e9]:  # Print for differrent frequencies
    scene.frequency = f
    print(f"\nRadioMaterial: {so.radio_material.name} @ {scene.frequency / 1e9:.2f}GHz")
    print("Conductivity:", so.radio_material.conductivity.numpy())
    print("Relative permittivity:", so.radio_material.relative_permittivity.numpy())
    print("Complex relative permittivity:", so.radio_material.complex_relative_permittivity.numpy())
    print("Relative permeability:", so.radio_material.relative_permeability.numpy())
    print("Scattering coefficient:", so.radio_material.scattering_coefficient.numpy())
    print("XPD coefficient:", so.radio_material.xpd_coefficient.numpy())

# Compute propagation paths
paths = scene.compute_paths(max_depth=max_num_refl,
                            num_samples=1e7)  # Number of rays shot into directions defined
# by a Fibonacci sphere , too few rays can
# lead to missing paths

paths.normalize_delays = False  # without this, the first path is normalized to 0s

######################
# Complete Sionna RT
#####################


# Visualize paths in the 3D preview
scene.preview(paths, show_devices=True, show_paths=True)  # Use the mouse to focus on the visualized paths
# Default parameters in the PUSCHConfig

# Print shape of channel coefficients before the application of Doppler shifts
# The last dimension corresponds to the number of time steps which defaults to one
# as there is no mobility
print("Shape of `a` before applying Doppler shifts: ", paths.a.shape)

# Apply Doppler shifts
paths.apply_doppler(sampling_frequency=subcarrier_spacing,  # Set to 15e3 Hz
                    num_time_steps=14,  # Number of OFDM symbols
                    tx_velocities=[3., 0, 0],  # We can set additional tx speeds
                    rx_velocities=[0, 7., 0])  # Or rx speeds

print("Shape of `a` after applying Doppler shifts: ", paths.a.shape)

# convert frequency response to range profile
# freq_response = all_results[0, 0, 0, 0]  # center pulse?
# num_bins = 512
# range_profile = np.fft.ifft(np.fft.ifftshift(freq_response, axes=0), axis=0, n=num_bins) * num_bins
# # high to low
# # plot range profile
# range_profile = np.fmax(np.abs(range_profile), 1.e-30)
# time_domain = np.linspace(0, time_domain[-1], num_bins)
#
# # sort range_profile from largest to smallest, keep on the top N points. Use the same index values to create a new array of time_domain with the corroponding values
# N = 48
# idxs = np.argsort(range_profile, axis=0)[::-1]  # high to low
# range_profile = range_profile[idxs]
# range_profile = range_profile[:N]  # keep only the top N points
# time_domain = time_domain[idxs] * 1e9  # ns
# time_domain = time_domain[:N]
# range_profile = range_profile / np.max(range_profile)  # normalize to 1
#
# a, tau = paths.cir()
# print("Shape of tau: ", tau.shape)
#
# t = tau[0, 0, 0, :] / 1e-9  # Scale to ns
# a_abs = np.abs(a)[0, 0, 0, 0, 0, :, 0]
#
# a_max = np.max(a_abs)
#
# range_profile = range_profile * a_max  # make results the same scale, range profile has previously been normalized to 1

# Line of Sight between Tx/Rx is 82.1188 meters
# or 273.33ns at speed of light

# Add dummy entry at start/end for nicer figure
# t = np.concatenate([(0.,), t, (np.max(t)*1.1,)])
# a_abs = np.concatenate([(np.nan,), a_abs, (np.nan,)])
# pem_cir = np.concatenate([(np.nan,), pem_cir, (np.nan,)])
# And plot the CIR
# plt.figure()
# plt.title("Channel impulse response realization")
# plt.stem(t, a_abs, markerfmt='r', label='Sionna RT')
# plt.stem(time_domain, range_profile, markerfmt='gD', label='Perceive EM')
# # plt.xlim([0, np.max(t)])
# # plt.ylim([-2e-6, a_max*1.1])
# plt.xlabel(r"$\tau$ [ns]")
# plt.ylabel(r"$|a|$")
# plt.legend()
# plt.show()

# Compute frequencies of subcarriers and center around carrier frequency
# frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
# # Compute the frequency response of the channel at frequencies.
# h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)  # Non-normalized includes path-loss
# h_freq_pem = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)  # Perceive EM
# # Verify that the channel power is normalized
# h_avg_power = tf.reduce_mean(tf.abs(h_freq) ** 2).numpy()
#
# print("Shape of h_freq: ", h_freq.shape)
# print("Average power h_freq: ", h_avg_power)  # Channel is normalized
#
# # Placeholder for tx signal of shape
# # [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
# x = tf.zeros([h_freq.shape.as_list()[i] for i in [0, 3, 4, 5, 6]], tf.complex64)
#
# no = 0.1  # noise variance
#
# # Init channel layer
# channel = ApplyOFDMChannel(add_awgn=True)
#
# # Apply channel
# y = channel([x, h_freq, no])
#
# # [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
# print(y.shape)
#
# # Init pusch_transmitter
# pusch_config = PUSCHConfig()
#
# # Instantiate a PUSCHTransmitter from the PUSCHConfig
# pusch_transmitter = PUSCHTransmitter(pusch_config)
#
# # Create a PUSCHReceiver using the PUSCHTransmitter
# pusch_receiver = PUSCHReceiver(pusch_transmitter)
#
# # Simulate transmissions over the
# batch_size = 100  # h_freq is broadcast, i.e., same CIR for all samples but different AWGN realizations
# ebno_db = 2.  # SNR in dB
#
# no = ebnodb2no(ebno_db,
#                pusch_transmitter._num_bits_per_symbol,
#                pusch_transmitter._target_coderate,
#                pusch_transmitter.resource_grid)
#
# x, b = pusch_transmitter(batch_size)  # Generate transmit signal and info bits
#
# y = channel([x, h_freq, no])  # Simulate channel output
#
# b_hat = pusch_receiver([y, no])  # Recover the info bits
#
# # Compute BER
# print(f"BER: {compute_ber(b, b_hat).numpy():.5f}")

#
cm = scene.coverage_map(max_depth=5,
                        diffraction=True, # Disable to see the effects of diffraction
                        cm_cell_size=(5., 5.), # Grid size of coverage map cells in m
                        combining_vec=None,
                        precoding_vec=None,
                        num_samples=int(20e6)) # Reduce if your hardware does not have enough memory
# Create new camera
tx_pos = scene.transmitters["tx"].position.numpy()
bird_pos = tx_pos.copy()
bird_pos[-1] = 1000 # Set height of coverage map to 1000m above tx
bird_pos[-2]-= 0.01 # Slightly move the camera for correct orientation

# Create new camera
bird_cam = Camera("birds_view", position=bird_pos, look_at=tx_pos)

scene.add(bird_cam)


# Open 3D preview (only works in Jupyter notebook)
scene.preview(coverage_map=cm)

cm.show(tx=0); # If multiple transmitters exist, tx selects for which transmitter the cm is shown

