import os
import matplotlib.pyplot as plt
import numpy as np
import time


gpu_num = 0 # Use "" to use the CPU
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
sys.path.append("..") # directory above this directory is where api_core exists
import pem_utilities.pem_core as pem_core
api = pem_core.api

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actor
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaDevice

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
tf.random.set_seed(1) # Set global random seed for reproducibility

resolution = [480,320] # increase for higher quality of renderings


######################
# Setup and Run Perceive EM
#####################

mat_manager = MaterialManager(generate_itu_materials=True, itu_freq_ghz=2.14)
# output file will be stored in this directory
output_path = '../output/'
os.makedirs(output_path, exist_ok=True)
output_path = os.path.abspath(output_path)

all_scene_actors = {}

# Load integrated scene
# env_filename = 'C:\\ProgramData\\miniforge3\\envs\\sionna\\lib\\site-packages\\sionna\\rt\\scenes\\munich\\munich.xml'
env_filename = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Scripting\github\perceive_em\example_scripts\output\munich_scene\scene.json'
all_scene_actors['city'] = Actor(filename=env_filename,material_manager = mat_manager)

pem_core.isOK(api.setPrivateKey("FieldOfView", "360"))
ant_device_tx = AntennaDevice(file_name='example_1tx_2.14GHz_38901.json')
ant_device_tx.initialize_mode(mode_name='mode1')
# ant_device_tx.coord_sys.rot = euler_to_rot(phi=45, theta=0, order='zyz', deg=True)
#ant_device_tx.coord_sys.pos = (-0.2,0.0,0.2) for hand attachment
#for body attachment:
ant_device_tx.coord_sys.pos = (8.5,21,27)
ant_device_tx.coord_sys.lin = (3.,0.,0)
ant_device_tx.coord_sys.update()
ant_device_tx.add_antennas(mode_name='mode1',load_pattern_as_mesh=True,scale_pattern=10)
ant_device_tx.add_mode(mode_name='mode1')

# If we want to visualize the antennas, we can add them to the scene as actors. This is not required for the simulation
# but is used only for visualization purposes. Also required that load_pattern_as_mesh is set to True
for each in ant_device_tx.all_antennas_properties:
    all_scene_actors[each] = ant_device_tx.all_antennas_properties[each]['Actor']

######################
# Rx1
#####################
ant_device_rx1 = AntennaDevice(file_name='example_1rx_2.14GHz.json')
ant_device_rx1.initialize_mode(mode_name='mode1')
# ant_device_rx1.coord_sys.rot = euler_to_rot(phi=90, theta=90, order='zyz', deg=True)
ant_device_rx1.coord_sys.pos = (45,90,1.5)
ant_device_rx1.coord_sys.lin = (0.,7.,0)
ant_device_rx1.coord_sys.update()
ant_device_rx1.add_antennas(mode_name='mode1',load_pattern_as_mesh=True,scale_pattern=10)
ant_device_rx1.add_mode(mode_name='mode1')

for each in ant_device_rx1.all_antennas_properties:
    all_scene_actors[each] = ant_device_rx1.all_antennas_properties[each]['Actor']

rComp = pem_core.RssPy.ResponseComposition.INDIVIDUAL
pem_core.isOK(api.setTxResponseComposition(ant_device_tx.modes['mode1'],rComp))

pem_core.isOK(api.setDoP2PCoupling(ant_device_tx.h_node_platform,ant_device_rx1.h_node_platform ,True))

# assign modes to devices
print(api.listGPUs())
devIDs = [0]; devQuotas = [0.8]; # limit RTR to use 80% of available gpu memory
pem_core.isOK(api.setGPUDevices(devIDs,devQuotas))
maxNumRayBatches = 25
pem_core.isOK(api.autoConfigureSimulation(maxNumRayBatches))

# initialize solver settings
pem_core.isOK(api.setMaxNumRefl(5))
pem_core.isOK(api.setMaxNumTrans(1))
pem_core.isOK(api.setTargetRayDensity(0.1)) # global ray density setting

# optional check if RSS is configured
# this will also be checked before response computation
if not api.isReady():
  print("RSS is not ready to execute a simulation:\n")
  print(api.getLastWarnings())

# we are really only running 1 frame, nothing is moving
fps = 1; dt = 1/fps
T = 4; numFrames = int(T/dt)

which_mode = ant_device_tx.modes['mode1']  # tell it which mode we want to get respones from
ant_device_tx.waveforms['mode1'].get_response_domains(which_mode)
rng_domain = ant_device_tx.waveforms['mode1'].rng_domain *2 # this is calculated for round trip
print(f"Range domain max: {rng_domain[-1]}")
time_domain = rng_domain / 2.99792458e8
print(f"Time domain max: {time_domain[-1]}")

output_movie_name = os.path.join(output_path, 'out_vis.mp4')
modeler = ModelVisualization(all_scene_actors,
                             show_antennas=True,
                             overlay_results = False,
                             fps=fps,
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name)

all_results = []
modeler.pl.show_grid()
for iFrame in tqdm(range(numFrames)):
    time = iFrame * dt
    # update all coordinate systems
    for actor in all_scene_actors:
        all_scene_actors[actor].update_actor(time=time)
    pem_core.isOK(api.computeResponseSync())
    (ret, response) = api.retrieveP2PResponse(ant_device_tx.modes['mode1'],
                                              ant_device_rx1.modes['mode1'],
                                              pem_core.RssPy.ResponseType.FREQ_PULSE)
    all_results.append(response)
    modeler.update_frame()
modeler.close()
all_results = np.array(all_results)

######################
# Perceive EM Simulation Complete
#####################



######################
# Setup and Run Sionna RT
#####################

scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile
# scene.preview()

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
tx = Transmitter(name="tx",
                 position=[8.5,21,27])

# Add transmitter instance to scene
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx",
              position=[45,90,1.5],
              orientation=[0,0,0])

# Add receiver instance to scene
scene.add(rx)

tx.look_at(rx) # Transmitter points towards receiver


scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials

scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

# Select an example object from the scene
so = scene.get("Altes_Rathaus-itu_marble")

# Print name of assigned radio material for different frequenies
for f in [3.5e9, 2.14e9]: # Print for differrent frequencies
    scene.frequency = f
    print(f"\nRadioMaterial: {so.radio_material.name} @ {scene.frequency/1e9:.2f}GHz")
    print("Conductivity:", so.radio_material.conductivity.numpy())
    print("Relative permittivity:", so.radio_material.relative_permittivity.numpy())
    print("Complex relative permittivity:", so.radio_material.complex_relative_permittivity.numpy())
    print("Relative permeability:", so.radio_material.relative_permeability.numpy())
    print("Scattering coefficient:", so.radio_material.scattering_coefficient.numpy())
    print("XPD coefficient:", so.radio_material.xpd_coefficient.numpy())


# Compute propagation paths
paths = scene.compute_paths(max_depth=5,
                            num_samples=1e7)  # Number of rays shot into directions defined
                                              # by a Fibonacci sphere , too few rays can
                                              # lead to missing paths

paths.normalize_delays = False # without this, the first path is normalized to 0s

######################
# Complete Sionna RT
#####################


# Visualize paths in the 3D preview
scene.preview(paths, show_devices=True, show_paths=True) # Use the mouse to focus on the visualized paths
# Default parameters in the PUSCHConfig
subcarrier_spacing = 15e3
fft_size = 48
# Print shape of channel coefficients before the application of Doppler shifts
# The last dimension corresponds to the number of time steps which defaults to one
# as there is no mobility
print("Shape of `a` before applying Doppler shifts: ", paths.a.shape)

# Apply Doppler shifts
paths.apply_doppler(sampling_frequency=subcarrier_spacing, # Set to 15e3 Hz
                    num_time_steps=14, # Number of OFDM symbols
                    tx_velocities=[3.,0,0], # We can set additional tx speeds
                    rx_velocities=[0,7.,0]) # Or rx speeds

print("Shape of `a` after applying Doppler shifts: ", paths.a.shape)


# convert frequency response to range profile
freq_response = all_results[0,0,0,0] #center pulse?
num_bins = 512
range_profile = np.fft.ifft(np.fft.ifftshift(freq_response, axes=0), axis=0, n=num_bins) * num_bins
 # high to low
# plot range profile
range_profile = np.fmax(np.abs(range_profile), 1.e-30)
time_domain = np.linspace(0,time_domain[-1],num_bins)

# sort range_profile from largest to smallest, keep on the top N points. Use the same index values to create a new array of time_domain with the corroponding values
N=512
idxs = np.argsort(range_profile, axis=0)[::-1]  # high to low
range_profile = range_profile[idxs]
range_profile = range_profile[:N] # keep only the top N points
time_domain = time_domain[idxs]*1e9 #ns
time_domain = time_domain[:N]
range_profile = range_profile/np.max(range_profile) # normalize to 1







a, tau = paths.cir()
print("Shape of tau: ", tau.shape)



t = tau[0,0,0,:]/1e-9 # Scale to ns
a_abs = np.abs(a)[0,0,0,0,0,:,0]

a_max = np.max(a_abs)

range_profile = range_profile*a_max # make results the same scale, range profile has previously been normalized to 1

# Line of Sight between Tx/Rx is 82.1188 meters
# or 273.33ns at speed of light

# Add dummy entry at start/end for nicer figure
# t = np.concatenate([(0.,), t, (np.max(t)*1.1,)])
# a_abs = np.concatenate([(np.nan,), a_abs, (np.nan,)])
# pem_cir = np.concatenate([(np.nan,), pem_cir, (np.nan,)])
# And plot the CIR
plt.figure()
plt.title("Channel impulse response realization")
plt.stem(time_domain, range_profile,'g',markerfmt='gD',label='Perceive EM')
plt.stem(t, a_abs,'b',markerfmt='b',linefmt='--',label='Sionna RT')

# plt.xlim([0, np.max(t)])
# plt.ylim([-2e-6, a_max*1.1])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$")
plt.legend()
plt.show()

plt.figure()
plt.title("Channel impulse response - Perceive EM")
plt.stem(time_domain, range_profile,'g',markerfmt='gD',label='Perceive EM')


# plt.xlim([0, np.max(t)])
# plt.ylim([-2e-6, a_max*1.1])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$")
plt.legend()
plt.show()

#
cm = scene.coverage_map(max_depth=5,
                        diffraction=True, # Disable to see the effects of diffraction
                        cm_cell_size=(5., 5.), # Grid size of coverage map cells in m
                        combining_vec=None,
                        precoding_vec=None,
                        num_samples=int(20e6)) # Reduce if your hardware does not have enough memory
# # Create new camera
tx_pos = scene.transmitters["tx"].position.numpy()
bird_pos = tx_pos.copy()
bird_pos[-1] = 1000 # Set height of coverage map to 1000m above tx
bird_pos[-2]-= 0.01 # Slightly move the camera for correct orientation

# Create new camera
bird_cam = Camera("birds_view", position=bird_pos, look_at=tx_pos)

scene.add(bird_cam)


# Open 3D preview (only works in Jupyter notebook)
# scene.preview(coverage_map=cm)

cm.show(tx=0); # If multiple transmitters exist, tx selects for which transmitter the cm is shown

scene.render_to_file(camera=bird_cam,  # Also try camera="preview"
                     filename="scene.png",
                     resolution=[650, 500])