#
# Copyright ANSYS. All rights reserved.
#

#######################################
#######################################
# IMPORTANT! edit api_core.py to use the correct API, P2P API is needed for this script to run
#######################################
#######################################


import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.animation import PillowWriter
import PIL.Image
import os
import sys

sys.path.append("..")  # directory above this directory is where api_core exists

#######################################################################################################################
# IMORTANT! edit api_core.py to use the correct API, P2P API is needed for this script to run
#######################################################################################################################
import pem_utilities.pem_core as pem_core

api = pem_core.api

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import AntennaDevice, Waveform
from pem_utilities.load_deepsense_6g import DeepSense_6G
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs

#######################################################################################################################
# Input Parameters
#######################################################################################################################


# simulation parameters
max_num_refl = 3
max_num_trans = 0
ray_density = 0.01

# Perceive EM waveform (some parameters are also used for Sionna)
center_freq = 60.0e9
num_freqs = 512
bandwidth = 100e6

cpi_duration = 1e-3
num_pulse_CPI = 3
fps = 20  # output video frame rate

export_debug = False

#######################################################################################################################
# End Input Parameters
#######################################################################################################################

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

# output file will be stored in this directory
output_path = '../output/'
os.makedirs(output_path, exist_ok=True)
debug_logs = DebuggingLogs(output_directory=output_path)

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

# actor_env_name = all_actors.add_actor(filename='../models/Scenario1/Scenario1.json')
actor_vehicle_name = all_actors.add_actor(filename='../models/Audi_A1_2010/Audi_A1_2010.json')
actor_bs_name = all_actors.add_actor()

# initial positions of each actor. Any value that is not set, will be set to zero. We can set pos/lin/rot/ang for each


# We are going to create a radar that is attached the vehicle, the hierarchy will look like this:
#
#  Ego Vehicle 1 --> Radar Platform --> Radar Device --> Radar Antenna (Tx/Rx)
#


# Radar Device
# create radar device and antennas, the radar platform is loaded from json file. It is then created in reference to the
# ego vehicle node. The position of the device is place 2.5 meters in front of the vehicle, and 1 meter above the ground
# the device itself does not have any meshes attached to it.

# Load antennas, these can be loaded from json or defined within the script. This example will load within the script
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_pulse_CPI,
    "tx_multiplex": "SIMULTANEOUS",
    "mode_delay": "CENTER_CHIRP"}
pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
mode_name = 'mode1'

######################
# Tx
#####################
########################## Setup Radar Platform  ##########################

# we are not loading an antenna device from file, so this will just create a blank antenna device,
# as a child of radar_device. Instead of loading from a file, this will be manually setup using a combination
# of direct API calls and the AntennaDevice class
ant_device_rx = AntennaDevice(file_name=None, parent_h_node=all_actors.actors[actor_bs_name].h_node)
ant_device_rx.initialize_device()
ant_device_rx.coord_sys.pos = [0, 0, 1]
ant_device_rx.coord_sys.update()
waveform = Waveform(waveform_dict)
ant_device_rx.waveforms[mode_name] = waveform
# configure radar mode
h_mode = pem_core.RssPy.RadarMode()
ant_device_rx.modes[mode_name] = h_mode
pem_core.isOK(api.addRadarMode(h_mode, ant_device_rx.h_device))

# positive angles are to the left of the scene, negative to the right. They define beam index 64 as to the right (or negatve angle)
peak_beam_direction = [43.71, 41.43, 38.86, 38., 36.29, 35.43, 34.57, 33.71,
                       31.14, 29.43, 27.71, 26.86, 25.14, 25.14, 23.43, 22.57,
                       20., 19.14, 17.43, 15.71, 13.14, 13.14, 12.29, 10.57,
                       8.86, 8.00, 6.286, 4.571, 3.714, 1.143, 1.143, -3.143,
                       -3.143, -4., -4.857, -6.571, -7.429, -9.41, -10.85, -13.42,
                       -13.43, -14.29, -16.0, -17.71, -18.57, -21.14, -23.71, -22.85,
                       -23.71, -25.43, -29.71, -30.57, -30.57, -31.43, -32.28, -33.14,
                       -34.86, -36.57, -38.29, -39.14, -40.0, -40.86, -43.42, -43.28]

peak_beam_direction = np.array(peak_beam_direction)
# peak_beam_direction = np.flip(np.linspace(-45,50,num=64))
antennas_dict = {}
for idx, direction in enumerate(peak_beam_direction):
    rot = euler_to_rot(phi=direction, deg=True)
    rx_dict = {
        "type": "parametric",
        "operation_mode": "rx",
        "polarization": "VERTICAL",
        "hpbwHorizDeg": 5,
        "hpbwVertDeg": 120.,
        "position": [0.0, 0.0, 0.0],
        "rotation": rot
    }

    antennas_dict[f"Rx{idx}"] = rx_dict
ant_device_rx.add_antennas(mode_name=mode_name, load_pattern_as_mesh=True, scale_pattern=1, antennas_dict=antennas_dict)
ant_device_rx.set_mode_active(mode_name)
ant_device_rx.add_mode(mode_name)

# add antennas to scene for visulization
for each in ant_device_rx.all_antennas_properties:
    name = all_actors.add_actor(name=each, actor=ant_device_rx.all_antennas_properties[each]['Actor'])

######################
# Tx
####################
ant_device_tx = AntennaDevice(file_name=None, parent_h_node=all_actors.actors[actor_vehicle_name].h_node)
ant_device_tx.initialize_device()
waveform = Waveform(waveform_dict)
ant_device_tx.waveforms[mode_name] = waveform
ant_device_tx.coord_sys.pos = [0, 0, 1.8]
ant_device_tx.coord_sys.update()
# configure radar mode
h_mode = pem_core.RssPy.RadarMode()
ant_device_tx.modes[mode_name] = h_mode
pem_core.isOK(api.addRadarMode(h_mode, ant_device_tx.h_device))

tx_dict = {
    "type": "ffd",
    "file_path": "dipole.ffd",
    "operation_mode": "tx",
    "position": [0, 0, 0]
}

antennas_dict = {"Tx1": tx_dict}
ant_device_tx.add_antennas(mode_name=mode_name, load_pattern_as_mesh=True, scale_pattern=1, antennas_dict=antennas_dict)
ant_device_tx.set_mode_active(mode_name)
ant_device_tx.add_mode(mode_name)

# add antennas to scene for visulization
for each in ant_device_tx.all_antennas_properties:
    name = all_actors.add_actor(name=each, actor=ant_device_tx.all_antennas_properties[each]['Actor'])

rComp = pem_core.RssPy.ResponseComposition.INDIVIDUAL
pem_core.isOK(api.setTxResponseComposition(ant_device_tx.modes[mode_name], rComp))

pem_core.isOK(api.setDoP2PCoupling(ant_device_tx.h_node_platform, ant_device_rx.h_node_platform, True))
pem_core.isOK(api.setPrivateKey("FieldOfView", "360"))

# assign modes to devices
print(api.listGPUs())
devIDs = [0];
devQuotas = [0.9];  # limit RTR to use 80% of available gpu memory
pem_core.isOK(api.setGPUDevices(devIDs, devQuotas))
maxNumRayBatches = 25
pem_core.isOK(api.autoConfigureSimulation(maxNumRayBatches))

# initialize solver settings
pem_core.isOK(api.setMaxNumRefl(max_num_refl))
pem_core.isOK(api.setMaxNumTrans(max_num_trans))
pem_core.isOK(api.setTargetRayDensity(ray_density))  # global ray density setting

# optional check if RSS is configured
# this will also be checked before response computation
if not api.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(api.getLastWarnings())

# display setup
# print(api.reportSettings())

# get response domains, this returns the range, velocity, frequency, and pulse domains for the radar which can be used
# for post-processing and scaling axes
which_mode = ant_device_tx.modes[mode_name]  # tell it which mode we want to get respones from
ant_device_tx.waveforms[mode_name].get_response_domains(which_mode)

vel_domain = ant_device_tx.waveforms[mode_name].vel_domain
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain
freq_domain = ant_device_tx.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx.waveforms[mode_name].pulse_domain

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

responses = []

# # create a pyvista plotter
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             fps=fps,
                             camera_attachment=actor_bs_name,
                             camera_orientation='follow',
                             overlay_results=True)

car_angle = []

x_vals = np.linspace(-3.1, 2.5, 7)
y_vals = np.linspace(2.3, 2.5, 3)
phi_vals = np.linspace(-2, 2,5 )
xy_vals = []
all_responses = []
for x in x_vals:
    for y in y_vals:
        for p in phi_vals:
            xy_vals.append([x, y])
            all_actors.actors[actor_bs_name].coord_sys.pos = [x, y, 0]
            all_actors.actors[actor_bs_name].coord_sys.rot = euler_to_rot(phi=p)
            all_actors.actors[actor_bs_name].coord_sys.update()

            print('running simulation...')
            # load deepsense 6g dataset
            for sequence_index in range(1, 2):
                ds_6g = DeepSense_6G(csv_file='scenario1.csv',
                                     scenario_folder='C:/Users/asligar/OneDrive - ANSYS, Inc/Documents/Applications/DeepSense/Scenario1/')
                ds_6g.load(seq_index=sequence_index)
                time_stamps = ds_6g.time_stamps
                scenario_time_step = time_stamps[1] - time_stamps[0]
                num_time_stamps = ds_6g.num_time_stamps
                time_idx = 0
                responses = []
                for time in tqdm(time_stamps):
                    # update all coordinate systems

                    all_actors.actors[actor_vehicle_name].coord_sys.pos = ds_6g.get_position(time)
                    car_angle.append(np.rad2deg(np.arctan(all_actors.actors[actor_vehicle_name].coord_sys.pos[1] /
                                                          all_actors.actors[actor_vehicle_name].coord_sys.pos[0])))
                    all_actors.actors[actor_vehicle_name].coord_sys.rot = ds_6g.get_rotation(time)
                    all_actors.actors[actor_vehicle_name].update_actor(time=time)

                    pem_core.isOK(api.computeResponseSync())
                    (ret, response) = api.retrieveP2PResponse(ant_device_tx.modes['mode1'], ant_device_rx.modes['mode1'],
                                                              pem_core.RssPy.ResponseType.FREQ_PULSE)
                    # calculate response in dB to overlay in pyvista plotter
                    # imData = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))
                    # modeler.mpl_ax_handle.set_data(imData) # update pyvista matplotlib plot
                    responses.append(response)
                    if export_debug:
                        debug_camera.generate_image()
                        # generate radar camera image debug_camera.current_image now has this image frame,
                        # debug_camera.camera_images has all the images from all frames
                        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file

                    frame_image = PIL.Image.open(ds_6g.frame_image[time_idx])

                    modeler.mpl_ax_handle.set_data(frame_image)

                    modeler.update_frame()
                    time_idx += 1

            responses = np.array(responses)


            all_results_abs = np.squeeze(np.abs(responses[:, :, :, 1, 256]))  # single chirp and frequency
            all_results_abs_norm = []
            all_pos = []
            for n in range(len(all_results_abs)):
                all_results_abs_norm.append(all_results_abs[n] / np.max(all_results_abs[n]))
            all_results_abs_norm = np.array(all_results_abs_norm)
            # all_results_abs_norm = np.fliplr(all_results_abs_norm) # index 64 is pointing to right side of scene(which is negative angles), opposite to how they were defined

            all_results_abs_measure_norm = []
            for n in range(len(ds_6g.power_vs_index)):
                all_results_abs_measure_norm.append(ds_6g.power_vs_index[n] / np.max(ds_6g.power_vs_index[n]))
            all_results_abs_measure_norm = np.array(all_results_abs_measure_norm)
            plt.close('all')

            number_of_frames = all_results_abs.shape[0]
            index = np.arange(1, 65, 1)

            fig, (ax, ax2) = plt.subplots(1, 2, dpi=100, figsize=(10, 4))
            # fig = plt.figure(figsize=(8, 4.25), dpi=100)
            # fig.tight_layout()
            # ax = plt.axes()
            ax.grid(zorder=0)
            ax.set_facecolor((20 / 255, 30 / 255, 47 / 255))
            # ax.xaxis.label.set_color('white')
            # ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            fig.patch.set_facecolor((33 / 255, 48 / 255, 56 / 255))
            plt.title("mmWave Beam Power Plot", color='white')
            ax.set_xlabel("Beam Index", fontweight='bold', color='white')
            ax.set_ylabel("Normalized Power", fontweight='bold', color='white')
            ax.set_xlim(index[0], index[-1])

            barcollection = ax.bar(index, all_results_abs_norm[0], zorder=3, label='Synthetic Data')
            barcollection2 = ax.bar(index, all_results_abs_measure_norm[0], zorder=3, label='Measured Data', alpha=.5)
            ax.legend()
            frame_image = PIL.Image.open(ds_6g.frame_image[0])
            ax2_handle = ax2.imshow(frame_image)


            def animate(i):
                idx_max = np.argmax(all_results_abs_norm[i])
                antenna_beam_selected = peak_beam_direction[idx_max]

                frame_image = PIL.Image.open(ds_6g.frame_image[i])
                ax2_handle.set_data(frame_image)
                y = all_results_abs_norm[i]
                y2 = all_results_abs_measure_norm[i]
                for n, b in enumerate(barcollection):
                    b.set_height(y[n])

                for n, b in enumerate(barcollection2):
                    b.set_height(y2[n])

                ax.set_title(f"mmWave Beam Power Plot, Time Index:{i}", color='white')


            animation = anim.FuncAnimation(fig, animate, number_of_frames, blit=False)

            # plt.show()


            f = f'./beam_power_16x1_seqindex_{sequence_index}_{x}_{y}_{p}.gif'
            # writervideo = anim.FFMpegWriter(fps=fps)
            # animation.save(f, writer=writervideo)
            animation.save(f, writer=PillowWriter(fps=25))
