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
import PIL.Image
import os
import sys

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx, add_single_rx, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.stk_utils import STK_Utils, STK_Results_Reader
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 

import_stk_results = False
fps = 1
total_time = 2

# waveform to plot
center_freq = 0.9e9
num_freqs = 128
bandwidth = 50e6
cpi_duration = 1000e-3
num_pulse_CPI = 101

# simulation parameters
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 3
max_num_trans = 0
ray_density = .1

export_debug = True #use this to export the movie at the end of the simulation (.gif showing radar camera and response)

# material manager used to load predefined materials, defined in material_library.json. This will load all the materials
# and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
mat_manager = MaterialManager()

time_stamps = np.linspace(0,total_time,int(total_time*fps))

# create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
all_actors = Actors()

agi_hq_name = all_actors.add_actor(filename =os.path.join(paths.models,'agi_hq/agi_hq.json'))
car_name = all_actors.add_actor(filename =os.path.join(paths.models,'Toyota_Prius_2012/Toyota_Prius_2012.json'))
actor_tx_name = all_actors.add_actor()
actor_rx_name = all_actors.add_actor()

all_actors.actors[agi_hq_name].coord_sys.rot = euler_to_rot(phi=0,theta=180,psi=180)
all_actors.actors[agi_hq_name].coord_sys.update()
# IDENTIFY THE OBJECTS IN THE STK SCENARIO
#
# - We first define the Global Coordinate System
#
global_object_path = 'Facility/Facility1'     # Global Coordinate STK Object
global_cs = "Body"                       # Global Coordinate Axes from AWB
#
#   We then assign PY script variables for the intended STK Objects defined below
#
#   THE OBSERVER NAME
#
stk_scaling=1e3
stk_tx = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)
stk_rx = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)

stk_car = STK_Utils(global_object_path,
                     global_cs,
                     dataFrequency=1/fps,
                     time_start_idx=None,
                     time_stop_idx=None,
                     scaling=stk_scaling)


# IDENTIFY THE CORRESPONDING STK OBJECT NAME AND AXES
#
#        This is the STK OBSERVER Platform and Coordinate System
local_object_path_tx = 'Place/Place1/Sensor/Sensor2'
local_cs_tx = 'Body'
stk_tx.getDataFromSTK(local_object_path_tx, local_cs_tx)

local_object_path_rx = 'Place/Place2/Sensor/Sensor1'
local_cs_rx = 'Body'
stk_rx.getDataFromSTK(local_object_path_rx, local_cs_rx)


local_object_path_car = 'GroundVehicle/GroundVehicle1'
local_cs_car = 'Body'
stk_car.getDataFromSTK(local_object_path_car, local_cs_car)

# Antenna Device

mode_name = 'mode1' # name of mode so we can reference it in post processing
# input_power_dbm = 10.0 # dBm
# # convert to watts
# input_power_watts = 10 ** ((input_power_dbm - 30) / 10)
input_power_watts = 1
# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more
# flexibility and control over the parameters, without having to create/modify a json file. This is the same as the
# results as if the same parameters had been created in a json file.
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "FreqPulse",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_pulse_CPI,
    "tx_multiplex": "INDIVIDUAL",
    "mode_delay": "CENTER_CHIRP",
    "tx_incident_power": input_power_watts}

pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)


######################
# Add 2 antenna to teh scene, one tx and one rx
####################


# add a single Tx antenna to the scene at tx_pos
ant_device_tx = add_single_tx(all_actors,waveform,mode_name,
                              pos=[0,0,0],
                              ffd_file='dipole.ffd',
                              scale_pattern=20,
                              parent_h_node=all_actors.actors[actor_tx_name].h_node)

######################
# Rx on walking user
####################

ant_device_rx = add_single_rx(all_actors,waveform,mode_name,
                              pos=[0,0,0],
                              ffd_file='dipole.ffd',
                              scale_pattern=20,
                              parent_h_node=all_actors.actors[actor_rx_name].h_node)

enable_coupling(mode_name,ant_device_tx, ant_device_rx)

# rays can be defined as density or spacing, this is the conversion between the two, if ray_density is set, then
freq_center = ant_device_tx.waveforms[mode_name].center_freq
lambda_center = 2.99792458e8 / freq_center
ray_spacing = np.sqrt(2) * lambda_center / ray_density

print(pem.listGPUs())
sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
sim_options.auto_configure_simulation()

which_mode = ant_device_tx.modes[mode_name] # tell it which mode we want to get respones from
ant_device_tx.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device_tx.waveforms[mode_name].vel_domain
rng_domain = ant_device_tx.waveforms[mode_name].rng_domain
freq_domain = ant_device_tx.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx.waveforms[mode_name].pulse_domain

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_logs = DebuggingLogs(output_directory=paths.output)
    debug_camera = DebuggingCamera(hMode=ant_device_tx.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

dt = 1/fps
num_frames = int(total_time/dt)

responses = []

# # create a pyvista plotter
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             freq_domain=freq_domain,pulse_domain=pulse_domain,
                             fps=fps,
                             camera_attachment=None,
                             camera_orientation=None)

print('running simulation...')
for frame_idx in tqdm(range(num_frames)):
    time = frame_idx*dt
    # update all coordinate systems


    all_actors.actors[actor_tx_name].coord_sys.pos = stk_tx.pos[frame_idx]
    all_actors.actors[actor_tx_name].coord_sys.rot = stk_tx.rot[frame_idx]
    all_actors.actors[actor_rx_name].coord_sys.pos = stk_rx.pos[frame_idx]
    all_actors.actors[actor_rx_name].coord_sys.rot = stk_rx.rot[frame_idx]
    all_actors.actors[car_name].coord_sys.pos = stk_car.pos[frame_idx]
    all_actors.actors[car_name].coord_sys.rot = stk_car.rot[frame_idx]

    all_actors.actors[actor_tx_name].coord_sys.update()
    all_actors.actors[actor_rx_name].coord_sys.update()
    all_actors.actors[car_name].coord_sys.update()

    # write out scene summary, only writing out first and last frame, useful for debugging

    pem_api_manager.isOK(pem.computeResponseSync())
    (ret, response) = pem.retrieveP2PResponse(ant_device_tx.modes[mode_name],ant_device_rx.modes[mode_name], RssPy.ResponseType.FREQ_PULSE)
    # calculate response in dB to overlay in pyvista plotter
    im_data = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))
    if export_debug:
        if frame_idx == 0:
            debug_logs.write_scene_summary(file_name=f'out_{frame_idx}.json')
        debug_camera.generate_image()
        # generate radar camera image debug_camera.current_image now has this image frame,
        # debug_camera.camera_images has all the images from all frames
        # debug_camera.write_image_to_file(f'out_{iFrame}.png') # write current image to file
        responses.append(response)
    modeler.update_frame(plot_data=im_data, plot_limits=[im_data.max() - 60, im_data.max()])
# modeler.close()

# post-process images into gif

if export_debug:
    # debug_camera.write_camera_to_gif(file_name='camera.gif')
    # debug_camera.write_response_to_gif(responses,file_name='response.gif',rng_domain=rng_domain,vel_domain=vel_domain)
    debug_camera.write_camera_and_response_to_gif(responses,
                                                  file_name='camera_and_response.gif',
                                                  rng_domain=rng_domain,
                                                  vel_domain=vel_domain)

responses = np.array(responses)
plt.close('all')

if import_stk_results:
    reader = STK_Results_Reader()
    rfcm_chandata = reader.read_rsp()
    # RFCM data
    links = rfcm_chandata.keys()
    print("Available Link Data: \n")
    for mylink in links:
        print(mylink)

    # Begin Processing ---------------------------------------------------------------------------------------
    for mylink in links:
        # for mylink in myspeciallink:
        print("Working on Link: " + mylink)
        respdata, Tdat, freqsweepvec, timesweepvec, fc, bw = reader.get_link_data(mylink)

        # Compile the complete data structure and fill into structured array respdata -------

        Tsamps, nTx, nRx, nPulses, nSamples = respdata.shape

        for myTx in range(0, nTx):
            for myRx in range(0, nRx):
                print('Running Tx, Rx: ' + str(myTx) + ',' + str(myRx))
                # Plot the FD data (Re, Im) for the first time interval, center sounding/pulse
                plt.figure()
                plt.plot(freqsweepvec, 20*np.log10(np.abs(respdata[0][0][0][int(np.floor(nPulses / 2))][:])),
                         label='RFCM')

                plt.plot(freqsweepvec, 20*np.log10(np.abs(responses[0,0,0,int(np.floor(nPulses / 2)),:])),
                         label='Perceive EM')
                plt.title('Freq Domain - First Sweep')
                plt.ylabel('S-parameter forward coupling (dB)')
                plt.xlabel('Channel Frequency (Hz)')
                plt.legend()
                plt.tight_layout()
                plt.show()



                # Plot the FD data (Re, Im) for the first time interval, center sounding/pulse
                plt.figure()
                plt.plot(freqsweepvec, np.real(respdata[0][0][0][int(np.floor(nPulses / 2))][:]),
                         label='RFCM - Real')
                plt.plot(freqsweepvec, np.imag(respdata[0][0][0][int(np.floor(nPulses / 2))][:]),
                         label='RFCM - Imag')
                plt.plot(freqsweepvec, np.real(responses[0,0,0,int(np.floor(nPulses / 2)),:]),
                         label='Perceive EM - Real')
                plt.plot(freqsweepvec, np.imag(responses[0,0,0,int(np.floor(nPulses / 2)),:]),
                         label='Perceive EM - Imag')
                plt.title('Freq Domain - First Sweep')
                plt.ylabel('S-parameter forward coupling (Re, Im)')
                plt.xlabel('Channel Frequency (Hz)')
                plt.legend()
                plt.tight_layout()
                plt.show()


np.save(paths.output + 'responses.npy', responses)
np.save(paths.output + 'freq_domain.npy', freq_domain)
np.save(paths.output + 'pulse_domain.npy', pulse_domain)
np.save(paths.output + 'simulation_timestamps.npy', time_stamps)



