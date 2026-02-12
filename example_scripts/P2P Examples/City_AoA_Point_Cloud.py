import json
from tqdm import tqdm
import pyvista as pv
import numpy as np
import os
import sys
import time as walltime
import utm
import matplotlib.pyplot as plt

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import  Waveform, add_single_tx, add_single_rx, enable_coupling, AntennaArray
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.open_street_maps_geometry import BuildingsPrep, TerrainPrep, get_z_elevation_from_mesh, find_random_location
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.post_processing import create_target_list
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 

# range of values that lat/lon can be generated around
scene_lat_lon = (40.747303, -73.985701) #nyc

# simulation timing parameters
# time step for simulation, this is the time step that the solver will use, within each time step the solver will do the
# number of channel soundings defined by num_soundings_per_sim, at an update rate of cpi_duration/num_soundings_per_sim
dt = 0.1
include_mobility = True # if True, the rx antennas will move around the scene, assigned a random velocity vector
# how many time steps to wait before changing the random selected position for each rx, if include_mobility
# is set to True. This means we will run total_time = time_steps_per_positions * dt for the entire simulation
time_steps_per_position = 300
radius = 500
# waveform parameters
center_freq = 3e9
num_freqs = 512
bandwidth = 150e6
cpi_duration = 100e-3 # a simulation will issue num_soundings_per_sim channel soundings within this time
num_soundings_per_sim = 101 # this will result in a pulse interval of 1ms

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5 # how many reflections to compute
max_num_trans = 1 # how many transmissions to compute
ray_density = .4 # ray density as defined at 100m. Alternatively we can set the ray_spacing parameter or assign adaptive ray spacing to each object in the scene

# export debug logs and show modeler for visualization
export_debug = True


# Tx Antenna positions (z will be adjusted to be above the highest mesh at xy position)
ant_1_pos = [-48 ,6 ,3]
rx_pos = [-170, 70, 1.5] # always place the Rx antennas at 2m height of highest mesh at xy position


print(f'Generating Scene: Lat/Lon: {scene_lat_lon}...')
output_path = paths.output
terrain_prep = TerrainPrep(output_path)
terrain = terrain_prep.get_terrain(center_lat_lon=scene_lat_lon, max_radius=radius, grid_size=5, buffer_percent=0,flat_surface=False,shape='rectangle')

buildings_prep = BuildingsPrep(output_path)
# terrain is not yet created, I will create it later, using the exact same points as used for the heatmap surface
building_image_path = os.path.join(output_path, 'buildings.png')
buildings = buildings_prep.generate_buildings(scene_lat_lon, terrain_mesh=terrain['mesh'], max_radius=radius,
                                              export_image_path=building_image_path)


all_meshes_for_elevation_check = [buildings['mesh'],terrain['mesh']]

#######################################################################################################################
# Setup and Run Perceive EM
#######################################################################################################################

debug_logs = DebuggingLogs(output_directory=output_path)



mat_manager = MaterialManager()
all_actors = Actors()  # all actors, using the same material library for everyone


buildings_name = all_actors.add_actor(filename=buildings['file_name'],
                                 mat_idx=mat_manager.get_index('concrete'),
                                 color='grey',transparency=0.0)

terrain_name = all_actors.add_actor(filename=terrain['file_name'],
                                 mat_idx=mat_manager.get_index('asphalt'),
                                 color='black',transparency=0.5)



# Instead of defining the waveform parameters in a json file, I am going to do it manually here, this allows more flexibility
# and control over the parameters, without having to create/modify a json file
waveform_dict = {
    "mode": "PulsedDoppler",
    "output": "RangeDoppler",
    "center_freq": center_freq,
    "bandwidth": bandwidth,
    "num_freq_samples": num_freqs,
    "cpi_duration": cpi_duration,
    "num_pulse_CPI": num_soundings_per_sim,
    "tx_multiplex": "SIMULTANEOUS",
    "mode_delay": "CENTER_CHIRP"}

pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
wavelength = 3e8 / center_freq
mode_name = 'mode1'
range_pixels = 512
doppler_pixels = 256

# waveform will be used for both Tx and Rx
waveform = Waveform(waveform_dict)


#################
# Tx antenna

# ant_device_tx1 = add_single_tx(all_actors,waveform,mode_name,pos=ant_1_pos,ffd_file='dipole.ffd',scale_pattern=20.0)
ant_array = AntennaArray(all_actors=all_actors,
                            name = 'array',
                            waveform = waveform,
                            mode_name = mode_name,
                            file_name = None,
                            beamwidth_H = 140,
                            beamwidth_V = 120,
                            polarization = 'V',
                            rx_shape = [0, 0],
                            tx_shape = [2, 2],
                            spacing_wl_x = 0.5,
                            spacing_wl_y = 0.5,
                            load_pattern_as_mesh = True,
                            scale_pattern = 10,
                            parent_h_node = None,
                            normal = 'x',
                            range_pixels=range_pixels,
                            doppler_pixels=doppler_pixels)

ant_device_tx1 = ant_array.antenna_device

z_intersect = get_z_elevation_from_mesh(ant_1_pos[:2], all_meshes_for_elevation_check)
ant_1_pos[2] += np.max(z_intersect)
ant_device_tx1.coord_sys.pos = np.array(ant_1_pos)
ant_device_tx1.coord_sys.rot = euler_to_rot(phi=135,theta=-15)
ant_device_tx1.coord_sys.update()

#################
# Rx antennas

# always place the antennas at 1.5m height
z_intersect = get_z_elevation_from_mesh(rx_pos[:2], all_meshes_for_elevation_check)
rx_pos[2] += np.max(z_intersect)
rx_actor_name = all_actors.add_actor()
all_actors.actors[rx_actor_name].coord_sys.pos = np.array(rx_pos)
if include_mobility:
    # randomize the velocity vector of actor,
    lin_x = np.random.uniform(low=-3, high=3)
    lin_y = np.random.uniform(low=-3, high=3)
    all_actors.actors[rx_actor_name].coord_sys.lin = np.array([lin_x, lin_y, 0])
all_actors.actors[rx_actor_name].coord_sys.update()
# import this antenna into an existing actor, we can just move the actor and the antenna will move with it
ant_device_rx = add_single_rx(all_actors,waveform,mode_name,parent_h_node=all_actors.actors[rx_actor_name].h_node,ffd_file='dipole.ffd')
# between all the existing tx, and rx antennas, which ones do we want to compute coupling between
enable_coupling(mode_name,ant_device_tx1, ant_device_rx)


# setup can be ray density or ray spacing, if ray density is set, it will override ray spacing
if ray_density is not None:
    freq_center = ant_device_tx1.waveforms[mode_name].center_freq
    lambda_center = 2.99792458e8 / freq_center
    ray_spacing = np.sqrt(2) * lambda_center / ray_density


sim_options = SimulationOptions()
sim_options.ray_spacing = ray_spacing
sim_options.max_reflections = max_num_refl
sim_options.max_transmissions = max_num_trans
sim_options.go_blockage = go_blockage
sim_options.field_of_view = 360
sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
# you must set GPU device to use, if you have multiple GPUs, you can select which one to use or use multiple with [0,1...etc]
# sim_options.gpu_device = 0
sim_options.auto_configure_simulation()


# if we want to query the domain information for the solution, we can do that all for just tx1 becuase they all use
# the same waveform, they will be all the same. In this case, these domains are just used for plotting
which_mode = ant_device_tx1.modes[mode_name] # tell it which mode we want to get respones from
ant_device_tx1.waveforms[mode_name].get_response_domains(which_mode)
vel_domain = ant_device_tx1.waveforms[mode_name].vel_domain
rng_domain = ant_device_tx1.waveforms[mode_name].rng_domain*2 # this is round trip, multiply by 2 to get one way
freq_domain = ant_device_tx1.waveforms[mode_name].freq_domain
pulse_domain = ant_device_tx1.waveforms[mode_name].pulse_domain

fps = 1 / dt

print(f'Maximum Range: {rng_domain[-1]} meters')

# activate radar camera view, this is useful for debugging. This is what the solver sees, no need to activate this
# if we don't care about seeing the output. There is a cost with doing this, for high performance don't activate
if export_debug:
    debug_camera = DebuggingCamera(hMode=ant_device_tx1.modes[mode_name],
                                   display_mode='coating',
                                   output_size=(512, 512),
                                   background_color=255,
                                   frame_rate=fps)

output_movie_name = os.path.join(output_path, 'out_vis_synt_data_gen.mp4')

# going to create a rolling plot that is num_rx * time_steps_per_position *10 long
output_size = (num_soundings_per_sim, num_freqs)
modeler = ModelVisualization(all_actors,
                             show_antennas=True,
                             overlay_results=False,
                             fps=fps,
                             shape=output_size,
                             x_domain=np.linspace(-600,600,num=512),
                             y_domain=np.linspace(-600,600,num=512),
                             camera_orientation=None,
                             camera_attachment=None,
                             output_movie_name=output_movie_name,)

modeler.pl.show_grid()
print('running simulation...')



timestamps = np.linspace(0, time_steps_per_position*dt-dt, time_steps_per_position)

all_data = []
for time in timestamps:
    all_responses = []
    # update all coordinate systems
    for actor in all_actors.actors:
        all_actors.actors[actor].update_actor(time=time)

    start_sim_time = walltime.time()
    pem_api_manager.isOK(pem.computeResponseSync())
    end_sim_time = walltime.time()

    # print out the simulation time, this is the time it takes to run the simulation.
    # I am printing this out because the overhead in the model visualization can be significant and is not related
    # to Perceive EM simulation resources
    print(f'Simulation Time: {end_sim_time - start_sim_time} seconds per {num_soundings_per_sim} soundings')

    start_ret_time = walltime.time()

    # retrieve the response for each Rx device, this will return between tx and rx devices. If we have multiple
    # tx or rx on each device, all combinations will be returned. In this case, we only have one tx and one rx
    # per device, so we will get one response per device pair

    (ret, response) = pem.retrieveP2PResponse(ant_device_tx1.modes[mode_name],
                                          ant_device_rx.modes[mode_name],
                                          RssPy.ResponseType.RANGE_DOPPLER)

    end_ret_time = walltime.time()

    # below is just displaying the data in the modeler, this is not needed for the simulation, but useful for
    # visualization
    print(f'Retrieval Time: {end_ret_time - start_ret_time} seconds')




    rd_all_channels_az = response[0:2, 0]
    rd_all_channels_el = response[::2, 0]
    target_list, fps_create_target = create_target_list(rd_all_channels_az=rd_all_channels_az,
                                                       rd_all_channels_el=rd_all_channels_el,
                                                       rngDomain=rng_domain,
                                                       velDomain=vel_domain,
                                                       azPixels=512, elPixels=512,
                                                       antenna_spacing_wl=0.5,
                                                       radar_fov_az=[-90, 90],
                                                       radar_fov_el=[-90, 90],
                                                       centerFreq=center_freq,
                                                       rcs_min_detect=0,
                                                       min_detect_range=7.5,
                                                       rel_peak_threshold=1e-2,
                                                       max_detections=200,
                                                       return_cfar=False)

    # modeler.mpl_ax_handle.set_data(range_angle_response.T)  # update pyvista matplotlib plot
    # max_of_data = np.max(range_angle_response)
    # modeler.mpl_ax_handle.set_clim(vmin=max_of_data-30, vmax=max_of_data)

    # velocity_min = -10 and max is just for visualization purposes, to scale color of the point cloud
    modeler.add_point_cloud_to_scene(target_list,
                                     tx_pos=ant_1_pos,
                                     tx_rot=ant_device_tx1.coord_sys.rot,
                                     color_min=-100,
                                     color_max=-40,
                                     color_mode='p_received',
                                     size_mode='p_received',)
    x = []
    y = []
    all_mag = []
    max_size = 2
    for target in target_list:
        x.append(target_list[target]['xpos'])
        y.append(target_list[target]['ypos'])
        p_rec = target_list[target]['p_received']
        p_rec_db = 10*np.log10(p_rec)
        all_mag.append(p_rec_db)
    p_rec_db = np.array(all_mag)
    p_rec_db_norm = (p_rec_db - np.min(p_rec_db)) / (np.max(p_rec_db) - np.min(p_rec_db))

    p_rec_db_norm = -100/p_rec_db*10 # set a baseline size for all points relative to the numerator

    if time==0:
        f, ax = plt.subplots(tight_layout=True)
        scat = ax.scatter(x,y,s=p_rec_db_norm*7,c=p_rec_db,vmin=-100, vmax=-40,cmap='jet')
        ax.set_xlim(-600,600)
        ax.set_ylim(-600,600)
        h_chart = pv.ChartMPL(f, size=(0.275, 0.375), loc=(0.7, 0.6))
        h_chart.background_color = (0.5, 0.5, 0.5)
        modeler.pl.add_chart(h_chart)
    else:
        scat.set_offsets(np.c_[x,y])
        scat.set_sizes(p_rec_db_norm*7)
        scat.set_array(p_rec_db)


    modeler.update_frame() # update the visualization and save the image as part of a video

    # exporting radar camera images
    if export_debug:
        if time==0: # export a debug log only for first frame
            debug_logs.write_scene_summary(file_name='out_0.json')
        debug_camera.generate_image()



modeler.close()
if export_debug: # this will create a gif of the output as rendered from the view of the Tx antenna
    debug_camera.write_camera_to_gif(file_name='camera.gif')
