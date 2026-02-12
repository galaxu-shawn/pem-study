# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:51:32 2021

@author: asligar
"""

import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits import mplot3d
import PIL.Image
import pyvista as pv
import os
import sys
import uuid
import shutil

from pathlib import Path
# Dynamically resolve the path to the api_core module
script_path = Path(__file__).resolve()
api_core_path = script_path.parent.parent  # Adjusted to point to the correct parent directory
model_path = os.path.join(script_path.parent.parent, 'models')
output_path = os.path.join(script_path.parent.parent, 'output')
# output file will be stored in this directory
os.makedirs(output_path, exist_ok=True)
if api_core_path not in sys.path:
    sys.path.insert(0, str(api_core_path))
import pem_utilities.pem_core as pem_core
RssPy = pem_core.RssPy
api = pem_core.api

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot

from pem_utilities.antenna_device import  Waveform, add_single_tx, add_single_rx, enable_coupling
from pem_utilities.heat_map import HeatMap
from pem_utilities.utils import generate_rgb_from_array
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.open_street_maps_geometry import BuildingsPrep, TerrainPrep, find_random_location, convert_heatmap_to_vtk
from pem_utilities.simulation_options import SimulationOptions

# output file will be stored in this directory

temp_path = os.path.join(output_path,'tmp_cache')
os.makedirs(temp_path, exist_ok=True)
# total variation run will be num_scenes * num_variations_per_scene
num_scenes = 5
num_variations_per_scene = 2
max_radius = 30
z_tx_pos = 1.5
# range of values that lat/lon can be generated around
scene_lat_lon_range = (40.713444,40.762394,-74.008942, -73.957211)
polarization = 'Z'

sampling_spacing_wl = 10

# simulation options
go_blockage = 1 # set to -1 if no GO blockage, set to 0 or higher for GO blockage
max_num_refl = 5
max_num_trans = 0
ray_density = 0.1

center_freq = 3.5e9
num_freqs = 2
bandwidth = 200e6
cpi_duration = 0.9e-3
num_pulse_CPI = 2

show_modeler = True
copy_results_for_sim_ai_inputs= True
export_debug = False

#######################################################################################################################
# Input Parameters
#######################################################################################################################
scene_has_been_initialized = False
output_id = 'Gen_' + str(uuid.uuid4().hex[:6])
output_path = os.path.join(output_path, output_id)
os.makedirs(output_path, exist_ok=True)

if copy_results_for_sim_ai_inputs:
    output_path_sim_ai = os.path.join(output_path, 'sim_ai_results')
    output_path_sim_ai = os.path.join(output_path_sim_ai, output_id)
    if not os.path.exists(output_path_sim_ai):
        os.makedirs(output_path_sim_ai)



summary_dict = {'output_id': output_id}
summary_filename = output_id + '_summary.json'
summary_output_filename = os.path.join(output_path, summary_filename)

for scene_idx in range(num_scenes):
    print(f'Generating Scene {scene_idx}...')
    # lat_lon = (40.739524, -73.990127)
    lat = np.random.uniform(low=scene_lat_lon_range[0], high=scene_lat_lon_range[1])
    lon = np.random.uniform(low=scene_lat_lon_range[2], high=scene_lat_lon_range[3])
    lat_lon = (lat, lon)
    print('Lat/Lon:', lat_lon)
    try:
        wl = 299792458.0 / center_freq
        grid_size = sampling_spacing_wl * wl
        buildings_prep = BuildingsPrep(temp_path)
        # terrain is not yet created, I will create it later, using the exact same points as used for the heatmap surface
        building_image_path = os.path.join(temp_path, 'buildings.png')
        buildings = buildings_prep.generate_buildings(lat_lon, terrain_mesh=None, max_radius=max_radius,
                                                      export_image_path=building_image_path)
        scene_idx_creation = True

    except:
        scene_idx_creation = False
        api.reset()
        pem_core.isOK(api.selectApiLicenseMode(pem_core.RssPy.ApiLicenseMode.PERCEIVE_EM))
        print('Scene Creation Failed, trying new lat/lon')



    for var_idx in range(num_variations_per_scene):

        print(f'Generating Variation {var_idx}...')
        scene_idx_str = str(scene_idx).zfill(4)
        var_idx_str = str(var_idx).zfill(4)

        instance_id = f'Result_{scene_idx_str}_{var_idx_str}'
        output_path_sim_ai_instance = os.path.join(output_path_sim_ai, f'{scene_idx_str}_{var_idx_str}')

        if scene_idx_creation == False:
            # generation of builing, or terrain has failed, mark all variations for this scene as failed
            summary_dict[instance_id] = 'Failed'
            break

        instance_path = os.path.join(output_path, instance_id)
        if not os.path.exists(instance_path):
            os.makedirs(instance_path)

        if copy_results_for_sim_ai_inputs:
            if not os.path.exists(output_path_sim_ai_instance):
                os.makedirs(output_path_sim_ai_instance)


        boundary_conditions_filename = 'boundary_conditions.json'

        meta_data_dict = {'scene_idx': scene_idx,var_idx: var_idx,
                          'instance_id': instance_id,
                          'output_id': output_id,
                          'lat_lon': None,
                          'max_radius': max_radius,
                          'tx_pos': [0,0,0],
                          'polarization': polarization,
                          'num_reflections': max_num_refl,
                          'num_transmissions': max_num_trans,
                          'ray_density': ray_density,
                          'go_blockage': go_blockage,
                          'sampling_spacing_wl': sampling_spacing_wl,
                          'waveform': None,
                          'output_path': instance_id,
                          'building_mesh': None,
                          'terrain_mesh': None,
                          'heatmap_data_npy': None,
                          'heatmap_xvals_npy': None,
                          'heatmap_yvals_npy': None,
                          'heatmap_data_png': None,
                          'scene_image': None,
                          'building_vtp': None,
                          'terrain_vtp': None,
                          'total_vtp': None,
                          'boundary_conditions_path':boundary_conditions_filename,
                          }



        meta_data_output_filename = os.path.abspath(os.path.join(instance_path, f'meta_data_{scene_idx_str}_{var_idx_str}.json'))

        meta_data_dict['lat_lon'] = lat_lon
        max_radius = max_radius

        heatmap_x_size = max_radius*2 #meters
        heatmap_y_size = max_radius*2 #meters

        try:
            # copy building_image_path to instance_path
            shutil.copy(building_image_path,os.path.join(instance_path,os.path.basename(building_image_path)))
            shutil.copy(buildings['file_name'], os.path.join(instance_path, os.path.basename(buildings['file_name'])))

            meta_data_dict['scene_image'] = os.path.basename(building_image_path)
            meta_data_dict['building_mesh'] = os.path.basename(buildings['file_name'])


            xy = find_random_location(buildings['mesh'],outdoors=True)
            if xy:
                tx_pos = [xy[0],xy[1], z_tx_pos]
                meta_data_dict['tx_pos'] = tx_pos
            else:
                print("No outdoor location found, exiting")
                summary_dict[instance_id] = 'Failed'
                api.reset()
                pem_core.isOK(api.selectApiLicenseMode(pem_core.RssPy.ApiLicenseMode.PERCEIVE_EM))
                break
        except:
            summary_dict[instance_id] = 'Failed'
            api.reset()
            pem_core.isOK(api.selectApiLicenseMode(pem_core.RssPy.ApiLicenseMode.PERCEIVE_EM))
            break


        # grid bounds where heatmap will be calculated [xmin,xmax,ymin,ymax]
        grid_bounds = [-heatmap_x_size/2, heatmap_x_size/2,
                       -heatmap_y_size/2, heatmap_y_size/2]
        rx_zpos = tx_pos[2]

        # Perceive EM waveform




        #######################################################################################################################
        # Setup and Run Perceive EM
        #######################################################################################################################

        debug_logs = DebuggingLogs(output_directory=output_path)

        mat_manager = MaterialManager()
        all_actors = Actors()  # all actors, using the same material library for everyone


        buildings_name = all_actors.add_actor(filename=buildings['file_name'],
                                         mat_idx=mat_manager.get_index('concrete'),
                                         color='grey',transparency=0.0)


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

        meta_data_dict['waveform'] = waveform_dict

        # waveform will be used for both Tx and Rx
        waveform = Waveform(waveform_dict)

        # initialize the antenna device, one for Tx, one for Rx
        antenna_device_tx = add_single_tx(all_actors,
                                          waveform,
                                          mode_name,
                                          pos=tx_pos,
                                          ffd_file='dipole.ffd',
                                          scale_pattern=10)


        heatmap = HeatMap(all_actors=all_actors,
                          sampling_spacing_wl=sampling_spacing_wl,
                          bounds=grid_bounds,
                          z_elevation=rx_zpos,
                          waveform=waveform,
                          mode_name=mode_name,
                          num_subgrid_samples_nx=10,
                          num_subgrid_samples_ny=10,
                          polarization=polarization,
                          show_patterns=False,
                          )

        enable_coupling(mode_name, antenna_device_tx, heatmap.probe_device)

        terrain_prep = TerrainPrep(temp_path)
        terrain = terrain_prep.get_terrain(all_grid_pos=heatmap.all_grid_and_subgrid_positions,
                                           flat_surface=True,
                                           shape='grid')
        terrain_name = all_actors.add_actor(filename=terrain['file_name'],
                                         mat_idx=mat_manager.get_index('asphalt'),
                                         color='black',transparency=0.5)
        shutil.copy(terrain['file_name'], os.path.join(instance_path, os.path.basename(terrain['file_name'])))
        meta_data_dict['terrain_mesh'] = os.path.basename(terrain['file_name'])
        if export_debug:
            debug_camera = DebuggingCamera(hMode=antenna_device_tx.modes['mode1'],
                                            display_mode='coating',
                                            output_size=(512,512),
                                            background_color=255,
                                            frame_rate=1)

        # assign modes to devices
        if ray_density is not None:
            lambda_center = 2.99792458e8 / antenna_device_tx.waveforms[mode_name].center_freq
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

        which_mode = antenna_device_tx.modes[mode_name]  # tell it which mode we want to get response from
        antenna_device_tx.waveforms[mode_name].get_response_domains(which_mode)

        # this is calculated for round trip, multiply by 2 to get one way
        rng_domain = antenna_device_tx.waveforms[mode_name].rng_domain * 2
        time_domain = rng_domain / 3e8

        print(f"Range domain max: {rng_domain[-1]}")
        print(f"Time domain max: {time_domain[-1]}")


        # video output speed
        fps = 100



        if show_modeler:
            output_movie_name = os.path.join(output_path, 'out_vis.mp4')
            modeler = ModelVisualization(all_actors,
                                         show_antennas=True,
                                         fps=fps,
                                         shape=(heatmap.total_samples_x, heatmap.total_samples_y),
                                         x_domain=heatmap.x_domain,
                                         y_domain=heatmap.y_domain,
                                         camera_attachment=None,
                                         camera_orientation=None,
                                         output_movie_name=output_movie_name)

            create_surface=True
            add_mesh_to_overlay=True
            modeler.pl.show_grid()
        else:
            modeler = None
            create_surface=False
            add_mesh_to_overlay=False

        print("Running Perceive EM Simulation...")

        heatmap.update_heatmap(tx_mode=antenna_device_tx.modes[mode_name],
                               probe_mode=heatmap.probe_device.modes[mode_name],
                               function='db',
                               modeler=modeler,
                               create_surface=create_surface,
                               add_mesh_to_overlay=add_mesh_to_overlay,
                               plot_min=-100,
                               plot_max=-50,
                               pulse_idx = 0,
                               freq_idx = 0,
                               include_complex_data=True)

        # tx_mode = None,
        # probe_mode = None,
        # function = 'db',
        # modeler = None,
        # create_surface = True,
        # sub_grid_updates = True,
        # plot_min = -100,
        # plot_max = -50,
        # add_mesh_to_overlay = True,
        # td_output_size = None,
        # window = 'flat',
        # pulse_idx = 0,
        # freq_idx = 0

        image = generate_rgb_from_array(heatmap.image,function='dB',
                                        plot_min=None,plot_max=None,
                                        plot_dynamic_range=None,
                                        resize_window=None,
                                        colormap='jet',
                                        smooth_image=False,
                                        show_image=False)

        terrain_mesh, building_mesh, total_mesh = convert_heatmap_to_vtk(heatmap.complex_data,
                                                   heatmap.all_grid_and_subgrid_positions,
                                                   building_mesh=buildings['mesh'],
                                                   terrain_mesh=terrain['mesh'])

        terrain_mesh.save(os.path.join(instance_path,'terrain.vtp'))
        building_mesh.save(os.path.join(instance_path, 'buildings.vtp'))

        total_mesh.save(os.path.join(instance_path, 'total.vtp'))
        meta_data_dict['terrain_vtp'] = 'terrain.vtp'
        meta_data_dict['building_vtp'] = 'buildings.vtp'
        meta_data_dict['total_vtp'] = f'total_{scene_idx_str}_{var_idx_str}_surface.vtp'


        np.save(os.path.join(instance_path,'x_vals.npy'),heatmap.x_domain)
        np.save(os.path.join(instance_path,'y_vals.npy'),heatmap.y_domain)
        np.save(os.path.join(instance_path,'data_complex.npy'),heatmap.complex_data)
        meta_data_dict['heatmap_data_npy'] = 'data_complex.npy'
        meta_data_dict['heatmap_xvals_npy'] = 'x_vals.npy'
        meta_data_dict['heatmap_yvals_npy'] = 'y_vals.npy'

        image=PIL.Image.fromarray(image)
        full_path = os.path.join(instance_path,'heatmap.png')
        image.save(full_path)

        meta_data_dict['heatmap_data_png'] = 'heatmap.png'

        if export_debug:
            debug_logs.write_scene_summary(file_name=f'out.json')
            debug_camera.generate_image()
            debug_camera.write_image_to_file('debug_camera.png')

        json_object = json.dumps(meta_data_dict, indent=4)
        with open(meta_data_output_filename, "w") as outfile:
            outfile.write(json_object)

        boundary_conditions_dict = {'tx_xpos':tx_pos[0],'tx_ypos':tx_pos[1],'tx_zpos':tx_pos[2],'rx_zpos':rx_zpos,'center_freq':center_freq}
        # write boundary conditions to file
        boundary_conditions_filename = os.path.join(instance_path, boundary_conditions_filename)
        json_object = json.dumps(boundary_conditions_dict, indent=4)
        with open(boundary_conditions_filename, "w") as outfile:
            outfile.write(json_object)

        if show_modeler:
            modeler.update_frame(write_frame=True) # if write_frame=False, no video will be created, just the modeler shown.
            modeler.close()

        summary_dict[instance_id] = 'Success'
        api.reset()
        pem_core.isOK(api.selectApiLicenseMode(pem_core.RssPy.ApiLicenseMode.PERCEIVE_EM))
        json_object = json.dumps(summary_dict, indent=4)
        with open(summary_output_filename, "w") as outfile:
            outfile.write(json_object)

        if copy_results_for_sim_ai_inputs:
            shutil.copy(os.path.join(instance_path, 'total.vtp'), os.path.join(output_path_sim_ai_instance, meta_data_dict['total_vtp']))
            shutil.copy(boundary_conditions_filename, os.path.join(output_path_sim_ai_instance, os.path.basename(boundary_conditions_filename)))
# json_object = json.dumps(summary_dict, indent=4)
# with open(summary_output_filename, "w") as outfile:
#     outfile.write(json_object)

