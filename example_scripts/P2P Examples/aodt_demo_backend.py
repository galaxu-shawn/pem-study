#
# Copyright ANSYS. All rights reserved.
#
#######################################
# Example of P2P coupling between a pedestrian and a building using multiple animated people


import numpy as np
import os
from pathlib import Path

from pem_utilities.materials import MaterialManager
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_diversity_antenna_pair, enable_coupling
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.simulation_options import SimulationOptions

paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 

class AODT_DEMO:
    def __init__(self,show_modeler=False):

        # material manager used to load predefined materials, defined in material_library.json. This will load all the materials
        # and allow us to assign them to actors by name using the get_mat_id function: mat_manager.get_index('aluminum')
        mat_manager = MaterialManager()

        self.export_debug = True

        # output file will be stored in this directory
        self.output_path = paths.output
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.debug_logs = DebuggingLogs(output_directory=self.output_path)

        
        # create dictionary of all scene actors for easier organization and visualization. The key is the name of the actor
        self.all_actors = Actors()  # all actors, using the same material library for everyone



        self.center_freq = 3.6e9
        self.num_freqs = 4096
        self.bandwidth = 122.88e6
        self.cpi_duration = 100.0e-3
        self.num_pulse_CPI = 3

        self.num_ru = 0
        self.num_ue = 0
        self.num_ant_per_ru = 2
        self.num_ant_per_ue = 2
        waveform_dict = {
            "mode": "PulsedDoppler",
            "output": "FreqPulse",
            "center_freq": self.center_freq,
            "bandwidth": self.bandwidth,
            "num_freq_samples": self.num_freqs,
            "cpi_duration": self.cpi_duration,
            "num_pulse_CPI": self.num_pulse_CPI,
            "tx_multiplex": "simultaneous",
            "mode_delay": "CENTER_CHIRP"}
        pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
        wavelength = 3e8 / self.center_freq
        self.mode_name_comms = 'mode_comms'
        self.waveform = Waveform(waveform_dict=waveform_dict)

        self.all_ru = []
        self.all_ue = []
        self.all_antenna_ru_actors = {}
        self.all_antenna_ue_actors = {}

        self.modeler=None
        self.show_modeler = show_modeler


    def add_scene_element(self,filename,target_ray_spacing=None):

        
        actor_name = self.all_actors.add_actor(name='actor',
                        filename=filename,
                        target_ray_spacing=target_ray_spacing)
        return actor_name

    def add_ru(self, pos=[0, 0, 0], rot=np.eye(3), name='ru', scale_pattern=1):
        self.num_ru += 1

        ru_root = self.all_actors.add_actor(name=name)



        ant_device_tx = add_diversity_antenna_pair(self.all_actors,
                                        self.waveform,
                                        self.mode_name_comms ,
                                        operation_mode='tx',
                                        pos=pos,rot=rot,lin=np.zeros(3),ang=np.zeros(3),
                                        parent_h_node=self.all_actors.actors[ru_root].h_node,
                                        ffd_files=['dipole.ffd','dipole_y.ffd'],
                                        scale_pattern=scale_pattern,
                                        polarization='VH',
                                        spatial_diversity_offset=np.array([0.0,0.0,0.0]),
                                        load_pattern_as_mesh=True,
                                        fov=360.0)


        # position of each antenna device

        self.all_antenna_ru_actors[ru_root] = ant_device_tx
        

    def add_ue(self, pos=[0, 0, 0], rot=np.eye(3), lin=[0, 0, 0], ang=[0, 0, 0], name='ue',scale_pattern=1):
        self.num_ue += 1
        ue_root = self.all_actors.add_actor(name=name)
        ant_device_rx = add_diversity_antenna_pair(self.all_actors,
                                self.waveform,
                                self.mode_name_comms ,
                                operation_mode='rx',
                                pos=pos,rot=rot,lin=np.zeros(3),ang=np.zeros(3),
                                parent_h_node=self.all_actors.actors[ue_root].h_node,
                                ffd_files=['dipole.ffd','dipole_y.ffd'],
                                scale_pattern=scale_pattern,
                                polarization='VH',
                                spatial_diversity_offset=np.array([0.0,0.0,0.0]),
                                load_pattern_as_mesh=True,
                                fov=360.0)
        self.all_antenna_ue_actors[ue_root] = ant_device_rx

    def _set_all_couplings(self):
        # should be called before running simulation
        for ru in self.all_antenna_ru_actors:
            for ue in self.all_antenna_ue_actors:
                enable_coupling(self.mode_name_comms,self.all_antenna_ru_actors[ru], self.all_antenna_ue_actors[ue])

    def configure_sim(self):
        '''
        this should be called right before run_simulations
        '''

        self._set_all_couplings()

        if self.show_modeler:
            output_movie_name = os.path.join(paths.output, 'out_vis_aodt_example.mp4')

            self.modeler = ModelVisualization(self.all_actors,
                                        show_antennas=True,
                                        rng_domain=None,
                                        vel_domain=None, 
                                        overlay_results=False,
                                        fps=10,
                                        camera_orientation=None,
                                        camera_attachment=None,
                                        output_movie_name=output_movie_name)
            # actors haven't been added yet, this is just a quick and dirty fix
            # ToDo, update ModelVisualization to dynamcially add actors into the scene
            # removing old actors if needed.

            self.modeler.pl.show_grid()


        # rays can be defined as density or spacing, this is the conversion between the two, if ray_density is set, then


        print(pem.listGPUs())
        sim_options = SimulationOptions()
        sim_options.ray_spacing = 5
        sim_options.max_reflections = 5
        sim_options.max_transmissions = 3
        sim_options.go_blockage = 1
        sim_options.field_of_view = 360
        sim_options.bounding_box = -1 # bounding box to truncate scene is not used, set to -1
        sim_options.auto_configure_simulation()

    def update_ue(self,pos=None,rot=None,lin=None,ang=None,name='ue'):
        ant_device_rx = self.all_antenna_ue_actors[name]
        if pos is not None:
            ant_device_rx.coord_sys.pos = pos
        if rot is not None:
            ant_device_rx.coord_sys.rot = rot
        if lin is not None:
            ant_device_rx.coord_sys.lin = lin
        if ang is not None:
            ant_device_rx.coord_sys.ang = ang
        ant_device_rx.coord_sys.update()


    def run_simulation(self):

        print('running simulation...')

        pem_api_manager.isOK(pem.computeResponseSync())

        self.debug_logs.write_scene_summary(file_name=f'out_test{0}.json')

        if self.show_modeler:

            self.modeler.update_frame(write_frame=True)

    def retrive_results(self):
        all_responses  = []
        # should be called before running simulation
        num_ru = len(self.all_antenna_ru_actors)
        num_ue = len(self.all_antenna_ue_actors)
        responses = np.zeros((num_ue,self.num_ant_per_ue,self.num_ant_per_ru,self.num_pulse_CPI,self.num_freqs),dtype=np.complex64)
        for ru in self.all_antenna_ru_actors:
            ue_idx = 0
            for ue_idx, ue in enumerate(self.all_antenna_ue_actors):

                (ret, response) = pem.retrieveP2PResponse(self.all_antenna_ru_actors[ru].modes[self.mode_name_comms],
                                                          self.all_antenna_ue_actors[ue].modes[self.mode_name_comms],
                                                          RssPy.ResponseType.FREQ_PULSE)
                responses[ue_idx] = response
                ue_idx+=1
            all_responses.append(responses)


        all_responses = np.array(all_responses)
        print(f'Results Shape: {all_responses.shape}')
        return all_responses
    








