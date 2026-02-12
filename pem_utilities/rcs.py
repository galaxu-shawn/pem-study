# -*- coding: utf-8 -*-
"""
RCS Setup Module

This module provides the RCS (Radar Cross Section) setup for radar simulations. Simplifying the setup for users
by encapsulating the configuration of radar parameters, target actors, and simulation options. Users can define
The setup in terms of polarization, and incident theta and phi ranges to extract. Incident plane waves are used
where a user can specify polarization as 'VV', 'HH', 'HV','VH', RHCPLHCP, RHCPRHCP, LHCPLHCP, LHCPRHCP. Or all or some of these polarization.




"""

from tqdm import tqdm
import numpy as np
import os
import sys
import scipy
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# Import utility modules for various radar processing functions
from pem_utilities.actor import Actors, Actor
from pem_utilities.rotation import theta_phi_to_rot, rot_to_euler
from pem_utilities.antenna_device import add_plane_wave, Waveform, add_plane_wave_bistatic_observers
from pem_utilities.debugging_utils import DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.utils import apply_math_function
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.theta_phi_sweep_generator import ThetaPhiSweepGenerator

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

class RCS:
    """

    """
    
    def __init__(self,all_actors=None,rcs_mode='monostatic',rayshoot_method='grid'):
        """
        Initialize the RCS setup with configuration parameters.
        
        Todo: add bistatic mode support

        """
        self._rcs_mode = rcs_mode.lower()
        self._all_actors = all_actors

        # Initialize all private attributes for properties
        self._center_freq = 10e9
        self._num_freqs = 3
        self._bandwidth = 300e6
        self._num_pulse_CPI = 3
        self._range_pixels = 1400
        self._r_specs = 'hann'

        self._phase_reference = [0,0,0]

        self._go_blockage = -1
        self._max_num_refl = 3
        self._max_num_trans = 0
        self._ray_spacing = 0.1
        # default value should be consistent
        lambda_center = 2.99792458e8 / 10e9  # Speed of light / frequency
        self._ray_density = np.sqrt(2) * lambda_center / self.ray_spacing
        self.rayshoot_method = rayshoot_method # 'grid' or 'sbr'

        self._polarization = 'VV'
        self._output_path = paths.output
        self._export_debug = False
        self.show_antenna_pattern = False  # Show antenna pattern in modeler

        self.incident_wave = ThetaPhiSweepGenerator()
        self.observer_wave = ThetaPhiSweepGenerator()
        self.all_theta_phi_vals = None  # This will be set later based on incident_wave parameters
        self.all_theta_phi_vals_observer = None  # This will be set later based on observer_wave parameters

        self.num_waves_to_solve_in_parallel = 1
        self.trim_final_results_size = 0 # this is for dealing with batch sizes that are not a multiple of the number of theta/phi samples

        self.vel_domain = None    # Velocity domain
        self.rng_domain = None     # Range domain
        self.freq_domain = None   # Frequency domain
        self.pulse_domain = None  # Pulse domain

        self.modeler = None  # Model visualization object
        self.results_df = pd.DataFrame(columns=['theta', 'phi', 'frequency_response'])

        self.total_time = 0.0  # Total time for the simulation

        self.solver_time_total = 0.0  # Total simulation time
        self.solver_per_excitation = 0.0  # Time per observation
        self.solver_time_per_batch = 0.0  # Time per batch of observations

        self.retreive_time_total = 0.0  # Total time for retrieving results
        self.retreive_per_excitation = 0.0  # Time per observation for retrieving results
        self.retreive_time_per_batch = 0.0  # Time per batch of observations for retrieving results
        self.total_excitations = 0  # Total number of excitations processed
        self.num_batches = 0
    @property
    def rcs_mode(self):
        """Get the RCS mode."""
        return self._rcs_mode
    @rcs_mode.setter
    def rcs_mode(self, value):
        """Set the RCS mode with validation."""
        # valid_modes = ['monostatic', 'bistatic']
        valid_modes = ['monostatic']

        if not isinstance(value, str):
            raise TypeError("RCS mode must be a string")
        if value.lower() not in valid_modes:
            raise ValueError(f"RCS mode must be one of {valid_modes}")
        self._rcs_mode = value.lower()
        print(f"RCS mode set to {self._rcs_mode}")

    # all scene actors
    @property
    def all_actors(self):
        """Get the collection of all actors in the scene."""
        return self._all_actors
    @all_actors.setter
    def all_actors(self, value):
        """Set the collection of all actors in the scene."""
        if not isinstance(value, Actors):
            raise TypeError("all_actors must be an instance of Actors")

        for each in value.actors:
            if not isinstance(value.actors[each], Actor):
                raise TypeError("all_actors must be an instance of Actor")
        self._all_actors = value
        print(f"All actors set with {len(self._all_actors.actors)} actors") 

    # Radar waveform parameters
    @property
    def center_freq(self):
        """Get the center frequency in Hz."""
        return self._center_freq
    
    @center_freq.setter
    def center_freq(self, value):
        """Set the center frequency in Hz with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Center frequency must be a number")
        if value <= 0:
            raise ValueError("Center frequency must be positive")
        self._center_freq = float(value)
        print(f"Center frequency set to {self._center_freq/1e9:.2f} GHz")

    @property
    def num_freqs(self):
        """Get the number of frequency samples."""
        return self._num_freqs
    
    @num_freqs.setter
    def num_freqs(self, value):
        """Set the number of frequency samples with validation."""
        if not isinstance(value, int):
            raise TypeError("Number of frequencies must be an integer")
        if value <= 0:
            raise ValueError("Number of frequencies must be positive")
        self._num_freqs = value

    @property
    def bandwidth(self):
        """Get the bandwidth in Hz."""
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self, value):
        """Set the bandwidth in Hz with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Bandwidth must be a number")
        if value <= 0:
            raise ValueError("Bandwidth must be positive")
        self._bandwidth = float(value)

    @property
    def num_pulse_CPI(self):
        """Get the number of pulses in Coherent Processing Interval."""
        return self._num_pulse_CPI
    
    @num_pulse_CPI.setter
    def num_pulse_CPI(self, value):
        """Set the number of pulses in CPI with validation."""
        if not isinstance(value, int):
            raise TypeError("Number of pulses must be an integer")
        if value <= 0:
            raise ValueError("Number of pulses must be positive")
        self._num_pulse_CPI = value
        # Update dependent properties
        self._doppler_pixels = value

    @property
    def range_pixels(self):
        """Get the range dimension pixels."""
        return self._range_pixels
    
    @range_pixels.setter
    def range_pixels(self, value):
        """Set the range dimension pixels with validation."""
        if not isinstance(value, int):
            raise TypeError("Range pixels must be an integer")
        if value <= 0:
            raise ValueError("Range pixels must be positive")
        self._range_pixels = value


    @property
    def r_specs(self):
        """Get the range window specification."""
        return self._r_specs
    
    @r_specs.setter
    def r_specs(self, value):
        """Set the range window specification with validation."""
        valid_specs = ['hann', 'hamming', 'blackman', 'bartlett', 'kaiser', 'rectangular']
        if not isinstance(value, str):
            raise TypeError("Range window specification must be a string")
        if value not in valid_specs:
            raise ValueError(f"Range window must be one of {valid_specs}")
        self._r_specs = value

    # Scene geometry
    @property
    def phase_reference(self):
        """Get the target center coordinates."""
        return self._phase_reference
    
    @phase_reference.setter
    def phase_reference(self, value):
        """Set the phase center coordinates with validation."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError("Phase center must be a list, tuple, or numpy array")
        if len(value) != 3:
            raise ValueError("Phase center must have exactly 3 coordinates [x, y, z]")
        if not all(isinstance(coord, (int, float)) for coord in value):
            raise TypeError("Phaset center coordinates must be numbers")
        self._phase_reference = list(value)

    # Ray tracing simulation options
    @property
    def go_blockage(self):
        """Get the GO blockage setting."""
        return self._go_blockage
    
    @go_blockage.setter
    def go_blockage(self, value):
        """Set the GO blockage setting with validation."""
        if not isinstance(value, int):
            raise TypeError("GO blockage must be an integer")
        if value < -1:
            raise ValueError("GO blockage must be -1 (disabled) or >= 0 (enabled)")
        self._go_blockage = value

    @property
    def max_num_refl(self):
        """Get the maximum number of reflections."""
        return self._max_num_refl
    
    @max_num_refl.setter
    def max_num_refl(self, value):
        """Set the maximum number of reflections with validation."""
        if not isinstance(value, int):
            raise TypeError("Maximum reflections must be an integer")
        if value < 0:
            raise ValueError("Maximum reflections must be non-negative")
        self._max_num_refl = value

    @property
    def max_num_trans(self):
        """Get the maximum number of transmissions."""
        return self._max_num_trans
    
    @max_num_trans.setter
    def max_num_trans(self, value):
        """Set the maximum number of transmissions with validation."""
        if not isinstance(value, int):
            raise TypeError("Maximum transmissions must be an integer")
        if value < 0:
            raise ValueError("Maximum transmissions must be non-negative")
        self._max_num_trans = value

    @property
    def ray_spacing(self):
        """Get the ray density for simulation."""
        return self._ray_spacing
    
    @ray_spacing.setter
    def ray_spacing(self, value):
        """Set the ray density for simulation with validation."""
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError("Ray density must be a number or None")
            if value <= 0:
                raise ValueError("Ray density must be positive")
            self._ray_spacing = float(value)
        else:
            self._ray_spacing = None

    @property
    def ray_density(self):
        """Get the ray density based on the ray spacing."""
        if self._ray_density is not None:
            return self._ray_density
        else:
            return None
        # Calculate ray spacing based on ray density
    @ray_density.setter
    def ray_density(self, value):
        """Set the ray density based on the ray spacing with validation."""
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError("Ray density must be a number or None")
            if value <= 0:
                raise ValueError("Ray density must be positive")
            self._ray_density = float(value)
        else:
            self._ray_density = None
        # Calculate ray spacing based on ray density
        if self._ray_density is not None:
            lambda_center = 2.99792458e8 / self.center_freq
            self._ray_spacing = np.sqrt(2) * lambda_center / self._ray_density


    @property
    def polarization(self):
        """Get the antenna polarization."""
        return self._polarization
    
    @polarization.setter
    def polarization(self, value):
        """Set the antenna polarization with validation."""
        valid_polarizations = ['VV', 'HH', 'HV', 'VH', 'RHCPLHCP', 'RHCPRHCP', 'LHCPLHCP', 'LHCPRHCP']
        if not isinstance(value, str):
            raise TypeError("Polarization must be a string")
        if value not in valid_polarizations:
            raise ValueError(f"Polarization must be one of {valid_polarizations}")
        self._polarization = value

    # Output configuration
    @property
    def output_path(self):
        """Get the output directory path."""
        return self._output_path
    
    @output_path.setter
    def output_path(self, value):
        """Set the output directory path with validation."""
        if not isinstance(value, str):
            raise TypeError("Output path must be a string")
        os.makedirs(value, exist_ok=True)  # Create temp directory if it doesn't exist
        self._output_path = value

    # Debug options
    @property
    def export_debug(self):
        """Get the debug export flag."""
        return self._export_debug
    
    @export_debug.setter
    def export_debug(self, value):
        """Set the debug export flag with validation."""
        if not isinstance(value, bool):
            raise TypeError("Export debug must be a boolean")
        self._export_debug = value

    def intialize_solver(self):
        """

        """

        self.actor_radar_name = self.all_actors.add_actor()
        output = 'FreqPulse'
        
        # Waveform dictionary with all radar parameters
        waveform_dict = {
            "mode": "PulsedDoppler",           # Pulsed Doppler radar mode
            "output": output,                   # Output format
            "center_freq": self.center_freq,   # Center frequency
            "bandwidth": self.bandwidth,       # Signal bandwidth
            "num_freq_samples": self.num_freqs,         # Number of frequency samples
            "cpi_duration": 1,          # CPI duration
            "num_pulse_CPI": self.num_pulse_CPI,        # Pulses per CPI
            "mode_delay": "CENTER_CHIRP"                # Timing reference
        }
        
        # Calculate derived parameters
        pulse_interval = waveform_dict['cpi_duration'] / waveform_dict['num_pulse_CPI']
        wavelength = 3e8 / self.center_freq
        self.mode_name = 'mode1'
        waveform = Waveform(waveform_dict)
        
        self.all_ant_devices = []
        if self.rcs_mode == 'monostatic':
            # Add single transmitter/receiver radar to the scene
            for each in range(self.num_waves_to_solve_in_parallel):
                self.all_ant_devices.append(add_plane_wave(
                    self.all_actors,
                    waveform,
                    self.mode_name,
                    parent_h_node=self.all_actors.actors[self.actor_radar_name].h_node,
                    polarization=self.polarization,
                    range_pixels=self.range_pixels,
                    doppler_pixels=128,
                    r_specs=self.r_specs,
                    d_specs='hann',  # Doppler window specification
                    scale_pattern=10,
                    load_pattern_as_mesh=self.show_antenna_pattern
                )   
                )
        elif self.rcs_mode == 'bistatic':
            # Add bistatic radar with separate transmitter and receiver
            all_observer = []
            all_incident = []
            if self.all_theta_phi_vals_observer is None:
                self.all_theta_phi_vals_observer = self.observer_wave.get_all_theta_phi_vals()
            for each in self.observer_wave.phi_domain:
                # Convert theta and phi to rotation matrix
                # rot = theta_phi_to_rot(each[0], each[1])
                rot=np.eye(3) # rotate them later
                all_observer.append({'pos':np.zeros(3),'rot':rot,'lin':np.zeros(3),'ang':np.zeros(3)})

            if self.all_theta_phi_vals is None:
                self.all_theta_phi_vals = self.incident_wave.get_all_theta_phi_vals()

            # incident orientations
            for each in self.all_theta_phi_vals:
                # Convert theta and phi to rotation matrix
                rot = theta_phi_to_rot(each[0], each[1])
                all_incident.append({'pos':np.zeros(3),'rot':rot,'lin':np.zeros(3),'ang':np.zeros(3)})

            
            # Add transmitter
            ant_device = add_plane_wave_bistatic_observers(
                self.all_actors,
                waveform,
                self.mode_name,
                parent_h_node=self.all_actors.actors[self.actor_radar_name].h_node,
                observer_orientations=all_observer,
                incident_orientations=all_incident,
                polarization=self.polarization,
                range_pixels=self.range_pixels,
                doppler_pixels=3,
                r_specs=self.r_specs,
                d_specs='hann',  # Doppler window specification
                scale_pattern=10,
                load_pattern_as_mesh=self.show_antenna_pattern
            )
            self.all_ant_devices.append(ant_device)
            

        self.ray_density = 0.5
        # Configure simulation options
        sim_options = SimulationOptions(center_freq=self.center_freq)
        sim_options.ray_spacing = self.ray_spacing
        sim_options.max_reflections = self.max_num_refl
        sim_options.max_transmissions = self.max_num_trans
        sim_options.go_blockage = self.go_blockage
        sim_options.field_of_view = 180
        sim_options.bounding_box = -1
        sim_options.ray_shoot_method = self.rayshoot_method  # 'grid' or 'sbr'
        # Note: GPU device selection can be done with sim_options.gpu_device = 0
        sim_options.auto_configure_simulation()
        
        # Get response domains for post-processing and axis scaling
        which_mode = self.all_ant_devices[0].modes[self.mode_name]
        self.all_ant_devices[0].waveforms[self.mode_name].get_response_domains(which_mode)
        
        # Extract domains for each dimension
        self.vel_domain = self.all_ant_devices[0].waveforms[self.mode_name].vel_domain     # Velocity domain
        self.rng_domain = self.all_ant_devices[0].waveforms[self.mode_name].rng_domain     # Range domain
        self.freq_domain = self.all_ant_devices[0].waveforms[self.mode_name].freq_domain   # Frequency domain
        self.pulse_domain = self.all_ant_devices[0].waveforms[self.mode_name].pulse_domain # Pulse domain

        # For planewave sources
        dist_to_center = 0
        radial_velocity = 0
        
        # Set pixel reference point
        ref_pixel = RssPy.ImagePixelReference.MIDDLE
        
        # Display maximum range
        print(f'Max Range: {self.rng_domain[-1]} m')
        
        
        # Initialize debug systems if enabled
        if self.export_debug:
            self.debug_logs = DebuggingLogs(output_directory=self.output_path)
            self.debug_logs.write_scene_summary(file_name='out.json')

    def _get_batch_of_theta_phi_vals(self, batch_size=10):
        """
        Generate batches of theta and phi values for parallel processing.
        
        This method creates batches of theta and phi values to be processed in parallel,
        allowing efficient simulation of multiple radar waveforms.
        
        Args:
            batch_size (int): Number of theta-phi pairs per batch
        
        Returns:
            list: List of batches, each containing theta-phi pairs
        """
        if self.all_theta_phi_vals is None:
            self.all_theta_phi_vals = self.incident_wave.get_all_theta_phi_vals()
        
        num_batches = int(np.ceil(len(self.all_theta_phi_vals) / batch_size))
        list_of_values = [self.all_theta_phi_vals[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        # Ensure the last batch is not empty
        if len(list_of_values[-1]) == 0:
            list_of_values.pop()
        # check if last bathch is smaller than batch_size, pad with zeros
        if len(list_of_values[-1]) < batch_size:
            self.trim_final_results_size = len(list_of_values[-1])
            list_of_values[-1] = np.pad(list_of_values[-1], ((0, batch_size - len(list_of_values[-1])), (0, 0)), mode='constant', constant_values=0)
            
        return list_of_values
    
    def _run_bistatic(self, show_modeler=False):
        pos = self.phase_reference
        response_type = RssPy.ResponseType.FREQ_PULSE
        self.num_batches = len(self.observer_wave.theta_domain)

        all_results_numpy=[]
        theta_vals_observer = self.observer_wave.theta_domain
        phi_vals_observer = self.observer_wave.phi_domain
        all_results = []
        # phi values will be the same for not change, we are running 1 simulation for each theta value
        for th in tqdm(theta_vals_observer):
            ant_device = self.all_ant_devices[0]  # Assuming the first and only device is the bistatic one
            rx_ant_idx = 0
            for each in ant_device.all_antennas_properties:
                # only rx antenna
                if ant_device.all_antennas_properties[each]['operation_mode'].lower() == 'rx':
                    # each of these should correspond to a different phi antenna location
                    ant_device.all_antennas_properties[each]['rotation'] = theta_phi_to_rot(th, phi_vals_observer[rx_ant_idx])
                    self.all_actors.actors[each].coord_sys.rot = theta_phi_to_rot(th, phi_vals_observer[rx_ant_idx])
                    self.all_actors.actors[each].coord_sys.update()
                    rx_ant_idx+=1

            # Update plane wave position and orientation    
            # Time the solver computation
            solver_start_time = time.perf_counter()
            pem_api_manager.isOK(pem.computeResponseSync())
            solver_end_time = time.perf_counter()
            solver_time = solver_end_time - solver_start_time
            self.solver_time_total += solver_time


            # Retrieve simulation response
            retrieve_start_time = time.perf_counter()
            (ret, response) = pem.retrieveResponse(ant_device.modes[self.mode_name], response_type)
            retrieve_end_time = time.perf_counter()
            retrieve_time = retrieve_end_time - retrieve_start_time
            self.retreive_time_total += retrieve_time

            if response is None:
                raise ValueError("Response is None, check simulation parameters and actors")
            if isinstance(response, np.ndarray):
                # if response is a 3D array, we need to reshape it to 2D
                if response.ndim == 3:
                    response = response[:,1] # reformt so 1D response in frequency dimension
                                        # scale power to get RCS
                else:
                    response = response[0,:,1]
                all_results_numpy.append(response)
                Pr = np.sqrt(4*np.pi)*np.abs(response)
                # save results to pandas dataframe with theta_phi values and frequency response (self.freq_domain
                for ph_idx, ph in enumerate(self.observer_wave.phi_domain):
                    all_results.append({
                        'theta': th,
                        'phi': ph,
                        'frequency_response': Pr[ph_idx]
                    })

        all_results_numpy = np.array(all_results_numpy)
        # save numpy array to file
        np.save(os.path.join(paths.output, 'bistatic_rcs_results.npy'), all_results_numpy) 
        np.save(os.path.join(paths.output, 'phi_domain.npy'), self.observer_wave.phi_domain)
        np.save(os.path.join(paths.output, 'theta_domain.npy'), self.observer_wave.theta_domain)
        np.save(os.path.join(paths.output, 'frequency_domain.npy'), self.freq_domain)
        return all_results


    def _run_monostatic(self,show_modeler=False):

        pos = self.phase_reference
        response_type = RssPy.ResponseType.FREQ_PULSE

        vals = self._get_batch_of_theta_phi_vals(len(self.all_ant_devices))
        self.num_batches = len(vals)
        # Initialize batch timing
        batch_solver_times = []
        batch_retrieve_times = []
        all_results = []  # List to store results for each theta-phi pair
        for val_idx, theta_phi in enumerate(tqdm(vals)):
            batch_start_time = time.perf_counter()
            
            for idx, ant_device in enumerate(self.all_ant_devices):
                # Update plane wave position and orientation
                ant_device.coord_sys.pos = pos
                # For monostatic, use the same theta and phi for both transmitter and receiver
                ant_device.coord_sys.rot = theta_phi_to_rot(theta_phi[idx,0], theta_phi[idx,1])
                ant_device.coord_sys.update()

            # Time the solver computation
            solver_start_time = time.perf_counter()
            pem_api_manager.isOK(pem.computeResponseSync())
            solver_end_time = time.perf_counter()
            solver_time = solver_end_time - solver_start_time
            self.solver_time_total += solver_time
            batch_solver_times.append(solver_time)
   
            if show_modeler:
                self.modeler.update_frame(write_frame=True)
            # Generate debug output if enabled
            if self.export_debug:
                self.debug_logs.write_scene_summary(file_name='out.json')

            # Time the response retrieval
            
            retrieve_time_per_batch = 0
            for idx, ant_device in enumerate(self.all_ant_devices):
                # skip part of the last batch if it is smaller than batch size, this was just padded with zero to make equal batch sizes
                if val_idx == len(vals)-1 and idx >=self.trim_final_results_size and self.trim_final_results_size != 0:
                    continue
                # Retrieve simulation response
                retrieve_start_time = time.perf_counter()
                (ret, response) = pem.retrieveResponse(ant_device.modes[self.mode_name], response_type)
                retrieve_end_time = time.perf_counter()
                retrieve_time = retrieve_end_time - retrieve_start_time
                retrieve_time_per_batch += retrieve_time 
                self.retreive_time_total += retrieve_time

                if response is None:
                    raise ValueError("Response is None, check simulation parameters and actors")
                if isinstance(response, np.ndarray):

                    response = response[0,0,1] # reformt so 1D response in frequency dimension
                                        # scale power to get RCS
                    Pr = np.sqrt(4*np.pi)*np.abs(response)
                    # save results to pandas dataframe with theta_phi values and frequency response (self.freq_domain
                    all_results.append({
                        'theta': theta_phi[idx, 0],
                        'phi': theta_phi[idx, 1],
                        'frequency_response': Pr
                    })

                self.total_excitations += 1
            batch_retrieve_times.append(retrieve_time)
        return all_results

    def run_simulation(self,show_modeler=False):
        """
        Execute the radar simulation and generate SAR image.
        
        This method performs the complete simulation workflow:
        - Updates radar and target positions
        - Computes radar response using ray tracing
        - Applies selected SAR imaging algorithm
        - Returns processed SAR image
        
        Args:
            function (str): Mathematical function to apply to image data.
                          Options: 'dB', 'abs', 'real', 'imag', or other supported functions
        
        Returns:
            numpy.ndarray: Processed SAR image as 2D array
        """

        # Reset timing statistics
        self.solver_time_total = 0.0
        self.retreive_time_total = 0.0
        self.total_time = 0.0
        
        # initialize an empty pandas DataFrame to store results
  
        self.intialize_solver()

        if show_modeler:
            self.modeler = ModelVisualization(self.all_actors,overlay_results=False,show_antennas=self.show_antenna_pattern)



        # bistatic the incident and observer wave are different and have already been set

        if self.rcs_mode == 'bistatic':
            all_results = self._run_bistatic(show_modeler=show_modeler)

        elif self.rcs_mode == 'monostatic':
            all_results = self._run_monostatic(show_modeler=show_modeler)


        # Calculate timing statistics
        self.total_time = self.solver_time_total+self.retreive_time_total
        
        # Calculate per-excitation times
        if self.total_excitations > 0:
            self.solver_per_excitation = self.solver_time_total / self.total_excitations
            self.retreive_per_excitation = self.retreive_time_total / self.total_excitations
        


        # After all loops are complete, create the DataFrame:
        self.results_df = pd.DataFrame(all_results)
        
    def print_timing_summary(self):
        # Print timing summary
        print(f"\nSimulation Timing Summary:")
        print(f"Total simulation time: {self.total_time:.3f} seconds")
        print(f"Total solver time: {self.solver_time_total:.3f} seconds")
        print(f"Total retrieve time: {self.retreive_time_total:.3f} seconds")
        print(f"Solver time per excitation: {self.solver_per_excitation:.6f} seconds")
        print(f"Retrieve time per excitation: {self.retreive_per_excitation:.6f} seconds")
        print(f"Average solver time per batch: {self.solver_time_per_batch:.3f} seconds")
        print(f"Average retrieve time per batch: {self.retreive_time_per_batch:.3f} seconds")
        print(f"Total excitations processed: {self.total_excitations}")
        print(f"Number of batches: {self.num_batches}")


    def show_modeler(self,show_antennas=False):
        self.modeler = ModelVisualization(self.all_actors,overlay_results=False,show_antennas=show_antennas)
        self.modeler.update_frame(write_frame=False)

    def _plot_points(self, points_xyz):
        """
        Plot 3D visualization of sampling points on unit sphere.
        
        This private method creates a 3D scatter plot showing the distribution
        of observation points on a unit sphere with a wireframe reference.
        
        Args:
            points_xyz (numpy.ndarray): XYZ coordinates of points to plot (N×3 array)
        
        Note:
            This method is used internally by the sampling methods when
            show_plot=True is specified.
        """
        # Convert to spherical coordinates for reference (currently unused)
        r = np.ones(points_xyz.shape[0])
        theta = np.arccos(points_xyz[:, 1])  # Polar angle
        phi = np.arctan2(points_xyz[:, 2], points_xyz[:, 0])  # Azimuthal angle
        
        # Create 3D plot
        fig = plt.figure(figsize=(6, 6))
        ax2 = fig.add_subplot(111, projection='3d')
        
        # Plot sampling points
        ax2.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], 
                   c='blue', s=15, label='Sampling Points')
        
        # Add wireframe reference sphere for context
        u_sphere = np.linspace(0, 2 * np.pi, 20)
        v_sphere = np.linspace(0, np.pi, 10)
        u_sphere, v_sphere = np.meshgrid(u_sphere, v_sphere)
        x_sphere = np.sin(v_sphere) * np.cos(u_sphere)
        y_sphere = np.sin(v_sphere) * np.sin(u_sphere)
        z_sphere = np.cos(v_sphere)
        ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
        
        # Configure plot appearance
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Uniform Sampling Observation Points')
        
        # Set equal aspect ratio for proper sphere visualization
        ax2.set_box_aspect([
            (np.max(x_sphere) - np.min(x_sphere)) / 2,
            (np.max(y_sphere) - np.min(y_sphere)) / 2,
            (np.max(z_sphere) - np.min(z_sphere)) / 2
        ])
        
        plt.tight_layout()
        plt.show()


    def plot_rectangular(self, phi=None, theta=None, freq_idx=0, function='dB', 
                        title=None, xlabel=None, ylabel=None, 
                        save_path=None, figsize=(10, 6)):
        """
        Create a rectangular plot of RCS data for a specific theta or phi slice.
        
        This method plots RCS data along one angular dimension while holding the other constant.
        For example, if theta=0 is specified, it plots RCS vs phi for all phi values at theta=0.
        
        Args:
            phi (float, optional): Fixed phi value in degrees. If specified, plot RCS vs theta.
            theta (float, optional): Fixed theta value in degrees. If specified, plot RCS vs phi.
            freq_idx (int): Frequency index to plot (default: 0)
            function (str): Mathematical function to apply ('dB', 'linear', 'abs', etc.)
            title (str, optional): Plot title. If None, auto-generated.
            xlabel (str, optional): X-axis label. If None, auto-generated.
            ylabel (str, optional): Y-axis label. If None, auto-generated.
            show_plot (bool): Whether to display the plot
            save_path (str, optional): Path to save the plot
            figsize (tuple): Figure size (width, height) in inches
            
        Returns:
            tuple: (fig, ax, x_values, y_values) where x_values are angles and y_values are RCS values
        """
        plt.ion()
        # Validate inputs
        if phi is None and theta is None:
            raise ValueError("At least one of phi or theta must be specified")
        if phi is not None and theta is not None:
            raise ValueError("Only one of phi or theta can be specified at a time")
        
        if self.results_df is None or self.results_df.empty:
            raise ValueError("No simulation results available. Run simulation first.")
            
        # Validate frequency index
        if freq_idx < 0 or freq_idx >= len(self.freq_domain):
            raise ValueError(f"Frequency index must be between 0 and {len(self.freq_domain)-1}")
        
        # Filter data based on specified angle
        if phi is not None:
            if not isinstance(phi, (int, float)):
                raise TypeError("phi must be a number")
            if phi < self.incident_wave.phi_start or phi > self.incident_wave.phi_stop:
                raise ValueError(f"phi must be between {self.incident_wave.phi_start} and {self.incident_wave.phi_stop}")
            
            # Filter for constant phi, varying theta
            df = self.results_df[self.results_df['phi'] == phi].copy()
            varying_angle = 'theta'
            fixed_angle = 'phi'
            fixed_value = phi
            varying_label = 'Theta'
            varying_unit = '°'
            
        else:  # theta is not None
            if not isinstance(theta, (int, float)):
                raise TypeError("theta must be a number")
            if theta < self.incident_wave.theta_start or theta > self.incident_wave.theta_stop:
                raise ValueError(f"theta must be between {self.incident_wave.theta_start} and {self.incident_wave.theta_stop}")
            
            # Filter for constant theta, varying phi
            df = self.results_df[self.results_df['theta'] == theta].copy()
            varying_angle = 'phi'
            fixed_angle = 'theta'
            fixed_value = theta
            varying_label = 'Phi'
            varying_unit = '°'
        
        if df.empty:
            raise ValueError(f"No data found for the specified {fixed_angle} value")
        
        # Sort by the varying angle for proper plotting
        df = df.sort_values(by=varying_angle)
        
        # Extract frequency response data at the specified frequency index
        x_values = df[varying_angle].values
        y_data = []
        
        for _, row in df.iterrows():
            freq_response = row['frequency_response']
            if isinstance(freq_response, np.ndarray) and len(freq_response) > freq_idx:
                y_data.append(freq_response[freq_idx])
            else:
                raise ValueError(f"Frequency response data is not available or insufficient for index {freq_idx}")
        
        y_data = np.array(y_data)
        
        # Apply mathematical function
        try:
            y_values = apply_math_function(y_data, function=function)
            y_label_default = f'RCS ({function})'
        except:
            raise ValueError(f"Unsupported function: {function}")
    
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_values, y_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax.grid(True, alpha=0.3)
        
        # Set labels and title
        if xlabel is None:
            xlabel = f'{varying_label} ({varying_unit})'
        if ylabel is None:
            ylabel = y_label_default
        if title is None:
            freq_ghz = self.freq_domain[freq_idx] / 1e9
            title = f'RCS vs {varying_label} at {fixed_angle.capitalize()}={fixed_value}° (f={freq_ghz:.2f} GHz)'
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Set reasonable axis limits
        ax.set_xlim(x_values.min(), x_values.max())
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig, ax, x_values, y_values

    def plot_polar(self, phi=None, theta=None, freq_idx=0, function='dB', 
                   title=None, show_plot=True, save_path=None, figsize=(8, 8),
                   r_min=None, r_max=None):
        """
        Create a polar plot of RCS data for a specific theta or phi slice.
        
        This method creates polar plots which are particularly useful for RCS visualization
        as they naturally represent the angular nature of the data.
        
        Args:
            phi (float, optional): Fixed phi value in degrees. If specified, plot RCS vs theta in polar coordinates.
            theta (float, optional): Fixed theta value in degrees. If specified, plot RCS vs phi in polar coordinates.
            freq_idx (int): Frequency index to plot (default: 0)
            function (str): Mathematical function to apply ('dB', 'linear', 'abs', etc.)
            title (str, optional): Plot title. If None, auto-generated.
            show_plot (bool): Whether to display the plot
            save_path (str, optional): Path to save the plot
            figsize (tuple): Figure size (width, height) in inches
            r_min (float, optional): Minimum radius for plot scaling
            r_max (float, optional): Maximum radius for plot scaling
            
        Returns:
            tuple: (fig, ax, angles_rad, r_values) where angles_rad are angles in radians and r_values are RCS values
        """
        # Get rectangular plot data first
        fig_rect, ax_rect, x_values, y_values = self.plot_rectangular(
            phi=phi, theta=theta, freq_idx=freq_idx, function=function, 
            show_plot=False
        )
        plt.close(fig_rect)  # Close the rectangular plot
        
        # Convert angles to radians for polar plot
        angles_rad = np.deg2rad(x_values)
        
        # Handle negative values for polar plotting
        if function.lower() == 'db':
            # For dB plots, we need to handle the radial scaling carefully
            r_values = y_values
            if r_min is None:
                r_min = np.min(r_values) - 5  # Add some margin
            if r_max is None:
                r_max = np.max(r_values) + 5
        else:
            # For linear plots, use absolute values
            r_values = np.abs(y_values)
            if r_min is None:
                r_min = 0
            if r_max is None:
                r_max = np.max(r_values) * 1.1
        
        # Create polar plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles_rad, r_values, 'b-', linewidth=2, marker='o', markersize=3)
        ax.fill(angles_rad, r_values, alpha=0.3)
        
        # Set radial limits
        ax.set_ylim(r_min, r_max)
        
        # Configure angular axis
        if phi is not None:
            # Plotting theta variation at constant phi
            ax.set_theta_zero_location('N')  # 0 degrees at top (North)
            ax.set_theta_direction(1)  # Clockwise
            varying_label = 'Theta'
        else:
            # Plotting phi variation at constant theta  
            ax.set_theta_zero_location('E')  # 0 degrees at right (East)
            ax.set_theta_direction(1)  # Counter-clockwise
            varying_label = 'Phi'
        
        # Set title
        if title is None:
            if phi is not None:
                freq_ghz = self.freq_domain[freq_idx] / 1e9
                title = f'RCS Polar Plot vs {varying_label} at Phi={phi}° (f={freq_ghz:.2f} GHz)'
            else:
                freq_ghz = self.freq_domain[freq_idx] / 1e9
                title = f'RCS Polar Plot vs {varying_label} at Theta={theta}° (f={freq_ghz:.2f} GHz)'
        
        ax.set_title(title, pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Polar plot saved to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig, ax, angles_rad, r_values

    def get_rcs_data(self, theta_range=None, phi_range=None, freq_idx=None, 
                     sort_by='theta', function='linear', return_format='dataframe'):
        """
        Parse and extract RCS data from simulation results with flexible filtering options.
        
        This function provides an intuitive interface to extract specific slices of RCS data
        based on theta, phi, and frequency criteria. It supports various input formats and
        return options to make data analysis convenient.
        
        Args:
            theta_range (float, list, tuple, or None): 
                - float: Extract data for specific theta value
                - list: Extract data for multiple theta values [theta1, theta2, ...]
                - tuple: Extract data for theta range (start, stop) or (start, stop, step)
                - None: Include all theta values
            
            phi_range (float, list, tuple, or None):
                - float: Extract data for specific phi value  
                - list: Extract data for multiple phi values [phi1, phi2, ...]
                - tuple: Extract data for phi range (start, stop) or (start, stop, step)
                - None: Include all phi values
            
            freq_idx (int, list, slice, or None):
                - int: Extract data for specific frequency index
                - list: Extract data for multiple frequency indices [idx1, idx2, ...]
                - slice: Extract data for frequency slice (e.g., slice(0, 10, 2))
                - None: Include all frequencies
            
            sort_by (str): How to sort the returned data
                - 'theta': Sort by theta values (ascending)
                - 'phi': Sort by phi values (ascending) 
                - 'theta_phi': Sort by theta first, then phi
                - 'phi_theta': Sort by phi first, then theta
                - 'none': No sorting (preserve original order)
            
            function (str): Mathematical function to apply to frequency response data
                - 'linear': Return linear magnitude values
                - 'dB': Convert to decibels (20*log10)
                - 'abs': Absolute magnitude
                - 'real': Real part only
                - 'imag': Imaginary part only
                - 'phase': Phase in degrees
                
            return_format (str): Format of returned data
                - 'dataframe': Return as pandas DataFrame
                - 'dict': Return as dictionary with keys for each data type
                - 'arrays': Return as tuple of (theta, phi, rcs_data) arrays
        
        Returns:
            pandas.DataFrame, dict, or tuple: Filtered and sorted RCS data in requested format
        
        Examples:
            # Get data for specific theta and phi
            data = rcs.get_rcs_data(theta_range=45, phi_range=90, freq_idx=0)
            
            # Get data for theta range and multiple phi values  
            data = rcs.get_rcs_data(theta_range=(0, 90, 10), phi_range=[0, 45, 90, 135])
            
            # Get all data in dB scale, sorted by phi then theta
            data = rcs.get_rcs_data(function='dB', sort_by='phi_theta')
            
            # Get data as arrays for custom processing
            theta, phi, rcs = rcs.get_rcs_data(return_format='arrays')
        """
        
        # Validate that simulation results exist
        if self.results_df is None or self.results_df.empty:
            raise ValueError("No simulation results available. Run simulation first.")
        
        # Start with full dataset
        df = self.results_df.copy()
        
        # Filter by theta range
        if theta_range is not None:
            theta_values = self._parse_angle_range(theta_range, 'theta', 
                                                 self.incident_wave.theta_start, self.incident_wave.theta_stop, self.incident_wave.theta_step_deg)
            df = df[df['theta'].isin(theta_values)]
        
        # Filter by phi range  
        if phi_range is not None:
            phi_values = self._parse_angle_range(phi_range, 'phi',
                                               self.incident_wave.phi_start, self.incident_wave.phi_stop, self.incident_wave.phi_step_deg)
            df = df[df['phi'].isin(phi_values)]
        
        if df.empty:
            raise ValueError("No data found matching the specified theta and phi criteria")
        
        # Process frequency response data and apply function
        processed_data = []
        for _, row in df.iterrows():
            freq_response = row['frequency_response']
            
            # Apply frequency filtering if specified
            if freq_idx is not None:
                if isinstance(freq_idx, int):
                    if freq_idx < 0 or freq_idx >= len(freq_response):
                        raise ValueError(f"Frequency index {freq_idx} out of range [0, {len(freq_response)-1}]")
                    filtered_response = freq_response[freq_idx:freq_idx+1]  # Keep as array
                elif isinstance(freq_idx, list):
                    if any(idx < 0 or idx >= len(freq_response) for idx in freq_idx):
                        raise ValueError(f"One or more frequency indices out of range [0, {len(freq_response)-1}]")
                    filtered_response = freq_response[freq_idx]
                elif isinstance(freq_idx, slice):
                    filtered_response = freq_response[freq_idx]
                else:
                    raise TypeError("freq_idx must be int, list, slice, or None")
            else:
                filtered_response = freq_response
            

            try:
                processed_response = apply_math_function(filtered_response, function=function)
            except:
                raise ValueError(f"Unsupported function: {function}")
            
            processed_data.append({
                'theta': row['theta'],
                'phi': row['phi'],
                'rcs_data': processed_response
            })
        
        # Convert to DataFrame for easier manipulation
        result_df = pd.DataFrame(processed_data)
        
        # Apply sorting
        if sort_by.lower() == 'theta':
            result_df = result_df.sort_values('theta', ignore_index=True)
        elif sort_by.lower() == 'phi':
            result_df = result_df.sort_values('phi', ignore_index=True)
        elif sort_by.lower() == 'theta_phi':
            result_df = result_df.sort_values(['theta', 'phi'], ignore_index=True)
        elif sort_by.lower() == 'phi_theta':
            result_df = result_df.sort_values(['phi', 'theta'], ignore_index=True)
        elif sort_by.lower() != 'none':
            raise ValueError(f"Invalid sort_by option: {sort_by}. Must be 'theta', 'phi', 'theta_phi', 'phi_theta', or 'none'")
        
        # Return data in requested format
        if return_format.lower() == 'dataframe':
            return result_df
        elif return_format.lower() == 'dict':
            return {
                'theta': result_df['theta'].values,
                'phi': result_df['phi'].values, 
                'rcs_data': result_df['rcs_data'].values,
                'frequencies': self.freq_domain if freq_idx is None else self.freq_domain[freq_idx] if isinstance(freq_idx, (int, list, slice)) else self.freq_domain
            }
        elif return_format.lower() == 'arrays':
            return (result_df['theta'].values, result_df['phi'].values, result_df['rcs_data'].values)
        else:
            raise ValueError(f"Invalid return_format: {return_format}. Must be 'dataframe', 'dict', or 'arrays'")

    def _parse_angle_range(self, angle_input, angle_type, angle_start, angle_stop, angle_step):
        """
        Parse various input formats for angle ranges into a list of specific angle values.
        
        This helper method handles different input types for theta and phi ranges,
        converting them into lists of discrete angle values that exist in the data.
        
        Args:
            angle_input: Input angle specification (float, list, or tuple)
            angle_type (str): Type of angle ('theta' or 'phi') for error messages
            angle_start (float): Minimum angle value in the domain
            angle_stop (float): Maximum angle value in the domain  
            angle_step (float): Step size between angle values
            
        Returns:
            list: List of angle values to filter for
        """
        if isinstance(angle_input, (int, float)):
            # Single angle value - find closest match in domain
            angle_domain = np.arange(angle_start, angle_stop + angle_step, angle_step)
            closest_idx = np.argmin(np.abs(angle_domain - angle_input))
            return [angle_domain[closest_idx]]
            
        elif isinstance(angle_input, list):
            # List of specific angle values - find closest matches
            angle_domain = np.arange(angle_start, angle_stop + angle_step, angle_step)
            matched_values = []
            for target_angle in angle_input:
                closest_idx = np.argmin(np.abs(angle_domain - target_angle))
                matched_values.append(angle_domain[closest_idx])
            return list(set(matched_values))  # Remove duplicates
            
        elif isinstance(angle_input, tuple):
            # Range specification
            if len(angle_input) == 2:
                # (start, stop) format
                start, stop = angle_input
                step = angle_step
            elif len(angle_input) == 3:
                # (start, stop, step) format  
                start, stop, step = angle_input
            else:
                raise ValueError(f"{angle_type}_range tuple must have 2 or 3 elements: (start, stop) or (start, stop, step)")
            
            # Validate range
            if start > stop:
                raise ValueError(f"{angle_type}_range start ({start}) cannot be greater than stop ({stop})")
            if step <= 0:
                raise ValueError(f"{angle_type}_range step ({step}) must be positive")
                
            # Generate range and find closest matches in domain
            target_angles = np.arange(start, stop + step, step)
            angle_domain = np.arange(angle_start, angle_stop + angle_step, angle_step)
            matched_values = []
            for target_angle in target_angles:
                if angle_start <= target_angle <= angle_stop:
                    closest_idx = np.argmin(np.abs(angle_domain - target_angle))
                    matched_values.append(angle_domain[closest_idx])
            
            if not matched_values:
                raise ValueError(f"No {angle_type} values in range [{start}, {stop}] overlap with simulation domain [{angle_start}, {angle_stop}]")
                
            return list(set(matched_values))  # Remove duplicates
            
        else:
            raise TypeError(f"{angle_type}_range must be a number, list, or tuple")

    def summary_stats(self, theta_range=None, phi_range=None, freq_idx=None, function='linear'):
        """
        Generate summary statistics for RCS data in the specified range.
        
        This function computes useful statistics like min, max, mean, std deviation
        for the RCS data, which can help with analysis and visualization scaling.
        
        Args:
            theta_range, phi_range, freq_idx: Same as get_rcs_data()
            function (str): Mathematical function to apply before computing stats
            
        Returns:
            dict: Dictionary containing summary statistics
        """
        # Get the data using the main parsing function
        data = self.get_rcs_data(theta_range=theta_range, phi_range=phi_range, 
                                freq_idx=freq_idx, function=function, return_format='dict')
        
        # Flatten all RCS data for statistics
        all_rcs_values = []
        for rcs_array in data['rcs_data']:
            if isinstance(rcs_array, np.ndarray):
                all_rcs_values.extend(rcs_array.flatten())
            else:
                all_rcs_values.append(rcs_array)
        
        all_rcs_values = np.array(all_rcs_values)
        
        # Remove any invalid values
        valid_values = all_rcs_values[np.isfinite(all_rcs_values)]
        
        if len(valid_values) == 0:
            raise ValueError("No valid RCS values found in the specified range")
        
        stats = {
            'count': len(valid_values),
            'min': np.min(valid_values),
            'max': np.max(valid_values), 
            'mean': np.mean(valid_values),
            'std': np.std(valid_values),
            'median': np.median(valid_values),
            'percentile_25': np.percentile(valid_values, 25),
            'percentile_75': np.percentile(valid_values, 75),
            'theta_range': [np.min(data['theta']), np.max(data['theta'])],
            'phi_range': [np.min(data['phi']), np.max(data['phi'])],
            'num_theta_points': len(np.unique(data['theta'])),
            'num_phi_points': len(np.unique(data['phi'])),
            'function_applied': function
        }
        
        return stats

