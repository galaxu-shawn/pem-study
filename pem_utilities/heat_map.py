import copy
import time as walltime
import numpy as np
import pyvista as pv
from tqdm import tqdm
import os
import asyncio
import threading
from pem_utilities.actor import Actors
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.antenna_device import AntennaDevice, AntennaArray, add_single_rx, add_antenna_device_from_json, enable_coupling
from pem_utilities.utils import create_mesh_from_image2
from pem_utilities.post_processing import channel_capacity
from pem_utilities.open_street_maps_geometry import get_z_elevation_from_mesh
from pem_utilities.wireless_kpis import siso_capacity,mse_lmmse,mse_ls,snr_estimation_error
from enum import Enum
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 


class Constants:
    """This class stores constants used in the toolkit."""
    T = 300
    k_b = 1.38e-23

class KPIType(Enum):
    """This class stores the types of KPIs."""
    SISO_CAPACITY = 1
    SISO_CAPACITY_LMMSE = 2
    SISO_CAPACITY_LS = 3
    SNR = 4
    SNR_LMMSE = 5
    SNR_LS = 6
    OUTAGE = 7
    OUTAGE_LMMSE = 8
    OUTAGE_LS = 9
    MSE_LMMSE = 10
    MSE_LS = 11

class KPIProps:
    """This class stores the properties of the KPIs."""
    def __init__(self):
        pass
    
    @property
    def quantity(self):
        kpiToString = {
        KPIType.SISO_CAPACITY: "siso_capacity",
        KPIType.SISO_CAPACITY_LMMSE: "siso_capacity_lmmse",
        KPIType.SISO_CAPACITY_LS: "siso_capacity_ls",
        KPIType.SNR: "snr",
        KPIType.SNR_LMMSE: "snr_lmmse",
        KPIType.SNR_LS: "snr_ls",
        KPIType.OUTAGE: "outage",
        KPIType.OUTAGE_LMMSE: "outage_lmmse",
        KPIType.OUTAGE_LS: "outage_ls",
        KPIType.MSE_LMMSE: "mse_lmmse",
        KPIType.MSE_LS: "mse_ls"
        }
        return kpiToString[self._quantity]
    
    @quantity.setter
    def quantity(self, type):
        self._quantity = type

    @property 
    def noise_power(self):
       return self._noise_power

    @noise_power.setter
    def noise_power(self, value):
        self._noise_power = value


    @property 
    def target_rate(self):
        return self._target_rate

    @target_rate.setter
    def target_rate(self, value):
        self._target_rate = value
    

    @property 
    def num_pilots(self):
        return self._num_pilots

    @num_pilots.setter
    def num_pilots(self, value):
        self._num_pilots = value
      
class HeatMapArbitraryPoints:
    """
    The HeatMap class is used to create a heatmap representation of a scene using a radar platform. using a list of points
    Only works for 2d reports

    Attributes:
        reference_actor (Actor): The reference actor for the heatmap. Will be used for node id of probe device.
        waveform (Waveform): The waveform used by the radar platform.
        mode_name (str): The mode name of the radar platform.
        bounds (list): The bounds of the scene. length=4 is 2d, length=6 is 3d.
        polarization (str): The polarization of the radar platform, 'X' 'Y' or 'Z'.
        show_patterns (bool): Whether to show the antenna patterns on subgrid. This can be used for debugging purposes.
        quantity (str): s_params, channel_capacity Whether to calculate channel capacity. Defaults to False.
    """

    def __init__(self,
                 ant_device_tx,
                 all_actors=None,
                 list_of_points=[],
                 progressive_refinement=True,
                 probe_antenna_file='dipole.ffd',
                 waveform=None,
                 mode_name='default',
                 num_parallel_points=10,
                 polarization='Z',
                 show_patterns=False,
                 Kpi_props=None,
                 show_results_in_modeler=True,
                 cmap='jet',
                 opacity=0.99):
        """
        The constructor for the HeatMap class.

        Parameters:
            reference_actor (Actor): The reference actor for the heatmap. Will be used for node id of probe device.
            waveform (Waveform): The waveform used by the radar platform. Defaults to None.
            mode_name (str): The mode name of the radar platform. Defaults to 'default'.
            bounds (list): The bounds of the scene. Defaults to [0,1,0,1]. length=4 is 2d, length=6 is 3d.
            z_elevation (float): The z elevation of the radar platform. Defaults to 0, ignored if is_3d=True.
            sampling_spacing_wl (int): The sampling spacing in wavelengths. Defaults to 10.
            num_subgrid_samples_nx (int): The number of subgrid samples in the x direction. Defaults to 10.
            num_subgrid_samples_ny (int): The number of subgrid samples in the y direction. Defaults to 10.
            polarization (str): The polarization of the radar platform. Defaults to 'Z'.  'X' 'Y' or 'Z'.
            show_patterns (bool): Whether to show the antenna patterns. Defaults to False.
            progressive_refinement (bool): Whether to use progressive refinement. Defaults to True. Interleaved updates, full results appear faster

        """

        self.ant_device_tx = ant_device_tx
        self.probe_antenna_file = probe_antenna_file
        self.show_patterns = show_patterns
        self.polarization = polarization

        self.num_parallel_points = num_parallel_points
        if progressive_refinement:
            # self._list_of_points = self.reorder_points_progressive(list_of_points)
            self._list_of_points = self.reorder_points_random(list_of_points) # this performs faster, and the impact on immediate visualizaiton seems to be the same
        else:
            self._list_of_points = list_of_points

        self.total_num_batches = int(np.ceil(len(self._list_of_points)/self.num_parallel_points))
        self.cmap = cmap
        self.opacity = opacity
        self.all_actors = all_actors

        self.Kpi_props = Kpi_props
        if Kpi_props is not None:
            self.quantity = Kpi_props.quantity.lower().replace("_","")
        else:
            self.quantity = 's_params'
        self.show_results_in_modeler = show_results_in_modeler
        self.waveform = waveform
        self.mode_name = mode_name
        self.wavelength = 299792458.0 / waveform.center_freq

        self.heatmap_mesh_for_overlay = None


        self.image = None
        self.image1 = None

        self.mesh = None
        self.image_max = 1.1e-15
        self.image_min = 1.0e-15
        self.complex_data = None
        self.solved_xyz_points = None
        # values needed for time domain heatmaps.
        self.image_time_domain = None
        self.image_time_domain_3D = None
        self.fast_time_domain = None
        self.range_domain = None
        self.probe_devices = []
        self.create_probes()

        self.simulation_time = 0.0
        
        # Threading attributes
        self._computation_thread = None
        self._stop_thread = threading.Event()
        self._data_lock = threading.Lock()
        self._computation_complete = threading.Event()
        self._current_progress = 0.0

    @property
    def list_of_points(self):
        """Get the list of points."""
        return self._list_of_points
    
    @list_of_points.setter
    def list_of_points(self, value):
        """Set the list of points and update total_num_batches."""
        self._list_of_points = self.reorder_points_random(value)
        self.total_num_batches = int(np.ceil(len(self._list_of_points)/self.num_parallel_points))

    def update_quantity(self):
        if self.Kpi_props is not None:
            self.quantity = self.Kpi_props.quantity.lower().replace("_","")

    def create_probes(self):
        """
        This method sets up the radar platform. It creates an AntennaArray object which represents a blank antenna device.
        The antenna device is manually set up using a combination of direct API calls and the AntennaDevice class.
        The created antenna device is then assigned to the probe_device attribute of the HeatMap instance.

        The AntennaArray is initialized with the following parameters:
            name (str): The name of the antenna array. Here it is set as 'array'.
            waveform (Waveform): The waveform used by the radar platform. It is taken from the HeatMap instance.
            mode_name (str): The mode name of the radar platform. It is taken from the HeatMap instance.
            file_name (str): The file name of the antenna device. Here it is set as 'dipole.ffd'.
            polarization (str): The polarization of the radar platform. It is taken from the HeatMap instance.
            rx_shape (list): The shape of the receiver antenna array. It is a list of two integers representing the number of subgrid samples in the x and y directions.
            tx_shape (int): The shape of the transmitter antenna array. Here it is set as 0.
            spacing_wl_x (int): The spacing in wavelengths in the x direction. It is taken from the HeatMap instance.
            spacing_wl_y (int): The spacing in wavelengths in the y direction. It is taken from the HeatMap instance.
            parent_h_node (HNode): The parent HNode for the antenna device. It is taken from the reference actor of the HeatMap instance.
            load_pattern_as_mesh (bool): Whether to load the antenna pattern as a mesh. It is taken from the HeatMap instance.
            scale_pattern (float): The scale of the antenna pattern. Here it is set as 0.3.
        """

        for each in range(self.num_parallel_points):

            if self.polarization.lower() == 'x':
                # 3x3 rotation matrix, where the z axis now becomes the x axis
                rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            elif self.polarization.lower() == 'y':
                # 3x3 rotation matrix, where the z axis now becomes the y axis
                rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            else: # z polarization
                rotation = np.eye(3)
                
            # get extension of self.probe_antenna_file
            if self.probe_antenna_file is not None:
                if self.probe_antenna_file.endswith('.ffd'):
                    probe_device = add_single_rx(self.all_actors,
                                                waveform=self.waveform,
                                                mode_name=self.mode_name,
                                                pos=np.zeros(3), rot=rotation, lin=np.zeros(3), ang=np.zeros(3),
                                                parent_h_node=None,
                                                ffd_file=self.probe_antenna_file,
                                                load_pattern_as_mesh=self.show_patterns,
                                                scale_pattern=0.3)
                elif self.probe_antenna_file.endswith('.json'):
                    probe_device = add_antenna_device_from_json(self.all_actors,
                                                                self.probe_antenna_file,
                                                                mode_name=self.mode_name,
                                                                pos=np.zeros(3), rot=rotation, 
                                                                lin=np.zeros(3), ang=np.zeros(3),
                                                                parent_h_node=None,
                                                                load_pattern_as_mesh=self.show_patterns,
                                                                scale_pattern=0.3)
                else:
                    raise ValueError('Probe antenna file must be either .ffd or .json')
                    

            enable_coupling(self.mode_name, self.ant_device_tx, probe_device)

            self.probe_devices.append(probe_device)


    def reorder_points_random(self,points):
        """
        Randomly shuffle the order of points.
        
        Args:
            points: Array of shape (N, 3) representing grid points
        
        Returns:
            Randomly reordered array of points
        """
        
        # Create a copy to avoid modifying the original
        reordered_points = points.copy()
        
        # Randomly shuffle the points
        np.random.shuffle(reordered_points)
        
        print('Reordering points randomly...Done')
        return reordered_points

    def reorder_points_progressive(self,points):
        """
        Reorder a flattened grid of points for progressive refinement.
        This will show the full image right away, but with less detail, and then refine the image over time.
        This reordering can be slow for large grids, suggest using random reordering instead.
        Args:
            points: Array of shape (N, 3) representing grid points
            num_x: Number of points in X direction
            num_y: Number of points in Y direction
        
        Returns:
            Reordered array of points for progressive rendering
        """
        print('Reordering points for progressive refinement...')
        num_x = len(np.unique(points[:,0]))
        num_y = len(np.unique(points[:,1]))
        # Reshape to 2D grid for easier manipulation
        grid_shape = (num_y, num_x, 3)  # note: meshgrid produces (ny, nx) shape
        grid = points.reshape(grid_shape)
        
        # Determine maximum stride (power of 2)
        max_stride = 1
        while max_stride * 2 <= min(num_x, num_y):
            max_stride *= 2
        
        reordered_indices = []
        
        # Start from coarsest level and refine
        stride = max_stride
        while stride >= 1:
            # Generate indices for this refinement level
            for y_idx in range(0, num_y, stride):
                for x_idx in range(0, num_x, stride):
                    flat_idx = y_idx * num_x + x_idx
                    
                    # Only add if not already computed in previous level
                    if flat_idx not in reordered_indices:
                        reordered_indices.append(flat_idx)
            
            stride //= 2
        
        # Reorder the points array
        reordered_points = points[reordered_indices]
        print('Reordering points for progressive refinement...Done')
        return reordered_points
    
    def _update_batch(self,idx):
        """
        This method updates the batch index and calculates the start and end indices for the x and y directions.
        It also updates the position of the probe device based on the current batch index.

        Parameters:
            idx (int): The current batch index.

        Attributes updated:
            batch_idx (int): The current batch index.
            idx_x_start (int): The start index for the x direction.
            idx_x_end (int): The end index for the x direction.
            idx_y_start (int): The start index for the y direction.
            idx_y_end (int): The end index for the y direction.
            probe_device.coord_sys.pos (list): The position of the probe device.
        """

        self.batch_idx = idx
        # ToDo, deal with extra points
        if (idx*self.num_parallel_points+self.num_parallel_points)<len(self.list_of_points):
            points_to_use = self.list_of_points[idx*self.num_parallel_points:idx*self.num_parallel_points+self.num_parallel_points]
        else:
            points_to_use = self.list_of_points[
                            idx * self.num_parallel_points:len(self.list_of_points)]

        for idx, xyz in enumerate(points_to_use):

            self.probe_devices[idx].coord_sys.pos = xyz
            self.probe_devices[idx].coord_sys.update()

        return points_to_use


    def update_heatmap_only(self,
                                function='db',
                                plot_min=-100,
                                plot_max=-50,
                                pulse_idx=0,
                                freq_idx=0,
                                tx_idx=0,
                                rx_idx=0):
        """
        This method updates the 2D heatmap representation of the scene using a radar platform.
        This is the bare minimum version that does not use modeler for visualization. or other advanced features.
        Intended for use with omniverse visualization or custom visualization.

        Parameters:
            function (str): The function used to calculate the response. Defaults to 'db'.
            plot_min (float): The minimum value for the plot. Defaults to -100.
            plot_max (float): The maximum value for the plot. Defaults to -50.

        Attributes updated:
            image (ndarray): The heatmap image.
            mesh (PolyData): The mesh of the heatmap.
            image_min (float): The minimum value of the image.
            image_max (float): The maximum value of the image.
            heatmap_mesh_for_overlay (PolyData): The heatmap mesh for overlay.
        """

        options = {}
        options['cmap'] = self.cmap
        options['opacity'] = self.opacity

        self.complex_data = []
        # if self.complex_data is None:
        #     self.complex_data = np.zeros(len(self.list_of_points), dtype='complex')
        xyz_points = []
        responses = []
        num_tx = 1
        num_rx = 1
        mesh_update_counter = 0
        how_often_to_update_mesh= max(1,int(self.total_num_batches/10)) # update mesh 10 times during the simulation
        for idx in tqdm(range(self.total_num_batches)):
            points_one_batch = self._update_batch(idx)
            xyz_points.extend(points_one_batch)
            start_time = walltime.time()
            pem_api_manager.isOK(pem.computeResponseSync())
            self.simulation_time += walltime.time() - start_time
            response = []

            if self.ant_device_tx.waveforms[self.mode_name].output.lower() == 'rangedoppler':
                ouput_format = RssPy.ResponseType.RANGE_DOPPLER
            else:
                ouput_format = RssPy.ResponseType.FREQ_PULSE

            for probe_idx, xyz in enumerate(points_one_batch):
                (ret, response_temp) = pem.retrieveP2PResponse(self.ant_device_tx.modes[self.mode_name],
                                                          self.probe_devices[probe_idx].modes[self.mode_name],
                                                          ouput_format)

                response.append(response_temp[tx_idx,rx_idx,pulse_idx,freq_idx])
            self.complex_data.extend(np.array(response))
            
            should_update_modeler = mesh_update_counter%how_often_to_update_mesh
            mesh_update_counter+=1
            if should_update_modeler==0:
                # create a pyvista mesh from the series of xyz points with the scalar data being self.complex_data
                cloud = pv.PolyData(xyz_points)
                self.mesh = cloud.delaunay_2d()
                self.mesh[function] = self._apply_function(self.complex_data, function=function)
    

    def update_heatmap(self,
                       function='db',
                       modeler=None,
                       sub_grid_updates=True,
                       td_output_size=None,
                       window='flat',
                       plot_min=-100,
                       plot_max=-50,
                       add_mesh_to_overlay=True,
                       pulse_idx=0,
                       freq_idx=0,
                       tx_idx=0,
                       rx_idx=0,
                       time_domain_output=False,
                       save_all_data=False):
        """
        This method updates the 2D heatmap representation of the scene using a radar platform.

        Parameters:
            function (str): The function used to calculate the response. Defaults to 'db'.
            modeler (Modeler): The modeler used to visualize the heatmap. Defaults to None.
            create_surface (bool): Whether to create a surface for the heatmap. Defaults to True.
            sub_grid_updates (bool): Whether to update the subgrid. Defaults to True.
            plot_min (float): The minimum value for the plot. Defaults to -100.
            plot_max (float): The maximum value for the plot. Defaults to -50.
            add_mesh_to_overlay (bool): Whether to add the mesh to the overlay. Defaults to True.

        Attributes updated:
            image (ndarray): The heatmap image.
            mesh (PolyData): The mesh of the heatmap.
            image_min (float): The minimum value of the image.
            image_max (float): The maximum value of the image.
            heatmap_mesh_for_overlay (PolyData): The heatmap mesh for overlay.
        """

        options = {}
        options['cmap'] = self.cmap
        options['opacity'] = self.opacity
        if modeler is not None:
            pyvistafunc = modeler.pl.add_mesh
        else:  # can't overlay results if we don't have a modeler
            self.show_results_in_modeler = False
            create_surface = False

        if td_output_size is None:
            td_output_size = self.waveform.num_freq_samples

        self.complex_data = []
        self.all_data = []
        # if self.complex_data is None:
        #     self.complex_data = np.zeros(len(self.list_of_points), dtype='complex')
        xyz_points = []
        responses = []
        num_tx = 1
        num_rx = 1

        # Performance of visualization updates is not great, so lets limit the number of updates
        how_often_to_update_modeler = max(1,int(self.total_num_batches/10)) # update modeler 10 times during the simulation
        model_update_counter = 0

        for idx in tqdm(range(self.total_num_batches)):
            points_one_batch = self._update_batch(idx)
            xyz_points.extend(points_one_batch)
            start_time = walltime.time()
            pem_api_manager.isOK(pem.computeResponseSync())
            self.simulation_time += walltime.time() - start_time
            response = []

            if self.ant_device_tx.waveforms[self.mode_name].output.lower() == 'rangedoppler':
                ouput_format = RssPy.ResponseType.RANGE_DOPPLER
            else:
                ouput_format = RssPy.ResponseType.FREQ_PULSE

            for probe_idx, xyz in enumerate(points_one_batch):
                (ret, response_temp) = pem.retrieveP2PResponse(self.ant_device_tx.modes[self.mode_name],
                                                          self.probe_devices[probe_idx].modes[self.mode_name],
                                                          ouput_format)
                if tx_idx is None and rx_idx is None: # do all tx and all rx
                    # get number of tx and rx in response
                    num_tx = response_temp.shape[0]
                    num_rx = response_temp.shape[1]
                elif tx_idx is None:
                    num_tx = response_temp.shape[0]
                    num_rx = 1
                elif rx_idx is None:
                    num_tx = 1
                    num_rx = response_temp.shape[0]
                else:
                    num_tx = 1
                    num_rx = 1
                num_pulse = response_temp.shape[2]
                if pulse_idx is not None:
                    # only 1 pulse, if it is None, do all pulses
                    num_pulse = 1
                num_freq = response_temp.shape[-1]
                if freq_idx is not None:
                    # only 1 pulse, if it is None, do all pulses
                    num_freq = 1

                if time_domain_output: # must be a time domain
                    win_range, _ = self._window_function(function=window, size=response_temp[0,0,0].shape[-1])
                    sf_upsample = td_output_size / response_temp[0,0,0].shape[-1] # scale factor for up/down sampling
                    response_temp2 = np.zeros((num_tx, num_rx, num_pulse, td_output_size), dtype='complex')
                    for tidx in range(num_tx):
                        for ridx in range(num_rx):
                            for pidx in range(num_pulse):
                                if pulse_idx is None:
                                    pulse_idx=pidx
                                windowed_data = win_range*response_temp[tidx,ridx, pulse_idx, :]
                                response_temp2[tidx,ridx,pidx] = sf_upsample * np.fft.ifft(windowed_data, n=td_output_size)
                    response.append(response_temp2)
                else:
                    # use a 4 bit truth table to determine how to slice the data. None for tx_idx, rx_idx, pulse_idx, freq_idx is all values
                    # of the array. if an integer is passed in, it will slice the array at that index
                    # if None, then all values are used
                    if tx_idx is not None:
                        func_tx = tx_idx
                    else:
                        func_tx = slice(tx_idx)
                    if rx_idx is not None:
                        func_rx = rx_idx
                    else:
                        func_rx = slice(rx_idx)
                    if pulse_idx is not None:
                        func_pulse = pulse_idx
                    else:
                        func_pulse = slice(pulse_idx)
                    if freq_idx is not None:
                        func_freq = freq_idx
                    else:
                        func_freq = slice(freq_idx)
                    response.append(response_temp[func_tx,func_rx,func_pulse,func_freq])
                    if save_all_data:
                        self.all_data.append(response_temp)

            self.complex_data.extend(np.array(response))
            
            should_update_modeler = model_update_counter%how_often_to_update_modeler
            model_update_counter+=1
            if self.show_results_in_modeler and sub_grid_updates and should_update_modeler==0:
                model_update_counter+=1
                # calculate response in dB to overlay in pyvista plotter

                # create a pyvista mesh from the series of xyz points with the scalar data being self.complex_data
                cloud = pv.PolyData(xyz_points)
                self.mesh = cloud.delaunay_2d()
                self.mesh[function] = self._apply_function(self.complex_data, function=function)

                im_min = self.mesh[function].min()
                im_max = self.mesh[function].max()

                if add_mesh_to_overlay:
                    if self.heatmap_mesh_for_overlay is None:
                        pyvistafunc(self.mesh, **options)
                        self.heatmap_mesh_for_overlay = self.mesh
                    else:
                        self.heatmap_mesh_for_overlay.copy_from(self.mesh)

                    modeler.update_frame()  # update pyvista plotter
                    # update scalar bar for current range of values
                    # 'image_mag' is default name of color bar associated with heatmaps
                    modeler.pl.update_scalar_bar_range([plot_min, plot_max], function)
        xyz_points = np.array(xyz_points)
        self.complex_data = np.array(self.complex_data)
        self.solved_xyz_points = xyz_points

        if self.show_results_in_modeler:
            # calculate response in dB to overlay in pyvista plotter

            # create a pyvista mesh from the series of xyz points with the scalar data being self.complex_data
            cloud = pv.PolyData(self.solved_xyz_points)
            self.mesh = cloud.delaunay_2d()
            self.mesh[function] = self._apply_function(self.complex_data, function=function)

            im_min = self.mesh[function].min()
            im_max = self.mesh[function].max()

            if add_mesh_to_overlay:
                if self.heatmap_mesh_for_overlay is None:
                    pyvistafunc(self.mesh, **options)
                    self.heatmap_mesh_for_overlay = self.mesh
                else:
                    self.heatmap_mesh_for_overlay.copy_from(self.mesh)

                if sub_grid_updates:
                    modeler.update_frame()  # update pyvista plotter
                # update scalar bar for current range of values
                # 'image_mag' is default name of color bar associated with heatmaps
                modeler.pl.update_scalar_bar_range([plot_min, plot_max], function)

        if self.show_results_in_modeler:
            modeler.update_frame()

        self.complex_data = np.array(self.complex_data)
        if save_all_data:
            self.all_data = np.array(self.all_data)

    def update_heatmap_time_domain(self,
                                   function='db',
                                   modeler=None,
                                   plot_min=-100,
                                   plot_max=-50,
                                   add_mesh_to_overlay=True,
                                   td_output_size=None,
                                   window='flat',
                                   pulse_idx=0,
                                   tx_idx=0,
                                   rx_idx=0,
                                   start_animation_after_time=None,
                                   end_animation_after_time=None,
                                   use_slider_widget=False,
                                   loop_animation=False):
        """
        Updates the heatmap in the time domain based on the provided parameters.
        Parameters:
            function (str): The function to apply to the data (default: 'db').
            modeler (object): The modeler object used for visualization (default: None).
            plot_min (float): Minimum value for the plot range (default: -100).
            plot_max (float): Maximum value for the plot range (default: -50).
            add_mesh_to_overlay (bool): Whether to add the mesh to the overlay (default: True).
            td_output_size (tuple or None): Size of the time domain output (default: None).
            window (str): Windowing function to apply to the data (default: 'flat').
            pulse_idx (int): Index of the pulse to use (default: 0).
            tx_idx (int): Index of the transmitter to use (default: 0).
            rx_idx (int): Index of the receiver to use (default: 0).
            start_animation_after_time (float or None): Start time for animation (default: None).
            end_animation_after_time (float or None): End time for animation (default: None).
            use_slider_widget (bool): Whether to use a slider widget for time steps (default: False).
            loop_animation (bool): Whether to loop the animation (default: False).
        Returns:
            None
        Notes:
            - If the quantity is 'channelcapacity', it will be replaced with 's-parameters' as channel capacity is not supported for time domain heatmaps.
            - The method calculates the range and time domains based on the waveform and updates the heatmap accordingly.
            - Time stamps are generated for all time steps and can be truncated based on user-specified start and end times.
            - If `use_slider_widget` is True, a slider widget is added to control the time steps interactively.
            - If `loop_animation` is True, the animation will loop through the time steps.
            - The method supports adding a mesh overlay to the visualization and updating the scalar bar range dynamically.
        """
        if self.quantity == 'channelcapacity':
            print("WARNING: Channel capacity is not supported for time domain heatmaps")
            print("Using S-Parameters")
            self.quantity = 's-parameters'


        self.waveform.get_response_domains(self.ant_device_tx.modes[self.mode_name])
        # this is calculated for round trip, so we multiply by 2 to get 1 way
        self.range_domain = self.waveform.rng_domain * 2
        self.fast_time_domain = self.waveform.fast_time_domain * 2
        print(f"Range domain max: {self.range_domain[-1]}")
        print(f"Time domain max: {self.fast_time_domain[-1]}")

        # after this is called, we will have all the data we need to create the time domain heatmaps.
        # this will create a new parameter image_time_domain,which will be a 3D array of the x,y,time domain data
        # ToDo add upsampling and windowing to the data
        self.update_heatmap(add_mesh_to_overlay=False,
                            td_output_size=td_output_size,
                            window=window,
                            pulse_idx=pulse_idx,
                            freq_idx=None,
                            tx_idx=0,
                            rx_idx=0,
                            time_domain_output=True)

        # time stamps are same lenght as frequncy domain, so we can use the same time stamps for all time steps
        time_stamps = np.linspace(0,self.fast_time_domain[-1],num=self.complex_data.shape[-1])

        # truncate time if users specifies

        if end_animation_after_time is not None:
            # find the closest time to user asked time
            idx = (np.abs(time_stamps - end_animation_after_time)).argmin()
            if idx<len(time_stamps):
                time_stamps = time_stamps[:idx]
        if start_animation_after_time is not None:
            if start_animation_after_time >= end_animation_after_time:
                print('Start animation time must be less than end animation time')
                start_animation_after_time=0
            idx = (np.abs(time_stamps - start_animation_after_time)).argmin()
            if idx<len(time_stamps):
                time_stamps = time_stamps[idx:]
        if add_mesh_to_overlay:
            options = {}
            options['opacity'] = self.opacity
            options['cmap'] = self.cmap
            pyvistafunc = modeler.pl.add_mesh

            cloud = pv.PolyData(self.solved_xyz_points)
            mesh = cloud.delaunay_2d()
            self.mesh = mesh
            if use_slider_widget:
                all_meshes = []
                for time_idx, time in tqdm(enumerate(time_stamps)):
                    mesh[function] = self._apply_function(self.complex_data[time_idx,tx_idx,rx_idx,pulse_idx], function=function)
                    all_meshes.append(mesh)

                def get_mesh_time_step(value):
                    time_idx = int((np.abs(time_stamps - value)).argmin())
                    mesh = all_meshes[time_idx]
                    if self.heatmap_mesh_for_overlay is None:
                        pyvistafunc(mesh, **options)
                        self.heatmap_mesh_for_overlay = mesh
                    else:
                        self.heatmap_mesh_for_overlay.copy_from(mesh)
                    # pyvistafunc(all_meshes[time_idx], **options)
                    modeler.pl.update_scalar_bar_range([plot_min, plot_max], 'image_mag')
                    if hasattr(modeler,'mpl_ax_handle'):
                        modeler.mpl_ax_handle.set_cmap(self.cmap)
                        modeler.mpl_ax_handle.set_data(self.image_time_domain[:, :, time_idx])
                        modeler.mpl_ax_handle.set_clim(vmin=plot_min, vmax=plot_max)
                    return
                modeler.pl.add_slider_widget(get_mesh_time_step, [0, time_stamps[-1]], title='adsf')
                modeler.update_frame(write_frame=False)

            else:
                time = 0
                time_idx = 0
                num_loops = 0
                while time<=time_stamps[-1]:
                    if loop_animation:
                        if time_idx>=len(time_stamps):
                            time_idx=0
                            num_loops+=1
                        time = time_stamps[time_idx]
                    else:
                        if time_idx>=len(time_stamps):
                            break
                        time = time_stamps[time_idx]

                    temp_mesh = copy.deepcopy(self.mesh)
                    temp_mesh[function] = self._apply_function(self.complex_data[:,tx_idx,rx_idx,pulse_idx,time_idx], function=function)

                    im_min = temp_mesh[function].min()
                    im_max = temp_mesh[function].max()


                    if add_mesh_to_overlay:
                        if self.heatmap_mesh_for_overlay is None:
                            pyvistafunc(temp_mesh, **options)
                            self.heatmap_mesh_for_overlay = temp_mesh
                        else:
                            self.heatmap_mesh_for_overlay.copy_from(temp_mesh)
                        # update scalar bar for current range of values
                        # 'image_mag' is default name of color bar associated with heatmaps
                        modeler.pl.update_scalar_bar_range([plot_min, plot_max], function)
                        if loop_animation:
                            # ToDo, this prevents the video to be infnietly long during loop animation, but
                            # causes a bad crash when modeler window is closed
                            if num_loops>=1:
                                modeler.pl.update(force_redraw=False)
                            else:
                                modeler.update_frame(write_frame=True)
                        else:
                            modeler.update_frame()
                    time_idx += 1


    def _apply_function(self,data, function='db'):
        function = function.lower().replace('_','')

        if function.lower() == 'db':
            data = 20 * np.log10(np.fmax(np.abs(data), 1.e-30))
        elif function.lower() == 'db10':
            data = 10 * np.log10(np.fmax(np.abs(data), 1.e-30))
        elif function.lower() == 'db20':
            data = 20 * np.log10(np.fmax(np.abs(data), 1.e-30))
        elif function.lower() == 'real':
            data = np.fmax(np.real(data), 1.e-30)
        elif function.lower() == 'imag':
            data = np.fmax(np.imag(data), 1.e-30)
        else:
            data = np.fmax(np.abs(data), 1.e-30)

        return data

    def _window_function(self,function='flat', size=512, dbDown=30):
        if function.lower() == 'hann':
            win = np.hanning(size)
        elif function.lower() == 'hamming':
            win = np.hamming(size)
        elif function.lower() == 'blackman':
            win = np.blackman(size)
        elif function.lower() == 'bartlett':
            win = np.bartlett(size)
        elif function.lower() == 'kaiser':
            win = np.kaiser(size, dbDown)
        elif function.lower() == 'flat':
            win = np.ones(size)
        else:
            print('Warning: Invalid window function, defaulting to flat')
            win = np.ones(size)
        win_sum = np.sum(win)
        win *= size/win_sum
        return win, win_sum

    def start_background_computation(self, pulse_idx=0, freq_idx=0, tx_idx=0, rx_idx=0):
        """
        Start the background computation thread that continuously updates self.complex_data.
        
        Parameters:
            pulse_idx (int): The pulse index. Defaults to 0.
            freq_idx (int): The frequency index. Defaults to 0.
            tx_idx (int): The transmitter index. Defaults to 0.
            rx_idx (int): The receiver index. Defaults to 0.
        """
        if self._computation_thread is not None and self._computation_thread.is_alive():
            print("Background computation is already running")
            return
        
        # Reset stop event
        self._stop_thread.clear()
        self._computation_complete.clear()
        
        # Start the computation thread
        self._computation_thread = threading.Thread(
            target=self._background_computation_worker,
            args=(pulse_idx, freq_idx, tx_idx, rx_idx),
            daemon=True
        )
        self._computation_thread.start()
        print("Background computation started")
    
    def _background_computation_worker(self, pulse_idx, freq_idx, tx_idx, rx_idx):
        """
        Worker function that runs in the background thread.
        Continuously computes and updates self.complex_data and self.solved_xyz_points.
        """
        # Thread-local storage for data
        local_complex_data = []
        local_xyz_points = []
        
        for idx in range(self.total_num_batches):
            # Check if we should stop
            if self._stop_thread.is_set():
                print("Background computation stopped by user")
                return
            
            # Update batch and get points
            points_one_batch = self._update_batch(idx)
            local_xyz_points.extend(points_one_batch)
            
            # Compute response
            start_time = walltime.time()
            pem_api_manager.isOK(pem.computeResponseSync())
            self.simulation_time += walltime.time() - start_time
            
            if self.ant_device_tx.waveforms[self.mode_name].output.lower() == 'rangedoppler':
                ouput_format = RssPy.ResponseType.RANGE_DOPPLER
            else:
                ouput_format = RssPy.ResponseType.FREQ_PULSE

            # Retrieve responses
            response = []
            for probe_idx, xyz in enumerate(points_one_batch):
                (ret, response_temp) = pem.retrieveP2PResponse(
                    self.ant_device_tx.modes[self.mode_name],
                    self.probe_devices[probe_idx].modes[self.mode_name],
                    ouput_format
                )
                response.append(response_temp[tx_idx, rx_idx, pulse_idx, freq_idx])
            
            local_complex_data.extend(np.array(response))
            
            # Update shared data with thread lock
            with self._data_lock:
                self.complex_data = local_complex_data.copy()
                self.solved_xyz_points = local_xyz_points.copy()
                self._current_progress = (idx + 1) / self.total_num_batches
        
        # Mark computation as complete
        self.update_mesh_threaded(function='db')
        self._computation_complete.set()
        print("Background computation completed")
    
    def stop_background_computation(self):
        """
        Stop the background computation thread.
        """
        if self._computation_thread is None or not self._computation_thread.is_alive():
            print("No background computation is running")
            return
        
        print("Stopping background computation...")
        self._stop_thread.set()
        self._computation_thread.join(timeout=5.0)
        
        if self._computation_thread.is_alive():
            print("Warning: Thread did not stop gracefully")
        else:
            print("Background computation stopped")
    
    def get_computation_progress(self):
        """
        Get the current progress of the background computation.
        
        Returns:
            float: Progress value between 0.0 and 1.0
        """
        with self._data_lock:
            return self._current_progress
    
    def is_computation_complete(self):
        """
        Check if the background computation has completed.
        
        Returns:
            bool: True if computation is complete, False otherwise
        """
        return self._computation_complete.is_set()
    
    def get_current_mesh(self, function='db'):
        """
        Create and return a mesh from the current state of complex_data.
        This is thread-safe and can be called while background computation is running.
        
        Parameters:
            function (str): The function to apply to the data. Defaults to 'db'.
            
        Returns:
            pv.PolyData: The mesh with current data, or None if no data available
        """
        with self._data_lock:
            if self.solved_xyz_points is None or len(self.solved_xyz_points) == 0:
                return None
            
            # Create copies to avoid holding the lock too long
            xyz_points_copy = self.solved_xyz_points.copy()
            complex_data_copy = self.complex_data.copy()
        
        # Create mesh outside the lock
        cloud = pv.PolyData(xyz_points_copy)
        mesh = cloud.delaunay_2d()
        if mesh.n_cells >0:
            mesh[function] = self._apply_function(complex_data_copy, function=function)
            return mesh
        return None
    
    def update_mesh_threaded(self, function='db'):
        """
        Update self.mesh with the current state of complex_data.
        This replaces update_mesh_async() for threaded usage.
        
        Parameters:
            function (str): The function to apply to the data. Defaults to 'db'.
            
        Returns:
            bool: True if mesh was updated, False if no data available
        """
        mesh = self.get_current_mesh(function)
        if mesh is not None:
            self.mesh = mesh
            return True
        return False

class HeatMap:
    """
    The HeatMap class is used to create a heatmap representation of a scene using a radar platform.

    Attributes:
        reference_actor (Actor): The reference actor for the heatmap. Will be used for node id of probe device.
        waveform (Waveform): The waveform used by the radar platform.
        mode_name (str): The mode name of the radar platform.
        bounds (list): The bounds of the scene. length=4 is 2d, length=6 is 3d.
        z_elevation (float): The z elevation where the plane will be evaluated
        sampling_spacing_wl (int): The sampling spacing in wavelengths.
        num_subgrid_samples_nx (int): The number of subgrid samples in the x direction.
        num_subgrid_samples_ny (int): The number of subgrid samples in the y direction.
        polarization (str): The polarization of the radar platform, 'X' 'Y' or 'Z'.
        show_patterns (bool): Whether to show the antenna patterns on subgrid. This can be used for debugging purposes.
        quantity (str): s_params, channel_capacity Whether to calculate channel capacity. Defaults to False.
    """

    def __init__(self,all_actors,
                 waveform=None,
                 mode_name='default',
                 bounds=[0,1,0,1],
                 z_elevation=None,
                 sampling_spacing_wl=10,
                 num_subgrid_samples_nx=10,
                 num_subgrid_samples_ny=10,
                 polarization='Z',
                 show_patterns=False,
                 probe_device=None,
                 Kpi_props=None,
                 show_results_in_modeler=True,
                 cmap='jet',
                 opacity=0.99):
        """
        The constructor for the HeatMap class.

        Parameters:
            reference_actor (Actor): The reference actor for the heatmap. Will be used for node id of probe device.
            waveform (Waveform): The waveform used by the radar platform. Defaults to None.
            mode_name (str): The mode name of the radar platform. Defaults to 'default'.
            bounds (list): The bounds of the scene. Defaults to [0,1,0,1]. length=4 is 2d, length=6 is 3d.
            z_elevation (float): The z elevation of the radar platform. Defaults to 0, ignored if is_3d=True.
            sampling_spacing_wl (int): The sampling spacing in wavelengths. Defaults to 10.
            num_subgrid_samples_nx (int): The number of subgrid samples in the x direction. Defaults to 10.
            num_subgrid_samples_ny (int): The number of subgrid samples in the y direction. Defaults to 10.
            polarization (str): The polarization of the radar platform. Defaults to 'Z'.  'X' 'Y' or 'Z'.
            show_patterns (bool): Whether to show the antenna patterns. Defaults to False.

        """



        self.show_patterns = show_patterns
        self.polarization = polarization
        self.total_num_batches = 1
        self.num_subgrid_samples_nx = num_subgrid_samples_nx
        self.num_subgrid_samples_ny = num_subgrid_samples_ny
        self.sample_spacing_in_wl = sampling_spacing_wl
        self.z_elevation = z_elevation
        self.all_z_elevations = []
        self.bounds = bounds
        self.cmap = cmap
        self.opacity = opacity
        self.all_actors = all_actors
        probe_name = self.all_actors.add_actor(name='probe')
        self.probe_actor = all_actors.actors[probe_name]
        self.Kpi_props = Kpi_props
        if Kpi_props is not None:
            self.quantity = Kpi_props.quantity.lower().replace("_","")
        else:
            self.quantity = 's_params'
        self.show_results_in_modeler = show_results_in_modeler
        self.waveform = waveform
        self.mode_name = mode_name
        self.wavelength = 299792458.0 / waveform.center_freq
        self.z_slices = []
        self.heatmap_mesh_for_overlay = None
        self.point_cloud_mesh = pv.PolyData()

        self.batch_idx = 0
        self.image = None
        self.image1 = None
        self.image_3d = None
        self.mesh = None
        self.image_max = 1.1e-15
        self.image_min = 1.0e-15
        self.complex_data = None
        self.create_sub_grid()
        self.create_grid()

        # values needed for time domain heatmaps.
        self.image_time_domain = None
        self.image_time_domain_3D = None
        self.fast_time_domain = None
        self.range_domain = None

        # only create the probe device if it is not passed in (already created)
        self.probe_device = probe_device
        if self.probe_device is None:
            self.create_probes()

        if len(self.bounds) == 4:
            self.is_3d = False
            if self.z_elevation is None:
                print("Warning: 2D heatmap created without z_elevation, setting to 0")
                self.z_elevation = 0
                self.all_z_elevations.append(self.z_elevation)
        elif len(self.bounds) == 6:
            if self.bounds[-1] != self.bounds[-2]:
                self.is_3d = True
            else: # if z bounds are equal, then just make it 2D
                self.is_3d = False
                if self.z_elevation is None:
                    self.z_elevation = self.bounds[-1]
                    self.all_z_elevations.append(self.z_elevation)
                    print(f"Warning: 2D heatmap created without z_elevation, setting to {self.z_elevation}")
        else:
            raise ValueError('Bounds must be either 4 (2D) or 6 (3D) elements')

        if self.is_3d:
            z_spacing = self.wavelength * sampling_spacing_wl
            z_num = int((self.bounds[5] - self.bounds[4]) / z_spacing)
            if z_num==0:
                z_num=1
                z_vals = np.array([self.bounds[4]])
            else:
                z_vals = np.linspace(self.bounds[4], self.bounds[5], num=z_num)
            for z in z_vals:
                self.all_z_elevations.append(z)
                self.z_slices.append(HeatMap(self.all_actors,
                                     waveform=self.waveform,
                                     mode_name=self.mode_name,
                                     bounds=self.bounds[:4],
                                     z_elevation=z,
                                     sampling_spacing_wl=self.sample_spacing_in_wl,
                                     num_subgrid_samples_nx=self.num_subgrid_samples_nx,
                                     num_subgrid_samples_ny=self.num_subgrid_samples_ny,
                                     polarization=self.polarization,
                                     show_patterns=self.show_patterns,
                                     probe_device=self.probe_device,
                                     Kpi_props=None,
                                     show_results_in_modeler=self.show_results_in_modeler))
        self.all_z_elevations = np.array(self.all_z_elevations)

    def update_quantity(self):
        if self.Kpi_props is not None:
            self.quantity = self.Kpi_props.quantity.lower().replace("_","")

    def create_sub_grid(self):

        ############
        # Design our probe patch, these are all the number of samples that will be collected in one simulation. Effecitly,
        # this is a sub sampling of the overall larger grid. This is used to accelerate the calculation.

        self.total_size_in_wl_sub_x = (self.num_subgrid_samples_nx-1) * self.sample_spacing_in_wl
        self.total_size_in_wl_sub_y = (self.num_subgrid_samples_ny-1) * self.sample_spacing_in_wl
        self.total_size_in_meter_sub_x = self.total_size_in_wl_sub_x * self.wavelength
        self.total_size_in_meter_sub_y = self.total_size_in_wl_sub_y * self.wavelength

        self.subgrid_rx_positions_x = np.linspace(0, self.total_size_in_meter_sub_x , num=self.num_subgrid_samples_nx)
        self.subgrid_rx_positions_y = np.linspace(0, self.total_size_in_meter_sub_y , num=self.num_subgrid_samples_ny)
        self.all_subgrid_rx_positions = np.array(np.meshgrid(self.subgrid_rx_positions_x, self.subgrid_rx_positions_y)).T
        self.all_subgrid_rx_positions = self.all_subgrid_rx_positions.reshape((self.num_subgrid_samples_nx*self.num_subgrid_samples_ny,2))

    def create_grid(self):
        # create a grid of samples that will be used to sample the scene, which is a grid that is spaced by the size of
        # our subgrid/probe defined above. This will sample the entire scene, but could be limited if we wanted to

        self.total_size_scene_x = self.bounds[1] - self.bounds[0]
        self.total_size_scene_y = self.bounds[3] - self.bounds[2]
        # there might be some unequal spacing at the end of the grid,
        # so we will just choose the number of samples to be the integer spacing that gets close the edge
        self.num_grid_samples_nx = int(self.total_size_scene_x/(self.total_size_in_meter_sub_x+self.wavelength*self.sample_spacing_in_wl))+1
        self.num_grid_samples_ny = int(self.total_size_scene_y/(self.total_size_in_meter_sub_y+self.wavelength*self.sample_spacing_in_wl))+1

        self.total_actual_size_x = ((self.num_grid_samples_nx-1)*self.num_subgrid_samples_nx) * self.wavelength*self.sample_spacing_in_wl
        self.total_actual_size_y = ((self.num_grid_samples_ny-1)*self.num_subgrid_samples_ny) * self.wavelength*self.sample_spacing_in_wl
        self.grid_positions_x = np.linspace(self.bounds[0],self.bounds[0]+self.total_actual_size_x,num=self.num_grid_samples_nx)
        self.grid_positions_y = np.linspace(self.bounds[2],self.bounds[2]+self.total_actual_size_y,num=self.num_grid_samples_ny)
        self.all_grid_positions = np.array(np.meshgrid(self.grid_positions_x, self.grid_positions_y)).T
        self.all_grid_positions = self.all_grid_positions.reshape((self.num_grid_samples_nx*self.num_grid_samples_ny,2))

        self.total_samples_x = int(self.num_grid_samples_nx * self.num_subgrid_samples_nx)
        self.total_samples_y = int(self.num_grid_samples_ny * self.num_subgrid_samples_ny)

        self.center_point_x = self.grid_positions_x[int(self.num_grid_samples_nx / 2)]
        self.center_point_y = self.grid_positions_y[int(self.num_grid_samples_ny / 2)]

        self.xs = np.linspace(self.grid_positions_x[0], self.grid_positions_x[-1] + self.subgrid_rx_positions_x[-1], num=self.total_samples_x)
        self.ys = np.linspace(self.grid_positions_y[0], self.grid_positions_y[-1] + self.subgrid_rx_positions_y[-1], num=self.total_samples_y)

        self.all_grid_and_subgrid_positions = np.array(np.meshgrid(self.xs, self.ys)).T
        self.all_grid_and_subgrid_positions = self.all_grid_and_subgrid_positions.reshape((self.total_samples_x * self.total_samples_y, 2))

        self.x_domain = np.linspace(self.grid_positions_x[0],
                                    self.grid_positions_x[-1] + self.subgrid_rx_positions_x[-1],
                                    num=self.total_samples_x)
        self.y_domain = np.linspace(self.grid_positions_y[0],
                                   self.grid_positions_y[-1]+self.subgrid_rx_positions_y[-1],
                                   num=self.total_samples_y)

        self.total_num_batches = len(self.all_grid_positions)

    def create_probes(self):
        """
        This method sets up the radar platform. It creates an AntennaArray object which represents a blank antenna device.
        The antenna device is manually set up using a combination of direct API calls and the AntennaDevice class.
        The created antenna device is then assigned to the probe_device attribute of the HeatMap instance.

        The AntennaArray is initialized with the following parameters:
            name (str): The name of the antenna array. Here it is set as 'array'.
            waveform (Waveform): The waveform used by the radar platform. It is taken from the HeatMap instance.
            mode_name (str): The mode name of the radar platform. It is taken from the HeatMap instance.
            file_name (str): The file name of the antenna device. Here it is set as 'dipole.ffd'.
            polarization (str): The polarization of the radar platform. It is taken from the HeatMap instance.
            rx_shape (list): The shape of the receiver antenna array. It is a list of two integers representing the number of subgrid samples in the x and y directions.
            tx_shape (int): The shape of the transmitter antenna array. Here it is set as 0.
            spacing_wl_x (int): The spacing in wavelengths in the x direction. It is taken from the HeatMap instance.
            spacing_wl_y (int): The spacing in wavelengths in the y direction. It is taken from the HeatMap instance.
            parent_h_node (HNode): The parent HNode for the antenna device. It is taken from the reference actor of the HeatMap instance.
            load_pattern_as_mesh (bool): Whether to load the antenna pattern as a mesh. It is taken from the HeatMap instance.
            scale_pattern (float): The scale of the antenna pattern. Here it is set as 0.3.
        """
        probe_device = AntennaArray(name='array', waveform=self.waveform,
                                    mode_name=self.mode_name,
                                    file_name='dipole.ffd',
                                    polarization=self.polarization,
                                    rx_shape=[self.num_subgrid_samples_nx,self.num_subgrid_samples_ny],
                                    tx_shape=0,
                                    spacing_wl_x=self.sample_spacing_in_wl,
                                    spacing_wl_y=self.sample_spacing_in_wl,
                                    parent_h_node=self.probe_actor.h_node,
                                    load_pattern_as_mesh=self.show_patterns,
                                    scale_pattern=.3,
                                    all_actors=self.all_actors)
        self.probe_device = probe_device.antenna_device


    #
    def _update_batch(self,idx):
        """
        This method updates the batch index and calculates the start and end indices for the x and y directions.
        It also updates the position of the probe device based on the current batch index.

        Parameters:
            idx (int): The current batch index.

        Attributes updated:
            batch_idx (int): The current batch index.
            idx_x_start (int): The start index for the x direction.
            idx_x_end (int): The end index for the x direction.
            idx_y_start (int): The start index for the y direction.
            idx_y_end (int): The end index for the y direction.
            probe_device.coord_sys.pos (list): The position of the probe device.
        """

        self.batch_idx = idx
        row = idx % self.num_grid_samples_ny
        col = idx // self.num_grid_samples_ny
        idx_x_start = col * self.num_subgrid_samples_nx
        idx_x_end = (col + 1) * self.num_subgrid_samples_nx

        idx_y_start = row * self.num_subgrid_samples_ny
        idx_y_end = (row + 1) * self.num_subgrid_samples_ny

        probe_pos = [self.all_grid_positions[idx][0], self.all_grid_positions[idx][1], self.z_elevation]
        self.probe_device.coord_sys.pos = probe_pos
        self.probe_device.coord_sys.update()

        self.idx_x_start = idx_x_start
        self.idx_x_end = idx_x_end
        self.idx_y_start = idx_y_start
        self.idx_y_end = idx_y_end


    def update_heatmap_3d(self,
                       tx_mode=None,
                       probe_mode=None,
                       function='db',
                       modeler=None,
                       plot_min=-120,
                       plot_max=-70,
                       output_format='point_cloud',
                       freq_idx=0,
                       pulse_idx=0):
        """
        This method updates the 3D heatmap representation of the scene using a radar platform.

        Parameters:
            tx_mode (str): The transmission mode of the radar platform. Defaults to None.
            probe_mode (str): The probe mode of the radar platform. Defaults to None.
            function (str): The function used to calculate the response. Defaults to 'db'.
            modeler (Modeler): The modeler used to visualize the heatmap. Defaults to None.
            plot_min (float): The minimum value for the plot. Defaults to -120.
            plot_max (float): The maximum value for the plot. Defaults to -70.
            output_format (str): The output format of the heatmap. Defaults to 'point_cloud'.

        Attributes updated:
            point_cloud_mesh (PolyData): The point cloud mesh of the heatmap.
        """
        options = {}
        options['clim'] = [plot_min, plot_max]
        options['opacity'] = self.opacity
        options['cmap'] = self.cmap

        pyvistafunc = modeler.pl.add_volume

        all_z_pos = []
        all_images_mag = []
        self.image_3d = []
        for slice_idx, z_slice in enumerate(self.z_slices):
            all_z_pos.append(z_slice.z_elevation)
            print(f"\nUpdating heatmap for z={z_slice.z_elevation}, {slice_idx + 1} of {len(self.z_slices)}\n")
            # each z slice is a 2d heatmap
            # with create_point_cloud=True, this will not generate a surface, just polydata with points and magnitudes
            z_slice.update_heatmap(tx_mode=tx_mode,
                                   probe_mode=probe_mode,
                                   function=function,
                                   plot_min=plot_min,
                                   plot_max=plot_max,
                                   modeler=modeler,
                                   create_surface=False,
                                   sub_grid_updates=False,
                                   add_mesh_to_overlay=False,
                                   freq_idx=freq_idx,
                                   pulse_idx=pulse_idx
                                   )

            if self.show_results_in_modeler:
                self.point_cloud_mesh += z_slice.mesh
            self.image_3d.append(z_slice.image.T) # image data is transposed to match the x,y grid
        self.image_3d = np.array(self.image_3d)
        # if all we want in the end is all the points that make up the 3d heatmap, and don't want to plot, this can
        # save some time.
        if self.show_results_in_modeler:
            xrng = z_slice.xs
            yrng = z_slice.ys
            zrng = np.array(all_z_pos)
            grid = pv.RectilinearGrid(xrng, yrng, zrng)
            data = np.ndarray.flatten(self.image_3d, order='C')
            data = self._apply_function(data, function=function)

            grid['image_mag'] = data
            self.point_cloud_mesh = grid
            output_format = output_format.lower().replace('_','')

            if output_format == 'isosurface' or output_format == 'contour':
                options = {}
                options['clim'] = [plot_min, plot_max]
                options['opacity'] = self.opacity
                options['cmap'] = self.cmap
                mesh = self.point_cloud_mesh.contour(isosurfaces=6, rng=[plot_min, plot_max])
                pyvistafunc = modeler.pl.add_mesh
                pyvistafunc(mesh, **options)
            elif output_format == 'contourclipplane' or output_format == 'contourcutplane':
                options = {}
                options['clim'] = [plot_min, plot_max]
                options['opacity'] = self.opacity
                options['cmap'] = self.cmap
                mesh = self.point_cloud_mesh.contour(isosurfaces=6, rng=[plot_min, plot_max])
                # pyvistafunc = modeler.pl.add_mesh
                # pyvistafunc(mesh, **options)
                modeler.pl.add_mesh_clip_plane(mesh,**options)
            elif output_format == 'slice':
                options['opacity'] = 1.0
                options['cmap'] = self.cmap
                modeler.pl.add_mesh_slice(self.point_cloud_mesh, **options)
            elif output_format == 'cutplane' or output_format == 'clipplane':
                options['opacity'] = 1.0
                options['cmap'] = self.cmap
                modeler.pl.add_mesh_clip_plane(self.point_cloud_mesh, **options)
            else: # pointcloud
                if output_format != 'pointcloud':
                    print(f"Warning: Invalid output format {output_format}, defaulting to point cloud")
                options = {}
                options['opacity'] = 'sigmoid'
                options['mapper'] = 'smart'
                options['clim'] = [plot_min, plot_max]
                options['cmap'] = self.cmap
                pyvistafunc(self.point_cloud_mesh, **options)
            # modeler.pl.add_mesh_clip_plane(self.point_cloud_mesh)
            # modeler.pl.add_mesh_slice(self.point_cloud_mesh)
            modeler.update_frame()  # update pyvista plotter

    def update_heatmap(self,
                       tx_mode=None,
                       probe_mode=None,
                       function='db',
                       modeler=None,
                       create_surface=True,
                       sub_grid_updates=True,
                       plot_min=-100,
                       plot_max=-50,
                       add_mesh_to_overlay=True,
                       td_output_size=None,
                       window='flat',
                       pulse_idx=0,
                       freq_idx=0,
                       include_complex_data = False):
        """
        This method updates the 2D heatmap representation of the scene using a radar platform.

        Parameters:
            tx_mode (str): The transmission mode of the radar platform. Defaults to None.
            probe_mode (str): The probe mode of the radar platform. Defaults to None.
            function (str): The function used to calculate the response. Defaults to 'db'.
            modeler (Modeler): The modeler used to visualize the heatmap. Defaults to None.
            create_surface (bool): Whether to create a surface for the heatmap. Defaults to True.
            sub_grid_updates (bool): Whether to update the subgrid. Defaults to True.
            plot_min (float): The minimum value for the plot. Defaults to -100.
            plot_max (float): The maximum value for the plot. Defaults to -50.
            add_mesh_to_overlay (bool): Whether to add the mesh to the overlay. Defaults to True.

        Attributes updated:
            image (ndarray): The heatmap image.
            mesh (PolyData): The mesh of the heatmap.
            image_min (float): The minimum value of the image.
            image_max (float): The maximum value of the image.
            heatmap_mesh_for_overlay (PolyData): The heatmap mesh for overlay.
        """

        options = {}
        options['cmap'] = self.cmap
        options['opacity'] = self.opacity
        if modeler is not None:
            pyvistafunc = modeler.pl.add_mesh
        else: #can't overlay results if we don't have a modeler
            self.show_results_in_modeler = False
            create_surface = False

        # make this so if multple modes are passed it will work, taking the sum of all modes
        # as the heatmap output
        if not isinstance(tx_mode,list):
            tx_mode= [tx_mode]

        for idx in tqdm(range(self.total_num_batches)):
            # print(modeler.pl.camera_position)
            # modeler.pl.camera_position = [(-6.628138388036804, -13.533762431546673, 4.416518949264247),
            # (4.839783862233162, -4.109425738453865, 1.3774030953645706),
            # (0.14661084374231628, 0.13749220738163118, 0.9795923404184482)]
            self._update_batch(idx)
            pem_api_manager.isOK(pem.computeResponseSync())
            temp_responses = []



            for tx_m in tx_mode:
                if self.probe_device.waveforms[self.mode_name].output.lower() == 'rangedoppler':
                    ouput_format = RssPy.ResponseType.RANGE_DOPPLER
                else:
                    ouput_format = RssPy.ResponseType.FREQ_PULSE
                (ret, temp_response) = pem.retrieveP2PResponse(tx_m,
                                                        probe_mode,
                                                        ouput_format)
                temp_responses.append(temp_response)
            response = np.sum(np.array(temp_responses),axis=0)

            if td_output_size is None:
                # use original size
                # should probaby upsample to closest power of 2 for faster processing, but not going to for now
                td_output_size = response.shape[-1]

            if self.image is None:
                # initialize heatmap image
                self.image = np.random.uniform(low=1e-11,high=1e-10,size=(self.total_samples_x, self.total_samples_y))
            if include_complex_data:
                if self.complex_data is None:
                    self.complex_data = np.zeros((self.total_samples_x, self.total_samples_y),dtype='complex')

            # only need to initialize this if we are doing time domain heatmaps,defined by freq_idx=None
            if self.image_time_domain is None and freq_idx is None:
                # initialize heatmap image
                self.image_time_domain = np.random.uniform(low=1e-11,high=1e-10,size=(self.total_samples_x,
                                                                                      self.total_samples_y,
                                                                                      td_output_size))
            if self.quantity == 'sisocapacity':    
                # channel capacity is in bps/Hz, convert to Mbps
                # channel capacity is calculated for each Rx (1 tx in the subgrid (not for entire array)
                # this is not supported for time domain heatmaps, so we don't need to worry about multi-freq output
                # only a single freq_idx can be used.
                #noise_power = self.waveform.bandwidth*Constants.k_b*Constants.T
                noise_power = self.Kpi_props.noise_power
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    capacity_b_symbol = siso_capacity(channel_power/noise_power)
                    temp_mag[rx_idx] = capacity_b_symbol
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == 'sisocapacitylmmse':
                noise_power = self.Kpi_props.noise_power
                temp_mag = np.zeros((response.shape[1],))
                Np = self.Kpi_props.num_pilots
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    mse = mse_lmmse(channel_power/noise_power,Np)
                    snr = snr_estimation_error(channel_power,noise_power,mse)
                    capacity_b_symbol = siso_capacity(snr)
                    temp_mag[rx_idx] = capacity_b_symbol
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == 'sisocapacityls':
                noise_power = self.Kpi_props.noise_power
                temp_mag = np.zeros((response.shape[1],))
                Np = self.Kpi_props.num_pilots
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    mse = mse_ls(channel_power/noise_power,Np)
                    snr = snr_estimation_error(channel_power,noise_power,mse)
                    capacity_b_symbol = siso_capacity(snr)
                    temp_mag[rx_idx] = capacity_b_symbol
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == 'snr':
                noise_power = self.Kpi_props.noise_power
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    snr = channel_power/noise_power
                    temp_mag[rx_idx] = snr
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == 'snrlmmse':
                noise_power = self.Kpi_props.noise_power
                Np = self.Kpi_props.num_pilots
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    mse = mse_lmmse(channel_power/noise_power,Np)
                    snr = snr_estimation_error(channel_power,noise_power,mse)
                    temp_mag[rx_idx] = snr
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == 'snrls':
                noise_power = self.Kpi_props.noise_power
                Np = self.Kpi_props.num_pilots
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    mse = mse_ls(channel_power/noise_power,Np)
                    snr = snr_estimation_error(channel_power,noise_power,mse)
                    temp_mag[rx_idx] = snr
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == 'outage':
                noise_power = self.Kpi_props.noise_power
                target_rate = self.Kpi_props.target_rate
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    capacity_b_symbol = siso_capacity(channel_power/noise_power)
                    outage = target_rate>capacity_b_symbol
                    temp_mag[rx_idx] = outage
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == 'outagelmmse':
                noise_power = self.Kpi_props.noise_power
                target_rate = self.Kpi_props.target_rate
                Np = self.Kpi_props.num_pilots
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    mse = mse_lmmse(channel_power/noise_power,Np)
                    snr = snr_estimation_error(channel_power,noise_power,mse)
                    capacity_b_symbol = siso_capacity(snr)
                    outage = target_rate>capacity_b_symbol
                    temp_mag[rx_idx] = outage
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == 'outagels':
                noise_power = self.Kpi_props.noise_power
                target_rate = self.Kpi_props.target_rate
                Np = self.Kpi_props.num_pilots
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    mse = mse_ls(channel_power/noise_power,Np)
                    snr = snr_estimation_error(channel_power,noise_power,mse)
                    capacity_b_symbol = siso_capacity(snr)
                    outage = target_rate>capacity_b_symbol
                    temp_mag[rx_idx] = outage
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == "mselmmse":
                noise_power = self.Kpi_props.noise_power
                Np = self.Kpi_props.num_pilots
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    mse = mse_lmmse(channel_power/noise_power,Np)
                    temp_mag[rx_idx] = mse
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            elif self.quantity == "msels":
                noise_power = self.Kpi_props.noise_power
                Np = self.Kpi_props.num_pilots
                temp_mag = np.zeros((response.shape[1],))
                for rx_idx in range(response.shape[1]):
                    channel_power = abs(response[:, rx_idx, pulse_idx, freq_idx])**2
                    mse = mse_ls(channel_power/noise_power,Np)
                    temp_mag[rx_idx] = mse
                temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
            else:
                if freq_idx is None:
                    win_range, _ = self._window_function(function=window, size=response.shape[-1])
                    sf_upsample = td_output_size / response.shape[-1] # scale factor for up/down sampling
                    tdr_temp = np.zeros((response.shape[1],td_output_size)) # for every rx we will populate in probe array
                    for rx_idx in range(response.shape[1]):
                        windowed_data = win_range*response[0, rx_idx, pulse_idx, :]
                        fft_data = sf_upsample * np.fft.ifft(windowed_data, n=td_output_size)
                        tdr_temp[rx_idx] = self._apply_function(np.fmax(np.abs(fft_data), 1e-30))
                    # also update the self.image with the first frequency
                    temp_mag = np.fmax(np.abs(response[0, :, pulse_idx, 0]), 1e-30)  # [tx][all_rx][one chirp][one freq]
                    temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
                    if include_complex_data:
                        temp_cmplx = np.array(response[0, :, pulse_idx, 0]).reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
                else: # single freq index is used
                    if include_complex_data:
                        temp_cmplx = np.array(response[0, :, pulse_idx, freq_idx]).reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))
                    temp_mag = np.fmax(np.abs(response[0, :, pulse_idx, freq_idx]), 1e-30)  # [tx][all_rx][one chirp][one freq]
                    temp_mag = temp_mag.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny))

            # self.image does not change the function of the data yet (still real/imagniary)
            self.image[self.idx_x_start:self.idx_x_end, self.idx_y_start:self.idx_y_end] = temp_mag
            if include_complex_data:
                self.complex_data[self.idx_x_start:self.idx_x_end, self.idx_y_start:self.idx_y_end] = temp_cmplx
            # special case, image_time_domain is only availble for freq_idx=None
            if freq_idx is None:
                tdr_temp = tdr_temp.reshape((self.num_subgrid_samples_nx, self.num_subgrid_samples_ny, td_output_size))
                self.image_time_domain[self.idx_x_start:self.idx_x_end, self.idx_y_start:self.idx_y_end,:] = tdr_temp
                # no need to do anymore, we have all the data we need, time domain heatmaps are calculated for all time steps
                # and we don't need to plot anything in this step


            if self.show_results_in_modeler:
                # calculate response in dB to overlay in pyvista plotter

                self.mesh = create_mesh_from_image2(self.image,
                                                    all_positions=self.all_grid_and_subgrid_positions,
                                                    z_offset=self.z_elevation,
                                                    function=function,
                                                    create_surface=create_surface)

                image_with_function = self._apply_function(self.image, function=function)
                im_min = image_with_function.min()
                im_max = image_with_function.max()

                # imshow (pyvista doesn't (?) have origin arugments) expects the origin to be at the top left,
                # so we flip the image so origin in lower left. And transpose the axes so that the x and y are correct
                # because imshow uses M,N indexing where M is vertical and N is horizontal. In our original data, X and Y
                # are the horizontal and vertical axes, respectively.
                if hasattr(modeler,'mpl_ax_handle'):
                    modeler.mpl_ax_handle.set_cmap(self.cmap)
                    modeler.mpl_ax_handle.set_data(image_with_function.T)  # update pyvista matplotlib plot


                    # print(f"Heatmap Cutplane: Min: {im_min_db}dB, Max: {im_max_db}dB")
                    modeler.mpl_ax_handle.set_clim(vmin=plot_min, vmax=plot_max)
                    modeler.ax.set_title(f'Heatmap:Z:{self.z_elevation:.2f}m\n Min: {im_min:.2f}, Max: {im_max:.2f} {function}')

                if add_mesh_to_overlay:
                    if self.heatmap_mesh_for_overlay is None:
                        pyvistafunc(self.mesh, **options)
                        self.heatmap_mesh_for_overlay = self.mesh
                    else:
                        self.heatmap_mesh_for_overlay.copy_from(self.mesh)

                    if sub_grid_updates:
                        modeler.update_frame() # update pyvista plotter
                    # update scalar bar for current range of values
                    # 'image_mag' is default name of color bar associated with heatmaps
                    modeler.pl.update_scalar_bar_range([plot_min,plot_max],'image_mag')
                    

        if self.show_results_in_modeler:
            modeler.update_frame()


    def update_heatmap_time_domain(self,
                                   tx_mode=None,
                                   probe_mode=None,
                                   function='db',
                                   modeler=None,
                                   create_surface=True,
                                   plot_min=-100,
                                   plot_max=-50,
                                   add_mesh_to_overlay=True,
                                   td_output_size=None,
                                   window='flat',
                                   pulse_idx=0,
                                   start_animation_after_time=None,
                                   end_animation_after_time=None,
                                   use_slider_widget=False,
                                   loop_animation=False):

        if self.quantity == 'channelcapacity':
            print("WARNING: Channel capacity is not supported for time domain heatmaps")
            print("Using S-Parameters")

        if not isinstance(tx_mode,list):
            tx_mode= [tx_mode]
        self.waveform.get_response_domains(tx_mode[0]) # if muliple modes, they should all have same domains
        # this is calculated for round trip, so we multiply by 2 to get 1 way
        self.range_domain = self.waveform.rng_domain * 2
        self.fast_time_domain = self.waveform.fast_time_domain * 2
        print(f"Range domain max: {self.range_domain[-1]}")
        print(f"Time domain max: {self.fast_time_domain[-1]}")

        # after this is called, we will have all the data we need to create the time domain heatmaps.
        # this will create a new parameter image_time_domain,which will be a 3D array of the x,y,time domain data
        # ToDo add upsampling and windowing to the data
        self.update_heatmap(tx_mode=tx_mode,
                            probe_mode=probe_mode,
                            function=function,
                            modeler=modeler,
                            create_surface=False,
                            sub_grid_updates=False,
                            add_mesh_to_overlay=False,
                            td_output_size=td_output_size,
                            window=window,
                            pulse_idx=pulse_idx,
                            freq_idx=None)

        # time stamps are same lenght as frequncy domain, so we can use the same time stamps for all time steps
        time_stamps = np.linspace(0,self.fast_time_domain[-1],num=self.image_time_domain.shape[-1])

        # truncate time if users specifies

        if end_animation_after_time is not None:
            # find the closest time to user asked time
            idx = (np.abs(time_stamps - end_animation_after_time)).argmin()
            if idx<len(time_stamps):
                time_stamps = time_stamps[:idx]
        if start_animation_after_time is not None:
            if start_animation_after_time >= end_animation_after_time:
                print('Start animation time must be less than end animation time')
                start_animation_after_time=0
            idx = (np.abs(time_stamps - start_animation_after_time)).argmin()
            if idx<len(time_stamps):
                time_stamps = time_stamps[idx:]
        if add_mesh_to_overlay:
            options = {}
            options['opacity'] = self.opacity
            options['cmap'] = self.cmap
            pyvistafunc = modeler.pl.add_mesh

            if use_slider_widget:
                all_meshes = []
                for time_idx, time in tqdm(enumerate(time_stamps)):
                    # calculate response in to overlay in pyvista plotter
                    # function has already been applied to the data, so we just need to create the mesh
                    mesh = create_mesh_from_image2(self.image_time_domain[:,:,time_idx],
                                                        all_positions=self.all_grid_and_subgrid_positions,
                                                        z_offset=self.z_elevation,
                                                        function='real',
                                                        create_surface=create_surface)
                    all_meshes.append(mesh)

                def get_mesh_time_step(value):
                    time_idx = int((np.abs(time_stamps - value)).argmin())
                    mesh = all_meshes[time_idx]
                    if self.heatmap_mesh_for_overlay is None:
                        pyvistafunc(mesh, **options)
                        self.heatmap_mesh_for_overlay = mesh
                    else:
                        self.heatmap_mesh_for_overlay.copy_from(mesh)
                    # pyvistafunc(all_meshes[time_idx], **options)
                    modeler.pl.update_scalar_bar_range([plot_min, plot_max], 'image_mag')
                    if hasattr(modeler,'mpl_ax_handle'):
                        modeler.mpl_ax_handle.set_cmap(self.cmap)
                        modeler.mpl_ax_handle.set_data(self.image_time_domain[:, :, time_idx])
                        modeler.mpl_ax_handle.set_clim(vmin=plot_min, vmax=plot_max)
                    return
                modeler.pl.add_slider_widget(get_mesh_time_step, [0, time_stamps[-1]], title='adsf')
                modeler.update_frame(write_frame=False)

            else:
                time = 0
                time_idx = 0
                num_loops = 0
                while time<=time_stamps[-1]:
                    if loop_animation:
                        if time_idx>=len(time_stamps):
                            time_idx=0
                            num_loops+=1
                        time = time_stamps[time_idx]
                    else:
                        if time_idx>=len(time_stamps):
                            break
                        time = time_stamps[time_idx]

                    # for time_idx, time in enumerate(time_stamps):
                    # calculate response in to overlay in pyvista plotter
                    # function has already been applied to the data, so we just need to create the mesh
                    self.mesh = create_mesh_from_image2(self.image_time_domain[:,:,time_idx],
                                                        all_positions=self.all_grid_and_subgrid_positions,
                                                        z_offset=self.z_elevation,
                                                        function='real',
                                                        create_surface=create_surface)


                    im_min = self.image_time_domain[:,:,time_idx].min()
                    im_max = self.image_time_domain[:,:,time_idx].max()

                    # imshow (pyvista doesn't (?) have origin arugments) expects the origin to be at the top left,
                    # so we flip the image so origin in lower left. And transpose the axes so that the x and y are correct
                    # because imshow uses M,N indexing where M is vertical and N is horizontal. In our original data, X and Y
                    # are the horizontal and vertical axes, respectively.
                    if hasattr(modeler, 'mpl_ax_handle'):
                        modeler.mpl_ax_handle.set_cmap(self.cmap)
                        modeler.mpl_ax_handle.set_data(self.image_time_domain[:,:,time_idx].T)  # update pyvista matplotlib plot

                        # print(f"Heatmap Cutplane: Min: {im_min_db}dB, Max: {im_max_db}dB")
                        modeler.mpl_ax_handle.set_clim(vmin=plot_min, vmax=plot_max)
                        time_us = time * 1e6
                        modeler.ax.set_title(f'Heatmap:Z:{self.z_elevation:.2f}m\n Time: {time_us:.2f}us, Min: {im_min:.2f}, Max: {im_max:.2f} {function}')

                    if add_mesh_to_overlay:
                        if self.heatmap_mesh_for_overlay is None:
                            pyvistafunc(self.mesh, **options)
                            self.heatmap_mesh_for_overlay = self.mesh
                        else:
                            self.heatmap_mesh_for_overlay.copy_from(self.mesh)
                        # update scalar bar for current range of values
                        # 'image_mag' is default name of color bar associated with heatmaps
                        modeler.pl.update_scalar_bar_range([plot_min, plot_max], 'image_mag')
                        if loop_animation:
                            # ToDo, this prevents the video to be infnietly long during loop animation, but
                            # causes a bad crash when modeler window is closed
                            if num_loops>=1:
                                modeler.pl.update(force_redraw=False)
                            else:
                                modeler.update_frame(write_frame=True)
                        else:
                            modeler.update_frame()
                    time_idx += 1

    def update_heatmap_time_domain_3d(self,
                                       tx_mode=None,
                                       probe_mode=None,
                                       function='db',
                                       modeler=None,
                                       create_surface=True,
                                       plot_min=-100,
                                       plot_max=-50,
                                       add_mesh_to_overlay=True,
                                       td_output_size=None,
                                       window='flat',
                                       pulse_idx=0,
                                       start_animation_after_time = None,
                                       end_animation_after_time=None,
                                       use_slider_widget=False,
                                       loop_animation=False,
                                       numpy_data_path=None,
                                       ):

        self.max_at_idx = []
        if self.quantity == 'channelcapacity':
            print("WARNING: Channel capacity is not supported for time domain heatmaps")
            print("Using S-Parameters")
            self.quantity = 's-parameters'
        if not isinstance(tx_mode,list):
            tx_mode= [tx_mode]
        self.waveform.get_response_domains(tx_mode[0]) # if muliple modes, they should all have same domains
        # this is calculated for round trip, so we multiply by 2 to get 1 way
        self.range_domain = self.waveform.rng_domain * 2
        self.fast_time_domain = self.waveform.fast_time_domain * 2
        print(f"Range domain max: {self.range_domain[-1]}")
        print(f"Time domain max: {self.fast_time_domain[-1]}")



        # after this is called, we will have all the data we need to create the time domain heatmaps.
        # this will create a new parameter image_time_domain,which will be a 3D array of the x,y,time domain data
        # ToDo add upsampling and windowing to the data
        all_z_pos = []
        self.image_time_domain_3D = []

        compute_heatmap = True
        if numpy_data_path is not None:
            if os.path.exists(numpy_data_path):
                print(f"Loading precomputed heatmap data from {numpy_data_path}")
                self.image_time_domain_3D = np.load(numpy_data_path)
                print(f"Loaded data has shape {self.image_time_domain_3D.shape}")
                compute_heatmap=False
            else:
                print(f"Precomputed data not found: {numpy_data_path}, recomputing heatmap data and saving at this location")
                compute_heatmap=True

        
        for slice_idx, z_slice in enumerate(self.z_slices):
            all_z_pos.append(z_slice.z_elevation)
            if compute_heatmap:
                print(f"\nUpdating heatmap for z={z_slice.z_elevation}, {slice_idx + 1} of {len(self.z_slices)}\n")

                z_slice.update_heatmap(tx_mode=tx_mode,
                                    probe_mode=probe_mode,
                                    function=function,
                                    modeler=modeler,
                                    create_surface=False,
                                    sub_grid_updates=False,
                                    add_mesh_to_overlay=False,
                                    td_output_size=td_output_size,
                                    window=window,
                                    pulse_idx=pulse_idx,
                                    freq_idx=None)
                self.image_time_domain_3D.append(z_slice.image_time_domain.swapaxes(0,1))  # image data is transposed to match the x,y grid

        if compute_heatmap:    
            self.image_time_domain_3D = np.array(self.image_time_domain_3D)
        # save image data for numpy array for future processing
        if numpy_data_path is not None:
            np.save(numpy_data_path,self.image_time_domain_3D)

        # adjust time stamps if td_output_size is different from original size.
        time_stamps = np.linspace(0,self.fast_time_domain[-1],num=self.image_time_domain_3D.shape[-1])
        end_time = time_stamps[-1]
        # ToDo, add something to handle aliasing
        # truncate time if users specifies
        if end_animation_after_time is not None:
            # find the closest time to user asked time
            idx = (np.abs(time_stamps - end_animation_after_time)).argmin()
            if idx<len(time_stamps):
                time_stamps = time_stamps[:idx]
                self.image_time_domain_3D = self.image_time_domain_3D[:, :, :, :idx]
        if start_animation_after_time is not None:
            if start_animation_after_time >= end_animation_after_time:
                print('Start animation time must be less than end animation time')
                start_animation_after_time=0
            idx = (np.abs(time_stamps - start_animation_after_time)).argmin()
            if idx<len(time_stamps):
                time_stamps = time_stamps[idx:]
                self.image_time_domain_3D = self.image_time_domain_3D[:, :, :, idx:]
        if add_mesh_to_overlay:
            options = {}
            options['opacity'] = self.opacity
            options['cmap'] = self.cmap
            pyvistafunc = modeler.pl.add_mesh

            z_slice = self.z_slices[0]
            xrng = z_slice.xs
            yrng = z_slice.ys
            zrng = np.array(all_z_pos)
            grid = pv.RectilinearGrid(xrng, yrng, zrng)

            if len(all_z_pos) > 1:
                z_idx = 1
            else:
                z_idx = 0

            # you can have a slider bar for interactive stepping through time, or just create a video
            if use_slider_widget:
                all_meshes = []
                for time_idx, time in tqdm(enumerate(time_stamps)):
                    data = np.ndarray.flatten(self.image_time_domain_3D[:,:,:,time_idx], order='C')
                    grid['image_mag'] = data
                    self.point_cloud_mesh = grid
                    data_min = data.min()
                    data_max = data.max()
                    if plot_min > data_max:
                        print(f"Warning: plot_min {plot_min} is greater than max data{data_max}, resetting to data min")
                        plot_min = data_min
                    if plot_max < data_min:
                        print(f"Warning: plot_max {plot_max} is less than min data{data_min}, resetting to data max")
                        plot_max = data_max

                    all_meshes.append(self.point_cloud_mesh.contour(isosurfaces=9, rng=[plot_min, plot_max]))

                def get_mesh_time_step(value):
                    time_idx = int((np.abs(time_stamps - value)).argmin())
                    mesh = all_meshes[time_idx]
                    if self.heatmap_mesh_for_overlay is None:
                        pyvistafunc(mesh, **options)
                        self.heatmap_mesh_for_overlay = mesh
                    else:
                        self.heatmap_mesh_for_overlay.copy_from(mesh)
                    # pyvistafunc(all_meshes[time_idx], **options)
                    modeler.pl.update_scalar_bar_range([plot_min, plot_max], 'image_mag')
                    if hasattr(modeler, 'mpl_ax_handle'):
                        modeler.mpl_ax_handle.set_cmap(self.cmap)
                        modeler.mpl_ax_handle.set_data(self.image_time_domain_3D[z_idx, :, :, time_idx])
                        modeler.mpl_ax_handle.set_clim(vmin=plot_min, vmax=plot_max)
                    return
                modeler.pl.add_slider_widget(get_mesh_time_step, [0, time_stamps[-1]], title='adsf')
                modeler.update_frame(write_frame=False)
            else:
                time = 0
                time_idx = 0
                num_loops = 0
                while time <= time_stamps[-1]:
                    if loop_animation:
                        if time_idx >= len(time_stamps):
                            time_idx = 0
                            num_loops += 1
                        time = time_stamps[time_idx]
                    else:
                        if time_idx>=len(time_stamps):
                            break
                        time = time_stamps[time_idx]
                    data = np.ndarray.flatten(self.image_time_domain_3D[:,:,:,time_idx], order='C')
                    grid['image_mag'] = data
                    self.point_cloud_mesh = grid
                    # output_format = output_format.lower().replace('_', '')

                    data_min = data.min()
                    data_max = data.max()
                    if plot_min > data_max:
                        print(f"Warning: plot_min {plot_min} is greater than max data{data_max}, resetting to data min")
                        plot_min = data_min
                    if plot_max < data_min:
                        print(f"Warning: plot_max {plot_max} is less than min data{data_min}, resetting to data max")
                        plot_max = data_max

                    options = {}
                    options['clim'] = [plot_min, plot_max]
                    options['opacity'] = self.opacity
                    options['cmap'] = self.cmap
                    self.mesh = self.point_cloud_mesh.contour(isosurfaces=8, rng=[plot_min, plot_max])
                    # pyvistafunc = modeler.pl.add_mesh
                    # pyvistafunc(mesh, **options)

                    im_min = self.image_time_domain_3D[:,:,:,time_idx].min()
                    im_max = self.image_time_domain_3D[:,:,:,time_idx].max()

                    # imshow (pyvista doesn't (?) have origin arugments) expects the origin to be at the top left,
                    # so we flip the image so origin in lower left. And transpose the axes so that the x and y are correct
                    # because imshow uses M,N indexing where M is vertical and N is horizontal. In our original data, X and Y
                    # are the horizontal and vertical axes, respectively.
                    if hasattr(modeler,'mpl_ax_handle'):
                        modeler.mpl_ax_handle.set_cmap(self.cmap)
                        modeler.mpl_ax_handle.set_data(self.image_time_domain_3D[z_idx,:,:,time_idx])  # update pyvista matplotlib plot

                        # print(f"Heatmap Cutplane: Min: {im_min_db}dB, Max: {im_max_db}dB")
                        modeler.mpl_ax_handle.set_clim(vmin=plot_min, vmax=plot_max)
                        time_us = time * 1e6
                        z_val = all_z_pos[z_idx]
                        modeler.ax.set_title(f'Heatmap:Z:{z_val:.2f}m\n Time: {time_us:.2f}us, Min: {im_min:.2f}, Max: {im_max:.2f} {function}')
                    if add_mesh_to_overlay:
                        if self.heatmap_mesh_for_overlay is None:
                            pyvistafunc(self.mesh, **options)
                            self.heatmap_mesh_for_overlay = self.mesh
                        else:
                            self.heatmap_mesh_for_overlay.copy_from(self.mesh)
                        # update scalar bar for current range of values
                        # 'image_mag' is default name of color bar associated with heatmaps
                        modeler.pl.update_scalar_bar_range([plot_min, plot_max], 'image_mag')

                        if loop_animation:
                            # ToDo, this prevents the video to be infnietly long during loop animation, but
                            # causes a bad crash when modeler window is closed
                            if num_loops>=1:
                                modeler.pl.update()
                            else:
                                modeler.update_frame(write_frame=True)
                        else:
                            modeler.update_frame()
                    time_idx += 1

    def _apply_function(self,data, function='db'):
        function = function.lower().replace('_','')

        if function.lower() == 'db':
            data = 20 * np.log10(np.fmax(np.abs(data), 1.e-30))
        elif function.lower() == 'db10':
            data = 10 * np.log10(np.fmax(np.abs(data), 1.e-30))
        elif function.lower() == 'db20':
            data = 20 * np.log10(np.fmax(np.abs(data), 1.e-30))
        elif function.lower() == 'real':
            data = np.fmax(np.real(data), 1.e-30)
        elif function.lower() == 'imag':
            data = np.fmax(np.imag(data), 1.e-30)
        else:
            data = np.fmax(np.abs(data), 1.e-30)

        return data

    def _window_function(self,function='flat', size=512, dbDown=30):
        if function.lower() == 'hann':
            win = np.hanning(size)
        elif function.lower() == 'hamming':
            win = np.hamming(size)
        elif function.lower() == 'blackman':
            win = np.blackman(size)
        elif function.lower() == 'bartlett':
            win = np.bartlett(size)
        elif function.lower() == 'kaiser':
            win = np.kaiser(size, dbDown)
        elif function.lower() == 'flat':
            win = np.ones(size)
        else:
            print('Warning: Invalid window function, defaulting to flat')
            win = np.ones(size)
        win_sum = np.sum(win)
        win *= size/win_sum
        return win, win_sum
