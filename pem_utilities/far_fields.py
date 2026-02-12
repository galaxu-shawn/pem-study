"""
Far-Field Antenna Pattern Analysis Module

This module provides comprehensive tools for reading, processing, and analyzing far-field antenna patterns.
It supports various file formats including ASCII (.ffd), binary (.ffld), and NetCDF (.bffd) formats.
The module enables antenna array pattern combination, beam steering calculations, and 3D visualization
of far-field radiation patterns.

Key Features:
- Reading far-field data from multiple file formats
- Combining multiple antenna element patterns with arbitrary weights and phases
- Beam steering and array factor calculations
- 3D mesh generation for pattern visualization
- Export capabilities to various formats (GLTF, CSV, binary)

Classes:
    FarFields: Main class for handling far-field antenna pattern data

Functions:
    summing: Interpolate 3D pattern from vertical and horizontal slices
    convert_ffd_to_gltf: Convert far-field mesh to GLTF format
    combine_farfields: Combine multiple antenna patterns with weights
    get_beam_scan_weights: Calculate beam steering weights for antenna arrays
    calc_relative_phase: Calculate relative phase for beam steering
    calculate_most_center_location: Find centroid and closest point in array
    find_center_and_maxmin_array_pos: Find center position and extents of antenna array
"""

import os
import time as walltime
import numpy as np
import pyvista as pv
from netCDF4 import Dataset
import struct
from pem_utilities.FarFieldData import FF_Data

def summing(
    vertical_slice,
    horizonal_slice):
    """
    Interpolate a full 3-D pattern from a vertical and horizontal slice using the summing algorithm.
    
    This function creates a 3D radiation pattern by combining vertical and horizontal pattern slices
    using a simple averaging approach. This is useful when full 3D pattern data is not available
    but principal plane cuts are known.
    
    Parameters
    ----------
    vertical_slice : numpy.ndarray
        Vertical plane pattern data (theta variation)
    horizonal_slice : numpy.ndarray
        Horizontal plane pattern data (phi variation)

    Returns
    -------
    numpy.ndarray
        Interpolated 3D pattern as a 2D array (theta x phi)
    """
    theta_resolution = (vertical_slice.size // 2) + 1
    return 0.5 * (
        vertical_slice[:theta_resolution, np.newaxis] + horizonal_slice[np.newaxis, :]
    )

def convert_ffd_to_gltf(ff_mesh, output_path, cmap='jet', show_plot=False):
    """
    Convert a far-field mesh to GLTF format for 3D visualization.
    
    This function exports a PyVista mesh containing far-field pattern data to the
    GLTF format, which can be used in web browsers and other 3D visualization tools.
    
    Parameters
    ----------
    ff_mesh : pyvista.StructuredGrid
        Far-field mesh containing the 3D radiation pattern
    output_path : str
        Full path where the GLTF file should be saved
    cmap : str, optional
        Colormap to use for visualization (default: 'jet')
    show_plot : bool, optional
        Whether to display the plot before export (default: False)
    """
    pl = pv.Plotter()
    pl.add_mesh(ff_mesh, cmap=cmap)
    pl.export_gltf(output_path)  # export to gltf
    if show_plot:
        pl.show()
    pl.close()

def combine_farfields(fields, source_weights, freq_idx=0):
    """
    Combine multiple far-field patterns using complex weights and array geometry.
    
    This function performs array pattern synthesis by combining individual element
    patterns with specified amplitude and phase weights. It accounts for element
    positions to calculate proper array factors and supports multi-frequency analysis.
    
    Parameters
    ----------
    fields : FarFields
        FarFields object containing individual element patterns
    source_weights : dict
        Dictionary with port names as keys and weight dictionaries as values.
        Each weight dict should contain:
        - 'mag': amplitude weight (linear scale)
        - 'phase': phase weight in degrees
        - 'position': optional [x,y,z] position in meters
    freq_idx : int or str, optional
        Frequency index to process. Use 'All' for all frequencies (default: 0)
    
    Returns
    -------
    FF_Data
        Combined far-field pattern data object
        
    Notes
    -----
    The array factor is calculated using the formula:
    AF = sum(w_n * exp(j*k*(x_n*sin(θ)cos(φ) + y_n*sin(θ)sin(φ))))

    where w_n is the complex weight and (x_n, y_n) is the element position.
    """
    start_time = walltime.time()
    data = FF_Data()
    all_port_names = fields.all_port_names
    num_ports = len(all_port_names)

    # Determine frequency range to process
    if str(freq_idx)=='All':
        num_freq = fields.data_dict[all_port_names[0]].Num_Freq
        freqs = fields.data_dict[all_port_names[0]].Frequencies
    else:
        num_freq = 1
        freqs = [fields.data_dict[all_port_names[0]].Frequencies[freq_idx]]

    # Get angular grids from first port (assuming all ports have same grid)
    phi = fields.data_dict[all_port_names[0]].Phi
    theta = fields.data_dict[all_port_names[0]].Theta

    # Initialize combined field arrays
    rEtheta_total = np.zeros((num_freq, fields.data_dict[all_port_names[0]].Total_Samples), dtype=complex)
    rEphi_total = np.zeros((num_freq, fields.data_dict[all_port_names[0]].Total_Samples), dtype=complex)

    # Process each frequency
    for idx, freq in enumerate(freqs):
        # Create wave vector grids for array factor calculation
        phi_grid, theta_grid = np.meshgrid(np.deg2rad(phi), np.deg2rad(theta))
        ko = 2 * np.pi * freq / 3e8  # Free space wave number
        kx_grid = ko * np.sin(theta_grid) * np.cos(phi_grid)
        ky_grid = ko * np.sin(theta_grid) * np.sin(phi_grid)

        # Flatten grids for vectorized computation
        kx_flat = kx_grid.ravel()
        ky_flat = ky_grid.ravel()

        weights = np.zeros(num_ports, dtype=complex)
        incident_power = 0

        # Process each antenna element/port
        for n, port in enumerate(all_port_names):
            if port in source_weights.keys():
                # Extract amplitude and phase weights
                w_mag = source_weights[port]['mag']
                incident_power += w_mag
                w_phase = np.deg2rad(source_weights[port]['phase'])
                weights[n] = np.sqrt(w_mag) * np.exp(1j * w_phase)

                # Get element position for array factor calculation
                if 'position' in source_weights[port]:
                    xyz_pos = source_weights[port]['position']
                elif hasattr(fields.data_dict[port], 'Position'):
                    xyz_pos = fields.data_dict[port].Position
                else:
                    xyz_pos = np.zeros(3)
                    print('No position information found for port, assuming 0,0,0')
            else:
                # Port not excited
                weights[n] = np.sqrt(0) * np.exp(1j * 0)
                xyz_pos = np.zeros(3)

            # Calculate array factor for this element
            array_factor = np.exp(1j * (xyz_pos[0] * kx_flat + xyz_pos[1] * ky_flat)) * weights[n]
            
            # Select appropriate frequency index
            if str(freq_idx) == 'All':
                current_idx = idx
            else:
                current_idx = freq_idx
                
            # Add weighted contribution to total field
            rEtheta_total[idx] += array_factor * fields.data_dict[port].rETheta[current_idx]
            rEphi_total[idx] += array_factor * fields.data_dict[port].rEPhi[current_idx]

    # Populate output data structure
    data.rETheta = rEtheta_total
    data.rEPhi = rEphi_total
    data.Theta = fields.data_dict[all_port_names[0]].Theta
    data.Phi = fields.data_dict[all_port_names[0]].Phi
    data.Frequencies = fields.data_dict[all_port_names[0]].Frequencies
    data.Delta_Theta = fields.data_dict[all_port_names[0]].Delta_Theta
    data.Delta_Phi = fields.data_dict[all_port_names[0]].Delta_Phi
    data.Diff_Area = fields.data_dict[all_port_names[0]].Diff_Area
    data.Total_Samples = len(data.rETheta[0])
    data.Num_Freq = len(data.Frequencies)
    data.Lattice_Vector = fields.data_dict[all_port_names[0]].Lattice_Vector
    data.Lattice_Vector = fields.data_dict[all_port_names[0]].Lattice_Vector
    data.Is_Component_Array = fields.data_dict[all_port_names[0]].Is_Component_Array
    data.Incident_Power = incident_power

    stop_time = walltime.time()
    elapsed_time = stop_time-start_time
    #print(f'combined Fields: {elapsed_time}')
    return data


def get_beam_scan_weights(pos_dict,
                          theta=0,
                          phi=0,
                          freq=1e9,
                          center_pos=[0,0,0],
                          max_dist=0,
                          tapering='flat',
                          cosine_power=1,
                          edge_taper=-200):
    """
    Calculate beam steering weights for an antenna array.
    
    This function computes the complex weights (amplitude and phase) needed to steer
    an antenna array beam to a specific direction. It supports various tapering
    functions to control sidelobe levels.
    
    Parameters
    ----------
    pos_dict : dict
        Dictionary mapping port names to their 3D positions [x,y,z] in meters
    theta : float, optional
        Beam steering angle in elevation (degrees from z-axis, default: 0)
    phi : float, optional  
        Beam steering angle in azimuth (degrees from x-axis, default: 0)
    freq : float, optional
        Operating frequency in Hz (default: 1e9)
    center_pos : list, optional
        Reference center position [x,y,z] in meters (default: [0,0,0])
    max_dist : float or list, optional
        Maximum distance for tapering calculation. If scalar, same for x,y.
        If list, [max_x, max_y] (default: 0)
    tapering : str, optional
        Tapering function: 'flat', 'cosine', 'triangular', 'hamming' (default: 'flat')
    cosine_power : float, optional
        Power for cosine tapering (default: 1)
    edge_taper : float, optional
        Edge taper level in dB (default: -200, essentially zero)
        
    Returns
    -------
    dict
        Dictionary with port names as keys and weight dictionaries as values.
        Each weight dict contains:
        - 'mag': amplitude weight
        - 'phase': phase weight in degrees  
        - 'position': element position
        
    Notes
    -----
    The phase weights implement progressive phase shift for beam steering:
    φ_n = k * (x_n*sin(θ)cos(φ) + y_n*sin(θ)sin(φ) + z_n*cos(θ))
    """
    source_weights = {}

    # Convert tapering parameters
    cosinePow = cosine_power
    edgeTaper_dB = edge_taper
    edgeTaper = 10 ** ((float(edgeTaper_dB)) / 20)  # Convert dB to linear
    threshold = 1e-10
    w1 = 1
    w2 = 1
    max_length_in_dir1 = max_length_in_dir2 = max_dist

    for port in pos_dict:
        temp_dict = {}
        pos = pos_dict[port]
        
        # Calculate distance from center for tapering
        dist_vector = np.abs(pos - np.array(center_pos))
        distx = dist_vector[0]
        disty = dist_vector[1]
        
        # Apply selected tapering function
        if tapering.lower() == 'cosine':
            if max_dist[0] < threshold:
                w1 = 1
            else:
                w1 = (1 - edgeTaper) * (np.cos(np.pi * distx / max_dist[0])) ** cosinePow + edgeTaper
            if max_dist[1] < threshold:
                w2 = 1
            else:
                w2 = (1 - edgeTaper) * (np.cos(np.pi * disty / max_dist[1])) ** cosinePow + edgeTaper
                
        elif tapering.lower() == 'triangular':
            if max_dist[0] < threshold:
                w1 = 1
            else:
                w1 = (1-edgeTaper)*(1-(np.fabs(distx)/(max_dist[0]/2))) + edgeTaper
            if max_dist[1] < threshold:
                w2 = 1
            else:
                w2 = (1-edgeTaper)*(1-(np.fabs(disty)/(max_dist[1]/2))) + edgeTaper
                
        elif tapering.lower() == 'hamming':
            if max_dist[0] < threshold:
                w1 = 1
            else:
                w1 = 0.54 - 0.46 * np.cos(2*np.pi*(distx/max_dist[0]-0.5))
            if max_dist[1] < threshold:
                w2 = 1
            else:
                w2 = 0.54 - 0.46 * np.cos(2*np.pi*(disty/max_dist[1]-0.5))

        # Calculate beam steering phase
        phase = calc_relative_phase(pos, freq, theta, phi)
        
        # Store amplitude and phase weights
        temp_dict['mag'] = np.abs(w1*w2)
        temp_dict['phase'] = np.rad2deg(phase)
        temp_dict['position'] = pos
        source_weights[port] = temp_dict

    return source_weights


def calc_relative_phase(pos, freq, theta, phi):
    """
    Calculate the relative phase shift for beam steering.
    
    This function computes the phase shift required at each antenna element
    to steer the beam to a specified direction (theta, phi).
    
    Parameters
    ----------
    pos : array_like
        Element position [x, y, z] in meters
    freq : float
        Operating frequency in Hz
    theta : float
        Elevation angle in degrees (0° = zenith, 90° = horizon)
    phi : float
        Azimuth angle in degrees (0° = +x axis)
        
    Returns
    -------
    float
        Phase shift in radians
        
    Notes
    -----
    The phase calculation uses the standard array theory formula:
    φ = k₀ * r⃗ · k̂
    where k₀ is the wave number and k̂ is the unit vector in the beam direction.
    """
    wavelength = 3e8 / freq
    phaseConstant = (2 * np.pi / wavelength)

    # Convert angles to radians for calculation
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)

    # Calculate wave vector components
    xVector = -pos[0] * np.sin(theta_rad) * np.cos(phi_rad)
    yVector = -pos[1] * np.sin(theta_rad) * np.sin(phi_rad)
    zVector = -pos[2] * np.cos(theta_rad)

    # Calculate total phase shift
    phaseOfIncidentWaveAtElement = phaseConstant * (xVector + yVector + zVector)

    return phaseOfIncidentWaveAtElement


def calculate_most_center_location(points):
    """
    Calculate the centroid of 3D points and find the closest point to it.
    
    This function finds the geometric center (centroid) of a set of 3D points
    and determines which input point is closest to this centroid. It also
    calculates the maximum extents of the point cloud.
    
    Parameters
    ----------
    points : list of tuples
        List of (x, y, z) coordinate tuples representing 3D points
        
    Returns
    -------
    tuple
        - closest_point (numpy.ndarray): Point closest to centroid
        - closest_index (int): Index of the closest point
        - centroid (tuple): Calculated centroid coordinates
        - max_dist (numpy.ndarray): Maximum extents [max_x, max_y]
    """
    # Calculate centroid coordinates
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    centroid_x = sum(x) / len(x)
    centroid_y = sum(y) / len(y)
    centroid_z = sum(z) / len(z)
    centroid = (centroid_x, centroid_y, centroid_z)
    
    # Find point closest to centroid
    closest_point = None
    closest_distance = float('inf')
    distance_all = []
    dist_x = []
    dist_y = []
    
    for i, point in enumerate(points):
        # Calculate Euclidean distance to centroid
        distance = np.sqrt((point[0] - centroid_x) ** 2 + (point[1] - centroid_y) ** 2 + (point[2] - centroid_z) ** 2)
        distance_all.append(distance)
        dist_x.append(point[0])
        dist_y.append(point[1])
        
        if distance < closest_distance:
            closest_distance = distance
            closest_point = point
            closest_index = i
            
    # Calculate maximum extents
    max_dist = np.max(np.array(distance_all))
    max_dist = np.array([np.max(np.array(dist_x)),np.max(np.array(dist_y))])
    
    return np.array(closest_point), closest_index, centroid, max_dist


def find_center_and_maxmin_array_pos(pos_dict):
    """
    Find the center position and extents of an antenna array.
    
    Given a dictionary mapping port names to positions, this function determines
    the most centrally located element and calculates the array extents.
    
    Parameters
    ----------
    pos_dict : dict
        Dictionary mapping port names (strings) to position tuples (x, y, z)
        
    Returns
    -------
    tuple
        - closest_point (numpy.ndarray): Position of most central element
        - port_name (str): Name of the most central port
        - centroid (tuple): Geometric center of all elements
        - max_dist (numpy.ndarray): Maximum array extents [max_x, max_y]
    """
    pos = []
    port_names = list(pos_dict.keys())
    
    # Extract all positions
    for port in pos_dict:
        pos.append(pos_dict[port])
        
    # Find most central location
    closest_point, closest_index, centroid, max_dist = calculate_most_center_location(pos)
    
    return closest_point, port_names[closest_index], centroid, max_dist


class FarFields:
    """
    Main class for handling far-field antenna pattern data.
    
    This class provides comprehensive functionality for reading, processing, and analyzing
    far-field antenna patterns from various file formats. It supports individual element
    patterns as well as array configurations with embedded element patterns.
    
    Attributes
    ----------
    max_distance : numpy.ndarray or None
        Maximum dimensions of the antenna array
    centroid : tuple or None
        Geometric center of the antenna array
    metadata_file_path : str or None
        Path to metadata file containing additional information
    path_dict : dict or None
        Mapping of port names to far-field file paths
    location_dict : dict or None
        Mapping of port names to 3D positions
    data_dict : dict or None
        Storage for loaded far-field data objects
    incident_power_for_binary_import : float
        Power scaling factor for binary file imports
    lattice_vector : numpy.ndarray or None
        Lattice vectors for periodic array structures
    geo_file_path : str or None
        Path to geometry file (.obj format)
    exported_path : str or None
        Path where exported data was saved
    center_most_pos : tuple
        Position of the most central array element
    center_most_port : str or None
        Name of the most central port
    max_array_pos : tuple
        Maximum position coordinates in the array
    min_array_pos : tuple
        Minimum position coordinates in the array
    model_units : str
        Units used in the model ('mm', 'm', etc.)
    ff_meshes : dict
        Storage for 3D visualization meshes
    all_port_names : list
        List of all available port names
    """
    
    def __init__(self, incident_power_for_binary_import=1, location_dict=None):
        """
        Initialize FarFields object.
        
        Parameters
        ----------
        incident_power_for_binary_import : float, optional
            Power scaling factor for binary imports (default: 1)
        location_dict : dict, optional
            Pre-defined mapping of port names to positions (default: None)
        """
        self.max_distance = None
        self.centroid = None
        self.metadata_file_path = None
        self.path_dict = None
        self.location_dict = location_dict
        self.data_dict = None
        self.incident_power_for_binary_import = incident_power_for_binary_import
        self.lattice_vector = None
        self.geo_file_path = None
        self.exported_path = None
        self.center_most_pos = (0, 0, 0)  # position
        self.center_most_port = None  # position
        self.max_array_pos = (0, 0, 0)  # maximum position of the array
        self.min_array_pos = (0, 0, 0)
        self.model_units = 'mm'
        self.ff_meshes = {}

    def read_eep_map(self, file_name):
        """
        Read an embedded element pattern (EEP) mapping file.
        
        This method parses a text file that maps port names to far-field data files
        and optionally includes position information for each antenna element.
        
        Parameters
        ----------
        file_name : str
            Path to the EEP mapping file (.txt format)
            
        Notes
        -----
        The mapping file format should be:
        Port_Name File_Base_Name [x_pos y_pos z_pos]
        
        Associated files that may be read:
        - .latvec: Lattice vector information
        - .obj: Geometry file
        - .info: Metadata file with units and other information
        """
        eep_map_path, eep_map_file = os.path.split(file_name)
        base_file_name, file_extension = os.path.splitext(os.path.abspath(file_name))
        
        # Define associated file paths
        lattice_vec_file = os.path.join(eep_map_path, base_file_name + '.latvec')
        geo_file = os.path.join(eep_map_path, base_file_name + '.obj')
        metadata_file = os.path.join(eep_map_path, base_file_name + '.info')
        
        # Read the mapping file
        with open(file_name, 'r') as reader:
            lines = [line.strip().split(None) for line in reader]
        reader.close()
        lines = lines[1:]  # remove header

        self.path_dict = {}
        self.location_dict = {}
        
        # Parse each line in the mapping file
        for pattern in lines:
            # Earlier versions did not contain position information
            if len(pattern) >= 2:
                port = pattern[0]
                if ':' in port:
                    port = port.split(':')[0]  # Remove any port qualifiers
                    
                # Check for different file format options (priority order)
                if os.path.exists(eep_map_path + '/' + pattern[1] + '.bffd'): # Binary NetCDF format
                    self.path_dict[port] = eep_map_path + '/' + pattern[1] + '.bffd'
                elif os.path.exists(eep_map_path + '/' + pattern[1] + '.ffld'): # Binary solver format
                    self.path_dict[port] = eep_map_path + '/' + pattern[1] + '.ffld'
                else: # ASCII format (assumed to exist)
                    self.path_dict[port] = eep_map_path + '/' + pattern[1] + '.ffd'
                    
            # Parse position information if available
            if len(pattern) == 5:  # Contains position information
                x = float(pattern[2])
                y = float(pattern[3])
                z = float(pattern[4])
                xyz = np.array([x, y, z])
                self.location_dict[port] = xyz
                
        # Calculate array center and extents if positions are available
        if self.location_dict:
            self.center_most_pos, self.center_most_port, self.centroid, self.max_distance = find_center_and_maxmin_array_pos(self.location_dict)

        # Read lattice vector file if it exists
        if os.path.exists(lattice_vec_file):
            with open(lattice_vec_file, 'r') as reader:
                for line in reader:
                    lat_vec = line.split(',')
            reader.close()
            lattice_vec = []
            for each in lat_vec:
                try:
                    lattice_vec.append(float(each))
                except:
                    lattice_vec.append(0.0)
            if len(lattice_vec) > 6:
                lattice_vec = lattice_vec[:5]
            self.lattice_vector = np.array(lattice_vec)

        # Store geometry file path if it exists
        if os.path.exists(geo_file):
            self.geo_file_path = geo_file
            
        # Read metadata file if it exists
        if os.path.exists(metadata_file):
            self.metadata_file_path = metadata_file
            with open(self.metadata_file_path , 'r') as reader:
                for line in reader:
                    if 'Units:' in line:
                        self.model_units = line.split(':')[1]
            reader.close()


    def read_ffd(self, ffd_input, create_farfield_mesh=False, scale_pattern=1, name='FarFieldData'):
        """
        Read far-field data from various file formats.
        
        This method is the main entry point for loading far-field antenna pattern data.
        It supports single files, dictionaries of files, or embedded element mapping files.
        
        Parameters
        ----------
        ffd_input : str or dict
            Either:
            - String path to a single .ffd file
            - String path to an EEP mapping file (.txt)  
            - Dictionary mapping port names to file paths
        create_farfield_mesh : bool, optional
            Whether to create 3D visualization meshes (default: False)
        scale_pattern : float, optional
            Scaling factor for 3D mesh size (default: 1)
        name : str, optional
            Name for the far-field data in meshes (default: 'FarFieldData')
            
        Returns
        -------
        bool
            False if loading failed, otherwise None
            
        Notes
        -----
        Supported file formats:
        - .ffd: ASCII far-field format
        - .ffld: Binary solver format  
        - .bffd: Binary NetCDF format
        - .txt: Embedded element pattern mapping file
        """
        time_before = walltime.time()
        print('Loading Embedded Element Patterns...')
        self.data_dict = {}
        valid_ffd = True

        # Check if input is an embedded element mapping file
        if isinstance(ffd_input, str):
            file_name, file_extension = os.path.splitext(os.path.abspath(ffd_input))
            if file_extension == '.txt':  # EEP mapping file
                self.read_eep_map(ffd_input)  # Parse mapping file first
                ffd_input = self.path_dict

        # Process input format
        if isinstance(ffd_input, dict):
            ffd_dict = ffd_input
            all_ports = list(ffd_input.keys())
        elif isinstance(ffd_input, str):  # Single file
            all_ports = ['1']
            ffd_dict = {'1': ffd_input}
        else:
            print('ffd dict or ffd file name must be defined')
            return False

        all_freq = []
        index = 0
        
        # Load data for each port
        for port in ffd_dict.keys():
            if os.path.exists(ffd_dict[port]):
                base_path, file_name = os.path.split(os.path.abspath(ffd_dict[port]))
                base_name, extension = os.path.splitext(file_name)
                
                # Read file based on extension
                if create_farfield_mesh:
                    if extension == '.ffd':  # ASCII format
                        data = self.read_ascii_ffd(ffd_dict[port])
                    elif extension == '.ffld':  # Binary solver format
                        data = self.read_binary_ffd(ffd_dict[port], incident_power=self.incident_power_for_binary_import)
                    elif extension == '.bffd':  # Binary NetCDF format
                        data = self.read_binary_ffd_netcdf(ffd_dict[port])
                else:
                    data= FF_Data()

                # Add position information if available
                if self.location_dict is not None:
                    if port in self.location_dict.keys():
                        data.Position = self.location_dict[port]
                        
                # Add lattice vector information if available
                if self.lattice_vector is not None:
                    data.Lattice_Vector = self.lattice_vector
                    data.Is_Component_Array = True
                    
                # Create 3D mesh if requested
                if create_farfield_mesh:
                    self.ff_meshes[port] = self.gen_farfield_mesh_from_ffd(
                        data.calc_RealizedGainTotal(),
                        data.Theta,
                        data.Phi,
                        scale_pattern=scale_pattern,
                        name=name
                    )
            else:
                print('Port Missing')
                return False

            self.data_dict[port] = data

        self.all_port_names = list(self.data_dict.keys())

        elapsed_time = walltime.time() - time_before
        print(f'Loading Embedded Element Patterns...Done: {elapsed_time} seconds')

    def export_aerial6g_data(self, output_directory=None):
        """
        Export far-field data in Aerial 6G compatible CSV format.
        
        This method exports the loaded far-field data to CSV files that can be
        imported into Aerial 6G for propagation studies and system-level simulations.
        
        Parameters
        ----------
        output_directory : str, optional
            Directory where CSV files should be saved. If None, uses current directory.
            
        Notes
        -----
        Each port's data is saved to a separate CSV file with the port name.
        The format includes theta, phi angles and complex field components.
        Only the first frequency is exported for each port.
        """
        if self.data_dict is None:
            return

        if output_directory is None:
            output_directory = os.getcwd()

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Define CSV header format
        header_str = "Theta,Phi,Re(Complex_amplitude_theta),Im(Complex_amplitude_theta),Re(Complex_amplitude_phi),Im(Complex_amplitude_phi)"

        # Export each port to separate CSV file
        for port in self.data_dict:
            file_path = os.path.join(output_directory, port + '.csv')
            with open(file_path, 'w') as writer:
                # Write header information
                writer.write("# File Type: Far field"+ '\n'+ '\n')
                writer.write(f"#Frequency: {self.data_dict[port].Frequencies[0]}" + '\n')
                writer.write("#Coordinate System: Spherical" + '\n')
                writer.write(f"#No. of Theta Samples: {len(self.data_dict[port].Theta)}" + '\n')
                writer.write(f"#No. of Phi Samples: {len(self.data_dict[port].Phi)}" + '\n'+ '\n')
                writer.write("# Theta and phi are in degree, and fields are in linear scale"+ '\n'+ '\n')
                writer.write("#No. of Header Lines: 1" + '\n' + '\n')
                writer.write("#Date:" + '\n' + '\n')
                writer.write(header_str + '\n')

                # Write data points
                n = 0
                for theta in self.data_dict[port].Theta:
                    for phi in self.data_dict[port].Phi:
                        writer.write(f'{theta},{phi},{np.real(self.data_dict[port].rETheta[0,n])},'
                                     f'{np.imag(self.data_dict[port].rETheta[0,n])},'
                                     f'{np.real(self.data_dict[port].rEPhi[0,n])},'
                                     f'{np.imag(self.data_dict[port].rEPhi[0,n])}\n')
                        n+=1
        writer.close()

    def export_binary_data(self, ff_inst):
        """
        Export far-field data to binary NetCDF format.
        
        This method converts ASCII far-field files to efficient binary NetCDF format
        for faster loading and reduced file sizes.
        
        Parameters
        ----------
        ff_inst : FarFields
            FarFields instance containing the data to export
            
        Notes
        -----
        Creates .bffd files in the same directory as the original files.
        The NetCDF format preserves all frequency and field component information.
        """
        if ff_inst is None:
            print('No Data to Export, please load data')
            return
            
        ports = list(ff_inst.data_dict.keys())
        
        for port in ports:
            try:
                ncfile.close()  # Close any existing dataset
            except:
                pass

            # Generate output file path
            base_path, file_name = os.path.split(os.path.abspath(ff_inst.path_dict[port]))
            base_name, extension = os.path.splitext(file_name)
            new_file_name = os.path.join(base_path, base_name + '.bffd')
            print(f'Writing binary file {port}, {new_file_name}')
            
            # Create NetCDF file
            ncfile = Dataset(new_file_name, mode='w', format='NETCDF4_CLASSIC')
            ncfile.description = f"Far Field Export {port}"

            # Get data dimensions
            num_freq = ff_inst.data_dict[port].Num_Freq
            num_theta = ff_inst.data_dict[port].Theta.shape[0]
            num_phi = ff_inst.data_dict[port].Phi.shape[0]
            
            # Create dimensions
            theta_dim = ncfile.createDimension('Theta', num_theta)
            phi_dim = ncfile.createDimension('Phi', num_phi)
            freq_dim = ncfile.createDimension('Frequency', num_freq)

            # Create coordinate variables
            theta = ncfile.createVariable('Theta', np.float64, ('Theta',))
            theta.units = 'degree'
            theta.long_name = 'Theta'

            phi = ncfile.createVariable('Phi', np.float64, ('Phi',))
            phi.units = 'degree'
            phi.long_name = 'Phi'

            freq = ncfile.createVariable('Frequency', np.float64, ('Frequency',))
            freq.units = 'Hz'
            freq.long_name = 'Frequency'

            # Prepare data arrays
            theta_data = ff_inst.data_dict[port].Theta
            phi_data = ff_inst.data_dict[port].Phi
            freq_data = ff_inst.data_dict[port].Frequencies
            re_theta_data = ff_inst.data_dict[port].rETheta.reshape((num_freq, num_theta, num_phi))
            re_phi_data = ff_inst.data_dict[port].rEPhi.reshape((num_freq, num_theta, num_phi))

            # Create field component variables (real and imaginary parts)
            re_theta_re = ncfile.createVariable('rETheta_Real', np.float64, ('Frequency', 'Theta', 'Phi'))
            re_theta_re.units = 'Volt Per Meter'
            re_theta_re.standard_name = 'rETheta_Real'

            re_theta_im = ncfile.createVariable('rETheta_Imag', np.float64, ('Frequency', 'Theta', 'Phi'))
            re_theta_im.units = 'Volt Per Meter'
            re_theta_im.standard_name = 'rETheta_Imag'

            re_phi_re = ncfile.createVariable('rEPhi_Real', np.float64, ('Frequency', 'Theta', 'Phi'))
            re_phi_re.units = 'Volt Per Meter'
            re_phi_re.standard_name = 'rEPhi_Real'

            re_phi_im = ncfile.createVariable('rEPhi_Imag', np.float64, ('Frequency', 'Theta', 'Phi'))
            re_phi_im.units = 'Volt Per Meter'
            re_phi_im.standard_name = 'rEPhi_Imag'

            # Write data to file
            theta[:] = theta_data
            phi[:] = phi_data
            freq[:] = freq_data
            re_theta_re[:, :, :] = re_theta_data.real
            re_phi_re[:, :, :] = re_phi_data.real
            re_theta_im[:, :, :] = re_theta_data.imag
            re_phi_im[:, :, :] = re_phi_data.imag
            ncfile.close()
            
        print('Binary Export Completed')

    def read_binary_ffd_netcdf(self, filepath, incident_power=1):
        """
        Read far-field data from NetCDF binary format.
        
        Parameters
        ----------
        filepath : str
            Path to the .bffd NetCDF file
        incident_power : float, optional
            Power scaling factor (default: 1)
            
        Returns
        -------
        FF_Data
            Far-field data object containing the loaded information
        """
        data = FF_Data()
        print('Far field pattern is binary (NetCDF)')
        base_path, filename_only = os.path.split(filepath)

        # Open NetCDF file
        nc_fid = Dataset(filepath, 'r')

        # Extract coordinate and field data
        theta = np.array(nc_fid.variables['Theta'][:])
        phi = np.array(nc_fid.variables['Phi'][:])
        freq = np.array(nc_fid.variables['Frequency'][:])
        re_theta_re = np.array(nc_fid.variables['rETheta_Real'][:].flatten())
        re_theta_im = np.array(nc_fid.variables['rETheta_Imag'][:].flatten())
        re_phi_re = np.array(nc_fid.variables['rEPhi_Real'][:].flatten())
        re_phi_im = np.array(nc_fid.variables['rEPhi_Imag'][:].flatten())

        # Calculate dimensions
        num_freq = freq.shape[0]
        all_freq = freq
        samples_per_freq = theta.shape[0] * phi.shape[0]
        
        # Reconstruct complex field arrays
        data.rETheta = np.vectorize(complex)(re_theta_re, re_theta_im).reshape((num_freq, samples_per_freq))
        data.rEPhi = np.vectorize(complex)(re_phi_re, re_phi_im).reshape((num_freq, samples_per_freq))
        
        # Store coordinate and metadata
        data.Theta = theta
        data.Phi = phi
        data.Frequencies = np.array(all_freq)
        data.Delta_Theta = np.abs(theta[1] - theta[0])
        data.Delta_Phi = np.abs(phi[1] - phi[0])
        data.Diff_Area = np.abs(np.radians(data.Delta_Theta) * np.radians(data.Delta_Phi) * np.sin(np.radians(theta)))
        data.Total_Samples = len(data.rETheta[0])
        data.Num_Freq = len(data.Frequencies)
        
        return data

    def read_binary_ffd(self, filepath, incident_power=1):
        """
        Read far-field data from binary solver format (experimental).
        
        Parameters
        ----------
        filepath : str
            Path to the .ffld binary file
        incident_power : float, optional
            Power scaling factor (default: 1)
            
        Returns
        -------
        FF_Data
            Far-field data object containing the loaded information
            
        Notes
        -----
        This is an experimental reader for binary files produced directly
        by electromagnetic simulation solvers. Format may vary by solver.
        """
        data = FF_Data()
        print('Far field pattern is binary (Experimental)')
        base_path, filename_only = os.path.split(filepath)
        
        with open(filepath, "rb") as fp:
            # Read header information (binary format)
            mStartTheta = struct.unpack('d', fp.read(8))[0]
            mEndTheta = struct.unpack('d', fp.read(8))[0]
            ntheta = struct.unpack('i', fp.read(4))[0]
            mStartPhi = struct.unpack('d', fp.read(8))[0]
            mEndPhi = struct.unpack('d', fp.read(8))[0]
            nphi = struct.unpack('i', fp.read(4))[0]

            # Read complex field data
            numSize = ntheta * nphi
            Etheta = [None] * numSize
            Ephi = [None] * numSize
            
            # Read theta component
            for i in range(numSize):
                Etheta[i] = complex(*struct.unpack('dd', fp.read(16)))
            # Read phi component    
            for i in range(numSize):
                Ephi[i] = complex(*struct.unpack('dd', fp.read(16)))

            # Set up coordinate grids
            num_freq = 1
            all_freq = [1e9]  # Default frequency
            theta_range = np.linspace(mStartTheta, mEndTheta, num=ntheta)
            phi_range = np.linspace(mStartPhi, mEndPhi, num=nphi)
            samples_per_freq = ntheta * nphi
            
            # Apply scaling factor
            scale_factor = incident_power * np.sqrt(2)
            data.rETheta = np.array(Etheta, dtype='complex').reshape((num_freq, samples_per_freq)) * scale_factor
            data.rEPhi = np.array(Ephi, dtype='complex').reshape((num_freq, samples_per_freq)) * scale_factor
            
            # Store metadata
            data.Theta = theta_range
            data.Phi = phi_range
            data.Frequencies = np.array(all_freq)
            data.Delta_Theta = np.abs(theta_range[1] - theta_range[0])
            data.Delta_Phi = np.abs(phi_range[1] - phi_range[0])
            data.Diff_Area = np.abs(np.radians(data.Delta_Theta) * np.radians(data.Delta_Phi) * np.sin(np.radians(theta_range)))
            data.Total_Samples = len(data.rETheta[0])
            data.Num_Freq = len(data.Frequencies)
            
        return data

    def read_ascii_ffd(self, filepath):
        """
        Read far-field data from ASCII format file.
        
        This method parses ASCII .ffd files containing far-field antenna pattern data.
        The format includes header information followed by complex field components.
        
        Parameters
        ----------
        filepath : str
            Path to the .ffd ASCII file
            
        Returns
        -------
        FF_Data
            Far-field data object containing the loaded information
            
        Notes
        -----
        Expected file format:
        Line 1: theta_start theta_end num_theta
        Line 2: phi_start phi_end num_phi  
        Line 3: Frequency num_frequencies
        Data: Re(E_theta) Im(E_theta) Re(E_phi) Im(E_phi) for each point
        """
        data = FF_Data()
        all_freq = []
        temp_dict = {}
        base_path, filename_only = os.path.split(filepath)
        
        with open(filepath, 'r') as reader:
            # Read header information
            theta = [int(i) for i in reader.readline().split()]  # [start, end, num_points]
            phi = [int(i) for i in reader.readline().split()]
            num_freq = int(reader.readline().split()[1])
            
            # Generate coordinate arrays
            theta_range = np.linspace(*theta)
            phi_range = np.linspace(*phi)
            ntheta = len(theta_range)
            nphi = len(phi_range)
            samples_per_freq = ntheta * nphi

            # Parse frequency information from file
            freq_index = -1
            field_index = 0
            for line in reader:
                if 'Frequency' in line:
                    freq = float(line.split()[1])
                    all_freq.append(freq)
                    freq_index += 1
                    field_index = 0
        reader.close()

        # Load numerical data (skip header lines, ignore frequency markers)
        eep_txt = np.loadtxt(filepath, skiprows=4, comments='Frequency')
        
        # Reconstruct complex field arrays
        Etheta = np.vectorize(complex)(eep_txt[:, 0], eep_txt[:, 1])
        Ephi = np.vectorize(complex)(eep_txt[:, 2], eep_txt[:, 3])

        # Store in data object
        data.rETheta = Etheta.reshape((num_freq, samples_per_freq))
        data.rEPhi = Ephi.reshape((num_freq, samples_per_freq))
        data.Theta = theta_range
        data.Phi = phi_range
        data.Frequencies = np.array(all_freq)
        data.Delta_Theta = np.abs(theta_range[1] - theta_range[0])
        data.Delta_Phi = np.abs(phi_range[1] - phi_range[0])
        data.Diff_Area = np.abs(np.radians(data.Delta_Theta) * np.radians(data.Delta_Phi) * np.sin(np.radians(theta_range)))
        data.Total_Samples = len(data.rETheta[0])
        data.Num_Freq = len(data.Frequencies)
        
        return data


    def gen_farfield_mesh_from_ffd(self, ff_data, theta, phi,freq_idx=0, mesh_limits=[0, 1], scale_pattern=1, name='FarField'):
        """
        Generate a 3D mesh for visualizing far-field radiation patterns.
        
        This method creates a PyVista StructuredGrid that represents the 3D radiation
        pattern shape, with the radial distance proportional to the field magnitude.
        
        Parameters
        ----------
        ff_data : numpy.ndarray
            Far-field magnitude data (e.g., gain, directivity)
        theta : numpy.ndarray
            Theta angle coordinates in degrees
        phi : numpy.ndarray
            Phi angle coordinates in degrees
        mesh_limits : list, optional
            [min_value, max_scale] for mesh normalization (default: [0, 1])
        scale_pattern : float, optional
            Overall scaling factor for mesh size (default: 1)
        name : str, optional
            Name for the scalar data array in the mesh (default: 'FarField')
            
        Returns
        -------
        pyvista.StructuredGrid
            3D mesh representing the radiation pattern
            
        Notes
        -----
        The mesh shape follows the radiation pattern magnitude, with:
        - Radial distance proportional to field strength
        - Color mapping based on original field values
        - Spherical coordinate conversion to Cartesian
        """

        ff_data_shape = ff_data.shape
        if len(ff_data_shape) > 1:
            ff_data = ff_data[freq_idx, :]
            
        # Handle negative values by shifting data to ensure all positive
        if ff_data.min() < 0:
            ff_data_renorm = ff_data + np.abs(ff_data.min())
        else:
            ff_data_renorm = ff_data

        # Convert angles to radians for coordinate transformation
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        # Reshape data to match coordinate grids
        r_no_renorm = np.reshape(ff_data, (len(theta), len(phi)))  # Original values for coloring
        r = np.reshape(ff_data_renorm, (len(theta), len(phi)))     # Normalized values for shape
        r_max = np.max(r)
        
        # Convert to Cartesian coordinates with radial scaling
        x = mesh_limits[1] * r / r_max * np.sin(theta_grid) * np.cos(phi_grid)
        y = mesh_limits[1] * r / r_max * np.sin(theta_grid) * np.sin(phi_grid)
        z = mesh_limits[1] * r / r_max * np.cos(theta_grid)

        # Flatten original data for color mapping
        mag = np.ndarray.flatten(r_no_renorm, order='F')

        # Create PyVista structured grid
        ff_mesh = pv.StructuredGrid(x, y, z)
        ff_mesh.scale(scale_pattern, inplace=True)
        
        # Store original field values for color visualization
        ff_mesh[name] = mag
        
        return ff_mesh
    
    def plot(self):
        """
        Display 3D visualization of loaded far-field meshes.
        
        This method creates an interactive 3D plot showing all loaded far-field
        patterns using PyVista. Each port's pattern is displayed as a separate mesh.
        
        Notes
        -----
        Requires that create_farfield_mesh=True was used when loading the data.
        The plot will show all ports with individual scalar bars for field values.
        """
        if self.ff_meshes is None:
            print('No Far Field Meshes to plot')
            return

        pl = pv.Plotter()
        for port in self.ff_meshes:
            pl.add_mesh(self.ff_meshes[port], show_scalar_bar=True, name=port)
        pl.show()
        pl.close()