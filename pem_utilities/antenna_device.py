"""
Created on Fri Jan 26 15:00:00 2024

@author: asligar
"""

import json
import os
import numpy as np
from dataclasses import dataclass

from pathlib import Path

from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API
from pem_utilities.far_fields import FarFields
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.actor import Actor
from pem_utilities.rotation import euler_to_rot

pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object
RssPy = pem_api_manager.RssPy

paths = get_repo_paths()
default_antenna_path = paths.antenna_device_library
model_path = paths.models


@dataclass
class Waveform:
    """
    A class to represent and configure radar waveform parameters.
    Attributes:
        waveform_dict (dict): Dictionary containing waveform configuration parameters.
        vel_domain (numpy.ndarray): Velocity domain for the waveform.
        rng_domain (numpy.ndarray): Range domain for the waveform.
        freq_domain (numpy.ndarray): Frequency domain for the waveform.
        pulse_domain (numpy.ndarray): Pulse domain for the waveform.
        r_specs (str): Range window specifications (default: Hann window with 50 dB sidelobe level).
        d_specs (str): Doppler window specifications (default: Hann window with 50 dB sidelobe level).
        mode (str): Radar mode, either 'pulseddoppler' or 'fmcw' (default: 'pulseddoppler').
        output (str): Output type, either 'rangepulse', 'freqpulse', 'rangedoppler', or 'adc_samples' (default: 'freqpulse'. 'rangepulse' available >= 25.2).
        center_freq (float): Center frequency of the radar waveform (default: 76.5 GHz).
        bandwidth (float): Bandwidth of the radar waveform (default: 1 GHz).
        num_freq_samples (int): Number of frequency samples (default: 101).
        num_pulse_cpi (int): Number of pulses per coherent processing interval (default: 201).
        cpi_duration (float): Duration of the coherent processing interval (default: 1 ms).
        pulse_interval (float): Interval between pulses (calculated based on cpi_duration and num_pulse_cpi).
        mode_delay (rss_py.ModeDelayReference): Mode delay reference, either CENTER_CHIRP or FIRST_CHIRP (default: CENTER_CHIRP).
        tx_multiplex (rss_py.TxMultiplex): Transmit multiplexing mode, either SIMULTANEOUS or INTERLEAVED (default: SIMULTANEOUS). depracated: INDIVIDUAL will be equal to simultaneous.
        adc_sample_rate (float): ADC sample rate (default: 50 MHz).
        is_iq_channel (bool): Indicates whether the channel is IQ (default: True).
        tx_incident_power (float): Transmit incident power (default: 1.0).
        rx_noise_db (float): Receiver noise level in dB (default: None).
        rx_gain_db (float): Receiver gain in dB (default: None).
    Methods:
        __init__(waveform_dict):
            Initializes the Waveform object with the given configuration dictionary.
        get_response_domains(h_mode):
            Calculates and sets the response domains for the waveform based on the radar mode.
        _lowercase(obj):
            Converts all keys and values in a dictionary to lowercase. Handles nested dictionaries, lists, sets, tuples, and strings.
    """

    def __init__(self, waveform_dict):
        waveform_dict = self._lowercase(waveform_dict)
        self.waveform_dict = waveform_dict
        self.vel_domain = None
        self.rng_domain = None
        self.freq_domain = None
        self.pulse_domain = None

        sideLobeLevelDb = 50.
        self.r_specs = "hann," + str(sideLobeLevelDb)
        self.d_specs = "hann," + str(sideLobeLevelDb)

        if 'mode' in waveform_dict.keys():  # can be PulsedDoppler or FMCW
            self.mode = waveform_dict.get('mode').lower().strip()
        else:
            self.mode = 'pulseddoppler'
        if 'output' in waveform_dict.keys():  # can be FreqPulse, RangeDoppler, RangePulse (25.2 or later) or ADC_SAMPLES
            self.output = waveform_dict.get('output').lower().strip()
        else:
            self.mode = 'freqpulse'

        if 'center_freq' in waveform_dict.keys():
            self.center_freq = waveform_dict.get('center_freq')
        else:
            self.center_freq = 76.5e9
        if 'bandwidth' in waveform_dict.keys():
            self.bandwidth = waveform_dict.get('bandwidth')
        else:
            self.bandwidth = 1.0e9
        if 'num_freq_samples' in waveform_dict.keys():
            self.num_freq_samples = waveform_dict.get('num_freq_samples')
        else:
            self.num_freq_samples = 101

        if 'num_pulse_cpi' in waveform_dict.keys():
            self.num_pulse_cpi = waveform_dict.get('num_pulse_cpi')
        else:
            self.num_pulse_cpi = 201
        if 'cpi_duration' in waveform_dict.keys():
            if 'pulse_interval' in waveform_dict.keys():
                print('Both cpi_duration and pulse_interval are defined. Using cpi_duration')
            self.cpi_duration = waveform_dict.get('cpi_duration')
            self.pulse_interval = self.cpi_duration / self.num_pulse_cpi
        else:
            if 'pulse_interval' in waveform_dict.keys():
                self.pulse_interval = waveform_dict.get('pulse_interval')
            else:
                self.pulse_interval = self.cpi_duration / self.num_pulse_cpi
            self.cpi_duration = 1.0e-3

        if 'mode_delay' in waveform_dict.keys():
            if waveform_dict.get('mode_delay').lower().strip() == 'first_chirp':
                self.mode_delay = RssPy.ModeDelayReference.FIRST_CHIRP
            else:
                self.mode_delay = RssPy.ModeDelayReference.CENTER_CHIRP
        else:
            self.mode_delay = RssPy.ModeDelayReference.CENTER_CHIRP  # CENTER_CHIRP or FIRST_CHIRP
        if 'tx_multiplex' in waveform_dict.keys():
            if waveform_dict.get('tx_multiplex').lower().strip() == 'interleaved':
                self.tx_multiplex = RssPy.TxMultiplex.INTERLEAVED
            else: # default state if use INDIVIDUAL or any other value that is not interleaved. Accounts for depracation of INDIVIDUAL
                self.tx_multiplex = RssPy.TxMultiplex.SIMULTANEOUS
        else:
            self.tx_multiplex = RssPy.TxMultiplex.SIMULTANEOUS  # SIMULTANEOUS or INTERLEAVED
        if 'adc_sample_rate' in waveform_dict.keys():
            self.adc_sample_rate = waveform_dict.get('adc_sample_rate')
        else:
            self.adc_sample_rate = 50.0e6
        if 'is_iq_channel' in waveform_dict.keys():
            self.is_iq_channel = waveform_dict.get('is_iq_channel')
        else:
            self.is_iq_channel = True

        if 'tx_incident_power' in waveform_dict.keys():
            self.tx_incident_power = waveform_dict.get('tx_incident_power')
        else:
            self.tx_incident_power = 1.0
        if 'rx_noise_db' in waveform_dict.keys():
            self.rx_noise_db = waveform_dict.get('rx_noise_db')
        else:
            self.rx_noise_db = None
        if 'rx_gain_db' in waveform_dict.keys():
            self.rx_gain_db = waveform_dict.get('rx_gain_db')
        else:
            self.rx_gain_db = None

    def get_response_domains(self, h_mode):
        """
        Computes and sets the response domains based on the specified output type and hardware mode.
        Parameters:
            h_mode (str): The hardware mode used for domain computation.
        Attributes:
            vel_domain (numpy.ndarray): Velocity domain values computed based on CPI duration and center frequency.
            rng_domain (numpy.ndarray): Range domain values computed based on bandwidth and number of frequency samples.
            pulse_domain (numpy.ndarray): Pulse domain values computed based on CPI duration and number of pulses per CPI.
            freq_domain (numpy.ndarray): Frequency domain values computed based on center frequency and bandwidth.
            max_range (float): Maximum range computed based on bandwidth and number of frequency samples.
            vel_res (float): Velocity resolution computed based on CPI duration and center frequency.
            vel_win (float): Velocity window computed based on velocity resolution and number of frequency samples.
            fast_time_domain (numpy.ndarray): Fast time domain values computed based on range domain and speed of light.
        Notes:
            - For 'rangedoppler' or 'dopplerrange' output types, the response domains are computed for RANGE_DOPPLER.
            - For 'adc_samples' or 'freqpulse' output types, the response domains are computed for ADC_SAMPLES or FREQ_PULSE respectively.
            - The range domain assumes round-trip calculations. For one-way simulations (e.g., P2P), adjustments may be needed (x2).
        """
        # response domain for the waveform are assumed to be round trip, if you need one way for P2P simulation you
        # may need to multiply the range/time domain by 2

        if self.output.lower().replace('_','') == 'rangedoppler' or self.output == 'dopplerrange':
            (ret, self.vel_domain, self.rng_domain) = pem.responseDomains(h_mode, RssPy.ResponseType.RANGE_DOPPLER)

            self.pulse_domain = np.linspace(-self.cpi_duration / 2, self.cpi_duration / 2, num=self.num_pulse_cpi)
            # self.freq_domain = np.linspace(self.center_freq - self.bandwidth / 2, self.center_freq + self.bandwidth / 2,
            #                                num=self.num_freq_samples)
            freq_step = self.bandwidth/(self.num_freq_samples)
            self.freq_domain = np.linspace(self.center_freq - self.bandwidth / 2+freq_step/2, self.center_freq + self.bandwidth / 2-freq_step/2,
                                           num=self.num_freq_samples)
        elif self.output.lower().replace('_','') == 'rangepulse' or self.output == 'pulserange':
            # this output is only supported in 25.2 or later
            if float(pem_api_manager.version) < 252:
                raise RuntimeError("RANGE_PULSE output is only supported in Perceive EM API version 25.2 or later.")

            (ret, self.pulse_domain, self.rng_domain) = pem.responseDomains(h_mode, RssPy.ResponseType.RANGE_PULSE)

            self.vel_res = 2.99792458e8 / 2 / self.cpi_duration / self.center_freq
            self.vel_win = self.vel_res * self.num_freq_samples
            self.vel_domain = np.linspace(-self.vel_win, self.vel_win, num=int(self.vel_win / self.vel_res))
            # self.freq_domain = np.linspace(self.center_freq - self.bandwidth / 2, self.center_freq + self.bandwidth / 2,
            #                                num=self.num_freq_samples)
            # deal with half frequency step
            freq_step = self.bandwidth/(self.num_freq_samples)
            self.freq_domain = np.linspace(self.center_freq - self.bandwidth / 2+freq_step/2, self.center_freq + self.bandwidth / 2-freq_step/2,
                                           num=self.num_freq_samples)

        else:
            if self.output.lower().replace('_','') == 'adcsamples':
                (ret, self.pulse_domain, self.freq_domain) = pem.responseDomains(h_mode,
                                                                                 RssPy.ResponseType.ADC_SAMPLES)
            elif self.output.lower().replace('_','') == 'freqpulse':
                (ret, self.pulse_domain, self.freq_domain) = pem.responseDomains(h_mode, RssPy.ResponseType.FREQ_PULSE)
            rng_res = 2.99792458e8 / 2 / self.bandwidth
            self.max_range = rng_res * self.num_freq_samples-rng_res # bin offset by one range resolution to mark the start of the bin
            self.rng_domain = np.linspace(0, self.max_range, num=int(self.max_range / rng_res))
            self.vel_res = 2.99792458e8 / 2 / self.cpi_duration / self.center_freq
            self.vel_win = self.vel_res * self.num_freq_samples
            self.vel_domain = np.linspace(-self.vel_win, self.vel_win, num=int(self.vel_win / self.vel_res))
            self.fast_time_domain = self.rng_domain / 2.99792458e8

    def _lowercase(self,obj):
        """ Make dictionary lowercase """
        if isinstance(obj, dict):
            return {k.strip().lower(): self._lowercase(v) for k, v in obj.items()}
        elif isinstance(obj, (list, set, tuple)):
            t = type(obj)
            return t(self._lowercase(o) for o in obj)
        elif isinstance(obj, str):
            return obj.strip().lower()
        else:
            return obj

class AntennaDevice:
    """
    A class to represent an antenna device with various parameters and methods to initialize and manage the device.

    Attributes:
    ------------
    parent_h_node : object
        The parent node of the antenna device.

    h_node_platform : object
        The platform node of the antenna device, initialized to None.

    h_device : object
        The device handle, initialized to None.

    name : str
        The name of the antenna device.

    device : object
        The device object, initialized to None.

    post_processing : object
        The post-processing object, initialized to None.

    antennas_rx : dict
        A dictionary to store the receive antennas.

    antennas_tx : dict
        A dictionary to store the transmit antennas.

    device_json : dict
        A dictionary to store the device configuration in JSON format.

    coord_sys : object
        The coordinate system of the antenna device.

    modes : dict
        A dictionary to store the modes of the antenna device.

    waveforms : dict
        A dictionary to store the waveforms of the antenna device.

    all_actors : object
        An object to store all actors related to the antenna device. used for visualization. if
        not provided, visualization will not be available. does not impact simulation

    all_antennas_properties : dict
        A dictionary to store all properties for all antennas, needed for visualization.

    range_pixels : int
        The number of range pixels, default is 256.

    doppler_pixels : int
        The number of Doppler pixels, default is 128.

    center_vel : float
        The center velocity, default is 0.0.

    r_specs : str
        The range specifications for the antenna device.

    d_specs : str
        The Doppler specifications for the antenna device.

    path_of_antenna_device : str
        The path of the antenna device file.

    full_path_antenna_device : str
        The full path of the antenna device file.

    rss_py : object
        The RSS Python API core object.

    api : object
        The API core object.

    Methods:
    --------
    __init__(file_name=None, parent_h_node=None, name='AntennaDevice', all_actors=None):
        Constructs all the necessary attributes for the AntennaDevice object.

    _lowercase(obj):
        Converts all keys in a dictionary to lowercase.

    initialize_device():
        Initializes the antenna device.

    initialize_mode(mode_name=None):
        Initializes the mode for the antenna device.

    add_mode(mode_name):
        Adds a mode to the antenna device.

    set_mode_active(mode_name, status=True):
        Sets the mode active for the antenna device.

    add_antennas(mode_name=None, load_pattern_as_mesh=False, scale_pattern=1, antennas_dict=None):
        Adds antennas to the antenna device.
    """
    def __init__(self, file_name=None, parent_h_node=None,name='AntennaDevice',all_actors=None):


        self.parent_h_node = parent_h_node
        self.h_node_platform = None
        self.h_device = None
        self.name = name
        self.device = None
        self.post_processing = None
        self.antennas_rx = {}
        self.antennas_tx = {}
        self.device_json = None
        self.coord_sys = None
        self.modes = {}  # actual mode objects created in API
        self.active_mode_name = None
        self.waveforms = {}  # parameters of waveform
        self.all_actors = all_actors
        self.all_antennas_properties = {}  # will be populated with all properties for all antennas, needed for visualization

        # ToDo, these should be assigned to the mode, not the antenna device
        self.range_pixels = 256
        self.doppler_pixels = 128
        self.center_vel = 0.0
        sideLobeLevelDb = 50.
        self.r_specs = "hann," + str(sideLobeLevelDb)
        self.d_specs = "hann," + str(sideLobeLevelDb)

        self.fov = 180.0

        self.antenna_file_name = file_name
        self.full_path_antenna_device = None
        self.path_of_antenna_device = None

        self.rss_py = RssPy
        self.pem = pem
        file_exists = False

        if file_name is not None:
            if os.path.exists(file_name): # full path was provided
                self.full_path_antenna_device = file_name
                self.path_of_antenna_device = os.path.dirname(file_name)
            elif os.path.exists(os.path.join(default_antenna_path,file_name)): # check default location for antenna file
                self.full_path_antenna_device = os.path.abspath(os.path.join(default_antenna_path,file_name))
                self.path_of_antenna_device = os.path.dirname(self.full_path_antenna_device)
            else:
                self.path_of_antenna_device = default_antenna_path
        else:
            self.path_of_antenna_device= os.path.abspath(default_antenna_path)
            print("File name not provided, Initializing Empty Antenna Device Object")
            return

        with open(self.full_path_antenna_device ) as f:
            self.device_json = json.load(f)
        # make entire dictionary lowercase
        self.device_json = self._lowercase(self.device_json)

    def _lowercase(self,obj):
        """ Make dictionary lowercase """
        if isinstance(obj, dict):
            return {k.strip().lower(): self._lowercase(v) for k, v in obj.items()}
        elif isinstance(obj, (list, set, tuple)):
            t = type(obj)
            return t(self._lowercase(o) for o in obj)
        elif isinstance(obj, str):
            return obj.strip().lower()
        else:
            return obj

    def initialize_device(self):
        """
        Initializes the radar device and its associated platform.
        This method sets up the radar platform and radar device, linking them
        appropriately. If a parent node is provided, the radar platform is added
        as a child of the parent node. Otherwise, the radar platform is added
        independently. The radar device is then added as a child of the radar
        platform node. Additionally, a coordinate system is initialized for the
        radar platform.
        Attributes:
            h_node_platform: An instance of `rss_py.RadarPlatform` representing
                the radar platform node.
            h_device: An instance of `rss_py.RadarDevice` representing the radar
                device node.
            coord_sys: An instance of `CoordSys` representing the coordinate system
                associated with the radar platform.
        Raises:
            RuntimeError: If any API call fails during the initialization process.
        """
        self.h_node_platform = RssPy.RadarPlatform()
        if self.parent_h_node is None:
            pem_api_manager.isOK(self.pem.addRadarPlatform(self.h_node_platform))
        else:
            pem_api_manager.isOK(self.pem.addRadarPlatform(self.h_node_platform, self.parent_h_node))

        self.h_device = RssPy.RadarDevice()
        # this indicates that the radar device is a child of the radar_device node
        pem_api_manager.isOK(self.pem.addRadarDevice(self.h_device, self.h_node_platform))
        self.coord_sys = CoordSys(h_node=self.h_node_platform, parent_h_node=self.parent_h_node)

    def initialize_mode(self, mode_name=None):
        """
        Initializes the radar mode for the antenna device.
        This method sets up the radar mode based on the provided mode name or defaults
        to the first mode found in the device's waveform configuration. It ensures the
        device is initialized, loads post-processing parameters if available, and
        configures the radar mode and waveform.
        Args:
            mode_name (str, optional): The name of the radar mode to initialize.
                           Defaults to None, in which case the first mode
                           in the waveform configuration is used.
        Raises:
            KeyError: If the specified mode name is not found in the device's waveform
                  configuration.
        Notes:
            - If the device is not initialized, it will be initialized automatically.
            - Post-processing parameters such as `range_pixels`, `doppler_pixels`, and
              `center_vel` are loaded if available in the device configuration.
            - The radar mode is added to the device using the API and set as active.
        """
        if self.h_node_platform is None and self.h_device is None:
            self.initialize_device()
        if 'post_processing' in self.device_json.keys():
            self.post_processing = self.device_json['post_processing']
            self.range_pixels = self.post_processing['range_pixels']
            self.doppler_pixels = self.post_processing['doppler_pixels']
            self.center_vel = 0.0

        if mode_name is None and len(self.device_json['waveform'].keys()) > 0:
            mode_name = list(self.device_json['waveform'].keys())[0]
        if mode_name not in self.device_json['waveform'].keys():
            print(f"Mode {mode_name} not found in device file, using first mode found")
            mode_name = list(self.device_json['waveform'].keys())[0]

        self.waveforms[mode_name] = Waveform(self.device_json['waveform'][mode_name])

        self.modes[mode_name] = RssPy.RadarMode()
        pem_api_manager.isOK(self.pem.addRadarMode(self.modes[mode_name], self.h_device))
        self.set_mode_active(mode_name)

    def add_mode(self, mode_name):
        """
        Configures and adds a radar mode to the antenna device based on the specified waveform parameters.
        Args:
            mode_name (str): The name of the radar mode to be added. This mode must exist in the `self.modes` and
                             `self.waveforms` dictionaries.
        Raises:
            Exception: If any API call fails, an exception is raised indicating the failure.
        Details:
            - Sets the radar mode start delay to 0.
            - Configures transmission parameters such as multiplexing, center frequency, bandwidth, and incident power.
            - Configures reception parameters such as noise level and gain.
            - Depending on the waveform mode (`pulseddoppler` or `fmcw`), sets the corresponding system specifications.
            - Activates range-Doppler response if the output type is `rangedoppler` or `dopplerrange`.
        Notes:
            - The `self.waveforms` dictionary contains waveform-specific parameters such as `tx_multiplex`, `center_freq`,
              `bandwidth`, `num_freq_samples`, `pulse_interval`, `num_pulse_cpi`, `mode`, `is_iq_channel`,
              `tx_incident_power`, `rx_noise_db`, `rx_gain_db`, `adc_sample_rate`, `output`, `r_specs`, and `d_specs`.
            - The `self.modes` dictionary maps mode names to their corresponding radar mode objects.
            - The `self.antennas_tx` list is checked to determine if the antenna type is transmission-capable before
              activating range-Doppler response.
        """

        pem_api_manager.isOK(self.pem.setRadarModeStartDelay(self.modes[mode_name], 0., self.waveforms[mode_name].mode_delay))

        tx_multiplex = self.waveforms[mode_name].tx_multiplex
        center_freq = self.waveforms[mode_name].center_freq
        bandwidth = self.waveforms[mode_name].bandwidth
        num_freq_samples = self.waveforms[mode_name].num_freq_samples
        pulse_interval = self.waveforms[mode_name].pulse_interval
        num_pulse_cpi = self.waveforms[mode_name].num_pulse_cpi
        mode = self.waveforms[mode_name].mode
        is_iq_channel = self.waveforms[mode_name].is_iq_channel
        tx_incident_power = self.waveforms[mode_name].tx_incident_power
        rx_noise_db = self.waveforms[mode_name].rx_noise_db
        rx_gain_db = self.waveforms[mode_name].rx_gain_db

        if tx_incident_power != 1.0:
            pem_api_manager.isOK(self.pem.setRadarModeTxIncidentPower(self.modes[mode_name],tx_incident_power))
        if rx_noise_db:
            pem_api_manager.isOK(self.pem.setRadarModeRxThermalNoise(self.modes[mode_name],True,rx_noise_db))
        if rx_gain_db:
            rx_gain_type = RssPy.RxChannelGainSpecType.USER_DEFINED
            pem_api_manager.isOK(self.pem.setRadarModeRxChannelGain(self.modes[mode_name],rx_gain_type,rx_gain_db))

        if mode.lower() == 'pulseddoppler':
            pem_api_manager.isOK(self.pem.setPulseDopplerWaveformSysSpecs(self.modes[mode_name],
                                                              center_freq,
                                                              bandwidth,
                                                              num_freq_samples,
                                                              pulse_interval,
                                                              num_pulse_cpi,
                                                              tx_multiplex))
        else:
            if mode.lower() == 'fmcw':
                adc_samples = self.waveforms[mode_name].adc_sample_rate
                chirpType = RssPy.FmcwChirpType.ASCENDING_RAMP
                pem_api_manager.isOK(self.pem.setChirpSequenceFMCWFromSysSpecs(self.modes[mode_name],
                                                                   chirpType,
                                                                   center_freq,
                                                                   bandwidth,
                                                                   adc_samples,
                                                                   num_freq_samples,
                                                                   pulse_interval,
                                                                   num_pulse_cpi,
                                                                   is_iq_channel,
                                                                   tx_multiplex
                                                                   ))

        output = self.waveforms[mode_name].output.lower().replace('_','')
        if output == 'rangedoppler' or output == 'dopplerrange':
            self.r_specs = self.waveforms[mode_name].r_specs
            self.d_specs = self.waveforms[mode_name].d_specs
            if len(self.antennas_tx)>0: # only do if this is a tx antenna type
                pem_api_manager.isOK(self.pem.activateRangeDopplerResponse(self.modes[mode_name],
                                                               self.range_pixels,
                                                               self.doppler_pixels,
                                                               self.center_vel,
                                                               self.r_specs,
                                                               self.d_specs))
        elif output == 'rangepulse' or output == 'pulserange':
            self.r_specs = self.waveforms[mode_name].r_specs
            self.d_specs = self.waveforms[mode_name].d_specs
            if len(self.antennas_tx)>0: # only do if this is a tx antenna type
                pem_api_manager.isOK(self.pem.activateRangePulseResponse(self.modes[mode_name],
                                                               self.range_pixels,
                                                               RssPy.ImagePixelReference.BEGIN, # range reference pixel
                                                               0,
                                                               self.r_specs))

    def set_mode_active(self, mode_name, status=True):
        """
        Activates or deactivates a radar mode based on the given mode name.
        Args:
            mode_name (str): The name of the radar mode to activate or deactivate.
            status (bool, optional): The activation status of the mode.
                Set to True to activate the mode, or False to deactivate it.
                Defaults to True.
        Returns:
            bool: True if the mode was successfully activated or deactivated,
            False if the mode name does not exist in the available modes.
        """

        if mode_name in self.modes.keys():
            pem_api_manager.isOK(self.pem.setRadarModeActive(self.modes[mode_name], status))
            self.active_mode_name = mode_name
            return True
        else:
            return False

    def add_antennas(self, mode_name=None, load_pattern_as_mesh=False, scale_pattern=1,antennas_dict=None):
        """
        Adds antennas to the radar device based on the provided configuration.
        Parameters:
            mode_name (str, optional): The name of the operational mode to use for the antennas.
                If not provided, the first mode in the device file or waveform dictionary is used.
            load_pattern_as_mesh (bool, optional): Whether to load the antenna pattern as a mesh for visualization.
                Defaults to False.
            scale_pattern (float, optional): Scaling factor for the antenna pattern mesh. Defaults to 1.
            antennas_dict (dict, optional): Dictionary containing antenna configurations.
                If not provided, the antenna configurations from the device file are used.
        Raises:
            FileNotFoundError: If the specified FFD file for an antenna is not found.
        Notes:
            - Supports three types of antennas: parametric, FFD/file-based, and planewave.
            - Configures antenna properties such as position, rotation, type, and operation mode.
            - Handles transmission (TX) and reception (RX) antennas separately.
            - Loads far-field data for antennas and optionally creates meshes for visualization.
            - Updates the coordinate system and actor properties for each antenna.
        Returns:
            None
        """


        if self.device_json is not None:
            if mode_name is None:
                mode_name = list(self.device_json['waveform'].keys())[0]
            if mode_name not in self.device_json['waveform'].keys():
                print(f"Mode {mode_name} not found in device file, using first mode found")
                mode_name = list(self.device_json['waveform'].keys())[0]
        else:
            if mode_name is None:
                mode_name = list(self.waveforms.keys())[0]
            if mode_name not in self.waveforms.keys():
                print(f"Mode {mode_name} not found in device file, using first mode found")
                mode_name = list(self.waveforms.keys())[0]

        if antennas_dict is None:
            antennas_dict = self.device_json['antenna']
        else:
            antennas_dict = self._lowercase(antennas_dict)

        for ant in antennas_dict:
            # will be overwritten if ffd file is found, this is used only for visualization for parametric antennas
            ffd_file_path = 'parametric_beam_dummy.ffd'
            ffd_file_path = os.path.abspath(os.path.join(default_antenna_path, ffd_file_path))
            ant_name = ant
            ant_type = antennas_dict[ant]['type']
            ant_op_mode = antennas_dict[ant]['operation_mode']
            ant_pos = np.array([0, 0, 0])
            ant_rot = np.eye(3)
            if 'position' in antennas_dict[ant].keys():
                ant_pos = np.array(antennas_dict[ant]['position'])
            if 'rotation' in antennas_dict[ant].keys():
                ant_rot = np.array(antennas_dict[ant]['rotation'])
            h_ant = RssPy.RadarAntenna()
            if ant_type == 'parametric':
                hpbw_horiz_deg = float(antennas_dict[ant]['hpbwhorizdeg'])
                hpbw_vert_deg = float(antennas_dict[ant]['hpbwvertdeg'])
                ant_pol = antennas_dict[ant]['polarization']
                if ant_pol.lower() == 'vertical':
                    ant_pol = RssPy.AntennaPolarization.VERT
                else:
                    ant_pol = RssPy.AntennaPolarization.HORZ
                pem_api_manager.isOK(
                    self.pem.addRadarAntennaParametricBeam(h_ant, self.h_device, ant_pol, hpbw_vert_deg, hpbw_horiz_deg, 1.))

            elif ant_type == 'ffd' or ant_type == 'file':

                if self.path_of_antenna_device is not None:
                    ffd_file_path = os.path.abspath(
                        os.path.join(self.path_of_antenna_device, antennas_dict[ant]['file_path']))
                else:
                    ffd_file_path = os.path.abspath(antennas_dict[ant]['file_path'])
                if os.path.exists(ffd_file_path):
                    fftbl = self.pem.loadFarFieldTable(ffd_file_path)
                    pem_api_manager.isOK(self.pem.addRadarAntennaFromTable(h_ant, self.h_device, fftbl))
                else:
                    raise FileNotFoundError(f"FFD file not found for {ffd_file_path}")

            elif ant_type.lower() == 'planewave':
                ant_pol = antennas_dict[ant]['polarization']
                if ant_pol.lower() == 'vertical':
                    ant_pol = RssPy.AntennaPolarization.VERT
                elif ant_pol.lower() == 'rhcp':
                    ant_pol = RssPy.AntennaPolarization.RHCP
                elif ant_pol.lower() == 'lhcp':
                    ant_pol = RssPy.AntennaPolarization.LHCP
                else:
                    ant_pol = RssPy.AntennaPolarization.HORZ
                power = self.waveforms[mode_name].tx_incident_power
                pem_api_manager.isOK(self.pem.addPlaneWaveIllumination(h_ant, self.h_device, ant_pol,power))


            # set locaiton of antenna with respect to the device
            pem_api_manager.isOK(self.pem.setCoordSysInParent(h_ant,
                                                       np.ascontiguousarray(ant_rot, dtype=np.float64),
                                                       np.ascontiguousarray(ant_pos, dtype=np.float64),
                                                       np.ascontiguousarray(np.zeros(3), dtype=np.float64),
                                                       np.ascontiguousarray(np.zeros(3), dtype=np.float64)))

            temp_dict = {'handle': h_ant,
                         'type': ant_type,
                         'operation_mode': ant_op_mode,
                         'position': ant_pos,
                         'rotation': ant_rot,
                         'ffd_file_path': ffd_file_path}
            self.all_antennas_properties[ant_name] = temp_dict
            if ant_op_mode.lower() == 'tx':
                pem_api_manager.isOK(self.pem.addTxAntenna(self.modes[mode_name], h_ant))
                if int(pem_api_manager.version) >= 252:
                    if self.fov !=180.0:
                        if self.fov != 360.0:
                            print("WARNING: FOV can only be 180 or 360")
                        self.pem.setAntennaFieldOfView(h_ant,360.0,(1,0,0))

                self.antennas_tx[ant_name] = h_ant
            else:
                pem_api_manager.isOK(self.pem.addRxAntenna(self.modes[mode_name], h_ant))
                self.antennas_rx[ant_name] = h_ant

        dict_for_loading_ffds = {}
        location_dict = {}  # build a dictionary of points can be used for beamforming
        for each in self.all_antennas_properties.keys():
            dict_for_loading_ffds[each] = self.all_antennas_properties[each]['ffd_file_path']
            location_dict[each] = self.all_antennas_properties[each]['position']
        ff_data = FarFields(location_dict=location_dict)
        # load all the ffd files into ff_data.meshes dictionary

        ff_data.read_ffd(dict_for_loading_ffds,
                        create_farfield_mesh=load_pattern_as_mesh,
                        scale_pattern=scale_pattern,
                        name=self.name)
        ff_mesh = None
        for port in ff_data.all_port_names:
            if load_pattern_as_mesh:
                self.all_antennas_properties[port]['mesh'] = ff_data.ff_meshes[port]
                ff_mesh = ff_data.ff_meshes[port]
            else:
                self.all_antennas_properties[port]['mesh'] = None
            coord_sys = CoordSys(h_node=self.all_antennas_properties[port]['handle'], parent_h_node=self.h_device)
            coord_sys.pos = self.all_antennas_properties[port]['position']
            coord_sys.rot = self.all_antennas_properties[port]['rotation']
            coord_sys.update()
            actor_name = f"{port}_actor"
            self.all_antennas_properties[port]['Actor'] = Actor(name=actor_name,
                                                                mesh=ff_mesh,
                                                                parent_h_node=self.parent_h_node,
                                                                coord_sys=coord_sys,
                                                                is_antenna=True)
            self.all_antennas_properties[port]['Fields'] = ff_data.data_dict[port]

        self.all_antennas_properties[port]['Fields'] = ff_data.data_dict[port]
        # all_antennas_properties[each]['mesh']

        if self.all_actors is not None:
            for each in self.all_antennas_properties:
                name = self.all_actors.add_actor(name=each, actor=self.all_antennas_properties[each]['Actor'])


class AntennaArray:
    """
    AntennaArray Class
    This class represents an antenna array configuration for radar systems. It provides functionality to initialize
    antenna arrays, configure radar modes, and populate antenna elements based on array size and shape.
    Attributes:
        name (str): Name of the antenna array. Default is 'array'.
        waveform (object): Waveform object used for radar signal processing. Must be provided.
        mode_name (str): Name of the radar mode. Default is 'default'.
        file_name (str or None): File name for antenna pattern data. If None, parametric antennas are used.
        beamwidth_H (float): Horizontal beamwidth of the antenna in degrees. Default is 140.
        beamwidth_V (float): Vertical beamwidth of the antenna in degrees. Default is 120.
        polarization (str): Polarization of the antenna ('V' for vertical, 'H' for horizontal). Default is 'V'.
        rx_shape (list): Shape of the receiver antenna array as [rows, columns]. Default is [1, 1].
        tx_shape (list): Shape of the transmitter antenna array as [rows, columns]. Default is [1, 1].
        spacing_wl_x (float): Spacing between antenna elements in the X direction, in wavelengths. Default is 0.5.
        spacing_wl_y (float): Spacing between antenna elements in the Y direction, in wavelengths. Default is 0.5.
        load_pattern_as_mesh (bool): Whether to load antenna patterns as mesh. Default is False.
        scale_pattern (float): Scaling factor for antenna patterns. Default is 1.
        parent_h_node (object or None): Parent node for hierarchical device configuration. Default is None.
        normal (str): Normal vector of the array ('x', 'y', or 'z'). Default is 'z'.
        range_pixels (int): Number of range pixels for radar processing. Default is 256.
        doppler_pixels (int): Number of Doppler pixels for radar processing. Default is 128.
        all_actors (object or None): Scene actors for visualization. Default is None.
    Methods:
        __init__(self, name, waveform, mode_name, file_name, beamwidth_H, beamwidth_V, polarization, rx_shape,
                 tx_shape, spacing_wl_x, spacing_wl_y, load_pattern_as_mesh, scale_pattern, parent_h_node,
                 normal, range_pixels, doppler_pixels, all_actors):
            Initializes the AntennaArray object with the provided parameters.
        _loop_positions(self, shape, operation_mode):
            Generates antenna element positions based on the array shape and operation mode ('tx' or 'rx').
    Usage:
        This class is used to configure and manage antenna arrays for radar systems. It supports both parametric
        and file-based antenna patterns, and allows customization of array shape, spacing, polarization, and
        beamwidths.
    """

    def __init__(self, name='array',
                 waveform=None,
                 mode_name='default',
                 file_name=None,
                 beamwidth_H=140,
                 beamwidth_V=120,
                 polarization='V',
                 planewave=False,
                 rx_shape=[1,1],
                 tx_shape=[1,1],
                 spacing_wl_x=0.5,
                 spacing_wl_y=0.5,
                 array_elements_centered=False,
                 load_pattern_as_mesh=False,
                 scale_pattern=1,
                 parent_h_node=None,
                 normal='z',
                 range_pixels = 256,
                 doppler_pixels = 128,
                 all_actors=None):

        # I added normal later, zo spacing_wl_x and spacing_wl_y are only consistent with normal='z'
        # this is the normal vector of the array.
        # else:
        # normal = 'x' means spacing_wl_x is actually the spacing in Z and spacing_wl_y is still Y
        # normal = 'y' means spacing_wl_x is still the spacing in X and spacing_wl_y is actuall Z
        self.normal = normal
        if waveform is None:
            raise ValueError("Waveform must be provided")

        # check shape of rx_shape and tx_shape, if they are not lists, convert them to lists
        if isinstance(rx_shape, int) or isinstance(rx_shape, float):
            rx_shape = [rx_shape, 1]
        if isinstance(tx_shape, int) or isinstance(tx_shape, float):
            tx_shape = [tx_shape, 1]


        if file_name is None:
            parametric = True
            if planewave:
                parametric = False
        else:
            parametric = False

        polarization = _parse_polarization(polarization)
        # polarization is used for parametric antennas or planewave, or if x,y or z is defined along with the dipole antenna
        # it will be used for that orientation

        # if we want to center the array so the elements are +/- around the origin
        self.array_elements_centered = array_elements_centered

        self.wavelength = 299792458.0/waveform.center_freq
        self.spacing_meter_x = spacing_wl_x*self.wavelength
        self.spacing_meter_y = spacing_wl_y*self.wavelength

        # can be parametric or have an ffd defined
        self.parametric = parametric
        self.file_name = file_name
        self.planewave = planewave

        self.rx_shape = rx_shape
        self.tx_shape = tx_shape

        # used for parametric beam, or if dipole it can be used to orient xyz
        self.polarization = polarization
        self.beamwidth_H = beamwidth_H
        self.beamwidth_V = beamwidth_V

        self.name = name
        self.waveform = waveform

        self.range_pixels = range_pixels
        self.doppler_pixels = doppler_pixels

        # empty antenna device that will be populated with antenna patterns
        self.antenna_device = AntennaDevice(file_name=None, parent_h_node=parent_h_node)


        self.antenna_device.name = name
        self.antenna_device.initialize_device()
        self.antenna_device.range_pixels = range_pixels
        self.antenna_device.doppler_pixels = doppler_pixels
        self.antenna_device.waveforms[mode_name] = waveform

        # configure radar mode
        h_mode = RssPy.RadarMode()
        self.antenna_device.modes[mode_name] = h_mode
        pem_api_manager.isOK(pem.addRadarMode(h_mode, self.antenna_device.h_device))

        # populate dictionary of all antenna elements based on array size
        ant_dict = {}
        if self.rx_shape[0] > 0 and self.rx_shape[1] > 0:
            ant_dict.update(self._loop_positions(self.rx_shape, operation_mode='rx'))
        if self.tx_shape[0] > 0 and self.tx_shape[1] > 0:
            ant_dict.update(self._loop_positions(self.tx_shape, operation_mode='tx'))

        self.antenna_device.add_antennas(mode_name=mode_name,
                                         load_pattern_as_mesh=load_pattern_as_mesh,
                                         scale_pattern=scale_pattern,
                                         antennas_dict=ant_dict)
        self.antenna_device.set_mode_active(mode_name)
        self.antenna_device.add_mode(mode_name)

        # if we provide teh existing scene actors, this array will also be added to the actors object for visuzalization
        if all_actors is not None:
            for each in self.antenna_device.all_antennas_properties:
                name = all_actors.add_actor(name=each, actor=self.antenna_device.all_antennas_properties[each]['Actor'])

    def shift_antenna_positions(self,antenna_operation_mode_to_update='rx',pos_x=None,pos_y=None,pos_z=None,all_actors=None):
        """
        Shifts the positions of antennas in the array by a specified offset. Used for when we
        want to adjust only certain antennas in the array, for example, if we created a 1Tx by 1x128Rx array
        and we want ot keep the Tx constant but shift the Rx antennas by a certain offset. to simulation
        a sub portion of a larger array. This could be used to simulate a larger (1Tx by 128x128Rx) by simulation
        1Tx by 1x128Rx. Then moving the rx array up in position by a certain offset and appending all the results
        Args:
            antenna_operation_mode_to_update (str): The operation mode of antennas to update ('rx' or 'tx').
            position_offset (numpy.ndarray): The offset vector to shift the antenna positions.
        """


        if all_actors is not None:
            for each in self.antenna_device.all_antennas_properties:
                if self.antenna_device.all_antennas_properties[each]['operation_mode'].lower() == antenna_operation_mode_to_update.lower():
                    cur_pos1 = self.antenna_device.all_antennas_properties[each]['position']
                    cur_pos2 = all_actors.actors[each].coord_sys.pos
                    if pos_x is not None:
                        cur_pos1[0] = pos_x
                        cur_pos2[0] = pos_x
                    if pos_y is not None:
                        cur_pos1[1] = pos_y
                        cur_pos2[1] = pos_y
                    if pos_z is not None:
                        cur_pos1[2] = pos_z
                        cur_pos2[2] = pos_z

                    self.antenna_device.all_antennas_properties[each]['position'] = cur_pos1
                    all_actors.actors[each].coord_sys.pos = cur_pos2
                    all_actors.actors[each].coord_sys.update()



    def _loop_positions(self,shape,operation_mode='tx'):

        total_count = 0
        antennas_dict = {}

        pos_offset_x = 0
        pos_offset_y = 0
        if self.array_elements_centered:
            # if we want to center the array, we need to adjust the position of the antennas
            pos_offset_x = -0.5*(shape[0]-1)*self.spacing_meter_x
            pos_offset_y = -0.5*(shape[1]-1)*self.spacing_meter_y

        polarization_idx = 0 # this is the polarization for Tx antennas
        if operation_mode.lower() == 'rx':
            polarization_idx = 1
        for x in range(shape[0]):
            pos_x = x*self.spacing_meter_x+pos_offset_x
            for y in range(shape[1]):
                pos_y = y*self.spacing_meter_y+pos_offset_y
                if self.normal.lower() == 'x':
                    pos = [0.0, pos_y, pos_x]
                elif self.normal.lower() == 'y':
                    pos = [pos_x, 0.0, pos_y]
                else:
                    pos = [pos_x, pos_y, 0.0]



                if self.parametric:
                    rx_dict = {
                        "type": "parametric",
                        "operation_mode": operation_mode,
                        "polarization": self.polarization[polarization_idx],
                        "hpbwHorizDeg": self.beamwidth_H,
                        "hpbwVertDeg": self.beamwidth_V,
                        "position": pos
                        }
                elif self.planewave:
                    rx_dict = {
                        "type": "planewave",
                        "operation_mode": operation_mode,
                        "polarization": self.polarization[polarization_idx],
                        "position": pos
                    }
                else:
                    rotation = np.eye(3) # default polarizaiton, is veritical/Z oriented
                    if 'dipole.ffd' in self.file_name.lower():
                        if self.polarization[polarization_idx].lower() == 'x':
                            # 3x3 rotation matrix, where the z axis now becomes the x axis
                            rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
                        elif self.polarization[polarization_idx] == 'y':
                            # 3x3 rotation matrix, where the z axis now becomes the y axis
                            rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
                    rx_dict = {
                        "type": "ffd",
                        "operation_mode": operation_mode,
                        "file_path": self.file_name ,
                        "position": pos,
                        "rotation": rotation
                        }
                antennas_dict[f"{operation_mode}{total_count}"] = rx_dict
                total_count += 1
        return antennas_dict


def add_single_tx(all_actors,
                  waveform,
                  mode_name,
                  pos=np.zeros(3),rot=np.eye(3),lin=np.zeros(3),ang=np.zeros(3),
                  parent_h_node=None,
                  ffd_file=None,
                  beamwidth_H=140,
                  beamwidth_V=120,
                  range_pixels=512,
                  doppler_pixels=256,
                  scale_pattern=10,
                  polarization='V',
                  planewave=False,
                  load_pattern_as_mesh=True,
                  fov=360.0):

    """
    Adds a single transmitter (Tx) antenna device to the simulation environment for Radar simulation.
    This function initializes and configures a transmitter antenna device with specified parameters,
    including waveform, position, orientation, and antenna properties. The antenna device is added
    to the scene for visualization purposes and configured with the provided radar mode.
    Args:
        all_actors (object): An object responsible for managing actors in the scene.
        waveform (object): The waveform to be used by the antenna device.
        mode_name (str): The name of the radar mode to be added to the antenna device.
        pos (numpy.ndarray, optional): Position of the antenna device in the scene (default: np.zeros(3)).
        rot (numpy.ndarray, optional): Rotation matrix for the antenna device (default: np.eye(3)).
        lin (numpy.ndarray, optional): Linear velocity of the antenna device (default: np.zeros(3)).
        ang (numpy.ndarray, optional): Angular velocity of the antenna device (default: np.zeros(3)).
        parent_h_node (object, optional): Parent hierarchical node for the antenna device (default: None).
        ffd_file (str, optional): File path to the FFD (Far-Field Data) file for antenna pattern (default: None).
        beamwidth_H (float, optional): Horizontal beamwidth of the antenna in degrees (default: 140).
        beamwidth_V (float, optional): Vertical beamwidth of the antenna in degrees (default: 120).
        range_pixels (int, optional): Number of range pixels for the radar mode (default: 512).
        doppler_pixels (int, optional): Number of Doppler pixels for the radar mode (default: 256).
        scale_pattern (float, optional): Scaling factor for the antenna pattern visualization (default: 10).
        fov (float, optional): Field of view of the antenna device in degrees (default: 360.0).
    Returns:
        AntennaDevice: The configured transmitter antenna device object.
    Notes:
        - If `ffd_file` is provided, the antenna pattern is loaded from the specified FFD file.
        - If `ffd_file` is not provided, a parametric antenna pattern is used with the specified beamwidths.
        - The antenna device is added to the scene for visualization using the `all_actors` object.
    """
    polarization = _parse_polarization(polarization)
    # initialize the antenna device, one for Tx, one for Rx
    antenna_device_tx = AntennaDevice(parent_h_node=parent_h_node)
    antenna_device_tx.initialize_device()
    antenna_device_tx.range_pixels = range_pixels
    antenna_device_tx.doppler_pixels = doppler_pixels
    antenna_device_tx.waveforms[mode_name] = waveform
    antenna_device_tx.fov = fov

    h_mode = RssPy.RadarMode()
    antenna_device_tx.modes[mode_name] = h_mode
    pem_api_manager.isOK(pem.addRadarMode(h_mode, antenna_device_tx.h_device))
    antennas_dict = {}

    if ffd_file is not None:
        ant_type_tx = {
            "type": "ffd",
            "file_path": ffd_file,
            "operation_mode": "tx",
            "position": [0, 0, 0]
        }  # position is offset location from where antenna device is placed

    elif planewave:
        ant_type_tx = {
            "type": "planewave",
            "operation_mode": "tx",
            "polarization": polarization[0]
        }

    else:  # parameteric
        pol_tx = polarization[0]
        ant_type_tx = {
            "type": "parametric",
            "operation_mode": "tx",
            "polarization": pol_tx,
            "hpbwHorizDeg": beamwidth_H,
            "hpbwVertDeg": beamwidth_V,
            "position": [0, 0, 0]
        }

    antennas_dict["Tx"] = ant_type_tx
    antenna_device_tx.add_antennas(mode_name=mode_name,
                                   load_pattern_as_mesh=load_pattern_as_mesh,
                                   scale_pattern=scale_pattern,
                                   antennas_dict=antennas_dict)
    antenna_device_tx.set_mode_active(mode_name)
    antenna_device_tx.add_mode(mode_name)

    # position of each antenna device
    antenna_device_tx.coord_sys.pos = np.array(pos)
    antenna_device_tx.coord_sys.rot = np.array(rot)
    antenna_device_tx.coord_sys.lin = np.array(lin)
    antenna_device_tx.coord_sys.ang = np.array(ang)
    antenna_device_tx.coord_sys.update()

    # just for visualization purposes, we can add the antennas as actors to the scene.
    for each in antenna_device_tx.all_antennas_properties:
        if all_actors is not None:
            name = all_actors.add_actor(name=each, actor=antenna_device_tx.all_antennas_properties[each]['Actor'])

    return antenna_device_tx



def add_diversity_antenna_pair(all_actors,
                                waveform,
                                mode_name,
                                operation_mode='tx',
                                pos=np.zeros(3),rot=np.eye(3),lin=np.zeros(3),ang=np.zeros(3),
                                parent_h_node=None,
                                ffd_files=None,
                                beamwidth_H=140,
                                beamwidth_V=120,
                                range_pixels=512,
                                doppler_pixels=256,
                                scale_pattern=10,
                                polarization='VH',
                                spatial_diversity_offset=np.array([0.0,0.0,0.0]),
                                load_pattern_as_mesh=True,
                                fov=360.0):

    """
    Adds two transmitter with spatial and/or polarization diversity, antenna device to the simulation environment for Radar simulation.
    This function initializes and configures a transmitter antenna device with specified parameters,
    including waveform, position, orientation, and antenna properties. The antenna device is added
    to the scene for visualization purposes and configured with the provided radar mode.
    Args:
        all_actors (object): An object responsible for managing actors in the scene.
        waveform (object): The waveform to be used by the antenna device.
        mode_name (str): The name of the radar mode to be added to the antenna device.
        pos (numpy.ndarray, optional): Position of the antenna device in the scene (default: np.zeros(3)).
        rot (numpy.ndarray, optional): Rotation matrix for the antenna device (default: np.eye(3)).
        lin (numpy.ndarray, optional): Linear velocity of the antenna device (default: np.zeros(3)).
        ang (numpy.ndarray, optional): Angular velocity of the antenna device (default: np.zeros(3)).
        parent_h_node (object, optional): Parent hierarchical node for the antenna device (default: None).
        ffd_files (list, optional): List of 2 file path to the FFD (Far-Field Data) file for antenna pattern (default: None).
        beamwidth_H (float, optional): Horizontal beamwidth of the antenna in degrees (default: 140).
        beamwidth_V (float, optional): Vertical beamwidth of the antenna in degrees (default: 120).
        range_pixels (int, optional): Number of range pixels for the radar mode (default: 512).
        doppler_pixels (int, optional): Number of Doppler pixels for the radar mode (default: 256).
        scale_pattern (float, optional): Scaling factor for the antenna pattern visualization (default: 10).
        fov (float, optional): Field of view of the antenna device in degrees (default: 360.0).
    Returns:
        AntennaDevice: The configured transmitter antenna device object.
    Notes:
        - If `ffd_file` is provided, the antenna pattern is loaded from the specified FFD file.
        - If `ffd_file` is not provided, a parametric antenna pattern is used with the specified beamwidths.
        - The antenna device is added to the scene for visualization using the `all_actors` object.
    """
    if ffd_files is not None and not isinstance(ffd_files, list) and len(ffd_files)!=2:
        raise ValueError("ffd_files must be a list of 2 file paths for add_diversity_antenna_pair()")

    polarization = _parse_polarization(polarization)
    # initialize the antenna device, one for Tx, one for Rx
    antenna_device = AntennaDevice(parent_h_node=parent_h_node)
    antenna_device.initialize_device()
    antenna_device.range_pixels = range_pixels
    antenna_device.doppler_pixels = doppler_pixels
    antenna_device.waveforms[mode_name] = waveform
    antenna_device.fov = fov

    h_mode = RssPy.RadarMode()
    antenna_device.modes[mode_name] = h_mode
    pem_api_manager.isOK(pem.addRadarMode(h_mode, antenna_device.h_device))
    antennas_dict = {}

    pol_tx = polarization[0]
    theta_rot = 0
    if pol_tx == 'HORIZONTAL':
        theta_rot=90
    pol_tx2 = polarization[1]
    theta_rot = 0
    if pol_tx2 == 'HORIZONTAL':
        theta_rot2=90

    if ffd_files is not None:
        ant_type_1 = {
            "type": "ffd",
            "file_path": ffd_files[0],
            "operation_mode": operation_mode,
            "position": [0, 0, 0]
        }  # position is offset location from where antenna device is placed
        ant_type_2 = {
            "type": "ffd",
            "file_path": ffd_files[1],
            "operation_mode": operation_mode,
            "position": spatial_diversity_offset
        }  # position is offset location from where antenna device is placed
    else:  # parameteric
        pol_tx = polarization[0]
        ant_type_1 = {
            "type": "parametric",
            "operation_mode": operation_mode,
            "polarization": pol_tx,
            "hpbwHorizDeg": beamwidth_H,
            "hpbwVertDeg": beamwidth_V,
            "position": [0, 0, 0]
        }
        ant_type_2 = {
            "type": "parametric",
            "operation_mode": operation_mode,
            "polarization": pol_tx,
            "hpbwHorizDeg": beamwidth_H,
            "hpbwVertDeg": beamwidth_V,
            "position": spatial_diversity_offset
        }

    name = f"{operation_mode}1"
    antennas_dict[name] = ant_type_1
    name = f"{operation_mode}2"
    antennas_dict[name] = ant_type_2

    antenna_device.add_antennas(mode_name=mode_name,
                                   load_pattern_as_mesh=load_pattern_as_mesh,
                                   scale_pattern=scale_pattern,
                                   antennas_dict=antennas_dict)
    antenna_device.set_mode_active(mode_name)
    antenna_device.add_mode(mode_name)

    # position of each antenna device
    antenna_device.coord_sys.pos = np.array(pos)
    antenna_device.coord_sys.rot = np.array(rot)
    antenna_device.coord_sys.lin = np.array(lin)
    antenna_device.coord_sys.ang = np.array(ang)
    antenna_device.coord_sys.update()

    # just for visualization purposes, we can add the antennas as actors to the scene.
    for each in antenna_device.all_antennas_properties:
        if all_actors is not None:
            name = all_actors.add_actor(name=each, actor=antenna_device.all_antennas_properties[each]['Actor'])

    return antenna_device



def add_single_rx(all_actors,
                  waveform,
                  mode_name,
                  pos=np.zeros(3), rot=np.eye(3), lin=np.zeros(3), ang=np.zeros(3),
                  parent_h_node=None,
                  ffd_file=None,
                  beamwidth_H=140,
                  beamwidth_V=120,
                  range_pixels=512,
                  doppler_pixels=256,
                  polarization='V',
                  load_pattern_as_mesh=True,
                  scale_pattern=10):
    """
    Adds a single receiver (Rx) antenna device to the simulation environment for Radar simulation.
    This function initializes and configures a receiver antenna device with specified parameters,
    including waveform, position, orientation, and antenna properties. The antenna device is added
    to the scene for visualization purposes and configured with the provided radar mode.
    Args:
        all_actors (object): An object responsible for managing actors in the scene.
        waveform (object): The waveform to be used by the antenna device.
        mode_name (str): The name of the radar mode to be added to the antenna device.
        pos (numpy.ndarray, optional): Position of the antenna device in the scene (default: np.zeros(3)).
        rot (numpy.ndarray, optional): Rotation matrix for the antenna device (default: np.eye(3)).
        lin (numpy.ndarray, optional): Linear velocity of the antenna device (default: np.zeros(3)).
        ang (numpy.ndarray, optional): Angular velocity of the antenna device (default: np.zeros(3)).
        parent_h_node (object, optional): Parent hierarchical node for the antenna device (default: None).
        ffd_file (str, optional): File path to the FFD (Far-Field Data) file for antenna pattern (default: None).
        beamwidth_H (float, optional): Horizontal beamwidth of the antenna in degrees (default: 140).
        beamwidth_V (float, optional): Vertical beamwidth of the antenna in degrees (default: 120).
        range_pixels (int, optional): Number of range pixels for the radar mode (default: 512).
        doppler_pixels (int, optional): Number of Doppler pixels for the radar mode (default: 256).
        scale_pattern (float, optional): Scaling factor for the antenna pattern visualization (default: 10).
        fov (float, optional): Field of view of the antenna device in degrees (default: 360.0).
    Returns:
        AntennaDevice: The configured transmitter antenna device object.
    Notes:
        - If `ffd_file` is provided, the antenna pattern is loaded from the specified FFD file.
        - If `ffd_file` is not provided, a parametric antenna pattern is used with the specified beamwidths.
        - The antenna device is added to the scene for visualization using the `all_actors` object.
    """
    polarization = _parse_polarization(polarization)
    # initialize the antenna device, one for Tx, one for Rx
    antenna_device_rx = AntennaDevice(parent_h_node=parent_h_node)
    antenna_device_rx.initialize_device()
    antenna_device_rx.range_pixels = range_pixels
    antenna_device_rx.doppler_pixels = doppler_pixels
    antenna_device_rx.waveforms[mode_name] = waveform

    h_mode = RssPy.RadarMode()
    antenna_device_rx.modes[mode_name] = h_mode
    pem_api_manager.isOK(pem.addRadarMode(h_mode, antenna_device_rx.h_device))
    antennas_dict = {}
    if ffd_file is not None:
        ant_type_rx = {
            "type": "ffd",
            "file_path": ffd_file,
            "operation_mode": "rx",
            "position": [0, 0, 0]
        }  # position is offset location from where antenna device is placed
    else:  # parameteric
        ant_type_rx = {
            "type": "parametric",
            "operation_mode": "rx",
            "polarization": polarization[0],
            "hpbwHorizDeg": beamwidth_H,
            "hpbwVertDeg": beamwidth_V,
            "position": [0, 0, 0]
        }

    antennas_dict["Rx"] = ant_type_rx
    antenna_device_rx.add_antennas(mode_name=mode_name,
                                   load_pattern_as_mesh=load_pattern_as_mesh,
                                   scale_pattern=scale_pattern,
                                   antennas_dict=antennas_dict)
    antenna_device_rx.set_mode_active(mode_name)
    antenna_device_rx.add_mode(mode_name)

    # position of each antenna device
    antenna_device_rx.coord_sys.pos = np.array(pos)
    antenna_device_rx.coord_sys.rot = np.array(rot)
    antenna_device_rx.coord_sys.lin = np.array(lin)
    antenna_device_rx.coord_sys.ang = np.array(ang)
    antenna_device_rx.coord_sys.update()

    # just for visualization purposes, we can add the antennas as actors to the scene.
    for each in antenna_device_rx.all_antennas_properties:
        if all_actors is not None:
            name = all_actors.add_actor(name=each, actor=antenna_device_rx.all_antennas_properties[each]['Actor'])

    return antenna_device_rx

def _parse_polarization(polarization):
    # parse a string value that may be an combination of two of the strings from the list ['V', 'H', 'RHCP', 'LHCP'].
    # if the string is a single character, it will be used for both Tx and Rx

    if not isinstance(polarization, str):
        raise ValueError(f"Polarization must be a string, got {type(polarization)} instead.")

    if len(polarization) == 1:
        if polarization.upper() == 'V':
            return ['VERTICAL', 'VERTICAL']
        elif polarization.upper() == 'H':
            return ['HORIZONTAL', 'HORIZONTAL']
        elif polarization.upper() == 'R':
            return ['RHCP', 'RHCP']
        elif polarization.upper() == 'L':
            return ['LHCP', 'LHCP']
        elif polarization.upper() == 'X':
            return ['X', 'X']
        elif polarization.upper() == 'Y':
            return ['Y', 'Y']
        elif polarization.upper() == 'Z':
            return ['Z', 'Z']
    elif len(polarization) == 2:
        if polarization[0].upper() == 'V':
            pol_tx = 'VERTICAL'
        elif polarization[0].upper() == 'H':
            pol_tx = 'HORIZONTAL'
        elif polarization[0].upper() == 'X':
            pol_tx = 'X'
        elif polarization[0].upper() == 'Y':
            pol_tx = 'Y'
        elif polarization[0].upper() == 'Z':
            pol_tx = 'Z'
        else:
            raise ValueError(f"Invalid Tx polarization: {polarization[0]}")

        if polarization[1].upper() == 'V':
            pol_rx = 'VERTICAL'
        elif polarization[1].upper() == 'H':
            pol_rx = 'HORIZONTAL'
        elif polarization[1].upper() == 'X':
            pol_rx = 'X'
        elif polarization[1].upper() == 'Y':
            pol_rx = 'Y'
        elif polarization[1].upper() == 'Z':
            pol_rx = 'Z'
        else:
            raise ValueError(f"Invalid Rx polarization: {polarization[1]}")
        return [pol_tx, pol_rx]
    elif len(polarization) == 4:
        if polarization[0:2].upper() == 'RH':
            pol_tx = 'RHCP'
            pol_rx = 'RHCP'
        elif polarization[0:2].upper() == 'LH':
            pol_tx = 'LHCP'
            pol_rx = 'LHCP'
        else:
            raise ValueError(f"Invalid polarization: {polarization}")

        return [pol_tx, pol_rx]
    elif len(polarization) == 5:
        if polarization[0:2].upper() == 'RH':
            pol_tx = 'RHCP'
        elif polarization[0:2].upper() == 'LH':
            pol_tx = 'LHCP'
        elif polarization[0].upper() == 'V':
            pol_tx = 'VERTICAL'
        elif polarization[0].upper() == 'H':
            pol_tx = 'HORIZONTAL'
        elif polarization[0].upper() == 'X':
            pol_tx = 'X'
        elif polarization[0].upper() == 'Y':
            pol_tx = 'Y'
        elif polarization[0].upper() == 'Z':
            pol_tx = 'Z'
        else:
            raise ValueError(f"Invalid Tx polarization: {polarization}")

        if polarization[1:5].upper() == 'RH':
            pol_rx = 'RHCP'
        elif polarization[1:5].upper() == 'LH':
            pol_rx = 'LHCP'
        elif polarization[-1].upper() == 'V':
            pol_rx = 'VERTICAL'
        elif polarization[-1].upper() == 'H':
            pol_rx = 'HORIZONTAL'
        elif polarization[-1].upper() == 'X':
            pol_rx = 'X'
        elif polarization[-1].upper() == 'Y':
            pol_rx = 'Y'
        elif polarization[-1].upper() == 'Z':
            pol_rx = 'Z'
        else:
            raise ValueError(f"Invalid Rx polarization: {polarization}")
        return [pol_tx, pol_rx]
    elif len(polarization) == 8:
        if polarization[0:2].upper() == 'RH':
            pol_tx = 'RHCP'
        elif polarization[0:2].upper() == 'LH':
            pol_tx = 'LHCP'
        else:
            raise ValueError(f"Invalid Tx polarization: {polarization}")
        if polarization[4:6].upper() == 'RH':
            pol_rx = 'RHCP'
        elif polarization[4:6].upper() == 'LH':
            pol_rx = 'LHCP'
        else:
            raise ValueError(f"Invalid Rx polarization: {polarization}")
        return [pol_tx, pol_rx]
    else:
        raise ValueError(f"Invalid polarization string: {polarization}. It should be one of the following: "
                         "'V', 'H', 'RHCPRHCP', 'VV', 'HH', 'RHCPVV', 'LHCPVV', etc.")


def add_single_tx_rx(all_actors,
                      waveform,
                      mode_name,
                      pos=np.zeros(3),rot=np.eye(3),lin=np.zeros(3),ang=np.zeros(3),
                      parent_h_node=None,
                      ffd_file=None,
                      planewave=False,
                      beamwidth_H=140,
                      beamwidth_V=120,
                      polarization='VV',
                      range_pixels=512,
                      doppler_pixels=256,
                      scale_pattern=10,
                      load_pattern_as_mesh=True,
                      r_specs='hann,50',
                      d_specs='hann,50'):

    """
    Adds a single transmit (Tx) and receive (Rx) antenna device to the scene for P2P simulation.
    Coupling between tx and rx is automatically configured.
    This function initializes an antenna device, configures its properties, and adds it to the scene.
    It supports different antenna types including parametric, planewave, and file-based (FFD).
    The function also allows customization of antenna parameters such as beamwidth, polarization, and waveform specifications.
    Args:
        all_actors (object): The scene object to which antenna actors will be added for visualization.
        waveform (object): The waveform object containing radar signal specifications.
        mode_name (str): The name of the radar mode to be added.
        pos (np.ndarray, optional): Position of the antenna device in the scene (default: np.zeros(3)).
        rot (np.ndarray, optional): Rotation matrix for the antenna device (default: np.eye(3)).
        lin (np.ndarray, optional): Linear velocity of the antenna device (default: np.zeros(3)).
        ang (np.ndarray, optional): Angular velocity of the antenna device (default: np.zeros(3)).
        parent_h_node (object, optional): Parent hierarchical node for the antenna device (default: None).
        ffd_file (str, optional): Path to the FFD file for file-based antenna configuration (default: None).
        planewave (bool, optional): If True, configures the antenna as a planewave type (default: False).
        beamwidth_H (float, optional): Horizontal beamwidth in degrees for parametric antennas (default: 140).
        beamwidth_V (float, optional): Vertical beamwidth in degrees for parametric antennas (default: 120).
        polarization (str, optional): Polarization type for the antenna (e.g., 'VV', 'HH', 'RHCP', 'LHCP') (default: 'VV').
        range_pixels (int, optional): Number of range pixels for the radar device (default: 512).
        doppler_pixels (int, optional): Number of Doppler pixels for the radar device (default: 256).
        scale_pattern (float, optional): Scaling factor for the antenna pattern visualization (default: 10).
        r_specs (str, optional): Range specifications for the waveform (default: 'hann,50').
        d_specs (str, optional): Doppler specifications for the waveform (default: 'hann,50').
    Returns:
        AntennaDevice: The configured antenna device object with Tx and Rx properties.
    Notes:
        - The function supports three types of antennas: parametric, planewave, and file-based (FFD).
        - Polarization can be specified as a single string (e.g., 'VV') or concatenated (e.g., 'RHCPVV').
        - The antenna device is added to the scene for visualization purposes.
    """

    # dictionary defining polarization

    # initialize the antenna device, one for Tx, one for Rx
    antenna_device_tx_rx = AntennaDevice(parent_h_node=parent_h_node)
    antenna_device_tx_rx.initialize_device()
    antenna_device_tx_rx.range_pixels = range_pixels
    antenna_device_tx_rx.doppler_pixels = doppler_pixels
    waveform.r_specs = r_specs
    waveform.d_specs = d_specs
    antenna_device_tx_rx.waveforms[mode_name] = waveform

    polarization = _parse_polarization(polarization)
    h_mode = RssPy.RadarMode()
    antenna_device_tx_rx.modes[mode_name] = h_mode
    pem_api_manager.isOK(pem.addRadarMode(h_mode, antenna_device_tx_rx.h_device))
    antennas_dict = {}
    if ffd_file is not None:
        ant_type_tx = {
            "type": "ffd",
            "file_path": ffd_file,
            "operation_mode": "tx",
            "position": [0, 0, 0]
        }  # position is offset location from where antenna device is placed
        ant_type_rx = {
            "type": "ffd",
            "file_path": ffd_file,
            "operation_mode": "rx",
            "position": [0, 0, 0]
        }  # position is offset location from where antenna device is placed
    elif planewave:
        ant_type_tx = {
            "type": "planewave",
            "operation_mode": "tx",
            "polarization": polarization[0]
        }
        ant_type_rx = {
            "type": "planewave",
            "operation_mode": "rx",
            "polarization": polarization[1]
        }
    else:  # parameteric
        pol_tx = polarization[0]
        pol_rx = polarization[1]



        ant_type_tx = {
            "type": "parametric",
            "operation_mode": "tx",
            "polarization": pol_tx,
            "hpbwHorizDeg": beamwidth_H,
            "hpbwVertDeg": beamwidth_V,
            "position": [0, 0, 0]
        }
        ant_type_rx = {
            "type": "parametric",
            "operation_mode": "rx",
            "polarization": pol_rx,
            "hpbwHorizDeg": beamwidth_H,
            "hpbwVertDeg": beamwidth_V,
            "position": [0, 0, 0]
        }

    antennas_dict["Tx"] = ant_type_tx
    antennas_dict["Rx"] = ant_type_rx
    antenna_device_tx_rx.add_antennas(mode_name=mode_name,
                                   load_pattern_as_mesh=load_pattern_as_mesh,
                                   scale_pattern=scale_pattern,
                                   antennas_dict=antennas_dict)
    antenna_device_tx_rx.set_mode_active(mode_name)
    antenna_device_tx_rx.add_mode(mode_name)

    # position of each antenna device
    antenna_device_tx_rx.coord_sys.pos = np.array(pos)
    antenna_device_tx_rx.coord_sys.rot = np.array(rot)
    antenna_device_tx_rx.coord_sys.lin = np.array(lin)
    antenna_device_tx_rx.coord_sys.ang = np.array(ang)
    antenna_device_tx_rx.coord_sys.update()

    # just for visualization purposes, we can add the antennas as actors to the scene.
    for each in antenna_device_tx_rx.all_antennas_properties:
        if all_actors is not None:
            name = all_actors.add_actor(name=each, actor=antenna_device_tx_rx.all_antennas_properties[each]['Actor'])

    return antenna_device_tx_rx

def add_plane_wave(all_actors,
                      waveform,
                      mode_name,
                      pos=np.zeros(3),rot=np.eye(3),lin=np.zeros(3),ang=np.zeros(3),
                      parent_h_node=None,
                      polarization='VV',
                      range_pixels=512,
                      doppler_pixels=256,
                      scale_pattern=1,
                      r_specs='hann,50',
                      d_specs='hann,50',
                      load_pattern_as_mesh=True):

    """
    """

    pol = _parse_polarization(polarization)


    # initialize the antenna device, one for Tx, one for Rx
    antenna_device_tx_rx = AntennaDevice(parent_h_node=parent_h_node)
    antenna_device_tx_rx.initialize_device()
    antenna_device_tx_rx.range_pixels = range_pixels
    antenna_device_tx_rx.doppler_pixels = doppler_pixels
    waveform.r_specs = r_specs
    waveform.d_specs = d_specs
    antenna_device_tx_rx.waveforms[mode_name] = waveform


    h_mode = RssPy.RadarMode()
    antenna_device_tx_rx.modes[mode_name] = h_mode
    pem_api_manager.isOK(pem.addRadarMode(h_mode, antenna_device_tx_rx.h_device))
    antennas_dict = {}

    ant_type_tx = {
        "type": "planewave",
        "operation_mode": "tx",
        "polarization": pol[0].lower()
    }
    ant_type_rx = {
        "type": "planewave",
        "operation_mode": "rx",
        "polarization": pol[1].lower()
    }


    antennas_dict["Tx"] = ant_type_tx
    antennas_dict["Rx"] = ant_type_rx
    antenna_device_tx_rx.add_antennas(mode_name=mode_name,
                                   load_pattern_as_mesh=load_pattern_as_mesh,
                                   scale_pattern=scale_pattern,
                                   antennas_dict=antennas_dict)
    antenna_device_tx_rx.set_mode_active(mode_name)
    antenna_device_tx_rx.add_mode(mode_name)

    # position of each antenna device
    antenna_device_tx_rx.coord_sys.pos = np.array(pos)
    antenna_device_tx_rx.coord_sys.rot = np.array(rot)
    antenna_device_tx_rx.coord_sys.lin = np.array(lin)
    antenna_device_tx_rx.coord_sys.ang = np.array(ang)
    antenna_device_tx_rx.coord_sys.update()

    # just for visualization purposes, we can add the antennas as actors to the scene.
    for each in antenna_device_tx_rx.all_antennas_properties:
        if all_actors is not None:
            name = all_actors.add_actor(name=each, actor=antenna_device_tx_rx.all_antennas_properties[each]['Actor'])

    return antenna_device_tx_rx


def add_plane_wave_bistatic_observers(all_actors,
                            waveform,
                            mode_name,
                            pos=np.zeros(3),rot=np.eye(3),lin=np.zeros(3),ang=np.zeros(3),
                            incident_orientations=[{'pos':np.zeros(3),'rot':np.eye(3),'lin':np.zeros(3),'ang':np.zeros(3)}],
                            observer_orientations=[{'pos':np.zeros(3),'rot':np.eye(3),'lin':np.zeros(3),'ang':np.zeros(3)}],
                            parent_h_node=None,
                            polarization='VV',
                            range_pixels=512,
                            doppler_pixels=256,
                            scale_pattern=1,
                            r_specs='hann,50',
                            d_specs='hann,50',
                            load_pattern_as_mesh=True):

    """
    """

    pol = _parse_polarization(polarization)


    # initialize the antenna device, one for Tx, one for Rx
    antenna_device_tx_rx = AntennaDevice(parent_h_node=parent_h_node)
    antenna_device_tx_rx.initialize_device()
    antenna_device_tx_rx.range_pixels = range_pixels
    antenna_device_tx_rx.doppler_pixels = doppler_pixels
    waveform.r_specs = r_specs
    waveform.d_specs = d_specs
    antenna_device_tx_rx.waveforms[mode_name] = waveform


    h_mode = RssPy.RadarMode()
    antenna_device_tx_rx.modes[mode_name] = h_mode
    pem_api_manager.isOK(pem.addRadarMode(h_mode, antenna_device_tx_rx.h_device))
    antennas_dict = {}

    print(f"Adding {len(incident_orientations)} incident wave to the scene.")
    for i in range(len(incident_orientations)):
        if i < len(incident_orientations):
            pos2 = incident_orientations[i]['pos']
            rot2 = incident_orientations[i]['rot']
            lin2 = incident_orientations[i]['lin']
            ang = incident_orientations[i]['ang']
        else:
            pos2 = np.zeros(3)
            rot2 = np.eye(3)
            lin2 = np.zeros(3)
            ang2 = np.zeros(3)

        # create a unique name for each incident antenna
        ant_name = f"Incident_{i}"
        antennas_dict[ant_name] = {
            "type": "planewave",
            "operation_mode": "tx",
            "polarization": pol[0].lower(),
            "position": pos2,
            "rotation": rot2
        }

    print(f"Adding {len(observer_orientations)} observers to the scene.")
    for i in range(len(observer_orientations)):
        if i < len(observer_orientations):
            pos1 = observer_orientations[i]['pos']
            rot1 = observer_orientations[i]['rot']
            lin1 = observer_orientations[i]['lin']
            ang1 = observer_orientations[i]['ang']
        else:
            pos1 = np.zeros(3)
            rot1 = np.eye(3)
            lin1 = np.zeros(3)
            ang1 = np.zeros(3)

        # create a unique name for each observer
        observer_name = f"Observer_{i}"
        antennas_dict[observer_name] = {
            "type": "planewave",
            "operation_mode": "rx",
            "polarization": pol[1].lower(),
            "position": pos1,
            "rotation": rot1
        }




    antenna_device_tx_rx.add_antennas(mode_name=mode_name,
                                   load_pattern_as_mesh=load_pattern_as_mesh,
                                   scale_pattern=scale_pattern,
                                   antennas_dict=antennas_dict)
    antenna_device_tx_rx.set_mode_active(mode_name)
    antenna_device_tx_rx.add_mode(mode_name)

    # position of each antenna device
    antenna_device_tx_rx.coord_sys.pos = np.array(pos)
    antenna_device_tx_rx.coord_sys.rot = np.array(rot)
    antenna_device_tx_rx.coord_sys.lin = np.array(lin)
    antenna_device_tx_rx.coord_sys.ang = np.array(ang)
    antenna_device_tx_rx.coord_sys.update()

    # just for visualization purposes, we can add the antennas as actors to the scene.
    for each in antenna_device_tx_rx.all_antennas_properties:
        if all_actors is not None:
            name = all_actors.add_actor(name=each, actor=antenna_device_tx_rx.all_antennas_properties[each]['Actor'])

    return antenna_device_tx_rx

def add_multi_channel_radar_az_el(all_actors,
                                  waveform,
                                  mode_name,
                                  num_rx_az=2,
                                  num_rx_el=0,
                                  spacing_wl = 0.5,
                                  pos=np.zeros(3),
                                  rot=np.eye(3),
                                  lin=np.zeros(3),
                                  ang=np.zeros(3),
                                  normal='x',
                                  parent_h_node=None,
                                  ffd_file=None,
                                  beamwidth_H=140,
                                  beamwidth_V=120,
                                  polarization='VV',
                                  range_pixels=512,
                                  doppler_pixels=256,
                                  load_pattern_as_mesh=True,
                                  scale_pattern=10):
    # create multi channel radar that has 1 Tx and N Rx antennas in Az and M Rx antennas in El

    spacing_m = spacing_wl*299792458.0/waveform.center_freq
    spacing_az = []
    spacing_el = []
    for i in range(num_rx_az):
        if normal.lower() == 'x':
            spacing_az.append([0.0,i*spacing_m, 0.0])
        elif normal.lower() == 'y':
            spacing_az.append([i*spacing_m,0.0, 0.0])
        else:
            spacing_az.append([i*spacing_m,0.0, 0.0])
    for i in range(num_rx_el):
        if normal.lower() == 'x':
            spacing_el.append([0.0, 0.0,i*spacing_m])
        elif normal.lower() == 'y':
            spacing_el.append([0.0, 0,0, i*spacing_m])
        else:
            spacing_el.append([0.0, i*spacing_m, 0.0])


    # dictionary defining polarization

    polarization = _parse_polarization(polarization)
    # initialize the antenna device, one for Tx, one for Rx
    antenna_device = AntennaDevice(parent_h_node=parent_h_node)
    antenna_device.initialize_device()
    antenna_device.range_pixels = range_pixels
    antenna_device.doppler_pixels = doppler_pixels
    antenna_device.waveforms[mode_name] = waveform

    h_mode = RssPy.RadarMode()
    antenna_device.modes[mode_name] = h_mode
    pem_api_manager.isOK(pem.addRadarMode(h_mode, antenna_device.h_device))
    antennas_dict = {}
    if ffd_file is not None:
        ant_type_tx = {
            "type": "ffd",
            "file_path": ffd_file,
            "operation_mode": "tx",
            "position": [0, 0, 0]
        }  # position is offset location from where antenna device is placed

        for i in range(num_rx_az):
            ant_type_rx = {
                "type": "ffd",
                "file_path": ffd_file,
                "operation_mode": "rx",
                "position": spacing_az[i]
            }
            antennas_dict[f"Rx{i}_az"] = ant_type_rx
        for i in range(num_rx_el):
            ant_type_rx = {
                "type": "ffd",
                "file_path": ffd_file,
                "operation_mode": "rx",
                "position": spacing_el[i]
            }
            antennas_dict[f"Rx{i}_el"] = ant_type_rx


    else:  # parameteric
        # only support VV, VH, HV, HH, or RHCPLHCP, LHCPRHCP
        pol_tx = polarization[0]
        pol_rx = polarization[1]



        ant_type_tx = {
            "type": "parametric",
            "operation_mode": "tx",
            "polarization": pol_tx,
            "hpbwHorizDeg": beamwidth_H,
            "hpbwVertDeg": beamwidth_V,
            "position": [0, 0, 0]
        }
        for i in range(num_rx_az):
            ant_type_rx = {
                "type": "parametric",
                "operation_mode": "rx",
                "polarization": pol_rx,
                "hpbwHorizDeg": beamwidth_H,
                "hpbwVertDeg": beamwidth_V,
                "position": spacing_az[i]
            }
            antennas_dict[f"Rx{i}_az"] = ant_type_rx
        for i in range(num_rx_el):
            ant_type_rx = {
                "type": "parametric",
                "operation_mode": "rx",
                "polarization": pol_rx,
                "hpbwHorizDeg": beamwidth_H,
                "hpbwVertDeg": beamwidth_V,
                "position": spacing_el[i]
            }
            antennas_dict[f"Rx{i}_el"] = ant_type_rx


    antennas_dict["Tx"] = ant_type_tx

    antenna_device.add_antennas(mode_name=mode_name,
                                   load_pattern_as_mesh=load_pattern_as_mesh,
                                   scale_pattern=scale_pattern,
                                   antennas_dict=antennas_dict)
    antenna_device.set_mode_active(mode_name)
    antenna_device.add_mode(mode_name)

    # position of each antenna device
    antenna_device.coord_sys.pos = np.array(pos)
    antenna_device.coord_sys.rot = np.array(rot)
    antenna_device.coord_sys.lin = np.array(lin)
    antenna_device.coord_sys.ang = np.array(ang)
    antenna_device.coord_sys.update()

    # just for visualization purposes, we can add the antennas as actors to the scene.
    for each in antenna_device.all_antennas_properties:
        if all_actors is not None:
            name = all_actors.add_actor(name=each, actor=antenna_device.all_antennas_properties[each]['Actor'])

    return antenna_device


def add_antenna_device_from_json(all_actors,
                                json_file,
                                mode_name=None,
                                pos=np.zeros(3),rot=np.eye(3),lin=np.zeros(3),ang=np.zeros(3),
                                parent_h_node=None,
                                scale_pattern=10,
                                load_pattern_as_mesh=True,
                                fov=360.0):


    if not os.path.exists(json_file):
        print(f"File {json_file} not found, searching antenna_device_library folder")
        if os.path.exists(os.path.join(default_antenna_path,json_file)):
            json_file = os.path.join(default_antenna_path,json_file)
            print(f"File {json_file} found")
        else:
            raise FileNotFoundError(f"File {json_file} not found. Provide full path or place in antenna_device_library folder")

    ant_device = AntennaDevice(json_file,
                               parent_h_node=parent_h_node,
                               all_actors=all_actors)

    if mode_name is None:
        mode_name = list(ant_device.device_json['waveform'].keys())[0]

    ant_device.initialize_mode(mode_name=mode_name)
    # put this antenna on the roof
    ant_device.coord_sys.pos = np.array(pos)
    ant_device.coord_sys.rot = np.array(rot)
    ant_device.coord_sys.lin = np.array(lin)
    ant_device.coord_sys.ang = np.array(ang)
    ant_device.coord_sys.update()
    ant_device.add_antennas(mode_name=mode_name,load_pattern_as_mesh=load_pattern_as_mesh,scale_pattern=scale_pattern)
    ant_device.add_mode(mode_name=mode_name)



    # just for visualization purposes, we can add the antennas as actors to the scene.
    for each in ant_device.all_antennas_properties:
        if all_actors is not None:
            name = all_actors.add_actor(name=each, actor=ant_device.all_antennas_properties[each]['Actor'])

    return ant_device


def enable_coupling(mode_name,tx_device, rx_device):
    # set coupling between antennas
    rComp = RssPy.ResponseComposition.INDIVIDUAL
    pem_api_manager.isOK(pem.setTxResponseComposition(tx_device.modes[mode_name], rComp))
    pem_api_manager.isOK(pem.setDoP2PCoupling(tx_device.h_node_platform, rx_device.h_node_platform, True))