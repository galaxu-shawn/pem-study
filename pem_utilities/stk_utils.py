"""
STK Utilities Module

This module provides utilities for interfacing with STK (Systems Tool Kit) software
and processing STK simulation results. It includes classes for extracting motion data
from STK scenarios and reading STK response files for channel characterization analysis.

Classes:
    STK_Utils: Extracts and manages object motion data from STK scenarios
    STK_Results_Reader: Reads and processes STK response (.rsp) files

Dependencies:
    - STK Python API (agi.stk12)
    - NumPy, SciPy for numerical operations
    - Matplotlib for plotting and animations
    - H5PY for reading HDF5 files
    - Tkinter for file dialogs

Author: ANSYS Inc.
"""

import tkinter as tk
from tkinter import filedialog
import os
import sys
import numpy as np
import scipy.interpolate
import h5py
import matplotlib.animation
import matplotlib.pyplot as plt

try:
    from agi.stk12.stkdesktop import STKDesktop
    from agi.stk12.stkobjects import *
    from agi.stk12.stkutil import *
    from agi.stk12.vgt import *
except ImportError as e:
    # Install STK python library if package is not already installed
    # Issue an error message
    print("STK Python library not found. Please install STK Python library")
    print("pip install \"<STK installation directory>/bin/AgPythonAPI/agi.stk12-12.9.0-py3-none-any.whl\"")


class STK_Utils():
    """
    Utility class for extracting and managing object motion data from STK scenarios.
    
    This class provides methods to extract position, orientation, velocity, and angular
    velocity data from STK objects. It can either connect to a live STK session or
    load previously saved data from numpy archives.
    
    Attributes:
        stk: STK Desktop application instance
        stkRoot: STK application root object
        global_cs: Global coordinate system reference
        global_object_path: Path to the global reference object
        scenario_name: Name of the current STK scenario
        dataFrequency: Data sampling frequency (Hz)
        time_start_idx: Start index for time slicing
        time_stop_idx: Stop index for time slicing
        scaling: Scaling factor for position and velocity data
    """

    def __init__(self, global_object_path=None, global_cs=None, time_start_idx=None, 
                 time_stop_idx=None, dataFrequency=1.0, scaling=1, load_from_npz=False, 
                 npz_filename=None):
        """
        Initialize STK_Utils instance.
        
        Args:
            global_object_path (str, optional): Path to global reference object in STK
            global_cs (str, optional): Global coordinate system name
            time_start_idx (int, optional): Starting time index for data slicing
            time_stop_idx (int, optional): Ending time index for data slicing
            dataFrequency (float): Data sampling frequency in Hz (default: 1.0)
            scaling (float): Scaling factor for position/velocity data (default: 1)
            load_from_npz (bool): Whether to load from saved numpy file (default: False)
            npz_filename (str, optional): Path to numpy archive file
        """
        if load_from_npz and npz_filename is not None:
            # Load previously saved data from numpy archive
            data = np.load(npz_filename)
            self._time_stamps = data['time_stamps']
            self._pos = data['pos']
            self._rot = data['rot']
            self._lin = data['lin']
            self._ang = data['ang']
        else:
            # Connect to STK application
            stk = STKDesktop.AttachToApplication()
            self.stk = stk
            self.stkRoot = self.stk.Root

            # Store configuration parameters
            self.global_cs = global_cs
            self.global_object_path = global_object_path
            self.scenario_name = self.stkRoot.CurrentScenario.InstanceName
            self.dataFrequency = dataFrequency
            self.time_start_idx = time_start_idx
            self.time_stop_idx = time_stop_idx
            self.scaling = scaling
        
    def __del__(self):
        """Destructor for cleanup operations."""
        print("STK Clean Up")

    def getDataFromSTK(self, local_object_path, local_cs):
        """
        Extract motion data from STK for a specified object.
        
        This method retrieves position, velocity, attitude, and angular velocity data
        from STK for a given object relative to a global reference frame.
        
        Args:
            local_object_path (str): STK path to the target object
            local_cs (str): Local coordinate system for the target object
            
        Note:
            Requires utilities.rotation module for Euler angle conversions
        """
        from pem_utilities.rotation import euler_to_rot
        
        print(f"Getting Data from Scenario Name:{self.scenario_name}, {local_object_path}, {local_cs}")
        print(local_object_path)
        
        # Get STK objects
        target = self.stkRoot.GetObjectFromPath(local_object_path)
        reference_object = self.stkRoot.GetObjectFromPath(self.global_object_path)
        
        # Create truncated path strings for data provider configuration
        targetTruncatedPath = target.Path.split('/')[-2] + '/' + target.Path.split('/')[-1]
        referenceTruncatedPath = reference_object.Path.split('/')[-2] + '/' + reference_object.Path.split('/')[-1]
        
        # Determine vector axes string based on object type
        if 'sensor' in local_object_path.lower() or 'receiver' in local_object_path.lower() or 'transmitter' in local_object_path.lower():
            vector_choose_axes_str = f'Vector Choose Axes/{target.Parent.InstanceName}-{target.InstanceName}'
        else:
            vector_choose_axes_str = f'Vector Choose Axes/{target.InstanceName}'

        # Set up data provider for position and velocity data
        dataPrv = reference_object.DataProviders.GetDataPrvTimeVarFromPath(vector_choose_axes_str)
        preData = f'{referenceTruncatedPath} {self.global_cs}'
        dataPrv.PreData = preData

        # Position information comes from Data Provider from Global report TO object
        dataRequestpos = [['Time'], ['x'], ['y'], ['z']]
        dataResultspos = dataPrv.ExecElements(self.stkRoot.CurrentScenario.StartTime,
                                           self.stkRoot.CurrentScenario.StopTime, 
                                           self.dataFrequency, 
                                           dataRequestpos)

        # Linear Velocity information comes from Data Provider from Global report TO object
        dataRequestlin = [['Time'], ['Derivative x'], ['Derivative y'], ['Derivative z']]
        dataResultslin = dataPrv.ExecElements(self.stkRoot.CurrentScenario.StartTime,
                                              self.stkRoot.CurrentScenario.StopTime,
                                              self.dataFrequency,
                                              dataRequestlin)

        # Code to get attitude data out of STK in custom VGT coordinate system
        dataPrvAttitude = target.DataProviders.GetDataPrvTimeVarFromPath(f'Axes Choose Axes/{local_cs}')
        preDataAttitude = f'{referenceTruncatedPath} {self.global_cs}'
        dataPrvAttitude.PreData = preDataAttitude

        # Rotation information comes from the Data Provider wrt the Global Axes
        dataRequestrot = ['Euler323 precession', 'Euler323 nutation', 'Euler323 spin']
        dataResultsrot = dataPrvAttitude.ExecElements(self.stkRoot.CurrentScenario.StartTime,
                                                            self.stkRoot.CurrentScenario.StopTime, 
                                                            self.dataFrequency, 
                                                            dataRequestrot)

        # Angular Velocity information comes from the Data Provider wrt to Global
        dataRequestang = ['wx', 'wy', 'wz']
        dataResultsang = dataPrvAttitude.ExecElements(self.stkRoot.CurrentScenario.StartTime,
                                                      self.stkRoot.CurrentScenario.StopTime,
                                                      self.dataFrequency,
                                                      dataRequestang)

        # Store data in dictionary object
        data = {}

        # Extract position data relative to the global axes
        data['time'] = np.array(dataResultspos.DataSets.GetDataSetByName('Time').GetValues())
        data['xPos'] = np.array(dataResultspos.DataSets.GetDataSetByName('x').GetValues()) * 1
        data['yPos'] = np.array(dataResultspos.DataSets.GetDataSetByName('y').GetValues()) * 1
        data['zPos'] = np.array(dataResultspos.DataSets.GetDataSetByName('z').GetValues()) * 1

        # Extract linear velocity data relative to the global axes
        data['xVel'] = np.array(dataResultslin.DataSets.GetDataSetByName('Derivative x').GetValues()) * 1
        data['yVel'] = np.array(dataResultslin.DataSets.GetDataSetByName('Derivative y').GetValues()) * 1
        data['zVel'] = np.array(dataResultslin.DataSets.GetDataSetByName('Derivative z').GetValues()) * 1

        # Extract Euler 323 rotation of custom body axes relative to custom global axes
        data['Euler Angle A'] = np.array(dataResultsrot.DataSets.GetDataSetByName('Euler323 precession').GetValues())
        data['Euler Angle B'] = np.array(dataResultsrot.DataSets.GetDataSetByName('Euler323 nutation').GetValues())
        data['Euler Angle C'] = np.array(dataResultsrot.DataSets.GetDataSetByName('Euler323 spin').GetValues())

        # Extract Angular Velocity of custom body axes relative to custom global axes
        data['AngVel A'] = np.array(dataResultsang.DataSets.GetDataSetByName('wx').GetValues())
        data['AngVel B'] = np.array(dataResultsang.DataSets.GetDataSetByName('wy').GetValues())
        data['AngVel C'] = np.array(dataResultsang.DataSets.GetDataSetByName('wz').GetValues())

        # Apply time slicing if specified
        if self.time_stop_idx is not None:
            for each in data:
                data[each] = data[each][self.time_start_idx:self.time_stop_idx]
                
        # Combine components into vector arrays
        data['pos'] = np.array([data['xPos'], data['yPos'], data['zPos']]).T * 1
        data['lin'] = np.array([data['xVel'], data['yVel'], data['zVel']]).T
        data['ang'] = np.array([data['AngVel A'], data['AngVel B'], data['AngVel C']]).T

        # Store processed data
        self.data = data
        self._pos = self.data['pos'] * self.scaling
        self._rot = euler_to_rot(self.data['Euler Angle A'], self.data['Euler Angle B'], self.data['Euler Angle C'], order='ZYZ')
        self._lin = self.data['lin'] * self.scaling
        self._ang = self.data['ang'] * np.pi / 180  # Convert from degrees to radians
        
    def pos_interp(self, times):
        """
        Interpolate position data at specified times.
        
        Args:
            times (array-like): Time points for interpolation
            
        Returns:
            ndarray: Interpolated position data
        """
        pos_interp_func = scipy.interpolate.interp1d(self.time_stamps, self._pos, axis=0, 
                                                      assume_sorted=True, bounds_error=False, 
                                                      fill_value='extrapolate')
        pos = pos_interp_func(times)
        return pos 

    def rot_interp(self, times):
        """
        Interpolate rotation data at specified times.
        
        Args:
            times (array-like): Time points for interpolation
            
        Returns:
            ndarray: Interpolated rotation data
        """
        rot_interp_func = scipy.interpolate.interp1d(self.time_stamps, self._rot, axis=0, 
                                                      assume_sorted=True, bounds_error=False, 
                                                      fill_value='extrapolate')        
        rot = rot_interp_func(times)
        return rot

    def lin_interp(self, times):
        """
        Interpolate linear velocity data at specified times.
        
        Args:
            times (array-like): Time points for interpolation
            
        Returns:
            ndarray: Interpolated linear velocity data
        """
        lin_interp_func = scipy.interpolate.interp1d(self.time_stamps, self._lin, axis=0, 
                                                      assume_sorted=True, bounds_error=False, 
                                                      fill_value='extrapolate')
        lin = lin_interp_func(times)
        return lin

    def ang_interp(self, times):
        """
        Interpolate angular velocity data at specified times.
        
        Args:
            times (array-like): Time points for interpolation
            
        Returns:
            ndarray: Interpolated angular velocity data
        """
        ang_interp_func = scipy.interpolate.interp1d(self.time_stamps, self._ang, axis=0, 
                                                      assume_sorted=True, bounds_error=False, 
                                                      fill_value='extrapolate')
        ang = ang_interp_func(times)
        return ang

    def save_values_as_numpy(self, filename):
        """
        Save motion data to a numpy archive file.
        
        Args:
            filename (str): Output filename for the numpy archive
        """
        np.savez(filename, time_stamps=self.time_stamps, pos=self._pos, rot=self._rot, 
                 lin=self._lin, ang=self._ang)

    @property
    def time_stamps(self):
        """
        Convert STK time format to epoch seconds.
        
        Returns:
            ndarray: Time stamps in epoch seconds format
        """
        time_stamps = []
        converter = self.stkRoot.ConversionUtility
        for each in self.data['time']:
            epochSecsTime = converter.ConvertDate('UTCG', 'EpSec', each)
            time_stamps.append(float(epochSecsTime))
        return np.array(time_stamps)

    @property
    def pos(self):
        """Position data getter."""
        return self._pos

    @pos.setter
    def pos(self, value):
        """Position data setter."""
        self._pos = value

    @property
    def rot(self):
        """Rotation data getter."""
        return self._rot

    @rot.setter
    def rot(self, value):
        """Rotation data setter."""
        self._rot = value

    @property
    def lin(self):
        """Linear velocity data getter."""
        return self._lin

    @lin.setter
    def lin(self, value):
        """Linear velocity data setter."""
        self._lin = value

    @property
    def ang(self):
        """Angular velocity data getter."""
        return self._ang

    @ang.setter
    def ang(self, value):
        """Angular velocity data setter."""
        self._ang = value

    @property
    def phi(self):
        """Phi Euler angle (precession) getter."""
        phi = self.data['Euler Angle A']
        return phi

    @property
    def theta(self):
        """Theta Euler angle (nutation) getter."""
        theta = self.data['Euler Angle B']
        return theta

    @property
    def psi(self):
        """Psi Euler angle (spin) getter."""
        psi = self.data['Euler Angle C']
        return psi

    @property
    def unit_vec(self):
        """Unit vector in direction of position vector."""
        v_hat = self.data['pos'].T / np.linalg.norm(self.data['pos'], axis=1)
        return v_hat.T

    @property
    def distance(self):
        """Distance magnitude from origin."""
        v_hat = np.linalg.norm(self.data['pos'], axis=1)
        return v_hat


class STK_Results_Reader():
    """
    Reader class for STK response (.rsp) files containing channel characterization data.
    
    This class provides methods to read HDF5-formatted STK response files and extract
    channel response data for analysis and visualization.
    """

    def __init__(self):
        """Initialize STK_Results_Reader instance."""
        pass

    def read_rsp(self, filename=None):
        """
        Read STK response file (.rsp) containing channel characterization data.
        
        Args:
            filename (str, optional): Path to .rsp file. If None, opens file dialog.
            
        Returns:
            dict: Dictionary containing channel data for all links, or None if file invalid
        """
        if filename is None:
            # Open file dialog to select .rsp file
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(filetypes=[('rsp files', '*.rsp'), ('All Files', '*.*')])
            
        if not filename:
            print(f'{filename} is not a real file path with a file name!')
            return None

        s = {}
        with h5py.File(filename, 'r') as fid:
            # Get all available links in the file
            lnks = list(fid['/Links'])
            s = {}
            
            for i, lnk in enumerate(lnks):
                d = {}
                # Extract link configuration parameters
                d['nTx'] = fid['/Links/' + lnk + '/Transmitter'].attrs['Antenna Count'][0]
                d['nRx'] = fid['/Links/' + lnk + '/Receiver'].attrs['Antenna Count'][0]
                d['nP'] = fid['/Links/' + lnk + '/Waveform'].attrs['Channel Soundings'][0]
                d['nS'] = fid['/Links/' + lnk + '/Waveform'].attrs['Sample Count'][0]
                d['bw'] = fid['/Links/' + lnk + '/Waveform'].attrs['Bandwidth'][0]
                d['fc'] = fid['/Links/' + lnk + '/Waveform'].attrs['Frequency'][0]
                
                # Extract time and response arrays for each link
                d['t'] = fid['/Links/' + lnk + '/Channel Characterization/Time Array'][:]
                
                # Read the response data that is stored as separate real and imag parts
                # and convert to complex data for storage in the data dictionary
                respdata = fid['/Links/' + lnk + '/Channel Characterization/Response'][:]
                d['resp'] = respdata[:, 0::2] + 1j * respdata[:, 1::2]
                s[lnk] = d
                
            self.data = s
        return s

    def get_link_data(self, designated_link):
        """
        Extract and structure data for a specific link.
        
        Args:
            designated_link (str): Name/identifier of the link to extract
            
        Returns:
            tuple: (responsedata, Tdatvec, fsweepvec, tsweepvec, fc, bw)
                - responsedata: 5D array [time, Tx, Rx, pulse, sample]
                - Tdatvec: Time vector in epoch seconds
                - fsweepvec: Frequency sweep vector in GHz
                - tsweepvec: Time sweep vector in microseconds
                - fc: Center frequency
                - bw: Bandwidth
        """
        # Extract basic parameters
        Tdatvec = self.data[designated_link]['t'][0]  # Sounding simulation time samples (epoch seconds)
        Tsamps = len(Tdatvec)  # Number of multi-sounding simulations in this data set
        nTx = self.data[designated_link]['nTx']  # Number of transmitters in this link
        nRx = self.data[designated_link]['nRx']  # Number of receivers in this link
        nPulses = self.data[designated_link]['nP']  # Number of soundings per simulation
        nSamples = self.data[designated_link]['nS']  # Number of freq samples per sounding
        bw = self.data[designated_link]['bw']  # Total simulation bandwidth for soundings
        fc = self.data[designated_link]['fc']  # Carrier (center) frequency of bw
        
        # Extract and reshape response data
        responsedata = self.data[designated_link]['resp']  # Extract all the real,imag response data for this link
        responsedata = responsedata.reshape(Tsamps, nTx, nRx, nPulses, nSamples)  # Structure data in n-dim matrix

        # Create frequency and time vectors for analysis
        fsweepvec = np.arange(fc - bw / 2, fc + bw / 2, bw / nSamples) / 1e9  # Express data in GHz
        tsweepvec = np.arange(0, nSamples) * 1.0 / bw * 1e6  # Express data in microseconds
        
        return (responsedata, Tdatvec, fsweepvec, tsweepvec, fc, bw)


def main():
    """
    Main function for processing and visualizing STK response data.
    
    This function provides a complete workflow for:
    1. Reading STK response files
    2. Processing channel data for each link
    3. Generating time/frequency domain plots
    4. Creating waterfall plots and animations
    """
    # Set up GUI for file selection
    root = tk.Tk()
    root.withdraw()
    animation_dir = filedialog.askdirectory(title="Pick a folder in directory to save plots")
    do_animation_plots = False  # Set to True to generate animation files

    # Initialize reader and load data
    reader = STK_Results_Reader()
    chandata = reader.read_rsp()

    # Display available links
    links = chandata.keys()
    print("Available Link Data: \n")
    for mylink in links:
        print(mylink)

    # Begin Processing each link
    for mylink in links:
        print("Working on Link: " + mylink)
        respdata, Tdat, freqsweepvec, timesweepvec, fc, bw = reader.get_link_data(mylink)

        # Get data dimensions
        Tsamps, nTx, nRx, nPulses, nSamples = respdata.shape

        # Process each transmitter-receiver pair
        for myTx in range(0, nTx):
            for myRx in range(0, nRx):
                print('Running Tx, Rx: ' + str(myTx) + ',' + str(myRx))

                # Plot the frequency domain data (Re, Im) for the first time interval, center sounding/pulse
                plt.figure()
                plt.plot(freqsweepvec, np.real(respdata[0][0][0][int(np.floor(nPulses/2))][:]))
                plt.plot(freqsweepvec, np.imag(respdata[0][0][0][int(np.floor(nPulses/2))][:]))
                plt.title('Freq Domain - First Sweep')
                plt.ylabel('S-parameter forward coupling (Re, Im)')
                plt.xlabel('Channel Frequency (Hz)')
                plt.tight_layout()
                plt.show()

                # Prepare data for time domain analysis
                blackman_window = np.blackman(nSamples)
                Tdata = []
                Fdata = []
                TdataWindowed = []
                
                # Process each time step
                for tcount, t in enumerate(Tdat):
                    Fdata.append(respdata[tcount][myTx][myRx][int(np.floor(nPulses / 2))][:])
                    Tdata.append(np.fft.ifft(Fdata[tcount]))
                    TdataWindowed.append(np.fft.ifft(Fdata[tcount] * blackman_window))
                    
                # Convert to numpy arrays
                Fdata = np.array(Fdata)
                Tdata = np.array(Tdata)
                TdataWindowed = np.array(TdataWindowed)
                TdataMax = np.max(np.abs(Tdata))

                def T_plot_update(i):
                    """Update function for time domain animation plots."""
                    ax.clear()
                    ax.plot(timesweepvec, 20.0 * np.log10(np.abs(TdataWindowed[i])))
                    ax.set_xlim([0, timesweepvec[-1]])
                    ax.set_ylim([-150.0, -50.0])
                    ax.set_ylabel('Channel Power Gain (dB)')
                    ax.set_xlabel('Delay Time (usec)')
                    ax.set_title('Power Delay Profile (dB), Time = {:3.1f}'.format(Tdat[i] - Tdat[0]) + 's',
                                 fontsize=10)
                    plt.suptitle(mylink)
                    plt.tight_layout()

                def F_plot_update(i):
                    """Update function for frequency domain animation plots."""
                    ax.clear()
                    ax.plot(freqsweepvec, 20.0 * np.log10(np.abs(Fdata[i])))
                    ax.set_xlim([freqsweepvec[0], freqsweepvec[-1]])
                    ax.set_ylim([-120.0, -20.0])
                    ax.set_ylabel('Channel Forward Voltage Gain (dB)')
                    ax.set_xlabel('Channel Frequency (GHz)')
                    ax.set_title('Channel Spectrum Magnitude (dB), Time = {:3.1f}'.format(Tdat[i] - Tdat[0]) + 's',
                                 fontsize=10)
                    plt.suptitle(mylink)
                    plt.tight_layout()

                def Fri_plot_update(i):
                    """Update function for frequency domain real/imaginary animation plots."""
                    ax.clear()
                    ax.plot(freqsweepvec, np.real(Fdata[i]))  # Plot Real part
                    ax.plot(freqsweepvec, np.imag(Fdata[i]))  # Plot Imaginary part
                    ax.set_xlim([freqsweepvec[0], freqsweepvec[-1]])
                    ax.set_ylim([-0.0002, 0.0002])
                    ax.set_ylabel('Channel Forward Voltage Gain')
                    ax.set_xlabel('Channel Frequency (GHz)')
                    ax.set_title('Channel Spectrum Real/Imag Data, Time = {:3.1f}'.format(Tdat[i] - Tdat[0]) + 's',
                                 fontsize=10)
                    plt.suptitle(mylink)
                    plt.tight_layout()

                # Generate animation plots if enabled
                if do_animation_plots:
                    fig, ax = plt.subplots()

                    # Time domain animation
                    ani = matplotlib.animation.FuncAnimation(fig, T_plot_update, frames=Tsamps, interval=200)
                    f_const = animation_dir + 'T_' + mylink + '_Tx_' + str(myTx) + '_Rx_' + str(myRx) + '.mp4'

                    # Configure ffmpeg path
                    base_env_path = sys.prefix
                    ffmpeg_path = f'{base_env_path}/Library/bin/ffmpeg.exe'
                    if os.path.exists(ffmpeg_path):
                        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
                    else:
                        print(f'ffmpeg not found at {ffmpeg_path}')
                        print(f'Please download ffmpeg and extract to {ffmpeg_path}: https://www.gyan.dev/ffmpeg/builds/')

                    writervideo = matplotlib.animation.FFMpegWriter(fps=10)
                    ani.save(f_const, writer=writervideo)

                    # Frequency domain magnitude animation
                    ani = matplotlib.animation.FuncAnimation(fig, F_plot_update, frames=Tsamps, interval=200)
                    f_const = animation_dir + 'F_' + mylink + '_Tx_' + str(myTx) + '_Rx_' + str(myRx) + '.mp4'
                    writervideo = matplotlib.animation.FFMpegWriter(fps=10)
                    ani.save(f_const, writer=writervideo)

                    # Frequency domain real/imaginary animation
                    ani = matplotlib.animation.FuncAnimation(fig, Fri_plot_update, frames=Tsamps, interval=200)
                    f_const = animation_dir + 'Fri_' + mylink + '_Tx_' + str(myTx) + '_Rx_' + str(myRx) + '.mp4'
                    writervideo = matplotlib.animation.FFMpegWriter(fps=10)
                    ani.save(f_const, writer=writervideo)

                # Generate frequency domain waterfall plot
                plt.figure()
                plt.imshow(20.0 * np.log10(np.abs(np.flipud(Fdata.T))), cmap='jet', vmin=-100.0, vmax=-40.0,
                           aspect='auto',
                           extent=[0, Tdat[-1] - Tdat[0], (fc - (bw / 2.0)) / 1E9, (fc + (bw / 2.0)) / 1E9])
                plt.title('Spectrum Power Gain vs. Time')
                plt.xlabel('Scenario Time (s)')
                plt.ylabel('Frequency (GHz)')
                plt.colorbar()
                plt.suptitle(mylink)
                plt.tight_layout()
                plt.savefig(animation_dir + 'Fwtr_' + mylink + '_Tx_' + str(myTx) + '_Rx_' + str(myRx) + '.png')

                # Generate time domain waterfall plot
                plt.figure()
                plt.imshow(20.0 * np.log10(np.abs(np.flipud(TdataWindowed.T))), cmap='magma', vmin=-140, vmax=-80,
                           aspect='auto', extent=[0, Tdat[-1] - Tdat[0], 0, timesweepvec[-1]])
                plt.title('Power Delay (CIR) Waterfall')
                plt.xlabel('Scenario Time (s)')
                plt.ylabel('Channel Signal Delay (usec)')
                plt.colorbar()
                plt.suptitle(mylink)
                plt.tight_layout()
                plt.savefig(animation_dir + 'Twtr_' + mylink + '_Tx_' + str(myTx) + '_Rx_' + str(myRx) + '.png')

        print('End ' + mylink)

    print('Complete')


if __name__ == "__main__":
    main()

