# -*- coding: utf-8 -*-
"""
Created on Fri April 19, 2024

@author: asligar

Copyright ANSYS. All rights reserved.

In this project, they want to simulate an entire city with N base stations (J antennas each) and M UE (K antennas
each). All the positions for the BS and the UE's are generated at random. A list of integer values of each
parameter (J,K,M,N) can be definded.  The benchmarking will sweep through all possible combinations of these
parameters and report the solve times Along with sweeping the number of devices and number of antennas per device,
the user can also choose which group of antennas will act as the transmitter or receiver. This is done becase
simulation times are primarily impacted by the number of Tx antennas, not the number or Rx antennas
"""

import os
import sys
import numpy as np
import csv
import time as walltime
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import json

#######################################################################################################################
#
# USER INPUT
#
#######################################################################################################################

# create api
libDir = 'C:/Program Files/AnsysEM/Perceive_EM/v241/Win64/lib/'
path_to_benchmark_geometry = r'..\models\benchmark_town02.stl'
output_directory = '../output/benchmark'  # output_directory: Directory to save the output files

number_of_gpus = 1
case_to_run = 'case1' # defined below, but currently available cases are case1, case2a, case2b, case2c, case3


##################################################################################################################
# Start Case Definitions
##################################################################################################################

##################
# Cases defined below, we don't need to edit them directly, just change them directly, but for easier running
# of benchmark, we just need to change what key/case we want to run
case_dict = {}

# case 1: Number of UE devices created in the scene

output_summary_name = f'results_summary_case1_{number_of_gpus}gpu.json'
num_ue_sweep = [2,4,8,16,32,64,128,256,512,1024,2048,4096,5120]  # number of rx antennas
num_bs_sweep = [1] # num of antenna devices (tx only)
num_ant_bs_sweep = [1]  # num of Tx antenna per Antenna Device
case_dict['case1'] = {'output_summary_name':output_summary_name,
                      'num_ue_sweep':num_ue_sweep,
                      'num_bs_sweep':num_bs_sweep,
                      'num_ant_bs_sweep':num_ant_bs_sweep}

# case 2a: Number of antenna per UE device
output_summary_name = f'results_summary_case2a_{number_of_gpus}gpu.json'
num_ue_sweep = [1]  # number of rx antennas
num_bs_sweep = [1] # num of antenna devices (tx only)
num_ant_bs_sweep = [2,4,8,16,32,64,128,256,512,1024,2048,4096,5120]  # num of Tx antenna per Antenna Device
case_dict['case2a'] = {'output_summary_name':output_summary_name,
                      'num_ue_sweep':num_ue_sweep,
                      'num_bs_sweep':num_bs_sweep,
                      'num_ant_bs_sweep':num_ant_bs_sweep}


# case 2b: Number of antenna per UE device
output_summary_name = f'results_summary_case2b_{number_of_gpus}gpu.json'
num_ue_sweep = [1]  # number of rx antennas
num_bs_sweep = [2,4,8,16,32,64,128] # num of antenna devices (tx only)
num_ant_bs_sweep = [1]  # num of Tx antenna per Antenna Device
case_dict['case2a'] = {'output_summary_name':output_summary_name,
                      'num_ue_sweep':num_ue_sweep,
                      'num_bs_sweep':num_bs_sweep,
                      'num_ant_bs_sweep':num_ant_bs_sweep}

# case 2c: Number of antenna per UE device
output_summary_name = f'results_summary_case2c_{number_of_gpus}gpu.json'
num_ue_sweep = [1]  # number of rx antennas
num_bs_sweep = [2,4,8,16,32,64,128] # num of antenna devices (tx only)
num_ant_bs_sweep = [2,4,8,16,32,64,128] # num of Tx antenna per Antenna Device
case_dict['case2a'] = {'output_summary_name':output_summary_name,
                      'num_ue_sweep':num_ue_sweep,
                      'num_bs_sweep':num_bs_sweep,
                      'num_ant_bs_sweep':num_ant_bs_sweep}

# case 3: Number of antenna per UE device
output_summary_name = f'results_summary_case3_{number_of_gpus}gpu.json'
num_ue_sweep = [1,8,16,32,64,128]
num_ant_ue_sweep = [1]
num_bs_sweep = [2,8,16,32]
num_ant_bs_sweep = [2,8,16,32]
case_dict['case3'] = {'output_summary_name':output_summary_name,
                      'num_ue_sweep':num_ue_sweep,
                      'num_bs_sweep':num_bs_sweep,
                      'num_ant_bs_sweep':num_ant_bs_sweep}


##################################################################################################################
# End Case Definitions
##################################################################################################################



output_summary_name =case_dict[case_to_run]['output_summary_name']
num_ue_sweep = case_dict[case_to_run]['num_ue_sweep']
num_bs_sweep = case_dict[case_to_run]['num_bs_sweep']
num_ant_bs_sweep = case_dict[case_to_run]['num_ant_bs_sweep']


# I reverse them so largest runs first, jst incase there is an issue the bigger values will fail first
num_ant_bs_sweep.reverse()
num_bs_sweep.reverse()
num_ue_sweep.reverse()

#######################################################################################################################
# END USER INPUT
#######################################################################################################################
# num_ant_ue_sweep = [1] # number of rx antenna per ue device
which_tx_sweep = ['bs']  # which_tx_sweep: Which group of antennas will act as the transmitter, either 'bs' or 'ue'
sys.path.append(libDir)
import P2Py

api = P2Py.P2PApi()

# output file will be stored in this directory
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(path_to_benchmark_geometry):
    raise Exception('Geometry file does not exist')

licensingclient_dir = libDir
if os.path.exists(os.path.join(licensingclient_dir, 'licensingclient')):
    os.environ["RTR_LICENSE_DIR"] = licensingclient_dir
    print(f'Found licensingclient at {licensingclient_dir}')



##############################################################################
#
# Scene define
#
#############################################################################
scenario_name = "Benchmark_Speedtest"

save_data = False  # save data as numpy array
cpi = 100e-3
numPulseCPI = 100
numFreqSamples = 512
center_freq = 3e9
bandwidth = 100e6


# function to check for warnings or errors
def isOK(rGpuCallStat):
    if (rGpuCallStat == P2Py.RGpuCallStat.OK):
        return
    elif (rGpuCallStat == P2Py.RGpuCallStat.RGPU_WARNING):
        print(api.getLastWarnings())
    else:
        print(api.getLastError())
        exit()



results_dict = {}
for num_ue in num_ue_sweep:
    for num_ant_ue in num_ant_ue_sweep:
        for num_bs in num_bs_sweep:
            for num_ant_bs in num_ant_bs_sweep:
                for which_tx in which_tx_sweep:
                    result_log = []
                    result_dict = {}
                    devIDs = []
                    devQuotas = []
                    for each in range(number_of_gpus):
                        devIDs.append(each)  # multiple devices are supported for multiple modes in 2022R1
                        devQuotas.append(.99)  # float between 0 and 1, percentage of GPU memory that can be used
                    isOK(api.setGPUDevices(devIDs, devQuotas))

                    # nothing in the scene is moving, this is just used to average the results across multiple frames
                    # I am using 10 frames which seems like enough
                    scenario_time_step = .1  # same as my CPI length, although it doesn't really matter for this example
                    scenario_length = .4  # simulation time is averaged over this total simulation time
                    ##############################################################################
                    total_ant_all_ue = num_ue

                    # paths to supporting files. Scene is just one stl file,
                    geometry_path = path_to_benchmark_geometry


                    def euler_to_rot(phi, theta, psi, order='zyz', deg=True):
                        if isinstance(phi, float) or isinstance(phi, int):
                            phi = np.array([phi])
                        if isinstance(theta, float) or isinstance(theta, int):
                            theta = np.array([theta])
                        if isinstance(psi, float) or isinstance(psi, int):
                            psi = np.array([psi])

                        # fix size of arrays so they are matching in length
                        length_of_array = max(len(phi), len(theta), len(psi))
                        if len(phi) != len(theta) or len(phi) != len(psi) or len(theta) != len(psi):
                            which_is_longest = np.argmax([len(phi), len(theta), len(psi)])
                            if which_is_longest == 0:
                                if len(theta) != 1 and len(theta) != len(phi):
                                    Exception('Theta must be a scalar or same length as phi')
                                elif len(theta) == 1:
                                    theta = np.repeat(theta, len(phi))
                                if len(psi) != 1 and len(psi) != len(phi):
                                    Exception('Psi must be a scalar or same length as phi')
                                elif len(psi) == 1:
                                    psi = np.repeat(psi, len(phi))
                            elif which_is_longest == 1:
                                if len(phi) != 1 and len(phi) != len(theta):
                                    Exception('Phi must be a scalar or same length as theta')
                                elif len(phi) == 1:
                                    phi = np.repeat(phi, len(theta))
                                if len(phi) != 1 or len(psi) != len(theta):
                                    Exception('Psi must be a scalar or same length as theta')
                                elif len(psi) == 1:
                                    psi = np.repeat(psi, len(theta))
                            else:
                                if len(phi) != 1 or len(phi) != len(psi):
                                    Exception('Phi must be a scalar or same length as psi')
                                elif len(phi) == 1:
                                    phi = np.repeat(phi, len(psi))

                                if len(theta) != 1 or len(theta) != len(psi):
                                    Exception('Theta must be a scalar or same length as psi')
                                elif len(theta) == 1:
                                    theta = np.repeat(theta, len(psi))

                        all_angs = zip(phi, theta, psi)
                        n = 0
                        rot = np.zeros((length_of_array, 3, 3))
                        for phi_val, theta_val, psi_val in all_angs:
                            rot_temp = Rotation.from_euler(order, [phi_val, theta_val, psi_val], degrees=deg)
                            rot_temp = rot_temp.as_matrix()
                            rot[n] = rot_temp
                            n += 1
                        return rot

                    scene_extents = (-116.98009490966797, 369.10833740234375, -389.11541748046875, 15.075408935546875,
                                     -7.051324844360352, 44.40016174316406)
                    pos_x = np.random.uniform(scene_extents[0], scene_extents[1], num_ue)
                    pos_y = np.random.uniform(scene_extents[2], scene_extents[3], num_ue)
                    pos_z = np.random.uniform(scene_extents[4], scene_extents[5], num_ue)
                    all_ue_pos = np.array(np.stack((pos_x, pos_y, pos_z)).T, dtype=float)

                    rot_x = np.random.uniform(0, 360, num_ue)
                    rot_y = np.random.uniform(-10, 10, num_ue)
                    rot_z = np.random.uniform(0, 10, num_ue)
                    rot = np.stack((rot_x, rot_y, rot_z)).T
                    all_ue_rot = np.array(euler_to_rot(rot[:, 0], rot[:, 1], rot[:, 2]), dtype=float)
                    if all_ue_rot.ndim == 2:
                        all_ue_rot =  np.swapaxes(np.atleast_3d(all_ue_rot),0,-1)

                    # all resulting positions and rotations of UE's and BS
                    pos_x = np.random.uniform(scene_extents[0], scene_extents[1], num_bs)
                    pos_y = np.random.uniform(scene_extents[2], scene_extents[3], num_bs)
                    pos_z = np.random.uniform(scene_extents[4], scene_extents[5], num_bs)
                    all_bs_pos = np.array(np.stack((pos_x, pos_y, pos_z)).T, dtype=float)

                    rot_x = np.random.uniform(0, 360, num_bs)
                    rot_y = np.random.uniform(-10, 10, num_bs)
                    rot_z = np.random.uniform(0, 10, num_bs)
                    rot = np.stack((rot_x, rot_y, rot_z)).T
                    all_bs_rot = np.array(euler_to_rot(rot[:, 0], rot[:, 1], rot[:, 2]), dtype=float)
                    if all_bs_rot.ndim == 2:
                        all_bs_rot =  np.swapaxes(np.atleast_3d(all_bs_rot),0,-1)
                    # time steps to use, although time doesn't really matter, this is just used to step through all the UE's
                    time_stamps = np.linspace(0, scenario_length, num=int(scenario_length / scenario_time_step))

                    # optional environment variable to specify custom location for RTR licensing directory
                    # the variable should point to the licensingclient/ directory
                    # the shared_files directory MUST be at least one directory level above licensingclient/
                    # os.environ["RTR_LICENSE_DIR"] = libDir + "/licensingclient"

                    # for debugging
                    api.setVerbose(False)
                    # api.setPrivateKey("VerboseLevel1", "true");
                    # api.setPrivateKey("ShowErrorDetails","True")

                    isOK(api.selectApiLicenseMode(P2Py.ApiLicenseMode.PERCEIVE_EM))
                    # depending on the license, select the appropriate HPC license HPC_ANSYS_PACK or HPC_ANSYS_POOL
                    isOK(api.selectPreferredHpcLicense(P2Py.HpcLicenseType.HPC_ANSYS_PACK))

                    # class to hold coordinate system data
                    class CoordSys:
                        # default constructor
                        def __init__(self, hNode=None, hElem=None):
                            if hNode is None:
                                hNode = P2Py.SceneNode()
                                isOK(api.addSceneNode(hNode))
                            self.hNode = hNode
                            self.hElem = hElem
                            self.rot = np.eye(3)  # 3x3 rotation matrix
                            self.pos = np.zeros(3)  # position in meters
                            self.lin = np.zeros(3)  # linear velocity in meters/sec
                            self.ang = np.zeros(3)  # angular velocity
                            if hElem is not None:
                                isOK(api.setSceneElement(self.hNode, self.hElem))

                        # update coordinate system
                        def update_ant(self, pos, rot):
                            # update which UE position to use, depending on th etime step, and how many UE's are requested
                            isOK(api.setCoordSysInGlobal(self.hNode, rot, pos, self.lin, self.ang))


                    def loadMesh(filename):
                        mesh = api.loadTriangleMesh(filename)
                        hMesh = P2Py.SceneElement()
                        isOK(api.addSceneElement(hMesh))
                        # set triangles from mesh
                        isOK(api.setTriangles(hMesh, mesh.vertices, mesh.triangles, 1))
                        return (hMesh, mesh)


                    hWall = P2Py.Coating()
                    isOK(api.addCoating(hWall,
                                        "DielectricLayers 0.2,5,0.0,1.0,0.0,8 VACUUM"))  # material properties as defined by customer
                    isOK(api.mapCoatingToIndex(hWall, 1))  # assign to everything
                    (hSceneMesh, _) = loadMesh(geometry_path)
                    sceneCS = CoordSys(None, hSceneMesh)


                    class RadarPlatform:
                        def __init__(self, numTx=1, numRx=1, waveforms=[]):
                            self._hPlatform = P2Py.RadarPlatform()
                            hPlatform = self._hPlatform
                            isOK(api.addRadarPlatform(hPlatform))
                            self._platformCS = CoordSys(hPlatform, None)
                            self._platformCS.pos = (0., 0., 0.)
                            self._platformCS.lin = (0., 0., 0.)
                            self._hDevice = P2Py.RadarDevice()
                            hDevice = self._hDevice
                            isOK(api.addRadarDevice(hDevice, hPlatform))

                            # add antennas, each TX/RX in Nokia Example is dual polarized, although it doesn't really matter for this example
                            polV = P2Py.AntennaPolarization.VERT
                            polH = P2Py.AntennaPolarization.HORZ
                            (vBW, hBW) = (120., 120.)

                            max_num_rows = 2  # how big can the array be in one dimensions, the real UE is only 2 antennas
                            row_num = 0  # index to keep track of which row in the array I am on
                            self._hTxAntennas = []
                            for iTx in range(numTx):
                                hAnt = P2Py.RadarAntenna()
                                pol = polV
                                if iTx % 2 == 0:  # alternate odd and even as horizontal and vert polarization
                                    pol = polH
                                isOK(api.addRadarAntennaParametricBeam(hAnt, hDevice, pol, vBW, hBW, 1.))
                                # just moving each element so that is not exactly on top of each other. I don't care about exact layout
                                pos = np.array([(iTx % max_num_rows) * .04, 0, row_num * .04])
                                if iTx % max_num_rows == 0:
                                    row_num += 1  # increment which row I'm on
                                isOK(api.setCoordSysInParent(hAnt, np.eye(3), pos, np.zeros(3), np.zeros(3)))
                                self._hTxAntennas.append(hAnt)

                            max_num_rows = 8  # how big can the array be in one dimensions, the real array is 8x8
                            row_num = 0  # index to keep track of which row in the array I am on
                            self._hRxAntennas = []
                            max_num_rows = 8  # how big can the array be in one dimensions, the real array is 8x8
                            row_num = 0  # index to keep track of which row in the array I am on
                            for iRx in range(numRx):
                                hAnt = P2Py.RadarAntenna()
                                pol = polV
                                if iRx % 2 == 0:  # alternate odd and even as horizontal and vert polarization
                                    pol = polH
                                isOK(api.addRadarAntennaParametricBeam(hAnt, hDevice, pol, vBW, hBW, 1.))
                                # just moving each element so that is not exactly on top of each other. I don't care about exact layout
                                pos = np.array([(iRx % max_num_rows) * .04, 0, row_num * .04])
                                if iRx % max_num_rows == 0:
                                    row_num += 1  # increment which row I'm on
                                isOK(api.setCoordSysInParent(hAnt, np.eye(3), pos, np.zeros(3), np.zeros(3)))
                                self._hRxAntennas.append(hAnt)

                            # add waveforms
                            self._waveforms = waveforms
                            self._hModes = []
                            for wf in waveforms:
                                hMode = P2Py.RadarMode()
                                self._hModes.append(hMode)
                                isOK(api.addRadarMode(hMode, self._hDevice))
                                isOK(api.setRadarModeActive(hMode, True))
                                for hTx in self._hTxAntennas:
                                    isOK(api.addTxAntenna(hMode, hTx))
                                for hRx in self._hRxAntennas:
                                    isOK(api.addRxAntenna(hMode, hRx))
                                isOK(api.setRadarModeStartDelay(hMode, 0., wf.modeDelay))
                                isOK(api.setPulseDopplerWaveformSysSpecs(
                                    hMode, wf.centerFreq, wf.bandWidthHz, wf.numFreqSamples,
                                    wf.pulseInterval(), wf.numPulseCPI, wf.txMultiplex))


                    # waveform config
                    class Waveform():
                        def __init__(self):
                            self.centerFreq = center_freq
                            self.bandWidthHz = bandwidth
                            self.numFreqSamples = numFreqSamples
                            self.cpiDuration = cpi
                            self.numPulseCPI = numPulseCPI
                            self.modeDelay = P2Py.ModeDelayReference.CENTER_CHIRP
                            self.txMultiplex = P2Py.TxMultiplex.SIMULTANEOUS

                        def pulseInterval(self):
                            return self.cpiDuration / self.numPulseCPI


                    # specify multiple waveforms with different center frequencies
                    wf0 = Waveform()

                    waveforms = (wf0,)

                    ue_platforms = []
                    ue_cs = []
                    # create platforms
                    for each in range(num_ue):  # how many UE's to solve simultaneously
                        if which_tx.lower() == 'ue':
                            uePlatform = RadarPlatform(num_ant_ue, 0, waveforms)  # each UE has N Tx antennas
                        else:
                            uePlatform = RadarPlatform(0, num_ant_ue, waveforms)  # each UE has N Tx antennas
                        ue_platforms.append(uePlatform)
                        uePlatformCS = uePlatform._platformCS
                        uePlatformCS.pos = (0., 0., 0.0)
                        uePlatformCS.lin = (0., 0., 0.)
                        uePlatformCS.rot = np.eye(3)
                        ue_cs.append(uePlatformCS)

                    bs_platforms = []
                    bs_cs = []
                    for each in range(num_bs):  # how many UE's to solve simultaneously
                        if which_tx.lower() == 'ue':
                            bsPlatform = RadarPlatform(0, num_ant_bs, waveforms)  # each UE has N Tx antennas
                        else:
                            bsPlatform = RadarPlatform(num_ant_bs, 0, waveforms)  # each UE has N Tx antennas
                        bs_platforms.append(bsPlatform)
                        bsPlatformCS = bsPlatform._platformCS
                        bsPlatformCS.pos = (0., 0., 0.0)
                        bsPlatformCS.lin = (0., 0., 0.)
                        bsPlatformCS.rot = np.eye(3)
                        bs_cs.append(bsPlatformCS)

                    # range doppler specs for Tx mode
                    sideLobeLevelDb = 50.
                    rSpecs = "hann," + str(sideLobeLevelDb)
                    dSpecs = "hann," + str(sideLobeLevelDb)
                    rPixels = 1024
                    dPixels = 1024
                    centerVel = 0.

                    # enable output from individual Tx
                    rComp = P2Py.ResponseComposition.INDIVIDUAL
                    for ue in ue_platforms:
                        for hMode in ue._hModes:
                            isOK(api.setTxResponseComposition(hMode, rComp))

                    # enable tx -> rx platform coupling
                    if which_tx.lower() == 'ue':
                        for ue in ue_platforms:
                            hTxPlatform = ue._hPlatform
                            for bs in bs_platforms:
                                hRxPlatform = bs._hPlatform
                                # I assume this is like what we could get out of HFSS-SBR+, with s-parameters for every Tx and Rx pair
                                isOK(api.setDoP2PCoupling(hTxPlatform, hRxPlatform, True))
                    else:
                        for bs in bs_platforms:
                            hTxPlatform = bs._hPlatform
                            for ue in ue_platforms:
                                hRxPlatform = ue._hPlatform
                                # I assume this is like what we could get out of HFSS-SBR+, with s-parameters for every Tx and Rx pair
                                isOK(api.setDoP2PCoupling(hTxPlatform, hRxPlatform, True))

                    # assign modes to devices
                    print(api.listGPUs())

                    # if I have multiple GPU's how is this set? if I comment this line out it still runs, so I assume setGPUDevices does
                    # something similiar
                    # isOK(api.assignAllRadarModesToGPU(0))

                    # initialize solver settings
                    isOK(api.setMaxNumRefl(3))
                    isOK(api.setMaxNumTrans(2))
                    isOK(api.setTargetRayDensity(0.01))
                    try:
                        isOK(api.autoConfigureSimulation(10))
                    except:
                        api.reset()
                        print('Autoconfigure failed, resetting and moving to next setup')
                        break
                    # optional check if RSS is configured
                    # this will also be checked before response computation
                    if not api.isReady():
                        print("RSS is not ready to execute a simulation:\n")
                        print(api.getLastWarnings())

                    # get coupled modes
                    mode_pairs = []
                    if which_tx.lower() == 'ue':
                        for ue in ue_platforms:
                            hTxPlatform = ue._hPlatform
                            for bs in bs_platforms:
                                hRxPlatform = bs._hPlatform
                                (ret, txrxModePairs) = api.p2pCouplingModes(hTxPlatform, hRxPlatform)
                                mode_pairs.append(txrxModePairs)
                    else:
                        for bs in bs_platforms:
                            hTxPlatform = bs._hPlatform
                            for ue in ue_platforms:
                                hRxPlatform = ue._hPlatform
                                (ret, txrxModePairs) = api.p2pCouplingModes(hTxPlatform, hRxPlatform)
                                mode_pairs.append(txrxModePairs)

                    # compute radar response
                    images = []

                    # place all antennas in scene
                    for ue_idx, cs in enumerate(ue_cs):
                        cs.update_ant(all_ue_pos[ue_idx], all_ue_rot[ue_idx])
                    for bs_idx, cs in enumerate(bs_cs):
                        cs.update_ant(all_bs_pos[bs_idx], all_bs_rot[bs_idx])

                    total_solver_time = 0
                    total_retrieve_time = 0
                    time_start_simulation = walltime.time()  #this will be overwritten on the second frame to run, ignoring licensing check

                    for iFrame in range(len(time_stamps)):

                        # print('simulating frame {:d} of {:d}'.format(1+iFrame,len(time_stamps)))
                        time_before = walltime.time()
                        isOK(api.computeResponseSync())
                        time_after = walltime.time()
                        if iFrame != 0:  #ignore first frame, it include license checkout
                            print(f'Frame Number {iFrame} of {len(time_stamps)}')
                            total_time = time_after - time_before
                            if total_time < 0:
                                total_time = 0
                            total_solver_time += total_time
                            avg_sim_time = total_solver_time / (iFrame)
                            if total_time > 0:
                                current_fps = 1 / total_time
                            else:
                                current_fps = 'inf'
                            if total_time > 0:
                                avg_fps = 1 / avg_sim_time
                            else:
                                avg_fps = 'inf'
                            print(f'Estimated Solve Time: {avg_sim_time * (len(time_stamps) - 1)}')
                            print(f'Average FPS: {avg_fps}')
                            print(f'Current FPS: {current_fps}')

                            # I am not saving all the respones, but I think this is getting all the responses, I would need to save every
                            # response from within this For loop, and also the outer For loop to save all the data
                            # becuase this step can be the slowest, I am only going to call it on the final step in the simulation
                            if iFrame == len(time_stamps)-1:
                                time_before_retrieve = walltime.time()
                                responses = []
                                for idx, pair in enumerate(mode_pairs):
                                    (ret, response) = api.retrieveP2PResponse(pair[0][0], pair[0][1],
                                                                              P2Py.ResponseType.FREQ_PULSE)
                                    # all_data[iFrame*idx+idx] = np.array(response)
                                    # responses.append(response)

                                # isOK(ret)
                                time_after_retrieve = walltime.time()
                                time_retrieve = time_after_retrieve - time_before_retrieve
                                total_retrieve_time += time_retrieve
                                if time_retrieve > 0:
                                    fps_time_retrieve = 1 / time_retrieve
                                    print(f'Current Retrieve FPS: {fps_time_retrieve}')
                                else:
                                    fps_time_retrieve=-1
                                result_dict['fps_avg_retrieve'] = fps_time_retrieve
                        else:
                            time_start_simulation = walltime.time()

                    time_stop_simulation = walltime.time()
                    total_sim_time = time_stop_simulation - time_start_simulation
                    if which_tx.lower() == 'bs':
                        num_rx = num_ue * num_ant_ue
                        num_tx_per_tx = num_ant_bs
                        num_tx_units = num_bs
                        num_tx = num_bs * num_ant_bs
                    else:
                        num_rx = num_bs * num_ant_bs
                        num_tx = num_ue * num_ant_ue
                        num_tx_per_tx = num_ant_ue
                        num_tx_units = num_ue

                    total_channel_pairs = num_ue * num_ant_ue * num_bs * num_ant_bs
                    result_dict['num_ue'] = num_ue
                    result_dict['num_bs'] = num_bs
                    result_dict['num_rx'] = num_rx
                    result_dict['num_tx_per_tx'] = num_tx_per_tx
                    result_dict['num_tx_units'] = num_tx_units
                    result_dict['num_tx'] = num_ue
                    result_dict['total_channel_pairs'] = total_channel_pairs
                    result_dict['which_tx'] = which_tx

                    result_log.append('------------------------------------------')
                    result_log.append('----------Results Summary: Setup----------')
                    result_log.append(f'Number of GPUs: {number_of_gpus}')
                    result_log.append(f'Number of BS: {num_bs} with {num_ant_bs} antennas per BS')
                    result_log.append(f'Number of UE: {num_ue} with {num_ant_ue} antennas per UE')
                    result_log.append(f'Total Rx Antennas: {num_rx}')
                    result_log.append(f'Total Tx Antennas: {num_tx}')
                    result_log.append(f'Total Channel Pairs: {total_channel_pairs}')
                    result_log.append(f'Center Freq: {center_freq * 1e-9}GHz')
                    result_log.append(f'Bandwidth: {bandwidth * 1e-6}MHz, with {numFreqSamples} Samples')
                    result_log.append(f'CPI: {cpi * 1e-3}ms, with {numPulseCPI} Time Samples')
                    result_log.append(f'Antenna Group Acting as Transmitter: {which_tx}')
                    result_log.append('----------Results Summary: Performance----------')
                    if avg_sim_time > 0:
                        fps_avg = 1 / avg_sim_time
                    else:
                        fps_avg = 'inf'


                    result_dict['avg_sim_time'] = avg_sim_time
                    if fps_avg != 'inf':
                        result_dict['fps_avg'] = fps_avg
                    else:
                        result_dict['fps_avg'] = -1

                    result_log.append(f'Average Simulation FPS: {fps_avg}')
                    result_log.append(f'Average Results Retrieval FPS: {fps_time_retrieve}')
                    result_log.append(f'Total Solve Time: {total_solver_time}')
                    result_log.append(f'Total Retrieve Time: {total_retrieve_time}')
                    result_log.append(f'Total Simulation Time: {total_sim_time}')

                    parameter_str = f'gpu{number_of_gpus}_bs{num_bs}_ants{num_ant_bs}_ue{num_ue}_ants{num_ant_ue}_txgroup{which_tx}.txt'
                    output_file = os.path.join(output_directory, parameter_str)
                    with open(output_file, 'w') as f:
                        for line in result_log:
                            f.write(line + '\n')
                    print(f'Output File: {output_file}')

                    # results_dict

                    if save_data:
                        os.path.join(output_directory, 'out_cmplx.npy')
                        np.save(os.path.join(output_directory, 'out_cmplx.npy'), np.array(response))
                        np.save(os.path.join(output_directory, 'out_real.npy'), np.array(np.real(response)))
                        np.savez_compressed(os.path.join(output_directory, 'outz_cmplx.npy'), a=np.array(response))
                        np.savez_compressed(os.path.join(output_directory, 'outz_real.npy'),
                                            a=np.array(np.real(response)))
                    api.reset()
                    results_dict[parameter_str] = result_dict


json_object = json.dumps(results_dict, indent=4)
filename = os.path.join(output_directory, output_summary_name)
with open(filename, "w") as outfile:
    outfile.write(json_object)


