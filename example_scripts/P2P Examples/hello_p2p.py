"""
(c) 2012-2025 ANSYS, Inc. All rights reserved. Unauthorized use, distribution,
  or duplication is prohibited.

THIS ANSYS SOFTWARE PRODUCT AND PROGRAM DOCUMENTATION INCLUDE TRADE SECRETS AND
ARE CONFIDENTIAL AND PROPRIETARY PRODUCTS OF ANSYS, INC., ITS SUBSIDIARIES, OR
LICENSORS. The software products and documentation are furnished by ANSYS,
Inc., its subsidiaries, or affiliates under a software license agreement that
contains provisions concerning non-disclosure, copying, length and nature of
use, compliance with exporting laws, warranties, disclaimers, limitations of
liability, and remedies, and other provisions.  The software products and
documentation may be used, disclosed, transferred, or copied only in accordance
with the terms and conditions of that software license agreement.
"""

import os
import sys
import numpy as np
import math as m

import PIL.Image
import time as perftime
import matplotlib.pyplot as plt

# create api
libDir = 'C:/Program Files/AnsysEM/Perceive_EM/v252/Win64/lib/'

sys.path.insert(0, libDir)
print(sys.path)
import RssPy

api = RssPy.RssApi()


# function to check for warnings or errors
def isOK(rGpuCallStat):
    if (rGpuCallStat == RssPy.RGpuCallStat.OK):
        return
    elif (rGpuCallStat == RssPy.RGpuCallStat.RGPU_WARNING):
        print(api.getLastWarnings())
    else:
        print(api.getLastError())
        exit()


# optional environment variable to specify custom location for RTR licensing directory
# the variable should point to the licensingclient/ directory
# the shared_files directory MUST be at least one directory level above licensingclient/
if "RTR_LICENSE_DIR" not in os.environ:
    os.environ["RTR_LICENSE_DIR"] = libDir + "/licensingclient"

# for debugging
# api.setVerbose(True)


# print version and copyright info
print(api.copyright())
print(api.version(True))

# Customers who are not Perceive EM users disable this line or pass DEFAULT
isOK(api.selectApiLicenseMode(RssPy.ApiLicenseMode.PERCEIVE_EM))
# Perceive EM users optionally enable this line with the type of HPC license
# they have to potentially save initialization time.
isOK(api.selectPreferredHpcLicense(RssPy.HpcLicenseType.HPC_ANSYS_PACK))


# class to hold coordinate system data
class CoordSys:
    # default constructor
    def __init__(self, hNode=None, hElem=None):
        if hNode is None:
            hNode = RssPy.SceneNode()
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
    # in this example, only linear velocity is considered
    def update(self, time):
        newPos = np.asarray(self.pos) + time * np.asarray(self.lin)
        isOK(api.setCoordSysInGlobal(
            self.hNode, self.rot, newPos, self.lin, self.ang
        ))


# add a road
verts = np.ascontiguousarray([[-1, -1, 0], [-1, +1, 0], [+1, +1, 0], [+1, -1, 0]], dtype=np.float32) * 1.e3
tris = np.ascontiguousarray([[0, 1, 2], [2, 3, 0]], dtype=np.int32)
coat = 1
hPlateMesh = RssPy.SceneElement()
isOK(api.addSceneElement(hPlateMesh))
isOK(api.setTriangles(hPlateMesh, verts, tris, coat))
plateCS = CoordSys(None, hPlateMesh)


class RadarPlatform:
    def __init__(self, numTx=1, numRx=1, waveforms=[]):
        self._hPlatform = RssPy.RadarPlatform()
        hPlatform = self._hPlatform
        isOK(api.addRadarPlatform(hPlatform))
        self._platformCS = CoordSys(hPlatform, None)
        self._platformCS.pos = (0., 0., 0.)
        self._platformCS.lin = (0., 0., 0.)
        self._hDevice = RssPy.RadarDevice()
        hDevice = self._hDevice
        isOK(api.addRadarDevice(hDevice, hPlatform))

        # add antennas
        (pol, vBW, hBW) = (RssPy.AntennaPolarization.VERT, 30., 60.)
        self._hTxAntennas = []
        for iTx in range(numTx):
            hAnt = RssPy.RadarAntenna()
            isOK(api.addRadarAntennaParametricBeam(hAnt, hDevice, pol, vBW, hBW, 1.))
            self._hTxAntennas.append(hAnt)
        self._hRxAntennas = []
        for iRx in range(numRx):
            hAnt = RssPy.RadarAntenna()
            isOK(api.addRadarAntennaParametricBeam(hAnt, hDevice, pol, vBW, hBW, 1.))
            self._hRxAntennas.append(hAnt)

        # add waveforms
        self._waveforms = waveforms
        self._hModes = []
        for wf in waveforms:
            hMode = RssPy.RadarMode()
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
        self.centerFreq = 76.5e9
        self.bandWidthHz = 0.1e9
        self.numFreqSamples = 267
        self.cpiDuration = 0.004
        self.numPulseCPI = 200
        self.modeDelay = RssPy.ModeDelayReference.CENTER_CHIRP
        self.txMultiplex = RssPy.TxMultiplex.SIMULTANEOUS

    def pulseInterval(self):
        return self.cpiDuration / self.numPulseCPI


# specify multiple waveforms with different center frequencies
wf0 = Waveform()
wf1 = Waveform()
wf1.centerFreq -= 3.
wf2 = Waveform()
wf2.centerFreq -= 6.
waveforms = (wf0, wf1, wf2)

# create platforms
txPlatform = RadarPlatform(2, 0, waveforms)
txPlatformCS = txPlatform._platformCS
txPlatformCS.pos = (0., 0., 0.5)
txPlatformCS.lin = (0., 0., 0.)
txPlatformCS.rot = np.eye(3)

rxPlatform = RadarPlatform(0, 2, waveforms)
rxPlatformCS = rxPlatform._platformCS
rxPlatformCS.pos = (+200, 0., 0.5)
rxPlatformCS.lin = (-20., 0., 0.)
rxPlatformCS.rot = np.diag((-1., -1., 1.))

# list of coordinate systems to update between frames
csList = (plateCS, txPlatformCS, rxPlatformCS)

# range doppler specs for Tx mode
sideLobeLevelDb = 50.
rSpecs = "hann," + str(sideLobeLevelDb)
dSpecs = "hann," + str(sideLobeLevelDb)
rPixels = 1024
dPixels = 1024
centerVel = 0.

for hMode in txPlatform._hModes:
    isOK(api.activateRangeDopplerResponse(
        hMode, rPixels, dPixels, centerVel, rSpecs, dSpecs))
responseType = RssPy.ResponseType.RANGE_DOPPLER

# enable output from individual Tx
txRespComp = RssPy.ResponseComposition.INDIVIDUAL
for hMode in txPlatform._hModes:
    isOK(api.setTxResponseComposition(hMode, txRespComp))

# optional: radar mode Rx thermal noise and channel gain/loss
# only available for Tx composite responses
if txRespComp == RssPy.ResponseComposition.COMPOSITE:
    for hMode, wf in zip(rxPlatform._hModes, rxPlatform._waveforms):
        rxGainType = RssPy.RxChannelGainSpecType.USER_DEFINED
        rxGain_dB = 20.
        isOK(api.setRadarModeRxChannelGain(hMode, rxGainType, rxGain_dB))
        T = 290  # temperature in kelvin
        kB = 1.380649e-23  # Boltzmann constant
        rxNoise_dB = 10 * m.log10(kB * T * wf.bandWidthHz)  # assuming Rx channel bandwidth is same as Tx
        isOK(api.setRadarModeRxThermalNoise(hMode, True, rxNoise_dB))

# enable tx -> rx platform coupling
hTxPlatform = txPlatform._hPlatform
hRxPlatform = rxPlatform._hPlatform
isOK(api.setDoP2PCoupling(hTxPlatform, hRxPlatform, True))

# assign modes to devices
print(api.listGPUs())
isOK(api.setGPUDevices([0]))
isOK(api.assignAllRadarModesToGPU(0))

# initialize solver settings
isOK(api.setMaxNumRefl(3))
isOK(api.setMaxNumTrans(3))
isOK(api.setRaySpacing(0.5))

# optional check if RSS is configured
# this will also be checked before response computation
if not api.isReady():
    print("RSS is not ready to execute a simulation:\n")
    print(api.getLastWarnings())

# display setup
# print(api.reportSettings())

# get coupled modes
(ret, txrxModePairs) = api.p2pCouplingModes(hTxPlatform, hRxPlatform)

# get response domains
(idxTxRxModePair, idxTx, idxRx) = (0, 0, 0)
hTxMode = txrxModePairs[idxTxRxModePair][0]
hRxMode = txrxModePairs[idxTxRxModePair][1]
(ret, velDomain, rngDomain) = api.p2pResponseDomains(hTxMode, hRxMode, responseType)
isOK(ret)

# initialize plots
cLims_dB = (-180, -120)
(fig, axes) = plt.subplots()
imData = np.zeros((len(rngDomain), len(velDomain)))
image = plt.imshow(imData, interpolation='bilinear', cmap='jet', \
                   aspect='auto', vmin=cLims_dB[0], vmax=cLims_dB[1], \
                   extent=[velDomain[0], velDomain[-1], rngDomain[0], rngDomain[-1]])
axes.set_xlabel('Doppler velocity (m/s)')
axes.set_ylabel('Range (m)')

# compute radar response
fps = 2;
dt = 1 / fps
T = 10;
numFrames = int(T / dt)
images = []
reportRate = 1
for iFrame in range(numFrames):
    doPrint = numFrames < reportRate or (iFrame + 1) % reportRate == 0
    if doPrint:
        print('simulating frame {:d} of {:d}'.format(1 + iFrame, numFrames))
    time = iFrame * dt
    for cs in csList:
        cs.update(time)
    start = perftime.perf_counter()
    isOK(api.computeResponseSync())
    end = perftime.perf_counter()
    if doPrint:
        print("compute response time [s]: {:.4e}".format(end - start))
    (ret, response) = api.retrieveP2PResponse(hTxMode, hRxMode, responseType)
    # test radar modes
    # (ret,response) = api.retrieveResponse(hTxMode,responseType)
    # (ret,response) = api.retrieveResponse(hRxMode,responseType); idxTx = 0
    isOK(ret)
    if doPrint:
        print('processing frame {:d} of {:d}'.format(1 + iFrame, numFrames))
    response = response[idxTx][idxRx]
    imData = np.rot90(20 * np.log10(np.fmax(np.abs(response), 1.e-30)))
    image.set_data(imData)
    fig.canvas.draw()

    # Updated method to create image from canvas
    buf = np.asarray(fig.canvas.buffer_rgba())
    radarIm = PIL.Image.fromarray(buf[:,:,:3])  # Only take RGB channels

    images.append(radarIm)

print('writing hello_p2p.gif...')
images[0].save('hello_p2p.gif',save_all=True,append_images=images[1:],optimize=False,duration=int(1000/fps),loop=0)
