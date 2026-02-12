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
import matplotlib.pyplot as plt
import PIL.Image

# create api
libDir = 'C:/Program Files/AnsysEM/Perceive_EM/v252/Win64/lib/'

sys.path.append(libDir)
import RssPy
api = RssPy.RssApi()

# optional environment variable to specify custom location for RTR licensing directory
# the variable should point to the licensingclient/ directory
# the shared_files directory MUST be at least one directory level above licensingclient/
if "RTR_LICENSE_DIR" not in os.environ:
  os.environ["RTR_LICENSE_DIR"] = libDir + "/licensingclient"

# function to check for warnings or errors
def isOK(rGpuCallStat):
  if(rGpuCallStat == RssPy.RGpuCallStat.OK):
    return
  elif(rGpuCallStat == RssPy.RGpuCallStat.RGPU_WARNING):
    print(api.getLastWarnings())
  else:
    print(api.getLastError())
    exit()
isOK(api.selectApiLicenseMode(RssPy.ApiLicenseMode.PERCEIVE_EM))
# for debugging
#api.setVerbose(True)

# print version and copyright info
print(api.copyright())
print(api.version(True))

# class to hold coordinate system data
class CoordSys:
  # default constructor
  def __init__(self,hNode = None, hElem = None):
    if hNode is None:
      hNode = RssPy.SceneNode()
      isOK(api.addSceneNode(hNode))
    self.hNode = hNode
    self.hElem = hElem
    self.rot = np.eye(3)   # 3x3 rotation matrix
    self.pos = np.zeros(3) # position in meters
    self.lin = np.zeros(3) # linear velocity in meters/sec
    self.ang = np.zeros(3) # angular velocity
    if hElem is not None:
      isOK(api.setSceneElement(self.hNode,self.hElem))
  # update coordinate system
  # in this example, only linear velocity is considered
  def update(self,time):
    newPos = np.asarray(self.pos) + time*np.asarray(self.lin)
    isOK(api.setCoordSysInGlobal(
      self.hNode,self.rot,newPos,self.lin,self.ang
    ))

# create small triangle mesh
tgtLen = 0.2 # meters
tgtVerts = np.asarray([
  [0. ,0. ,0.],
  [0. ,tgtLen,0.],
  [0. ,0. ,tgtLen],
  [0. ,tgtLen,tgtLen]],
  dtype=np.float32)
tgtTris = np.asarray(
  [[0,1,2],
   [1,2,3]],
   dtype=np.int32)
hTgtMesh = RssPy.SceneElement()
isOK(api.addSceneElement(hTgtMesh))
isOK(api.setTriangles(hTgtMesh,tgtVerts,tgtTris,0))

# initialize target coordinate system
tgtCS = CoordSys(hNode=None,hElem=hTgtMesh)
tgtCS.pos = (+100.,0.,0.)
tgtCS.lin = (+40.,0.,0.)

# create EGO vehicle platform
hPlatform = RssPy.RadarPlatform()
isOK(api.addRadarPlatform(hPlatform))
platformCS = CoordSys(hPlatform,None)

# list of coordinate systems to update between frames
csList = (tgtCS,platformCS)

# create radar device and antennas
hDevice = RssPy.RadarDevice()
isOK(api.addRadarDevice(hDevice, hPlatform))
hTx = RssPy.RadarAntenna()
isOK(api.addRadarAntennaParametricBeam(hTx,hDevice,RssPy.AntennaPolarization.VERT,45.,45.,1.))
hRx = RssPy.RadarAntenna()
isOK(api.addRadarAntennaParametricBeam(hRx,hDevice,RssPy.AntennaPolarization.VERT,45.,45.,1.))

# configure radar mode
hMode = RssPy.RadarMode()
isOK(api.addRadarMode(hMode, hDevice))
isOK(api.setRadarModeActive(hMode,True))
isOK(api.addTxAntenna(hMode,hTx))
isOK(api.addRxAntenna(hMode,hRx))
isOK(api.setRadarModeStartDelay(hMode,0.,RssPy.ModeDelayReference.CENTER_CHIRP))

# configure arbitrary waveform
numChirps = 101
centerFreq = 10e9# Hz
numFreq = 128
pulse_interval = 10e-3
bandwidth = 100e6 # Hz

centerFreqs = np.ones(numChirps,dtype=np.float64)*centerFreq # Hz
bandwidths = np.ones(numChirps,dtype=np.float64)*bandwidth # Hz
numFreqSamples = numFreq*np.ones(numChirps,dtype=np.int32)
PulseIntervals = np.ones(numChirps,dtype=np.float64)*pulse_interval # sec

txMultiplex = RssPy.TxMultiplex.SIMULTANEOUS
isOK(api.setArbitraryPulseDopplerWaveform(hMode, txMultiplex,numChirps,centerFreqs,bandwidths,numFreqSamples,PulseIntervals))

responseType = RssPy.ResponseType.ARBITRARY_WAVEFORM

# calculate max range and range resolution based on bandwidth and number of frequency samples
rangeResolution = 3e8 / bandwidth / 2  
maxRange = numFreq * rangeResolution

print("Max range: {:.2f} m, Range resolution: {:.2f} m".format(maxRange, rangeResolution))


# assign modes to devices
print(api.listGPUs())
devIDs = [0]; devQuotas = [0.8]; # limit PEM to use 80% of available gpu memory
isOK(api.setGPUDevices(devIDs,devQuotas))
isOK(api.assignAllRadarModesToGPU(0))

# initialize solver settings
isOK(api.setMaxNumRefl(3))
isOK(api.setMaxNumTrans(3))
isOK(api.setTargetRayDensity(0.1))

maxNumRayBatches = 1000
isOK(api.autoConfigureSimulation(maxNumRayBatches))

# optional check if RSS is configured
# this will also be checked before response computation
if not api.isReady():
  print("RSS is not ready to execute a simulation:\n")
  print(api.getLastWarnings())

# display setup
#print(api.reportSettings())

# get response domains
(ret,txRespIdxs,rxRespIdxs,chirpTimes,frequencies) = api.responseDomains(hMode,responseType)
isOK(ret)

# compute radar response
dt = 0.1; numFrames = 10
responses = []
for iFrame in range(numFrames):
  print('simulating frame {:d} of {:d}'.format(1+iFrame,numFrames))
  time = iFrame*dt
  for cs in csList:
    cs.update(time)
  isOK(api.computeResponseSync())
  (ret,response) = api.retrieveResponse(hMode,responseType)
  isOK(ret)
  responses.append(response)

responses = np.asarray(responses)
# response is in the form of [frame_idx,tx_idx,rx_idx,chirp_idx*adc_sample_idx]
# reshape the response to [frame_idx,tx_idx,rx_idx,numChirps,numADCSamples]
numTx = responses.shape[1]
numRx = responses.shape[2]


responses = responses.reshape((numFrames,numTx,numRx,numChirps,numFreq))

# plot the first frame of the response
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(responses[0, 0, 0, :, :]), aspect='auto', cmap='jet')
plt.title('Magnitude of Response (Tx 0, Rx 0)')
plt.xlabel('Freq Samples')
plt.ylabel('Chirps')
plt.colorbar(label='Magnitude')
plt.subplot(1, 2, 2)
plt.imshow(np.angle(responses[0, 0, 0, :, :]), aspect='auto', cmap='jet')
plt.title('Phase of Response (Tx 0, Rx 0)')
plt.xlabel('Freq Samples')
plt.ylabel('Chirps')
plt.colorbar(label='Phase (radians)')
plt.tight_layout()
plt.show()

num_rng_samples = 512
rng_domain = np.linspace(0, maxRange, num_rng_samples)
# plot the ifft of the first frame of the response
ifftResponse = np.fft.ifft(responses[0, 0, 0, :, :], axis=-1,n=num_rng_samples)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(20*np.log10(np.abs(ifftResponse)), aspect='auto', cmap='jet')
plt.title('Magnitude of IFFT Response (Tx 0, Rx 0)')
plt.xticks(ticks=np.linspace(0, num_rng_samples-1, 9), labels=np.round(np.linspace(0, maxRange, 9), 2))
plt.xlabel('Range Samples')
plt.ylabel('Chirps')
plt.colorbar(label='Magnitude')
plt.subplot(1, 2, 2)
plt.imshow(np.angle(ifftResponse), aspect='auto', cmap='jet')
plt.title('Phase of IFFT Response (Tx 0, Rx 0)')
plt.xticks(ticks=np.linspace(0, num_rng_samples-1, 9), labels=np.round(np.linspace(0, maxRange, 9), 2))
plt.xlabel('Range Samples')
plt.ylabel('Chirps')
plt.colorbar(label='Phase (radians)')
plt.tight_layout()
plt.show()

# create a plot showing each chirp in the first frame
plt.figure(figsize=(12, 6))
for iChirp in range(numChirps):
    plt.plot(rng_domain, np.abs(ifftResponse[iChirp]), label=f'Chirp {iChirp+1}')
plt.title('Magnitude of IFFT Response for Each Chirp (Tx 0, Rx 0)')
plt.xlabel('Range (m)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# plot the peak of each chirp in the first frame
peak_magnitudes = np.max(np.abs(ifftResponse), axis=1)
# peak at which rng_domain
peak_indices = np.argmax(np.abs(ifftResponse), axis=1)
peak_ranges = rng_domain[peak_indices]
plt.figure(figsize=(12, 6))
plt.plot(np.arange(numChirps), peak_ranges, marker='o')
plt.title('Peak Magnitude of Each Chirp (Tx 0, Rx 0)')
plt.xlabel('Chirp Index')
plt.ylabel('Peak Magnitude')
plt.xticks(np.arange(numChirps), [f'Chirp {i+1}' for i in range(numChirps)], rotation=45)
plt.grid()
plt.tight_layout()
plt.show()