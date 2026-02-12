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
import matplotlib.pyplot as plt
from pem_utilities.path_helper import get_repo_paths # common paths to reference
paths = get_repo_paths() 

# create api
libDir = 'C:/Program Files/AnsysEM/Perceive_EM/v252/Win64/lib/'
sys.path.append(libDir)
import RssPy
api = RssPy.RssApi()
# acl.update_license_env_vars(libDir)
os.environ["ANSYSCL252_DIR"] = libDir + "/licensingclient"
# Select Response Type
outputOptions = [RssPy.ResponseType.FREQ_PULSE,
                 RssPy.ResponseType.RANGE_PULSE,
                 RssPy.ResponseType.RANGE_DOPPLER,
                 RssPy.ResponseType.ADC_SAMPLES]
activeOutput = 2

mesh_to_include = 'terrain' # 'full_scene', 'lead_car', 'semi', 'terrain'

# function to check for warnings or errors
def isOK(rGpuCallStat):
  if(rGpuCallStat == RssPy.RGpuCallStat.OK):
    return
  elif(rGpuCallStat == RssPy.RGpuCallStat.RGPU_WARNING):
    print(api.getLastWarnings())
  else:
    print(api.getLastError())
    exit()

# for debugging
#api.setVerbose(True)

# print version and copyright info
print(api.copyright())
print(api.version(True))

# Customers who are not Perceive EM users disable this line or pass DEFAULT
isOK(api.selectApiLicenseMode(RssPy.ApiLicenseMode.PERCEIVE_EM))
# Perceive EM users optionally enable this line with the type of HPC license
# they have to potentially save initialization time.
#isOK(api.selectPreferredHpcLicense(RssPy.HpcLicenseType.HPC_ANSYS))

 # load meshes and create scene elements
def loadMesh(filename):
  mesh = api.loadTriangleMesh(filename)
  hMesh = RssPy.SceneElement()
  isOK(api.addSceneElement(hMesh))
  # set triangles from mesh
  isOK(api.setTriangles(hMesh,mesh))
  # set triangles, 1 coating index for each triangle
  #isOK(api.setTriangles(hMesh,mesh.vertices,mesh.triangles,mesh.coatings))
  # set triangles, 1 coating index for all triangles
  #coatIdx = 1; isOK(api.setTriangles(hMesh,mesh.vertices,mesh.triangles,coatIdx))
  return (hMesh,mesh)

if mesh_to_include == 'full_scene':
  (hSceneMesh,_) = loadMesh(os.path.join(paths.models,'whole-scene-static.stl'))
  (hCarMesh,_)   = loadMesh(os.path.join(paths.models,'mustang-no-wheels.stl'))
  (hAxleMesh,_)  = loadMesh(os.path.join(paths.models,'mustang-axle.stl'))
  (hSemiMesh,_)  = loadMesh(os.path.join(paths.models,'tractor-trailor.stl'))
elif mesh_to_include == 'lead_car':
  (hCarMesh,_)   = loadMesh(os.path.join(paths.models,'mustang-no-wheels.stl'))
  (hAxleMesh,_)  = loadMesh(os.path.join(paths.models,'mustang-axle.stl'))
elif mesh_to_include == 'semi':
  (hSemiMesh,_)  = loadMesh(os.path.join(paths.models,'tractor-trailor.stl'))
elif mesh_to_include == 'terrain':
  (hSceneMesh,_) = loadMesh(os.path.join(paths.models,'whole-scene-static.stl'))  



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


def update_wheels(time):
  for ix,x in enumerate((1.9,-0.9)):
    radius = 0.35
    ang = carCS.lin[0]/radius
    c = m.cos(ang*time)
    s = m.sin(ang*time)
    rot = np.asarray([[+c,0,+s],[0,1,0],[-s,0,+c]])
    pos = np.asarray((x,0.,radius))
    lin = np.zeros(3)
    ang = ang*np.asarray((0,1,0))
    isOK(api.setCoordSysInParent(hAxles[ix],rot,pos,lin,ang))

# instantiate scene elements and initialize coordinate systems
if mesh_to_include == 'full_scene':
  sceneCS = [CoordSys(None,hSceneMesh),CoordSys(None,hSceneMesh)]
  sceneCS[1].pos = (150.,0.,0.)
  carCS = CoordSys(None,hCarMesh)
  carCS.pos = (6.,0.,0.)
  carCS.lin = (15.,0.,0.)
  semiCS = CoordSys(None,hSemiMesh)
  semiCS.rot[0,0] = -1; semiCS.rot[1,1] = -1
  semiCS.pos = (42.,10.5,0.)
  semiCS.lin = (-25.,0.,0.)
  # create car wheels (front/back axle)
  hAxles = []
  for n in range(2):
    hNode = RssPy.SceneNode()
    isOK(api.addSceneNode(hNode,carCS.hNode))
    isOK(api.setSceneElement(hNode,hAxleMesh))
    hAxles.append(hNode)
  update_wheels(0)
elif mesh_to_include == 'lead_car':
  carCS = CoordSys(None,hCarMesh)
  carCS.pos = (6.,0.,0.)
  carCS.lin = (15.,0.,0.)
  hAxles = []
  for n in range(2):
    hNode = RssPy.SceneNode()
    isOK(api.addSceneNode(hNode,carCS.hNode))
    isOK(api.setSceneElement(hNode,hAxleMesh))
    hAxles.append(hNode)
  update_wheels(0)
elif mesh_to_include == 'semi':
  semiCS = CoordSys(None,hSemiMesh)
  semiCS.rot[0,0] = -1; semiCS.rot[1,1] = -1
  semiCS.pos = (42.,10.5,0.)
  semiCS.lin = (-25.,0.,0.)
elif mesh_to_include == 'terrain':
  sceneCS = [CoordSys(None,hSceneMesh),CoordSys(None,hSceneMesh)]
  sceneCS[1].pos = (150.,0.,0.)





# create and assign coatings
hAsphalt = RssPy.Coating()
isOK(api.addCoating(hAsphalt,"DielectricLayers -1,3.18,-0.1,1.0,0.0,0.0"))
isOK(api.mapCoatingToIndices(hAsphalt,(0,1)))
stdDevSurfHt_mm = 1; roughness = 0.2
isOK(api.setCoatingRoughness(hAsphalt,stdDevSurfHt_mm,roughness))
hGlass = RssPy.Coating()
isOK(api.addCoating(hGlass, "DielectricLayers 4.0,6.5,0.0,1.0,0.0,0.0000325 VACUUM"))
isOK(api.mapCoatingToIndex(hGlass,4))

# create EGO vehicle platform, with a semi as the EGO vehicle
hPlatform = RssPy.RadarPlatform()
isOK(api.addRadarPlatform(hPlatform))
platformCS = CoordSys(hPlatform,None)
platformCS.pos = (-0.475,0.,0.)
platformCS.lin = (14.,0.,0.)

# list of coordinate systems to update between frames
if mesh_to_include == 'full_scene':
  csList = (sceneCS[0],sceneCS[1],carCS,semiCS,platformCS)
elif mesh_to_include == 'lead_car':
  csList = (carCS,platformCS)
elif mesh_to_include == 'semi':
  csList = (semiCS,platformCS)
elif mesh_to_include == 'terrain':
  csList = (sceneCS[0],sceneCS[1],platformCS)

# create radar device and antennas
hDevice = RssPy.RadarDevice()
isOK(api.addRadarDevice(hDevice, hPlatform))
radarCS = CoordSys(hDevice)
# set radar device so its initial position is (1.,0.,0.5) m, 5mm below the semi front surface
isOK(api.setCoordSysInParent(hDevice, radarCS.rot, (1.475,0.,0.5), radarCS.lin, radarCS.ang))
hTx = RssPy.RadarAntenna()
isOK(api.addRadarAntennaParametricBeam(hTx,hDevice,RssPy.AntennaPolarization.VERT,30.,60.,1.))
hRx = RssPy.RadarAntenna()
fftbl = api.loadFarFieldTable(os.path.join(paths.antenna_device_library,"beam.ffd"))
isOK(api.addRadarAntennaFromTable(hRx,hDevice,fftbl))

# configure radar mode
hMode = RssPy.RadarMode()
isOK(api.addRadarMode(hMode, hDevice))
isOK(api.setRadarModeActive(hMode,True))
isOK(api.addTxAntenna(hMode,hTx))
isOK(api.addRxAntenna(hMode,hRx))
isOK(api.setRadarModeStartDelay(hMode,0.,RssPy.ModeDelayReference.CENTER_CHIRP))

# configure Pulse-Doppler or FMCW 
centerFreq = 77.e9
bandWidth  = 500.e6
chirpInterval = 60.e-6
numChirpsPerCPI = 256
txMultiplex = RssPy.TxMultiplex.INTERLEAVED
# While FREQ_PULSE output requires a Pulse-Doppler waveform, ADC_SAMPLES
# output requires an FMCW waveform. RANGE_PULSE and RANGE_DOPPLER outputs
# can be activated with those two waveforms. Here, an FMCW waveform is
# selected for RANGE_PULSE and RANGE_DOPPLER outputs. 
if activeOutput == 0: # only FREQ_PULSE
  numFreq = 256
  isOK(api.setPulseDopplerWaveformSysSpecs(
    hMode,
    centerFreq,
    bandWidth,
    numFreq,
    chirpInterval,
    numChirpsPerCPI,
    txMultiplex))
else:
  chirpType = RssPy.FmcwChirpType.ASCENDING_RAMP
  adcSampleRate = 10.e6
  numADCSPerChirp = 256
  isIQChannel = True
  isOK(api.setChirpSequenceFMCWFromSysSpecs(
    hMode,
    chirpType,
    centerFreq,
    bandWidth,
    adcSampleRate,
    numADCSPerChirp,
    chirpInterval,
    numChirpsPerCPI,
    isIQChannel,
    txMultiplex))

# optional: radar mode Rx thermal noise and channel gain/loss
# rxGainType = RssPy.RxChannelGainSpecType.USER_DEFINED
# rxGain_dB = 20.
# isOK(api.setRadarModeRxChannelGain(hMode,rxGainType,rxGain_dB))
# T = 290 # temperature in kelvin
# kB = 1.380649e-23 # Boltzmann constant
# rxNoise_dB = 10*m.log10(kB*T*bandWidthHz) # assuming Rx channel bandwidth is same as Tx
# isOK(api.setRadarModeRxThermalNoise(hMode,True,rxNoise_dB))

# optional Tx phase noise
# pedHeight_dB = -80
# pedWidth_Hz = 1.e6
# kRolloff = 3.
# psdType = RssPy.PhaseNoisePSDModel.PEDESTAL
# isOK(api.setTxPhaseNoisePSD(hMode, psdType, pedHeight_dB, pedWidth_Hz, kRolloff))

# optional: set Tx amplitudes, can be updated at any time before or after compute response
# isOK(api.setTxAntennaAmplitude(hMode,hTx,1+1j))
# (ret,numPulseAmpls) = api.numTxAntennaAmplitudes(hMode,hTx); isOK(ret)
# txAmpls = [(1+1j)/np.sqrt(2) for i in range(numPulseAmpls)]
# isOK(api.setTxAntennaAmplitudes(hMode,hTx,txAmpls))

# assign modes to devices
print(api.listGPUs())
devIDs = [0]; devQuotas = [0.8]; # limit RTR to use 80% of available gpu memory
isOK(api.setGPUDevices(devIDs,devQuotas))
maxNumRayBatches = 25
isOK(api.autoConfigureSimulation(maxNumRayBatches))
#isOK(api.assignAllRadarModesToGPU(0))
#isOK(api.assignRadarModesToGPUs([hMode],[0]))

# activate range-pulse or range-doppler output
sideLobeLevelDb = 50.
rSpecs = "hann," + str(sideLobeLevelDb)
dSpecs = "hann," + str(sideLobeLevelDb)
rPixels = 1024
dPixels = 1024
centerVel = 0.
rangeRefVal = 0.
if activeOutput == 0 or activeOutput == 3:
  pass # do not activate RANGE_DOPPLER or RANGE_PULSE
elif activeOutput == 1:
  isOK(api.activateRangePulseResponse(hMode, rPixels, RssPy.ImagePixelReference.BEGIN, rangeRefVal, rSpecs))
elif activeOutput == 2:
  isOK(api.activateRangeDopplerResponse(hMode, rPixels, dPixels, centerVel, rSpecs, dSpecs))

# initialize solver settings
isOK(api.setMaxNumRefl(1))
isOK(api.setMaxNumTrans(0))
terrainRaySpacing = 0.5; # m
isOK(api.setRaySpacing(terrainRaySpacing))
isOK(api.setPrivateKey("SkipTerminalBncPOBlockage", "true"))
# add focused ray groups around vehicles
groupRaySpacing = 0.01; # m
if mesh_to_include == 'full_scene':
  isOK(api.addFocusedRayGroup(carCS.hNode, groupRaySpacing))
  isOK(api.addFocusedRayGroup(semiCS.hNode, groupRaySpacing))
elif mesh_to_include == 'lead_car':
  isOK(api.addFocusedRayGroup(carCS.hNode, groupRaySpacing))
elif mesh_to_include == 'semi':
  isOK(api.addFocusedRayGroup(semiCS.hNode, groupRaySpacing))

# optional check if RSS is configured
# this will also be checked before response computation
if not api.isReady():
  print("RSS is not ready to execute a simulation:\n")
  print(api.getLastWarnings())

# display setup
#print(api.reportSettings())

# get response domains
(ret,xDomain,yDomain) = api.responseDomains(hMode,outputOptions[activeOutput])
isOK(ret)

# activate radar camera view
isOK(api.initRadarCamera(RssPy.CameraProjection.FISHEYE,RssPy.CameraColorMode.COATING,512,512,255,True,1))
isOK(api.activateRadarCamera())

# compute radar response
fps = 10; dt = 1/fps
T = 10; numFrames = int(T/dt)
responses = []
cameraImages = []
reportRate = 20
for iFrame in range(numFrames):
  if numFrames < reportRate or (iFrame+1) % reportRate == 0:
    print('simulating frame {:d} of {:d}'.format(1+iFrame,numFrames))
  time = iFrame*dt
  for cs in csList:
    cs.update(time)
  if mesh_to_include == 'lead_car':
    update_wheels(time)
  isOK(api.computeResponseSync())
  (ret,response) = api.retrieveResponse(hMode,outputOptions[activeOutput])
  isOK(ret)
  (ret,radarCamera,_,_) = api.retrieveRadarCameraImage(hMode)
  isOK(ret)
  responses.append(response)
  cameraImages.append(radarCamera)

# save responses to numpy file
responses = np.asarray(responses)
np.save(f"{mesh_to_include}_{outputOptions[activeOutput]}_1bounce.npy",responses)

# post-process images into gif
(fig,axes) = plt.subplots()
imData = np.zeros((len(yDomain),len(xDomain)))
if activeOutput == 0: # FREQ_PULSE
  cLims_dB = (-110,-60)
  xDomain *= 1.e3
  axes.set_xlabel('Pulse Time (ms)')
  yDomain *= 1.e-9
  axes.set_ylabel('Frequency (GHz)')
elif activeOutput == 1: # RANGE_PULSE
  cLims_dB = (-160,-80)
  xDomain *= 1.e3
  axes.set_xlabel('Pulse Time (ms)')
  axes.set_ylabel('Range (m)')
elif activeOutput == 2: # RANGE_DOPPLER
  cLims_dB = (-180,-100)
  axes.set_xlabel('Doppler velocity (m/s)')
  axes.set_ylabel('Range (m)')
elif activeOutput == 3: # ADC_SAMPLES
  cLims_dB = (-110,-60)
  xDomain *= 1.e3
  axes.set_xlabel('Pulse Time (ms)')
  yDomain *= 1.e6
  axes.set_ylabel('Chirp local time (us)')
image = plt.imshow(imData,interpolation='bilinear',cmap='jet',\
  aspect='auto',vmin=cLims_dB[0],vmax=cLims_dB[1],\
  extent=[xDomain[0],xDomain[-1],yDomain[0],yDomain[-1]])
images = []
for iFrame,response in enumerate(responses):
    if numFrames < reportRate or (iFrame+1) % reportRate == 0:
      print('processing frame {:d} of {:d}'.format(1+iFrame,numFrames))
    imData = np.rot90(20*np.log10(np.fmax(np.abs(response[0][0]),1.e-30)))
    image.set_data(imData)
    fig.canvas.draw()
    # Fetch ARGB data
    raw_data = fig.canvas.tostring_argb()
    width, height = fig.canvas.get_width_height()
    # Create numpy array
    image_data = np.frombuffer(raw_data, dtype=np.uint8)
    image_data = image_data.reshape((height, width, 4))
    # Discard A channel
    rgb_data = image_data[..., 1:4]
    # Form RGB image
    radarIm = PIL.Image.fromarray(rgb_data, 'RGB')
    cameraIm = PIL.Image.frombytes('RGB',cameraImages[iFrame].shape[0:2],cameraImages[iFrame])
    radarIm = radarIm.resize((int(cameraIm.height*radarIm.width/radarIm.height),cameraIm.height))
    finalIm = PIL.Image.new('RGB',(radarIm.width+cameraIm.width,radarIm.height))
    finalIm.paste(cameraIm,(0, 0))
    finalIm.paste(radarIm,(cameraIm.width, 0))
    images.append(finalIm)
shortname = lambda x: str(x).split('.')[-1]
filename = f"{mesh_to_include}_hello_radar_{shortname(outputOptions[activeOutput])}.gif"
print('writing {}...'.format(filename))
images[0].save(filename,save_all=True,append_images=images[1:],optimize=False,duration=int(1000/fps),loop=0)