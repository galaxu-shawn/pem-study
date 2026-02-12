#
# Copyright ANSYS. All rights reserved.
#
import os
import sys
import numpy as np
import math as m
import matplotlib.pyplot as plt
import PIL.Image
import glob

# create api
libDir = 'C:/Program Files/AnsysEM/Perceive_EM/v252/Win64/lib/'

sys.path.insert(0,libDir)
print(sys.path)
import RssPy
api = RssPy.RssApi()

# function to check for warnings or errors
def isOK(rGpuCallStat):
  if(rGpuCallStat == RssPy.RGpuCallStat.OK):
    return
  elif(rGpuCallStat == RssPy.RGpuCallStat.RGPU_WARNING):
    print(api.getLastWarnings())
  else:
    print(api.getLastError())
    exit()


# new licensing mode, starting Feb 12th, 2024 you will be able to generate a new license with this key
isOK(api.selectApiLicenseMode(RssPy.ApiLicenseMode.PERCEIVE_EM))
# The HPC license can be pre-selected, if it is not, this will first look for HPC_ANSYS, then HPC_ANSYS_POOL
# RssPy.HpcLicenseType.HPC_ANSYS_POOL or RssPy.HpcLicenseType.HPC_ANSYS

# optional environment variable to specify custom location for RTR licensing directory
# the variable should point to the licensingclient/ directory
# the shared_files directory MUST be at least one directory level above licensingclient/
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

# for debugging
#api.setVerbose(True)

# print version and copyright info
print(api.copyright())
print(api.version(True))

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
(hSceneMesh,_) = loadMesh('example_scripts/models/whole-scene-static.stl')
(hCarMesh,_)   = loadMesh('example_scripts/models/mustang-no-wheels.stl')
(hAxleMesh,_)  = loadMesh('example_scripts/models/mustang-axle.stl')
(hSemiMesh,_)  = loadMesh('example_scripts/models/tractor-trailor.stl')

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

# instantiate scene elements and initialize coordinate systems
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
update_wheels(0)

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
platformCS = CoordSys(hPlatform,hSemiMesh)
platformCS.pos = (-0.475,0.,0.)
platformCS.lin = (14.,0.,0.)

# list of coordinate systems to update between frames
csList = (sceneCS[0],sceneCS[1],carCS,semiCS,platformCS)

# create radar device and antennas
hDevice = RssPy.RadarDevice()
isOK(api.addRadarDevice(hDevice, hPlatform))
radarCS = CoordSys(hDevice)
# set radar device so its initial position is (1.,0.,0.5) m, 5mm below the semi front surface
isOK(api.setCoordSysInParent(hDevice, radarCS.rot, (1.475,0.,0.5), radarCS.lin, radarCS.ang))
hTx = RssPy.RadarAntenna()
isOK(api.addRadarAntennaParametricBeam(hTx,hDevice,RssPy.AntennaPolarization.VERT,30.,60.,1.))
hRx = RssPy.RadarAntenna()
fftbl = api.loadFarFieldTable("./antenna_device_library/beam.ffd")
isOK(api.addRadarAntennaFromTable(hRx,hDevice,fftbl))

# configure radar mode
hMode = RssPy.RadarMode()
isOK(api.addRadarMode(hMode, hDevice))
isOK(api.setRadarModeActive(hMode,True))
isOK(api.addTxAntenna(hMode,hTx))
isOK(api.addRxAntenna(hMode,hRx))
isOK(api.setRadarModeStartDelay(hMode,0.,RssPy.ModeDelayReference.CENTER_CHIRP))
minFreqHz = 76.4004e9
maxFreqHz = 76.5996e9
centerFreq = 0.5*(minFreqHz+maxFreqHz)
bandWidthHz = maxFreqHz-minFreqHz
numFreqSamples = 267
cpiDuration = 0.004
numPulseCPI = 200
pulseInterval = cpiDuration / numPulseCPI
txMultiplex = RssPy.TxMultiplex.INTERLEAVED
isOK(api.setPulseDopplerWaveformSysSpecs(hMode, centerFreq, bandWidthHz,
  numFreqSamples, pulseInterval, numPulseCPI, txMultiplex))

# optional: radar mode Rx thermal noise and channel gain/loss
# rxGainType = RssPy.RxChannelGainSpecType.USER_DEFINED
# rxGain_dB = 20.
# isOK(api.setRadarModeRxChannelGain(hMode,rxGainType,rxGain_dB))
# T = 290 # temperature in kelvin
# kB = 1.380649e-23 # Boltzmann constant
# rxNoise_dB = 10*m.log10(kB*T*bandWidthHz) # assuming Rx channel bandwidth is same as Tx
# isOK(api.setRadarModeRxThermalNoise(hMode,True,rxNoise_dB))

# isOK(api.setTxAntennaAmplitude(hMode,hTx,1+1j))
# (ret,numPulseAmpls) = api.numTxAntennaAmplitudes(hMode,hTx); isOK(ret)
# txAmpls = [(1+1j)/np.sqrt(2) for i in range(numPulseAmpls)]
# isOK(api.setTxAntennaAmplitudes(hMode,hTx,txAmpls))
# optional: Tx amplitudes
# can be updated at any time before or after compute response

# assign modes to devices
print(api.listGPUs())
devIDs = [0]; devQuotas = [0.8]; # limit RTR to use 80% of available gpu memory
isOK(api.setGPUDevices(devIDs,devQuotas))
maxNumRayBatches = 25
isOK(api.autoConfigureSimulation(maxNumRayBatches))
#isOK(api.assignAllRadarModesToGPU(0))
#isOK(api.assignRadarModesToGPUs([hMode],[0]))

# activate range-doppler output
sideLobeLevelDb = 50.
rSpecs = "hann," + str(sideLobeLevelDb)
dSpecs = "hann," + str(sideLobeLevelDb)
rPixels = 1024
dPixels = 1024
centerVel = 0.
isOK(api.activateRangeDopplerResponse(hMode, rPixels, dPixels, centerVel, rSpecs, dSpecs))

# initialize solver settings
isOK(api.setMaxNumRefl(3))
isOK(api.setMaxNumTrans(3))
terrainRaySpacing = 0.5; # m
isOK(api.setRaySpacing(terrainRaySpacing))

# add focused ray groups around vehicles
groupRaySpacing = 0.01; # m
isOK(api.addFocusedRayGroup(carCS.hNode, groupRaySpacing))
isOK(api.addFocusedRayGroup(semiCS.hNode, groupRaySpacing))

# optional check if RSS is configured
# this will also be checked before response computation
if not api.isReady():
  print("RSS is not ready to execute a simulation:\n")
  print(api.getLastWarnings())

# display setup
#print(api.reportSettings())

# get response domains
(ret,velDomain,rngDomain) = api.responseDomains(hMode,RssPy.ResponseType.RANGE_DOPPLER)
isOK(ret)

# activate radar camera view
isOK(api.initRadarCamera(RssPy.CameraProjection.FISHEYE,RssPy.CameraColorMode.COATING,512,512,255,True,1))
isOK(api.activateRadarCamera())

# compute radar response
fps = 10; dt = 1/fps
T = 10; numFrames = int(T/dt)
responses = []
cameraImages = []
for iFrame in range(numFrames):
  print('simulating frame {:d} of {:d}'.format(1+iFrame,numFrames))
  time = iFrame*dt
  for cs in csList:
    cs.update(time)
  update_wheels(time)
  isOK(api.computeResponseSync())
  (ret,response) = api.retrieveResponse(hMode,RssPy.ResponseType.RANGE_DOPPLER)
  isOK(ret)
  (ret,radarCamera,_,_) = api.retrieveRadarCameraImage(hMode)
  isOK(ret)
  responses.append(response)
  cameraImages.append(radarCamera)


# post-process images into gif
cLims_dB = (-240,-120)
(fig,axes) = plt.subplots()
imData = np.zeros((len(rngDomain),len(velDomain)))
image = plt.imshow(imData,interpolation='bilinear',cmap='jet',\
  aspect='auto',vmin=cLims_dB[0],vmax=cLims_dB[1],\
  extent=[velDomain[0],velDomain[-1],rngDomain[0],rngDomain[-1]])
axes.set_xlabel('Doppler velocity (m/s)')
axes.set_ylabel('Range (m)')
images = []
for iFrame,response in enumerate(responses):
    print('processing frame {:d} of {:d}'.format(1+iFrame,numFrames))
    imData = np.rot90(20*np.log10(np.fmax(np.abs(response[0][0]),1.e-30)))
    image.set_data(imData)
    fig.canvas.draw()

    # Updated method to create image from canvas
    buf = np.asarray(fig.canvas.buffer_rgba())
    radarIm = PIL.Image.fromarray(buf[:,:,:3])  # Only take RGB channels

    cameraIm = PIL.Image.frombytes('RGB',cameraImages[iFrame].shape[0:2],cameraImages[iFrame])
    radarIm = radarIm.resize((int(cameraIm.height*radarIm.width/radarIm.height),cameraIm.height))
    finalIm = PIL.Image.new('RGB',(radarIm.width+cameraIm.width,radarIm.height))
    finalIm.paste(cameraIm,(0, 0))
    finalIm.paste(radarIm,(cameraIm.width, 0))
    images.append(finalIm)
print('writing hello_radar.gif...')
images[0].save('hello_radar.gif',save_all=True,append_images=images[1:],optimize=False,duration=int(1000/fps),loop=0)
