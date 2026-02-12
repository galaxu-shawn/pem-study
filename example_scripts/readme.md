# Examples

## P2P Examples
P2P examples calculate antenna-to-antenna coupling. These examples return total fields. The antenna devices do not have to be co-located. Each antenna device can be placed on any antenna platform, located anywhere in the scene.

## Radar Examples
Radar examples return only scattered field components. With radar, the Tx/Rx are required to be co-located on the same platform.

## Utils
Utils provides a library of functions that can be used to create and manipulate the geometry, mesh, and setup the simulation.

The utilities are provided as a wrapper around the Perceive EM API, intended to automate and simplify the process of setting up and running simulations. The utilities are designed to be used in a Python environment. 

### Using utilities

#### Actors()
Example usage: actors.py Actors() class
This class can be used to store all actors, this class is a useful class that extends the use of Actor() class.
See Example_Using_Helper_Python_Utilities.py for an example using Actors() class.

```python
from utilities.actor import Actors
all_actors = Actors()
actor_name = all_actors.add_actor('actor_name',filename='path/to/stl/file.stl')

# set position of actor
all_actors.actors[actor_name].coord_sys.pos = (3.5, -5, 0.)
all_actors.actors[actor_name].coord_sys.update()

# update actor based on time step, used when actors include animations or are moving (ie if a velocity is assigned, position can be updated by updating while including a time stamp)
all_actors.actors[actor_name].update_actor(time=time)

all_actors_names = all_actors.get_actor_names()

```


#### Actor()
Example usage: actor.py Actor()
This class is used to initialize and store all information related to scene actors. It is also called by Actors() class if we want to help manage multiple actors in the scene. But it can also be called directly to manage a single actor.
```python

from utilities.actor import Actor

# dictionary to store all actors, is used for visualization and scene updates
all_scene_actors = {} 

# load an empty actor
all_scene_actors['actor_name'] = Actor()
# add a mesh element to this actor
all_scene_actors['actor_name'].add_part(filename='path/to/stl/file.stl',name='name')

# load an actor and add a mesh element to it in a single step
all_scene_actors['actor_name2'] = Actor(filename='path/to/stl/file.stl')

# load a multi-part actor from a file (.json). This json file will contain
# the actor's name and the file paths to the stl files. There are special
# actors types that can be defined and update functions to deal with animations
# for example, a vehicle defined with this json file will have a function to
# update the position and velocity of the wheels to add rotation and movement.
# each part of the vehicle will be a separate mesh element, and imported as a 
# separate part of the actor. This actor will have children actors that deal
# with the hierarchy of the vehicle, for example, the wheels will be children
# of the vehicle actor
all_scene_actors['actor_name3'] = Actor(filename='path/to/json/file.json')

# load a multi-part actor from a file (.dae). This colloda file contains animations of the actor
# right now this supports an animated pedestrian that is wakling. 
all_scene_actors['actor_name3'] = Actor(filename='path/to/json/file.json')

```
#### Initialize position of actor
```python

# initialize position of an actor. This is teh generic initialization, see below for vehicle and pedestrian.
all_scene_actors['actor_name'].coord_sys.pos = (-15.,-0.0, 0.)
all_scene_actors['actor_name'].coord_sys.lin = (10.,-0.0, 0.)

# or if Actors() class is used setting the same parameters would look like the following:
all_actors.actors[actor_name].coord_sys.pos = (-15.,-0.0, 0.)
all_actors.actors[actor_name].coord_sys.lin = (10.,-0.0, 0.)

# multi-part actors can be initialized with one additional property, velocity_mag
# velocity_mag allows the user to define the velocity magnitude instead of a vector.
# when this is assigned, the velocity vector will be calculated based on the direction
# the actor is facing. This is useful for vehicles and pedestrians, where we can set the
# orientation and not worry about directly supplying the velocity vector.
all_scene_actors['actor_name3'].coord_sys.pos = (0.,-0.0, 0.)
# additional euler_to_rot() can be used to convert euler angles to rotation matrix
# this function is found within the rotation class
all_scene_actors['actor_name3'].coord_sys.rot = np.eye(3) # no rotation applied, actor is facing +X direction 
all_scene_actors['actor_name3'].velocity_mag = 3.

```


#### AntennaDevice
Example usage: antenna_device.py
```python

from utilities.antenna_device import AntennaDevice

# We are going to create a radar that is attached an actor, the hierarchy will look like this:
#
#  Actor --> Radar Platform --> Antenna Device -->  Antenna (Tx/Rx)
#

# Antenna Device
# create antenna device and antennas, the antenna platform is loaded from json file. It is then created in reference to the
# actor node. 
ant_device = AntennaDevice('example_1tx_1rx.json',parent_h_node=all_scene_actors['actor_name3'].h_node)

# When loading an antenna from json, we need to first initialize the mode defined in the json file, if multiple modes are
# defined we need to select which one.
ant_device.initialize_mode(mode_name='mode1')

# The position of the device is place 2.5 meters in front of the actor, and 1 meter above the ground
# the device itself does not have any meshes attached to it.
ant_device.coord_sys.pos = (2.5,0.,1)
ant_device.coord_sys.update()

# Once initialized, we can add the antennas to the device, select which mode we
# and add the antennas to that node. The antennas are defined inside the json file. We have an additional option
# to load the far-field pattern data as a mesh, and create a pyvista actor that we can later visualize
# once the antennas are added, we can add the mode to the device
ant_device.add_antennas(mode_name='mode1',load_pattern_as_mesh=True,scale_pattern=1)
ant_device.add_mode(mode_name='mode1')

```

#### MaterialManager()
Example usage: material_manager.py
```python
from utilities.materials import MaterialManager
mat_manager = MaterialManager()

# load a material, the actor is expecting a index value. This value is defined in the material_list.json file
# getting the index by name is shown below.
all_scene_actors['actor_name'] = Actor(filename ='../models/A10.stl',mat_idx=mat_manager.get_material('aluminum'))

```


#### ModelVisualization()
Example usage: model_visualization.py
```python
from utilities.model_visualization import ModelVisualization

# initialize the scene modeler
modeler = ModelVisualization(all_scene_actors,
                             show_antennas=True,
                             output_movie_name = f'output_geometry.mp4',
                             fps=10,
                             overlay_results=True,
                             vel_domain=None,
                             rng_domain=None,
                             freq_domain=None,
                             pulse_domain=None,
                             shape = (100,100),
                             cmap='jet',
                             camera_orientation='actor_name',
                             camera_attachment='follow')

# arguments for this class are:
# all_scene_actors: dictionary of all actors in the scene
# show_antennas: boolean, if true, the antennas will be shown in the scene
# output_movie_name: string, name of the output movie
# fps: int, frames per second
# overlay_results: boolean, if true, the results will be overlaid on the scene
# vel_domain: array, velocity domain, used to label axes in plot
# rng_domain: array, range domain, used to label axes in plot
# freq_domain: array, frequency domain, used to label axes in plot
# pulse_domain: array, pulse domain, used to label axes in plot
# shape: tuple, shape of the plot, used to create the matplotlib plot
# cmap: string, colormap to use in the plot
# camera_attachment: string, name of the actor to follow. If None, it will use default pyvista view
# camera_orientation: string, how should the camera be attached to the actor, can be: 'follow' 'scene_top' 'top' 'front' 'side' 'radar'

# Using the dictionary of actors that was created earlier, we can visualize the scene using pyvista. This method
# will query every node in the scene, get the position/orientation and display the mesh. This is not used for 
# simulation, but just to see what is going on in the scene. This will impact performance, so for high performance
# applications, do not use this method.

# example simulation loop updating the scene and updating the pyvista plotter
for iFrame in tqdm(range(numFrames)):
    time = iFrame*dt
    # update all coordinate systems
    for actor in all_scene_actors:
        # Actors can have actor_type value, this will handle sub-parts of the actors. For example, if we have a vehicle
        # it will update the tires rotation and velocity, and the vehicle linear velocity based on its orientation
        if all_scene_actors[actor].actor_type == 'vehicle':
            all_scene_actors[actor].update_vehicle_4_wheel(time=time,dt=dt)
        elif all_scene_actors[actor].actor_type == 'pedestrian':
            all_scene_actors[actor].update_pedestrian(time=time,dt=dt)
        else:
            # general update based on linear velocity
            cur_pos = all_scene_actors[actor].coord_sys.pos
            cur_lin = all_scene_actors[actor].coord_sys.lin
            new_pos = cur_pos + dt*cur_lin
            all_scene_actors[actor].coord_sys.pos = new_pos
            all_scene_actors[actor].coord_sys.update()

        # run simulation
        api_core.isOK(api.computeResponseSync())
        (ret, response) = api.retrieveResponse(ant_device.modes['mode1'], api_core.RssPy.ResponseType.RANGE_DOPPLER)
        # calculate response in dB to overlay in pyvista plotter
        imData = np.rot90(20 * np.log10(np.fmax(np.abs(response[0][0]), 1.e-30)))
        modeler.mpl_ax_handle.set_data(imData) # update pyvista matplotlib plot

    # update the pyvista plotter, this will use the updated actor positions and orientations to update the scene
    modeler.update_frame()
    
# close the plotter when completed
modeler.close()

```