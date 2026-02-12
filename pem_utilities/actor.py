"""
Created on Fri Jan 26 15:00:00 2024

@author: asligar
"""
import numpy as np
import json
import os
import cv2
import pyvista as pv
import copy
import glob
import collada
import scipy
import scipy.interpolate
import random
import string
from scipy.spatial.transform import Rotation
from pathlib import Path

# this is only used for webcam based actor creation, only used for that specific example, not needed for general use
# and is a larger library, so will not require it for general use
try:
    import mediapipe as mp
    mediapipe_installed = True
except:
    mediapipe_installed = False

try:
    import mitsuba as mi

    mi.set_variant('scalar_rgb')
except:
    pass

from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API
from pem_utilities.path_helper import get_repo_paths
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.load_mesh import MeshLoader
from pem_utilities.usd_util import UsdActor
from pem_utilities.materials import MaterialManager
from pem_utilities.rotation import euler_to_rot, rotate_vector_from_rot, rot_to_euler

pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 



class Actors:
    def __init__(self, material_manager=None):
        """
        Initialize an Actors instance.

        This class is used to store multiple actors in a scene. It is used to manage the actors in a scene.

        Parameters:
        ------------
        material_manager : MaterialManager instance, optional
            The material manager for the actor. If not provided, a new MaterialManager instance will be created.
            If provided, the material_manager will be used for all actors in the scene, example usage would be when an ITU
            material that is generated on the fly and is used for all actors in the scene. No material .json file would be
            availble in this case and the materials can be shared across all actors.
        
        Attributes:
        ------------
        mat_manager : MaterialManager instance
            The material manager for the actor. It is either the provided material_manager or a new MaterialManager instance.
        actors : dict
            A dictionary to store the actors. The keys are the actor names and the values are the actor instances.

        Examples:
        ---------
        Create and manage multiple actors in a complex scene with vehicles, pedestrians, and infrastructure:

        >>> import os
        >>> from utilities.actor import Actors
        >>> from utilities.materials import MaterialManager
        >>> 
        >>> # Initialize material manager and actors container
        >>> mat_manager = MaterialManager()
        >>> all_actors = Actors(material_manager=mat_manager)
        >>> 
        >>> # Add terrain/infrastructure actors
        >>> road_name = all_actors.add_actor(
        ...     filename=os.path.join(model_path, 'intersection.stl'),
        ...     mat_idx=mat_manager.get_index('asphalt'),
        ...     color='black'
        ... )
        >>> 
        >>> overpass_name = all_actors.add_actor(
        ...     filename=os.path.join(model_path, 'Overpass.stl'),
        ...     mat_idx=mat_manager.get_index('pec'),
        ...     color='grey'
        ... )
        >>> 
        >>> # Add multi-part vehicles from JSON (preserves hierarchy, enables wheel rotation)
        >>> vehicle1_name = all_actors.add_actor(
        ...     filename=os.path.join(model_path, 'Audi_A1_2010/Audi_A1_2010.json')
        ... )
        >>> vehicle2_name = all_actors.add_actor(
        ...     filename=os.path.join(model_path, 'Audi_A1_2010/Audi_A1_2010.json')
        ... )
        >>> 
        >>> # Add animated windmill (multi-part actor with rotating blades)
        >>> windmill_name = all_actors.add_actor(
        ...     filename=os.path.join(model_path, 'Wind_Turbine/Wind_Turbine.json')
        ... )
        >>> 
        >>> # Add animated pedestrian from DAE file
        >>> pedestrian_name = all_actors.add_actor(
        ...     filename=os.path.join(model_path, 'Walking_speed50_armspace50.dae'),
        ...     target_ray_spacing=0.1  # Adaptive ray spacing for detailed simulation
        ... )
        >>> 
        >>> # Add simple mesh actor
        >>> truck_name = all_actors.add_actor(
        ...     filename=os.path.join(model_path, 'tractor-trailor.stl'),
        ...     mat_idx=mat_manager.get_index('pec')
        ... )
        >>> 
        >>> # Set initial positions and velocities for vehicles
        >>> all_actors.actors[vehicle1_name].coord_sys.pos = (-15.0, -0.0, 0.0)
        >>> all_actors.actors[vehicle1_name].coord_sys.rot = euler_to_rot(psi=90, deg=True)
        >>> all_actors.actors[vehicle1_name].velocity_mag = 10.0  # 10 m/s velocity
        >>> all_actors.actors[vehicle1_name].coord_sys.update()
        >>> 
        >>> all_actors.actors[vehicle2_name].coord_sys.pos = (15.0, 0.0, 0.0)
        >>> all_actors.actors[vehicle2_name].coord_sys.rot = euler_to_rot(psi=-90, deg=True)
        >>> all_actors.actors[vehicle2_name].velocity_mag = 8.0
        >>> all_actors.actors[vehicle2_name].coord_sys.update()
        >>> 
        >>> # Set pedestrian position and walking speed
        >>> all_actors.actors[pedestrian_name].coord_sys.pos = (0.0, 10.0, 0.0)
        >>> all_actors.actors[pedestrian_name].velocity_mag = 1.5  # Walking speed
        >>> all_actors.actors[pedestrian_name].coord_sys.update()
        >>> 
        >>> # Position static infrastructure
        >>> all_actors.actors[windmill_name].coord_sys.pos = (50.0, 50.0, 0.0)
        >>> all_actors.actors[windmill_name].coord_sys.update()
        >>> 
        >>> # Get list of all actor names
        >>> actor_names = all_actors.get_actor_names()
        >>> print(f"Scene contains {len(actor_names)} actors: {actor_names}")
        >>> 
        >>> # Update all actors for animation (in simulation loop)
        >>> for time_step in range(100):
        ...     current_time = time_step * 0.1  # 0.1 second steps
        ...     for actor_name in actor_names:
        ...         all_actors.actors[actor_name].update_actor(time=current_time)
        ...     # Perform electromagnetic simulation at this time step
        ...     # ...

        Notes:
        ------
        - Multi-part actors (vehicles, windmills, pedestrians) automatically handle complex animations
        - Use velocity_mag for vehicles/pedestrians to set speed while direction follows rotation
        - JSON-defined actors preserve hierarchical structure and enable specialized behaviors
        - The update_actor() method handles position updates and animations based on actor type
        - target_ray_spacing parameter creates adaptive simulation grids for detailed objects
        """
        if material_manager is not None:
            self.mat_manager = material_manager
        else:
            self.mat_manager = MaterialManager()
        self.actors = {}

    def add_actor(self, name='actor', mesh=None, h_mesh=None, h_node=None, parent_h_node=None, coord_sys=None,
                  filename=None, is_antenna=False, mat_idx=0, target_ray_spacing=None,
                  use_linear_velocity_equation_update=True,update_rot_based_on_ang_vel=True,
                  color=None, transparency=None, scale_mesh=1.0,
                  material_manager=None, actor=None, include_texture=False,map_texture_to_material=False,
                  clip_at_bounds=None,center_geometry=None,add_mesh_offset=None,
                  webcam=None,webcam_upper_body_only=False,generator=None,
                  dynamic_generator_updates=True,use_curved_physics=False,use_experimental_dae_load=False):
        """
        Add an actor to the scene.

        Parameters:
        ------------
        name : str, optional
            The name of the actor. If not provided, 'actor' will be used. If the name already exists in the scene,
            it will be incremented until a unique name is found.
        mesh : pyvista mesh, optional
            The pyvista mesh of the actor, not related to the simulation.
        h_mesh : handle, optional
            The handle for the mesh that is used in the simulation.
        h_node : handle, optional
            The handle for the node that is used in the simulation.
        parent_h_node : handle, optional
            The handle for the parent node that is used in the simulation.
        coord_sys : CoordSys instance, optional
            The coordinate system for the actor.
        filename : str, optional
            The filename of the actor's mesh.
        is_antenna : bool, optional
            If the actor is an antenna or not.
        mat_idx : int, optional
            The material index for the actor.
        target_ray_spacing : float, optional
            The target ray spacing for the actor.
        use_linear_velocity_equation_update : bool, optional
            If True, an actor's position will be updated using its linear velocity and time.
        color : list, optional
            The color of the actor.
        transparency : float, optional
            The transparency of the actor.
        scale_mesh : float, optional
            The scale of the actor's mesh.
        material_manager : MaterialManager instance, optional
            The material manager for the actor.
        actor : Actor instance, optional
            An existing actor to be added to the scene.
        clip_at_bounds: list. optional
            [-x,+x,-y,+y] clip extents. useful for cutting larger terrain geometry to smaller size
        generator: function, optional
            A function that generates a mesh. Used for generating actors on the fly, such as ocean surface.

        Returns:
        ------------
        str
            The name of the actor added to the scene.
        """
        if material_manager is None:
            # if a material manager is provided, use it, else use the default provided by the class
            self.mat_manager = self.mat_manager
        else:
            self.mat_manager = material_manager

        if actor is None:
            actor = Actor(name=name,
                          mesh=mesh,
                          h_mesh=h_mesh,
                          h_node=h_node,
                          parent_h_node=parent_h_node,
                          coord_sys=coord_sys,
                          filename=filename,
                          is_antenna=is_antenna,
                          mat_idx=mat_idx,
                          target_ray_spacing=target_ray_spacing,
                          use_linear_velocity_equation_update=use_linear_velocity_equation_update,
                          update_rot_based_on_ang_vel=update_rot_based_on_ang_vel,
                          color=color, transparency=transparency,
                          scale_mesh=scale_mesh,
                          material_manager=self.mat_manager,
                          include_texture=include_texture,
                          map_texture_to_material=map_texture_to_material,
                          clip_at_bounds=clip_at_bounds,
                          center_geometry=center_geometry,
                          add_mesh_offset=add_mesh_offset,
                          webcam=webcam,webcam_upper_body_only=webcam_upper_body_only,
                          generator=generator,
                          dynamic_generator_updates=dynamic_generator_updates,
                          use_curved_physics=use_curved_physics,
                          use_experimental_dae_load=use_experimental_dae_load)
        elif actor is not None:  # if passing in an existing actor, then just add it to the list
            if not isinstance(actor, Actor):
                return None

        if name in self.actors:
            name = increment_name(name, self.actors.keys())
            actor.name = name
        self.actors[name] = actor
        pem_api_manager.isOK(pem.setName(self.actors[name].h_node,actor.name)) # example of setting the name of an actor in the pem API

        return name

    def get_actor_names(self):
        """
        Get a list of all actor names in the scene.

        Returns:
        --------
        list
            A list containing the names of all actors currently stored in the scene.
        """
        return list(self.actors.keys())
    
    def delete_actor(self, name):
        """
        Remove an actor from the scene by name.

        Parameters:
        ------------
        name : str
            The name of the actor to be removed.

        Returns:
        --------
        bool
            True if the actor was successfully removed, False if the actor was not found.
        """
        # ToDo, also remove from teh modeler visualization. Right now it will just remove it from the simulation
        # but leave it in the modeler
        if name in self.actors:
            # remove the actor from the pem simulation as well
            pem_api_manager.isOK(pem.deleteSceneNode(self.actors[name].h_node))
            del self.actors[name]


            return True
        else:
            return False


class Actor:
    def __init__(self, name='Actor',mesh=None, h_mesh=None, h_node=None, parent_h_node=None, coord_sys=None,
                 filename=None,is_antenna=False, mat_idx=0, target_ray_spacing=None,
                 use_linear_velocity_equation_update=True,update_rot_based_on_ang_vel=True,
                 color=None, transparency=None, scale_mesh=1.0, material_manager=None,include_texture=False,
                 map_texture_to_material=False,clip_at_bounds=None,center_geometry=False,add_mesh_offset=None,
                 webcam=None,webcam_upper_body_only=False,
                 generator=None,dynamic_generator_updates=True,
                 use_curved_physics=False,use_experimental_dae_load=False):
        """
        Initialize an Actor instance.

        Parameters:
        ------------
        mesh : pyvista mesh, optional
            The pyvista mesh of the object, not related to the simulation.

        h_mesh : handle, optional
            The handle for the mesh that is used in the simulation.

        coord_sys : CoordSys instance, optional
            The coordinate system for the actor.

        pv_actor : pyvista actor, optional
            May be used in the future to modify visualization.

        clip_at_bounds: list. optional
            [-x,+x,-y,+y] clip extents. useful for cutting larger terrain geometry to smaller size
        generator: function, optional
            A function that generates a mesh. Used for generating actors on the fly, such as ocean surface.

        dynamic_genertor_updates: bool, optional
            If True, the generator will be updated dynamically at each time step, otherwise it will be static.
        """

        paths = get_repo_paths()

        if material_manager is not None:
            self.mat_manager = material_manager
        else:
            self.mat_manager = MaterialManager()
        self.time = 0
        self.dt = 0

        self.name = name

        self.bounds = None
        self.color = color
        self.transparency = transparency
        # if True, an actors position will be updated using its linear velocity and time,
        # otherwise it will be updated only using
        self.use_linear_velocity_equation_update = use_linear_velocity_equation_update
        self.update_rot_based_on_ang_vel = update_rot_based_on_ang_vel
        if is_antenna:
            self.actor_type = 'antenna'
        else:
            self.actor_type = 'other'
        self.parts = {}
        self.h_mesh = h_mesh
        self.mesh = mesh
        self.h_node = h_node
        self.scale_mesh = scale_mesh
        self.usd_actor = None
        # properties used for multi-part actors where they individual parts may have different initial conditions,
        # for example a wind turbine has blades that have a constant angular velocity, but the base of the turbine
        # is static.
        self.initial_pos = None
        self.initial_rot = None
        self.initial_lin = None
        self.initial_ang = None
        self.mesh_loader = MeshLoader(material_manager=self.mat_manager)
        self.webcam = webcam # will be webcam object if webcam is used, or video
        self.webcam_frame = None # used to output video showing pose estimate
        self.media_pipe_vid_output = None # initialized vid output from mediapipe (webcam based actor creation)

        self.use_curved_physics = use_curved_physics

        self.use_experimental_dae_load = use_experimental_dae_load
        self.previous_transform = np.eye(4)
        self.is_antenna = is_antenna
        self.clip_at_bounds = clip_at_bounds
        self.center_geometry = center_geometry
        self.add_mesh_offset = add_mesh_offset
        self.json_base_path = None
        if coord_sys is None:
            self.coord_sys = CoordSys(h_node=self.h_node,
                                      h_mesh=self.h_mesh,
                                      parent_h_node=parent_h_node,
                                      target_ray_spacing=target_ray_spacing)
        else:
            # if coord_sys is provided
            self.coord_sys = coord_sys
        # moving h_node for easier access, but it will exist in both places
        self.h_node = self.coord_sys.h_node
        self.dynamic_generator_updates = dynamic_generator_updates

        if generator is not None:
            # a generator is a procedural method for generating geometry. It can be used if a geometry is generated
            # using another script and/or could be updated dynamically. One example of this is the seastate.py
            # generator. This will update a geometry/mesh that is defined parametrically and updated based on time.

            # a generator must have a function called generate_mesh() that returns a pyvista mesh (Polydata)
            self.add_parts_from_generator(generator=generator,mat_idx=mat_idx)
            self.generator = generator

        elif filename is not None and self.h_mesh is None:
            if filename == 'webcam':
                if not mediapipe_installed:
                    raise ImportError("mediapipe is not installed. Please install (pip install mediapipe) it to use webcam actors.")
                 # Dynamically resolve the path to the api_core module
                model_path = paths.models
                lower_body_parts = None
                if webcam_upper_body_only:
                    lower_body_parts = ['Hips.stl', 'LEFT_KNEE.stl', 'RIGHT_KNEE.stl', 'RIGHT_HIP.stl', 'LEFT_HIP.stl']
                self.add_parts_from_folder(folder=os.path.join(model_path,'webcam_person'),
                                           mat_idx=mat_idx,
                                           include_texture=include_texture,
                                           map_texture_to_material=map_texture_to_material,
                                           exclude_list=lower_body_parts)
                self.actor_type = 'webcam'
            else:
                extension = os.path.splitext(filename)[1][1:]
                extension = extension.lower()
                if extension == 'json':
                    self.add_parts_from_json(filename=filename)
                elif extension == 'dae' or extension == 'daez':
                    self.add_parts_from_dae(filename=filename)
                elif extension == 'xml':  # mitsuba scene file
                    self.add_parts_from_xml(filename=filename)
                elif extension == 'usd' or extension == 'usdz':  # usd scene file
                    self.add_parts_from_usd(filename=filename,mat_idx=mat_idx)
                elif os.path.isdir(filename):
                    self.add_parts_from_folder(folder=filename,mat_idx=mat_idx,include_texture=include_texture,map_texture_to_material=map_texture_to_material)
                else:
                    self.add_part(filename=filename, mat_idx=mat_idx,include_texture=include_texture,map_texture_to_material=map_texture_to_material)

    def add_part(self, name='geo', mesh=None, filename=None, parent_h_node=None, mat_idx=0, color=None,include_texture=False,map_texture_to_material=False):
        """
        Add a part to the actor.

        This method allows adding individual mesh parts to an actor. Parts can be loaded from files
        or provided as existing mesh objects. Each part becomes a child component of the actor.

        Parameters:
        ------------
        name : str, optional
            The name of the part. If not provided, 'geo' will be used. If the name already exists,
            it will be incremented until a unique name is found.
        mesh : pyvista mesh, optional
            A pre-existing pyvista mesh object to add as a part.
        filename : str, optional
            Path to a mesh file to load as a part. Supports various formats (.stl, .obj, .vtp, etc).
        parent_h_node : handle, optional
            The handle for the parent node. If not provided, uses the actor's root node.
        mat_idx : int, optional
            The material index for the part. Default is 0.
        color : list, optional
            RGB color values for the part [r, g, b] where values are between 0-1.
        include_texture : bool, optional
            Whether to include texture information when loading. Default is False.
        map_texture_to_material : bool, optional
            Whether to map texture to material properties. Default is False.

        Returns:
        --------
        str
            The name of the added part.
        """

        if name in self.parts:
            name = increment_name(name, self.parts.keys())

        # material must be loaded into api
        self.mat_manager.load_material(mat_idx)

        # use root CS h_node if parent_h_node is not provided
        if parent_h_node is None:
            parent_h_node = self.h_node

        h_mesh = None
        if mesh is None and filename is not None: #if no mesh is provided, then load the mesh from the file
            # mesh_loader = MeshLoader()
            if self.json_base_path is not None:
                filename = os.path.join(self.json_base_path, filename)
            h_mesh, mesh = self.mesh_loader.loadMesh(filename=filename,
                                                    mat_idx=mat_idx,
                                                    scale_mesh=self.scale_mesh,
                                                    include_texture=include_texture,
                                                    map_texture_to_material=map_texture_to_material,
                                                    clip_at_bounds=self.clip_at_bounds,
                                                    center_geometry=self.center_geometry,
                                                    add_mesh_offset=self.add_mesh_offset,
                                                    use_curved_physics=self.use_curved_physics)
            if hasattr(mesh, 'bounds'):
                self._update_actor_bounds(mesh.bounds)
        elif mesh is not None and filename is None: # mesh already provided,
            # mesh_loader = MeshLoader()
            h_mesh, mesh = self.mesh_loader.loadMesh(mesh=mesh,
                                                    mat_idx=mat_idx,
                                                    scale_mesh=self.scale_mesh,
                                                    include_texture=include_texture,
                                                    map_texture_to_material=map_texture_to_material,
                                                    add_mesh_offset=self.add_mesh_offset,
                                                    use_curved_physics=self.use_curved_physics)
            if hasattr(mesh, 'bounds'):
                self._update_actor_bounds(mesh.bounds)
        # if mesh and filename are both none, it will add an empty actor
        self.parts[name] = Actor(mesh=mesh,
                                 h_mesh=h_mesh,
                                 parent_h_node=parent_h_node,
                                 color=color,
                                 include_texture=include_texture,
                                 map_texture_to_material=map_texture_to_material,
                                 dynamic_generator_updates=self.dynamic_generator_updates)


        return name

    def add_parts_from_json(self, name='geo', filename=None):
        """
        Add parts to the actor from a JSON configuration file.

        This method loads actor parts defined in a JSON file. The JSON file contains
        information about the actor's class, parts, materials, and properties. It supports
        different actor types like vehicles, birds, quadcopters, and helicopters.

        Parameters:
        ------------
        name : str, optional
            Base name for the actor parts. Default is 'geo'.
        filename : str
            Path to the JSON configuration file containing actor definitions.

        Returns:
        --------
        str
            The name of the loaded actor parts.

        Raises:
        -------
        FileNotFoundError
            If the specified JSON file does not exist.

        Notes:
        ------
        The JSON file should contain:
        - 'name': Actor name (optional, random if not provided)
        - 'class': Actor type (vehicle, bird, quadcopter, helicopter, or other)
        - 'parts': Dictionary of part definitions with file paths and properties
        """

        if os.path.exists(filename):
            filename = os.path.abspath(filename)
            self.json_base_path = os.path.dirname(filename)
            with open(filename) as f:
                actor_json = json.load(f)
        else:
            raise FileNotFoundError(f"File not found: {filename}")

        if 'name' not in actor_json:
            # generate a random string of length 6
            actor_json['name'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        base_name = actor_json['name']

        if 'class' not in actor_json.keys():
            class_type = 'other'
        else:
            class_type = actor_json['class']

        available_classes = ['vehicle', 'pedestrian', 'bicycle', 'motorcycle','bird','quadcopter','helicopter']

        if class_type not in available_classes:
            self._load_other(actor_json)
        elif actor_json['class'].lower() == 'vehicle':
            # self._load_other(actor_json)
            self._load_vehicle_4_wheel(actor_json)
        elif actor_json['class'].lower() == 'bird':
            self._load_bird(actor_json)
        elif actor_json['class'].lower() == 'quadcopter':
            self._load_quadcopter(actor_json)
        elif actor_json['class'].lower() == 'helicopter':
            self._load_helicopter(actor_json)

        return name

    def add_parts_from_folder(self, name='geo',
                              folder=None,
                              mat_idx=0,
                              include_texture=False,
                              map_texture_to_material=False,
                              exclude_list=None):
        """
        Add parts to the actor from all mesh files in a folder.

        This method recursively searches through a folder and loads all supported mesh files
        (.stl, .obj, .vtp) as individual parts of the actor.

        Parameters:
        ------------
        name : str, optional
            Base name for the actor parts. Default is 'geo'.
        folder : str
            Path to the folder containing mesh files to load.
        mat_idx : int, optional
            The material index to apply to all loaded parts. Default is 0.
        include_texture : bool, optional
            Whether to include texture information when loading. Default is False.
        map_texture_to_material : bool, optional
            Whether to map texture to material properties. Default is False.
        exclude_list : list, optional
            List of filenames to exclude from loading. Default is None.

        Returns:
        --------
        str
            The base name used for the loaded parts.

        Raises:
        -------
        FileNotFoundError
            If the specified folder does not exist.

        Notes:
        ------
        Supported file formats: .stl, .obj, .vtp
        The search is recursive and will find files in subdirectories.
        """

        if os.path.exists(folder):
            all_possible_files_stl = glob.glob(f'{folder}/**/*.stl',recursive=True)
            all_possible_files_obj = glob.glob(f'{folder}/**/*.obj', recursive=True)
            all_possible_files_vtp = glob.glob(f'{folder}/**/*.vtp', recursive=True)
            all_possible_files = all_possible_files_stl + all_possible_files_obj + all_possible_files_vtp
        else:
            raise FileNotFoundError(f"File not found: {filename}")

        if exclude_list is not None:
            # remove items in the list that have a match in the exclude list
            all_possible_files = [x for x in all_possible_files if os.path.basename(x) not in exclude_list]

        for filename in all_possible_files:
            only_filename = os.path.basename(filename)
            name, ext = os.path.splitext(only_filename)
            self.add_part(name=name, filename=filename, mat_idx=mat_idx,include_texture=include_texture,map_texture_to_material=map_texture_to_material)

        return name

    def add_parts_from_usd(self, name=None, filename=None, mat_idx=0):
        """
        Add parts to the actor from a USD (Universal Scene Description) file.

        This method loads 3D scenes and models from USD files, which are commonly used
        in film and animation pipelines. It extracts all meshes from the USD file and
        adds them as individual parts to the actor.

        Parameters:
        ------------
        name : str, optional
            Name for the USD actor. If None, a random 6-character string will be generated.
        filename : str
            Path to the USD (.usd or .usdz) file to load.
        mat_idx : int, optional
            The material index to apply to all loaded parts. Default is 0.

        Returns:
        --------
        str
            The name assigned to the USD actor.

        Raises:
        -------
        FileNotFoundError
            If the specified USD file does not exist.

        Notes:
        ------
        This method sets the actor_type to 'usd' and scales meshes according to the
        scale_mesh parameter. The USD file format supports complex scene hierarchies
        and animations.
        """

        self.meshes_from_usd_dict = None
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        self.usd_actor = UsdActor(filename,scale_mesh=self.scale_mesh)
        # scale mesh is already applied in the usd_actor, now in future uses we won't want to scale again
        self.scale_mesh = 1.0
        self.usd_actor.get_all_meshes()

        if name is None:
            # generate a random string of length 6
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        self.actor_type = 'usd'
        self._load_usd(mat_idx=mat_idx)

        return name

    def add_parts_from_dae(self, filename=None):
        """
        Add parts to the actor from a DAE (Collada) animation file.

        This method loads animated characters from DAE files, typically exported from
        mixamo.com or other animation software. It creates an animated pedestrian actor
        with bone structure and animations.

        Parameters:
        ------------
        filename : str
            Path to the .dae or .daez animation file to load.

        Raises:
        -------
        FileNotFoundError
            If the specified DAE file does not exist.

        Notes:
        ------
        The method automatically detects the file extension:
        - .daez files are loaded using AnimatedDAE_CMU for CMU motion capture data
        - .dae files use either AnimatedDAE or AnimatedDAE_experimental based on
          the use_experimental_dae_load setting
        
        Sets the actor_type to 'pedestrian' and initializes velocity_mag to 0.
        """

        self.dae_actors = None
        self.velocity_mag = 0
        self.actor_type = 'pedestrian'
        if os.path.exists(filename):
            filename = os.path.abspath(filename)
            self.json_base_path = os.path.dirname(filename)
        else:
            raise FileNotFoundError(f"File not found: {filename}")

        extension = os.path.splitext(filename)[1][1:]
        if extension == 'daez':
            self.dae_actors  = AnimatedDAE_CMU(filename, speed_factor=1, scale_mesh=self.scale_mesh,
                                         parent_h_node=self.h_node)
        else:
            if self.use_experimental_dae_load:
                self.dae_actors  = AnimatedDAE_experimental(filename, speed_factor=1, scale_mesh=self.scale_mesh, parent_h_node=self.h_node)
            else:
                self.dae_actors  = AnimatedDAE(filename, speed_factor=1, scale_mesh=self.scale_mesh, parent_h_node=self.h_node)

        for name in self.dae_actors .all_actors.keys():
            if name in self.parts:
                name = increment_name(name, self.parts.keys())
            self.dae_actors .all_actors[name].actor_type = 'pedestrian'
            self.parts[name] = self.dae_actors .all_actors[name]

        return

    def add_parts_from_generator(self, name='geo', generator=None,mat_idx=0):
        """
        Add parts to the actor from a procedural geometry generator.

        This method adds parts that are generated procedurally using a generator function.
        The generator must have a generate_mesh() method that returns a pyvista mesh.
        This is useful for dynamic geometry like ocean surfaces or terrain.

        Parameters:
        ------------
        name : str, optional
            Name for the generated part. Default is 'geo'.
        generator : object
            Generator object that must implement a generate_mesh() method returning
            a pyvista mesh. Cannot be None.
        mat_idx : int, optional
            The material index to apply to the generated part. Default is 0.

        Raises:
        -------
        ValueError
            If generator is None or doesn't implement required methods.

        Notes:
        ------
        Sets the actor_type to 'generator' and can be updated dynamically if
        dynamic_generator_updates is True.
        """
        if generator is None:
            # if no generator is provided, then issue an error
            raise ValueError("Generator function must be provided to generate geometry")
        self.generator = generator
        self.actor_type = 'generator'
        init_mesh = generator.generate_mesh()
        name = self.add_part(name=name, mesh=init_mesh,mat_idx=mat_idx)

        return
    def add_parts_from_xml(self, name='geo', filename=None):
        """
        Add parts to the actor from a Mitsuba XML scene file.

        This method loads 3D scenes from Mitsuba XML files and converts them to a format
        compatible with the actor system. It automatically converts materials and exports
        meshes for use in the simulation.

        Parameters:
        ------------
        name : str, optional
            Base name for the actor parts. Default is 'geo'.
        filename : str
            Path to the XML scene file to load.

        Raises:
        -------
        FileNotFoundError
            If the specified XML file does not exist.
        ImportError
            If Mitsuba package is not available and cannot be installed.

        Notes:
        ------
        This method requires the Mitsuba renderer to be installed. If not available,
        it will attempt to install it via conda or pip. The XML file is converted
        to a JSON format internally before loading.
        """

        try:
            import mitsuba as mi
        except ImportError as e:
            # Install mitsuba if package is not already installed
            try:
                os.system("conda install mitsuba")
            except:
                try:
                    os.system("pip install mitsuba")
                except:
                    raise ImportError("Mitsuba could not be installed")

        if os.path.exists(filename):
            filename = os.path.abspath(filename)
            base_path = os.path.dirname(filename)
        else:
            raise FileNotFoundError(f"File not found: {filename}")

        print(f"Starting Mitsuba scene conversion: {filename}")
        scene = mi.load_file(filename)
        converted_file_name = self._load_mitsuba_scene_objects(scene)
        print(f"Sionna XML based scene converted: {converted_file_name}")
        self.add_parts_from_json(filename=converted_file_name)

    def _load_mitsuba_scene_objects(self, scene):
        """
        Load the scene objects available in the scene
        """

        scene_name = 'scene'
        scene_dict = {"name": scene_name, "version": 1, "type": "mitsuba_scene", "class": "other", "parts": {}}

        # Parse all shapes in the scene
        # create temp path for file output
        cur_dir = os.path.abspath(__file__)
        cur_dir = os.path.abspath(os.path.join(os.path.dirname(cur_dir), '..'))
        cur_dir = os.path.abspath(os.path.join(cur_dir, 'output/tmp_cache'))

        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

        all_names = []
        for obj_id, s in enumerate(scene.shapes()):
            # Only meshes are handled
            if not isinstance(s, mi.Mesh):
                raise TypeError('Only triangle meshes are supported')

            # Setup the material
            mat_name = s.bsdf().id()
            if mat_name.startswith("mat-"):
                mat_name = mat_name[4:]
            if mat_name.startswith("itu_"):
                mat_name = mat_name[4:]

            # Instantiate the scene objects
            name = s.id()
            if name.startswith('mesh-'):
                name = name[5:]
            name = 'part-' + name  # Percieve EM requires all parts to be appended with part- in order to import in
            name = increment_name(name, all_names)
            mesh_filename = f'{name}.ply'
            full_file_name = os.path.join(cur_dir, mesh_filename)
            s.write_ply(full_file_name)
            pv_mesh = pv.read(full_file_name)
            mesh_filename_obj = f'{name}.stl'
            full_file_name_obj = os.path.join(cur_dir, mesh_filename_obj)
            pv_mesh.save(full_file_name_obj)
            color_map = {'marble': [0.701101, 0.644479, 0.485150],
                         'metal': [0.219526, 0.219526, 0.254152],
                         'brick': [0.401968, 0.111874, 0.086764],
                         'wood': [0.266356, 0.109462, 0.059511],
                         'concrete': [0.539479, 0.539479, 0.539480]}
            if mat_name in color_map.keys():
                color = color_map[mat_name]
            else:
                color = [0.5, 0.5, 0.5]
            temp_dict = {"file_name": mesh_filename_obj, "properties": {"material": mat_name, "color": color}}
            scene_dict['parts'][name] = temp_dict

            if os.path.exists(full_file_name):
                os.remove(full_file_name)

            json_filename = f'{scene_name}.json'
            with open(os.path.join(cur_dir, json_filename), 'w') as f:
                json.dump(scene_dict, f, indent=4)
        return os.path.join(cur_dir, json_filename)

    def _get_children(self, part_dict):
        # parts are organized in a dictionary, with each part having sub fields. If that subfield is also a dict
        # and that sub-fields has a file_name, then it is a child part of this part.
        children = []
        for sub_field in part_dict.keys():
            if isinstance(part_dict[sub_field], dict):
                # for example this will be part_dict['wheel']['file_name'] , wheel will be the child
                if 'file_name' in part_dict[sub_field].keys():
                    children.append(sub_field)
        return children

    def _load_usd(self,mat_idx):
        for name, mesh in self.usd_actor.all_pv_meshes.items():
            # don't scale again, scaled when originally imported during add_parts_from_usd()
            self.add_part(name=name, mesh=mesh, mat_idx=mat_idx)

    def _load_other(self, actor_dict, parent_h_node=None, child_name=None):
        self.velocity_mag = None
        # top level parts are within parts dict
        if 'parts' in actor_dict.keys():
            parts_dict = actor_dict['parts']
        else:  # child parts are not in a 'parts' dictionary
            parts_dict = actor_dict[child_name]
            material = 'pec'
            if "properties" in parts_dict.keys():
                if "material" in parts_dict["properties"].keys():
                    material = parts_dict['properties']['material']

            mat_idx = self.mat_manager.get_index(material)
            name = self.add_part(name=child_name, filename=parts_dict['file_name'], parent_h_node=parent_h_node,
                                 mat_idx=mat_idx,color=self.color)
            if "properties" in parts_dict.keys():
                # these properties are optional, so they may not exist. If they do, then they will be used
                # these are primiarly used to set the intial conditions for actors in the scene that have a
                # constant pos/rot/lin/ang relative to their parent. Useful for things like windturbines, helichopters
                # these property values can be overriden by the user by setting coord_sys.pos/rot/lin/ang
                if "initial_pos" in parts_dict["properties"].keys():
                    pos = parts_dict['properties']['initial_pos']
                    self.parts[name].initial_pos = pos
                if "initial_rot" in parts_dict["properties"].keys():
                    rot = parts_dict['properties']['initial_rot']
                    self.parts[name].initial_rot = rot
                if "initial_lin" in parts_dict["properties"].keys():
                    lin = parts_dict['properties']['initial_lin']
                    self.parts[name].initial_lin = lin
                if "initial_ang" in parts_dict["properties"].keys():
                    ang = parts_dict['properties']['initial_ang']
                    self.parts[name].initial_ang = ang

        # parts should always be appended witih "part-" to avoid confusion with other keys
        list_of_parts = list(parts_dict.keys())
        list_of_parts = [x for x in list_of_parts if "part-" in x]

        for part in list_of_parts:
            material = 'pec'
            if "properties" in parts_dict[part].keys():
                if "material" in parts_dict[part]["properties"].keys():
                    material = parts_dict[part]['properties']['material']
            mat_idx = self.mat_manager.get_index(material)
            color = None
            if "color" in parts_dict[part]["properties"].keys():
                color = parts_dict[part]['properties']['color']
            name = self.add_part(name=part, filename=parts_dict[part]['file_name'], parent_h_node=parent_h_node,
                                 mat_idx=mat_idx, color=color)
            if "properties" in parts_dict[part].keys():
                # these properties are optional, so they may not exist. If they do, then they will be used
                # these are primiarly used to set the intial conditions for actors in the scene that have a
                # constant pos/rot/lin/ang relative to their parent. Useful for things like windturbines, helichopters
                # these property values can be overriden by the user by setting coord_sys.pos/rot/lin/ang
                if "initial_pos" in parts_dict[part]["properties"].keys():
                    pos = parts_dict[part]['properties']['initial_pos']
                    self.parts[name].initial_pos = pos
                if "initial_rot" in parts_dict[part]["properties"].keys():
                    rot = parts_dict[part]['properties']['initial_rot']
                    self.parts[name].initial_rot = rot
                if "initial_lin" in parts_dict[part]["properties"].keys():
                    lin = parts_dict[part]['properties']['initial_lin']
                    self.parts[name].initial_lin = lin
                if "initial_ang" in parts_dict[part]["properties"].keys():
                    ang = parts_dict[part]['properties']['initial_ang']
                    self.parts[name].initial_ang = ang

            # recursively go through the parts and add them to the scene
            children = self._get_children(parts_dict[part])
            for child in children:
                part_dict = parts_dict[part]
                self._load_other(part_dict, self.parts[part].h_node, child_name=child)

    def _load_quadcopter(self, actor_dict, parent_h_node=None, child_name=None):
        self.velocity_mag = None
        self.rotor_ang = [0,0,0]
        self.actor_type = 'quadcopter'
        # top level parts are within parts dict
        if 'parts' in actor_dict.keys():
            parts_dict = actor_dict['parts']
        else:  # child parts are not in a 'parts' dictionary
            parts_dict = actor_dict[child_name]
            material = 'pec'
            if "properties" in parts_dict.keys():
                if "material" in parts_dict["properties"].keys():
                    material = parts_dict['properties']['material']

            mat_idx = self.mat_manager.get_index(material)
            name = self.add_part(name=child_name, filename=parts_dict['file_name'], parent_h_node=parent_h_node,
                                 mat_idx=mat_idx,color=self.color)
            if "properties" in parts_dict.keys():
                # these properties are optional, so they may not exist. If they do, then they will be used
                # these are primiarly used to set the intial conditions for actors in the scene that have a
                # constant pos/rot/lin/ang relative to their parent. Useful for things like windturbines, helichopters
                # these property values can be overriden by the user by setting coord_sys.pos/rot/lin/ang
                if "initial_pos" in parts_dict["properties"].keys():
                    pos = parts_dict['properties']['initial_pos']
                    self.parts[name].initial_pos = [pos[0]*self.scale_mesh,pos[1]*self.scale_mesh,pos[2]*self.scale_mesh]
                if "initial_rot" in parts_dict["properties"].keys():
                    rot = parts_dict['properties']['initial_rot']
                    self.parts[name].initial_rot = rot
                if "initial_lin" in parts_dict["properties"].keys():
                    lin = parts_dict['properties']['initial_lin']
                    self.parts[name].initial_lin = lin
                if "initial_ang" in parts_dict["properties"].keys():
                    ang = parts_dict['properties']['initial_ang']
                    self.rotor_ang = ang
                    self.parts[name].initial_ang = ang

        # parts should always be appended witih "part-" to avoid confusion with other keys
        list_of_parts = list(parts_dict.keys())
        list_of_parts = [x for x in list_of_parts if "part-" in x]

        for part in list_of_parts:
            material = 'pec'
            if "properties" in parts_dict[part].keys():
                if "material" in parts_dict[part]["properties"].keys():
                    material = parts_dict[part]['properties']['material']
            mat_idx = self.mat_manager.get_index(material)
            color = None
            if "color" in parts_dict[part]["properties"].keys():
                color = parts_dict[part]['properties']['color']
            name = self.add_part(name=part, filename=parts_dict[part]['file_name'], parent_h_node=parent_h_node,
                                 mat_idx=mat_idx, color=color)
            if "properties" in parts_dict[part].keys():
                # these properties are optional, so they may not exist. If they do, then they will be used
                # these are primiarly used to set the intial conditions for actors in the scene that have a
                # constant pos/rot/lin/ang relative to their parent. Useful for things like windturbines, helichopters
                # these property values can be overriden by the user by setting coord_sys.pos/rot/lin/ang
                if "initial_pos" in parts_dict[part]["properties"].keys():
                    pos = parts_dict[part]['properties']['initial_pos']
                    self.parts[name].initial_pos = [pos[0]*self.scale_mesh,pos[1]*self.scale_mesh,pos[2]*self.scale_mesh]
                if "initial_rot" in parts_dict[part]["properties"].keys():
                    rot = parts_dict[part]['properties']['initial_rot']
                    self.parts[name].initial_rot = rot
                if "initial_lin" in parts_dict[part]["properties"].keys():
                    lin = parts_dict[part]['properties']['initial_lin']
                    self.parts[name].initial_lin = lin
                if "initial_ang" in parts_dict[part]["properties"].keys():
                    ang = parts_dict[part]['properties']['initial_ang']
                    self.rotor_ang = ang
                    # diagnole rotors go in opposite direction, if we have a part number than is even
                    # make go in 1 direction, if odd go in other
                    is_even = 1
                    try:
                        part_idx = int(part[-1])
                        if part_idx % 2 == 0:
                            is_even = -1
                    except:
                        pass
                    self.parts[name].initial_ang = is_even*ang

            # recursively go through the parts and add them to the scene
            children = self._get_children(parts_dict[part])
            for child in children:
                part_dict = parts_dict[part]
                self._load_other(part_dict, self.parts[part].h_node, child_name=child)

    def _load_helicopter(self, actor_dict, parent_h_node=None, child_name=None):
        self.velocity_mag = None
        self.rotor_ang = [0,0,0]
        self.rear_rotor_ang = [0, 0, 0]
        self.actor_type = 'helicopter'
        # top level parts are within parts dict
        if 'parts' in actor_dict.keys():
            parts_dict = actor_dict['parts']
        else:  # child parts are not in a 'parts' dictionary
            parts_dict = actor_dict[child_name]
            material = 'pec'
            if "properties" in parts_dict.keys():
                if "material" in parts_dict["properties"].keys():
                    material = parts_dict['properties']['material']

            mat_idx = self.mat_manager.get_index(material)
            name = self.add_part(name=child_name, filename=parts_dict['file_name'], parent_h_node=parent_h_node,
                                 mat_idx=mat_idx,color=self.color)
            if "properties" in parts_dict.keys():
                # these properties are optional, so they may not exist. If they do, then they will be used
                # these are primiarly used to set the intial conditions for actors in the scene that have a
                # constant pos/rot/lin/ang relative to their parent. Useful for things like windturbines, helichopters
                # these property values can be overriden by the user by setting coord_sys.pos/rot/lin/ang
                if "initial_pos" in parts_dict["properties"].keys():
                    pos = parts_dict['properties']['initial_pos']
                    self.parts[name].initial_pos = [pos[0]*self.scale_mesh,pos[1]*self.scale_mesh,pos[2]*self.scale_mesh]
                if "initial_rot" in parts_dict["properties"].keys():
                    rot = parts_dict['properties']['initial_rot']
                    self.parts[name].initial_rot = rot
                if "initial_lin" in parts_dict["properties"].keys():
                    lin = parts_dict['properties']['initial_lin']
                    self.parts[name].initial_lin = lin
                if "initial_ang" in parts_dict["properties"].keys():
                    ang = parts_dict['properties']['initial_ang']
                    if 'mainrotor' in name.lower().replace('_',''):
                        self.rotor_ang = ang
                    elif 'rearrotor' in name.lower().replace('_',''):
                        self.rear_rotor_ang = ang
                    self.parts[name].initial_ang = ang

        # parts should always be appended witih "part-" to avoid confusion with other keys
        list_of_parts = list(parts_dict.keys())
        list_of_parts = [x for x in list_of_parts if "part-" in x]

        for part in list_of_parts:
            material = 'pec'
            if "properties" in parts_dict[part].keys():
                if "material" in parts_dict[part]["properties"].keys():
                    material = parts_dict[part]['properties']['material']
            mat_idx = self.mat_manager.get_index(material)
            color = None
            if "color" in parts_dict[part]["properties"].keys():
                color = parts_dict[part]['properties']['color']
            name = self.add_part(name=part, filename=parts_dict[part]['file_name'], parent_h_node=parent_h_node,
                                 mat_idx=mat_idx, color=color)
            if "properties" in parts_dict[part].keys():
                # these properties are optional, so they may not exist. If they do, then they will be used
                # these are primiarly used to set the intial conditions for actors in the scene that have a
                # constant pos/rot/lin/ang relative to their parent. Useful for things like windturbines, helichopters
                # these property values can be overriden by the user by setting coord_sys.pos/rot/lin/ang
                if "initial_pos" in parts_dict[part]["properties"].keys():
                    pos = parts_dict[part]['properties']['initial_pos']
                    self.parts[name].initial_pos = [pos[0]*self.scale_mesh,pos[1]*self.scale_mesh,pos[2]*self.scale_mesh]
                if "initial_rot" in parts_dict[part]["properties"].keys():
                    rot = parts_dict[part]['properties']['initial_rot']
                    self.parts[name].initial_rot = rot
                if "initial_lin" in parts_dict[part]["properties"].keys():
                    lin = parts_dict[part]['properties']['initial_lin']
                    self.parts[name].initial_lin = lin
                if "initial_ang" in parts_dict[part]["properties"].keys():
                    ang = parts_dict[part]['properties']['initial_ang']
                    if 'mainrotor' in name.lower().replace('_',''):
                        self.rotor_ang = ang
                    elif 'rearrotor' in name.lower().replace('_',''):
                        self.rear_rotor_ang = ang
                    self.parts[name].initial_ang = ang

            # recursively go through the parts and add them to the scene
            children = self._get_children(parts_dict[part])
            for child in children:
                part_dict = parts_dict[part]
                self._load_other(part_dict, self.parts[part].h_node, child_name=child)

    def _load_bird(self, actor_dict, parent_h_node=None, child_name=None):
        self.velocity_mag = None
        self.flap_range = 45
        self.flap_freq = 3 # flaps per second

        self.actor_type = 'bird'
        # top level parts are within parts dict
        if 'parts' in actor_dict.keys():
            parts_dict = actor_dict['parts']
        else:  # child parts are not in a 'parts' dictionary
            parts_dict = actor_dict[child_name]
            material = 'pec'
            if "properties" in parts_dict.keys():
                if "material" in parts_dict["properties"].keys():
                    material = parts_dict['properties']['material']

            mat_idx = self.mat_manager.get_index(material)
            name = self.add_part(name=child_name, filename=parts_dict['file_name'], parent_h_node=parent_h_node,
                                 mat_idx=mat_idx,color=self.color)


        # parts should always be appended witih "part-" to avoid confusion with other keys
        list_of_parts = list(parts_dict.keys())
        list_of_parts = [x for x in list_of_parts if "part-" in x]

        for part in list_of_parts:
            material = 'pec'
            if "properties" in parts_dict[part].keys():
                if "material" in parts_dict[part]["properties"].keys():
                    material = parts_dict[part]['properties']['material']
            mat_idx = self.mat_manager.get_index(material)
            color = None
            if "color" in parts_dict[part]["properties"].keys():
                color = parts_dict[part]['properties']['color']
            name = self.add_part(name=part, filename=parts_dict[part]['file_name'], parent_h_node=parent_h_node,
                                 mat_idx=mat_idx, color=color)

            # recursively go through the parts and add them to the scene
            children = self._get_children(parts_dict[part])
            for child in children:
                part_dict = parts_dict[part]
                self._load_other(part_dict, self.parts[part].h_node, child_name=child)

    def _load_vehicle_4_wheel(self, actor_dict, parent_h_node=None, child_name=None):
        self.actor_type = 'vehicle'
        self.wheel_base = 2.5
        self.wheel_radius = 0.3
        self.wheel_to_wheel_width = 1.5
        self.velocity_mag = 0
        if 'wheel_base' in actor_dict:
            self.wheel_base = actor_dict['wheel_base']
        if 'wheel_radius' in actor_dict:
            self.wheel_radius = actor_dict['wheel_radius']
        if 'wheel_to_wheel_width' in actor_dict:
            self.wheel_to_wheel_width = actor_dict['wheel_to_wheel_width']

        front_driver_offset = [self.wheel_base, self.wheel_to_wheel_width / 2, self.wheel_radius]
        front_passenger_offset = [self.wheel_base, -self.wheel_to_wheel_width / 2, self.wheel_radius]
        rear_driver_offset = [0, self.wheel_to_wheel_width / 2, self.wheel_radius]
        rear_passenger_offset = [0, -self.wheel_to_wheel_width / 2, self.wheel_radius]
        wheel_offsets = {'front_driver': front_driver_offset,
                         'front_passenger': front_passenger_offset,
                         'rear_driver': rear_driver_offset,
                         'rear_passenger': rear_passenger_offset}

        front_driver_cs = self.add_part(name='front_driver_cs')
        front_passenger_cs = self.add_part(name='front_passenger_cs')
        rear_driver_cs = self.add_part(name='rear_driver_cs')
        rear_passenger_cs = self.add_part(name='rear_passenger_cs')

        # slight modification to load_other function. This function will add the wheels to the vehicle and a cs at each
        # axle location
        def _load_tree(actor_dict, parent_h_node=None, child_name=None):
            # top level parts are within parts dict
            if 'parts' in actor_dict.keys():
                parts_dict = actor_dict['parts']
            else:  # child parts are not in a 'parts' dictionary
                parts_dict = actor_dict[child_name]
                material = 'pec'
                if "properties" in parts_dict.keys():
                    if "material" in parts_dict["properties"].keys():
                        material = parts_dict['properties']['material']
                mat_idx = self.mat_manager.get_index(material)
                if "type" in parts_dict["properties"].keys():
                    if 'wheel' == parts_dict['properties']['type']:
                        name_front_drive = self.add_part(name=f"{child_name}_front_driver",
                                                         filename=parts_dict['file_name'],
                                                         parent_h_node=self.parts[front_driver_cs].h_node,
                                                         mat_idx=mat_idx)
                        name_front_passenger = self.add_part(name=f"{child_name}_front_passenger",
                                                             filename=parts_dict['file_name'],
                                                             parent_h_node=self.parts[front_passenger_cs].h_node,
                                                             mat_idx=mat_idx)
                        name_rear_driver = self.add_part(name=f"{child_name}_rear_driver",
                                                         filename=parts_dict['file_name'],
                                                         parent_h_node=self.parts[rear_driver_cs].h_node,
                                                         mat_idx=mat_idx)
                        name_rear_passenger = self.add_part(name=f"{child_name}_rear_passenger",
                                                            filename=parts_dict['file_name'],
                                                            parent_h_node=self.parts[rear_passenger_cs].h_node,
                                                            mat_idx=mat_idx)
                    else:
                        name = self.add_part(name=child_name, filename=parts_dict['file_name'],
                                             parent_h_node=parent_h_node,
                                             mat_idx=mat_idx)

            # parts should always be appended witih "part-" to avoid confusion with other keys
            list_of_parts = list(parts_dict.keys())
            list_of_parts = [x for x in list_of_parts if "part-" in x]

            is_wheel_assembly = False
            for part in list_of_parts:
                material = 'pec'
                if "properties" in parts_dict.keys():
                    if "material" in parts_dict["properties"].keys():
                        material = parts_dict['properties']['material']

                mat_idx = self.mat_manager.get_index(material)
                name = self.add_part(name=part, filename=parts_dict[part]['file_name'],
                                     parent_h_node=parent_h_node,
                                     mat_idx=mat_idx)

                # recursively go through the parts and add them to the scene
                children = self._get_children(parts_dict[part])
                for child in children:
                    part_dict = parts_dict[part]
                    test = 1
                    _load_tree(part_dict, self.parts[part].h_node, child_name=child)

        _load_tree(actor_dict, parent_h_node, child_name)
        # position cs for wheels
        self.parts[front_driver_cs].coord_sys.pos = wheel_offsets['front_driver']
        self.parts[front_passenger_cs].coord_sys.pos = wheel_offsets['front_passenger']
        self.parts[rear_driver_cs].coord_sys.pos = wheel_offsets['rear_driver']
        self.parts[rear_passenger_cs].coord_sys.pos = wheel_offsets['rear_passenger']

        wheel_actors = [front_driver_cs, front_passenger_cs, rear_driver_cs, rear_passenger_cs]

    def update_actor(self, time=0):
        """
        Update the actor's state based on the current time.

        This method updates the actor's position, rotation, and other properties based on its type
        and the elapsed time. Different actor types (vehicle, pedestrian, quadcopter, etc.) have
        specialized update behaviors.

        Parameters:
        ------------
        time : float, optional
            The current simulation time. Used to calculate time delta and update positions
            based on velocities. Default is 0.

        Notes:
        ------
        The update behavior depends on the actor_type:
        - 'vehicle': Updates wheel rotations and vehicle movement
        - 'pedestrian': Updates walking animation and movement
        - 'quadcopter': Updates rotor rotations and flight dynamics
        - 'helicopter': Updates main and tail rotor rotations
        - 'bird': Updates wing flapping animation
        - 'usd': Updates USD-based animations
        - 'webcam': Updates pose tracking from camera input
        - 'generator': Updates procedurally generated geometry
        - 'other': General purpose updates for other actor types
        """
        if time is not None:

            self.dt = time - self.time
        else:
            self.dt = 0
        self.time = time

        if self.actor_type == 'vehicle':
            self._update_vehicle_4_wheel()
        elif self.actor_type == 'pedestrian':
            self._update_pedestrian()
        elif self.actor_type == 'quadcopter':
            self._update_quadcopter()
        elif self.actor_type == 'helicopter':
            self._update_helicopter()
        elif self.actor_type == 'bird':
            self._update_bird()
        elif self.actor_type == 'usd':
            self._update_usd()
        elif self.actor_type == 'webcam':
            self._update_webcam()
        elif self.actor_type == 'generator':
            self._update_generator()
        else:
            self._update_other()



    def _update_generator(self):
        """
        Update an actor with procedurally generated geometry.

        This method handles updating actors that use procedural geometry generators.
        When dynamic_generator_updates is enabled, it regenerates the mesh at each
        time step using the generator's generate_mesh() method. This is useful for
        time-varying geometry like ocean surfaces, terrain, or other parametric models.

        The method performs the following operations:
        1. If dynamic updates are enabled:
           - Calls the generator's generate_mesh() method with current time
           - Updates all actor parts with the new mesh geometry
           - Deletes the old scene elements to avoid memory leaks
           - Updates the coordinate system with the current time
        2. If dynamic updates are disabled:
           - Falls back to the standard _update_other() method

        Notes:
        ----__
        - The generator object must implement a generate_mesh(t=time) method that
          returns a pyvista mesh object
        - Setting dynamic_generator_updates=False allows for static procedural
          geometry that is generated once and not updated
        - The old scene elements are explicitly deleted to prevent memory
          accumulation over time
        - This method is automatically called by update_actor() when the
          actor_type is 'generator'

        See Also:
        ---------
        add_parts_from_generator : Method to initialize generator-based actors
        update_actor : Main actor update method that calls this function
        """
        if self.dynamic_generator_updates:
            print(f"\nUpdating procedural geometry actor with generator at time:", self.time)
            new_mesh = self.generator.generate_mesh(t=self.time)
            
            for part in self.parts:
                self.parts[part].mesh_loader = self.mesh_loader
                self.parts[part].mesh_loader.update_mesh( h_node=self.parts[part].h_node, mesh=new_mesh, delete_old_scene_element=True)
                self.parts[part].mesh = new_mesh
            self.coord_sys.update(time=self.time)
        else:
            self._update_other()

    def _update_vehicle_4_wheel(self):
        """
        Updates the state of a 4-wheel vehicle, including its position, velocity, and wheel rotations.
        This method calculates the linear and angular velocities of the vehicle, updates its position 
        based on the current velocity, and computes the rotation of each wheel based on the vehicle's 
        speed. It also ensures that necessary attributes such as `wheel_base`, `wheel_radius`, 
        `wheel_to_wheel_width`, and `velocity_mag` are initialized if not already set.
        Attributes:
            wheel_base (float): Distance between the front and rear axles of the vehicle.
            wheel_radius (float): Radius of the vehicle's wheels.
            wheel_to_wheel_width (float): Distance between the left and right wheels.
            velocity_mag (float): Magnitude of the vehicle's velocity.
            coord_sys (object): Coordinate system of the vehicle, containing position, rotation, 
                    linear velocity, and angular velocity.
            parts (dict): Dictionary containing the coordinate systems of individual vehicle parts 
                  (e.g., wheels).
        Behavior:
            - Updates the vehicle's position using either a linear velocity equation or a time-based update.
            - Computes the angular velocity and rotation of each wheel based on the vehicle's speed.
            - Updates the coordinate systems of the wheels with their respective linear and angular velocities.
        Note:
            - Assumes `coord_sys.ang` is always zero for the vehicle.
            - Uses helper functions `rotate_vector_from_rot` and `euler_to_rot` for vector rotation and 
              Euler angle conversion, respectively.
        """

        self.wheel_base = 2.5
        self.wheel_radius = 0.3
        self.wheel_to_wheel_width = 1.5
        if not hasattr(self, 'wheel_base'):
            self.wheel_base = 2.5
        if not hasattr(self, 'wheel_radius'):
            self.wheel_radius = 0.3
        if not hasattr(self, 'wheel_to_wheel_width'):
            self.wheel_to_wheel_width = 1.5
        if not hasattr(self, 'velocity_mag'):
            self.velocity_mag = 0

        bulk_velocity = self.velocity_mag
        bulk_velocity_as_x_vector = [bulk_velocity, 0, 0]
        bulk_ang_velocity = self.coord_sys.ang  # this should always be zero for a vehicle
        bulk_pos = self.coord_sys.pos
        bulk_rot = self.coord_sys.rot

        self.coord_sys.lin = rotate_vector_from_rot(bulk_velocity_as_x_vector, bulk_rot)

        if self.use_linear_velocity_equation_update:
            new_pos = self.coord_sys.pos + self.dt * self.coord_sys.lin
            self.coord_sys.pos = new_pos
            self.coord_sys.update()
        else:
            self.coord_sys.update(time=self.time)

        vel_mag_temp = np.linalg.norm(bulk_velocity)
        wheel_speed_radps = vel_mag_temp / self.wheel_radius

        # get current rotation of tire component related to time
        theta = wheel_speed_radps / 2 / np.pi * self.time * 360
        theta = theta % 360
        rot = euler_to_rot(phi=0, theta=theta, psi=0)

        self.parts['rear_driver_cs'].coord_sys.lin = np.zeros(3)
        self.parts['rear_driver_cs'].coord_sys.ang = [0, wheel_speed_radps, 0]
        self.parts['rear_driver_cs'].coord_sys.rot = rot

        self.parts['rear_passenger_cs'].coord_sys.lin = np.zeros(3)
        self.parts['rear_passenger_cs'].coord_sys.ang = [0, wheel_speed_radps, 0]
        self.parts['rear_passenger_cs'].coord_sys.rot = rot

        self.parts['front_driver_cs'].coord_sys.lin = np.zeros(3)
        self.parts['front_driver_cs'].coord_sys.ang = [0, wheel_speed_radps, 0]
        self.parts['front_driver_cs'].coord_sys.rot = rot

        self.parts['front_passenger_cs'].coord_sys.lin = np.zeros(3)
        self.parts['front_passenger_cs'].coord_sys.ang = [0, wheel_speed_radps, 0]
        self.parts['front_passenger_cs'].coord_sys.rot = rot

    def _update_usd(self):
        # ToDo: Implement USD update, this might not be correct.
        def recurse_parts(part,pos,rot,time):
            part.coord_sys.pos = pos
            part.coord_sys.rot = rot
            part.coord_sys.update()
            for child in part.parts:
                recurse_parts(part.parts[child],pos,rot,time)

        print('USD update not yet completed - Only use for Static Actors')
        for prim in self.usd_actor.all_prims:
            for part in self.parts:
                pos, rot = self.usd_actor.update_actor(self.usd_actor.all_prims[prim], time=0)
                recurse_parts(self.parts[part],pos,rot,self.time)
                self.coord_sys.update(time=self.time)
    def _update_other(self):
        """
        Updates the state of the actor and its parts based on the current time, velocity, and angular velocity.
        This method performs the following operations:
        1. Updates the position and rotation of the actor and its parts based on their initial states and angular velocity.
        2. Updates the linear velocity and position of the actor using the linear velocity equation if enabled.
        3. Recursively updates the state of all child parts associated with the actor.
        Key Parameters:
        - `self.time`: The current simulation time.
        - `self.update_rot_based_on_ang_vel`: Flag indicating whether to update rotation based on angular velocity.
        - `self.use_linear_velocity_equation_update`: Flag indicating whether to update position using the linear velocity equation.
        - `self.velocity_mag`: Magnitude of the actor's velocity (if applicable).
        - `self.coord_sys`: The coordinate system of the actor, including position, rotation, linear velocity, and angular velocity.
        - `self.parts`: Dictionary of child parts associated with the actor.
        Notes:
        - If `self.use_linear_velocity_equation_update` is enabled, a INFO message is printed at time `0` to inform the user.
        - The rotation update based on angular velocity is performed using Euler angles and modular arithmetic to ensure values remain within valid ranges.
        - Recursive updates are applied to all child parts using their respective initial states and velocities.
        Raises:
        - None explicitly, but assumes valid input data for coordinate systems and velocity attributes.
        """

        def recurse_parts(part, time):
            if part.initial_pos is not None:
                part.coord_sys.pos = part.initial_pos
            if part.initial_rot is not None:
                part.coord_sys.rot = part.initial_rot
            if part.initial_lin is not None:
                part.coord_sys.lin = part.initial_lin
            if part.initial_ang is not None:
                part.coord_sys.ang = part.initial_ang

            intial_euler = rot_to_euler(part.coord_sys.rot, order='xyz', deg=True)

            rotional_speed_radps_1 = part.coord_sys.ang[0]
            rotional_speed_radps_2 = part.coord_sys.ang[1]
            rotional_speed_radps_3 = part.coord_sys.ang[2]

            # get current rotation of component related to time
            # ToDo, not sure if this correct at all, just a place holder
            if self.time is not None and self.update_rot_based_on_ang_vel:
                phi = rotional_speed_radps_1 / 2 / np.pi * self.time * 360
                phi = phi % 360
                theta = rotional_speed_radps_2 / 2 / np.pi * self.time * 360
                theta = theta % 360
                psi = rotional_speed_radps_3 / 2 / np.pi * self.time * 360
                psi = psi % 360
                new_rot = euler_to_rot(phi=intial_euler[0]+phi, theta=intial_euler[1]+theta, psi=intial_euler[2]+psi, order='xyz')
                part.coord_sys.rot = new_rot
            part.coord_sys.update()
            for child in part.parts:
                recurse_parts(part.parts[child], time)

        if self.use_linear_velocity_equation_update:
            if hasattr(self, 'velocity_mag'): # some objects will never have a velocity_mag, like an antenna
                if self.velocity_mag is not None:
                    bulk_velocity = self.velocity_mag
                    bulk_velocity_as_x_vector = [bulk_velocity, 0, 0]
                    bulk_ang_velocity = self.coord_sys.ang  # this should always be zero for a vehicle
                    bulk_pos = self.coord_sys.pos
                    bulk_rot = self.coord_sys.rot
                    self.coord_sys.lin = rotate_vector_from_rot(bulk_velocity_as_x_vector, bulk_rot)

            new_pos = self.coord_sys.pos + self.dt * self.coord_sys.lin
            self.coord_sys.pos = new_pos
            self.coord_sys.update()
        else:
            self.coord_sys.update(time=self.time)




        # get current rotation of component related to time
        # ToDo, not sure if this correct at all, just a place holder
        if self.time is not None and self.update_rot_based_on_ang_vel:

            intial_euler = rot_to_euler(self.coord_sys.rot, order='xyz', deg=True)
            rotional_speed_radps_1 = self.coord_sys.ang[0]
            rotional_speed_radps_2 = self.coord_sys.ang[1]
            rotional_speed_radps_3 = self.coord_sys.ang[2]

            phi = rotional_speed_radps_1 / 2 / np.pi * self.dt * 360
            phi = phi % 360
            theta = rotional_speed_radps_2 / 2 / np.pi * self.dt * 360
            theta = theta % 360
            psi = rotional_speed_radps_3 / 2 / np.pi * self.dt * 360
            psi = psi % 360
            # if psi !=0:
            #     print(psi)
            new_rot = euler_to_rot(phi=intial_euler[0]+phi, theta=intial_euler[1]+theta, psi=intial_euler[2]+psi, order='xyz')
            self.coord_sys.rot = new_rot
        self.coord_sys.update()


        for part in self.parts:
            recurse_parts(self.parts[part], self.time)

        # print a warning message to let users know that the velocity is being updated based on the linear velocity equation
        # it isn't always clear if the user wants to do this or not, so a warning message is printed so they are aware
        if self.time == 0:
            if self.use_linear_velocity_equation_update:
                print(f'Using Linear Velocity Equation for Position Update for Actor: {self.name}')

    def _update_quadcopter(self):
        """
        Updates the state of the quadcopter and its parts based on time, velocity, and angular velocity.
        This method handles the position, rotation, linear velocity, and angular velocity updates for the quadcopter 
        and its child parts. It also recursively updates the state of all child parts.
        Key functionalities:
        - Updates the position and rotation of the quadcopter based on linear and angular velocity.
        - Applies specific rotational behavior for blades and rotors based on their part names and indices.
        - Recursively updates the state of child parts.
        - Optionally updates the position using a linear velocity equation.
        - Prints a warning message at time 0 if the linear velocity equation update is enabled.
        Attributes:
            self.time (float): The current simulation time.
            self.dt (float): The time step for the simulation.
            self.coord_sys (object): The coordinate system of the quadcopter, containing position, rotation, linear velocity, and angular velocity.
            self.parts (dict): A dictionary of child parts belonging to the quadcopter.
            self.use_linear_velocity_equation_update (bool): Flag indicating whether to update position using the linear velocity equation.
            self.velocity_mag (float): The magnitude of the quadcopter's velocity (optional).
            self.update_rot_based_on_ang_vel (bool): Flag indicating whether to update rotation based on angular velocity.
            self.rotor_ang (list): Angular velocity values for the rotors.
        Notes:
            - The rotational behavior for blades and rotors depends on their part names and indices.
            - The method assumes that angular velocity is provided in radians per second.
            - The linear velocity equation update is only applied if `use_linear_velocity_equation_update` is True.
        Raises:
            None
        """

        def recurse_parts(part_name,part, time):
            if part.initial_pos is not None:
                part.coord_sys.pos = part.initial_pos
            if part.initial_rot is not None:
                part.coord_sys.rot = part.initial_rot
            if part.initial_lin is not None:
                part.coord_sys.lin = part.initial_lin
            if part.initial_ang is not None:
                if 'blade' in part_name or 'rotor' in part_name:
                    # diagnole rotors go in opposite direction, if we have a part number than is even
                    # make go in 1 direction, if odd go in other
                    is_even = 1
                    try:
                        part_idx = int(part_name[-1])
                        if part_idx % 2 == 0:
                            is_even = -1
                    except:
                        pass
                    part.coord_sys.ang = is_even*np.array(self.rotor_ang)

                else:
                    part.coord_sys.ang = part.initial_ang

            intial_euler = rot_to_euler(part.coord_sys.rot, order='xyz', deg=True)

            rotional_speed_radps_1 = part.coord_sys.ang[0]
            rotional_speed_radps_2 = part.coord_sys.ang[1]
            rotional_speed_radps_3 = part.coord_sys.ang[2]

            # get current rotation of component related to time
            # ToDo, not sure if this correct at all, just a place holder
            if self.time is not None and self.update_rot_based_on_ang_vel:
                phi = rotional_speed_radps_1 / 2 / np.pi * self.time * 360
                phi = phi % 360
                theta = rotional_speed_radps_2 / 2 / np.pi * self.time * 360
                theta = theta % 360
                psi = rotional_speed_radps_3 / 2 / np.pi * self.time * 360
                psi = psi % 360
                new_rot = euler_to_rot(phi=intial_euler[0]+phi, theta=intial_euler[1]+theta, psi=intial_euler[2]+psi, order='xyz')
                part.coord_sys.rot = new_rot
            part.coord_sys.update()
            for child in part.parts:
                recurse_parts(child,part.parts[child], time)

        if self.use_linear_velocity_equation_update:
            if hasattr(self, 'velocity_mag'): # some objects will never have a velocity_mag, like an antenna
                if self.velocity_mag is not None:
                    bulk_velocity = self.velocity_mag
                    bulk_velocity_as_x_vector = [bulk_velocity, 0, 0]
                    bulk_ang_velocity = self.coord_sys.ang  # this should always be zero for a vehicle
                    bulk_pos = self.coord_sys.pos
                    bulk_rot = self.coord_sys.rot
                    self.coord_sys.lin = rotate_vector_from_rot(bulk_velocity_as_x_vector, bulk_rot)

            new_pos = self.coord_sys.pos + self.dt * self.coord_sys.lin
            self.coord_sys.pos = new_pos
            self.coord_sys.update()
        else:
            self.coord_sys.update(time=self.time)

        for part in self.parts:
            recurse_parts(part,self.parts[part], self.time)

        # print a warning message to let users know that the velocity is being updated based on the linear velocity equation
        # it isn't always clear if the user wants to do this or not, so a warning message is printed so they are aware
        if self.time == 0:
            if self.use_linear_velocity_equation_update:
                print(f'Using Linear Velocity Equation for Position Update for Actor: {self.name}')
                

    def _update_helicopter(self):
        """
        Updates the state of the helicopter actor, including its position, rotation, 
        linear velocity, and angular velocity. This method handles both the main 
        helicopter body and its parts recursively.
        The update process includes:
        - Setting initial positions, rotations, linear velocities, and angular velocities 
          for each part.
        - Calculating new rotations based on angular velocity and elapsed time, if enabled.
        - Updating the position based on linear velocity using the linear velocity equation, 
          if enabled.
        - Recursively updating child parts of the helicopter.
        Attributes:
        - self.time (float): The current simulation time.
        - self.update_rot_based_on_ang_vel (bool): Flag to determine if rotation should be 
          updated based on angular velocity.
        - self.use_linear_velocity_equation_update (bool): Flag to determine if position 
          should be updated using the linear velocity equation.
        - self.velocity_mag (float): Magnitude of the velocity for the helicopter, if applicable.
        - self.coord_sys (object): The coordinate system of the helicopter, including position, 
          rotation, linear velocity, and angular velocity.
        - self.parts (dict): Dictionary of child parts associated with the helicopter.
        Notes:
        - Prints a warning message at time 0 if the linear velocity equation update is enabled.
        - Assumes that angular velocity for the main rotor and rear rotor is handled separately 
          based on part names.
        Raises:
        - None
        Returns:
        - None
        """

        def recurse_parts(part_name,part, time):
            if part.initial_pos is not None:
                part.coord_sys.pos = part.initial_pos
            if part.initial_rot is not None:
                part.coord_sys.rot = part.initial_rot
            if part.initial_lin is not None:
                part.coord_sys.lin = part.initial_lin
            if part.initial_ang is not None:
                if 'mainrotor' in part_name.lower().replace('_',''):
                    part.coord_sys.ang = self.rotor_ang
                elif 'rearrotor' in part_name.lower().replace('_',''):
                    part.coord_sys.ang = self.rear_rotor_ang
                else:
                    part.coord_sys.ang = part.initial_ang

            intial_euler = rot_to_euler(part.coord_sys.rot, order='xyz', deg=True)

            rotional_speed_radps_1 = part.coord_sys.ang[0]
            rotional_speed_radps_2 = part.coord_sys.ang[1]
            rotional_speed_radps_3 = part.coord_sys.ang[2]

            # get current rotation of component related to time
            # ToDo, not sure if this correct at all, just a place holder
            if self.time is not None and self.update_rot_based_on_ang_vel:
                phi = rotional_speed_radps_1 / 2 / np.pi * self.time * 360
                phi = phi % 360
                theta = rotional_speed_radps_2 / 2 / np.pi * self.time * 360
                theta = theta % 360
                psi = rotional_speed_radps_3 / 2 / np.pi * self.time * 360
                psi = psi % 360
                new_rot = euler_to_rot(phi=intial_euler[0]+phi, theta=intial_euler[1]+theta, psi=intial_euler[2]+psi, order='xyz')
                part.coord_sys.rot = new_rot
            part.coord_sys.update()
            for child in part.parts:
                recurse_parts(child,part.parts[child], time)

        if self.use_linear_velocity_equation_update:
            if hasattr(self, 'velocity_mag'): # some objects will never have a velocity_mag, like an antenna
                if self.velocity_mag is not None:
                    bulk_velocity = self.velocity_mag
                    bulk_velocity_as_x_vector = [bulk_velocity, 0, 0]
                    bulk_ang_velocity = self.coord_sys.ang  # this should always be zero for a vehicle
                    bulk_pos = self.coord_sys.pos
                    bulk_rot = self.coord_sys.rot
                    self.coord_sys.lin = rotate_vector_from_rot(bulk_velocity_as_x_vector, bulk_rot)

            new_pos = self.coord_sys.pos + self.dt * self.coord_sys.lin
            self.coord_sys.pos = new_pos
            self.coord_sys.update()
        else:
            self.coord_sys.update(time=self.time)

        for part in self.parts:
            recurse_parts(part,self.parts[part], self.time)

        # print a warning message to let users know that the velocity is being updated based on the linear velocity equation
        # it isn't always clear if the user wants to do this or not, so a warning message is printed so they are aware
        if self.time == 0:
            if self.use_linear_velocity_equation_update:
                print(f'Using Linear Velocity Equation for Position Update for Actor: {self.name}')

    def _update_bird(self):
        """
        Updates the bird's position, orientation, and parts based on the current time and configuration.
        This method handles the bird's movement and updates its parts recursively. It also supports
        updating the bird's position using a linear velocity equation if enabled.
        Key functionalities:
        - Updates the bird's position and orientation based on linear velocity or time.
        - Recursively updates the orientation and angular velocity of the bird's parts (e.g., wings).
        - Prints a warning message at time 0 if linear velocity equation update is enabled.
        Attributes:
            use_linear_velocity_equation_update (bool): Determines whether to update position using
                the linear velocity equation.
            velocity_mag (float, optional): Magnitude of the bird's velocity. Used for linear velocity
                updates if available.
            coord_sys (object): The bird's coordinate system containing position, rotation, angular
                velocity, and linear velocity.
            parts (dict): Dictionary of the bird's parts, where keys are part names and values are part
                objects.
            flap_range (float): Range of wing flapping motion.
            flap_freq (float): Frequency of wing flapping motion.
            time (float): Current simulation time.
            dt (float): Time step for position updates.
            name (str): Name of the bird actor.
        Notes:
            - The method assumes that the bird's wings are named with 'lwing' for the left wing and
              'rwing' for the right wing.
            - The `euler_to_rot` function is used to calculate rotation matrices for wing movements.
            - The `rotate_vector_from_rot` function is used to rotate the velocity vector based on the
              bird's rotation.
        Raises:
            AttributeError: If a required attribute is missing from the bird object.
        """

        def recurse_parts(part_name,part, time):
            if 'lwing' in part_name:
                part.coord_sys.rot = euler_to_rot(phi=0,
                                                  theta=self.flap_range * np.cos(2*np.pi * time / self.flap_freq),
                                                  order='zxz', deg=True)
                part.coord_sys.ang = [self.flap_range * np.sin(np.pi * time * self.flap_freq), 0, 0]
            elif 'rwing' in part_name:
                part.coord_sys.rot = euler_to_rot(phi=0,
                                                  theta=-self.flap_range * np.cos(2*np.pi * time / self.flap_freq),
                                                  order='zxz', deg=True)
                part.coord_sys.ang = [-self.flap_range * np.sin(np.pi * time * self.flap_freq), 0, 0]

            part.coord_sys.update()
            for child in part.parts:
                recurse_parts(child,part.parts[child], time)

        if self.use_linear_velocity_equation_update:
            if hasattr(self, 'velocity_mag'): # some objects will never have a velocity_mag, like an antenna
                if self.velocity_mag is not None:
                    bulk_velocity = self.velocity_mag
                    bulk_velocity_as_x_vector = [bulk_velocity, 0, 0]
                    bulk_ang_velocity = self.coord_sys.ang  # this should always be zero for a vehicle
                    bulk_pos = self.coord_sys.pos
                    bulk_rot = self.coord_sys.rot
                    self.coord_sys.lin = rotate_vector_from_rot(bulk_velocity_as_x_vector, bulk_rot)

            new_pos = self.coord_sys.pos + self.dt * self.coord_sys.lin
            self.coord_sys.pos = new_pos
            self.coord_sys.update()
        else:
            self.coord_sys.update(time=self.time)

        for part in self.parts:
            recurse_parts(part,self.parts[part], self.time)

        # print a warning message to let users know that the velocity is being updated based on the linear velocity equation
        # it isn't always clear if the user wants to do this or not, so a warning message is printed so they are aware
        if self.time == 0:
            if self.use_linear_velocity_equation_update:
                print(f'Using Linear Velocity Equation for Position Update for Actor: {self.name}')

    def _update_webcam(self):
        """
        Updates the webcam feed and processes pose landmarks using MediaPipe Pose.
        This method captures a frame from the webcam, processes it using MediaPipe Pose to extract pose landmarks,
        and calculates positions and rotations for various body parts. It also smooths the landmark positions over
        multiple frames and adjusts positions based on predefined bone lengths. The processed frame is displayed
        and optionally saved to a video file.
        Key functionalities:
        - Captures and processes webcam frames using MediaPipe Pose.
        - Extracts pose landmarks and calculates positions and rotations for body parts.
        - Smooths landmark positions over multiple frames.
        - Adjusts positions based on predefined bone lengths.
        - Updates the coordinate systems of body parts.
        - Displays the processed frame and optionally saves it to a video file.
        Attributes:
            self.webcam (cv2.VideoCapture): Webcam object for capturing frames.
            self.media_pipe_vid_output (cv2.VideoWriter): Video writer object for saving processed frames.
            self.webcam_frame (np.array): Current processed webcam frame.
            self.parts (dict): Dictionary containing body parts and their coordinate systems.
            self.time (float): Current time for updating coordinate systems.
        Notes:
            - Bone lengths are hardcoded for specific body parts.
            - The method uses smoothing over the last three frames to stabilize landmark positions.
            - Rotations are calculated using a "look-at" approach between landmarks.
        Raises:
            ValueError: If the webcam feed is empty or landmarks are not detected.
        Dependencies:
            - numpy (np)
            - cv2
            - mediapipe (mp.solutions.pose, mp.solutions.drawing_utils, mp.solutions.drawing_styles)
            - scipy.spatial.transform.Rotation
        Returns:
            None
        """
        shoulder_bone_length = 0.286

        def vec_length(v: np.array):
            return np.sqrt(sum(i ** 2 for i in v))

        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm

        def look_at(eye: np.array, target: np.array, alignment_vector: np.array = [0, 0, 1]):
            axis_z = normalize((target - eye))
            if vec_length(axis_z) == 0:
                axis_z = np.array((0, -1, 0))

            axis_x = np.cross(np.array(alignment_vector), axis_z)
            if vec_length(axis_x) == 0:
                axis_x = np.array((1, 0, 0))

            axis_y = np.cross(axis_z, axis_x)
            rot_matrix = np.matrix([axis_x, axis_y, axis_z]).transpose()
            return rot_matrix

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        all_parts_webcam_smoothing = []
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # while cap.isOpened():
            success, image = self.webcam.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                # continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            results = pose.process(image)
            all_parts_pos = {}
            all_parts_rot = {}
            landmakrs = results.pose_world_landmarks.landmark


            if landmakrs:
                for part in mp_pose.PoseLandmark:
                    part_idx = part.value
                    part_name = part.name.replace('_INDEX', '')
                    landmakrs[part_idx].z
                    all_parts_pos[part_name] = np.array([landmakrs[part_idx].z
                                                            , landmakrs[part_idx].x,
                                                         -landmakrs[part_idx].y,
                                                         landmakrs[part_idx].visibility])

               






                all_parts_webcam_smoothing.append(all_parts_pos)

                landmakrs_img = results.pose_landmarks.landmark

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                image = cv2.flip(image, 1)

                if self.media_pipe_vid_output is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.media_pipe_vid_output = cv2.VideoWriter('test.avi',
                                                                 fourcc,
                                                                 30, image.shape[:2], True)

                self.media_pipe_vid_output.write(image)
                # cv2.putText(image, str(np.rad2deg(angle)),
                #             (100, 100),
                #             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                #             0.5, (0, 255, 0), 2,
                #             cv2.LINE_AA)

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     pass
                # img_plot_data_bgr =np.moveaxis(np.stack([image[:,:,2], image[:,:,1],image[:,:,0]]),0,-1)
                # img_plot = Image.fromarray(img_plot_data_bgr)
                self.webcam_frame = image
               
                # if iFrame >= numFrames:
                #     break
                # if cv2.waitKey(5) & 0xFF == 27:
                #     break

                # the pair is used to determine unit vetor in direction of next joint
            # also, hardcode bone lengths
            pair = {}
            bone_lengths = {}

            pair['LEFT_SHOULDER'] = 'LEFT_ELBOW'
            bone_lengths['LEFT_SHOULDER'] = 0.286

            pair['LEFT_ELBOW'] = 'LEFT_WRIST'
            bone_lengths['LEFT_ELBOW'] = 0.286

            pair['LEFT_HIP'] = 'LEFT_KNEE'
            bone_lengths['LEFT_HIP'] = 0.447

            pair['LEFT_KNEE'] = 'LEFT_ANKLE'
            bone_lengths['LEFT_KNEE'] = 0.455

            pair['RIGHT_SHOULDER'] = 'RIGHT_ELBOW'
            bone_lengths['RIGHT_SHOULDER'] = 0.286

            pair['RIGHT_ELBOW'] = 'RIGHT_WRIST'
            bone_lengths['RIGHT_ELBOW'] = 0.286

            pair['RIGHT_HIP'] = 'RIGHT_KNEE'
            bone_lengths['RIGHT_HIP'] = 0.447

            pair['RIGHT_KNEE'] = 'RIGHT_ANKLE'
            bone_lengths['RIGHT_KNEE'] = 0.455

            current_idx = len(all_parts_webcam_smoothing) - 1
            num_cells_to_avg = 3
            if current_idx >= num_cells_to_avg:
                for part in all_parts_webcam_smoothing[current_idx]:
                    pos = np.zeros(3)
                    num_to_avg = 0
                    for idx in range(num_cells_to_avg):
                        if part in all_parts_webcam_smoothing[current_idx - idx].keys():
                            pos += all_parts_webcam_smoothing[current_idx - idx][part][:3]
                            num_to_avg += 1

                    all_parts_pos[part] = pos / num_to_avg
                    # self.all_parts_webcam_smoothing[current_idx][part][:3] = all_parts_pos[part]

            # ToDo, need to add in additional rotation for shoulder to hips
            if ('LEFT_SHOULDER' in all_parts_pos.keys() and 'RIGHT_SHOULDER' in all_parts_pos.keys()):
                rot = look_at(all_parts_pos['LEFT_SHOULDER'][:3],
                              all_parts_pos['RIGHT_SHOULDER'][:3])

                all_parts_pos['Chest'] = all_parts_pos['RIGHT_SHOULDER'][:3]
                all_parts_rot['Chest'] = rot
            elif 'RIGHT_SHOULDER' in all_parts_pos.keys():
                all_parts_pos['Chest'] = all_parts_pos['RIGHT_SHOULDER'][:3]
                all_parts_rot['Chest'] = np.eye(3)
            else:
                all_parts_pos['Chest'] = np.zeros(3)
                all_parts_rot['Chest'] = np.eye(3)

            if ('LEFT_HIP' in all_parts_pos.keys() and 'RIGHT_HIP' in all_parts_pos.keys()):
                rot = look_at(all_parts_pos['LEFT_HIP'][:3],
                              all_parts_pos['RIGHT_HIP'][:3])
                all_parts_pos['Hips'] = all_parts_pos['RIGHT_HIP'][:3]
                all_parts_rot['Hips'] = rot
            elif 'RIGHT_HIP' in all_parts_pos.keys():
                all_parts_pos['Hips'] = all_parts_pos['RIGHT_HIP'][:3]
                all_parts_rot['Hips'] = np.eye(3)
            else:
                all_parts_pos['Hips'] = np.zeros(3)
                all_parts_rot['Hips'] = np.eye(3)

            alignment_vector = np.array([1, 0, 0])
            if ('LEFT_EYE' in all_parts_pos.keys() and 'RIGHT_EYE' in all_parts_pos.keys()):

                rot = look_at(all_parts_pos['LEFT_EYE'][:3],
                              all_parts_pos['RIGHT_EYE'][:3],
                              alignment_vector=alignment_vector)
                all_parts_pos['NOSE'] = all_parts_pos['NOSE'][:3]
                all_parts_rot['NOSE'] = rot
            elif ('LEFT_EAR' in all_parts_pos.keys() and 'RIGHT_EAR' in all_parts_pos.keys()):

                rot = look_at(all_parts_pos['LEFT_EAR'][:3],
                              all_parts_pos['RIGHT_EAR'][:3],
                              alignment_vector=alignment_vector)
                all_parts_pos['NOSE'] = all_parts_pos['NOSE'][:3]
                all_parts_rot['NOSE'] = rot
            elif ('MOUTH_LEFT' in all_parts_pos.keys() and 'MOUTH_RIGHT' in all_parts_pos.keys()):

                rot = look_at(all_parts_pos['MOUTH_LEFT'][:3],
                              all_parts_pos['MOUTH_RIGHT'][:3],
                              alignment_vector=alignment_vector)
                all_parts_pos['NOSE'] = all_parts_pos['NOSE'][:3]
                all_parts_rot['NOSE'] = rot
            else:
                all_parts_pos['NOSE'] = np.zeros(3)
                all_parts_rot['NOSE'] = np.eye(3)

            # update all positions to account for bone length, this would allow any size person, but for now
            # these bone lenghts are hard coded above
            for part in self.parts:
                if part in all_parts_pos.keys():
                    pos = all_parts_pos[part][:3]
                    if part in pair.keys():
                        pos_next_joint = all_parts_pos[pair[part]]
                        pos_next_joint = pos_next_joint[:3] # last dimension is confidence
                        vec = pos_next_joint - pos
                        uvec = vec / np.linalg.norm(vec, axis=0)
                        # update next joint so it is in correct position based on bone length
                        all_parts_pos[pair[part]] = uvec * bone_lengths[part] + pos

            # with updated positions, get rotations
            for part in self.parts:
                if part in all_parts_pos.keys():
                    pos = all_parts_pos[part][:3]
                    if part in pair.keys():
                        pos_next_joint = all_parts_pos[pair[part]]
                        pos_next_joint = pos_next_joint[:3]
                    else:
                        pos_next_joint = None
                    self.parts[part].coord_sys.pos = pos

                    if pos_next_joint is not None:
                        # if rot has not been defined, in cases like torso,hips, then calculate from next joint position
                        rot = look_at(pos,pos_next_joint)
                        all_parts_rot[part] = rot
                    elif part not in all_parts_rot.keys():  # get rot from next joint position:
                        all_parts_rot[part] = np.eye(3)

                    # need to make sure it is a valid rotation matrix, otherwise it will skew geometry
                    r = Rotation.from_matrix(all_parts_rot[part])
                    self.parts[part].coord_sys.rot = r.as_matrix()

                    self.parts[part].coord_sys.update(time=self.time)

                    # if mesh.filename not in self.all_parts_webcam_smoothing_lin.keys():
                    #     self.all_parts_webcam_smoothing_lin[mesh.filename] = []
                    #     self.all_parts_webcam_smoothing_ang[mesh.filename] = []
                    #
                    # self.all_parts_webcam_smoothing_lin[mesh.filename].append(mesh.coords.lin)
                    # self.all_parts_webcam_smoothing_ang[mesh.filename].append(mesh.coords.ang)
                    #
                    # current_idx = len(self.all_parts_webcam_smoothing_lin[mesh.filename]) - 1
                    # num_cells_to_avg = 3
                    # if current_idx >= num_cells_to_avg:
                    #     lin = np.zeros(3)
                    #     ang = np.zeros(3)
                    #     num_to_avg = 0
                    #     for idx in range(num_cells_to_avg):
                    #         lin += np.array(self.all_parts_webcam_smoothing_lin[mesh.filename][current_idx - idx])
                    #         ang += np.array(self.all_parts_webcam_smoothing_ang[mesh.filename][current_idx - idx])
                    #         num_to_avg += 1
                    #
                    #     mesh.coords.lin = lin / num_to_avg
                    #     mesh.coords.ang = ang / num_to_avg
                    #     mesh.coords.update(time=None)

                    ###################
                    # Below is just for visualizaiton, we need to get the global CS position, since some objects may be in a parent CS
                    ###################
                    #
                    # (ret, rot, pos, lin, ang) = api.coordSysInGlobal(mesh.coords.hNode)
                    #
                    # T = np.concatenate((np.asarray(rot), np.asarray(pos).reshape((-1, 1))), axis=1)
                    # T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
                    #
                    # # we only will updat the mesh_pv_anim, mesh_pv will contain the original mesh in its orginal locaiton
                    # if mesh.mesh_pv_anim:
                    #     mesh.mesh_pv_anim = mesh.mesh_pv.transform(T, inplace=False)
                    # else:
                    #     mesh.mesh_pv_anim.transfo_animrm(T, inplace=True)
                    # mesh.T = T

    def _update_pedestrian(self):
        """
        Updates the state of the pedestrian actor, including its position, velocity, and associated parts.
        This method handles the update of the pedestrian's linear velocity and position based on the current
        velocity magnitude and coordinate system. It also recursively updates the coordinate systems of all
        associated parts.
        Attributes:
            velocity_mag (float): The magnitude of the pedestrian's velocity. Defaults to 0 if not set.
            coord_sys (object): The coordinate system of the pedestrian, containing position (`pos`), rotation (`rot`),
                    and linear velocity (`lin`).
            dt (float): The time step used for updating the position.
            time (float): The current simulation time, used for updating the coordinate systems of parts.
            use_linear_velocity_equation_update (bool): Flag indicating whether to update position using the linear
                                velocity equation or based on the simulation time.
        Behavior:
            - If `use_linear_velocity_equation_update` is True, the pedestrian's position is updated using the linear
              velocity equation.
            - Otherwise, the coordinate system is updated manually.
            - Recursively updates the coordinate systems of all child parts.
        Notes:
            - The `rotate_vector_from_rot` function is used to compute the linear velocity vector based on the rotation.
            - The `coord_sys.update()` method is called to apply changes to the coordinate system.
        """
        if not hasattr(self, 'velocity_mag'):
            self.velocity_mag = 0

        bulk_velocity = self.velocity_mag
        bulk_velocity_as_x_vector = [bulk_velocity, 0, 0]
        bulk_pos = self.coord_sys.pos
        bulk_rot = self.coord_sys.rot
        self.coord_sys.lin = rotate_vector_from_rot(bulk_velocity_as_x_vector, bulk_rot)

        if self.use_linear_velocity_equation_update:
            new_pos = self.coord_sys.pos + self.dt * self.coord_sys.lin
            self.coord_sys.pos = new_pos
            self.coord_sys.update()
        else:  # if the time argument is not None, the velocities will be estimated
            self.coord_sys.update(time=self.time)
            # pass

        def recurse_parts(part, time):
            part.coord_sys.update(time)
            for child in part.parts:
                recurse_parts(part.parts[child], time)

        for part in self.parts:
            recurse_parts(self.parts[part], self.time)

    def _update_actor_bounds(self, part_bounds):
        """
        Updates the actor's bounding box based on the provided part bounds.

        If the actor's bounds are not initialized, they are set to the provided
        part bounds. Otherwise, the actor's bounds are updated to encompass both
        the existing bounds and the provided part bounds.

        Args:
            part_bounds (list or None): A list containing the bounding box of a part
                in the format [xmin, xmax, ymin, ymax, zmin, zmax]. If None, the
                method does nothing.

        Returns:
            None
        """
        if part_bounds is None:
            return
        if self.bounds is None:
            self.bounds = part_bounds
        else:
            self.bounds = list(self.bounds)
            self.bounds[0] = np.minimum(self.bounds[0], part_bounds[0])
            self.bounds[1] = np.maximum(self.bounds[1], part_bounds[1])
            self.bounds[2] = np.minimum(self.bounds[2], part_bounds[2])
            self.bounds[3] = np.maximum(self.bounds[3], part_bounds[3])
            self.bounds[4] = np.minimum(self.bounds[4], part_bounds[4])
            self.bounds[5] = np.maximum(self.bounds[5], part_bounds[5])

    def get_mesh(self):
        """
        Get all meshes from the actor's parts.

        Returns:
        --------
        list
            A list of pyvista mesh objects from all parts of the actor that have meshes.
            Empty list if no parts have meshes.
        """
        all_meshes = []
        for part in self.parts:
            if self.parts[part].mesh is not None:
                all_meshes.append(self.parts[part].mesh)
        return all_meshes

    def get_bounds(self):
        """
        Get the bounding box of all meshes in the actor.

        This method calculates the overall bounding box that encompasses all mesh parts
        of the actor by finding the minimum and maximum extents in each dimension.

        Returns:
        --------
        numpy.ndarray
            Array containing [x_min, x_max, y_min, y_max, z_min, z_max] bounds.
            Returns array of infinities if no meshes are present.
        """
        bounds = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]
        for part in self.parts:
            # of all the returned meshes, bet the maximum bounds across all meshes
            # get the max/min bounds across all meshes
            if self.parts[part].mesh is not None:
                temp_bounds = self.parts[part].mesh.bounds
                bounds[0] = min(bounds[0], temp_bounds[0])
                bounds[1] = max(bounds[1], temp_bounds[1])
                bounds[2] = min(bounds[2], temp_bounds[2])
                bounds[3] = max(bounds[3], temp_bounds[3])
                bounds[4] = min(bounds[4], temp_bounds[4])
                bounds[5] = max(bounds[5], temp_bounds[5])
        # print(f'Bounds of the all mesh: {bounds}')
        return np.array(bounds)

    def get_center(self):
        """
        Get the center point of the actor's bounding box.

        Returns:
        --------
        numpy.ndarray
            Array containing [x_center, y_center, z_center] coordinates of the
            geometric center of all meshes in the actor.
        """
        bounds = self.get_bounds()
        center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        return np.array(center)

# class to load collada .dae animation from mixamo.com
class AnimatedDAE:
    def __init__(self,
                 filename,
                 material='human_avg',
                 speed_factor=1.0,
                 scale_mesh=1.0,
                 parent_h_node=None):
        """
        Initialize an AnimatedDAE instance for loading Mixamo animations.

        This class loads Collada (.dae) animation files from mixamo.com and creates
        an animated actor with joint transformations and mesh parts.

        Parameters:
        ------------
        filename : str
            Path to the .dae animation file.
        material : str, optional
            Material name to apply to the animated actor. Default is 'human_avg'.
        speed_factor : float, optional
            Speed multiplier for the animation playback. Default is 1.0.
        scale_mesh : float, optional
            Scale factor to apply to all meshes. Default is 1.0.
        parent_h_node : handle, optional
            Parent node handle for the animated actor. If None, creates a root actor.

        Attributes:
        -----------
        all_actors : dict
            Dictionary containing all actor parts of the animated character.
        clipLength : float
            Duration of the animation clip in seconds.
        sceneTree : dict
            Hierarchical structure of the animated actor parts.
        """

        self.mat_manager = MaterialManager()
        # the I've seen some clips 5x slower than it should be, so this hardcoded value can be adjusted
        original_clip_factor = 1
        self.speed_up_factor = speed_factor
        self.scale_mesh = scale_mesh

        self.mat_idx = self.mat_manager.get_index(material)
        self.all_actors = {}
        self.modelName = None
        self.root_handle = None

        paths = get_repo_paths()
        self.base_path = paths.models

        self.dae = collada.Collada(filename)


        # transform to scene coords
        # x -> +y y -> +z  z -> +x
        # quick fix, the mixamo models seem to be wit Y_up and I am expecting Z_Up
        # so for now I will just rotate the root object (which is the hips) to fix this

        self.local2global = np.asarray([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, +1]])

        self.global_rot = self.local2global

        self.global2local = np.linalg.inv(self.local2global)
        # rescale to meters
        self.units2meters = self.dae.assetInfo.unitmeter
        self.rescaleTransform = np.ones((4, 4))
        self.rescaleTransform[0:3, 3] = self.units2meters * self.scale_mesh
        self.color = [0.8, 0.8, 0.8]
        # build joint transform interpolations
        jointTransforms = {}
        self.clipLength = -1
        for anim in self.dae.animations:
            times = anim.sourceById[anim.name + "-Matrix-animation-input"]
            times = np.reshape(times, -1)
            times = times - np.min(times)  # zero shift
            times = times / (self.speed_up_factor * original_clip_factor)
            self.clipLength = max(self.clipLength, times[-1])
            transforms = anim.sourceById[anim.name + "-Matrix-animation-output-transform"]
            transforms = np.reshape(transforms, (-1, 4, 4))
            transforms = transforms * self.rescaleTransform
            if anim.name == 'mixamorig_Hips':  # this is the root fo rthis specific animation, so rotating it to global orientation
                for n in range(len(transforms)):
                    transforms[n] = np.dot(transforms[n], self.global_rot)
                    transforms[n][0:3, 3] = [0, 0, 1]  # offset in z direction to account for rotation around hips
            interpFunc = scipy.interpolate.interp1d(times, transforms, axis=0, assume_sorted=True)
            jointTransforms[anim.name] = interpFunc

        # instantiate scene tree
        def recurseJoints(
                sceneTree=None,
                nodeID=None
        ):
            if sceneTree is None:
                nodeID = "root"
                meshInst_root = Actor(parent_h_node=parent_h_node,color=self.color)
                meshInst_root.actor_type = 'pedestrian'
                self.all_actors[nodeID] = meshInst_root

                self.root_handle = meshInst_root.h_node

                sceneTree = {}
                children = [self.dae.scene.nodes[0]]
                sceneTree[nodeID] = (meshInst_root, children)
            (nodeCS, children) = sceneTree[nodeID]
            for child in children:
                childID = child.id
                meshInst_child = Actor(parent_h_node=nodeCS.h_node, mat_idx=self.mat_idx, scale_mesh=self.scale_mesh,color=self.color)
                meshInst_child.actor_type = 'pedestrian'
                self.all_actors[childID] = meshInst_child
                # meshInst_child.color = [1, .396, .266]

                if childID in jointTransforms:
                    meshInst_child.coord_sys.transforms = jointTransforms[childID]
                else:
                    childTransform = child.matrix * self.rescaleTransform
                    # meshInst_child.coord_sys.set(childTransform)
                # childCS.update(time=0.,inGlobal=False)
                sceneTree[childID] = (meshInst_child, child.children)
                recurseJoints(sceneTree, childID)
            return sceneTree

        self.sceneTree = recurseJoints()

        meshes = []
        self.cad_path = os.path.join(self.base_path, 'pedestrian_bones_stl')
        self.all_possible_files = glob.glob(f'{self.cad_path}/*.stl')

        # 'mixamorig_Hips': 'LEFT_HIP.stl', this one seems rotated weird
        joint_to_bone_stl_map = {'mixamorig_Hips': 'Hips_rotated.stl',
                                 'mixamorig_Neck': 'Neck.stl',
                                 'mixamorig_Head': 'Head.stl',
                                 'mixamorig_LeftArm': 'LeftShoulder.stl',
                                 'mixamorig_RightArm': 'RightShoulder.stl',
                                 'mixamorig_LeftForeArm': 'LeftElbow.stl',
                                 'mixamorig_RightForeArm': 'RightElbow.stl',
                                 'mixamorig_LeftUpLeg': 'LeftHip.stl',
                                 'mixamorig_LeftLeg': 'LeftKnee.stl',
                                 'mixamorig_RightLeg': 'RightKnee.stl',
                                 'mixamorig_RightUpLeg': 'RightHip.stl',
                                 'mixamorig_Spine1': 'Chest2.stl'}

        # create a function that will check if jointTransforms key is in the list of all possible files.
        # If it is, then load the mesh. The name of the joint should match within a wild card of the file name
        # I couldn't get this working quite right, so I just created a mapping dictionary
        def check_for_meshes(joint_name):
            if joint_name in joint_to_bone_stl_map:
                filename_of_stl = os.path.join(self.cad_path, joint_to_bone_stl_map[joint_name])
                return filename_of_stl

            return False

        for joint in jointTransforms.keys():
            filename = check_for_meshes(joint)
            if filename:
                self.sceneTree[joint][0].add_part(filename=filename, mat_idx=self.mat_idx,color=self.color)

class AnimatedDAE_experimental:
    def __init__(self,
                 filename,
                 material='human_avg',
                 speed_factor=1.0,
                 scale_mesh=1.0,
                 parent_h_node=None):
        """
        Initialize an AnimatedDAE_CMU instance for loading CMU motion capture data.

        This class loads Collada (.daez) animation files from CMU motion capture datasets
        and creates an animated actor with bone structure and animations. It's specifically
        designed to handle CMU BVH motion capture data that has been converted to DAE format.

        Parameters:
        ------------
        filename : str
            Path to the .daez animation file containing CMU motion capture data.
        material : str, optional
            Material name to apply to the animated actor. Default is 'human_avg'.
        speed_factor : float, optional
            Speed multiplier for the animation playback. Default is 1.0.
        scale_mesh : float, optional
            Scale factor to apply to all meshes. Default is 1.0.
        parent_h_node : handle, optional
            Parent node handle for the animated actor. If None, creates a root actor.

        Attributes:
        -----------
        all_actors : dict
            Dictionary containing all actor parts of the animated character.
        clipLength : float
            Duration of the animation clip in seconds.
        sceneTree : dict
            Hierarchical structure of the animated actor parts.
        animation_name : str
            Name of the loaded animation.
        times : numpy.ndarray
            Time values for the animation keyframes.

        Notes:
        ------
        This class is specifically designed for CMU motion capture data and includes
        specialized handling for CMU naming conventions and bone structures.
        """

        self.mat_manager = MaterialManager()
        # the I've seen some clips 5x slower than it should be, so this hardcoded value can be adjusted
        original_clip_factor = 1
        self.speed_up_factor = speed_factor
        self.scale_mesh = scale_mesh

        self.mat_idx = self.mat_manager.get_index(material)
        self.all_actors = {}
        self.modelName = None
        self.root_handle = None

        paths = get_repo_paths()
        self.base_path = paths.models

        self.dae = collada.Collada(filename)


        # transform to scene coords
        # x -> +y y -> +z  z -> +x
        # quick fix, the mixamo models seem to be wit Y_up and I am expecting Z_Up
        # so for now I will just rotate the root object (which is the hips) to fix this

        # self.local2global = np.asarray([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, +1]])
        self.local2global = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, +1]])
        self.global_rot = self.local2global

        self.global2local = np.linalg.inv(self.local2global)
        # rescale to meters
        self.units2meters = self.dae.assetInfo.unitmeter
        self.rescaleTransform = np.ones((4, 4))
        self.rescaleTransform[0:3, 3] = self.units2meters * self.scale_mesh
        self.color = [0.8, 0.8, 0.8]
        # build joint transform interpolations
        jointTransforms = {}
        self.clipLength = -1

        all_nodes = []
        jointTransforms = {}
        if len(self.dae.animations)  == 1:
            self.dae.animations = self.dae.animations[0].children
        all_nodes = list(self.dae.animations[0].sourceById.keys())

        # get time
        for node in all_nodes:
            if ('TIME',) == self.dae.animations[0].sourceById[node].components:
                times = self.dae.animations[0].sourceById[node]
                times = np.reshape(times, -1)
                times = times - np.min(times)  # zero shift
                times = times / (self.speed_up_factor * original_clip_factor)
                self.clipLength = max(self.clipLength, times[-1])
        
        all_nodes_only_transforms = [node for node in all_nodes if ('TRANSFORM',) == self.dae.animations[0].sourceById[node].components]

        # for each item in list, find the common string prefix and remove it
        common_prefix = os.path.commonprefix(all_nodes_only_transforms)
        all_nodes_only_transforms2 = [node.replace(common_prefix, '') for node in all_nodes_only_transforms]
        if 'transform-output' in all_nodes_only_transforms2:
            root_transform = all_nodes_only_transforms2[all_nodes_only_transforms2.index('transform-output')]
            all_nodes_only_transforms2.remove('transform-output')  # this is a root, uses a different naming convention, will remove and put back later
        common_prefix = os.path.commonprefix(all_nodes_only_transforms2)
        all_nodes_only_transforms3 = [node.replace(common_prefix, '') for node in all_nodes_only_transforms2]

        # remove common endings
        common_suffix = os.path.commonprefix([node[::-1] for node in all_nodes_only_transforms3])
        common_suffix = common_suffix[::-1]  # reverse it back
        all_nodes_only_transforms2 = [node.replace(common_suffix, '') for node in all_nodes_only_transforms3]
        all_nodes_only_transforms2 = [node.replace('_', '') for node in all_nodes_only_transforms2]  # remove underscores
        all_nodes_only_transforms2 = [node.replace('output', '') for node in all_nodes_only_transforms2]  # remove output suffix
        all_nodes_only_transforms2 = [node.replace('input', '') for node in all_nodes_only_transforms2]  # remove input suffix
        all_nodes_only_transforms2 = [node.replace('interpolation', '') for node in all_nodes_only_transforms2]  # remove interpolation suffix

        # add root node back into the list
        all_nodes_only_transforms2.insert(0,'transform-output')

        for n, anim in enumerate(self.dae.animations):
            if ('TRANSFORM',) == anim.sourceById[all_nodes_only_transforms[n]].components:
                transforms = anim.sourceById[all_nodes_only_transforms[n]]
                transforms = np.reshape(transforms, (-1, 4, 4))
                transforms = transforms * self.rescaleTransform 
                if n==0:  # this is the root fo rthis specific animation, so rotating it to global orientation
                    for m in range(len(transforms)):
                        transforms[m] = np.dot(transforms[m], self.global_rot)
                    interpFunc = scipy.interpolate.interp1d(times, transforms, axis=0, assume_sorted=True)
                    # jointTransforms2[self.dae2.animations[0].sourceById[node].id] = interpFunc
                    print(all_nodes_only_transforms[n])
                    jointTransforms['Root'] = interpFunc
                else:
                    interpFunc = scipy.interpolate.interp1d(times, transforms, axis=0, assume_sorted=True)
                    # jointTransforms2[self.dae2.animations[0].sourceById[node].id] = interpFunc
                    print(all_nodes_only_transforms[n])
                    jointTransforms[all_nodes_only_transforms2[n]] = interpFunc
        def recurseJoints(
                sceneTree=None,
                nodeID=None
        ):
            if sceneTree is None:
                nodeID = "root"
                meshInst_root = Actor(parent_h_node=parent_h_node,color=self.color)
                meshInst_root.actor_type = 'pedestrian'
                self.all_actors[nodeID] = meshInst_root
                # rotate root so it is in the correct orientation (dae file sayz Zup but it looks like it is Yup)
                meshInst_root.coord_sys.rot = np.array([[1, 0, 0],
                                                        [0, 0, 1],
                                                        [0, 1, 0]])    
                meshInst_root.coord_sys.update()
                self.root_handle = meshInst_root.h_node

                sceneTree = {}
                children = [self.dae.scene.nodes[0]]
                sceneTree[nodeID] = (meshInst_root, children)
            (nodeCS, children) = sceneTree[nodeID]
            for child in children:
                if hasattr(child, 'id'):
                    childID = child.id
                    childID = childID.split('_')[-1]  # remove the prefix, since it is not needed
                    meshInst_child = Actor(parent_h_node=nodeCS.h_node, mat_idx=self.mat_idx, scale_mesh=self.scale_mesh,color=self.color)
                    meshInst_child.actor_type = 'pedestrian'
                    self.all_actors[childID] = meshInst_child
                    # meshInst_child.color = [1, .396, .266]

                    if childID in jointTransforms:
                        meshInst_child.coord_sys.transforms = jointTransforms[childID]
                    else:
                        childTransform = child.matrix * self.rescaleTransform 
                        # meshInst_child.coord_sys.set(childTransform)
                    # childCS.update(time=0.,inGlobal=False)
                    sceneTree[childID] = (meshInst_child, child.children)
                    recurseJoints(sceneTree, childID)
            return sceneTree

        self.sceneTree = recurseJoints()

        meshes = []
        self.cad_path = os.path.join(self.base_path, 'pedestrian_bones_stl')
        self.all_possible_files = glob.glob(f'{self.cad_path}/*.stl')

        # 'mixamorig_Hips': 'LEFT_HIP.stl', this one seems rotated weird
        joint_to_bone_stl_map = {'mixamorig_Hips': 'Hips_rotated.stl',
                                 'mixamorig_Neck': 'Neck.stl',
                                 'mixamorig_Head': 'Head.stl',
                                 'mixamorig_LeftArm': 'LeftShoulder.stl',
                                 'mixamorig_RightArm': 'RightShoulder.stl',
                                 'mixamorig_LeftForeArm': 'LeftElbow.stl',
                                 'mixamorig_RightForeArm': 'RightElbow.stl',
                                 'mixamorig_LeftUpLeg': 'LeftHip.stl',
                                 'mixamorig_LeftLeg': 'LeftKnee.stl',
                                 'mixamorig_RightLeg': 'RightKnee.stl',
                                 'mixamorig_RightUpLeg': 'RightHip.stl',
                                 'mixamorig_Spine1': 'Chest2.stl',
                                 'root': 'Hips_rotated.stl',
                                 'Neck': 'Neck.stl',
                                 'Head': 'Head.stl',
                                 'LeftArm': 'LeftShoulder_rotated.stl',
                                 'RightArm': 'RightShoulder_rotated.stl',
                                 'LeftForeArm': 'LeftElbow_rotated.stl',
                                 'RightForeArm': 'RightElbow_rotated.stl',
                                 'LeftUpLeg': 'LeftHip_rotated.stl',
                                 'LeftLeg': 'LeftKnee_rotated.stl',
                                 'RightUpLeg': 'RightHip_rotated.stl',
                                 'RightLeg': 'RightKnee_rotated.stl',
                                 'Spine1': 'Chest2.stl'
                                 }

        # create a function that will check if jointTransforms key is in the list of all possible files.
        # If it is, then load the mesh. The name of the joint should match within a wild card of the file name
        # I couldn't get this working quite right, so I just created a mapping dictionary
        def check_for_meshes(joint_name):
            if joint_name in joint_to_bone_stl_map:
                filename_of_stl = os.path.join(self.cad_path, joint_to_bone_stl_map[joint_name])
                return filename_of_stl

            return False

        for joint in jointTransforms.keys():
            filename = check_for_meshes(joint)
            if filename:
                self.sceneTree[joint][0].add_part(filename=filename, mat_idx=self.mat_idx,color=self.color)

class AnimatedDAE_CMU:
    def __init__(self,
                 filename,
                 material='human_avg',
                 speed_factor=1.0,
                 scale_mesh=1.0,
                 parent_h_node=None,
                 loop_animation=False):

        filename = os.path.abspath(filename)
        directory = os.path.dirname(filename)

        paths = get_repo_paths()
        CMU_mesh_path = os.path.join( paths.models,'CMU_Database','meshes/')
        self.all_possible_files = glob.glob(f'{CMU_mesh_path}*.stl')

        self.mat_manager = MaterialManager()
        # the I've seen some clips 5x slower than it should be, so this hardcoded value can be adjusted
        original_clip_factor = 5
        self.speed_up_factor = speed_factor
        self.scale_mesh = scale_mesh
        self.animation_name = ''
        self.mat_idx = self.mat_manager.get_index(material)
        self.all_actors = {}
        self.modelName = None
        self.root_handle = None
        self.delay_animation = 0
        self.base_path = os.path.dirname(filename)

        if filename.endswith(".daez"):
            base_filename = os.path.basename(filename)
            base_filename = base_filename.replace(".daez", ".dae")
            self.dae = collada.Collada(filename, zip_filename=base_filename)
        else:
            self.dae = collada.Collada(filename)

        # self.local2global = np.asarray([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, +1]])

        self.global_rot = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, +1]])

        # self.global2local = np.linalg.inv(self.local2global)
        # rescale to meters
        self.units2meters = self.dae.assetInfo.unitmeter
        self.rescaleTransform = np.ones((4, 4))
        self.rescaleTransform[0:3, 3] = self.units2meters * self.scale_mesh
        self.loop_animation = loop_animation
        self.color = [0.8, 0.8, 0.8]
        # build joint transform interpolations
        jointTransforms = {}
        self.clipLength = -1
        for anim in self.dae.animations:
            name = list(anim.sourceById.keys())
            times = anim.sourceById[name[0]]
            times = np.reshape(times, -1)
            times = times - np.min(times)  # zero shift
            times = times / (self.speed_up_factor * original_clip_factor)
            self.times = times
            self.clipLength = max(self.clipLength, times[-1])
            self.animation_name = anim.name
            self.animation_id = anim.id.replace('action_container-', '')
            reference_part = f"{self.animation_id}_{self.animation_name}_Hips_pose_matrix-output"
            for each in name:
                if 'output' in each:
                    joint_name = each.replace('_pose_matrix-output', '')
                    joint_name = joint_name.replace(self.animation_id + '_', '')
                    joint_name = joint_name.replace(self.animation_name + '_', '')
                    transforms = anim.sourceById[each]
                    transforms = np.reshape(transforms, (-1, 4, 4))
                    transforms = transforms * self.rescaleTransform
                    if 'Hips_pose_matrix' in each:
                        # this is the root for this specific animation,
                        # so rotating it to global orientation
                        for n in range(len(transforms)):
                            # transforms2[:, :, -1][:, :3] = transforms2[:, :, -1][:, :3] - self.animation_original_pos
                            transforms[n] = np.dot(transforms[n], self.global_rot)
                            # transforms2[n][0:3, 3] = [0, 0, 1]  # offset in z direction to account for rotation around hips
                    interpFunc = scipy.interpolate.interp1d(times, transforms, axis=0, assume_sorted=True)
                    jointTransforms[joint_name] = interpFunc
        original_fps = 1 / np.average(np.diff(times))

        def check_for_meshes(joint_name):
            joint_name = joint_name + '.stl'
            for each in self.all_possible_files:
                each_filename = os.path.split(each)[1]
                if joint_name == each_filename:
                    filename_of_stl = each
                    return filename_of_stl
            return False

        def recurseJoints(
                sceneTree=None,
                nodeID=None):

            if sceneTree is None:
                nodeID = "root"
                meshInst_root = Actor(parent_h_node=parent_h_node,color=self.color)

                meshInst_root.actor_type = 'pedestrian'
                self.all_actors[nodeID] = meshInst_root

                self.root_handle = meshInst_root.h_node

                sceneTree = {}
                children = [self.dae.scene.nodes[0]]
                sceneTree[nodeID] = (meshInst_root, children)
            (nodeCS, children) = sceneTree[nodeID]
            for child in children:
                if hasattr(child, 'id'):
                    childID_orig = child.id
                    childID = child.id
                    childID = childID.replace(self.animation_id + '_', '')

                    meshInst_child = Actor(parent_h_node=nodeCS.h_node, mat_idx=self.mat_idx,
                                           scale_mesh=self.scale_mesh,color=self.color)
                    filename = check_for_meshes(childID)
                    if filename:
                        # filename = os.path.join(temp_path,'sphere.stl')
                        meshInst_child.add_part(filename=filename,
                                                name=childID,
                                                mat_idx=self.mat_idx,
                                                color=self.color)
                    meshInst_child.actor_type = 'pedestrian'

                    if childID in jointTransforms:
                        transforms = jointTransforms[childID]
                        meshInst_child.coord_sys.transforms = transforms
                    else:
                        # joints that are not in the animation will be set to the default position
                        childTransform = child.matrix * self.rescaleTransform
                        meshInst_child.coord_sys.transform4x4 = childTransform

                    self.all_actors[childID] = meshInst_child
                    sceneTree[childID] = (meshInst_child, child.children)
                    recurseJoints(sceneTree, childID)
            return sceneTree

        self.sceneTree = recurseJoints()


@staticmethod
def increment_name(base_name, existing_names):
    if not isinstance(existing_names, list):
        existing_names = list(existing_names)
    while base_name in existing_names:
        # split the name into a list of words by underscore
        name_list = base_name.split('_')
        # check if the last word is a number
        if name_list[-1].isdigit():
            # if it is, increment it
            name_list[-1] = str(int(name_list[-1]) + 1)
        else:
            # if it isn't, add a 2 to the end
            name_list.append('2')
        # join the list back into a string with underscores
        base_name = '_'.join(name_list)
    return base_name
