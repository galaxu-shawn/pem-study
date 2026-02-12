"""
Coordinate System Utilities for ANSYS Perceive EM

This module provides the CoordSys class for managing 3D coordinate systems
within the ANSYS Perceive EM electromagnetic simulation environment. It handles
position, rotation, linear and angular velocities, and transformations for
scene nodes, including support for animated transforms and focused ray groups.

Created on Fri Jan 26 15:00:00 2024

@author: asligar
"""
import numpy as np
import scipy.interpolate

from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API


class CoordSys:
    """
    A coordinate system manager for 3D scene nodes in ANSYS Perceive EM.
    
    This class provides a comprehensive interface for managing the position,
    orientation, and motion of objects in a 3D scene. It supports both static
    and dynamic transformations, velocity estimation, and focused ray groups
    for adaptive ray shooting.
    
    Attributes:
        h_node: Handle to the scene node
        h_mesh: Handle to the mesh element (optional)
        parent_h_node: Handle to the parent node (optional)
        transforms: Function for animated transformations (optional)
        vel_estimator: Velocity estimation object for complex motion
        pem: API reference for core operations
        rss_py: RSS Python interface
        dt: Time delta between updates
        time: Current simulation time
        
    Private Attributes:
        _rot: 3x3 rotation matrix
        _pos: 3D position vector
        _lin: 3D linear velocity vector
        _ang: 3D angular velocity vector
        _transform4x4: 4x4 transformation matrix
    """
    
    def __init__(self, h_node=None, h_mesh=None, parent_h_node=None, target_ray_spacing=None):
        """
        Initialize a CoordSys instance.

        Parameters:
        -----------
        h_node : handle, optional
            The handle for the scene node. If None, a new scene node is created.
            
        h_mesh : handle, optional
            The handle for the mesh element to associate with this coordinate system.
            
        parent_h_node : handle, optional
            The handle for the parent node. If provided, this node becomes a child
            of the parent in the scene hierarchy.
            
        target_ray_spacing : float, optional
            Spacing in meters for adaptive ray shooting. If provided, creates a
            focused ray group that dynamically follows this coordinate system.
            
        Raises:
        -------
        RuntimeError
            If API calls fail during initialization.
        """
        # Get API references
        self.pem_api_manager = Perceive_EM_API()
        pem = self.pem_api_manager.pem  # The configured API object
        rss_py = self.pem_api_manager.RssPy

        # Initialize transform and velocity estimation attributes
        # These are functions that can be assigned to model, primarily used for importing DAE files
        # If these have been assigned, when an update is called, it will use this function to determine the object transform
        self.transforms = None
        
        # Velocity estimator for complex motion that may exist in something that has been assigned movement 
        # through the previous transforms
        self.vel_estimator = None
        
        # Create or use provided scene node
        if h_node is None:
            h_node = rss_py.SceneNode()
            # Add to scene hierarchy
            if parent_h_node is None:
                # Add as root node
                self.pem_api_manager.isOK(pem.addSceneNode(h_node))
            else:
                # Add as child of parent node
                self.pem_api_manager.isOK(pem.addSceneNode(h_node, parent_h_node))

        # Store handles and references
        self.h_node = h_node
        self.h_mesh = h_mesh
        self.parent_h_node = parent_h_node
        self.pem = pem
        self.rss_py = rss_py
        
        # Initialize transformation parameters
        self._rot = np.eye(3)          # 3x3 identity rotation matrix
        self._pos = np.zeros(3)        # Zero position vector [x, y, z]
        self._lin = np.zeros(3)        # Zero linear velocity [vx, vy, vz]
        self._ang = np.zeros(3)        # Zero angular velocity [wx, wy, wz]
        self._transform4x4 = np.eye(4) # 4x4 identity transformation matrix
        
        # Initialize timing parameters
        self.dt = 0     # Time delta between updates
        self.time = 0   # Current simulation time

        # Associate mesh with scene node if provided
        if h_mesh is not None:
            self.pem_api_manager.isOK(pem.setSceneElement(self.h_node, self.h_mesh))

        # Set up adaptive ray shooting if requested
        # Target ray spacing is an adaptive ray shoot that will dynamically move with the target
        # Spacing is defined in meters
        if target_ray_spacing is not None:
            self.pem.addFocusedRayGroup(self.h_node, target_ray_spacing)

    def _update_with_transforms(self, time=None):
        """
        Update the coordinate system using assigned transform functions.
        
        This private method handles time-based animations and velocity estimation
        for objects with complex motion patterns, typically imported from DAE files.
        
        Parameters:
        -----------
        time : float, optional
            Current simulation time. Used for transform function evaluation
            and velocity estimation.
            
        Raises:
        -------
        RuntimeError
            If velocity estimation fails during the update process.
        """
        # Calculate time delta
        self.dt = time - self.time
        self.time = time

        # Apply transform functions if available
        if self.transforms is not None and time is not None:
            # Account for limited time animations by using modulo
            # This creates looping animations when time exceeds the transform duration
            temp_transform = self.transforms(np.mod(time, self.transforms.x[-1]))
            
            # Extract position and rotation from 4x4 transform matrix
            self._pos = temp_transform[0:3, 3]    # Translation vector
            self._rot = temp_transform[0:3, 0:3]  # Rotation matrix

        # Update velocity estimation
        temp_pos = self._pos
        
        # Initialize or reset velocity estimator if needed
        if (self.vel_estimator is None) or (self.dt <= 0):
            self.vel_estimator = self.rss_py.VelocityEstimate()
            # Order of estimate - 3 seems to work best for most applications
            self.vel_estimator.setApproximationOrder(3)
            
        # Push current state to velocity estimator
        ret = self.vel_estimator.push(
            time,
            np.ascontiguousarray(self._rot, dtype=np.float64),
            np.ascontiguousarray(temp_pos, dtype=np.float64)
        )
        
        if ret == False:
            raise RuntimeError("Error pushing velocity estimate")
            
        # Get updated linear and angular velocities
        (_, self._lin, self._ang) = self.vel_estimator.get()

    def update(self, time=None):
        """
        Update the coordinate system in the simulation environment.
        
        This method applies the current position, rotation, and velocity parameters
        to the scene node, updating either global or parent-relative coordinates
        depending on the node hierarchy.
        
        Parameters:
        -----------
        time : float, optional
            Current simulation time. If provided, triggers transform-based updates
            and velocity estimation.
            
        Raises:
        -------
        RuntimeError
            If API calls fail during the coordinate system update.
        """
        # Update with transform functions if time is provided
        if time is not None:
            self._update_with_transforms(time)

        # Update coordinate system in the simulation
        if self.parent_h_node is None:
            # Update as global coordinate system (root node)
            self.pem_api_manager.isOK(self.pem.setCoordSysInGlobal(
                self.h_node,
                np.ascontiguousarray(self._rot, dtype=np.float64),
                np.ascontiguousarray(self._pos, dtype=np.float64),
                np.ascontiguousarray(self._lin, dtype=np.float64),
                np.ascontiguousarray(self._ang, dtype=np.float64)
            ))
        else:
            # Update as parent-relative coordinate system (child node)
            self.pem_api_manager.isOK(self.pem.setCoordSysInParent(
                self.h_node,
                np.ascontiguousarray(self._rot, dtype=np.float64),
                np.ascontiguousarray(self._pos, dtype=np.float64),
                np.ascontiguousarray(self._lin, dtype=np.float64),
                np.ascontiguousarray(self._ang, dtype=np.float64)
            ))

    @property
    def transform4x4(self):
        """
        Get the current 4x4 transformation matrix in global coordinates.
        
        This property retrieves the current global transformation matrix,
        which combines rotation and translation into a single 4x4 matrix
        suitable for 3D graphics operations.
        
        Returns:
        --------
        numpy.ndarray
            4x4 transformation matrix in homogeneous coordinates:
            [[R11, R12, R13, Tx],
             [R21, R22, R23, Ty],
             [R31, R32, R33, Tz],
             [0,   0,   0,   1 ]]
            where R is the 3x3 rotation matrix and T is the translation vector.
        """
        # Ensure coordinate system is up-to-date
        self.update()
        
        # Get global coordinate system from API
        (ret, rot, pos, lin, ang) = self.pem.coordSysInGlobal(self.h_node)
        
        # Construct 4x4 transformation matrix
        # Combine rotation matrix (3x3) with position vector (3x1)
        self._transform4x4 = np.concatenate((np.asarray(rot), np.asarray(pos).reshape((-1, 1))), axis=1)
        # Add homogeneous coordinate row [0, 0, 0, 1]
        self._transform4x4 = np.concatenate((self._transform4x4, np.array([[0, 0, 0, 1]])), axis=0)
        
        return self._transform4x4

    @transform4x4.setter
    def transform4x4(self, value):
        """
        Set the coordinate system using a 4x4 transformation matrix.
        
        Parameters:
        -----------
        value : numpy.ndarray
            4x4 transformation matrix in homogeneous coordinates.
            The matrix should be structured as:
            [[R11, R12, R13, Tx],
             [R21, R22, R23, Ty],
             [R31, R32, R33, Tz],
             [0,   0,   0,   1 ]]
        """
        # Extract position and rotation from 4x4 matrix
        self._pos = value[0:3, 3]    # Translation vector (last column, first 3 rows)
        self._rot = value[0:3, 0:3]  # Rotation matrix (upper-left 3x3)
        
        # Apply the changes
        self.update()

    @property
    def pos(self):
        """
        Get the current position vector.
        
        Returns:
        --------
        numpy.ndarray
            3D position vector [x, y, z] as a copy of the internal position.
        """
        return np.array(self._pos)

    @pos.setter
    def pos(self, value):
        """
        Set the position vector.
        
        Parameters:
        -----------
        value : array-like
            3D position vector [x, y, z] in the same units as the simulation.
        """
        self._pos = value

    @property
    def rot(self):
        """
        Get the current rotation matrix.
        
        Returns:
        --------
        numpy.ndarray
            3x3 rotation matrix as a copy of the internal rotation matrix.
            The matrix represents the orientation of the local coordinate
            system relative to its parent or global coordinates.
        """
        return np.array(self._rot)

    @rot.setter
    def rot(self, value):
        """
        Set the rotation matrix.
        
        Parameters:
        -----------
        value : array-like
            3x3 rotation matrix. Should be orthogonal with determinant +1
            for proper rotation (no scaling or reflection).
        """
        self._rot = value

    @property
    def lin(self):
        """
        Get the current linear velocity vector.
        
        Returns:
        --------
        numpy.ndarray
            3D linear velocity vector [vx, vy, vz] as a copy of the internal
            linear velocity.
        """
        return np.array(self._lin)

    @lin.setter
    def lin(self, value):
        """
        Set the linear velocity vector.
        
        Parameters:
        -----------
        value : array-like
            3D linear velocity vector [vx, vy, vz] in units per time.
        """
        self._lin = value

    @property
    def ang(self):
        """
        Get the current angular velocity vector.
        
        Returns:
        --------
        numpy.ndarray
            3D angular velocity vector [wx, wy, wz] as a copy of the internal
            angular velocity, typically in radians per time unit.
        """
        return np.array(self._ang)

    @ang.setter
    def ang(self, value):
        """
        Set the angular velocity vector.
        
        Parameters:
        -----------
        value : array-like
            3D angular velocity vector [wx, wy, wz] in radians per time unit.
            The vector direction defines the rotation axis (right-hand rule),
            and the magnitude defines the rotation rate.
        """
        self._ang = value
