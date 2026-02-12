"""
Simulation Options Module

This module provides a SimulationOptions class that manages simulation parameters 
for electromagnetic field calculations, including ray tracing settings, GPU 
configuration, and various simulation options.
"""

import numpy as np
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API


class SimulationOptions:
    """
    A class to manage simulation options for electromagnetic field calculations.
    
    This class provides a convenient interface for configuring simulation parameters
    such as ray spacing/density, reflections, transmissions, GPU settings, and other
    simulation options. It acts as a wrapper around the api_core API calls.
    
    ray_shoot_method, grid is default method with a ray spacing value determining the number of rays defined
    on a grid at 200meters. SBR uses tessalated geometry to determine minimum number of rays, with at least 1
    ray per triangle, and subdivides the triangles to achieve the ray density if needed.

    Attributes:
        api: Reference to the api_core.api instance
        ray_shoot_method (str): Method used for ray shooting (default: "grid"), 
                                options are "grid", "sbr"
        ray_spacing (float): Spacing between rays in meters (default: 0.1)
        ray_density (float): Ray density override for ray_spacing (default: None)
        max_reflections (int): Maximum number of reflections (default: 3)
        max_transmissions (int): Maximum number of transmissions (default: 1)
        max_batches (int): Maximum number of ray batches (default: 100)
        go_blockage (int): GO blockage setting (-1 to disable, default: -1)
        gpu_device (int or list): GPU device ID(s) to use (default: 0)
        gpu_quota (float): GPU memory quota (default: 0.9)
        field_of_view (int): Field of view in degrees, 180 or 360 (default: 360)
        bounding_box (float): Maximum bounding box side length (default: None)
        has_gpu_been_set (bool): Flag to track if GPU has been configured
    """
    
    def __init__(self,center_freq=None):
        """
        Initialize SimulationOptions with default values.
        
        Sets up default simulation parameters and creates a reference to the API.
        """

        self.ray_shoot_method = "grid"  # Default ray shooting method
        self.pem_api_manager = Perceive_EM_API()
        self.pem  = self.pem_api_manager.pem  # The configured API object, all commands exist here
        self.RssPy = self.pem_api_manager.RssPy # 
        
        # Ray tracing parameters
        self._ray_spacing = 0.1
        if center_freq is not None:
            lambda_center = 2.99792458e8 / center_freq
            self._ray_density = np.sqrt(2) * lambda_center / self._ray_spacing
        else:
            self._ray_density = None  # ray spacing is default, if ray_density is set, it will override ray_spacing

        self.center_freq = center_freq  # Store center frequency for later use

        # Reflection and transmission limits
        self._max_reflections = 3
        self._max_transmissions = 1

        # Batch processing
        self._max_batches = 100

        # GO blockage setting: -1 means no go blockage, 0 means blockage starts at bounce 0, 
        # 1 means go blockage at 1st bounce
        self._go_blockage = -1

        # GPU configuration
        self._gpu_device = 0
        self._gpu_quota = 0.9

        # Field of view and bounding box settings
        self._field_of_view = 360  # 180 or 360 degrees
        self._bounding_box = None

        # Internal state tracking
        self.has_gpu_been_set = False

    @property
    def ray_shoot_method(self):
        """
        Get the current ray shooting method.
        
        Returns:
            str: The ray shooting method ("grid" or "sbr")
        """
        return self._ray_shoot_method
    @ray_shoot_method.setter
    def ray_shoot_method(self, value):
        """
        Set the ray shooting method.
        
        Args:
            value (str): The ray shooting method ("grid" or "sbr")
            
        Raises:
            ValueError: If the value is not "grid" or "sbr"
        """
        value = value.lower()
        if value not in ["grid", "sbr"]:
            raise ValueError("ray_shoot_method must be 'grid' or 'sbr'")
        if value == "sbr":
            if self.center_freq is None and int(self.pem_api_manager.version) <= 252:
                raise ValueError("ray_shoot_method 'sbr' requires a center frequency to be set in SimulationOptions")
            if int(self.pem_api_manager.version) <= 252:
                print('WARNING: SBR shoot method is a BETA feature in 25.2')
        self._ray_shoot_method = value

    # Ray spacing properties
    @property
    def ray_spacing(self):
        """
        Get the current ray spacing in meters.
        
        Returns:
            float: Ray spacing value in meters
        """
        return self._ray_spacing

    @ray_spacing.setter
    def ray_spacing(self, value):
        """
        Set the ray spacing in meters.
        
        Args:
            value (float): Ray spacing in meters
            
        Note:
            Setting ray_spacing will clear ray_density to None
        """
        if self.center_freq:
            lambda_center = 2.99792458e8 / self.center_freq
            self._ray_density = np.sqrt(2) * lambda_center / value
        else:
            self._ray_density = None
        self._ray_spacing = value


    @property
    def ray_density(self):
        """
        Get the current ray density.
        
        Returns:
            float or None: Ray density value, or None if using ray_spacing
        """
        return self._ray_density

    @ray_density.setter
    def ray_density(self, value):
        """
        Set the ray density, which overrides ray_spacing.
        
        Args:
            value (float): Ray density value
            
        Note:
            Setting ray_density will clear ray_spacing to None
        """
        if self.center_freq:
            lambda_center = 2.99792458e8 / self.center_freq
            self._ray_spacing = np.sqrt(2) * lambda_center / value
        else:
            print("Error: ray_density cannot be set without a center frequency")
        self._ray_density = value

    # Reflection and transmission properties
    @property
    def max_reflections(self):
        """
        Get the maximum number of reflections allowed.
        
        Returns:
            int: Maximum number of reflections
        """
        return self._max_reflections

    @max_reflections.setter
    def max_reflections(self, value):
        """
        Set the maximum number of reflections allowed.
        
        Args:
            value (int): Maximum number of reflections
        """
        self.pem_api_manager.isOK(self.pem.setMaxNumRefl(value))
        self._max_reflections = value

    @property
    def max_transmissions(self):
        """
        Get the maximum number of transmissions allowed.
        
        Returns:
            int: Maximum number of transmissions
        """
        return self._max_transmissions

    @max_transmissions.setter
    def max_transmissions(self, value):
        """
        Set the maximum number of transmissions allowed.
        
        Args:
            value (int): Maximum number of transmissions
        """
        self.pem_api_manager.isOK(self.pem.setMaxNumTrans(value))
        self._max_transmissions = value

    # Batch processing properties
    @property
    def max_batches(self):
        """
        Get the maximum number of ray batches.
        
        Returns:
            int: Maximum number of ray batches
        """
        return self._max_batches

    @max_batches.setter
    def max_batches(self, value):
        """
        Set the maximum number of ray batches.
        
        Args:
            value (int): Maximum number of ray batches
            
        Raises:
            ValueError: If value is not an integer
        """
        # Validate input type
        if not isinstance(value, int):
            raise ValueError("max_batches must be an integer")
        maxNumRayBatches = value

    # GO blockage properties
    @property
    def go_blockage(self):
        """
        Get the GO (Geometrical Optics) blockage setting.
        
        Returns:
            int: GO blockage setting (-1 to disable, 0+ for bounce number)
        """
        return self._go_blockage

    @go_blockage.setter
    def go_blockage(self, value):
        """
        Set the GO (Geometrical Optics) blockage setting.
        
        Args:
            value (int): GO blockage setting
                        -1 = disabled
                        0 = blockage starts at bounce 0
                        1+ = blockage starts at specified bounce number
                        
        Raises:
            ValueError: If value is not an integer
        """
        # Validate input type
        if not isinstance(value, int):
            raise ValueError("go_blockage must be an integer (-1 to disable)")
            
        # Apply GO blockage setting if enabled
        if value >= 0:
            self.pem_api_manager.isOK(self.pem.setDoGOBlockage(value))
        self._go_blockage = value

    @property
    def gpu_quota(self):
        """
        Get the current GPU memory quota.
        
        Returns:
            float: GPU memory quota (0.0 to 1.0)
        """
        return self._gpu_quota
    @gpu_quota.setter
    def gpu_quota(self, value):
        """
        Set the GPU memory quota.
        
        Args:
            value (float): GPU memory quota (0.0 to 1.0)
            
        Raises:
            ValueError: If value is not between 0.0 and 1.0
        """
        # Validate input range
        if value < 0.0 or value > 1.0:
            raise ValueError("gpu_quota must be between 0.0 and 1.0")
        self._gpu_quota = value

    @property
    def gpu_device(self):
        """
        Get the current GPU device
        
        Returns:
            int or list: GPU memory quota [0] or [0,1,...]
        """
        return self._gpu_device
    
    @gpu_device.setter
    def gpu_device(self, value):
        """
        Set the GPU device(s) to use for simulation.
        
        Args:
            value (int or list): GPU device ID or list of GPU device IDs
            
        Raises:
            ValueError: If value is not an integer or list of integers
        """
        # Display available GPUs
        print(self.pem.listGPUs())
        
        # Validate input type
        if not isinstance(value, int) and not isinstance(value, list):
            raise ValueError("gpu_device must be an integer or a list of integers")
            
        # Prepare device lists and quotas
        dev_quotas = []
        dev_ids = []
        
        if isinstance(value, list):
            dev_ids = value  # Fixed: was using undefined variable
            for id in dev_ids:
                dev_quotas.append(self._gpu_quota)
        else:
            dev_ids = [value]
            dev_quotas = [self._gpu_quota]

        # Configure GPU devices
        self.pem_api_manager.isOK(self.pem.setGPUDevices(dev_ids, dev_quotas))
        self.has_gpu_been_set = True
        self._gpu_device = value

    def set_gpu_device(self):
        """
        Set default GPU device configuration.
        
        If GPU device hasn't been explicitly set, this method configures
        GPU device 0 with 95% quota as the default.
        """
        # Default to GPU device 0 with quota
        self.pem_api_manager.isOK(self.pem.setGPUDevices([0], [self._gpu_quota]))
        self.has_gpu_been_set = True

    # Field of view properties
    @property
    def field_of_view(self):
        """
        Get the field of view setting in degrees.
        
        Returns:
            int: Field of view (180 or 360 degrees)
        """
        return self._field_of_view

    @field_of_view.setter
    def field_of_view(self, value):
        """
        Set the field of view for antenna patterns.
        
        Args:
            value (int): Field of view in degrees (must be 180 or 360)
            
        Raises:
            ValueError: If value is not 180 or 360
            
        Note:
            For 360-degree FOV, version-dependent behavior applies.
            In versions 252+, use setAntennaFieldOfView instead.
        """
        # Validate field of view value
        if value != 180 and value != 360:
            raise ValueError("field_of_view must be 180 or 360")
            
        # Handle 360-degree field of view with version compatibility
        if value == 360:
            if int(self.pem_api_manager.version) < 252:
                self.pem_api_manager.isOK(self.pem.setPrivateKey("FieldOfView", "360"))
            else:
                print("WARNING: FOV OPTION HAS MOVED")
                print("Field of view private key is deprecated in 2025 R2 and later. Use the setAntennaFieldOfView instead.")
        self._field_of_view = value

    # Bounding box properties
    @property
    def bounding_box(self):
        """
        Get the maximum bounding box side length.
        
        Returns:
            float or None: Maximum bounding box side length, or None if not set
        """
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, value):
        """
        Set the maximum bounding box side length.
        
        Args:
            value (float or None): Maximum bounding box side length in meters,
                                 or None to disable
        """
        if value is not None:
            self.pem_api_manager.isOK(self.pem.setPrivateKey("MaxBBoxSideLength", str(value)))
        self._bounding_box = value

    def auto_configure_simulation(self):
        """
        Automatically configure the simulation with current settings.
        
        This method ensures GPU devices are set (using defaults if not already
        configured) and then calls the API's auto-configuration routine with
        the specified maximum number of batches.
        """
        if self._ray_density is not None and self._ray_shoot_method == "sbr" and int(self.pem_api_manager.version) <= 252:
            print("Using SBR Style RayShoot with ray density:", self.ray_density)
            self.pem_api_manager.isOK(self.pem.setPrivateKey("RayShootGrid", "SBR," + str(self._ray_density)))
        else:
            if self._ray_shoot_method == "sbr" and int(self.pem_api_manager.version) > 252:
                ray_shoot_mode = self.RssPy.RayShootMode.SCENE_TESSELLATION
                # ray_shoot_mode = RssPy.RayShootMode.BOUNDING_BOX
                self.pem.setRayShootMode(ray_shoot_mode)
            self.pem_api_manager.isOK(self.pem.setRaySpacing(self._ray_spacing ))

        # Ensure GPU is configured
        if not self.has_gpu_been_set:
            self.set_gpu_device()

        print(f"Using GPU device(s): {self._gpu_device} with quota(s): {self._gpu_quota}")
        # Auto-configure simulation with current batch settings
        self.pem_api_manager.isOK(self.pem.autoConfigureSimulation(self.max_batches))

