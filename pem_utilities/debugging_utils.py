"""
Debugging Utilities for Radar Camera and Scene Analysis

This module provides debugging tools for radar simulation analysis, including:
- Camera visualization and image generation
- GIF creation for time-series analysis
- Scene logging and serialization
- Response data visualization

Classes:
    DebuggingCamera: Handles radar camera visualization and image generation
    DebuggingLogs: Manages scene logging and serialization utilities

Dependencies:
    - numpy: Numerical operations
    - PIL.Image: Image processing
    - matplotlib.pyplot: Plotting and visualization
    - tqdm: Progress bars
    - api_core: Core API functions for radar simulation
"""

import numpy as np
import PIL.Image
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 


class DebuggingCamera:
    """
    A debugging camera utility for radar simulation visualization.
    
    This class provides methods to capture, visualize, and export radar camera data
    as images and animated GIFs. It supports various display modes and can combine
    camera views with radar response data.
    
    Attributes:
        output_directory (str): Directory path for saving output files
        hMode: Handle mode for the radar camera
        frame_rate (int): Frame rate for GIF generation
        camera_images (list): List of captured camera images
        current_image (PIL.Image): Most recently captured image
    """
    
    def __init__(self, hMode,
                 display_mode='coating',
                 output_size=(512,512),
                 background_color=255,
                 frame_rate=10,
                 output_directory=None):
        """
        Initialize the debugging camera with specified parameters.
        
        Args:
            hMode: Handle mode for the radar camera
            display_mode (str, optional): Camera display mode. Options are:
                - 'blackwhite': Black and white display
                - 'normal': Normal color display
                - 'velocity': Velocity-based coloring
                - 'coating' (default): Coating-based coloring
            output_size (tuple, optional): Output image size as (width, height). 
                Defaults to (512, 512).
            background_color (int, optional): Background color value (0-255). 
                Defaults to 255 (white).
            frame_rate (int, optional): Frame rate for GIF generation. 
                Defaults to 10 fps.
            output_directory (str, optional): Directory for saving output files. 
                Defaults to '../output/'.
        """
        # Set up output directory
        if output_directory is None:
            self.output_directory = paths.output
        else:
            self.output_directory = output_directory
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)

        # Initialize camera parameters
        self.hMode = hMode
        self.frame_rate = frame_rate

        # Initialize image storage
        self.camera_images = []
        self.current_image = None
        self.raw_image = None

        # Convert string display mode to API enum
        if display_mode.lower() == 'blackwhite':
            display_mode = RssPy.CameraColorMode.BLACKWHITE
        elif display_mode.lower() == 'normal':
            display_mode = RssPy.CameraColorMode.NORMAL
        elif display_mode.lower() == 'velocity':
            display_mode = RssPy.CameraColorMode.VELOCITY
        else:
            display_mode = RssPy.CameraColorMode.COATING

        # Initialize and activate the radar camera
        pem_api_manager.isOK(pem.initRadarCamera(RssPy.CameraProjection.FISHEYE,
                                          display_mode,
                                          output_size[0],
                                          output_size[1],
                                          background_color,
                                          True,
                                          1))
        pem_api_manager.isOK(pem.activateRadarCamera())

    def activate(self):
        """
        activate the radar camera.
        
        """
        pem_api_manager.isOK(pem.activateRadarCamera())

    def deactivate(self):
        """
        Deactivate the radar camera.
        
        This method should be called when finished using the camera to properly
        clean up resources.
        """
        pem_api_manager.isOK(pem.deactivateRadarCamera())

    def generate_image(self,store_images=True,up_direction='+Z'):
        """
        Generate and capture a new camera image.
        
        Retrieves the current radar camera image, converts it to PIL format,
        and stores it in both current_image and the camera_images list for
        later use in GIF generation.
        """
        # Retrieve camera image data from API
        (ret, self.raw_image, _, _) = pem.retrieveRadarCameraImage(self.hMode)

        up_direction = up_direction.lower()

        if up_direction == '+y' or up_direction == 'y':
            self.raw_image = np.rot90(self.raw_image, k=3,axes=(0,1))
        elif up_direction == '-y':
            self.raw_image = np.rot90(self.raw_image, k=1,axes=(0,1))
        elif up_direction == '-z':
            self.raw_image = np.rot90(self.raw_image, k=2)

        # Convert to PIL image and store
        self.current_image = self._create_image_from_data(self.raw_image)
        if store_images:
            self.camera_images.append(self.current_image)

    def write_image_to_file(self, file_name):
        """
        Save the current camera image to a file.
        
        Args:
            file_name (str): Name of the output file (including extension, excluding path)
        """
        full_path = os.path.join(self.output_directory, file_name)
        self.current_image.save(full_path)

    def show(self):
        """
        Display the current camera image using the default image viewer.
        """
        self.current_image.show()

    def write_camera_to_gif(self, file_name='camera.gif'):
        """
        Create an animated GIF from all captured camera images.
        
        Args:
            file_name (str, optional): Name of the output GIF file. 
                Defaults to 'camera.gif'.
        """
        full_path = os.path.join(self.output_directory, file_name)

        # Set up matplotlib figure (used for progress tracking)
        (fig, axes) = plt.subplots()

        images = []
        print('generating debug camera gif...')
        # Process each captured frame
        for iFrame, camera_im in tqdm(enumerate(self.camera_images)):
            fig.canvas.draw()  # Update canvas for progress tracking
            images.append(camera_im)
        
        print(f'writing {file_name}...')
        # Save as animated GIF
        images[0].save(full_path, save_all=True, append_images=images[1:],
                       optimize=False,
                       duration=int(1000 / self.frame_rate),
                       loop=0)

    def write_response_to_gif(self, responses,
                                 file_name='response.gif',
                                 clim_db=(-240,-120),
                                 rng_domain=None,
                                 vel_domain=None,
                                 tx_index=0):
        """
        Create an animated GIF from radar response data.
        
        Visualizes radar response data as a range-Doppler plot over time,
        displaying the magnitude in dB scale with velocity and range axes.
        
        Args:
            responses (list): List of radar response data arrays for each frame
            file_name (str, optional): Name of the output GIF file. 
                Defaults to 'response.gif'.
            clim_db (tuple, optional): Color scale limits in dB as (min, max). 
                Defaults to (-240, -120).
            rng_domain (array-like, optional): Range axis values. If None, 
                creates normalized range from 0 to 1.
            vel_domain (array-like, optional): Velocity axis values. If None, 
                creates normalized velocity from -1 to 1.
            tx_index (int, optional): Transmitter index to visualize. 
                Defaults to 0.
        """
        full_path = os.path.join(self.output_directory, file_name)

        # Extract dimensions from response data
        num_frames = len(responses)
        num_ranges = responses[0][0].shape[-1]
        num_velocities = responses[0][0].shape[-2]

        # Set up matplotlib plot
        (fig,axes) = plt.subplots()
        
        # Create default domains if not provided
        if rng_domain is None:
            rng_domain = np.linspace(0,1,num_ranges)
        if vel_domain is None:
            vel_domain = np.linspace(-1,1,num_velocities)
        
        # Initialize image data
        imData = np.zeros((len(rng_domain),len(vel_domain)))
        image = plt.imshow(imData,
                           interpolation='bilinear',
                           cmap='jet',
                           aspect='auto',
                           vmin=clim_db[0],
                           vmax=clim_db[1],
                           extent=[vel_domain[0],vel_domain[-1],rng_domain[0],rng_domain[-1]])
        axes.set_xlabel('Doppler velocity (m/s)')
        axes.set_ylabel('Range (m)')
        
        images = []
        print('generating gif...')
        # Process each frame of response data
        for iFrame,response in tqdm(enumerate(responses)):
            # Convert to dB scale and rotate for proper orientation
            imData = np.rot90(20*np.log10(np.fmax(np.abs(response[tx_index][0]),1.e-30)))
            image.set_data(imData)
            fig.canvas.draw()
            # Convert matplotlib figure to PIL image
            radarIm = PIL.Image.frombytes('RGB',fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
            images.append(radarIm)

        print(f'writing {full_path}...')
        # Save as animated GIF
        images[0].save(full_path,save_all=True
                       ,append_images=images[1:],
                       optimize=False,
                       duration=int(1000/self.frame_rate),
                       loop=0)

    def write_camera_and_response_to_gif(self, responses,
                                         file_name='camera_and_response.gif',
                                         clim_db=(None,None),
                                         rng_domain=None,
                                         vel_domain=None,
                                         tx_index=0):
        """
        Create a side-by-side animated GIF combining camera and response data.
        
        Generates a combined visualization showing both the camera view and
        radar response data side by side for each frame.
        
        Args:
            responses (list): List of radar response data arrays for each frame
            file_name (str, optional): Name of the output GIF file. 
                Defaults to 'camera_and_response.gif'.
            clim_db (tuple, optional): Color scale limits in dB as (min, max).
                If (None, None), auto-scales based on first frame data.
            rng_domain (array-like, optional): Range axis values. If None, 
                creates normalized range from 0 to 1.
            vel_domain (array-like, optional): Velocity axis values. If None, 
                creates normalized velocity from -1 to 1.
            tx_index (int, optional): Transmitter index to visualize. 
                Defaults to 0.
        """
        # Auto-scale color limits if not provided
        if clim_db[0] is None or clim_db[1] is None:
            data = np.rot90(20 * np.log10(np.fmax(np.abs(responses[0][tx_index][0]), 1.e-30)))
            clim_db = (np.min(data),np.max(data))

        full_path = os.path.join(self.output_directory, file_name)

        # Extract dimensions from response data
        num_frames = len(responses)
        num_ranges = responses[0][0].shape[-1]
        num_velocities = responses[0][0].shape[-2]

        # Set up matplotlib plot
        (fig,axes) = plt.subplots()
        
        # Create default domains if not provided
        if rng_domain is None:
            rng_domain = np.linspace(0,1,num_ranges)
        if vel_domain is None:
            vel_domain = np.linspace(-1,1,num_velocities)
        
        # Initialize image data
        imData = np.zeros((len(rng_domain),len(vel_domain)))
        image = plt.imshow(imData,
                           interpolation='bilinear',
                           cmap='jet',
                           aspect='auto',
                           vmin=clim_db[0],
                           vmax=clim_db[1],
                           extent=[vel_domain[0],vel_domain[-1],rng_domain[0],rng_domain[-1]])
        axes.set_xlabel('Doppler velocity (m/s)')
        axes.set_ylabel('Range (m)')
        
        images = []
        print('generating gif...')
        # Process each frame
        for iFrame, response in tqdm(enumerate(responses)):
            # Convert response to dB scale and rotate for proper orientation
            imData = np.rot90(20 * np.log10(np.fmax(np.abs(response[tx_index][0]), 1.e-30)))
            image.set_data(imData)
            fig.canvas.draw()

            # Fix the color ordering from ARGB to RGBA
            img_data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_data = img_data.reshape((fig.canvas.get_width_height()[::-1] + (4,)))
            # Reorder from ARGB to RGBA
            img_data = np.roll(img_data, 3, axis=2)

            # Convert to PIL image
            radarIm = PIL.Image.fromarray(img_data, mode='RGBA')

            # Get corresponding camera image and resize radar image to match height
            cameraIm = self.camera_images[iFrame]
            radarIm = radarIm.resize((int(cameraIm.height * radarIm.width / radarIm.height), cameraIm.height))
            
            # Create combined image with camera on left, radar on right
            finalIm = PIL.Image.new('RGB', (radarIm.width + cameraIm.width, radarIm.height))
            finalIm.paste(cameraIm, (0, 0))
            finalIm.paste(radarIm, (cameraIm.width, 0))
            images.append(finalIm)

        print(f'writing {full_path}...')
        # Save as animated GIF
        images[0].save(full_path,save_all=True
                       ,append_images=images[1:],
                       optimize=False,
                       duration=int(1000/self.frame_rate),
                       loop=0)

    def _create_image_from_data(self,radar_camera):
        """
        Convert raw radar camera data to a PIL Image.
        
        Args:
            radar_camera: Raw camera data array from the API
            
        Returns:
            PIL.Image: Converted image object
        """
        # Ensure the array is C-contiguous before converting to PIL Image
        if not radar_camera.flags['C_CONTIGUOUS']:
            radar_camera = np.ascontiguousarray(radar_camera)
        return PIL.Image.frombytes('RGB', radar_camera.shape[0:2], radar_camera)

class DebuggingLogs:
    """
    A utility class for managing debugging logs and scene serialization.
    
    This class provides methods to enable verbose logging, export scene summaries,
    and serialize/deserialize scene data for debugging and analysis purposes.
    
    Attributes:
        output_directory (str): Directory path for saving output files
    """
    
    def __init__(self, output_directory=None):
        """
        Initialize the debugging logs utility.
        
        Args:
            output_directory (str, optional): Directory for saving output files. 
                Defaults to '../output/'.
        """
        if output_directory is None:
            self.output_directory = paths.output
        else:
            self.output_directory = output_directory
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)

    def enable_verbose_logs(self):
        """
        Enable verbose logging in the API.
        
        This method activates detailed logging output which can be useful
        for debugging simulation issues and understanding API behavior.
        """
        pem.setVerbose(True)

    def write_scene_summary(self,file_name='scene_summary.json'):
        """
        Export a JSON summary of the current scene settings.
        
        Retrieves and saves a comprehensive summary of all current scene
        parameters and settings in human-readable JSON format.
        
        Args:
            file_name (str, optional): Name of the output JSON file. 
                Defaults to 'scene_summary.json'.
        """
        # Get scene settings from API
        out = pem.reportSettings()
        out = json.loads(out)
        
        # Format as pretty-printed JSON
        json_object = json.dumps(out, indent=4)
        filename = os.path.join(self.output_directory, file_name)
        
        # Write to file
        with open(filename, "w") as outfile:
            outfile.write(json_object)

    def write_scene_to_proto(self,file_name='scene.proto'):
        """
        Serialize the current scene to a protocol buffer file.
        
        Saves the complete scene state in a binary protocol buffer format
        that can be loaded later for exact reproduction of the simulation.
        
        Args:
            file_name (str, optional): Name of the output proto file. 
                Defaults to 'scene.proto'.
        """
        filename = os.path.join(self.output_directory, file_name)
        out = pem.serialize(filename)

    def read_scene_from_proto(self,file_name):
        """
        Load a scene from a protocol buffer file.
        
        Restores a previously saved scene state from a protocol buffer file,
        allowing exact reproduction of simulation conditions.
        
        Args:
            file_name (str): Path to the protocol buffer file to load
        """
        pem.deserialize(file_name)

