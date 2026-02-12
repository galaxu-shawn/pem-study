"""
Mesh Loading Utilities for Perceive EM

This module provides the MeshLoader class for loading, processing, and managing 3D mesh data
for use with the Perceive EM simulation engine. It supports various mesh formats including
STL, OBJ, VTP, PLY, and FACET files, with capabilities for texture mapping, material
assignment, mesh clipping, and scaling.

Key features:
- Load and convert between multiple 3D mesh formats
- Apply material properties based on texture colors or predefined mappings
- Clip meshes to specified spatial bounds
- Scale and transform meshes
- Support for curved surface physics
- Memory management for mesh updates

Created on Fri Jan 26 15:00:00 2024

@author: asligar
"""
import os.path
import numpy as np
import pyvista as pv

from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API

import vtk
from tqdm import tqdm
from pathlib import Path
# Dynamically resolve the path to the api_core module


class MeshLoader:
    """
    A class for loading and processing 3D meshes for Perceive EM simulations.
    
    This class provides comprehensive mesh loading capabilities including format conversion,
    material assignment, texture mapping, geometric transformations, and integration with
    the Perceive EM API.
    
    Attributes:
        RssPy: Reference to the Perceive EM RssPy module
        api: Reference to the Perceive EM API
        mesh_memory_mat_idx (int): Stored material index for mesh updates
        mesh_memory_clip_at_bounds (list): Stored clipping bounds for mesh updates
        mesh_memory_center_geometry (bool): Stored centering flag for mesh updates
        h_mesh: Handle to the current mesh scene element
        material_manager: Material manager instance for handling material properties
    """
    
    def __init__(self, material_manager=None):
        """
        Initialize the MeshLoader with API connections and memory settings.
        
        Args:
            material_manager (optional): Material manager instance for handling 
                                       material properties and assignments
        """

        self.pem_api_manager = Perceive_EM_API()
        self.pem = self.pem_api_manager.pem  # The configured API object
        self.RssPy = self.pem_api_manager.RssPy


        # Used to preserve previous settings when update_mesh is used. Not all features can be used right now,
        # for example, scaling and textures will not be preserved on a generator mesh update. That is because
        # we are assuming what is coming from the generator is the mesh and it does not need to be changed
        self.mesh_memory_mat_idx = 0
        self.mesh_memory_clip_at_bounds = None
        self.mesh_memory_center_geometry = False
        self.h_mesh = None
        self.material_manager = material_manager

    def clip_mesh_at_bounds(self, mesh, bounds=None, center_geometry=False):
        """
        Clip a mesh to fit within specified spatial bounds.
        
        This method clips the input mesh to fit within the specified rectangular bounds
        and optionally centers the resulting geometry at the origin.
        
        Args:
            mesh (pv.PolyData): PyVista mesh to be clipped
            bounds (list, optional): Bounding box as [min_x, max_x, min_y, max_y].
                                   If None, no clipping is performed.
            center_geometry (bool, optional): If True, centers the clipped mesh at origin.
                                            Defaults to False.
        
        Returns:
            pv.PolyData: The clipped (and optionally centered) mesh
        """
        if bounds is not None:
            mesh_current_bounds = mesh.bounds

            # Extract bounding box coordinates
            minus_x = bounds[0]
            plus_x = bounds[1]
            minus_y = bounds[2]
            plus_y = bounds[3]

            # Calculate center coordinates if centering is requested
            if center_geometry:
                center_x = (minus_x + plus_x) / 2
                center_y = (minus_y + plus_y) / 2

            # Check if mesh is already within bounds to avoid unnecessary clipping
            if mesh_current_bounds[0] > minus_x and mesh_current_bounds[1] < plus_x and mesh_current_bounds[2] > minus_y and mesh_current_bounds[3] < plus_y:
                # Mesh is already within bounds, don't do any clipping
                if center_geometry:
                    mesh = mesh.translate([-center_x, -center_y, 0])
                return mesh
            
            # Perform clipping operations in sequence for each boundary
            _, keep = mesh.clip(return_clipped=True, normal='x', origin=(minus_x, 0, 0.0))
            keep, _ = keep.clip(return_clipped=True, normal='x', origin=(plus_x, 0, 0.0))
            _, keep = keep.clip(return_clipped=True, normal='y', origin=(0, minus_y, 0.0))
            keep, _ = keep.clip(return_clipped=True, normal='y', origin=(0, plus_y, 0.0))
            
            # There might be a defect with pyvista clip, the bounds do not update correctly. If I clip again
            # everything gets updated correctly.
            mesh, _ = keep.clip(return_clipped=True, normal='y', origin=(0, plus_y, 0.0))
            
            # Apply centering transformation if requested
            if center_geometry:
                mesh = mesh.translate([-center_x, -center_y, 0])
            return mesh

    def update_mesh(self, h_node=None, mesh=None, delete_old_scene_element=True):
        """
        Update an existing mesh node with new mesh data.
        
        This method replaces the mesh associated with a scene node while preserving
        previously stored settings like material index and clipping bounds.
        
        Args:
            h_node: Handle to the scene node to update
            mesh (pv.PolyData, optional): New mesh data to assign to the node
            delete_old_scene_element (bool, optional): Whether to delete the old scene
                                                     element before creating new one.
                                                     Defaults to True.
        """
        # Clean up old scene element if requested
        if delete_old_scene_element:
            if self.h_mesh is not None:
                try:
                    self.pem.deleteSceneElement(self.h_mesh)
                except:
                    print("INFO: Could not delete old scene element on update_mesh() for generator object")
        
        # Load new mesh with preserved settings
        (h_mesh, _) = self.loadMesh(mesh=mesh,
                                   mat_idx=self.mesh_memory_mat_idx,
                                   clip_at_bounds=self.mesh_memory_clip_at_bounds,
                                   center_geometry=self.mesh_memory_center_geometry)

        # This will update a node with a new scene element. I think using replace triangles will eventually be faster
        # but until 25R1 I will just use setSceneElement() which will replace the scene element on the node.
        # There might be some overhead if we don't delete previous scene_elements for large models with lots of updates
        self.pem_api_manager.isOK(self.pem.setSceneElement(h_node, h_mesh))

    def loadMesh(self, filename=None,
                 mesh=None,
                 mat_idx=0,
                 scale_mesh=None,
                 include_texture=False,
                 map_texture_to_material=False,
                 clip_at_bounds=None,
                 center_geometry=False,
                 add_mesh_offset=None,
                 use_curved_physics=False):
        """
        Load a 3D mesh from file or PyVista mesh object with various processing options.
        
        This is the main mesh loading method that handles multiple file formats, applies
        transformations, assigns materials, and integrates with the Perceive EM API.
        
        Args:
            filename (str, optional): Path to mesh file to load. Supports STL, OBJ, VTP, PLY, FACET formats.
            mesh (pv.PolyData, optional): Pre-loaded PyVista mesh object.
            mat_idx (int, optional): Material index to assign to mesh. Defaults to 0.
            scale_mesh (float, optional): Scale factor for mesh resizing. Defaults to 1.0.
            include_texture (bool, optional): Whether to load and apply textures. Defaults to False.
            map_texture_to_material (bool, optional): Whether to map texture colors to materials. Defaults to False.
            clip_at_bounds (list, optional): Bounding box for mesh clipping [min_x, max_x, min_y, max_y].
            center_geometry (bool, optional): Whether to center mesh at origin. Defaults to False.
            add_mesh_offset (list, optional): Translation offset to apply [x, y, z].
            use_curved_physics (bool, optional): Whether to enable curved surface physics. Defaults to False.
        
        Returns:
            tuple: (h_mesh, mesh) where h_mesh is the Perceive EM scene element handle
                   and mesh is the PyVista mesh object. Returns (None, None) if mesh is empty.
        
        Raises:
            FileNotFoundError: If specified filename does not exist.
            ValueError: If mesh scaling and texture are both requested (not supported).
        """

        paths = get_repo_paths()

        # Convert offset to numpy array if provided
        if add_mesh_offset is not None:
            add_mesh_offset = np.array(add_mesh_offset)

        # Scale mesh is used for pedestrian models, but can be used for any model

        # This mesh loads the file, but also returns the vertices and triangles id's that will make it easy to assign
        # material properties too. For visualization, we can either convert this to a pyvista mesh, or just reload the stl
        # directly as a pyvista mesh, I will do that because it is easier
        perceive_mesh = None
        h_mesh = None
        
        # Store material index for potential mesh updates
        self.mesh_memory_mat_idx = mat_idx
        
        # Handle file-based mesh loading
        if filename is not None:
            if os.path.exists(os.path.abspath(filename)):
                base_dir = os.path.dirname(filename)
                filename_only = os.path.basename(filename)
                ext = os.path.splitext(filename_only)[1]
                
                # Convert FACET files to STL format
                if ext.lower() == '.facet':
                    filename = self.facet_to_stl(filename)
                    filename_only = os.path.basename(filename)
                    ext = os.path.splitext(filename_only)[1]
                # Convert PLY files to OBJ format
                elif ext.lower() == '.ply':
                    filename = self.ply_to_obj(filename)
                    filename_only = os.path.basename(filename)
                    ext = os.path.splitext(filename_only)[1]
                
                # Load PyVista mesh which will be used for visualization
                mesh = pv.read(filename)
                
                # Apply mesh offset if specified
                if add_mesh_offset is not None:
                    mesh.translate(add_mesh_offset, inplace=True)
                
                # Apply clipping bounds if specified
                if clip_at_bounds is not None:
                    # Not sure if this will work for all file types, but it is a good start
                    mesh = self.clip_mesh_at_bounds(mesh, bounds=clip_at_bounds, center_geometry=center_geometry)

                # Check for incompatible options
                if scale_mesh != 1.0 and include_texture:
                    # Mesh scaling and texture is not supported
                    raise ValueError("Mesh scaling and texture is not supported")

                # Handle OBJ files with texture files (typically these are just for visualization)
                mtl_exists = os.path.exists(filename.replace('.obj', '.mtl'))
                if ext.lower() == '.obj' and mtl_exists and include_texture:
                    # Import OBJ with materials and textures using VTK
                    importer = vtk.vtkOBJImporter()
                    importer.SetFileName(filename)
                    importer.SetFileNameMTL(filename.replace('.obj', '.mtl'))
                    importer.SetTexturePath(base_dir)
                    importer.Update()

                    print('exporting temporary texture files (.vtk and .png) for visualization')
                    # Export to VTP format with textures
                    exporter = vtk.vtkSingleVTPExporter()
                    exporter.SetRenderWindow(importer.GetRenderWindow())
                    exporter.SetFilePrefix(os.path.join(base_dir, 'temp'))
                    exporter.Write()

                    # This will write 2 files (vtp and png), let's call them vtp_path and tex_path
                    mesh = pv.read(os.path.join(base_dir, 'temp.vtp'))
                    if add_mesh_offset is not None:
                        mesh.translate(add_mesh_offset, inplace=True)
                    tex = pv.read_texture(os.path.join(base_dir, 'temp.png'))

                    # ToDo - Fix this, it is not working in pyvista 0.46.0
                    # this appears to be broken in pyvista 0.46.0, so we will not use it, this is for visaualization of the mesh
                    mesh.textures = tex
                    
                    # Map texture colors to material indices if requested
                    if map_texture_to_material:
                        mat_idx = self.map_material_based_on_color(mesh, tex)
                    
                    # Apply clipping after texture processing
                    if clip_at_bounds is not None:
                        mesh = self.clip_mesh_at_bounds(mesh, bounds=clip_at_bounds, center_geometry=center_geometry)

                # Handle VTP files with potential texture support
                elif ext.lower() == '.vtp':
                    if include_texture:
                        # Check if texture image exists, assume same name as VTP file
                        texture_file = filename_only.replace('.vtp', '.jpg')
                        texture_full_path = os.path.join(base_dir, texture_file)
                        if os.path.exists(texture_full_path):
                            texture_exists = True
                        elif os.path.exists(texture_full_path.replace('.jpg', '.png')):
                            texture_full_path = texture_full_path.replace('.jpg', '.png')
                            texture_exists = True
                        else:
                            texture_exists = False
                        
                        # Load texture if found
                        if texture_exists:
                            tex = pv.read_texture(texture_full_path)
                            # ToDo - Fix this, it is not working in pyvista 0.46.0
                            # this appears to be broken in pyvista 0.46.0, so we will not use it, this is for visaualization of the mesh
                            mesh.textures = tex 
                        
                        # Apply material mapping for SUMS data format
                        if map_texture_to_material:
                            mesh = self.map_material_based_on_color_SUMS(mesh)
                        
                        # Apply clipping bounds
                        if clip_at_bounds is not None:
                            mesh = self.clip_mesh_at_bounds(mesh, bounds=clip_at_bounds,
                                                            center_geometry=center_geometry)

                # Handle mesh scaling operations
                # TODO: Scale mesh is done after clip_bounds is completed, should this be before?
                if scale_mesh is None:
                    scale_mesh = 1.0
                if scale_mesh != 1.0:
                    # Temp directory to save scaled mesh because it is needed for Perceive EM engine as well as viz
                    # TODO: Clean up temp files

                    mesh.scale([scale_mesh, scale_mesh, scale_mesh], inplace=True)
                    if add_mesh_offset is not None:
                        mesh.translate(add_mesh_offset, inplace=True)
                    
                    # Get the filename defined in the path
                    only_filename = os.path.basename(filename)
                    name, ext = os.path.splitext(only_filename)
                    new_filename = os.path.join(paths.cache, name + '_scaled.stl')  # PyVista can't save OBJ
                    mesh.save(new_filename)
                    filename = new_filename

                # Handle clipped mesh file saving
                if clip_at_bounds is not None:
                    # Temp directory to save clipped mesh because it is needed for Perceive EM engine as well as viz
                    # TODO: Clean up temp files

                    only_filename = os.path.basename(filename)
                    name, ext = os.path.splitext(only_filename)
                    new_filename = os.path.join(paths.cache, name + '.stl')  # PyVista can't save OBJ
                    if add_mesh_offset is not None:
                        mesh.translate(add_mesh_offset, inplace=True)
                    mesh.save(new_filename)
                    filename = new_filename

                # Special handling for VTP files with embedded materials
                if ext.lower() == '.vtp':
                    # If the file was VTP it didn't load correctly, we will manually populate it now, this could also
                    # be done for other file types but for those we are currently just converting gltf to obj then importing
                    # because VTP may have the materials baked into them, I will use these directly
                    mesh_faces = np.array(mesh.faces)
                    mesh_faces = np.delete(mesh_faces, np.arange(0, mesh_faces.size, 4))
                    mesh_faces = np.array(mesh_faces.reshape(-1, 3), dtype=np.int32)  # Must be type int32 for Perceive EM
                    triangles = mesh_faces
                    vertices = np.array(mesh.points, dtype=np.float32)
                    
                    # Use embedded material data if available
                    if 'material' in mesh.cell_data.keys():
                        print('Using materials from mesh.cell_data')
                        mat_idx = np.array(mesh.cell_data['material'], dtype=np.int32)
                else:
                    # Helper function to load STL or OBJ file, but ultimately we just need triangles
                    print(filename)
                    perceive_mesh = self.pem.loadTriangleMesh(filename)
                    triangles = perceive_mesh.triangles
                    vertices = perceive_mesh.vertices
                    perceive_mesh.coatings += mat_idx  # Coating by default will be 0 with this approach, increment to mat_idx
                
                # Check for empty mesh
                if vertices.shape[0] == 0:
                    print(f"INFO: Mesh is empty: {filename}")
                    return (None, None)

                # Create scene element and configure triangles
                h_mesh = self.RssPy.SceneElement()
                self.pem_api_manager.isOK(self.pem.addSceneElement(h_mesh))

                # Set up physics and geometry
                if not use_curved_physics or perceive_mesh is None:
                    self.pem_api_manager.isOK(self.pem.setTriangles(h_mesh, vertices, triangles, mat_idx))
                    if use_curved_physics == True:
                        print('Using Curved Physics Failed, check file type and support for surface normals (currently only obj and stl supported)')
                else:
                    print('Using curved physics')
                    # ToDo, make sure this is working correctly
                    self.pem_api_manager.isOK(self.pem.setTriangles(h_mesh, perceive_mesh))
                    self.pem_api_manager.isOK(self.pem.setDoCurvedSurfPhysics(True))
                    self.pem_api_manager.isOK(self.pem.setVertexNormals(h_mesh, self.RssPy.VertexNormalFormat.BY_FACE_LIST, perceive_mesh.vNormals))
                    
                    if len(perceive_mesh.vNormals) == 0:
                        # If no normals are found, calculate them, doesn't always work
                        print('No normals found in mesh, calculating normals - exeperimental')
                        # Calculate normals if not present
                        mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
                        normals = mesh.active_normals
                        self.pem_api_manager.isOK(self.pem.setVertexNormals(h_mesh, self.RssPy.VertexNormalFormat.BY_VERTEX_LIST, normals))
                        # self.pem_api_manager.isOK(self.pem.setVertexNormals(h_mesh, self.RssPy.VertexNormalFormat.BY_FACE_LIST, normals))
                    else:
                        print('Normals found in mesh, applying to geometry')

            else:
                raise FileNotFoundError(f"File not found: {filename}")
        
        # Handle direct mesh object input (no file loading)
        elif mesh is not None:
            # Apply transformations to the input mesh
            if add_mesh_offset is not None:
                mesh.translate(add_mesh_offset, inplace=True)
            if clip_at_bounds is not None:
                mesh = self.clip_mesh_at_bounds(mesh, bounds=clip_at_bounds, center_geometry=center_geometry)
            
            # Create scene element and configure geometry
            h_mesh = self.RssPy.SceneElement()
            self.pem_api_manager.isOK(self.pem.addSceneElement(h_mesh))
            
            # Extract faces and convert to proper format
            mesh_faces = np.array(mesh.faces)
            mesh_faces = np.delete(mesh_faces, np.arange(0, mesh_faces.size, 4))
            mesh_faces = np.array(mesh_faces.reshape(-1, 3), dtype=np.int32)  # Must be type int32 for Perceive EM
            self.pem_api_manager.isOK(self.pem.setTriangles(h_mesh, np.array(mesh.points, dtype=np.float32), mesh_faces, mat_idx))
        
        # Store settings for potential updates
        self.mesh_memory_clip_at_bounds = clip_at_bounds
        self.mesh_memory_center_geometry = center_geometry
        self.h_mesh = h_mesh

        return (h_mesh, mesh)

    def map_material_based_on_color_SUMS(self, mesh):
        """
        Map material indices based on predefined SUMS (aerial survey) material classifications.
        
        This method maps standard SUMS classification labels to appropriate material indices
        for electromagnetic simulation. It uses a predefined mapping dictionary to convert
        terrain classifications to material properties.
        
        Args:
            mesh (pv.PolyData): PyVista mesh with 'material' cell data containing SUMS classifications
        
        Returns:
            pv.PolyData: Modified mesh with updated material indices in cell_data['material']
        """
        # TODO: Using default material idx mappings. This should be updated to use the material names
        print('Mapping material based on default material_library.json. Please verify this is correct')

        # SUMS classification labels (swap the below dictionary key and values to get the material names)
        material_idx = {
            -1: 'unlabelled',
            0: 'unclassified', 
            1: 'terrain',
            2: 'high_vegetation',
            3: 'building',
            4: 'water',
            5: 'car',
            6: 'boat'
        }

        # Index values for PEC, earth, vegetation, water, concrete, aluminum, etc.
        material_map = {
            'unlabelled': 0,
            'unclassified': 0,
            'terrain': 4,
            'high_vegetation': 9,
            'building': 6,
            'water': 16,
            'car': 11,
            'boat': 0
        }

        # Direct mapping from SUMS indices to material library indices
        idx_mapping = {
            -1: 0,   # unlabelled -> default
            0: 0,    # unclassified -> default
            1: 4,    # terrain -> earth material
            2: 9,    # high_vegetation -> vegetation material
            3: 6,    # building -> concrete material
            4: 16,   # water -> water material
            5: 11,   # car -> metal material
            6: 0     # boat -> default
        }
        
        # Load the materials into the API
        for mat in idx_mapping.keys():
            self.material_manager.load_material(idx_mapping[mat])

        # Map mesh.cell_data['material'] to new material index using idx_mapping. If the material index is not in the
        # idx_mapping, then it is unlabelled and will be assigned to 0
        mesh.cell_data['material'] = np.array([idx_mapping.get(x, 0) for x in mesh.cell_data['material']], dtype=np.int32)
        return mesh

    def map_material_based_on_color(self, mesh, texture):
        """
        Map material indices based on texture color analysis.
        
        This method analyzes the texture colors at each mesh cell to determine appropriate
        material assignments. Currently implements a simple green/non-green classification
        but can be extended for more sophisticated color-based material mapping.
        
        Args:
            mesh (pv.PolyData): PyVista mesh with texture coordinates
            texture (pv.Texture): Texture object containing color information
        
        Returns:
            np.ndarray: Array of material indices for each mesh cell
        """
        texture = texture.to_array()
        # TODO: I am not sure if I have these backwards
        x_size = texture.shape[0]
        y_size = texture.shape[1]
        color_per_cell_idx = []

        print('Mapping material based on texture map... (slow)')
        for n in tqdm(range(mesh.n_cells)):
            cell = mesh.get_cell(n)
            pnt_ids = cell.point_ids
            colors_avg = []
            
            # Sample colors at each vertex of the cell
            for pnt_id in pnt_ids:
                uv_coord = mesh.active_texture_coordinates[pnt_id]
                closest_idx_x = int(np.abs(uv_coord[0]) * x_size)
                closest_idx_y = int(np.abs(uv_coord[1]) * y_size)
                colors_avg.append(np.array(texture[closest_idx_y, closest_idx_x][:3]))  # Only first three values
            
            # Calculate average color and convert to HSV
            rgb = self.average_rgb_colors(colors_avg)
            hsv = self.convert_rgb_to_hsb(rgb)
            
            # TODO: Assign meaningful index value. Right now just 1 for green, 0 for everything else
            if hsv[0] > 40 and hsv[0] < 160 and hsv[1] > 5 and hsv[1] < 250 and hsv[2] > 1 and hsv[2] < 250:  # Green
                color_per_cell_idx.append(1)
            else:
                color_per_cell_idx.append(0)
            colors_avg = []
        return np.asarray(color_per_cell_idx, dtype=np.int32)

    def average_rgb_colors(self, rgb_list):
        """
        Calculate the root mean square (RMS) average of RGB color values.
        
        This method computes a more perceptually accurate average of RGB colors
        using RMS averaging rather than simple arithmetic mean.
        
        Args:
            rgb_list (list): List of RGB color arrays
        
        Returns:
            list: RMS averaged RGB values as [r, g, b]
        """
        r = 0
        b = 0
        g = 0
        num = len(rgb_list)
        rgb_list = np.array(rgb_list, dtype=float)
        
        # Calculate RMS average for each color channel
        for rgb in rgb_list:
            r += rgb[0] * rgb[0]
            g += rgb[1] * rgb[1]
            b += rgb[2] * rgb[2]
        rgb_avg = [int(np.sqrt(r / num)), int(np.sqrt(g / num)), int(np.sqrt(b / num))]
        return rgb_avg

    def convert_rgb_to_hsb(self, rgb):
        """
        Convert RGB color values to HSB (Hue, Saturation, Brightness) color space.
        
        Args:
            rgb (list): RGB color values as [r, g, b] in range 0-255
        
        Returns:
            list: HSB color values as [h, s, b] where:
                  - h (hue) is in range 0-360 degrees
                  - s (saturation) is in range 0-100 percent
                  - b (brightness) is in range 0-100 percent
        """
        r, g, b = rgb
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        
        # Calculate hue
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        
        # Calculate saturation
        if mx == 0:
            s = 0
        else:
            s = (df / mx) * 100
        
        # Calculate brightness
        v = mx * 100
        return [h, s, v]

    def facet_to_stl(self, filename, outfile=None):
        """
        Convert a FACET file to STL format.
        
        This method reads a FACET file format and converts it to a binary STL file
        that can be used with standard 3D processing libraries.
        
        Args:
            filename (str): Path to the input FACET file
            outfile (str, optional): Path for the output STL file. If None, 
                                   uses the same directory with .stl extension.
        
        Returns:
            str: Path to the created STL file
        """
        base_dir = os.path.dirname(filename)
        filename_only = os.path.basename(filename)
        ext = os.path.splitext(filename_only)[1]

        if outfile is None:
            outfile = filename_only + '.stl'
            outfile = os.path.join(base_dir, outfile)

        # Read the FACET file
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Parse header information
        n_points = int(lines[4].strip())

        # Read vertex coordinates
        points = np.loadtxt(lines[5:5 + n_points])

        # Find the start of facets data
        facet_start = 5 + n_points + 3
        n_facets, n_sides = map(int, lines[facet_start - 1].strip().split())

        # Read face connectivity (convert to 0-based indexing)
        facets = np.loadtxt(lines[facet_start:], dtype=int)[:, :3] - 1  # Subtract 1 to convert to 0-based indexing

        # Write ASCII STL file
        with open(outfile, 'w') as f:
            f.write("solid \"design<stl unit=M>\"\n")

            for facet in facets:
                # Calculate normal vector (assuming counter-clockwise vertex order)
                v1 = points[facet[1]] - points[facet[0]]
                v2 = points[facet[2]] - points[facet[0]]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)

                # Write facet with normal and vertices
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("    outer loop\n")
                for vertex in facet:
                    point = points[vertex]
                    f.write(f"      vertex {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

            f.write("endsolid\n")

        # Convert ASCII STL to binary STL using PyVista
        mesh = pv.read(outfile)
        mesh.save(outfile, binary=True)
        print(f"STL file '{outfile}' has been created successfully.")
        return outfile

    def ply_to_obj(self, filename, outfile=None):
        """
        Convert a PLY file to OBJ format.
        
        Args:
            filename (str): Path to the input PLY file
            outfile (str, optional): Path for the output OBJ file. If None,
                                   uses the same directory with .obj extension.
        
        Returns:
            str: Path to the created OBJ file
        """
        base_dir = os.path.dirname(filename)
        filename_only = os.path.basename(filename)
        ext = os.path.splitext(filename_only)[1]
        filename_only = os.path.splitext(filename_only)[0]

        if outfile is None:
            outfile = filename_only + '.obj'
            outfile = os.path.join(base_dir, outfile)

        # Load and save using PyVista
        mesh = pv.read(filename)
        mesh.save(outfile, binary=True)
        print(f"OBJ file '{outfile}' has been created successfully.")
        return outfile

    def vtp_to_obj(self, filename, outfile=None):
        """
        Convert a VTP file to OBJ format.
        
        Args:
            filename (str): Path to the input VTP file
            outfile (str, optional): Path for the output OBJ file. If None,
                                   uses the same directory with .obj extension.
        
        Returns:
            str: Path to the created OBJ file
        """
        base_dir = os.path.dirname(filename)
        filename_only = os.path.basename(filename)
        ext = os.path.splitext(filename_only)[1]
        filename_only = os.path.splitext(filename_only)[0]

        if outfile is None:
            outfile = filename_only + '.obj'
            outfile = os.path.join(base_dir, outfile)

        # Load and save using PyVista
        mesh = pv.read(filename)
        mesh.save(outfile, binary=True)
        print(f"OBJ file '{outfile}' has been created successfully.")
        return outfile