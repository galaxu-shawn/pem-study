import pyvista as pv
import numpy as np
import os
import glob
from tqdm import tqdm
from pem_utilities.open_street_maps_geometry import find_random_location
from pem_utilities.path_helper import get_repo_paths
from pem_utilities.primitives import Cylinder, Sphere, Cone


class ForestGenerator:
    """
    A class for generating and visualizing forests on terrain meshes.
    """
    
    def __init__(self, num_trees=100, terrain_file='terrain2.stl', 
                 scale_min=0.5, scale_max=1.1, output_path=None, use_procedural_trees=False):
        """
        Initialize the ForestGenerator.
        
        Parameters:
        -----------
        num_trees : int
            Number of trees to place in the scene
        terrain_file : str or pv.PolyData
            Name of the terrain file to load
        scale_min : float
            Minimum scaling factor for trees
        scale_max : float
            Maximum scaling factor for trees
        output_path : str
            Output path for saving files (if None, will use paths.cache)
        use_procedural_trees : bool
            Whether to use procedural tree generation instead of loading from files
        """
        self.num_trees = num_trees
        self.terrain_file = terrain_file
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.use_procedural_trees = use_procedural_trees
        self.paths = get_repo_paths()
        if output_path is None:
            self.output_path = self.paths.cache
        else:
            self.output_path = output_path
        
        # Initialize attributes
        self.terrain_mesh = None
        self.forest_mesh = None
        self.all_tree_heights = []
        self.tree_positions = []
        self._load_terrain()
        
    def _load_terrain(self):
        """Load the terrain mesh from file."""
        if isinstance(self.terrain_file, pv.PolyData):
            self.terrain_mesh = self.terrain_file
            return
        terrain_path = os.path.join(self.paths.models, self.terrain_file)
        if not os.path.exists(terrain_path):
            raise FileNotFoundError(f"Terrain file not found: {terrain_path}")
        self.terrain_mesh = pv.read(terrain_path)
    
    def create_procedural_tree(self, tree_type='deciduous', trunk_height=None, crown_radius=None):
        """
        Create a procedural tree using PyVista primitives.
        
        Parameters:
        -----------
        tree_type : str
            Type of tree to generate ('deciduous', 'coniferous', 'palm', 'oak')
        trunk_height : float
            Height of the trunk (if None, will be randomly generated)
        crown_radius : float
            Radius of the crown (if None, will be randomly generated)
            
        Returns:
        --------
        pv.PolyData
            Combined tree mesh
        """
        # Random parameters if not specified
        if trunk_height is None:
            trunk_height = np.random.uniform(2.0, 8.0)
        if crown_radius is None:
            crown_radius = np.random.uniform(1.0, 3.0)
            
        trunk_radius = np.random.uniform(0.1, 0.3)
        
        # Create trunk (cylinder)
        trunk = Cylinder(radius=trunk_radius, height=trunk_height, 
                        num_theta=8, orientation=[0, 0, 1])
        trunk_mesh = trunk.generate_mesh()
        
        # Position trunk so bottom is at z=0
        trunk_mesh.translate([0, 0, trunk_height/2], inplace=True)
        
        # Create crown based on tree type
        if tree_type == 'deciduous':
            # Spherical crown
            crown = Sphere(radius=crown_radius, num_theta=16, num_phi=16)
            crown_mesh = crown.generate_mesh()
            # Position crown on top of trunk
            crown_mesh.translate([0, 0, trunk_height + crown_radius*0.7], inplace=True)
            
        elif tree_type == 'coniferous':
            # Conical crown
            crown_height = crown_radius * 2.5
            crown = Cone(radius=crown_radius, height=crown_height, num_theta=12)
            crown_mesh = crown.generate_mesh()
            # Position crown on top of trunk
            crown_mesh.translate([0, 0, trunk_height + crown_height/2], inplace=True)
            
        elif tree_type == 'palm':
            # Thin trunk, small spherical crown at top
            trunk_mesh.scale([0.5, 0.5, 1.2], inplace=True)  # Make trunk thinner and taller
            small_crown_radius = crown_radius * 0.4
            crown = Sphere(radius=small_crown_radius, num_theta=12, num_phi=12)
            crown_mesh = crown.generate_mesh()
            crown_mesh.translate([0, 0, trunk_height * 1.2 + small_crown_radius], inplace=True)
            
        elif tree_type == 'oak':
            # Wide, irregular crown using multiple spheres
            crown_mesh = pv.PolyData()
            num_crown_parts = np.random.randint(3, 7)
            for i in range(num_crown_parts):
                part_radius = crown_radius * np.random.uniform(0.6, 1.0)
                crown_part = Sphere(radius=part_radius, num_theta=12, num_phi=12)
                part_mesh = crown_part.generate_mesh()
                
                # Random offset for irregular shape
                offset_x = np.random.uniform(-crown_radius*0.5, crown_radius*0.5)
                offset_y = np.random.uniform(-crown_radius*0.5, crown_radius*0.5)
                offset_z = np.random.uniform(-crown_radius*0.3, crown_radius*0.3)
                
                part_mesh.translate([offset_x, offset_y, 
                                   trunk_height + crown_radius*0.7 + offset_z], inplace=True)
                crown_mesh += part_mesh
        else:
            # Default to deciduous
            crown = Sphere(radius=crown_radius, num_theta=16, num_phi=16)
            crown_mesh = crown.generate_mesh()
            crown_mesh.translate([0, 0, trunk_height + crown_radius*0.7], inplace=True)
        
        # Combine trunk and crown
        tree_mesh = trunk_mesh + crown_mesh
        
        return tree_mesh
    
    def get_random_tree_type(self):
        """
        Get a random tree type from available options.
        
        Returns:
        --------
        str
            Random tree type
        """
        tree_types = ['deciduous', 'coniferous', 'palm', 'oak']
        return np.random.choice(tree_types)

    def generate_forest_in_batches(self, batch_size=100):
        total_num_batches = int(self.num_trees // batch_size)
        temp_mesh=pv.PolyData()
        all_filenames = []
        for batch in tqdm(range(total_num_batches), desc="Generating Forest Batches"):
            mesh = self.generate_forest(num_trees=batch_size,show_progress=False)
            filename =os.path.join(self.paths.cache,f'forest_batch{batch}.stl')
            all_filenames.append(filename)
            mesh.save(filename)

        for each in tqdm(all_filenames, desc="Reading Forest Batches"):
            temp_mesh += pv.read(each)

        self.forest_mesh = temp_mesh

    def generate_forest(self, num_trees=None, show_progress=True):
        """
        Generate the forest by placing trees randomly on the terrain.
        
        Parameters:
        -----------
        num_trees : int
            Number of trees to generate (if None, uses self.num_trees)
        show_progress : bool
            Whether to show progress bar during generation
            
        Returns:
        --------
        pv.PolyData
            The combined forest mesh
        """
        if self.terrain_mesh is None:
            self._load_terrain()
            
        if num_trees is None:
            num_trees = self.num_trees
            
        # Initialize or get tree sources
        all_trees = []
        if not self.use_procedural_trees:
            # Get all available tree files
            tree_directory = os.path.join(self.paths.models, 'vegetation', '*.stl')
            all_trees = glob.glob(tree_directory)
            
            if not all_trees:
                print(f"Warning: No tree files found in {tree_directory}")
                print("Switching to procedural tree generation...")
                self.use_procedural_trees = True
        
        # Initialize forest mesh and tracking lists
        self.forest_mesh = pv.PolyData()
        self.all_tree_heights = []
        self.tree_positions = []

        # Generate trees with progress bar
        iterator = tqdm(range(num_trees), desc="Generating trees") if show_progress else range(num_trees)
        
        for i in iterator:
            if self.use_procedural_trees:
                # Generate procedural tree
                tree_type = self.get_random_tree_type()
                tree_mesh = self.create_procedural_tree(tree_type=tree_type)
            else:
                # Load tree from file
                tree_filename = np.random.choice(all_trees)
                tree_mesh = pv.read(tree_filename)

            # Apply random scaling
            scale_val = np.random.uniform(low=self.scale_min, high=self.scale_max)
            tree_mesh.scale((scale_val, scale_val, scale_val), inplace=True)
            
            # Apply random rotation around Z-axis
            tree_mesh.rotate_z(np.random.uniform(0, 360), inplace=True)
            
            # Record tree height (after scaling)
            tree_height = tree_mesh.bounds[5] - tree_mesh.bounds[4]
            self.all_tree_heights.append(tree_height)

            # Find random location on terrain
            x, y, z = find_random_location(
                self.terrain_mesh, 
                z_elevation_mesh=None, 
                outdoors=True, 
                return_highest_z=True, 
                max_tries=100
            )
            
            # Position the tree (move bottom of tree to terrain surface)
            tree_mesh.translate([x, y, z - tree_mesh.bounds[4]], inplace=True)
            self.tree_positions.append([x, y, z])
            
            # Add to forest mesh
            self.forest_mesh += tree_mesh
            
        if show_progress:
            print(f"Generated {num_trees} trees using {'procedural' if self.use_procedural_trees else 'file-based'} method")
            
        return self.forest_mesh
    
    def get_statistics(self):
        """
        Get statistics about the generated forest.
        
        Returns:
        --------
        dict
            Dictionary containing forest statistics
        """
        if not self.all_tree_heights:
            return {"message": "No forest generated yet. Call generate_forest() first."}
            
        return {
            "num_trees": len(self.all_tree_heights),
            "average_height": np.mean(self.all_tree_heights),
            "min_height": np.min(self.all_tree_heights),
            "max_height": np.max(self.all_tree_heights),
            "height_std": np.std(self.all_tree_heights)
        }
    
    def visualize(self, show_terrain=True, terrain_color='tan', forest_color='green', 
                  show_grid=True, show_edges=False):
        """
        Visualize the generated forest and terrain.
        
        Parameters:
        -----------
        show_terrain : bool
            Whether to display the terrain mesh
        terrain_color : str
            Color for the terrain mesh
        forest_color : str
            Color for the forest mesh
        show_grid : bool
            Whether to show the grid
        show_edges : bool
            Whether to show mesh edges
        """
        plotter = pv.Plotter()
        
        # Add terrain if requested and available
        if show_terrain and self.terrain_mesh is not None:
            plotter.add_mesh(
                self.terrain_mesh, 
                color=terrain_color, 
                show_edges=show_edges, 
                name='terrain'
            )
        
        # Add forest if available
        if self.forest_mesh is not None:
            plotter.add_mesh(
                self.forest_mesh, 
                color=forest_color, 
                show_edges=show_edges, 
                name='forest'
            )
        
        if show_grid:
            plotter.show_grid()
            
        plotter.show()
        
        return plotter
    
    def save_forest(self, filename, output_format='stl'):
        """
        Save the generated forest mesh to file.
        
        Parameters:
        -----------
        filename : str
            Name of the output file (without extension)
        output_format : str
            File format ('stl', 'ply', 'obj', etc.)
        """
        if self.forest_mesh is None:
            raise ValueError("No forest generated yet. Call generate_forest() first.")

        # if filename is only a file name and not a full path, use the output path appended to it
        if not os.path.splitext(filename)[1]:
            filename = os.path.join(self.output_path, filename)
        else:
            filename = os.path.join(self.output_path, os.path.basename(filename))    

        self.forest_mesh.save(filename)
        print(f"Forest saved to: {filename}")
        return filename


# Example usage (can be removed if not needed)
if __name__ == "__main__":
    # Create forest generator
    num_trees = 100000
    forest_gen = ForestGenerator(terrain_file='terrain3.stl',
                                 num_trees=num_trees,use_procedural_trees=False)
    
    # Generate the forest
    # timeit to see how long it takes
    import time
    # start_time = time.time()
    # print("Generating forest...")
    # forest_mesh = forest_gen.generate_forest()
    # end_time = time.time()
    # print(f"Forest generated in {end_time - start_time:.2f} seconds")

    # do the same with batches
    start_time = time.time()
    print("Generating forest in batches...")
    forest_gen.generate_forest_in_batches(batch_size=100)
    end_time = time.time()
    print(f"Forest generated in batches in {end_time - start_time:.2f} seconds")

    
    # Print statistics
    stats = forest_gen.get_statistics()
    print(f"Forest Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    

    forest_gen.save_forest(f"forest_{num_trees}_output.stl")

    # Visualize the result
    forest_gen.visualize()