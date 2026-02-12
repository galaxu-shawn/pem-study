import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


class ReportSettingsReader:
    """
    A class to read and parse Ansys Perceive EM output JSON files.
    Provides methods to access and recursively traverse all sections of the output.
    """
    
    def __init__(self, json_file_path: str):
        """
        Initialize the reader with a JSON file path.
        
        Args:
            json_file_path: Path to the out.json file
        """
        self.file_path = Path(json_file_path)
        self.data: Dict[str, Any] = {}
        self._load_json()
    
    def _load_json(self):
        """Load the JSON file into memory."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")
        
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
    
    # ==================== Product Info ====================
    
    def get_product_info(self) -> Dict[str, str]:
        """Get product information."""
        return self.data.get("ProductInfo", {})
    
    def get_version(self) -> str:
        """Get the product version."""
        return self.data.get("ProductInfo", {}).get("Version", "")
    
    # ==================== Scene Tree ====================
    
    def get_scene_tree(self) -> Dict[str, Any]:
        """Get the entire scene tree."""
        return self.data.get("SceneTree", {})
    
    def traverse_scene_tree(self, callback: Callable[[str, Dict[str, Any], int], None]):
        """
        Recursively traverse the scene tree and call a callback for each node.
        
        Args:
            callback: Function to call for each node. 
                     Signature: callback(node_name: str, node_data: dict, depth: int)
        """
        scene_tree = self.get_scene_tree()
        if scene_tree:
            self._traverse_node(scene_tree, callback, depth=0)
    
    def _traverse_node(self, node: Dict[str, Any], callback: Callable, depth: int = 0):
        """
        Recursively traverse a node and its children.
        
        Args:
            node: The current node dictionary
            callback: Function to call for each node
            depth: Current depth in the tree
        """
        for key, value in node.items():
            if isinstance(value, dict):
                # Call callback for this node
                callback(key, value, depth)
                
                # Recursively traverse child nodes
                self._traverse_node(value, callback, depth + 1)
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """
        Get all nodes of a specific type from the scene tree.
        
        Args:
            node_type: The type of node to search for (e.g., 'RadarDevice', 'SceneTreeNode')
        
        Returns:
            List of nodes matching the type
        """
        matching_nodes = []
        
        def collect_nodes(name: str, node_data: Dict[str, Any], depth: int):
            if node_data.get("Type") == node_type:
                matching_nodes.append({"name": name, "data": node_data, "depth": depth})
        
        self.traverse_scene_tree(collect_nodes)
        return matching_nodes
    
    def get_radar_platforms(self) -> List[Dict[str, Any]]:
        """Get all radar platforms from the scene tree."""
        return self.get_nodes_by_type("RadarPlatform")
    
    def get_radar_devices(self) -> List[Dict[str, Any]]:
        """Get all radar devices from the scene tree."""
        return self.get_nodes_by_type("RadarDevice")
    
    def get_radar_antennas(self) -> List[Dict[str, Any]]:
        """Get all radar antennas from the scene tree."""
        return self.get_nodes_by_type("RadarAntenna")
    
    def get_radar_modes(self) -> List[Dict[str, Any]]:
        """Get all radar modes from the scene tree."""
        return self.get_nodes_by_type("RadarMode")
    
    def flatten_scene_tree(self, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Get a flattened list of all nodes in the scene tree without hierarchy.
        
        Args:
            include_metadata: If True, includes node name, type, and depth in the result.
                            If False, returns only the node properties (without child nodes).
        
        Returns:
            List of all nodes in the scene tree as a flat structure.
            Each item contains only the node's own properties (Name, Type, Position, etc.)
            without nested child nodes.
        """
        flattened_nodes = []
        
        def collect_all_nodes(name: str, node_data: Dict[str, Any], depth: int):
            # Filter out child nodes - only keep primitive properties
            node_properties = {}
            for key, value in node_data.items():
                # Only include non-dict values (primitive properties)
                if not isinstance(value, dict):
                    node_properties[key] = value
            
            if include_metadata:
                flattened_nodes.append({
                    "name": name,
                    "type": node_data.get("Type", "Unknown"),
                    "depth": depth,
                    "properties": node_properties
                })
            else:
                flattened_nodes.append(node_properties)
        
        self.traverse_scene_tree(collect_all_nodes)
        return flattened_nodes
    
    def get_all_scene_tree_nodes(self) -> List[Dict[str, Any]]:
        """
        Get all scene tree nodes as a flat list with basic info.
        Convenience method that calls flatten_scene_tree with metadata.
        
        Returns:
            List of dictionaries with name, type, and key node properties
        """
        return self.flatten_scene_tree(include_metadata=True)
    
    # ==================== Meshes ====================
    
    def get_meshes(self) -> Dict[str, Any]:
        """Get all mesh information."""
        return self.data.get("Meshes", {})
    
    def get_mesh_by_name(self, mesh_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific mesh by name.
        
        Args:
            mesh_name: Name of the mesh (e.g., 'SceneElement_0')
        
        Returns:
            Mesh data or None if not found
        """
        return self.get_meshes().get(mesh_name)
    
    def get_mesh_count(self) -> int:
        """Get the total number of meshes."""
        return len(self.get_meshes())
    
    # ==================== Materials ====================
    
    def get_materials(self) -> Dict[str, Any]:
        """Get all material definitions."""
        return self.data.get("Materials", {})
    
    def get_material_by_name(self, material_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific material by name.
        
        Args:
            material_name: Name of the material
        
        Returns:
            Material data or None if not found
        """
        return self.get_materials().get(material_name)
    
    def get_materials_by_type(self, material_type: str) -> Dict[str, Any]:
        """
        Get all materials of a specific type.
        
        Args:
            material_type: Type of material (e.g., 'DIELECTRIC_COATING')
        
        Returns:
            Dictionary of matching materials
        """
        return {
            name: data for name, data in self.get_materials().items()
            if name.startswith(material_type)
        }
    
    # ==================== Coating Assignments ====================
    
    def get_coating_assignments(self) -> Dict[str, Any]:
        """Get all coating assignments."""
        return self.data.get("CoatingAssignments", {})
    
    # ==================== Solver Settings ====================
    
    def get_solver_settings(self) -> Dict[str, Any]:
        """Get solver settings."""
        return self.data.get("SolverSettings", {})
    
    def get_max_reflections(self) -> int:
        """Get the maximum number of reflections."""
        return int(self.get_solver_settings().get("MaxNumReflections", 0))
    
    def get_max_transmissions(self) -> int:
        """Get the maximum number of transmissions."""
        return int(self.get_solver_settings().get("MaxNumTransmissions", 0))
    
    def get_ray_groups(self) -> Dict[str, Any]:
        """Get ray group settings."""
        return self.get_solver_settings().get("RayGroups", {})
    
    # ==================== Hardware ====================
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return self.data.get("Hardware", {})
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get information about all GPUs."""
        hardware = self.get_hardware_info()
        num_gpus = int(hardware.get("NumGPUs", 0))
        
        gpus = []
        for i in range(num_gpus):
            gpu_key = f"GPU_{i}"
            if gpu_key in hardware:
                gpus.append(hardware[gpu_key])
        
        return gpus
    
    # ==================== Auxiliary Info ====================
    
    def get_auxiliary_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        return self.data.get("AuxiliaryInfo", {})
    
    # ==================== Utility Methods ====================
    
    def get_all_keys(self, section: Optional[str] = None) -> List[str]:
        """
        Get all keys in a section or the entire data structure.
        
        Args:
            section: Optional section name to get keys from
        
        Returns:
            List of keys
        """
        if section:
            return list(self.data.get(section, {}).keys())
        return list(self.data.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the entire data structure as a dictionary."""
        return self.data
    
    def search_by_key(self, key: str) -> List[tuple]:
        """
        Recursively search for a key in the entire data structure.
        
        Args:
            key: The key to search for
        
        Returns:
            List of tuples (path, value) where the key was found
        """
        results = []
        self._search_recursive(self.data, key, [], results)
        return results
    
    def _search_recursive(self, obj: Any, key: str, path: List[str], results: List[tuple]):
        """
        Helper method to recursively search for a key.
        
        Args:
            obj: Current object to search
            key: Key to search for
            path: Current path in the structure
            results: List to append results to
        """
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = path + [k]
                if k == key:
                    results.append((".".join(new_path), v))
                self._search_recursive(v, key, new_path, results)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = path + [f"[{i}]"]
                self._search_recursive(item, key, new_path, results)
    
    def print_summary(self):
        """Print a summary of the loaded data."""
        print("=" * 60)
        print("Perceive EM Output Summary")
        print("=" * 60)
        
        # Product Info
        product_info = self.get_product_info()
        print(f"\nProduct: {product_info.get('Product', 'N/A')}")
        print(f"Version: {product_info.get('Version', 'N/A')}")
        print(f"Build: {product_info.get('Build', 'N/A')}")
        
        # Scene Tree
        print(f"\nScene Tree Nodes:")
        radar_platforms = self.get_radar_platforms()
        radar_devices = self.get_radar_devices()
        radar_antennas = self.get_radar_antennas()
        radar_modes = self.get_radar_modes()
        print(f"  - Radar Platforms: {len(radar_platforms)}")
        print(f"  - Radar Devices: {len(radar_devices)}")
        print(f"  - Radar Antennas: {len(radar_antennas)}")
        print(f"  - Radar Modes: {len(radar_modes)}")
        
        # Meshes
        print(f"\nMeshes: {self.get_mesh_count()}")
        
        # Materials
        materials = self.get_materials()
        print(f"Materials: {len(materials)}")
        
        # Solver Settings
        solver = self.get_solver_settings()
        print(f"\nSolver Settings:")
        print(f"  - Max Reflections: {self.get_max_reflections()}")
        print(f"  - Max Transmissions: {self.get_max_transmissions()}")
        
        # Hardware
        gpus = self.get_gpu_info()
        print(f"\nHardware:")
        print(f"  - Number of GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.get('Model', 'N/A')}")
        
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Example: Read and analyze the output file
    reader = ReportSettingsReader("output/out.json")
    
    # Print summary
    reader.print_summary()
    
    # Example: Traverse the scene tree
    print("\n\nScene Tree Structure:")
    def print_node(name: str, node_data: Dict[str, Any], depth: int):
        indent = "  " * depth
        node_type = node_data.get("Type", "Unknown")
        print(f"{indent}{name} ({node_type})")
    
    reader.traverse_scene_tree(print_node)
    
    # Example: Search for all positions
    print("\n\nSearching for all 'Position' entries:")
    positions = reader.search_by_key("Position")
    for path, value in positions[:5]:  # Show first 5
        print(f"{path}: {value}")
