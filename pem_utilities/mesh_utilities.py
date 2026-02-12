import os
import numpy as np
import pyvista as pv
from pem_utilities.path_helper import get_repo_paths

def generate_cargo_on_deck(num_containers=100, filepath=None, bounds=[-25,110,-20,30], deck_z=47):
    print(f'Loading {num_containers} cargo containers on deck...')
    paths = get_repo_paths()
    mesh_container = pv.read(filepath)
    container_bounds = mesh_container.bounds
    # print(f'Container dimensions (LxWxH): {container_bounds[1]-container_bounds[0]:.2f} x {container_bounds[3]-container_bounds[2]:.2f} x {container_bounds[5]-container_bounds[4]:.2f} meters')
    
    # Container dimensions
    cont_length = container_bounds[1] - container_bounds[0]
    cont_width = container_bounds[3] - container_bounds[2]
    cont_height = container_bounds[5] - container_bounds[4]
    
    # Grid bounds in XY plane
    grid_x_min, grid_x_max = bounds[0], bounds[1]
    grid_y_min, grid_y_max = bounds[2], bounds[3]
    
    # Spacing factor
    spacing_factor = 1.05
    
    # Calculate how many containers fit in each direction
    num_x = int((grid_x_max - grid_x_min) / (cont_length * spacing_factor))
    num_y = int((grid_y_max - grid_y_min) / (cont_width * spacing_factor))
    
    print(f'Grid capacity per layer: {num_x} x {num_y} = {num_x * num_y} containers')
    
    # Place containers in a 3D grid
    container_count = 0
    current_layer = 0
    total_mesh = pv.PolyData()
    for i in range(num_containers):
        # Calculate position within the grid
        containers_per_layer = num_x * num_y
        if containers_per_layer == 0:
            print("Grid too small for containers!")
            break
            
        layer = i // containers_per_layer
        position_in_layer = i % containers_per_layer
        
        x_idx = position_in_layer % num_x
        y_idx = position_in_layer // num_x
        
        # Calculate actual position
        x_pos = grid_x_min + x_idx * cont_length * spacing_factor
        y_pos = grid_y_min + y_idx * cont_width * spacing_factor
        z_pos = deck_z + layer * cont_height * spacing_factor
        
        # Create and place container
        cont_copy = mesh_container.copy()
        cont_copy.translate([x_pos, y_pos, z_pos], inplace=True)
        
        # Color by layer for visualization
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        color = colors[layer % len(colors)]
        
        container_count += 1
        total_mesh+=cont_copy
    total_mesh.triangulate(inplace=True)
    total_mesh.save(os.path.join(paths.cache ,'cargo_on_deck.stl'))
    print(f'Total containers placed: {container_count}')
    return os.path.join(paths.cache,'cargo_on_deck.stl')

def get_z_elevation_from_mesh(xy,meshes,return_min_and_max=True):

    x_loc = xy[0]
    y_loc = xy[1]
    if not isinstance(meshes, list):
        meshes = [meshes]

    if len(meshes) == 0:
        if return_min_and_max:
            return [0,0]
        else:
            return 0

    intersection_points = []
    buffer = 10  # additional distance so intersection test is further away than directly on surface
    for mesh in meshes:
        bounds = mesh.bounds
        min_z = bounds[4]
        max_z = bounds[5]

        start_z = min_z - buffer
        stop_z = max_z + buffer

        # start from +Z and look down, intersection will be first point hit.
        stop_ray = [x_loc, y_loc, start_z]
        start_ray = [x_loc, y_loc, stop_z]
        intersection_point, _ = mesh.ray_trace(start_ray, stop_ray)

        # ray will intersect with a geometry and return a point (or list with size greater than 0)
        if len(intersection_point) > 0:
            intersection_points.append(intersection_point.flatten()[2])
    intersection_points = np.array(intersection_points)


    if return_min_and_max:
        if len(intersection_points) == 0:
            return [None, None]
        else:
            return [np.min(intersection_points), np.max(intersection_points)]
    elif len(intersection_points) == 0:
        return None
    else:
        return intersection_points[0]
    

def get_average_z_pos_and_normal(source_mesh, target_mesh, source_pos=None,source_orientation=None):
    """
    Compute the average z position and surface normal where two meshes intersect.
    
    Parameters
    ----------
    source_mesh : pyvista.PolyData
        The source mesh that intersects with the target mesh
    target_mesh : pyvista.PolyData
        The target mesh
        
    Returns
    -------
    avg_z : float or None
        Average z coordinate of intersection points, None if no intersection
    avg_normal : numpy.ndarray or None
        Average surface normal vector [nx, ny, nz] at intersection, None if no intersection
    """
    # Compute the intersection between the two meshes
    source_mesh.rotate_z(source_orientation,inplace=True)
    source_mesh.translate(source_pos,inplace=True)

    intersection = source_mesh.boolean_intersection(target_mesh)
    
    # Check if there is any intersection
    if intersection.n_points == 0:
        return None, None
    
    # Get all intersection points
    points = intersection.points
    
    # Calculate average z position
    avg_z = np.mean(points[:, 2])
    
    # Compute normals for the intersection surface if not already present
    if 'Normals' not in intersection.point_data:
        intersection = intersection.compute_normals(point_normals=True, cell_normals=False)
    
    # Get the normals at intersection points
    normals = intersection.point_data['Normals']
    
    # Calculate average normal (and normalize it)
    avg_normal = np.mean(normals, axis=0)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    
    return avg_z, avg_normal