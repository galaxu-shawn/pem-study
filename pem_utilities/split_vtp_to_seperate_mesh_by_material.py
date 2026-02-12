import numpy as np
import pyvista as pv
import os
import glob
import shutil

try:
    import pymeshlab
except ImportError:
    print("pymeshlab is not installed. Please install it to use this feature.")
    print("pip install pymeshlab")
    pymeshlab = None

# move all materials to a single folder, for example all geometry labelled as terrain will be in the terrain folder
move_material_to_folder = True

# This script is designed to split a PyVista mesh into separate meshes based on an array of indices.
# useful to segment meshes that have material properties encoded in the cell data.

def split_mesh_by_indices(mesh):
    """
    Split a PyVista mesh into separate meshes based on an array of indices.

    Parameters:
    -----------
    mesh : pyvista.DataSet
        The input mesh to split
    indices : array-like
        Array of integer indices, one per cell in the mesh

    Returns:
    --------
    dict
        Dictionary mapping each unique index to its corresponding mesh
    """

    if 'material' in mesh.cell_data.keys():
        print('Using materials from mesh.cell_data')
        mat_idxs = np.array(mesh.cell_data['material'], dtype=np.int32)
    else:
        raise ValueError("Mesh does not contain 'material' cell data")

    # Make sure the indices length matches the number of cells
    if len(mat_idxs) != mesh.n_cells:
        raise ValueError(
            f"Indices array length ({len(mat_idxs)}) does not match the number of cells in the mesh ({mesh.n_cells})")

    # Get unique indices
    unique_indices = np.unique(mat_idxs)

    # Dictionary to store results
    result_meshes = {}

    # Extract cells for each unique index
    for idx in unique_indices:
        # Get the cell IDs where indices equals idx
        mask = mat_idxs == idx
        cell_ids = np.arange(mesh.n_cells)[mask]

        # Extract those cells to a new mesh
        if len(cell_ids) > 0:
            submesh = mesh.extract_cells(cell_ids)
            result_meshes[idx] = submesh

    return result_meshes


def convert_unstructured_to_surface(mesh):
    """
    Convert an UnstructuredGrid to a surface mesh (PolyData).

    Parameters:
    -----------
    mesh : pyvista.UnstructuredGrid
        The input unstructured grid mesh

    Returns:
    --------
    pyvista.PolyData
        The extracted surface mesh
    """
    # Method 1: Extract the surface
    surface = mesh.extract_surface()

    # Method 2: Alternative approach that works well for some complex geometries
    # surface = mesh.extract_geometry()

    # Ensure the mesh is triangulated
    surface.triangulate(inplace=True)

    # Compute normals for proper rendering
    surface.compute_normals(inplace=True)

    return surface


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

material_name = {}
for each in material_idx.keys():
    material_name[material_idx[each]] = each

example_file = r'../models/Helsinki/Tile_+1984_+2688.vtp'
example_file = r'C:\tmp\Helsinki - Ground Truth\vtp'
out_dir = r'../output/helsinki_export/'


# Example usage
# Load your mesh
# check if example_file is folder or file
if os.path.isdir(example_file):
    example_files = glob.glob(os.path.join(example_file, '*.vtp'))  # Load the first .vtp file in the directory
else:
    example_files = [example_file]

all_files = {}
for mat in material_name.keys():
    all_files[mat] = []


for example_file in example_files:
    input_dir = os.path.dirname(example_file)
    mesh = pv.read(example_file)  # or any other format
    # get file name without extension or directory
    tile_name = os.path.splitext(os.path.basename(example_file))[0]
    image_name = os.path.splitext(os.path.basename(example_file))[0] + '.jpg'

    print(f"Processing file: {tile_name}")


    # Split the mesh
    split_meshes = split_mesh_by_indices(mesh)

    # Now you can access or save each submesh
    # create output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, tile_name), exist_ok=True)
    if os.path.isfile(os.path.join(input_dir,image_name)):
        # copy image_name file to the output directory, leaving the source file in place using shutil
        shutil.copy(os.path.join(input_dir,image_name), os.path.join(out_dir, tile_name, image_name))
        image_full_name = os.path.join(input_dir, image_name)
    else:
        image_full_name = None
    for idx, submesh in split_meshes.items():
        if isinstance(submesh, pv.UnstructuredGrid):
            submesh = convert_unstructured_to_surface(submesh)
        else:
            # If already a surface-type mesh, just triangulate
            submesh.triangulate(inplace=True) # triangulate the mesh
        print(f"Mesh {idx}: {submesh.n_cells} cells")
        # Optional: save each submesh
        if idx in material_idx.keys():
            mat_name = material_idx[idx]
        else:
            mat_name = 'unlabelled'

        output_full_path = os.path.join(out_dir, f"{tile_name}/{tile_name}_{mat_name}.obj")
        submesh.save(output_full_path)

        # open the saved obj file and write insert the mtl filename into the first line of the file
        with open(output_full_path, 'r') as f:
            lines = f.readlines()
        lines.insert(0,f"mtllib {tile_name}.mtl\n")
        lines.insert(1, f"usemtl {tile_name}_textured\n")
        with open(output_full_path, 'w') as f:
            f.writelines(lines)

        # write a single mtl file that can be used for all meshes
        output_full_path_mtl = os.path.join(out_dir, f"{tile_name}/{tile_name}.mtl")
        with open(output_full_path_mtl, 'w') as f:
            f.writelines("Ka 1 1 1\n")
            f.writelines("Kd 1 1 1\n")
            f.writelines("d 1\n")
            f.writelines("Ns 0\n")
            f.writelines("illum 1\n")
            f.writelines(f"map_Kd {tile_name}.jpg\n")
            f.writelines(f"newmtl {tile_name}_textured\n")
            f.writelines("Ka 0.501961 0.501961 0.501961\n")
            f.writelines("Kd 0.501961 0.501961 0.501961\n")
            f.writelines("d 1\n")
            f.writelines("Ns 0\n")
            f.writelines("illum 1\n")

            # Optionally visualize
            # submesh.plot()


        # save as better formated obj file that is compatible with cesium
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(output_full_path)
        # Convert the mesh
        ms.save_current_mesh(output_full_path)

        mtl_name = os.path.join(out_dir, f"{tile_name}/{tile_name}_{mat_name}.obj.mtl")
        all_files[mat_name].append([output_full_path,image_full_name,mtl_name])

if move_material_to_folder:
    # make directory for consilidated files
    consolidated_dir = os.path.join(out_dir, 'consolidated')
    os.makedirs(consolidated_dir, exist_ok=True)
    # make subdirectory for each material
    for mat in all_files.keys():
        os.makedirs(os.path.join(consolidated_dir, mat), exist_ok=True)
        for each in all_files[mat]:
            geo_file = each[0]
            image_file = each[1]
            mtl_file = each[2]
            # copy the files to consolidated directory
            shutil.copy(geo_file, os.path.join(consolidated_dir, mat, os.path.basename(geo_file)))
            if image_file is not None:
                shutil.copy(image_file, os.path.join(consolidated_dir, mat, os.path.basename(image_file)))
            shutil.copy(mtl_file, os.path.join(consolidated_dir, mat, os.path.basename(mtl_file)))


