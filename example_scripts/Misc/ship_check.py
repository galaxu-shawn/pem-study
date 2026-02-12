import pyvista as pv
import os
from glob import glob
from pem_utilities.path_helper import get_repo_paths # common paths to reference

paths = get_repo_paths() 
model_path = os.path.join(paths.models,'Ships')
output_path = os.path.join(paths.models,'Ships','decimate')
# model_path=output_path
ship_files = glob(os.path.join(model_path,'*.obj'))
# also check for stl files and extend the list
ship_files.extend(glob(os.path.join(model_path,'*.stl')))

# create a pyvista plotter

ship_files = [ship_files[8]]

for ship_file in ship_files:
    plotter = pv.Plotter()
    print(f'Loading ship file: {ship_file}')
    mesh = pv.read(ship_file)

    # reduce number of faces
    # initial_num_faces = mesh.n_cells
    # target_reduction = 0.80  # reduce to 20% of original
    # mesh = mesh.decimate(target_reduction)
    # final_num_faces = mesh.n_cells
    # print(f'Reduced number of faces from {initial_num_faces} to {final_num_faces}')

    flat_plate = pv.Plane(center=(0,0,5.0), direction=(0,0,1), i_size=500, j_size=500, i_resolution=1, j_resolution=1)
    # get file name no extension
    file_name_no_ext = os.path.splitext(os.path.basename(ship_file))[0]
    # mesh.save(os.path.join(output_path,file_name_no_ext+'.stl'))
    plotter.add_mesh(flat_plate, color='blue', opacity=0.5)
    plotter.add_mesh(mesh, color='white', show_edges=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title=os.path.basename(ship_file), window_size=[800,600])
    # print dimensions


    # plotter.close()