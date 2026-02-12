import os
import glob
import pyvista as pv
import sys

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pem_utilities.path_helper import get_repo_paths

paths = get_repo_paths()

path_to_vtp = os.path.join(paths.models, 'Helsinki')
output_path = os.path.join(paths.output, 'convert_vtp_to_gltf')
os.makedirs(output_path, exist_ok=True) 
vtp_files = glob.glob(os.path.join(path_to_vtp, '*.vtp'))
for vtp_file in vtp_files:
    mesh = pv.read(vtp_file)
    base_name = os.path.basename(vtp_file)
    file_name_no_ext = os.path.splitext(base_name)[0]
    output_file = os.path.join(output_path, f'{file_name_no_ext}.gltf')

    # Check for texture file
    texture = None
    for ext in ['.png', '.jpg', '.jpeg']:
        texture_path = os.path.join(path_to_vtp, file_name_no_ext + ext)
        if os.path.exists(texture_path):
            texture = pv.read_texture(texture_path)
            print(f"Found texture: {texture_path}")
            break
            
    print(f"Saving {output_file}")
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, texture=texture)
    pl.export_gltf(output_file)
    pl.close()
