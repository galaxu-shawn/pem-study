#
# Copyright ANSYS. All rights reserved.
#
# Using the Semantic Urban Mesh Segmentation (SUMS) tool, the geometry output can be converted to a format that can be used in the simulation.
# The output of the SUMS tool is a ply file that contains the material properties of the scene. This script converts the ply file to a vtp file that can be read by the simulation.
# Convert ply to vtp, where material properties are baked into the mesh cell data.
#
# The workflow is as follows:
# 1. Predict the material properties of a scene using the Semantic Urban Mesh Segmentation (SUMS) tool.
#       The work is based on this paper -
#       https://www.sciencedirect.com/science/article/pii/S0924271621001854
# 2. Convert the output of the SUMS tool to a format that can be used in the simulation.
#       convert_ply_to_btp_with_tex.py can convert the resulting ply file into a Perceive EM readable vtp format that
#       has the material properties baked into the mesh cell data. The mapping is currently based on the material index
#       defined in teh material_library.json file.

import os
import glob
import pyvista as pv
import numpy as np
import shutil
def convert(input_dir_or_file, output_dir,include_texture_image=True):
    """Converts all meshes defined by ply (output from SUMS) to standard ply format that can be read by pyminiply
    INPUT file must be ASCII ply, OUTPUT file will be ASCII ply with x,y,z,nx,ny,nz,red,green,blue,uv1,uv2
    """

    if include_texture_image:
        print('Including texture image if it exists')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # determine if input is a directory or a file
    if os.path.isdir(input_dir_or_file):
        # get all files in the input directory, recursively
        files = glob.glob(input_dir_or_file + '/**/*.ply', recursive=True)
    else:
        files = [input_dir_or_file]

    # Iterate over all files in the input directory
    file_counter = 0
    for filename in files:
        file_counter+=1
        basename = os.path.basename(filename)
        input_dir = os.path.dirname(filename)
        basename_no_ext = os.path.splitext(basename)[0]
        basename_no_ext = basename_no_ext.replace('_groundtruth_L1', '')
        output_path = os.path.join(output_dir,basename_no_ext+'.vtp')

        pv_tex = None
        if include_texture_image:
            texture_path= os.path.join(input_dir, basename_no_ext + '_0.jpg')
            if os.path.exists(texture_path):
                texture_path = texture_path
                shutil.copy(texture_path, os.path.join(output_dir, basename_no_ext + '.jpg'))
                pv_tex = pv.read_texture(texture_path)
            else:
                texture_path = None

        print(f'Converting {filename} to {output_path}, {file_counter} of {len(files)} ')

        # load all lines of the text file into memory
        with open(filename, 'r') as file:
            lines = file.readlines()


        end_of_header = None
        # header has some information that we need to extract to know which colunms are the colors and materials
        sample_line = ''
        n=1
        # if empty lines at end of file, go back until we find a line with content
        while len(sample_line) == 0:
            sample_line = lines[-1*n]  # grab a sample line to help determine length of lists in the header
            n+=1
        sample_line = sample_line.split(' ')
        num_columns = len(sample_line)
        face_properties_col_idx = None

        for line in lines:
            if 'element vertex' in line:
                num_element_vertex = line.split(' ')[-1].strip()
            elif 'element face' in line:
                num_element_face = line.split(' ')[-1].strip()
                face_properties_col_idx = lines.index(line)+1
            elif 'end_header' in line:
                end_of_header = lines.index(line)+1
                break
        if end_of_header is None:
            print(f'Error: end of header not found in {filename}')

        red_idx = None
        green_idx = None
        blue_idx = None
        material_idx = None
        confidence_idx = None
        normal_x_idx = None
        normal_y_idx = None
        normal_z_idx = None
        x_idx = None
        y_idx = None
        z_idx = None
        texture_idx = None

        face_property_lines = lines[face_properties_col_idx:end_of_header-1]
        idx_accum = 0
        for line in face_property_lines:
            # check if current property is a list, if it is use teh sample line to determine the number of elements
            if 'list' in line:
                idx_accum += int(sample_line[idx_accum])

            if 'vertex_indices' in line:
                x_idx = idx_accum-2
                y_idx = idx_accum-1
                z_idx = idx_accum+0
            elif 'property uchar r' in line or 'property float r' in line or 'property int r' in line:
                red_idx = idx_accum
            elif 'property uchar g' in line or 'property float g' in line or 'property int g' in line:
                green_idx = idx_accum
            elif 'property uchar b' in line or 'property float b' in line or 'property int b' in line:
                blue_idx = idx_accum
            elif 'property int label' in line:
                material_idx = idx_accum
            elif 'property float label_probabilities' in line:
                confidence_idx = idx_accum
            elif 'property float nx' in line:
                normal_x_idx = idx_accum
            elif 'property float ny' in line:
                normal_y_idx = idx_accum
            elif 'property float nz' in line:
                normal_z_idx = idx_accum
            elif 'texcoord' in line:
                texture_idx = idx_accum-5 # assuming texture is always 6 values (uv for each vertext)

            idx_accum += 1

        lines = lines[end_of_header:]  # remove the header


        # split into list of positions and list of indices
        verts = lines[:int(num_element_vertex)]
        faces = lines[int(num_element_vertex):]

        points = []
        for vert in verts:
            vert = vert.split(' ')
            points.append([np.float32(vert[0]), np.float32(vert[1]), np.float32(vert[2])])

        texture_coords_unique = np.zeros((len(points), 2))
        colors = []
        normals = []
        confidence = []
        tri_idxs = []
        tri_idxs_no_leading_size = []
        materials_idx = []
        texture_coords = []
        for face in faces:
            face = face.split(' ')
            tri_idxs.append([3, int(face[x_idx]), int(face[y_idx]), int(face[z_idx])])
            tri_idxs_no_leading_size.append([int(face[x_idx]), int(face[y_idx]), int(face[z_idx])])
            rgb = [float(face[red_idx]), float(face[green_idx]), float(face[blue_idx])]
            # normalize the color values to be between 0 and 1 if they are greater than 1
            for i in range(len(rgb)):
                if rgb[i] > 1:
                    rgb[i] = rgb[i]/255

            colors.append(rgb)
            if normal_x_idx is not None:
                normals.append([float(face[normal_x_idx]), float(face[normal_y_idx]), float(face[normal_z_idx])])
            if texture_idx is not None:
                texture_coords.append([float(face[texture_idx]), float(face[texture_idx+1]),
                                       float(face[texture_idx+2]),float(face[texture_idx+3]),
                                       float(face[texture_idx+4]),float(face[texture_idx+5])])
            materials_idx.append([int(face[material_idx])])
            confidence.append([float(face[confidence_idx])])
        # pyvista requires a single
        tri_idxs_no_leading_size = np.array(tri_idxs_no_leading_size)
        for pt_idx, cell in enumerate(tri_idxs_no_leading_size):
            texture_coords_unique[cell[0]] = [texture_coords[pt_idx][0], texture_coords[pt_idx][1]]
            texture_coords_unique[cell[1]] = [texture_coords[pt_idx][2], texture_coords[pt_idx][3]]
            texture_coords_unique[cell[2]] = [texture_coords[pt_idx][4], texture_coords[pt_idx][5]]
        # index locations where materials_idx is equal to 4
        # materials_idx = np.array(materials_idx)
        # idxs = np.where(materials_idx == 4)
        mesh = pv.PolyData(points,faces=tri_idxs)
        mesh.cell_data['color'] = np.array(colors)
        mesh.cell_data['material'] = materials_idx
        mesh.active_texture_coordinates = texture_coords_unique
        if normal_x_idx is not None:
            mesh.cell_data['normal'] = normals
        mesh.cell_data['confidence'] = confidence
        if pv_tex is not None:
            mesh.texture = pv_tex

        # i am having an issue with the texuture cordinates. As a work around. I will load the ply
        # using the pyvista reader, and then apply the scalars and texture to that mesh, then save as vtp
        # not sure what is wrong with my texture cordinates, but this works

        mesh2 = pv.read(filename)
        if pv_tex is not None:
            mesh.texture = pv_tex
        mesh2.cell_data['color'] = mesh.cell_data['color']
        mesh2.cell_data['material'] = mesh.cell_data['material']
        if normal_x_idx is not None:
            mesh2.cell_data['normal'] = mesh2.cell_data['normal']
        mesh2.cell_data['confidence'] = mesh.cell_data['confidence']
        mesh2.save(output_path)


if __name__ == "__main__":
    input = r"C:\tmp\data_demo\output\predict"
    output_dir = r"C:\tmp\data_demo\output\predict\converted"


    input = r"C:\tmp\Helsinki - Ground Truth\ply"
    output_dir = r"C:\tmp\Helsinki - Ground Truth\vtp"

    convert(input, output_dir)