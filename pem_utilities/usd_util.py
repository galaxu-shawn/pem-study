from pxr import Vt, Sdf, Usd, UsdGeom, Gf
from typing import List, Type, Tuple
import os
import math
import numpy as np
import pyvista as pv


class UsdActor:
    def __init__(self, filename,scale_mesh=1.0):
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File not found: {filename}')
        self.stage = Usd.Stage.Open(filename)
        self.scale_mesh = scale_mesh
        self.all_prims = {}
        self.all_pv_meshes = {}

    def to_cartesian(self, radius, theta, phi, scale):
        x = radius * math.cos(phi) * math.sin(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(theta)
        return Gf.Vec3f(x * scale, y * scale, z * scale)

    def decompose_matrix(self, mat: Gf.Matrix4d):
        translate: Gf.Vec3d = mat.ExtractTranslation()
        rotmat = mat.ExtractRotationMatrix().GetTranspose()
        scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in rotmat))
        return translate, rotmat, scale

    def get_world_transform_xform_euler(self, prim):
        xform = UsdGeom.Xformable(prim)
        mtrx = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return self.decompose_matrix(mtrx)

    def update_actor(self, prim, time=0):
        transform = self.get_world_transform_xform_euler(prim)
        # compute velocities
        new_pos = np.array(transform[0])
        # update position
        # actor.coord_sys.pos = newPos
        rot = np.array(transform[1])
        return new_pos, rot
        # actor.coord_sys.rot = rot
        # actor.update_actor(time)


    def find_prims_by_type(self, prim_type: Type[Usd.Typed]):
        found_prims = [x for x in self.stage.Traverse() if x.IsA(prim_type)]
        return found_prims

    def get_all_meshes(self, time_code=Usd.TimeCode.Default(), apply_transforms=True):
        """
        Extract all mesh primitives from the USD stage and convert to PyVista meshes.
        
        Args:
            time_code: USD time code to evaluate attributes at (default: Usd.TimeCode.Default())
            apply_transforms: Whether to apply world transforms to nested meshes (default: True)
        """
        usd_meshes = self.find_prims_by_type(UsdGeom.Mesh)
        print(f"Found {len(usd_meshes)} mesh primitives in USD file")

        for usd_mesh in usd_meshes:
            prim = UsdGeom.Mesh(usd_mesh)
            prim_path = f'{prim.GetPath()}'
            self.all_prims[prim_path] = prim

            try:
                # Get mesh attributes at specified time code
                points_attr = prim.GetPointsAttr()
                indices_attr = prim.GetFaceVertexIndicesAttr()
                counts_attr = prim.GetFaceVertexCountsAttr()
                
                # Check if attributes exist and are valid
                if not (points_attr and indices_attr and counts_attr):
                    print(f"Warning: Missing required attributes for mesh {prim_path}")
                    continue
                
                # Get attribute values at the specified time
                mesh_points = points_attr.Get(time_code)
                point_indices = indices_attr.Get(time_code)
                face_vertex_counts = counts_attr.Get(time_code)
                
                # Validate mesh data
                if mesh_points is None or point_indices is None or face_vertex_counts is None:
                    print(f"Warning: Invalid mesh data for {prim_path} at time {time_code}")
                    continue
                
                if len(mesh_points) == 0 or len(point_indices) == 0 or len(face_vertex_counts) == 0:
                    print(f"Warning: Empty mesh data for {prim_path}")
                    continue
                
                # Convert to numpy arrays
                mesh_points = np.array(mesh_points)
                point_indices = np.array(point_indices)
                face_vertex_counts = np.array(face_vertex_counts)
                
                # Apply world transform if requested (important for nested meshes)
                if apply_transforms:
                    try:
                        xform = UsdGeom.Xformable(usd_mesh)
                        world_matrix = xform.ComputeLocalToWorldTransform(time_code)
                        
                        # Transform points to world coordinates
                        transformed_points = []
                        for point in mesh_points:
                            # Convert to homogeneous coordinates
                            point_4d = Gf.Vec4f(point[0], point[1], point[2], 1.0)
                            # Apply transform
                            transformed_point = world_matrix * point_4d
                            # Convert back to 3D
                            transformed_points.append([transformed_point[0], transformed_point[1], transformed_point[2]])
                        
                        mesh_points = np.array(transformed_points)
                        print(f"Applied world transform for mesh: {prim_path}")
                        
                    except Exception as transform_error:
                        print(f"Warning: Could not apply transform for {prim_path}: {transform_error}")
                        # Continue with local coordinates
                
                # Build faces array for PyVista
                faces = []
                count = 0
                for face_count in face_vertex_counts:
                    if count + face_count > len(point_indices):
                        print(f"Warning: Face vertex count exceeds available indices for {prim_path}")
                        break
                    cur = list(point_indices[count:count + face_count])
                    faces.append([face_count, *cur])
                    count += face_count
                
                if not faces:
                    print(f"Warning: No valid faces found for {prim_path}")
                    continue
                
                faces = np.hstack(faces)
                
                # Create PyVista mesh
                surf = pv.PolyData(mesh_points, faces)
                surf.triangulate(inplace=True)
                
                # Apply scaling if specified (after world transform)
                if self.scale_mesh != 1.0:
                    surf.scale([self.scale_mesh, self.scale_mesh, self.scale_mesh], inplace=True)

                self.all_pv_meshes[prim_path] = surf
                
                # Show hierarchy information
                parent_path = usd_mesh.GetParent().GetPath() if usd_mesh.GetParent() else "/"
                print(f"Successfully processed mesh: {prim_path} (parent: {parent_path}) - {len(mesh_points)} points, {len(faces)//4} faces")
                
            except Exception as e:
                print(f"Error processing mesh {prim_path}: {str(e)}")
                continue

