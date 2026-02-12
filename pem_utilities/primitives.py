"""
Primitive 3D Geometry Classes for Electromagnetic Simulation

This module provides a collection of 3D geometric primitive classes for creating
meshes used in electromagnetic simulations. Each class can generate PyVista meshes
for various geometric shapes including spheres, cubes, cylinders, corner reflectors,
and rough surfaces.

Classes:
    Sphere: Generates spherical meshes with configurable resolution
    Cube: Generates cubic meshes with optional RCS-based sizing
    Cylinder: Generates cylindrical meshes with configurable orientation
    Dihedral: Generates dihedral corner reflector meshes
    RoughPlane: Generates rough surface planes with controlled statistical properties
    Capsule: Generates capsule (cylinder with hemispherical ends) meshes
    Plane: Generates planar meshes with configurable orientation
    CornerReflector: Generates triangular or square corner reflector meshes

Dependencies:
    numpy: For numerical computations
    pyvista: For 3D mesh generation and manipulation
    scipy.ndimage: For Gaussian filtering in rough surface generation

Author: [Author Name]
Date: [Date]
"""

import numpy as np
import pyvista as pv

class Cone:
    """
    A class for generating cone meshes.
    
    This class creates cone meshes with configurable radius, height, and angular resolution.
    
    Attributes:
        radius (float): Radius of the cone base
        height (float): Height of the cone
        num_theta (int): Angular resolution
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, radius=1, height=1, num_theta=45):
        """
        Initialize a Cone object.
        
        Args:
            radius (float, optional): Radius of the cone base. Defaults to 1.
            height (float, optional): Height of the cone. Defaults to 1.
            num_theta (int, optional): Angular resolution. Defaults to 45.
        """
        self.radius = radius
        self.height = height
        self.num_theta = num_theta

    def generate_mesh(self, t=None):
        """
        Generate the cone mesh.
        
        Args:
            t (float, optional): Time parameter for animations. Not used but required 
                               for interface compatibility. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated cone mesh
        """
        # Generate cone mesh and triangulate
        self.mesh = pv.Cone(radius=self.radius, height=self.height, resolution=self.num_theta)
        self.mesh = self.mesh.triangulate()
        return self.mesh

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)

class Sphere:
    """
    A class for generating spherical meshes.
    
    This class creates spherical meshes with configurable resolution and radius.
    The radius can be automatically calculated from a given radar cross section (RCS).
    
    Attributes:
        radius (float): Radius of the sphere
        num_theta (int): Number of theta divisions for mesh resolution
        num_phi (int): Number of phi divisions for mesh resolution
        create_as_geodesic_polyhedron (bool): Flag for geodesic polyhedron generation (not implemented)
        subdivisions (int): Number of subdivisions for geodesic polyhedron
        wl (float): Wavelength for RCS calculations
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, radius=1, num_theta=45, num_phi=45, create_as_geodesic_polyhedron=False, subdivisions=3, rcs=None, wl=0.3):
        """
        Initialize a Sphere object.
        
        Args:
            radius (float, optional): Radius of the sphere. Defaults to 1.
            num_theta (int, optional): Angular resolution in theta direction. Defaults to 45.
            num_phi (int, optional): Angular resolution in phi direction. Defaults to 45.
            create_as_geodesic_polyhedron (bool, optional): Create as geodesic polyhedron. Defaults to False.
            subdivisions (int, optional): Number of subdivisions for geodesic. Defaults to 3.
            rcs (float, optional): Radar cross section to calculate radius from. Defaults to None.
            wl (float, optional): Wavelength for RCS calculations. Defaults to 0.3.
        """
        self.radius = radius
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.create_as_geodesic_polyhedron = create_as_geodesic_polyhedron
        self.subdivisions = subdivisions
        
        # If RCS is provided, calculate radius from it
        if rcs is not None:
            self.wl = wl
            self.radius = self.calculate_length_from_rcs(rcs)
            print(f'Radius calculated from RCS {rcs}: {self.radius}')

    def calculate_length_from_rcs(self,rcs):
        """
        Calculate the radius of a sphere given its RCS value.
        
        Using the formula: RCS = π * r^2, where r is the radius.
        
        Args:
            rcs (float): Radar cross section value
            
        Returns:
            float: Calculated radius
        """
        radius = np.power(rcs / np.pi, 1/2)
        return radius

    def generate_mesh(self, t=None):
        """
        Generate the sphere mesh.
        
        Args:
            t (float, optional): Time parameter for animations. Not used but required 
                               for interface compatibility. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated sphere mesh
        """
        # TODO: Geodesic polyhedron creation is not working, might be pyvista version dependency
        # if self.create_as_geodesic_polyhedron:
        #     self.mesh = Icosphere(radius=self.radius, center=(0.0, 0.0, 0.0), nsub=self.subdivisions)
        # else:
        
        # Generate standard sphere mesh
        self.mesh = pv.Sphere(radius=self.radius, theta_resolution=self.num_theta, phi_resolution=self.num_phi)
        return self.mesh

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)


class Cube:
    """
    A class for generating cubic meshes.
    
    This class creates cubic meshes with configurable dimensions. The dimensions
    can be automatically calculated from a given radar cross section (RCS).
    
    Attributes:
        x_length (float): Length in x-direction
        y_length (float): Length in y-direction  
        z_length (float): Length in z-direction
        wl (float): Wavelength for RCS calculations
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, x_length=1, y_length=1, z_length=1, rcs=None, wl=0.3):
        """
        Initialize a Cube object.
        
        Args:
            x_length (float, optional): Length in x-direction. Defaults to 1.
            y_length (float, optional): Length in y-direction. Defaults to 1.
            z_length (float, optional): Length in z-direction. Defaults to 1.
            rcs (float, optional): Radar cross section to calculate dimensions from. Defaults to None.
            wl (float, optional): Wavelength for RCS calculations. Defaults to 0.3.
        """
        self.x_length = x_length
        self.y_length = y_length
        self.z_length = z_length
        
        # If RCS is provided, calculate dimensions from it (creates a cube)
        if rcs is not None:
            self.wl = wl
            self.x_length = self.calculate_length_from_rcs(rcs)
            self.y_length = self.x_length
            self.z_length = self.x_length

    def calculate_length_from_rcs(self, rcs):
        """
        Calculate the length of a cube side given its RCS value.
        
        Using the formula: RCS = 4π/3 * a^4/λ^2, where a is the side length
        and λ is the wavelength.
        
        Args:
            rcs (float): Radar cross section value
            
        Returns:
            float: Calculated side length
        """
        length = np.power(rcs * self.wl**2 / (4 * np.pi), 1/4)
        return length

    def generate_mesh(self, t=None):
        """
        Generate the cube mesh.
        
        Args:
            t (float, optional): Time parameter for animations. Not used but required 
                               for interface compatibility. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated cube mesh
        """
        # Generate cube mesh and triangulate for consistency
        self.mesh = pv.Cube(x_length=self.x_length, y_length=self.y_length, z_length=self.z_length)
        self.mesh = self.mesh.triangulate()
        return self.mesh

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)


class Cylinder:
    """
    A class for generating cylindrical meshes.
    
    This class creates cylindrical meshes with configurable radius, height,
    angular resolution, and orientation.
    
    Attributes:
        radius (float): Radius of the cylinder
        height (float): Height of the cylinder
        num_theta (int): Angular resolution
        orientation (list): Orientation vector [x, y, z]
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, radius=1, height=1, num_theta=45, orientation=[1, 0, 0]):
        """
        Initialize a Cylinder object.
        
        Args:
            radius (float, optional): Radius of the cylinder. Defaults to 1.
            height (float, optional): Height of the cylinder. Defaults to 1.
            num_theta (int, optional): Angular resolution. Defaults to 45.
            orientation (list, optional): Orientation vector [x, y, z]. Defaults to [1, 0, 0].
        """
        self.radius = radius
        self.height = height
        self.num_theta = num_theta
        self.orientation = orientation

    def generate_mesh(self, t=None):
        """
        Generate the cylinder mesh.
        
        Args:
            t (float, optional): Time parameter for animations. Not used but required 
                               for interface compatibility. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated cylinder mesh
        """
        # Generate cylinder mesh with capping and triangulate
        self.mesh = pv.Cylinder(
            radius=self.radius,
            height=self.height,
            resolution=self.num_theta,
            capping=True,
            direction=self.orientation
        )
        self.mesh = self.mesh.triangulate()
        return self.mesh

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)


class Dihedral:
    """
    A class for generating dihedral corner reflector meshes.
    
    This class creates dihedral corner reflectors consisting of two perpendicular
    planes. The size can be automatically calculated from a given RCS value.
    
    Attributes:
        width (float): Width of the dihedral
        height (float): Height of the dihedral
        rcs (float): Radar cross section
        wl (float): Wavelength for RCS calculations
        orientation (list): Orientation vector [x, y, z]
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, width=1.0, height=1.0, rcs=None, wl=0.3, orientation=[1, 0, 0]):
        """
        Initialize a Dihedral object.
        
        Args:
            width (float, optional): Width of the dihedral. Defaults to 1.0.
            height (float, optional): Height of the dihedral. Defaults to 1.0.
            rcs (float, optional): Radar cross section to calculate size from. Defaults to None.
            wl (float, optional): Wavelength for RCS calculations. Defaults to 0.3.
            orientation (list, optional): Orientation vector [x, y, z]. Defaults to [1, 0, 0].
        """
        self.width = width
        self.height = height
        self.rcs = rcs
        self.wl = wl
        self.orientation = orientation

    def calculate_dihedral_size_from_rcs(self):
        """
        Estimate the size of a dihedral corner reflector based on a given RCS value.
        
        This is a rough approximation using the high-frequency RCS formula for a dihedral.
        RCS = (4 * pi * a^4) / lambda^2, where:
        - a is the characteristic length of the dihedral
        - lambda is the wavelength of the incident radar wave
        
        Returns:
            float: Calculated characteristic size
        """
        c = 3e8  # Speed of light (m/s)
        size = ((self.rcs * self.wl ** 2) / (4 * np.pi)) ** 0.25  # Solving for a
        return size

    def generate_mesh(self, t=None):
        """
        Generate the dihedral corner reflector mesh.
        
        Args:
            t (float, optional): Time parameter for animations. Not used but required 
                               for interface compatibility. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated dihedral mesh
        """
        # Calculate size from RCS if provided
        if self.rcs is not None:
            size = self.calculate_dihedral_size_from_rcs()
            self.width, self.height = size, size
            
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        # Create two perpendicular planes
        plane1 = pv.Plane(
            center=(half_width / 2, 0, half_height / 2),
            direction=(0, 1, 0),
            i_size=self.width, 
            j_size=self.height
        )

        plane2 = pv.Plane(
            center=(0, half_width / 2, half_height / 2),
            direction=(1, 0, 0),
            i_size=self.width, 
            j_size=self.height
        )

        # Combine the two surfaces
        self.mesh = plane1 + plane2
        
        # Apply initial rotations
        self.mesh.rotate_z(-45, inplace=True)
        self.mesh.rotate_y(-90, inplace=True)
        
        # Convert orientation vector into a 4x4 transformation matrix
        normal = np.array(self.orientation, dtype=np.float64)
        normal /= np.linalg.norm(normal)  # Normalize the normal vector

        # Choose an arbitrary vector that is not parallel to the normal
        if abs(normal[0]) < abs(normal[1]):
            tangent = np.array([1, 0, 0], dtype=np.float64)
        else:
            tangent = np.array([0, 1, 0], dtype=np.float64)

        # Compute the right (X-axis) vector as the cross product
        x_axis = np.cross(tangent, normal)
        x_axis /= np.linalg.norm(x_axis)  # Normalize

        # Compute the new Y-axis vector as the cross product
        y_axis = np.cross(normal, x_axis)

        # Construct the 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, 0] = x_axis
        transform[:3, 1] = y_axis
        transform[:3, 2] = normal

        # Apply the transformation
        self.mesh.transform(transform)
        self.mesh = self.mesh.triangulate()
        return self.mesh

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)


class RoughPlane:
    """
    A class for generating rough surface plane meshes.
    
    This class creates plane meshes with controlled surface roughness using
    Gaussian height distributions and correlation lengths to simulate
    realistic rough surfaces.
    
    Attributes:
        i_size (float): Size in i-direction
        j_size (float): Size in j-direction
        num_i (int): Number of points in i-direction
        num_j (int): Number of points in j-direction
        orientation (list): Orientation vector [x, y, z]
        height_std_dev (float): Standard deviation of surface height variations
        roughness (float): Surface roughness parameter
        seed (int): Random seed for reproducibility
        wl (float): Wavelength parameter
        rcs (float): Radar cross section
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, i_size=1, j_size=1, num_i=50, num_j=50, orientation=[0, 0, 1],
                 height_std_dev=0.1, roughness=0.5, seed=None, wl=0.3, rcs=None):
        """
        Initialize a RoughPlane object.
        
        Args:
            i_size (float, optional): Size in i-direction. Defaults to 1.
            j_size (float, optional): Size in j-direction. Defaults to 1.
            num_i (int, optional): Number of points in i-direction. Defaults to 50.
            num_j (int, optional): Number of points in j-direction. Defaults to 50.
            orientation (list, optional): Orientation vector [x, y, z]. Defaults to [0, 0, 1].
            height_std_dev (float, optional): Standard deviation of height variations. Defaults to 0.1.
            roughness (float, optional): Surface roughness parameter. Defaults to 0.5.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            wl (float, optional): Wavelength parameter. Defaults to 0.3.
            rcs (float, optional): Radar cross section. Defaults to None.
        """
        self.i_size = i_size
        self.j_size = j_size
        self.num_i = num_i
        self.num_j = num_j
        self.orientation = orientation
        self.height_std_dev = height_std_dev
        self.roughness = roughness
        self.seed = seed
        self.wl = wl
        self.rcs = rcs
        self.mesh = None

    def generate_mesh(self, t=None):
        """
        Generate the rough plane mesh with controlled statistical properties.
        
        Args:
            t (float, optional): Time parameter for animations. Can be used as random seed. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated rough plane mesh
        """
        # Calculate the correlation length (L) based on height_std_dev and roughness
        correlation_length = self.height_std_dev / self.roughness

        # Set the random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)

        # Use time parameter as random seed if provided (for animations)
        if t is not None:
            np.random.seed(t)

        # Create a grid of points
        x = np.linspace(-self.i_size / 2, self.i_size / 2, self.num_i)
        y = np.linspace(-self.j_size / 2, self.j_size / 2, self.num_j)
        x, y = np.meshgrid(x, y)

        # Generate random heights with a Gaussian distribution
        heights = np.random.normal(0, self.height_std_dev, size=x.shape)

        # Apply a Gaussian filter to simulate the correlation length
        from scipy.ndimage import gaussian_filter
        smoothed_heights = gaussian_filter(heights, sigma=correlation_length)

        # Create the plane with the smoothed heights as z-values
        points = np.c_[x.ravel(), y.ravel(), smoothed_heights.ravel()]
        self.mesh = pv.PolyData(points)
        self.mesh = self.mesh.delaunay_2d()

        # Apply orientation transformation
        normal = np.array(self.orientation, dtype=np.float64)
        normal /= np.linalg.norm(normal)  # Normalize the normal vector

        # Choose an arbitrary vector that is not parallel to the normal
        if abs(normal[0]) < abs(normal[1]):
            tangent = np.array([1, 0, 0], dtype=np.float64)
        else:
            tangent = np.array([0, 1, 0], dtype=np.float64)

        # Compute the right (X-axis) vector as the cross product
        x_axis = np.cross(tangent, normal)
        x_axis /= np.linalg.norm(x_axis)  # Normalize

        # Compute the new Y-axis vector as the cross product
        y_axis = np.cross(normal, x_axis)

        # Construct the 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, 0] = x_axis
        transform[:3, 1] = y_axis
        transform[:3, 2] = normal

        # Apply the transformation
        self.mesh.transform(transform)
        return self.mesh.triangulate()

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)


class Capsule:
    """
    A class for generating capsule meshes (cylinder with hemispherical ends).
    
    This class creates capsule meshes with configurable radius, height,
    angular resolution, and orientation.
    
    Attributes:
        radius (float): Radius of the capsule
        height (float): Height of the cylindrical portion
        num_theta (int): Angular resolution
        orientation (list): Orientation vector [x, y, z]
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, radius=1, height=1, num_theta=45, orientation=[1, 0, 0]):
        """
        Initialize a Capsule object.
        
        Args:
            radius (float, optional): Radius of the capsule. Defaults to 1.
            height (float, optional): Height of the cylindrical portion. Defaults to 1.
            num_theta (int, optional): Angular resolution. Defaults to 45.
            orientation (list, optional): Orientation vector [x, y, z]. Defaults to [1, 0, 0].
        """
        self.radius = radius
        self.height = height
        self.num_theta = num_theta
        self.orientation = orientation

    def generate_mesh(self, t=None):
        """
        Generate the capsule mesh.
        
        Args:
            t (float, optional): Time parameter for animations. Not used but required 
                               for interface compatibility. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated capsule mesh
        """
        # Generate capsule mesh and triangulate
        self.mesh = pv.Capsule(
            radius=self.radius,
            cylinder_length=self.height,
            resolution=self.num_theta,
            direction=self.orientation
        )
        self.mesh = self.mesh.triangulate()
        return self.mesh

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)


class Plane:
    """
    A class for generating planar meshes.
    
    This class creates planar meshes with configurable dimensions, resolution,
    and orientation. The dimensions can be automatically calculated from a given RCS.
    
    Attributes:
        i_size (float): Size in i-direction
        j_size (float): Size in j-direction
        num_i (int): Number of divisions in i-direction
        num_j (int): Number of divisions in j-direction
        orientation (list): Orientation vector [x, y, z]
        wl (float): Wavelength for RCS calculations
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, i_size=1, j_size=1, num_i=10, num_j=10, orientation=[0, 0, 1], wl=0.3, rcs=None):
        """
        Initialize a Plane object.
        
        Args:
            i_size (float, optional): Size in i-direction. Defaults to 1.
            j_size (float, optional): Size in j-direction. Defaults to 1.
            num_i (int, optional): Number of divisions in i-direction. Defaults to 10.
            num_j (int, optional): Number of divisions in j-direction. Defaults to 10.
            orientation (list, optional): Orientation vector [x, y, z]. Defaults to [0, 0, 1].
            wl (float, optional): Wavelength for RCS calculations. Defaults to 0.3.
            rcs (float, optional): Radar cross section to calculate size from. Defaults to None.
        """
        self.i_size = i_size
        self.j_size = j_size
        self.num_i = num_i
        self.num_j = num_j
        self.orientation = orientation
        
        # If RCS is provided, calculate dimensions from it (creates a square)
        if rcs is not None:
            self.wl = wl
            self.i_size = self.calculate_length_from_rcs(rcs)
            self.j_size = self.i_size

    def calculate_length_from_rcs(self, rcs):
        """
        Calculate the length of a plane side given its RCS value.
        
        Using the formula: RCS = 4π/3 * a^4/λ^2, where a is the side length
        and λ is the wavelength.
        
        Args:
            rcs (float): Radar cross section value
            
        Returns:
            float: Calculated side length
        """
        length = np.power(rcs * self.wl**2 / (4 * np.pi), 1/4)
        return length

    def generate_mesh(self, t=None):
        """
        Generate the plane mesh.
        
        Args:
            t (float, optional): Time parameter for animations. Not used but required 
                               for interface compatibility. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated plane mesh
        """
        # Generate plane mesh and triangulate
        self.mesh = pv.Plane(
            i_size=self.i_size,
            j_size=self.j_size,
            i_resolution=self.num_i,
            j_resolution=self.num_j,
            direction=self.orientation
        )
        self.mesh = self.mesh.triangulate()
        return self.mesh

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)


class CornerReflector:
    """
    A class for generating corner reflector meshes.
    
    This class creates corner reflector meshes with either triangular or square faces.
    The size can be automatically calculated from a given radar cross section (RCS).
    
    Attributes:
        length (float): Length of the corner reflector sides
        rcs (float): Radar cross section
        wl (float): Wavelength for RCS calculations
        is_square (bool): Whether to create square or triangular faces
        orientation (str): Bore sight direction of the reflector
        mesh (pv.PolyData): Generated PyVista mesh
    """
    
    def __init__(self, length=1, rcs=None, wl=0.3, orientation='x', is_square=False):
        """
        Initialize a CornerReflector object.
        
        Args:
            length (float, optional): Length of the corner reflector sides. Defaults to 1.
            rcs (float, optional): Radar cross section to calculate size from. Defaults to None.
            wl (float, optional): Wavelength for RCS calculations. Defaults to 0.3.
            orientation (str, optional): Bore sight direction ('x', 'y', 'z', '-x', '-y', '-z'). Defaults to 'x'.
            is_square (bool, optional): Whether to create square or triangular faces. Defaults to False.
        """
        self.length = length
        self.rcs = rcs
        self.wl = wl
        self.is_square = is_square
        
        # Calculate length from RCS if provided
        if rcs is not None:
            self.length = self.calculate_length_from_rcs(rcs)
            print(f'Length calculated from RCS {rcs}: {self.length}')
            
        # Orientation is the bore sight direction of the reflector, or looking into the throat
        self.orientation = orientation

    def calculate_length_from_rcs(self, rcs):
        """
        Calculate the length of a corner reflector side given its RCS value.
        
        Uses different formulas for square vs triangular corner reflectors:
        - Square: RCS = 12π * a^4/λ^2
        - Triangular: RCS = 4π/3 * a^4/λ^2
        where a is the side length and λ is the wavelength.
        
        Args:
            rcs (float): Radar cross section value
            
        Returns:
            float: Calculated side length
        """
        if self.is_square:
            length = np.power(rcs * self.wl**2 / (12 * np.pi), 1/4)
        else:
            length = np.power(rcs * 3 * self.wl**2 / (4 * np.pi), 1/4)
        return length

    def create_triangular_face(self, center, direction):
        """
        Create a triangular face for the corner reflector.
        
        Args:
            center (array): Center position of the face
            direction (array): Direction vector for the face orientation
            
        Returns:
            pv.PolyData: Triangular face mesh
        """
        # Create three points for the triangle based on direction
        if direction[0] == 1:  # YZ plane
            points = np.array([
                [center[0], 0, 0],
                [center[0], self.length, 0],
                [center[0], 0, self.length]
            ], dtype=np.float32)
        elif direction[1] == 1:  # XZ plane
            points = np.array([
                [0, center[1], 0],
                [self.length, center[1], 0],
                [0, center[1], self.length]
            ], dtype=np.float32)
        else:  # XY plane
            points = np.array([
                [0, 0, center[2]],
                [self.length, 0, center[2]],
                [0, self.length, center[2]]
            ], dtype=np.float32)

        # Create a single triangle face
        face = np.array([3, 0, 1, 2])
        return pv.PolyData(points, faces=[face]).triangulate()

    def create_square_face(self, center, direction):
        """
        Create a square face using PyVista's Plane.
        
        Args:
            center (array): Center position of the face
            direction (array): Direction vector for the face orientation
            
        Returns:
            pv.PolyData: Square face mesh
        """
        mesh = pv.Plane(
            center=center,
            direction=direction,
            i_size=self.length,
            j_size=self.length,
            i_resolution=1,
            j_resolution=1
        )
        return mesh.triangulate()

    def rotate_mesh(self, mesh, orientation):
        """
        Rotate the mesh based on the specified orientation.
        
        Args:
            mesh (pv.PolyData): Mesh to rotate
            orientation (str): Orientation string ('x', 'y', 'z', '-x', '-y', '-z', 'none')
            
        Returns:
            pv.PolyData: Rotated mesh
        """
        orientation = orientation.lower()
        
        if 'x' in orientation:
            # Rotate to point along x-axis
            mesh.rotate_z(-45, inplace=True)
            mesh.rotate_y(35.2644, inplace=True)  # arctan(1/√2)
            if '-' in orientation:
                mesh.rotate_z(180, inplace=True)
        elif 'y' in orientation:
            # Rotate to point along y-axis
            mesh.rotate_z(45, inplace=True)
            mesh.rotate_x(-35.2644, inplace=True)
            if '-' in orientation:
                mesh.rotate_z(180, inplace=True)
        elif 'z' in orientation:
            # Rotate to point along z-axis
            mesh.rotate_z(45, inplace=True)
            mesh.rotate_x(54.7356, inplace=True)  # 90 - arctan(1/√2)
            if '-' in orientation:
                mesh.rotate_y(180, inplace=True)
        # 'none' orientation requires no rotation
        
        return mesh

    def create_corner_reflector(self):
        """
        Create a corner reflector with the given parameters.
        
        Returns:
            pv.PolyData: Complete corner reflector mesh
        """
        # Define the centers and directions for each face
        if self.is_square:
            face_creator = self.create_square_face
            # Centers for square faces (offset by half length)
            centers = np.array([
                (-self.length / 2, 0, 0), 
                (0, -self.length / 2, 0), 
                (0, 0, -self.length / 2)
            ])
        else:
            face_creator = self.create_triangular_face
            # Centers for triangular faces (all at origin)
            centers = np.array([
                (0, 0, 0), 
                (0, 0, 0), 
                (0, 0, 0)
            ])

        # Direction vectors for the three orthogonal faces
        directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        # Create the three faces
        faces = [face_creator(center, np.array(direction))
                 for center, direction in zip(centers, directions)]

        # Combine the faces into a single mesh
        corner_reflector = faces[0] + faces[1] + faces[2]

        # Apply rotation based on orientation
        corner_reflector = self.rotate_mesh(corner_reflector, self.orientation)

        return corner_reflector

    def generate_mesh(self, t=None):
        """
        Generate the corner reflector mesh.
        
        Args:
            t (float, optional): Time parameter for animations. Not used but required 
                               for interface compatibility. Defaults to None.
                               
        Returns:
            pv.PolyData: Generated corner reflector mesh
        """
        self.mesh = self.create_corner_reflector()
        return self.mesh

    def save_mesh(self, output_path):
        """
        Save the mesh to a file.
        
        Args:
            output_path (str): Path where the mesh file will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)


if __name__ == "__main__":
    """
    Example usage and testing of all primitive classes.
    
    This section demonstrates how to create different types of primitives,
    including examples with RCS-based sizing and visualization.
    """
    
    # Examples of primitive creation
    # Note: Some primitives can use rcs=Value and wl=c0/freq to calculate dimensions
    # These parameters will always override manually specified dimensions
    
    # Create various primitive examples
    prim = Sphere(radius=1, num_theta=45, num_phi=45)
    prim = Cube(x_length=1, y_length=1, z_length=1)
    prim = Cylinder(radius=1, height=10, num_theta=45, orientation=[0, 0, 1])
    prim = Capsule(radius=1, height=10, num_theta=45, orientation=[0, 0, 1])
    prim = Plane(i_size=1, j_size=1, num_i=10, num_j=10, orientation=[1, 0, 0])
    prim = CornerReflector(length=1, rcs=10, wl=3e8/76.5e9, orientation='x', is_square=False)
    prim = RoughPlane(
        i_size=50, j_size=50, num_i=100, num_j=100, orientation=[0, 0, 1],
        height_std_dev=0.05, roughness=0.5, seed=None, wl=0.3, rcs=None
    )

    # Generate the mesh
    prim.generate_mesh()
    
    # Optional: Save the mesh to file
    # prim.save_mesh('cr.stl')
    
    # Visualization using PyVista
    plotter = pv.Plotter()
    plotter.parallel_projection = True
    plotter.show_grid()

    # Set up the colormap for visualization
    colormap = "bone"  # Other options: "blues", "coolwarm", etc.
    plotter.add_mesh(prim.generate_mesh(), show_edges=True, cmap=colormap)
    plotter.show()
    plotter.close()