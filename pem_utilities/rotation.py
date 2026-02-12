"""
Rotation Utilities Module

This module provides a comprehensive set of utilities for handling 3D rotations in various formats.
It supports conversions between rotation matrices, Euler angles, yaw-pitch-roll representations,
and vector rotations. The module is particularly useful for electromagnetic simulation and 
antenna positioning calculations.

Key Features:
    - Yaw/Pitch/Roll to rotation matrix conversion and vice versa
    - Euler angle to rotation matrix conversion with configurable rotation orders
    - Vector rotation and normalization utilities
    - Camera "look-at" matrix generation for visualization
    - Support for both degrees and radians
    - Batch processing of multiple rotations

Created on Fri Jan 26 15:00:00 2024
@author: asligar
"""

import numpy as np
import math as m
from scipy.spatial.transform import Rotation

def spherical_to_cartesian(theta, phi):
    """Converts spherical (theta, phi) to Cartesian coordinates (r=1)."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def theta_phi_to_rot(theta, phi,deg=True):
    """
    Returns a scipy Rotation object that aligns the X-axis
    toward the origin from a spherical coordinate position (theta, phi).
    """

    if deg:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    # Step 1: Compute the viewing direction (from position to origin)
    view_dir = -spherical_to_cartesian(theta, phi)
    x_axis = view_dir / np.linalg.norm(view_dir)

    # Step 2: Define a global up vector
    up = np.array([0, 0, 1])
    if np.allclose(x_axis, up) or np.allclose(x_axis, -up):
        up = np.array([0, 1, 0])  # fallback if aligned with Z

    # Step 3: Compute Y and Z to form an orthonormal basis
    y_axis = np.cross(up, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Step 4: Stack into rotation matrix (columns = new axes)
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # Step 5: Return as a scipy Rotation object
    return Rotation.from_matrix(rot_matrix).as_matrix()

def yaw_pitch_roll_to_rotmat(yaw=0, pitch=0, roll=0, deg=True):
    """
    Convert yaw, pitch, and roll angles to a 3x3 rotation matrix.
    
    This function implements the ZYX Euler angle convention (Tait-Bryan angles),
    which is commonly used in aerospace applications. The rotation order is:
    1. Yaw (rotation about Z-axis)
    2. Pitch (rotation about Y-axis) 
    3. Roll (rotation about X-axis)

    Parameters
    ----------
    yaw : float, default=0
        Rotation angle about the Z-axis (yaw angle).
    pitch : float, default=0
        Rotation angle about the Y-axis (pitch angle).
    roll : float, default=0
        Rotation angle about the X-axis (roll angle).
    deg : bool, default=True
        If True, input angles are interpreted as degrees.
        If False, input angles are interpreted as radians.

    Returns
    -------
    rot : numpy.ndarray
        3x3 rotation matrix representing the combined rotation.
        Shape: (3, 3)

    Notes
    -----
    The rotation matrix is constructed using the following formula:
    R = R_z(yaw) * R_y(pitch) * R_x(roll)
    
    Where:
    - R_z is rotation about Z-axis
    - R_y is rotation about Y-axis  
    - R_x is rotation about X-axis

    Examples
    --------
    >>> # 90-degree yaw rotation
    >>> rot = yaw_pitch_roll_to_rotmat(yaw=90, pitch=0, roll=0)
    >>> print(rot.shape)
    (3, 3)
    
    >>> # Using radians
    >>> import numpy as np
    >>> rot = yaw_pitch_roll_to_rotmat(yaw=np.pi/2, deg=False)
    """
    # Initialize as identity matrix
    rot = np.eye(3)

    # Convert to radians if input is in degrees
    if deg:
        c1 = m.cos(m.radians(roll))   # cos(roll)
        c2 = m.cos(m.radians(pitch))  # cos(pitch)
        c3 = m.cos(m.radians(yaw))    # cos(yaw)

        s1 = m.sin(m.radians(roll))   # sin(roll)
        s2 = m.sin(m.radians(pitch))  # sin(pitch)
        s3 = m.sin(m.radians(yaw))    # sin(yaw)
    else:
        c1 = m.cos(roll)   # cos(roll)
        c2 = m.cos(pitch)  # cos(pitch)
        c3 = m.cos(yaw)    # cos(yaw)

        s1 = m.sin(roll)   # sin(roll)
        s2 = m.sin(pitch)  # sin(pitch)
        s3 = m.sin(yaw)    # sin(yaw)

    # Construct rotation matrix elements
    # First row
    rot[0, 0] = c2 * c1
    rot[0, 1] = s3 * s2 * c1 - c3 * s1
    rot[0, 2] = c3 * s2 * c1 + s3 * s1

    # Second row
    rot[1, 0] = c2 * s1
    rot[1, 1] = s3 * s2 * s1 + c3 * c1
    rot[1, 2] = c3 * s2 * s1 - s3 * c1

    # Third row
    rot[2, 0] = -s2
    rot[2, 1] = s3 * c2
    rot[2, 2] = c3 * c2

    return rot


def rotmat_to_yaw_pitch_roll(rot, deg=True):
    """
    Extract yaw, pitch, and roll angles from a rotation matrix.
    
    This function performs the inverse operation of yaw_pitch_roll_to_rotmat(),
    converting a 3x3 rotation matrix back to Euler angles using the ZYX convention.

    Parameters
    ----------
    rot : array_like
        3x3 rotation matrix or array of rotation matrices.
        Shape: (3, 3) or (N, 3, 3) for multiple matrices.
    deg : bool, default=True
        If True, output angles are in degrees.
        If False, output angles are in radians.

    Returns
    -------
    list
        List containing [yaw, pitch, roll] angles.
        - yaw : float or numpy.ndarray - Rotation about Z-axis
        - pitch : float or numpy.ndarray - Rotation about Y-axis  
        - roll : float or numpy.ndarray - Rotation about X-axis

    Notes
    -----
    This function can handle multiple rotation matrices simultaneously.
    The input is automatically reshaped to ensure proper dimensionality.
    
    The extraction uses the following relationships:
    - yaw = atan2(R[1,0], R[0,0])
    - pitch = acos(R[2,2])
    - roll = atan2(R[2,1], R[2,2])

    Examples
    --------
    >>> import numpy as np
    >>> rot = np.eye(3)  # Identity matrix
    >>> yaw, pitch, roll = rotmat_to_yaw_pitch_roll(rot)
    >>> print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")
    """
    # Set conversion factor based on desired output units
    if deg:
        convert = 180 / np.pi  # Convert radians to degrees
    else:
        convert = 1  # Keep in radians

    # Ensure proper dimensionality (minimum 3D array)
    rot = np.array(rot, ndmin=3)

    # Extract angles using inverse trigonometric functions
    yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0]) * convert
    pitch = np.arccos(rot[:, 2, 2]) * convert
    roll = np.arctan2(rot[:, 2, 1], rot[:, 2, 2]) * convert

    return [yaw, pitch, roll]


def euler_to_rot(phi=0., theta=0., psi=0., order='zyz', deg=True):
    """
    Convert Euler angles to a rotation matrix using specified rotation order.
    
    This function provides flexible Euler angle to rotation matrix conversion
    with support for various rotation sequences and both intrinsic/extrinsic rotations.

    Parameters
    ----------
    phi : float or array_like, default=0.0
        First Euler angle.
    theta : float or array_like, default=0.0
        Second Euler angle.
    psi : float or array_like, default=0.0
        Third Euler angle.
    order : str, default='zyz'
        Rotation sequence specification. Case-sensitive:
        - Lowercase (e.g., 'zyz'): Extrinsic rotations (fixed axes)
        - Uppercase (e.g., 'ZYZ'): Intrinsic rotations (moving axes)
        Common sequences: 'xyz', 'zyx', 'zyz', 'XYZ', 'ZYX', 'ZYZ'
    deg : bool, default=True
        If True, input angles are in degrees.
        If False, input angles are in radians.

    Returns
    -------
    rot : numpy.ndarray
        Rotation matrix or array of rotation matrices.
        Shape: (3, 3) for single rotation, (N, 3, 3) for multiple rotations.

    Notes
    -----
    **Important**: The capitalization of the rotation order is critical:
    
    - **Extrinsic rotations** (lowercase): Rotations about fixed, original axes.
      Each subsequent rotation is performed about the original coordinate system.
      Used by PyVista and some visualization libraries.
      
    - **Intrinsic rotations** (uppercase): Rotations about moving axes.
      Each rotation is performed about the current (rotated) coordinate system.
      Used by STK, AEDT, and many engineering applications.
    
    The function supports broadcasting: if arrays of different lengths are provided,
    scalar values are automatically replicated to match the longest array.

    Examples
    --------
    >>> # Single rotation using ZYZ Euler angles
    >>> rot = euler_to_rot(phi=30, theta=45, psi=60, order='zyz')
    >>> print(rot.shape)
    (3, 3)
    
    >>> # Multiple rotations with broadcasting
    >>> phi_vals = [0, 30, 60]
    >>> rot_array = euler_to_rot(phi=phi_vals, theta=45, psi=0)
    >>> print(rot_array.shape)
    (3, 3, 3)
    
    >>> # Intrinsic vs Extrinsic comparison
    >>> rot_ext = euler_to_rot(30, 45, 60, order='zyz')  # Extrinsic
    >>> rot_int = euler_to_rot(30, 45, 60, order='ZYZ')  # Intrinsic
    """
    # Convert scalar inputs to arrays for uniform processing
    if isinstance(phi, (float, int)):
        phi = np.array([phi])
    if isinstance(theta, (float, int)):
        theta = np.array([theta])
    if isinstance(psi, (float, int)):
        psi = np.array([psi])

    # Determine the maximum array length for broadcasting
    length_of_array = max(len(phi), len(theta), len(psi))
    
    # Handle array length mismatches through broadcasting
    if len(phi) != len(theta) or len(phi) != len(psi) or len(theta) != len(psi):
        which_is_longest = np.argmax([len(phi), len(theta), len(psi)])
        
        if which_is_longest == 0:  # phi is longest
            if len(theta) != 1 and len(theta) != len(phi):
                raise Exception('Theta must be a scalar or same length as phi')
            elif len(theta) == 1:
                theta = np.repeat(theta, len(phi))
            if len(psi) != 1 and len(psi) != len(phi):
                raise Exception('Psi must be a scalar or same length as phi')
            elif len(psi) == 1:
                psi = np.repeat(psi, len(phi))
                
        elif which_is_longest == 1:  # theta is longest
            if len(phi) != 1 and len(phi) != len(theta):
                raise Exception('Phi must be a scalar or same length as theta')
            elif len(phi) == 1:
                phi = np.repeat(phi, len(theta))
            if len(psi) != 1 and len(psi) != len(theta):
                raise Exception('Psi must be a scalar or same length as theta')
            elif len(psi) == 1:
                psi = np.repeat(psi, len(theta))
                
        else:  # psi is longest
            if len(phi) != 1 and len(phi) != len(psi):
                raise Exception('Phi must be a scalar or same length as psi')
            elif len(phi) == 1:
                phi = np.repeat(phi, len(psi))
            if len(theta) != 1 and len(theta) != len(psi):
                raise Exception('Theta must be a scalar or same length as psi')
            elif len(theta) == 1:
                theta = np.repeat(theta, len(psi))

    # Process each set of Euler angles
    all_angs = zip(phi, theta, psi)
    rot = np.zeros((length_of_array, 3, 3))
    
    for n, (phi_val, theta_val, psi_val) in enumerate(all_angs):
        # Use scipy's robust rotation implementation
        rot_temp = Rotation.from_euler(order, [phi_val, theta_val, psi_val], degrees=deg)
        rot[n] = rot_temp.as_matrix()
    
    # Return single matrix if only one rotation was computed
    if length_of_array == 1:
        rot = rot[0]
        
    return rot


def rot_to_euler(rot, order='zyz', deg=True):
    """
    Convert a rotation matrix to Euler angles using specified rotation order.
    
    This function extracts Euler angles from a rotation matrix using scipy's
    robust implementation, which handles edge cases and singularities properly.

    Parameters
    ----------
    rot : array_like
        3x3 rotation matrix to convert.
        Shape: (3, 3)
    order : str, default='zyz'
        Rotation sequence specification (same conventions as euler_to_rot).
        Case-sensitive for intrinsic vs extrinsic rotations.
    deg : bool, default=True
        If True, output angles are in degrees.
        If False, output angles are in radians.

    Returns
    -------
    euler_angles : numpy.ndarray
        Array of Euler angles [phi, theta, psi] in the specified order.
        Shape: (3,)

    Examples
    --------
    >>> import numpy as np
    >>> rot = np.eye(3)  # Identity matrix
    >>> angles = rot_to_euler(rot, order='zyz')
    >>> print(f"Euler angles: {angles}")
    """
    rot_obj = Rotation.from_matrix(rot)
    euler_angles = rot_obj.as_euler(order, degrees=deg)
    return euler_angles


def rotate_vector_from_rot(vec, rot, order='zyz', deg=True):
    """
    Rotate a vector using a rotation matrix.
    
    This function applies a rotation matrix to transform a vector from one
    coordinate system to another. Useful for transforming direction vectors,
    position vectors, or any 3D quantity.

    Parameters
    ----------
    vec : array_like
        3D vector to be rotated.
        Shape: (3,) or (N, 3) for multiple vectors.
    rot : array_like
        3x3 rotation matrix to apply.
        Shape: (3, 3)
    order : str, default='zyz'
        Rotation order (maintained for consistency, not used in this function).
    deg : bool, default=True
        Degree flag (maintained for consistency, not used in this function).

    Returns
    -------
    out_vector : numpy.ndarray
        Rotated vector(s).
        Shape: Same as input vector.

    Examples
    --------
    >>> import numpy as np
    >>> vec = [1, 0, 0]  # Unit vector along X-axis
    >>> rot = yaw_pitch_roll_to_rotmat(yaw=90)  # 90-degree rotation about Z
    >>> rotated_vec = rotate_vector_from_rot(vec, rot)
    >>> print(f"Rotated vector: {rotated_vec}")
    """
    rot_obj = Rotation.from_matrix(rot)
    out_vector = rot_obj.apply(vec)
    return out_vector


def vec_to_rot(vec):
    """
    Convert a rotation vector to a rotation matrix.
    
    A rotation vector represents a rotation as a vector whose direction
    indicates the axis of rotation and whose magnitude indicates the
    angle of rotation (in radians).

    Parameters
    ----------
    vec : array_like
        Rotation vector in axis-angle representation.
        Shape: (3,) where magnitude is angle in radians.

    Returns
    -------
    rot : numpy.ndarray
        3x3 rotation matrix equivalent to the rotation vector.
        Shape: (3, 3)

    Notes
    -----
    The rotation vector uses the axis-angle representation where:
    - Direction of vector = axis of rotation
    - Magnitude of vector = angle of rotation (radians)
    
    This is also known as the exponential map representation of rotations.

    Examples
    --------
    >>> import numpy as np
    >>> vec = [0, 0, np.pi/2]  # 90-degree rotation about Z-axis
    >>> rot = vec_to_rot(vec)
    >>> print(rot.shape)
    (3, 3)
    """
    rot_obj = Rotation.from_rotvec(vec)
    rot = rot_obj.as_matrix()
    return rot


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Compute the rotation matrix that aligns one vector with another.
    
    This function finds the rotation matrix that, when applied to vec1,
    will align it with vec2. This is useful for orienting objects or
    coordinate systems based on desired directions.

    Parameters
    ----------
    vec1 : array_like
        Source 3D vector to be aligned.
        Shape: (3,)
    vec2 : array_like
        Target 3D vector to align with.
        Shape: (3,)

    Returns
    -------
    rotation_matrix : numpy.ndarray
        3x3 rotation matrix that transforms vec1 to align with vec2.
        Shape: (3, 3)

    Notes
    -----
    The algorithm uses Rodrigues' rotation formula:
    R = I + [v]× + [v]²× * (1-cos(θ))/sin²(θ)
    
    Where:
    - v is the cross product of normalized vec1 and vec2
    - θ is the angle between the vectors
    - [v]× is the skew-symmetric matrix of v
    - I is the identity matrix

    The vectors are automatically normalized, so only their directions matter.

    Examples
    --------
    >>> import numpy as np
    >>> vec1 = [1, 0, 0]  # X-axis
    >>> vec2 = [0, 1, 0]  # Y-axis
    >>> rot = rotation_matrix_from_vectors(vec1, vec2)
    >>> # This should give a 90-degree rotation about Z-axis
    """
    # Normalize both vectors
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    
    # Compute cross product (rotation axis) and dot product (cosine of angle)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    # Construct skew-symmetric matrix from cross product
    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    
    # Apply Rodrigues' formula
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix


def vec_length(v):
    """
    Calculate the Euclidean length (magnitude) of a vector.
    
    This function computes the L2 norm of a vector, which is the square root
    of the sum of squared components.

    Parameters
    ----------
    v : numpy.ndarray
        Input vector of any dimension.

    Returns
    -------
    length : float
        The Euclidean length of the vector.

    Notes
    -----
    For a 3D vector [x, y, z], the length is sqrt(x² + y² + z²).
    This is equivalent to np.linalg.norm(v) but implemented explicitly.

    Examples
    --------
    >>> import numpy as np
    >>> v = [3, 4, 0]  # 3-4-5 triangle
    >>> length = vec_length(v)
    >>> print(f"Length: {length}")  # Should be 5.0
    """
    return np.sqrt(sum(i ** 2 for i in v))


def normalize(v):
    """
    Normalize a vector to unit length.
    
    This function scales a vector so that its magnitude becomes 1.0,
    preserving its direction but standardizing its length.

    Parameters
    ----------
    v : array_like
        Input vector to be normalized.

    Returns
    -------
    normalized_vector : numpy.ndarray
        Unit vector in the same direction as the input.
        If input has zero norm, returns the original vector unchanged.

    Notes
    -----
    Normalization is performed by dividing each component by the vector's magnitude.
    Zero vectors are returned unchanged to avoid division by zero.

    Examples
    --------
    >>> import numpy as np
    >>> v = [3, 4, 0]
    >>> v_norm = normalize(v)
    >>> print(f"Normalized: {v_norm}")  # Should be [0.6, 0.8, 0.0]
    >>> print(f"Length: {np.linalg.norm(v_norm)}")  # Should be 1.0
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v  # Return original vector if it has zero magnitude
    return v / norm


def look_at(eye, target, correct_rotation_matrix=True, alignment_vector=None):
    """
    Generate a rotation matrix for a camera looking at a target point.
    
    This function creates a rotation matrix that orients a coordinate system
    (typically a camera) so that its X-axis points toward a target location.
    This is commonly used in computer graphics and visualization applications.

    Parameters
    ----------
    eye : numpy.ndarray
        3D position of the camera/observer.
        Shape: (3,)
    target : numpy.ndarray  
        3D position of the target point to look at.
        Shape: (3,)
    correct_rotation_matrix : bool, default=True
        If True, uses scipy to correct any numerical imperfections in the
        computed rotation matrix. May cause issues with ambiguous angles.
    alignment_vector : numpy.ndarray, optional
        3D vector defining the "up" direction for orienting the Y and Z axes.
        Default is [0, 0, 1] (positive Z-axis).
        Shape: (3,)

    Returns
    -------
    rot_matrix : numpy.ndarray
        3x3 rotation matrix representing the camera orientation.
        - X-axis points from eye toward target
        - Y-axis is perpendicular to X-axis and alignment vector
        - Z-axis completes the right-handed coordinate system
        Shape: (3, 3)

    Notes
    -----
    The coordinate system is constructed as follows:
    1. X-axis = normalized(target - eye)  [pointing direction]
    2. Y-axis = normalized(alignment_vector × X-axis)  [right direction]
    3. Z-axis = X-axis × Y-axis  [up direction]
    
    This creates a right-handed coordinate system with X pointing toward the target.
    
    Special cases:
    - If eye and target are identical, X-axis defaults to [1, 0, 0]
    - If alignment vector is parallel to viewing direction, Y-axis defaults to [0, 1, 0]

    Examples
    --------
    >>> import numpy as np
    >>> eye = [0, 0, 0]      # Camera at origin
    >>> target = [1, 0, 0]   # Looking toward +X
    >>> rot = look_at(eye, target)
    >>> print("X-axis (forward):", rot[:, 0])  # Should be [1, 0, 0]
    
    >>> # Camera looking diagonally upward
    >>> eye = [0, 0, 0]
    >>> target = [1, 1, 1]
    >>> rot = look_at(eye, target)
    """
    # Ensure inputs are numpy arrays
    if not isinstance(eye, np.ndarray):
        eye = np.array(eye)
    if not isinstance(target, np.ndarray):
        target = np.array(target)
    
    # Compute forward direction (X-axis points toward target)
    axis_x = normalize(target - eye)
    
    # Handle degenerate case where eye and target are the same
    if vec_length(axis_x) == 0:
        axis_x = np.array([1, 0, 0])  # Default forward direction
    
    # Set default alignment vector if not provided
    if alignment_vector is None:
        alignment_vector = np.array([0, 0, 1])  # Default "up" is +Z
    
    # Compute right direction (Y-axis perpendicular to forward and up)
    axis_y = np.cross(alignment_vector, axis_x)
    
    # Handle degenerate case where alignment vector is parallel to forward
    if vec_length(axis_y) == 0:
        axis_y = np.array([0, 1, 0])  # Default right direction
    else:
        axis_y = normalize(axis_y)
    
    # Compute actual up direction (Z-axis completes right-handed system)
    axis_z = np.cross(axis_x, axis_y)
    axis_z = normalize(axis_z)
    
    # Construct rotation matrix with axes as columns
    rot_matrix = np.column_stack([axis_x, axis_y, axis_z])
    
    # Optionally correct numerical errors using scipy
    if correct_rotation_matrix:
        r = Rotation.from_matrix(rot_matrix)
        return r.as_matrix()
    else:
        return rot_matrix
