import numpy as np
import os
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.post_processing_radar_imaging import isar_3d

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 


# data_cube = np.load(os.path.join(paths.output, 'data_cube.npy'))
phi_domain = np.load(os.path.join(paths.output, 'phi_domain.npy'))
theta_domain = np.load(os.path.join(paths.output, 'theta_domain.npy'))
Freq = np.load(os.path.join(paths.output, 'frequency_domain.npy'))

model_path = os.path.join(paths.models,'cessna.stl')
cad = pv.read(model_path)
data_cube = np.load(os.path.join(r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Scripting\github\perceive_em\output', 'bistatic_rcs_results.npy'))

# data_cube = np.moveaxis(data_cube, 0,2)  # Ensure data_cube is in the correct shape (cross-range, range, cross-range)
y,x,z,isar_image = isar_3d(data_cube, freq_domain=Freq, phi_domain=phi_domain, theta_domain=theta_domain, function='abs', size=(128,128,128), window='hann')
data_flat = 20*np.log10(np.abs(isar_image.flatten()))
# create a plot of the all the data in the isar_image array and a 2D CDF plot of the data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(20*np.log10(np.abs(isar_image[:, :, isar_image.shape[2] // 2])), cmap='gray')
plt.title('ISAR Image Slice at Middle Cross-Range')
plt.colorbar(label='Intensity')
plt.subplot(1, 2, 2)
plt.hist(data_flat, bins=100, density=True, cumulative=True, color='blue', alpha=0.7)
plt.title('Cumulative Distribution Function (CDF) of ISAR Image Intensities')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Probability')
plt.tight_layout()
plt.show()

options = {}
# options['opacity'] = 'linear'
# options['mapper'] = 'smart'


plotter = pv.Plotter()
# pyvistafunc = plotter.add_volume
pyvistafunc = plotter.add_mesh
plotter.add_mesh(cad, color='grey', show_edges=True, smooth_shading=True)

# Create 3D isosurface plot
print(f"ISAR image shape: {isar_image.shape}")
print(f"ISAR image min/max: {np.min(isar_image):.6f} / {np.max(isar_image):.6f}")

min_val = np.min(data_flat)
max_val = np.max(data_flat)

actor = pv.RectilinearGrid(np.unique(x),np.unique(y),np.unique(z))
actor['isar'] = data_flat
actor = actor.contour(isosurfaces=10,rng=[max_val-30, max_val])
pyvistafunc(actor,**options)

plotter.show()

# Choose an isosurface threshold (e.g., 50% of max value)
threshold = 0.05 * np.max(isar_image)
print(f"Using isosurface threshold: {threshold:.6f}")

# Extract isosurface using marching cubes
try:
    vertices, faces, normals, values = measure.marching_cubes(isar_image, level=threshold)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the isosurface
    mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          triangles=faces, alpha=0.7, cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel('Cross-Range 1 (samples)')
    ax.set_ylabel('Range (samples)')
    ax.set_zlabel('Cross-Range 2 (samples)')
    ax.set_title(f'3D ISAR Isosurface (threshold = {threshold:.3e})')
    
    # Add colorbar
    plt.colorbar(mesh, shrink=0.5, aspect=5)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    output_filename = os.path.join(paths.output, 'isar_3d_isosurface.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"3D isosurface plot saved to: {output_filename}")
    
except Exception as e:
    print(f"Could not create isosurface at threshold {threshold}: {e}")
    print("Trying with a lower threshold...")
    
    # Try with a lower threshold (10% of max)
    threshold_low = 0.1 * np.max(isar_image)
    try:
        vertices, faces, normals, values = measure.marching_cubes(isar_image, level=threshold_low)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                              triangles=faces, alpha=0.7, cmap='viridis')
        
        ax.set_xlabel('Cross-Range 1 (samples)')
        ax.set_ylabel('Range (samples)')
        ax.set_zlabel('Cross-Range 2 (samples)')
        ax.set_title(f'3D ISAR Isosurface (threshold = {threshold_low:.3e})')
        
        plt.colorbar(mesh, shrink=0.5, aspect=5)
        ax.set_box_aspect([1,1,1])
        
        plt.tight_layout()
        plt.show()
        
        output_filename = os.path.join(paths.output, 'isar_3d_isosurface_low_threshold.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"3D isosurface plot saved to: {output_filename}")
        
    except Exception as e2:
        print(f"Could not create isosurface even with lower threshold: {e2}")
        
        # Alternative: Create a simple 3D scatter plot of high-intensity points
        print("Creating 3D scatter plot of high-intensity points instead...")
        
        # Find points above threshold
        high_intensity_mask = isar_image > threshold_low
        z_coords, y_coords, x_coords = np.where(high_intensity_mask)
        intensities = isar_image[high_intensity_mask]
        
        if len(x_coords) > 0:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                               c=intensities, cmap='viridis', alpha=0.6)
            
            ax.set_xlabel('Cross-Range 1 (samples)')
            ax.set_ylabel('Range (samples)')
            ax.set_zlabel('Cross-Range 2 (samples)')
            ax.set_title('3D ISAR High-Intensity Points')
            
            plt.colorbar(scatter, shrink=0.5, aspect=5)
            ax.set_box_aspect([1,1,1])
            
            plt.tight_layout()
            plt.show()
            
            output_filename = os.path.join(paths.output, 'isar_3d_scatter.png')
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"3D scatter plot saved to: {output_filename}")
        else:
            print("No high-intensity points found for visualization.")

