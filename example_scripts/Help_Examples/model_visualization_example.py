"""
Model Visualization Example Script
=================================

This script demonstrates the usage of the ModelVisualization class, which provides
a comprehensive 3D visualization framework for electromagnetic simulation results.
It combines PyVista for 3D rendering with Matplotlib for 2D data overlays.

The ModelVisualization class manages key visualization features including:
- 3D scene rendering with actors (antennas, structures, objects)
- Animated video generation with customizable camera controls
- Real-time data overlay plots (range-Doppler, field strength, etc.)
- Point cloud visualization for radar targets
- Support for multiple coordinate systems and transformations
- Flexible camera positioning and movement

Author: Example Script
Date: June 2025
"""

import os
import sys
import numpy as np

from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API

import pem_utilities.pem_core as pem_core
from pem_utilities.actor import Actors
from pem_utilities.materials import MaterialManager
from pem_utilities.model_visualization import ModelVisualization


def main():
    """
    Comprehensive demonstration of ModelVisualization usage
    """
    

    paths = get_repo_paths()

    print("=" * 70)
    print("MODEL VISUALIZATION DEMONSTRATION")
    print("=" * 70)
    
    # ========================================================================
    # 1. BASIC SETUP AND SCENE CREATION
    # ========================================================================
    print("\n1. Basic Setup and Scene Creation")
    print("-" * 40)
    
    print("   Setting up scene actors and materials...")
    
    # Initialize material manager and actors
    mat_manager = MaterialManager()
    all_actors = Actors()
    
    # Create some basic scene actors for demonstration
    print("   ✓ Creating basic geometric actors for visualization")
    
    # Create a simple cube actor
    import pyvista as pv
    cube_mesh = pv.Cube(center=(0, 0, 0), x_length=2, y_length=2, z_length=2)
    cube_actor_name = all_actors.add_actor(mesh=cube_mesh, 
                                          name='demo_cube',
                                          mat_idx=mat_manager.get_index('aluminum'),
                                          color='blue')
    
    # Create a simple sphere actor
    sphere_mesh = pv.Sphere(radius=1, center=(5, 0, 1))
    sphere_actor_name = all_actors.add_actor(mesh=sphere_mesh,
                                            name='demo_sphere', 
                                            mat_idx=mat_manager.get_index('pec'),
                                            color='red')
    
    # Create a ground plane
    ground_mesh = pv.Plane(center=(0, 0, -1), direction=(0, 0, 1), 
                          i_size=20, j_size=20, i_resolution=10, j_resolution=10)
    ground_actor_name = all_actors.add_actor(mesh=ground_mesh,
                                            name='ground_plane',
                                            mat_idx=mat_manager.get_index('concrete'),
                                            color='gray')
    
    print(f"   ✓ Created {len(all_actors.actors)} scene actors:")
    for actor_name in all_actors.actors.keys():
        print(f"     • {actor_name}")
    
    # ========================================================================
    # 2. BASIC MODEL VISUALIZATION CREATION
    # ========================================================================
    print("\n2. Basic ModelVisualization Creation")
    print("-" * 43)
    
    print("   Creating ModelVisualization with default settings...")
    
    # Create basic visualization
    output_movie_name = os.path.join(paths.output, 'basic_visualization.mp4')
    modeler = ModelVisualization(
        all_scene_actors=all_actors,
        show_antennas=True,
        output_movie_name=output_movie_name,
        fps=10,
        overlay_results=False  # Start without data overlay
    )
    
    print("   ✓ ModelVisualization created successfully")
    print("   ✓ PyVista plotter initialized")
    print("   ✓ Video recording configured")
    print("   ✓ Scene actors added to 3D visualization")
    
    # Show basic visualization
    print("\n   Displaying basic 3D scene...")
    modeler.pl.show_grid()  # Add grid for reference
    modeler.update_frame(write_frame=False)  # Display without recording
    print("   ✓ Basic scene displayed (close window to continue)")
    
    # Clean up basic visualization
    modeler.close()
    
    # ========================================================================
    # 3. CAMERA ORIENTATION OPTIONS
    # ========================================================================
    print("\n3. Camera Orientation Options")
    print("-" * 34)
    
    print("   ModelVisualization supports various camera orientation presets:")
    
    camera_orientations = [
        ('scene_top', 'High altitude overview from above'),
        ('follow', 'Standard third-person follow camera'),
        ('follow2', 'Medium distance follow camera'),
        ('follow3', 'Long distance follow camera'), 
        ('follow4', 'Close follow camera'),
        ('side', 'Side perspective view'),
        ('top', 'Overhead view with forward orientation'),
        ('front', 'Front-facing view'),
        ('radar', 'First-person radar perspective')
    ]
    
    print(f"     {'Orientation':<12} {'Description'}")
    print(f"     {'-'*12} {'-'*40}")
    for orientation, description in camera_orientations:
        print(f"     {orientation:<12} {description}")
    
    print("\n   Demonstrating camera attachment to moving actor...")
    
    # Animate the sphere actor
    sphere_actor = all_actors.actors[sphere_actor_name]
    
    # Create visualization with camera following the sphere
    output_movie_name = os.path.join(paths.output, 'camera_follow_demo.mp4')
    modeler = ModelVisualization(
        all_scene_actors=all_actors,
        show_antennas=True,
        output_movie_name=output_movie_name,
        fps=10,
        overlay_results=False,
        camera_orientation='follow',
        camera_attachment=sphere_actor_name
    )
    
    print("   ✓ Camera attached to sphere actor")
    print("   ✓ Camera will follow sphere movement")
    
    # Animate the sphere in a circle
    num_frames = 30
    for frame in range(num_frames):
        t = frame / num_frames * 2 * np.pi
        new_pos = [5 + 3*np.cos(t), 3*np.sin(t), 1 + np.sin(2*t)]
        sphere_actor.coord_sys.pos = new_pos
        sphere_actor.coord_sys.update()
        
        modeler.update_frame(write_frame=True)
    
    print(f"   ✓ Recorded {num_frames} frames of camera following animation")
    modeler.close()
    
    # ========================================================================
    # 4. DATA OVERLAY CAPABILITIES
    # ========================================================================
    print("\n4. Data Overlay Capabilities")
    print("-" * 32)
    
    print("   ModelVisualization supports various data overlay types:")
    print("   • Range-Doppler plots for radar applications")
    print("   • Range-Pulse plots for temporal radar analysis") 
    print("   • Frequency-Pulse plots for signal analysis")
    print("   • 2D spatial field distributions")
    print("   • 1D line plots for spatial profiles")
    print("   • Range-Angle plots for beam pattern analysis")
    
    # Create sample data domains
    vel_domain = np.linspace(-10, 10, 50)  # Velocity in m/s
    rng_domain = np.linspace(0, 100, 100)  # Range in meters
    freq_domain = np.linspace(76e9, 81e9, 256)  # Frequency in Hz
    
    print("\n   a) Range-Doppler visualization:")
    output_movie_name = os.path.join(paths.output, 'range_doppler_demo.mp4')
    modeler = ModelVisualization(
        all_scene_actors=all_actors,
        show_antennas=True,
        output_movie_name=output_movie_name,
        fps=10,
        overlay_results=True,
        vel_domain=vel_domain,
        rng_domain=rng_domain,
        figure_size=(0.3, 0.4)  # Size of overlay as fraction of screen
    )
    
    print("     ✓ Range-Doppler overlay configured")
    print(f"     ✓ Velocity domain: {vel_domain[0]:.1f} to {vel_domain[-1]:.1f} m/s")
    print(f"     ✓ Range domain: {rng_domain[0]:.1f} to {rng_domain[-1]:.1f} m")
    
    # Generate sample range-Doppler data and animate
    for frame in range(20):
        # Create synthetic range-Doppler response
        range_doppler_data = np.random.rand(len(vel_domain), len(rng_domain)) * 100
        # Add some peaks to make it interesting
        range_doppler_data[20:25, 40:45] += 50 + 30*np.sin(frame*0.3)
        range_doppler_data[30:35, 60:65] += 40 + 20*np.cos(frame*0.5)
        
        # Convert to dB scale
        range_doppler_db = 20 * np.log10(np.maximum(range_doppler_data, 1e-6))
        
        # Update visualization
        modeler.update_frame(write_frame=True, 
                           plot_data=range_doppler_db,
                           plot_limits=[range_doppler_db.min(), range_doppler_db.max()])
    
    print("     ✓ Range-Doppler animation completed")
    modeler.close()
    
    print("\n   b) Frequency-domain visualization:")
    output_movie_name = os.path.join(paths.output, 'frequency_demo.mp4')
    modeler = ModelVisualization(
        all_scene_actors=all_actors,
        show_antennas=True,
        output_movie_name=output_movie_name,
        fps=10,
        overlay_results=True,
        freq_domain=freq_domain,
        rng_domain=rng_domain,
        figure_size=(0.35, 0.4)
    )
    
    print("     ✓ Frequency-Range overlay configured")
    print(f"     ✓ Frequency domain: {freq_domain[0]/1e9:.1f} to {freq_domain[-1]/1e9:.1f} GHz")
    
    # Generate sample frequency response data
    for frame in range(15):
        freq_response = np.random.rand(len(freq_domain), len(rng_domain)) * 50
        # Add frequency-dependent features
        freq_response += 20 * np.exp(-((freq_domain.reshape(-1,1) - 78.5e9)**2)/(2*(1e9)**2))
        
        modeler.update_frame(write_frame=True,
                           plot_data=freq_response,
                           plot_limits=[freq_response.min(), freq_response.max()])
    
    print("     ✓ Frequency-domain animation completed")
    modeler.close()
    
    # ========================================================================
    # 5. POINT CLOUD VISUALIZATION
    # ========================================================================
    print("\n5. Point Cloud Visualization")
    print("-" * 32)
    
    print("   ModelVisualization supports point cloud rendering for radar targets:")
    print("   • Color mapping based on received power, velocity, or RCS")
    print("   • Size mapping based on target properties")
    print("   • Different shapes: spheres, cubes, points")
    print("   • Dynamic updates for moving targets")
    
    # Create point cloud visualization
    output_movie_name = os.path.join(paths.output, 'point_cloud_demo.mp4')
    modeler = ModelVisualization(
        all_scene_actors=all_actors,
        show_antennas=True,
        output_movie_name=output_movie_name,
        fps=10,
        overlay_results=False,
        camera_orientation='scene_top'
    )
    
    print("\n   Creating synthetic radar targets...")
    
    # Generate synthetic target list
    num_targets = 20
    target_list = {}
    
    for i in range(num_targets):
        target_name = f'target_{i}'
        target_list[target_name] = {
            'xpos': np.random.uniform(-15, 15),
            'ypos': np.random.uniform(-15, 15), 
            'zpos': np.random.uniform(0, 5),
            'rcs': np.random.uniform(0.1, 10),
            'p_received': np.random.uniform(1e-10, 1e-6),
            'velocity': np.random.uniform(-5, 5)
        }
    
    print(f"   ✓ Generated {num_targets} synthetic radar targets")
    
    # Animate point cloud with different visualization modes
    visualization_modes = [
        ('p_received', 'p_received', 'sphere'),
        ('velocity', 'velocity', 'cube'),
        ('rcs', 'rcs', 'sphere'),
        ('p_received', 'velocity', 'point')
    ]
    
    frame_counter = 0
    for color_mode, size_mode, shape in visualization_modes:
        print(f"   ✓ Visualizing with color={color_mode}, size={size_mode}, shape={shape}")
        
        for sub_frame in range(10):
            # Update target positions slightly for animation
            for target_name in target_list:
                target_list[target_name]['xpos'] += np.random.uniform(-0.1, 0.1)
                target_list[target_name]['ypos'] += np.random.uniform(-0.1, 0.1)
                
            # Add point cloud to scene
            modeler.add_point_cloud_to_scene(
                target_list=target_list,
                tx_pos=[0, 0, 2],  # Transmitter position
                color_mode=color_mode,
                size_mode=size_mode,
                shape=shape,
                max_radius=1.0,
                color_min=-80,  # dB scale for received power
                color_max=-40
            )
            
            modeler.update_frame(write_frame=True)
            frame_counter += 1
    
    print(f"   ✓ Point cloud animation completed ({frame_counter} total frames)")
    modeler.close()
    
    # ========================================================================
    # 6. CUSTOM CAMERA CONFIGURATION
    # ========================================================================
    print("\n6. Custom Camera Configuration")
    print("-" * 34)
    
    print("   ModelVisualization supports custom camera configurations:")
    print("   • Fixed camera positions")
    print("   • Custom camera orientations with dictionaries")
    print("   • Dynamic camera movements")
    
    # Example of custom camera dictionary
    custom_camera_config = {
        'cam_offset': [-10, -5, 8],     # Camera offset from attachment point
        'focal_offset': [20, 0, 0],     # Where camera looks relative to attachment
        'up': (0.0, 0.0, 1.0),          # Up vector
        'view_angle': 75                 # Field of view angle
    }
    
    print("\n   a) Custom camera configuration example:")
    output_movie_name = os.path.join(paths.output, 'custom_camera_demo.mp4')
    modeler = ModelVisualization(
        all_scene_actors=all_actors,
        show_antennas=True,
        output_movie_name=output_movie_name,
        fps=10,
        overlay_results=False,
        camera_orientation=custom_camera_config,
        camera_attachment=cube_actor_name
    )
    
    print("     ✓ Custom camera configuration applied")
    print(f"     ✓ Camera offset: {custom_camera_config['cam_offset']}")
    print(f"     ✓ Focal offset: {custom_camera_config['focal_offset']}")
    print(f"     ✓ View angle: {custom_camera_config['view_angle']}°")
    
    # Animate the cube with custom camera
    cube_actor = all_actors.actors[cube_actor_name]
    for frame in range(25):
        t = frame / 25 * 4 * np.pi
        new_pos = [2*np.cos(t), 2*np.sin(t), 1 + 0.5*np.sin(3*t)]
        cube_actor.coord_sys.pos = new_pos
        cube_actor.coord_sys.update()
        
        modeler.update_frame(write_frame=True)
    
    print("     ✓ Custom camera animation completed")
    modeler.close()
    
    print("\n   b) Fixed camera position example:")
    fixed_camera_pos = (10, 10, 10)  # Fixed camera position
    output_movie_name = os.path.join(paths.output, 'fixed_camera_demo.mp4')
    modeler = ModelVisualization(
        all_scene_actors=all_actors,
        show_antennas=True,
        output_movie_name=output_movie_name,
        fps=10,
        overlay_results=False,
        camera_position=fixed_camera_pos
    )
    
    print(f"     ✓ Fixed camera position: {fixed_camera_pos}")
    
    # Animate both objects with fixed camera
    for frame in range(20):
        t = frame / 20 * 2 * np.pi
        
        # Move cube
        cube_actor.coord_sys.pos = [3*np.cos(t), 3*np.sin(t), 1]
        cube_actor.coord_sys.update()
        
        # Move sphere
        sphere_actor.coord_sys.pos = [6 + 2*np.cos(2*t), 2*np.sin(2*t), 2]
        sphere_actor.coord_sys.update()
        
        modeler.update_frame(write_frame=True)
    
    print("     ✓ Fixed camera animation completed")
    modeler.close()
    
    # ========================================================================
    # 7. ADVANCED FEATURES AND BEST PRACTICES
    # ========================================================================
    print("\n7. Advanced Features and Best Practices")
    print("-" * 43)
    
    print("   Advanced ModelVisualization features:")
    
    advanced_features = [
        "Multi-domain data overlay (range-Doppler, frequency-pulse, etc.)",
        "Dynamic point cloud updates for real-time radar visualization",
        "Antenna pattern visualization with show_antennas=True",
        "Video recording with configurable frame rates and resolutions",
        "Automatic screen resolution detection for optimal output",
        "Actor transparency and material property visualization",
        "Coordinate system transformations and actor animations",
        "Grid and axes overlays for reference",
        "Custom colormaps and data scaling options",
        "Interactive display mode (write_frame=False) for development"
    ]
    
    for i, feature in enumerate(advanced_features, 1):
        print(f"   {i:2d}. {feature}")
    
    print("\n   Best practices for ModelVisualization:")
    
    best_practices = [
        "Always call modeler.close() when finished to release resources",
        "Use write_frame=False for interactive development and debugging",
        "Choose appropriate fps based on animation complexity (5-30 fps)",
        "Scale data appropriately for visualization (dB scale for power)",
        "Use camera_attachment for dynamic following shots",
        "Configure overlay_results=False if no data plots are needed",
        "Set reasonable figure_size for data overlays (0.2-0.4 screen fraction)",
        "Use show_grid() for spatial reference in complex scenes",
        "Test different camera orientations to find optimal viewpoints",
        "Consider output_video_size for target display resolution"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        print(f"   {i:2d}. {practice}")
    
    # ========================================================================
    # 8. SUMMARY AND WORKFLOW EXAMPLE
    # ========================================================================
    print("\n8. Summary and Workflow Example")
    print("-" * 34)
    
    print("   Typical ModelVisualization workflow:")
    
    workflow_steps = [
        "1. Create scene actors using Actors() class",
        "2. Initialize ModelVisualization with desired parameters",
        "3. Configure camera orientation and attachment",
        "4. Set up data domains for overlays if needed",
        "5. Create animation loop with actor updates",
        "6. Call update_frame() for each frame",
        "7. Update plot data if using overlays",
        "8. Call modeler.close() to finalize video and cleanup"
    ]
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print("\n   Example code pattern:")
    example_code = '''
    # Basic usage pattern
    modeler = ModelVisualization(
        all_scene_actors=all_actors,
        output_movie_name='my_animation.mp4',
        fps=15,
        camera_orientation='follow',
        camera_attachment='my_actor',
        overlay_results=True,
        vel_domain=velocity_range,
        rng_domain=range_values
    )
    
    for frame in range(num_frames):
        # Update actor positions
        my_actor.coord_sys.pos = calculate_new_position(frame)
        my_actor.coord_sys.update()
        
        # Generate new data
        plot_data = generate_radar_data(frame)
        
        # Update visualization
        modeler.update_frame(write_frame=True, plot_data=plot_data)
    
    # Cleanup
    modeler.close()
    '''
    
    print(example_code)
    
    print("\n" + "=" * 70)
    print("MODEL VISUALIZATION DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nAll ModelVisualization features demonstrated with examples.")
    print("Check the output folder for generated animation videos:")
    print(f"  • {paths.output}")
    print("\nKey videos created:")
    print("  • basic_visualization.mp4 - Basic 3D scene")
    print("  • camera_follow_demo.mp4 - Camera following animation")
    print("  • range_doppler_demo.mp4 - Range-Doppler data overlay")
    print("  • frequency_demo.mp4 - Frequency domain visualization")
    print("  • point_cloud_demo.mp4 - Radar target point clouds")
    print("  • custom_camera_demo.mp4 - Custom camera configuration")
    print("  • fixed_camera_demo.mp4 - Fixed camera position")


if __name__ == "__main__":
    """
    Run the model visualization demonstration
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()