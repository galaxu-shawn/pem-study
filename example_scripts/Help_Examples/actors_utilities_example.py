"""
Actors Utilities Example Script
===============================

This script demonstrates comprehensive usage of the Actors.py utilities module.
The Actors utilities provide a powerful framework for managing scene objects with
position tracking, animation, and specialized behaviors for different actor types.

Main Components:
- Actors class: Container/manager for multiple actors in a scene
- Actor class: Individual actor objects with coordinate systems and behaviors

* Types of Actors:
    * Generators 
        * Any mesh that can be dynamically updated at each call. Seastate is an example of a goemetry that can be dynamically updated. Mesh is offset when time parameters is passed into generator object
        * Primitives
            * Utility to generate basic CAD objects. Some allow for generation based on RCS value, ie. Corner Reflector, Plate, Cube...etc.
    * CAD (STL/OBJ/GLTF/VTP/FACET)
        * VTP Note: This format can include embedded material properties, format is generated from Material Segmentation AI workflow.
        * Material properties must be defined during import, if not it will assume PEC.
    * DAE/DAEZ
        * Collada, blender legacy format for animated sequences. Format used for Carnegie Mellon Motion capture database
    * USD
        * Limited USD format input, animations not tested
    * JSON
        * Nested CAD files with material properties. Maintains parent/child relationship.
        * Supports special vehicle types, quadcopters, UAV, helicopters, vehicles...etc where motion of rotating objects is automatically updated with time.
        * Intial velocities and positions of children objects can be included
    * Other - a few other various example have not been formalized, but are possible, ie. geotiff, mitsuba, xml, ply, webcam

Author: Example Script
Date: June 2025
"""
import os
import sys

import pem_utilities
from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API

import numpy as np
from pem_utilities.actor import Actors
from pem_utilities.primitives import *
from pem_utilities.materials import MaterialManager
from pem_utilities.rotation import euler_to_rot

def main():
    """
    Comprehensive demonstration of Actors utilities functionality
    """
    
    print("=" * 60)
    print("ACTORS UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # ========================================================================
    # 1. INITIALIZE ACTORS MANAGER
    # ========================================================================
    print("\n1. Creating Actors Manager")
    print("-" * 30)
    
    paths = get_repo_paths()

    # Initialize material manager for proper material handling
    mat_manager = MaterialManager()
    
    # Create the main actors container - this manages all actors in the scene
    # the material_manager to have a common library of material used across all actors
    # this uses the default materials_library.json file, but a custom one can be provided
    all_actors = Actors(material_manager=mat_manager)
    print("✓ Actors manager created successfully")
    
    # ========================================================================
    # 2. ADDING ACTORS TO THE SCENE - DIFFERENT METHODS
    # ========================================================================
    print("\n2. Adding Actors to Scene - Different Methods")
    print("-" * 50)
    
    # Method 1: Add actor from STL file
    print("   a) Adding actor from STL file:")

    # Example: Add a simple mesh actor (replace with actual file path)
    simple_actor_name = all_actors.add_actor(
        name='simple_mesh',
        filename=os.path.join(paths.models,'large_sheet.stl'),  # Replace with actual STL file
        mat_idx=mat_manager.get_index('pec'),  # Perfect Electric Conductor material
        color=[0.8, 0.8, 0.8]  # Light gray color
    )
    print(f"     ✓ Added simple actor: {simple_actor_name}")

    
    # Method 2: Add actor from JSON configuration (multi-part actors)
    print("   b) Adding actor from JSON configuration:")

    # JSON files define complex multi-part actors like vehicles with wheels
    vehicle_actor_name = all_actors.add_actor(
        name='vehicle',
        filename=os.path.join(paths.models,'Audi_A1_2010/Audi_A1_2010.json')  # Replace with actual JSON file
    )
    # material properties will be set automatically based on the JSON configuration
    # Note: The JSON file should define the actor's parts and materials
    print(f"     ✓ Added vehicle actor: {vehicle_actor_name}")

    # Method 3: Add actor dummy actors. Essentially just a coordinate system with no mesh
    # This is useful for actors that do not have a physical representation but need to be tracked
    # or for actors that server as parents of child actors
    print("   c) Adding actor with no mesh:")
    empty_actor_name = all_actors.add_actor(
        name='dummy_actor',
        color=[1.0, 0.5, 0.0]  # Orange color
    )
    print(f"     ✓ Added antenna actor: {empty_actor_name}")
    
    # Method 4: Add actor, modifying movement capabilities and updates
    # This method allows actors to automatically update their position and rotation based on velocity
    # This is useful for actors that need to move in the scene, such as vehicles or pedestrians
    # if use_linear_velocity_equation_update is True, the actor will automatically update its position
    # based on its linear velocity and the time step. If set to false, the script will need to manually
    # update the position based on the velocity and time step.
    print("   d) Adding actor with automated position updates based on velocity:")
    moving_actor_name = all_actors.add_actor(
        name='moving_object',
        filename=os.path.join(paths.models,'large_sheet.stl'),
        use_linear_velocity_equation_update=False,  # Enable automatic position updates to be turned off when FAlse, default is True
        update_rot_based_on_ang_vel=False,  # Enable automatic rotation updates, default is True
        mat_idx=mat_manager.get_index('glass'), # material of glass for actor
        parent_h_node=all_actors.actors[empty_actor_name].h_node,  # Attach to the empty dummy actor previouslly created
        color=[0.0, 1.0, 0.0]  # Green color
    )
    print(f"     ✓ Added moving actor: {moving_actor_name}")

    # Method 5: Add actor with adaptive mesh resolution and scaling the geometry
    print("   e) Adding scaled actor + adaptive mesh resolution:")
    scaled_mesh_name = all_actors.add_actor(
        name='moving_object',
        filename=os.path.join(paths.models,'large_sheet.stl'),
        scale_mesh=2.0,  # Scale the mesh by 2x
        target_ray_spacing=0.1,  # Target ray spacing for adaptive mesh resolution
        color=[0.0, 1.0, 0.0]  # Green color
    )
    print(f"     ✓ Added scaled actor + adaptive mesh resolution: {scaled_mesh_name}")

    # Method 6: Add actor based ona  primitive shape/generator
    print("   f) Adding actor from primitive shape generator:")
    # Example: Add a simple box actor using the primitive generator
    cube_generator = Cube(x_length=1, y_length=1, z_length=1, rcs=None, wl=0.3) # if rcs is give, the x_length, y_length, z_length will be calculated
    primitive_actor_name = all_actors.add_actor(
        name='primitive_cube',
        generator=cube_generator,  # Use the Cube generator
        mat_idx=mat_manager.get_index('pec'),  # Perfect Electric Conductor material
        dynamic_generator_updates=False,  # Enable dynamic updates for the generator, if True, each update will generate a new mesh. If False, it will only generate the mesh intitially and not on each update
        color=[0.5, 0.5, 1.0]  # Light blue color
    )
    print(f"     ✓ Added primitive actor: {primitive_actor_name}")

    # Method 7: Add actor for animated person (pedestrian)
    print("   g) Adding animated pedestrian actor:")
    pedestrian_actor_name = all_actors.add_actor(
        name='pedestrian',
        filename=os.path.join(paths.models,'Walking_speed50_armspace50.dae'), 
        target_ray_spacing=0.1)  # Adaptive mesh resolution
    print(f"     ✓ Added pedestrian actor: {pedestrian_actor_name}")
    # Note: The DAE file should contain the walking animation, and the mesh is located in the pedestrian_bones_stl folder

    # Method 8: Add actor for animated quadcopter
    print("   h) Adding animated quadcopter actor:")
    quadcopter_actor_name = all_actors.add_actor(
        name='quadcopter',
        filename=os.path.join(paths.models,'Quadcopter/Quadcopter.json'), 
        target_ray_spacing=0.1)  # Adaptive mesh resolution
    print(f"     ✓ Added quadcopter actor: {quadcopter_actor_name}")
    # Note: The JSON file should define the quadcopter's parts and materials
    

    
    # ========================================================================
    # 3. SETTING ACTOR POSITIONS AND ORIENTATIONS
    # ========================================================================
    print("\n3. Setting Actor Positions and Orientations")
    print("-" * 45)
    
    # Method 1: Set basic position
    print("   a) Setting basic positions:")
    all_actors.actors[simple_actor_name].coord_sys.pos = np.array([10.0, 0.0, 0.0])
    all_actors.actors[simple_actor_name].coord_sys.update()  # Update the coordinate system after setting position
    print(f"     ✓ {simple_actor_name} positioned at origin")
    

    # Method 2: Set position with rotation
    print("   b) Setting position with rotation:")
    all_actors.actors[vehicle_actor_name].coord_sys.pos = np.array([-15.0, 0.0, 0.0])
    # Rotate vehicle to face +Y direction (90 degrees around Z-axis)
    all_actors.actors[vehicle_actor_name].coord_sys.rot = euler_to_rot(psi=90, deg=True)
    all_actors.actors[vehicle_actor_name].coord_sys.update()  # Update the coordinate system after setting position and rotation
    print(f"     ✓ {vehicle_actor_name} positioned at (-15, 0, 0) facing +Y direction")
    
    # Method 3: Set position, rotation, and velocity
    print("   c) Setting position, rotation, and velocity:")
    all_actors.actors[moving_actor_name].coord_sys.pos = np.array([5.0, -10.0, 1.0])
    all_actors.actors[moving_actor_name].coord_sys.rot = euler_to_rot(phi=0, theta=15, psi=45, deg=True)
    all_actors.actors[moving_actor_name].coord_sys.lin = np.array([2.0, 1.0, 0.0])  # Linear velocity
    all_actors.actors[moving_actor_name].coord_sys.ang = np.array([0.0, 0.0, 0.5])  # Angular velocity
    print(f"     ✓ {moving_actor_name} positioned with velocity and rotation")
    
    # Method 4: Using velocity magnitude for vehicles/pedestrians
    print("   d) Setting velocity magnitude (for vehicles/pedestrians):")
    # For actors, velocity_mag allows setting speed while direction follows rotation value.
    # this allows the velocity magintude, instead of a vector to be defined. Must be combined with
    # use_linear_velocity_equation_update=True to have the actor update its position based on the velocity magnitude
    if hasattr(all_actors.actors[vehicle_actor_name], 'velocity_mag'):
        all_actors.actors[vehicle_actor_name].velocity_mag = 10.0  # 10 m/s
        print(f"     ✓ {vehicle_actor_name} velocity magnitude set to 10 m/s")
    
    # Update coordinate systems to apply changes, you can update them invidually or all at once
    # if individual updates are done it would look like: all_actors.actors[vehicle_actor_name].coord_sys.update()
    # or we can just do all of them in bulk like below:
    print("   e) Updating coordinate systems:")
    for actor_name in all_actors.get_actor_names():
        all_actors.actors[actor_name].coord_sys.update()
    print("     ✓ All coordinate systems updated")
    
    # ========================================================================
    # 4. UPDATING ACTOR POSITIONS OVER TIME
    # ========================================================================
    print("\n4. Updating Actor Positions Over Time")
    print("-" * 40)
    
    print("   Simulating time progression with position updates:")
    
    # Simulate 3 time steps
    # This is an example you would combine with calling the simulator, update the actor positions
    # for each time step, then call simulation engine. Here are a few ways they can be updated
    for time_step in range(3):
        current_time = time_step * 0.1  # 0.1 second increments
        
        print(f"\n   Time: {current_time:.1f}s")
        
        # Update all actors for this time step
        for actor_name in all_actors.get_actor_names():
            # Store previous position for comparison, just as a demonstrate (we don't need to do this in in real simulations)
            prev_pos = all_actors.actors[actor_name].coord_sys.pos.copy()
            
            # Update actor based on its type and movement settings. For example, if the actor has use_linear_velocity_equation_update=True,
            # it will automatically update its position based on its linear velocity and the time step. If it doesn't, the position will not change
            all_actors.actors[actor_name].update_actor(time=current_time)
            
            # Show position change
            new_pos = all_actors.actors[actor_name].coord_sys.pos
            displacement = np.linalg.norm(new_pos - prev_pos)
            
            if displacement > 1e-6:  # Only show if there was significant movement
                print(f"     {actor_name}: moved {displacement:.3f}m to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})")
            else:
                print(f"     {actor_name}: stationary at ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})")
    
    # ========================================================================
    # 5. DIFFERENT ACTOR TYPES AND SPECIALIZED BEHAVIORS
    # ========================================================================
    print("\n5. Different Actor Types and Specialized Behaviors")
    print("-" * 55)
    
    # Demonstrate different actor types that have specialized update behaviors
    actor_types_info = {
        'vehicle': 'Handles wheel rotations and vehicle dynamics',
        'pedestrian': 'Manages walking animations and movement',
        'quadcopter': 'Controls rotor rotations and flight dynamics',
        'helicopter': 'Manages main and tail rotor rotations',
        'bird': 'Handles wing flapping animations',
        'antenna': 'Static objects for electromagnetic simulation',
        'other': 'General purpose actors with basic physics'
    }
    
    print("   Actor type behaviors:")
    for actor_type, description in actor_types_info.items():
        print(f"     {actor_type:12}: {description}")
    
    # Show current actor types in scene
    print("\n   Current actors in scene:")
    for actor_name in all_actors.get_actor_names():
        actor_type = all_actors.actors[actor_name].actor_type

        # Set specific behaviors based on actor type, quadcopter/helicopter will have rotor speeds, birds will have flap angles and freq
        if actor_type == 'quadcopter':
            all_actors.actors[actor_name].rotor_ang = [0, 0, 100*2*np.pi] # in rad/s
        elif actor_type == 'bird':
            all_actors.actors[actor_name].flap_ang = 45
            all_actors.actors[actor_name].flap_freq = 3

        # also any json based actor can have intial_pos, intial_rot, initial_lin and/or intial_ang defined in the json definition file
        # if these exists, (see Wind_turbine.json for example) those value scan be overridden when the actor is loaded

        pos = all_actors.actors[actor_name].coord_sys.pos
        print(f"     {actor_name:15} (type: {actor_type:10}) at ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f})")
    
    # ========================================================================
    # 6. ADVANCED FEATURES
    # ========================================================================
    print("\n6. Advanced Features")
    print("-" * 25)
    
    # Get actor bounds and centers
    print("   a) Actor bounds and centers:")
    for actor_name in all_actors.get_actor_names():
        try:
            bounds = all_actors.actors[actor_name].get_bounds()
            center = all_actors.actors[actor_name].get_center()
            print(f"     {actor_name}: center at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        except:
            print(f"     {actor_name}: no mesh bounds available")
    
    
    
    # ========================================================================
    # 7. SUMMARY AND BEST PRACTICES
    # ========================================================================
    print("\n7. Summary and Best Practices")
    print("-" * 35)
    
    print("   Key points for using Actors utilities:")
    print("   • Use Actors() class to manage multiple actors in a scene")
    print("   • Set positions with coord_sys.pos = np.array([x, y, z])")
    print("   • Set rotations with coord_sys.rot = euler_to_rot(...)")
    print("   • Set velocities with coord_sys.lin/ang for linear/angular velocity")
    print("   • Use velocity_mag for vehicles/pedestrians to set speed magnitude")
    print("   • Call update_actor(time) in simulation loops for animations")
    print("   • Always call coord_sys.update() after changing position/rotation")
    print("   • Use appropriate material indices for electromagnetic properties")
    print("   • Different actor types have specialized behaviors (vehicle, pedestrian, etc.)")
    
    print(f"\n   Total actors in scene: {len(all_actors.get_actor_names())}")
    print("   Actor names:", all_actors.get_actor_names())
    
    print("\n" + "=" * 60)
    print("ACTORS UTILITIES DEMONSTRATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    """
    Run the actors utilities demonstration
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()