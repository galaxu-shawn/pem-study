"""
Antenna Insertion Methods Example Script
=======================================

This script demonstrates the various ways to insert antennas into a simulation scene
using the antenna_device.py utilities. The antenna utilities provide multiple methods
for creating and configuring antenna systems for electromagnetic simulations.

Main Insertion Methods:
- AntennaDevice class: Load complete antenna systems from JSON configurations
- Helper functions: add_single_tx(), add_single_rx(), add_single_tx_rx()
- AntennaArray class: Create array configurations programmatically
- Multi-channel functions: add_multi_channel_radar_az_el()
- Direct antenna creation: Manual antenna configuration

Author: Example Script
Date: June 2025
"""

import numpy as np

from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API

from pem_utilities.antenna_device import (
    AntennaDevice, AntennaArray, Waveform,
    add_single_tx, add_single_rx, add_single_tx_rx,
    add_multi_channel_radar_az_el, add_antenna_device_from_json,
    enable_coupling
)
from pem_utilities.actor import Actors
from pem_utilities.materials import MaterialManager
from pem_utilities.rotation import euler_to_rot

def main():
    """
    Comprehensive demonstration of antenna insertion methods

    There are 3 primary wasy to define an antenna device into a scene.
    1. Using the AntennaDevice class which loads a complete antenna system from a JSON configuration file.
    2. Using helper functions to quickly create single antennas or antenna arrays from a few inputs,
    3. Using a combindation of waveform and AntennaDevice classes to create antennas manually.

    In general, with Perceive EM there is a heirarchy of antenna definitions:
    - The Antenna Platform, or node in which we will be iserting our antenna device
    - - Antenna Devices: Represents a complete antenna system, can include multiple antennas.
    - - - Antennas: The actual radiating elements (which can be parametric or FFD based).

    There are also 2 primary modes of operation for the simulation, radar mode or communications mode.
    The radar mode is used for radar simulations, while the communications mode is used for point-to-point (P2P) communications.
    With radar mode, results are in terms of Scatter Fields, in Comms mode, results are in terms of Total Fields.

    If in radar mode, an antenna device must consists of at least one transmitter (TX) and one receiver (RX) antenna.
    If in communications mode, an antenna device can consist of only a transmitter (TX) or a receiver (RX) antenna, but each device much have a macthing waveform/mode

    In Comms mode, you must enable coupling between the Tx and Rx antennas in order to compute the coupling
    

    """
    
    print("=" * 70)
    print("ANTENNA INSERTION METHODS DEMONSTRATION")
    print("=" * 70)
    
    # ========================================================================
    # SETUP SCENE AND BASIC COMPONENTS
    # ========================================================================
    print("\n1. Scene Setup")
    print("-" * 20)
    
    # Initialize basic components
    mat_manager = MaterialManager()
    # make a dummy actors we will use to add antennas
    all_actors = Actors(material_manager=mat_manager)
    actor_name = all_actors.add_actor(name='actor1') # for all Tx/Rx radar and Tx for P2P we will use this node to attach the attennas
    actor2_name = all_actors.add_actor(name='actor2') # when we demonstrate antenna to antenna coupling (P2P) for comms we will use this actor as the Rx location
    

    
    print("✓ Scene components initialized")

    
    # ========================================================================
    # METHOD 1: ANTENNA DEVICE FROM JSON CONFIGURATION (manually load)
    # ========================================================================
    print("\n2. Method 1: AntennaDevice from JSON Configuration")
    print("-" * 55)
    
    print("   This method loads complete antenna systems from JSON files.")
    print("   Supports multi-antenna configurations with complex geometries.")
    print("   The JSON file includes the antenna definitions, modes, and waveforms.")
    
    # Method 1A: Direct AntennaDevice initialization
    print("   a) Direct AntennaDevice initialization:")

    # This would load from a JSON file in the antenna_device_library
    ant_device_json = AntennaDevice(
        file_name='example_1tx_1rx.json',  # JSON file with antenna definitions
        parent_h_node=all_actors.actors[actor_name].h_node,  # Attach to the actor node
        all_actors=all_actors # used for visualization, patterns will be loaded into the actors
    )
    ant_device_json.initialize_mode(mode_name='mode1')
    ant_device_json.coord_sys.pos = np.array([0.0, 0.0, 1.5])  # 1.5m above ground
    ant_device_json.coord_sys.update()
    ant_device_json.add_antennas(mode_name='mode1', load_pattern_as_mesh=True, scale_pattern=5.0)
    ant_device_json.add_mode(mode_name='mode1')
    ant_device_json.set_mode_active('mode1')
      

    # Method 1B: Helper function for JSON loading
    print("   b) Helper function for JSON loading:")

    ant_device_helper = add_antenna_device_from_json(
        all_actors=all_actors, # this is used for visualzation, patterns will be loaded into the actors
        parent_h_node=all_actors.actors[actor_name].h_node,  # Attach to the actor node
        json_file='example_1tx_30GHz.json',  # Another JSON configuration
        mode_name=None,  # Use first mode found in JSON
        pos=np.array([10.0, 0.0, 1.5]),
        scale_pattern=3.0 # for visulization purposes, this scales the antenna pattern
    )
    print("     ✓ Added antenna device using helper function")
        

    
    # ========================================================================
    # 3. METHOD 2: SINGLE ANTENNA HELPER FUNCTIONS
    # ========================================================================

    # Create a basic waveform for all antenna examples, see waveform example for variations of this waveform definition
    waveform_dict = {
        "mode": "PulsedDoppler",
        "output": "FreqPulse",
        "center_freq": 77.0e9,      # 77 GHz automotive radar
        "bandwidth": 1.0e9,         # 1 GHz bandwidth
        "num_freq_samples": 512,    # Frequency samples
        "cpi_duration": 4.0e-3,     # 4 ms coherent processing interval
        "num_pulse_CPI": 256,       # Number of pulses
        "tx_multiplex": "SIMULTANEOUS",
        "mode_delay": "CENTER_CHIRP"
    }
    
    waveform = Waveform(waveform_dict)
    mode_name = 'mode1'
    print(f"✓ Waveform created: {waveform.center_freq/1e9:.1f} GHz, {waveform.bandwidth/1e6:.0f} MHz BW")

    print("\n3. Method 2: Single Antenna Helper Functions")
    print("-" * 50)
    
    print("   These helper functions create simple antenna configurations quickly.")
    
    # Method 2A: Single transmitter (for Comms mode)
    print("   a) Single transmitter (TX) antenna:")
    tx_device = add_single_tx(
        all_actors=all_actors,
        parent_h_node=all_actors.actors[actor_name].h_node,  # Attach to the actor node
        waveform=waveform,
        mode_name=mode_name,
        pos=np.array([20.0, 0.0, 1.0]),           # Position
        ffd_file=None,                             # Use parametric (None) or FFD file
        beamwidth_H=30.0,                          # Horizontal beamwidth (degrees)
        beamwidth_V=15.0,                          # Vertical beamwidth (degrees)
        scale_pattern=4.0
    )
    print("     ✓ Single TX antenna created with parametric pattern")
    
    # Method 2B: Single receiver (for Comms mode)
    print("   b) Single receiver (RX) antenna:")
    rx_device = add_single_rx(
        all_actors=all_actors,
        parent_h_node=all_actors.actors[actor2_name].h_node,  # Attach to the actor 2 node
        waveform=waveform,
        mode_name=mode_name,
        pos=np.array([25.0, 0.0, 1.0]),
        ffd_file=None,                             # Parametric antenna
        beamwidth_H=45.0,
        beamwidth_V=20.0,
        scale_pattern=4.0
    )
    print("     ✓ Single RX antenna created")
    
    # this is required for P2P communications, where we need to enable coupling between the TX and RX antennas,
    # see below for how to enable coupling


    # Method 2C: Combined TX/RX device (for radar mode)
    print("   c) Combined TX/RX antenna device:")
    tx_rx_device = add_single_tx_rx(
        all_actors=all_actors,
        parent_h_node=all_actors.actors[actor_name].h_node,  # Attach to the actor node
        waveform=waveform,
        mode_name=mode_name,
        pos=np.array([30.0, 0.0, 1.0]),
        ffd_file='dipole.ffd',                 # Use FFD file for both TX and RX, this will look for a file in the antenna_device_library
        scale_pattern=4.0
    )
    print("     ✓ Combined TX/RX device created")
    
    
    # Method 2D: Planewave antenna (only for radar mode)
    print("   d) Planewave antenna:")
    planewave_device = add_single_tx_rx(
        all_actors=all_actors,
        waveform=waveform,
        mode_name=mode_name,
        pos=np.array([0.0, 0.0, 0.0]), # zero phase reference
        rot = np.eye(3),            # No rotation
        planewave=True,                            # Enable planewave mode
        polarization='HH'                         # Horizontal-Horizontal
    )
    print("     ✓ Planewave antenna created")
    
    # ========================================================================
    # 4. METHOD 3: ANTENNA ARRAYS
    # ========================================================================
    print("\n4. Method 3: Antenna Arrays")
    print("-" * 35)
    
    print("   AntennaArray class creates multi-element antenna arrays.")
    
    # Method 3A: Parametric antenna array
    print("   a) Parametric antenna array:")
    param_array = AntennaArray(
        name='parametric_array',
        parent_h_node= None,  # No parent node, will not be attached to an existing actor
        waveform=waveform,
        mode_name=mode_name,
        file_name=None,                            # None = parametric antennas
        beamwidth_H=45.0,
        beamwidth_V=20.0,
        polarization='V',                          # Vertical polarization
        rx_shape=[4, 1],                          # 4x1 RX array
        tx_shape=[1, 1],                          # 1x1 TX array
        spacing_wl_x=0.5,                         # 0.5 wavelength spacing in X
        spacing_wl_y=0.5,                         # 0.5 wavelength spacing in Y
        normal='z',                               # Array normal direction
        all_actors=all_actors
    )
    # Position the array
    param_array.antenna_device.coord_sys.pos = np.array([0.0, 20.0, 1.0])
    param_array.antenna_device.coord_sys.update()
    print("     ✓ 4x1 parametric RX array with 1x1 TX created")
    
    # Method 3B: FFD-based antenna array
    print("   b) FFD-based antenna array:")

    ffd_array = AntennaArray(
        name='ffd_array',
        waveform=waveform,
        mode_name=mode_name,
        file_name='dipole.ffd',                # Use FFD file for all elements
        rx_shape=[2, 2],                      # 2x2 RX array
        tx_shape=[1, 1],                      # 1x1 TX array
        spacing_wl_x=0.6,
        spacing_wl_y=0.6,
        parent_h_node=None,
        normal='z',
        all_actors=all_actors
    )
    ffd_array.antenna_device.coord_sys.pos = np.array([10.0, 20.0, 1.0])
    ffd_array.antenna_device.coord_sys.update()
    print("     ✓ 2x2 FFD-based array created")
    

    
    # ========================================================================
    # 5. METHOD 4: MULTI-CHANNEL RADAR ARRAYS
    # ========================================================================
    print("\n5. Method 4: Multi-Channel Radar Arrays")
    print("-" * 45)
    
    print("   Specialized function for azimuth/elevation antenna arrays.")
    
    # Method 4A: Azimuth array with parametric antennas
    print("   a) Azimuth array with parametric antennas:")
    az_array_device = add_multi_channel_radar_az_el(
        all_actors=all_actors,
        waveform=waveform,
        mode_name=mode_name,
        num_rx_az=8,                              # 8 RX antennas in azimuth
        num_rx_el=0,                              # No elevation antennas
        spacing_wl=0.5,                           # Half wavelength spacing
        pos=np.array([0.0, 40.0, 1.0]),
        normal='x',                               # Array normal along X-axis
        ffd_file=None,                            # Parametric antennas
        beamwidth_H=120.0,
        beamwidth_V=30.0,
        polarization='VV',
        scale_pattern=3.0
    )
    print("     ✓ 1TX + 8RX azimuth array created")
    
    # Method 4B: Combined azimuth/elevation array
    print("   b) Combined azimuth/elevation array:")
    az_el_array_device = add_multi_channel_radar_az_el(
        all_actors=all_actors,
        waveform=waveform,
        mode_name=mode_name,
        num_rx_az=4,                              # 4 RX in azimuth
        num_rx_el=2,                              # 2 RX in elevation
        spacing_wl=0.5,
        pos=np.array([20.0, 40.0, 1.0]),
        normal='x',
        beamwidth_H=90.0,
        beamwidth_V=45.0,
        polarization='HH',
        scale_pattern=3.0
    )
    print("     ✓ 1TX + 4RX(az) + 2RX(el) array created")
    
    # Method 4C: FFD-based multi-channel array
    print("   c) FFD-based multi-channel array:")

    ffd_multichannel = add_multi_channel_radar_az_el(
        all_actors=all_actors,
        waveform=waveform,
        mode_name=mode_name,
        num_rx_az=6,
        num_rx_el=0,
        spacing_wl=0.4,
        pos=np.array([40.0, 40.0, 1.0]),
        ffd_file='dipole.ffd',                # Use FFD pattern
        scale_pattern=3.0
    )
    print("     ✓ FFD-based multi-channel array created")
        

    
    # ========================================================================
    # 6. ANTENNA COUPLING AND P2P COMMUNICATION
    # ========================================================================
    print("\n6. Antenna Coupling and P2P Communication")
    print("-" * 48)
    
    print("   Demonstrating antenna coupling for point-to-point links.")
    
    # Enable coupling between TX and RX devices for P2P simulation

    enable_coupling(mode_name, tx_device, rx_device)
    print("     ✓ Coupling enabled between TX and RX devices")
    
        

    
    # ========================================================================
    # 8. ANTENNA POSITIONING AND ORIENTATION
    # ========================================================================
    print("\n8. Antenna Positioning and Orientation Examples")
    print("-" * 52)
    
    print("   Demonstrating different positioning and orientation methods.")
    
    # Method 8A: Rotated antenna
    print("   a) Rotated antenna device:")
    rotated_device = add_single_tx(
        all_actors=all_actors,
        waveform=waveform,
        mode_name=mode_name,
        pos=np.array([60.0, 0.0, 2.0]),
        rot=euler_to_rot(phi=0, theta=30, psi=45, deg=True),  # 30° tilt, 45° rotation
        beamwidth_H=40.0,
        beamwidth_V=20.0,
        scale_pattern=4.0
    )
    print("     ✓ Antenna with 30° tilt and 45° rotation")
    
    # Method 8B: Moving antenna
    print("   b) Antenna with linear velocity:")
    moving_device = add_single_tx_rx(
        all_actors=all_actors,
        waveform=waveform,
        mode_name=mode_name,
        pos=np.array([60.0, 20.0, 1.0]),
        lin=np.array([5.0, 0.0, 0.0]),              # 5 m/s in X direction
        polarization='VV',
        scale_pattern=4.0
    )
    print("     ✓ Antenna with 5 m/s linear velocity")
    
    # Method 8C: Antenna attached to actor
    print("   c) Antenna attached to moving actor:")
    
    # Create a simple actor (could be a vehicle)
    vehicle_actor = all_actors.add_actor(name='vehicle_platform')
    all_actors.actors[vehicle_actor].coord_sys.pos = np.array([80.0, 0.0, 0.5])
    all_actors.actors[vehicle_actor].coord_sys.lin = np.array([10.0, 0.0, 0.0])  # Moving at 10 m/s
    all_actors.actors[vehicle_actor].coord_sys.update()
    
    # Attach antenna to vehicle
    vehicle_antenna = add_single_tx_rx(
        all_actors=all_actors,
        waveform=waveform,
        mode_name=mode_name,
        pos=np.array([0.0, 0.0, 1.5]),              # 1.5m above vehicle
        parent_h_node=all_actors.actors[vehicle_actor].h_node,          # Attach to vehicle
        polarization='VV',
        scale_pattern=3.0
    )
    print("     ✓ Antenna attached to moving vehicle actor")
    
    # ========================================================================
    # 9. DIFFERENT ANTENNA TYPES AND PATTERNS
    # ========================================================================
    print("\n9. Different Antenna Types and Patterns")
    print("-" * 45)
    
    print("   Demonstrating various antenna pattern types.")
    
    antenna_examples = [
        {
            'name': 'Wide beam',
            'pos': [0.0, 80.0, 1.0],
            'beamwidth_H': 120.0,
            'beamwidth_V': 90.0,
            'description': 'Wide coverage antenna'
        },
        {
            'name': 'Narrow beam',
            'pos': [20.0, 80.0, 1.0],
            'beamwidth_H': 10.0,
            'beamwidth_V': 5.0,
            'description': 'High-gain narrow beam'
        },
        {
            'name': 'Asymmetric beam',
            'pos': [40.0, 80.0, 1.0],
            'beamwidth_H': 80.0,
            'beamwidth_V': 15.0,
            'description': 'Wide azimuth, narrow elevation'
        }
    ]
    
    for i, ant_example in enumerate(antenna_examples):
        ant_dev = add_single_tx(
            all_actors=all_actors,
            waveform=waveform,
            mode_name=mode_name,
            pos=np.array(ant_example['pos']),
            beamwidth_H=ant_example['beamwidth_H'],
            beamwidth_V=ant_example['beamwidth_V'],
            scale_pattern=3.0
        )
        print(f"     ✓ {ant_example['description']}: {ant_example['beamwidth_H']}°×{ant_example['beamwidth_V']}°")
    
    # ========================================================================
    # 10. SUMMARY AND BEST PRACTICES
    # ========================================================================
    print("\n10. Summary and Best Practices")
    print("-" * 35)
    
    # Count total antennas created
    total_actors = len(all_actors.get_actor_names())
    
    print("   Antenna insertion methods demonstrated:")
    print("   1. JSON configuration files - Most flexible for complex systems")
    print("   2. Helper functions - Quick setup for simple configurations")
    print("   3. Antenna arrays - Systematic multi-element arrangements")
    print("   4. Multi-channel functions - Specialized radar array configurations")
    print("   5. Manual creation - Maximum control over individual antennas")
    
    print("\n   Best practices:")
    print("   • Use JSON files for reusable, complex antenna configurations")
    print("   • Use helper functions for quick prototyping and simple setups")
    print("   • Use AntennaArray for systematic multi-element configurations")
    print("   • Always call coord_sys.update() after setting positions/orientations")
    print("   • Enable coupling between TX and RX for P2P simulations")
    print("   • Consider wavelength spacing (typically λ/2) for array elements")
    print("   • Use appropriate beamwidths for your application requirements")
    print("   • Scale antenna patterns appropriately for visualization")
    
    print(f"\n   Total scene actors created: {total_actors}")
    print("   Antenna types demonstrated: Parametric, FFD-based, Planewave")
    print("   Array configurations: Linear, 2D, Azimuth/Elevation")
    print("   Positioning: Static, Moving, Rotated, Attached to actors")
    
    print("\n" + "=" * 70)
    print("ANTENNA INSERTION METHODS DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    """
    Run the antenna insertion methods demonstration
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()