"""Test multiple wake functionality."""

import sys
sys.path.insert(0, 'pem_utilities')

from seastate3 import OceanSurface, Wake, OceanConfig
import numpy as np

print("Testing Multiple Wake Support")
print("=" * 80)

# Test 1: Add multiple wakes
print("\n1. Testing multiple wake addition...")
try:
    ocean = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
    
    # Add first wake
    wake1 = Wake.create(
        ship_length=110.0,
        ship_velocity=12.0,
        scene_length=500.0,
        num_grid=64,
        ship_position=(0.0, 0.0),
        ship_heading=0.0,
    )
    ocean.add_wake(wake1)
    print(f"   ✓ Added wake 1, total wakes: {ocean.get_num_wakes()}")
    
    # Add second wake
    wake2 = Wake.create(
        ship_length=50.0,
        ship_velocity=8.0,
        scene_length=500.0,
        num_grid=64,
        ship_position=(100.0, 100.0),
        ship_heading=45.0,
    )
    ocean.add_wake(wake2)
    print(f"   ✓ Added wake 2, total wakes: {ocean.get_num_wakes()}")
    
    # Add third wake
    wake3 = Wake.create(
        ship_length=80.0,
        ship_velocity=15.0,
        scene_length=500.0,
        num_grid=64,
        ship_position=(-100.0, 50.0),
        ship_heading=270.0,
    )
    ocean.add_wake(wake3)
    print(f"   ✓ Added wake 3, total wakes: {ocean.get_num_wakes()}")
    
    # Generate height field with all wakes
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Generated height field with 3 wakes: range [{height.min():.3f}, {height.max():.3f}] m")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Remove specific wake
print("\n2. Testing wake removal by index...")
try:
    ocean = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
    
    for i in range(3):
        wake = Wake.create(100.0, 10.0, 500.0, 64, ship_position=(i*50.0, 0.0))
        ocean.add_wake(wake)
    
    print(f"   ✓ Added 3 wakes, total: {ocean.get_num_wakes()}")
    
    # Remove middle wake
    ocean.remove_wake(index=1)
    print(f"   ✓ Removed wake at index 1, remaining: {ocean.get_num_wakes()}")
    
    # Verify we can still generate
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Generated height field with 2 wakes: range [{height.min():.3f}, {height.max():.3f}] m")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Remove all wakes
print("\n3. Testing removal of all wakes...")
try:
    ocean = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
    
    for i in range(5):
        wake = Wake.create(100.0, 10.0, 500.0, 64)
        ocean.add_wake(wake)
    
    print(f"   ✓ Added 5 wakes, total: {ocean.get_num_wakes()}")
    
    ocean.remove_wake()  # Remove all
    print(f"   ✓ Removed all wakes, remaining: {ocean.get_num_wakes()}")
    
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Generated height field with no wakes: range [{height.min():.3f}, {height.max():.3f}] m")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Time evolution with multiple wakes
print("\n4. Testing time evolution with multiple wakes...")
try:
    ocean = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
    
    # Add two ships moving in different directions
    wake1 = Wake.create(
        ship_length=100.0,
        ship_velocity=15.0,
        scene_length=500.0,
        num_grid=64,
        ship_position=(0.0, -200.0),
        ship_heading=0.0,  # Moving east
    )
    wake2 = Wake.create(
        ship_length=80.0,
        ship_velocity=12.0,
        scene_length=500.0,
        num_grid=64,
        ship_position=(-200.0, 0.0),
        ship_heading=90.0,  # Moving north
    )
    ocean.add_wake(wake1)
    ocean.add_wake(wake2)
    
    print(f"   ✓ Added 2 wakes moving in different directions")
    
    # Simulate over time
    times = [0.0, 5.0, 10.0, 20.0]
    for t in times:
        height = ocean.generate_height_field(t)
        print(f"   ✓ t={t:.1f}s: height range [{height.min():.3f}, {height.max():.3f}] m")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Factory method with_ship_wake still works
print("\n5. Testing backward compatibility with factory method...")
try:
    ocean = OceanSurface.with_ship_wake(
        ship_velocity=15.0,
        ship_length=100.0,
        num_grid=64,
        scene_length=500.0,
        random_seed=42,
    )
    
    print(f"   ✓ Created ocean with wake using factory method, wakes: {ocean.get_num_wakes()}")
    
    # Add another wake
    wake2 = Wake.create(80.0, 10.0, 500.0, 64, ship_position=(100.0, 0.0))
    ocean.add_wake(wake2)
    
    print(f"   ✓ Added second wake, total wakes: {ocean.get_num_wakes()}")
    
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Generated height field: range [{height.min():.3f}, {height.max():.3f}] m")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Independent wake parameters
print("\n6. Testing independent wake parameters...")
try:
    ocean = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
    
    # Large fast ship
    wake1 = Wake.create(
        ship_length=150.0,
        ship_velocity=20.0,
        scene_length=500.0,
        num_grid=64,
        ship_position=(0.0, 0.0),
        ship_heading=0.0,
    )
    
    # Small slow ship
    wake2 = Wake.create(
        ship_length=30.0,
        ship_velocity=5.0,
        scene_length=500.0,
        num_grid=64,
        ship_position=(150.0, 0.0),
        ship_heading=180.0,
    )
    
    ocean.add_wake(wake1)
    ocean.add_wake(wake2)
    
    print(f"   ✓ Added large fast ship (L={wake1.length_ship}m, V={wake1.velocity_ship}m/s)")
    print(f"   ✓ Added small slow ship (L={wake2.length_ship}m, V={wake2.velocity_ship}m/s)")
    
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Generated superimposed wake: range [{height.min():.3f}, {height.max():.3f}] m")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("All multi-wake tests completed!")
