"""Test script for refactored OceanSurface and Wake classes."""

import sys
sys.path.insert(0, 'pem_utilities')

from seastate3 import OceanSurface, Wake, OceanConfig, WakeConfig, SwellConfig

print("Testing refactored OceanSurface and Wake classes...")
print("=" * 60)

# Test 1: Factory method - calm_sea
print("\n1. Testing OceanSurface.calm_sea()...")
try:
    ocean = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Created calm sea: {height.shape}, height range: [{height.min():.3f}, {height.max():.3f}]")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Factory method - rough_sea
print("\n2. Testing OceanSurface.rough_sea()...")
try:
    ocean = OceanSurface.rough_sea(num_grid=64, scene_length=500.0, random_seed=42)
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Created rough sea: {height.shape}, height range: [{height.min():.3f}, {height.max():.3f}]")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Factory method - with_ship_wake
print("\n3. Testing OceanSurface.with_ship_wake()...")
try:
    ocean = OceanSurface.with_ship_wake(
        ship_velocity=15.0,
        ship_length=100.0,
        num_grid=64,
        scene_length=500.0,
        random_seed=42,
    )
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Created ocean with wake: {height.shape}, height range: [{height.min():.3f}, {height.max():.3f}]")
    print(f"   ✓ Wake component exists: {ocean.wake is not None}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Config-based initialization
print("\n4. Testing config-based initialization...")
try:
    ocean_config = OceanConfig(
        num_grid=64,
        scene_length=500.0,
        wind_speed=10.0,
        wave_amplitude=1.0,
        choppiness=0.5,
        random_seed=42,
    )
    wake_config = WakeConfig(
        enabled=True,
        velocity_ship=12.0,
        length_ship=110.0,
        initial_position=(0.0, 0.0),
    )
    ocean = OceanSurface(ocean_config, wake_config)
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Created ocean from configs: {height.shape}, height range: [{height.min():.3f}, {height.max():.3f}]")
    print(f"   ✓ Wake component exists: {ocean.wake is not None}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Adding wake separately
print("\n5. Testing add_wake() method...")
try:
    ocean = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
    wake = Wake.create(
        ship_length=100.0,
        ship_velocity=15.0,
        scene_length=500.0,
        num_grid=64,
    )
    ocean.add_wake(wake)
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Added wake to ocean: {height.shape}, height range: [{height.min():.3f}, {height.max():.3f}]")
    print(f"   ✓ Wake component exists: {ocean.wake is not None}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 6: Adding and removing swell
print("\n6. Testing add_swell() and remove_swell()...")
try:
    ocean = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
    ocean.add_swell(amplitude=0.5, wavelength=100.0, direction=(1.0, 0.0))
    height_with_swell = ocean.generate_height_field(0.0)
    print(f"   ✓ Added swell: height range [{height_with_swell.min():.3f}, {height_with_swell.max():.3f}]")
    
    ocean.remove_swell()
    height_no_swell = ocean.generate_height_field(0.0)
    print(f"   ✓ Removed swell: height range [{height_no_swell.min():.3f}, {height_no_swell.max():.3f}]")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 7: Legacy parameter initialization (backward compatibility)
print("\n7. Testing legacy parameter initialization...")
try:
    ocean = OceanSurface(
        num_grid=64,
        scene_length=500.0,
        wind_speed=10.0,
        wave_amplitude=1.0,
        choppiness=0.5,
        random_seed=42,
    )
    height = ocean.generate_height_field(0.0)
    print(f"   ✓ Legacy initialization works: {height.shape}, height range: [{height.min():.3f}, {height.max():.3f}]")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 8: Time evolution
print("\n8. Testing time evolution...")
try:
    ocean = OceanSurface.with_ship_wake(
        ship_velocity=15.0,
        ship_length=100.0,
        num_grid=64,
        scene_length=500.0,
        random_seed=42,
    )
    times = [0.0, 1.0, 2.0, 5.0]
    for t in times:
        height = ocean.generate_height_field(t)
        print(f"   ✓ t={t:.1f}s: height range [{height.min():.3f}, {height.max():.3f}]")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
