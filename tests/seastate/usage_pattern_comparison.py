"""
Comparison of different usage patterns for OceanSurface.

This script demonstrates the various ways to create and configure ocean surfaces,
from simple factory methods to advanced configuration objects.
"""

import sys
sys.path.insert(0, 'pem_utilities')

from seastate3 import OceanSurface, Wake, OceanConfig, WakeConfig, SwellConfig
import numpy as np

print("OceanSurface Usage Pattern Comparison")
print("=" * 80)

# Pattern 1: Simplest - Factory Methods
print("\n[Pattern 1] Factory Methods - Simplest for common scenarios")
print("-" * 80)
print("""
ocean = OceanSurface.calm_sea(num_grid=256, scene_length=1000.0)
# or
ocean = OceanSurface.rough_sea(num_grid=256, scene_length=1000.0)
# or
ocean = OceanSurface.with_ship_wake(
    ship_velocity=15.0,
    ship_length=100.0,
    num_grid=256,
    scene_length=1000.0,
)
""")

ocean1 = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
height1 = ocean1.generate_height_field(0.0)
print(f"✓ Created calm sea: height range [{height1.min():.3f}, {height1.max():.3f}] m")

ocean2 = OceanSurface.rough_sea(num_grid=64, scene_length=500.0, random_seed=42)
height2 = ocean2.generate_height_field(0.0)
print(f"✓ Created rough sea: height range [{height2.min():.3f}, {height2.max():.3f}] m")

ocean3 = OceanSurface.with_ship_wake(
    ship_velocity=15.0, ship_length=100.0, num_grid=64, scene_length=500.0, random_seed=42
)
height3 = ocean3.generate_height_field(0.0)
print(f"✓ Created ocean with wake: height range [{height3.min():.3f}, {height3.max():.3f}] m")

# Pattern 2: Progressive Disclosure - Start simple, add features
print("\n[Pattern 2] Progressive Disclosure - Start simple, add complexity")
print("-" * 80)
print("""
# Start with basic ocean
ocean = OceanSurface.calm_sea()

# Add wake later
wake = Wake.create(ship_length=100.0, ship_velocity=15.0, scene_length=1000.0, num_grid=256)
ocean.add_wake(wake)

# Add swell later
ocean.add_swell(amplitude=0.3, wavelength=80.0, direction=(1.0, 0.0))

# Remove components if needed
ocean.remove_wake()
ocean.remove_swell()
""")

ocean4 = OceanSurface.calm_sea(num_grid=64, scene_length=500.0, random_seed=42)
print(f"✓ Started with calm sea")

wake4 = Wake.create(ship_length=100.0, ship_velocity=15.0, scene_length=500.0, num_grid=64)
ocean4.add_wake(wake4)
height4a = ocean4.generate_height_field(0.0)
print(f"✓ Added wake: height range [{height4a.min():.3f}, {height4a.max():.3f}] m")

ocean4.add_swell(amplitude=0.3, wavelength=80.0, direction=(1.0, 0.0))
height4b = ocean4.generate_height_field(0.0)
print(f"✓ Added swell: height range [{height4b.min():.3f}, {height4b.max():.3f}] m")

ocean4.remove_wake()
height4c = ocean4.generate_height_field(0.0)
print(f"✓ Removed wake: height range [{height4c.min():.3f}, {height4c.max():.3f}] m")

# Pattern 3: Configuration Objects - Full control
print("\n[Pattern 3] Configuration Objects - Maximum control and clarity")
print("-" * 80)
print("""
ocean_config = OceanConfig(
    num_grid=512,
    scene_length=2000.0,
    wind_speed=15.0,
    wave_amplitude=1.5,
    choppiness=0.7,
    wind_direction=(1.0, 0.5),
    smooth=True,
    random_seed=42,
)

wake_config = WakeConfig(
    enabled=True,
    velocity_ship=18.0,
    length_ship=120.0,
    beam_ship=18.0,
    draft_ship=6.0,
    initial_position=(0.0, 0.0),
    rotation=15.0,
    amplitude_scale=1.2,
)

swell_config = SwellConfig(
    enabled=True,
    amplitude=0.3,
    wavelength=120.0,
    direction=(0.8, 0.6),
)

ocean = OceanSurface(ocean_config, wake_config, swell_config)
""")

ocean_config5 = OceanConfig(
    num_grid=64,
    scene_length=500.0,
    wind_speed=15.0,
    wave_amplitude=1.5,
    choppiness=0.7,
    random_seed=42,
)

wake_config5 = WakeConfig(
    enabled=True,
    velocity_ship=18.0,
    length_ship=120.0,
    beam_ship=18.0,
    draft_ship=6.0,
    rotation=15.0,
)

swell_config5 = SwellConfig(
    enabled=True,
    amplitude=0.3,
    wavelength=120.0,
)

ocean5 = OceanSurface(ocean_config5, wake_config5, swell_config5)
height5 = ocean5.generate_height_field(0.0)
print(f"✓ Created fully configured ocean: height range [{height5.min():.3f}, {height5.max():.3f}] m")

# Pattern 4: Hybrid - Factory + Customization
print("\n[Pattern 4] Hybrid Approach - Factory method + customization")
print("-" * 80)
print("""
# Start with factory method
ocean = OceanSurface.with_ship_wake(
    ship_velocity=15.0,
    ship_length=100.0,
    num_grid=256,
    scene_length=1000.0,
    wind_speed=12.0,
)

# Customize further
ocean.add_swell(amplitude=0.5, wavelength=100.0)
ocean.ocean_config.smooth = True
ocean.ocean_config.choppiness = 0.8
""")

ocean6 = OceanSurface.with_ship_wake(
    ship_velocity=15.0,
    ship_length=100.0,
    num_grid=64,
    scene_length=500.0,
    wind_speed=12.0,
    random_seed=42,
)
ocean6.add_swell(amplitude=0.5, wavelength=100.0)
height6 = ocean6.generate_height_field(0.0)
print(f"✓ Created and customized: height range [{height6.min():.3f}, {height6.max():.3f}] m")

# Comparison Table
print("\n" + "=" * 80)
print("USAGE PATTERN COMPARISON")
print("=" * 80)

comparison = """
┌─────────────────────┬──────────────┬─────────────┬───────────────┬──────────────┐
│ Pattern             │ Lines of Code│ Flexibility │ Readability   │ Best For     │
├─────────────────────┼──────────────┼─────────────┼───────────────┼──────────────┤
│ Factory Methods     │      1-3     │     ⭐⭐     │     ⭐⭐⭐⭐⭐   │ Quick start  │
│ Progressive         │      3-10    │    ⭐⭐⭐⭐   │     ⭐⭐⭐⭐    │ Prototyping  │
│ Config Objects      │     10-25    │   ⭐⭐⭐⭐⭐   │     ⭐⭐⭐     │ Production   │
│ Hybrid              │      4-8     │    ⭐⭐⭐⭐   │    ⭐⭐⭐⭐    │ Most cases   │
└─────────────────────┴──────────────┴─────────────┴───────────────┴──────────────┘

Recommendations:
• Beginners: Start with Factory Methods
• Exploratory work: Use Progressive Disclosure
• Production code: Use Config Objects for explicit contracts
• General usage: Hybrid approach balances simplicity and control
"""

print(comparison)

# Performance comparison
print("\n" + "=" * 80)
print("FEATURE AVAILABILITY BY PATTERN")
print("=" * 80)

features = """
                           Factory  Progressive  Config   Hybrid
                           Methods  Disclosure   Objects
Ocean customization          ✓         ✓          ✓       ✓
Wake component              ✓✓        ✓✓         ✓✓      ✓✓
Swell component              ~         ✓          ✓       ✓
Dynamic modification         ~         ✓          ~       ✓
Full parameter control       ~         ~          ✓       ~
Backward compatible          ✓         ✓          ✓       ✓
Type hints/IDE support       ✓         ✓         ✓✓       ✓

Legend: ✓✓ Excellent, ✓ Good, ~ Limited, - Not available
"""

print(features)

print("\n" + "=" * 80)
print("Conclusion: Choose the pattern that best fits your use case!")
print("All patterns maintain backward compatibility with legacy code.")
print("=" * 80)
