# OceanSurface & Wake - Quick Reference Card

## 🚀 Getting Started (Choose Your Style)

### Style 1: Instant Results (1 line)
```python
ocean = OceanSurface.calm_sea()
ocean = OceanSurface.rough_sea()
```

### Style 2: With Wake (3 lines)
```python
ocean = OceanSurface.with_ship_wake(
    ship_velocity=15.0, ship_length=100.0
)
```

### Style 3: Build Progressively (4-8 lines)
```python
ocean = OceanSurface.calm_sea()
wake = Wake.create(ship_length=100.0, ship_velocity=15.0, scene_length=1000.0, num_grid=256)
ocean.add_wake(wake)
ocean.add_swell(amplitude=0.3, wavelength=80.0)
```

### Style 4: Full Control (10-20 lines)
```python
ocean_config = OceanConfig(num_grid=512, wind_speed=15.0, ...)
wake_config = WakeConfig(velocity_ship=18.0, length_ship=120.0, ...)
ocean = OceanSurface(ocean_config, wake_config)
```

---

## 📚 Factory Methods

### OceanSurface
| Method | Description | Wind Speed | Wave Amplitude |
|--------|-------------|------------|----------------|
| `calm_sea()` | Gentle waves | 5 m/s | 0.3 m |
| `rough_sea()` | Large waves | 20 m/s | 2.0 m |
| `with_ship_wake()` | Ocean + wake | configurable | configurable |

### Wake
| Method | Description | Auto-calculates |
|--------|-------------|-----------------|
| `create()` | Simplified wake | beam, draft from length |

---

## ⚙️ Configuration Classes

### OceanConfig
```python
OceanConfig(
    num_grid=256,          # Grid resolution
    scene_length=1000.0,   # Domain size (m)
    wind_speed=10.0,       # Wind speed (m/s)
    wave_amplitude=1.0,    # Wave height (m)
    choppiness=0.5,        # Horizontal displacement
    wind_direction=(1,0),  # Wind direction
    smooth=False,          # Mesh smoothing
    random_seed=None,      # Reproducibility
)
```

### WakeConfig
```python
WakeConfig(
    enabled=True,
    velocity_ship=10.0,    # Ship speed (m/s)
    length_ship=110.0,     # Ship length (m)
    beam_ship=20.3,        # Ship beam (m)
    draft_ship=3.5,        # Ship draft (m)
    initial_position=(0,0),# Start position
    rotation=0.0,          # Heading (degrees)
    update_position=True,  # Auto-advance ship
    amplitude_scale=1.0,   # Wake scaling
)
```

### SwellConfig
```python
SwellConfig(
    enabled=True,
    amplitude=0.5,         # Swell height (m)
    wavelength=100.0,      # Swell wavelength (m)
    direction=(1,0),       # Swell direction
    phase=0.0,             # Initial phase (rad)
)
```

---

## 🔧 Methods

### Component Management
```python
ocean.add_wake(wake)           # Add wake component
ocean.remove_wake()            # Remove wake
ocean.add_swell(...)           # Add swell waves
ocean.remove_swell()           # Remove swell
```

### Surface Generation
```python
height = ocean.generate_height_field(t=0.0)
mesh = ocean.generate_mesh(t=0.0)
grid = ocean.create_ocean_surface_plot(t=0.0)
```

---

## 📋 Common Use Cases

### Use Case 1: Static Ocean Scene
```python
ocean = OceanSurface.calm_sea(num_grid=512, scene_length=2000.0)
mesh = ocean.generate_mesh(t=0.0)
mesh.save("ocean.stl")
```

### Use Case 2: Animated Ocean
```python
ocean = OceanSurface.rough_sea(random_seed=42)
for t in range(0, 100, 5):
    height = ocean.generate_height_field(t)
    # ... render or save
```

### Use Case 3: Ship Wake Tracking
```python
ocean = OceanSurface.with_ship_wake(
    ship_velocity=20.0,
    ship_length=150.0,
    ship_position=(0, -500),
    ship_heading=0.0,
)
# Ship automatically moves forward each timestep
for t in range(0, 60, 1):
    height = ocean.generate_height_field(t)
```

### Use Case 4: Dynamic Scene Building
```python
# Start basic
ocean = OceanSurface.calm_sea()

# Add ship later
if ship_appears:
    wake = Wake.create(100.0, 15.0, 1000.0, 256)
    ocean.add_wake(wake)

# Add swell from distant storm
if storm_effect:
    ocean.add_swell(0.8, 150.0, direction=(0.7, 0.3))
```

---

## 💡 Tips & Tricks

### Automatic Ship Dimensions
```python
# Don't specify beam and draft - they're auto-calculated!
ocean = OceanSurface.with_ship_wake(
    ship_velocity=15.0,
    ship_length=100.0,  # beam and draft calculated automatically
)

wake = Wake.create(
    ship_length=100.0,
    ship_velocity=15.0,  # beam and draft calculated automatically
    scene_length=1000.0,
    num_grid=256,
)
```

### Config Reuse
```python
# Save configs for reuse
base_config = OceanConfig(num_grid=512, scene_length=2000.0)

ocean1 = OceanSurface(base_config)
ocean2 = OceanSurface(base_config)  # Same config, different random state
```

### Hybrid Approach
```python
# Start with factory, customize config
ocean = OceanSurface.with_ship_wake(15.0, 100.0)
ocean.ocean_config.choppiness = 0.8  # Customize after creation
ocean.add_swell(0.3, 80.0)
```

---

## ⚠️ Common Pitfalls

### ❌ Don't mix old and new APIs in one call
```python
# Wrong - mixing config and legacy params
ocean = OceanSurface(ocean_config, wind_speed=15.0)  # BAD!
```

### ✅ Choose one approach per instance
```python
# Good - use config
ocean = OceanSurface(ocean_config)

# Good - use legacy
ocean = OceanSurface(num_grid=256, scene_length=1000.0, wind_speed=10.0, ...)
```

### ❌ Don't forget to call generate_height_field
```python
# Wrong - no height generated
ocean = OceanSurface.calm_sea()
# missing: height = ocean.generate_height_field(t=0.0)
```

---

## 🔄 Backward Compatibility

✅ **All old code works unchanged!**

```python
# This still works (legacy API)
ocean = OceanSurface(
    num_grid=400,
    scene_length=250.0,
    wind_speed=12.0,
    wave_amplitude=0.5,
    choppiness=0.1,
    # ... all 30+ parameters
)
```

But consider migrating to the new API for better maintainability.

---

## 📖 Documentation

- `SEASTATE_REFACTORING_GUIDE.md` - Complete guide with examples
- `BEFORE_AFTER_COMPARISON.md` - Side-by-side comparisons
- `REFACTORING_SUMMARY.md` - Technical implementation details
- `usage_pattern_comparison.py` - Live code demonstrations
- `test_refactored_seastate.py` - Test suite

---

## 🎯 When to Use Each Style

| Style | Best For | Learning Curve |
|-------|----------|----------------|
| Factory Methods | Quick prototypes, demos | ⭐ Easy |
| Progressive | Exploratory work, dynamic scenes | ⭐⭐ Medium |
| Config Objects | Production code, complex setups | ⭐⭐⭐ Advanced |
| Hybrid | General use, balanced approach | ⭐⭐ Medium |
| Legacy API | Existing code, no migration time | ⭐⭐⭐ Advanced |

---

**Recommendation:** Start with **Factory Methods**, add features with **Progressive** style, migrate to **Config Objects** for production.
