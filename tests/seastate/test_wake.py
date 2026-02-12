"""Test script to visualize the improved Kelvin wake."""

import numpy as np
import matplotlib.pyplot as plt
from pem_utilities.seastate3 import OceanSurface

# Create ocean surface with wake
num_grid = 400
scene_length = 250.0

ocean = OceanSurface(
    num_grid=num_grid,
    scene_length=scene_length,
    wind_speed=5.0,  # Reduced wind to see wake more clearly
    wave_amplitude=0.2,  # Reduced waves
    choppiness=0.05,
    wind_direction=(1.0, 0.0),
    include_wake=True,
    velocity_ship=12.0,
    length_ship=110.0,
    beam_ship=20.3,
    draft_ship=6.5,
    initial_wake_position=(0.0, 0.0),
    wake_amplitude_scale=2.0,  # Amplify wake for visibility
    wake_rotation=0,
    enable_swell=False,
    smooth=False,
    random_seed=42,
)

# Generate height field at t=0
height = ocean.generate_height_field(0.0)

# Create coordinate grids
x = np.linspace(-scene_length / 2, scene_length / 2, num_grid)
y = np.linspace(-scene_length / 2, scene_length / 2, num_grid)
xx, yy = np.meshgrid(x, y)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Top-down view
im1 = ax1.contourf(xx, yy, height, levels=50, cmap='seismic')
ax1.contour(xx, yy, height, levels=20, colors='black', linewidths=0.3, alpha=0.3)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Improved Kelvin Wake - Top View')
ax1.set_aspect('equal')
ax1.axhline(y=0, color='yellow', linestyle='--', alpha=0.5, label='Ship position')
ax1.axvline(x=0, color='yellow', linestyle='--', alpha=0.5)
ax1.legend()
plt.colorbar(im1, ax=ax1, label='Height (m)')

# Right plot: Cross-section at y = -50m (behind ship)
y_slice_idx = np.argmin(np.abs(y + 50))
ax2.plot(x, height[y_slice_idx, :], 'b-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Height (m)')
ax2.set_title(f'Cross-section at Y = {y[y_slice_idx]:.1f}m (behind ship)')
ax2.grid(True, alpha=0.3)

# Mark Kelvin angle boundaries
kelvin_angle = np.deg2rad(19.47)
distance = abs(y[y_slice_idx])
kelvin_width = distance * np.tan(kelvin_angle)
ax2.axvline(x=kelvin_width, color='red', linestyle='--', alpha=0.5, label=f'Kelvin angle ({19.47:.1f}°)')
ax2.axvline(x=-kelvin_width, color='red', linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
# plt.savefig('wake_visualization.png', dpi=150, bbox_inches='tight')
# print("\nVisualization saved as 'wake_visualization.png'")

# Print wake statistics
print(f"\nWake Statistics:")
print(f"Max height: {height.max():.3f} m")
print(f"Min height: {height.min():.3f} m")
print(f"Peak-to-peak: {height.max() - height.min():.3f} m")
print(f"RMS height: {np.sqrt(np.mean(height**2)):.3f} m")
print(f"\nKelvin angle: {19.47:.2f}°")
print(f"Wake width at y=-50m: ±{kelvin_width:.1f} m")

# Show angular spread by analyzing wake energy distribution
behind_ship = yy < -55  # Focus on wake region
for y_check in [-60, -80, -100]:
    idx = np.argmin(np.abs(y - y_check))
    cross_section = height[idx, :]
    energy = cross_section**2
    total_energy = np.sum(energy)
    if total_energy > 0:
        # Find where 90% of energy is contained
        cumsum = np.cumsum(np.sort(energy)[::-1])
        n90 = np.argmax(cumsum >= 0.9 * total_energy)
        print(f"\nAt y={y_check}m: wake energy spread contains 90% in widest {n90} points")

plt.show()
