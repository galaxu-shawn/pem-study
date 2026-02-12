"""Test script to visualize wake component blending and check for discontinuities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pem_utilities.seastate3 import OceanSurface

# Create ocean surface with wake
num_grid = 400
scene_length = 250.0

ocean = OceanSurface(
    num_grid=num_grid,
    scene_length=scene_length,
    wind_speed=0.0,  # No wind to see wake clearly
    wave_amplitude=0.0,  # No waves
    choppiness=0.0,
    wind_direction=(1.0, 0.0),
    include_wake=True,
    velocity_ship=12.0,
    length_ship=110.0,
    beam_ship=20.3,
    draft_ship=6.5,
    initial_wake_position=(0.0, 0.0),
    wake_amplitude_scale=2.0,
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

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Top-down view of complete wake
ax1 = fig.add_subplot(gs[0, :])
levels = np.linspace(height.min(), height.max(), 50)
im1 = ax1.contourf(xx, yy, height, levels=levels, cmap='RdBu_r')
ax1.contour(xx, yy, height, levels=20, colors='black', linewidths=0.3, alpha=0.3)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Complete Wake Pattern (Improved Blending)', fontsize=12, fontweight='bold')
ax1.set_aspect('equal')
ax1.axhline(y=0, color='yellow', linestyle='--', alpha=0.7, linewidth=2, label='Ship center')
ax1.axhline(y=-55, color='green', linestyle='--', alpha=0.5, label='Stern')
ax1.axhline(y=55, color='cyan', linestyle='--', alpha=0.5, label='Bow')
ax1.axvline(x=0, color='yellow', linestyle='--', alpha=0.7, linewidth=2)
ax1.legend(loc='upper right')
plt.colorbar(im1, ax=ax1, label='Height (m)')

# 2. Longitudinal cross-section (centerline)
ax2 = fig.add_subplot(gs[1, 0])
x_center_idx = num_grid // 2
ax2.plot(y, height[:, x_center_idx], 'b-', linewidth=2, label='Centerline')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=-55, color='green', linestyle='--', alpha=0.5, label='Stern')
ax2.axvline(x=55, color='cyan', linestyle='--', alpha=0.5, label='Bow')
ax2.axvline(x=-27.5, color='orange', linestyle=':', alpha=0.5, label='Quarter')
ax2.set_xlabel('Y - Longitudinal (m)')
ax2.set_ylabel('Height (m)')
ax2.set_title('Centerline Profile (Check for discontinuities)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Lateral cross-section at stern (y = -55m)
ax3 = fig.add_subplot(gs[1, 1])
y_stern_idx = np.argmin(np.abs(y + 55))
ax3.plot(x, height[y_stern_idx, :], 'r-', linewidth=2, label=f'Y = {y[y_stern_idx]:.1f}m')
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.axvline(x=10.15, color='purple', linestyle='--', alpha=0.5, label='Beam/2')
ax3.axvline(x=-10.15, color='purple', linestyle='--', alpha=0.5)
ax3.set_xlabel('X - Lateral (m)')
ax3.set_ylabel('Height (m)')
ax3.set_title('Cross-section at Stern')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Lateral cross-section behind ship (y = -80m)
ax4 = fig.add_subplot(gs[1, 2])
y_behind_idx = np.argmin(np.abs(y + 80))
ax4.plot(x, height[y_behind_idx, :], 'g-', linewidth=2, label=f'Y = {y[y_behind_idx]:.1f}m')
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
kelvin_angle = np.deg2rad(19.47)
kelvin_width = 80 * np.tan(kelvin_angle)
ax4.axvline(x=kelvin_width, color='red', linestyle='--', alpha=0.5, label=f'Kelvin angle')
ax4.axvline(x=-kelvin_width, color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('X - Lateral (m)')
ax4.set_ylabel('Height (m)')
ax4.set_title('Cross-section in Wake')
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. Gradient magnitude (to detect discontinuities)
ax5 = fig.add_subplot(gs[2, 0])
dy, dx = np.gradient(height)
grad_mag = np.sqrt(dx**2 + dy**2)
im5 = ax5.contourf(xx, yy, grad_mag, levels=50, cmap='hot')
ax5.set_xlabel('X (m)')
ax5.set_ylabel('Y (m)')
ax5.set_title('Gradient Magnitude (discontinuities show as bright spots)')
ax5.set_aspect('equal')
plt.colorbar(im5, ax=ax5, label='|∇h|')

# 6. Detail view near stern (blending region)
ax6 = fig.add_subplot(gs[2, 1])
zoom_range = 40
x_zoom = np.abs(x) <= zoom_range
y_zoom = (y >= -90) & (y <= 10)
x_z = x[x_zoom]
y_z = y[y_zoom]
xx_z, yy_z = np.meshgrid(x_z, y_z)
height_zoom = height[np.ix_(y_zoom, x_zoom)]
im6 = ax6.contourf(xx_z, yy_z, height_zoom, levels=30, cmap='RdBu_r')
ax6.contour(xx_z, yy_z, height_zoom, levels=15, colors='black', linewidths=0.5, alpha=0.4)
ax6.set_xlabel('X (m)')
ax6.set_ylabel('Y (m)')
ax6.set_title('Detail: Stern Region (blending zone)')
ax6.set_aspect('equal')
ax6.axhline(y=0, color='yellow', linestyle='--', alpha=0.7)
ax6.axhline(y=-55, color='green', linestyle='--', alpha=0.7)
plt.colorbar(im6, ax=ax6, label='Height (m)')

# 7. Statistics and smoothness metrics
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

# Calculate smoothness metrics
grad_mean = np.mean(grad_mag)
grad_std = np.std(grad_mag)
grad_max = np.max(grad_mag)

# Find regions with high gradients (potential discontinuities)
high_grad_threshold = grad_mean + 3 * grad_std
high_grad_points = np.sum(grad_mag > high_grad_threshold)

stats_text = f"""Wake Statistics:

Height Range:
  Max: {height.max():.3f} m
  Min: {height.min():.3f} m
  P-to-P: {height.max() - height.min():.3f} m

Smoothness Metrics:
  Gradient Mean: {grad_mean:.4f}
  Gradient Std: {grad_std:.4f}
  Gradient Max: {grad_max:.4f}
  
Discontinuity Check:
  High gradient points: {high_grad_points}
  (threshold: {high_grad_threshold:.4f})
  
  {'✓ SMOOTH' if high_grad_points < 100 else '⚠ CHECK EDGES'}

Ship Parameters:
  Length: 110 m
  Beam: 20.3 m
  Velocity: 12 m/s
  Froude: {12/np.sqrt(9.81*110):.3f}
"""

ax7.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Wake Component Blending Analysis', fontsize=14, fontweight='bold', y=0.995)
# plt.savefig('wake_blending_analysis.png', dpi=150, bbox_inches='tight')
# print("\nVisualization saved as 'wake_blending_analysis.png'")
print(f"\nSmoothness Check: {high_grad_points} points with high gradients")
print(f"{'✓ Wake appears smooth!' if high_grad_points < 100 else '⚠ Some discontinuities detected'}")

plt.show()
