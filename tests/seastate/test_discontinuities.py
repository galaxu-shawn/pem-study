"""Test to specifically check for discontinuities at component boundaries."""

import numpy as np
import matplotlib.pyplot as plt
from pem_utilities.seastate3 import OceanSurface

# Create ocean surface with wake (no background waves)
num_grid = 400
scene_length = 250.0

ocean = OceanSurface(
    num_grid=num_grid,
    scene_length=scene_length,
    wind_speed=0.0,
    wave_amplitude=0.0,
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

height = ocean.generate_height_field(0.0)

x = np.linspace(-scene_length / 2, scene_length / 2, num_grid)
y = np.linspace(-scene_length / 2, scene_length / 2, num_grid)

# Check for discontinuities along critical boundaries
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Centerline profile (check bow-kelvin-turbulent blending)
ax1 = axes[0, 0]
x_center_idx = num_grid // 2
centerline = height[:, x_center_idx]
ax1.plot(y, centerline, 'b-', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=-55, color='green', linestyle='--', alpha=0.5, label='Stern')
ax1.axvline(x=55, color='cyan', linestyle='--', alpha=0.5, label='Bow')
ax1.axvline(x=-36.7, color='orange', linestyle=':', alpha=0.5, label='L/3 (turbulent start)')
ax1.set_xlabel('Y (m)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Centerline Profile')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Derivative along centerline (sharp peaks = discontinuities)
ax2 = axes[0, 1]
dy_spacing = y[1] - y[0]
centerline_deriv = np.gradient(centerline, dy_spacing)
ax2.plot(y, centerline_deriv, 'r-', linewidth=1.5)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=-55, color='green', linestyle='--', alpha=0.5, label='Stern')
ax2.axvline(x=55, color='cyan', linestyle='--', alpha=0.5, label='Bow')
ax2.set_xlabel('Y (m)')
ax2.set_ylabel('dh/dy')
ax2.set_title('Centerline Derivative (spikes indicate discontinuities)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Cross-sections at different Y positions
ax3 = axes[1, 0]
y_positions = [-80, -55, -30, 0, 30]
colors = ['blue', 'green', 'orange', 'red', 'purple']
for y_pos, color in zip(y_positions, colors):
    idx = np.argmin(np.abs(y - y_pos))
    cross_section = height[idx, :]
    ax3.plot(x, cross_section, color=color, linewidth=1.5, label=f'Y={y_pos}m')
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Height (m)')
ax3.set_title('Lateral Cross-sections')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Check smoothness using Laplacian (detects sharp features)
ax4 = axes[1, 1]
# Compute Laplacian (second derivative test for smoothness)
laplacian = np.zeros_like(height)
for i in range(1, num_grid - 1):
    for j in range(1, num_grid - 1):
        laplacian[i, j] = (height[i+1, j] + height[i-1, j] + 
                          height[i, j+1] + height[i, j-1] - 4*height[i, j])

# Plot histogram of Laplacian values
laplacian_flat = laplacian.flatten()
ax4.hist(laplacian_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Laplacian Value')
ax4.set_ylabel('Frequency')
ax4.set_title('Laplacian Distribution (narrow peak = smooth)')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# Calculate smoothness metrics
std_laplacian = np.std(laplacian)
max_laplacian = np.max(np.abs(laplacian))
outliers = np.sum(np.abs(laplacian) > 3 * std_laplacian)

# Print statistics
print("\n" + "="*60)
print("WAKE SMOOTHNESS ANALYSIS")
print("="*60)
print(f"\nHeight Statistics:")
print(f"  Range: [{height.min():.3f}, {height.max():.3f}] m")
print(f"  Mean: {height.mean():.3f} m")
print(f"  Std: {height.std():.3f} m")

print(f"\nCenterline Derivative Statistics:")
print(f"  Max |dh/dy|: {np.max(np.abs(centerline_deriv)):.4f}")
print(f"  Mean |dh/dy|: {np.mean(np.abs(centerline_deriv)):.4f}")
print(f"  Std dh/dy: {np.std(centerline_deriv):.4f}")

print(f"\nLaplacian Statistics (smoothness indicator):")
print(f"  Std: {std_laplacian:.6f}")
print(f"  Max |∇²h|: {max_laplacian:.6f}")
print(f"  Outliers (>3σ): {outliers} points ({100*outliers/laplacian.size:.2f}%)")

# Check specific boundary regions for discontinuities
stern_region = (y >= -60) & (y <= -50)
stern_idx = np.where(stern_region)[0]
stern_centerline = centerline[stern_idx]
stern_deriv = centerline_deriv[stern_idx]
max_stern_jump = np.max(np.abs(np.diff(stern_centerline)))

print(f"\nBoundary Region Analysis:")
print(f"  Stern region (-60 to -50m):")
print(f"    Max jump: {max_stern_jump:.4f} m")
print(f"    Max derivative: {np.max(np.abs(stern_deriv)):.4f}")

# Overall assessment
print(f"\n{'='*60}")
if outliers < 1000 and max_stern_jump < 0.1:
    print("✓ RESULT: Wake blending is SMOOTH - no significant discontinuities")
elif outliers < 5000:
    print("⚠ RESULT: Wake blending is MOSTLY SMOOTH - minor artifacts")
else:
    print("✗ RESULT: Discontinuities present - needs improvement")
print("="*60 + "\n")

plt.suptitle('Wake Component Blending - Discontinuity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
# plt.savefig('wake_discontinuity_analysis.png', dpi=150, bbox_inches='tight')
# print("Saved: wake_discontinuity_analysis.png\n")

plt.show()
