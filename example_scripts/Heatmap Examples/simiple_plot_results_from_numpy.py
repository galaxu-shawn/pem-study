import numpy as np
import matplotlib.pyplot as plt

# load numpy results
data = np.load(r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Scripting\github\perceive_em\output\heatmap_time_domain_3D_narrow_beam2.npy')

print(f"Data shape: {data.shape}")
print(f"Data dimensions: x={data.shape[0]}, y={data.shape[1]}, z={data.shape[2]}, time={data.shape[3]}")

# Define the time offset to ignore early transients
TIME_OFFSET = 50
print(f"Ignoring first {TIME_OFFSET} time indices to avoid early transients")

# Create a subset of data excluding the first TIME_OFFSET time indices
data_filtered = data[:, :, :, TIME_OFFSET:]

# Find the global maximum across all space and filtered time
global_max_value = np.max(data_filtered)
global_max_indices_filtered = np.unravel_index(np.argmax(data_filtered), data_filtered.shape)
peak_x, peak_y, peak_z, peak_time_filtered = global_max_indices_filtered
peak_time_actual = peak_time_filtered + TIME_OFFSET  # Adjust back to original time index

print(f"\nGlobal Peak Analysis (excluding first {TIME_OFFSET} time indices):")
print(f"Maximum value: {global_max_value}")
print(f"Peak location (x, y, z): ({peak_x}, {peak_y}, {peak_z})")
print(f"Peak occurs at time index: {peak_time_actual} (filtered index: {peak_time_filtered})")

# Find the maximum value at each spatial location across filtered time
max_values_spatial = np.max(data_filtered, axis=3)  # Max across filtered time dimension
spatial_peak_indices = np.unravel_index(np.argmax(max_values_spatial), max_values_spatial.shape)
spatial_peak_x, spatial_peak_y, spatial_peak_z = spatial_peak_indices

print(f"\nSpatial Peak Analysis (max across filtered time for each location):")
print(f"Peak spatial location: ({spatial_peak_x}, {spatial_peak_y}, {spatial_peak_z})")
print(f"Maximum value at this location: {max_values_spatial[spatial_peak_x, spatial_peak_y, spatial_peak_z]}")

# Compare with your original centroid assumption
centroid_idx = [10, 51, 34]
centroid_idx = [11,41, 37]
centroid_idx = [5,21, 20]
centroid_idx = [5,15, 15]
centroid_results = data[centroid_idx[0], centroid_idx[1], centroid_idx[2], :]
centroid_results_filtered = centroid_results[TIME_OFFSET:]  # Apply same filtering
centroid_max_value = np.max(centroid_results_filtered)
centroid_max_time_filtered = np.argmax(centroid_results_filtered)
centroid_max_time_actual = centroid_max_time_filtered + TIME_OFFSET

print(f"\nCentroid Analysis:")
print(f"Assumed centroid location: ({centroid_idx[0]}, {centroid_idx[1]}, {centroid_idx[2]})")
print(f"Maximum value at centroid (filtered): {centroid_max_value}")
print(f"Peak time at centroid: {centroid_max_time_actual} (filtered index: {centroid_max_time_filtered})")

# Check if centroid matches actual peak location
is_centroid_correct = (centroid_idx[0] == spatial_peak_x and 
                      centroid_idx[1] == spatial_peak_y and 
                      centroid_idx[2] == spatial_peak_z)

print(f"\nComparison:")
print(f"Is centroid assumption correct? {is_centroid_correct}")
if not is_centroid_correct:
    print(f"Difference in indices: x={spatial_peak_x - centroid_idx[0]}, y={spatial_peak_y - centroid_idx[1]}, z={spatial_peak_z - centroid_idx[2]}")

# Extract time series at the actual peak location
peak_location_results = data[spatial_peak_x, spatial_peak_y, spatial_peak_z, :]

# Create comparison plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Signal at centroid location
ax1.plot(centroid_results, 'b-', label=f'Centroid ({centroid_idx[0]}, {centroid_idx[1]}, {centroid_idx[2]})', alpha=0.7)
ax1.plot(range(TIME_OFFSET, len(centroid_results)), centroid_results[TIME_OFFSET:], 'b-', linewidth=2, label='Analyzed region')
ax1.axvline(x=centroid_max_time_actual, color='b', linestyle='--', alpha=0.7, label=f'Peak at t={centroid_max_time_actual}')
ax1.axvline(x=TIME_OFFSET, color='gray', linestyle=':', alpha=0.5, label=f'Analysis start (t={TIME_OFFSET})')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Received Signal')
ax1.set_title('Time Domain Signal at Assumed Centroid')
ax1.grid(True)
ax1.legend()

# Plot 2: Signal at actual peak location
ax2.plot(peak_location_results, 'r-', label=f'Actual Peak ({spatial_peak_x}, {spatial_peak_y}, {spatial_peak_z})', alpha=0.7)
ax2.plot(range(TIME_OFFSET, len(peak_location_results)), peak_location_results[TIME_OFFSET:], 'r-', linewidth=2, label='Analyzed region')
peak_time_at_location_filtered = np.argmax(peak_location_results[TIME_OFFSET:])
peak_time_at_location_actual = peak_time_at_location_filtered + TIME_OFFSET
ax2.axvline(x=peak_time_at_location_actual, color='r', linestyle='--', alpha=0.7, label=f'Peak at t={peak_time_at_location_actual}')
ax2.axvline(x=TIME_OFFSET, color='gray', linestyle=':', alpha=0.5, label=f'Analysis start (t={TIME_OFFSET})')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Received Signal')
ax2.set_title('Time Domain Signal at Actual Peak Location')
ax2.grid(True)
ax2.legend()

# Plot 3: Comparison of both signals
ax3.plot(centroid_results, 'b-', label=f'Centroid ({centroid_idx[0]}, {centroid_idx[1]}, {centroid_idx[2]})', alpha=0.5)
ax3.plot(peak_location_results, 'r-', label=f'Peak Location ({spatial_peak_x}, {spatial_peak_y}, {spatial_peak_z})', alpha=0.5)
ax3.plot(range(TIME_OFFSET, len(centroid_results)), centroid_results[TIME_OFFSET:], 'b-', linewidth=2, alpha=0.8)
ax3.plot(range(TIME_OFFSET, len(peak_location_results)), peak_location_results[TIME_OFFSET:], 'r-', linewidth=2, alpha=0.8)
ax3.axvline(x=TIME_OFFSET, color='gray', linestyle=':', alpha=0.5, label=f'Analysis start (t={TIME_OFFSET})')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Received Signal')
ax3.set_title('Comparison: Centroid vs Actual Peak Location (Filtered Analysis)')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

# Optional: Create a 2D heatmap showing the maximum values across one spatial dimension
# This helps visualize where the peak occurs spatially
if data.shape[2] > 1:  # If we have multiple z layers, show middle z slice
    z_slice = data.shape[2] // 2
    max_vals_2d = np.max(data_filtered[:, :, z_slice, :], axis=2)  # Max across filtered time for middle z slice
    
    plt.figure(figsize=(10, 8))
    plt.imshow(max_vals_2d.T, origin='lower', aspect='auto', cmap='hot')
    plt.colorbar(label='Maximum Signal Value')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.title(f'2D Heatmap of Maximum Values (Z slice = {z_slice}, Time > {TIME_OFFSET})')
    
    # Mark the peak location and centroid
    if spatial_peak_z == z_slice:
        plt.plot(spatial_peak_x, spatial_peak_y, 'w*', markersize=15, label='Actual Peak')
    if centroid_idx[2] == z_slice:
        plt.plot(centroid_idx[0], centroid_idx[1], 'bo', markersize=10, label='Assumed Centroid', fillstyle='none')
    
    plt.legend()
    plt.show()

# Additional analysis: Show what would have been found without filtering
print(f"\n--- Comparison with unfiltered analysis ---")
global_max_unfiltered = np.max(data)
global_max_indices_unfiltered = np.unravel_index(np.argmax(data), data.shape)
unfiltered_peak_time = global_max_indices_unfiltered[3]

print(f"Unfiltered global max time index: {unfiltered_peak_time}")
print(f"Filtered global max time index: {peak_time_actual}")
print(f"Time difference: {peak_time_actual - unfiltered_peak_time}")

if unfiltered_peak_time < TIME_OFFSET:
    print(f"WARNING: The unfiltered peak occurred at time {unfiltered_peak_time}, which is within the ignored range (< {TIME_OFFSET})")
    print("This suggests the filtering successfully avoided an early transient peak.")

# create visually appealing plots to show the difference between the two locations
# and the impact of filtering on peak detection
# Plot 1: Time domain signal at assumed centroid location
plt.figure(figsize=(12, 6))
plt.plot(centroid_results, 'b-', label=f'Peak at Drone Location', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Received Signal')
plt.title('Time Domain Signal at Drone')
plt.grid(True)
plt.legend()
plt.show()


tx_pos = [7155.0598575 , 4687.35098269,   44.32637473]
tx2_pos = [7158.27984701, 4783.68505536,   39.59325487]
tx3_pos = [6997.59704313, 4684.60571833,   40.41413828]

# calculate centeroid of the three tx positions
centroid = [7080.3547 ,4734.908 ,41.5173]

# what angle from tx_pos, tx2_pos, tx3_pos to centroid
import math
def calculate_angle(from_pos, to_pos):
    delta_x = to_pos[0] - from_pos[0]
    delta_y = to_pos[1] - from_pos[1]
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg
angle_tx1_to_centroid = calculate_angle(tx_pos, centroid)
angle_tx2_to_centroid = calculate_angle(tx2_pos, centroid)
angle_tx3_to_centroid = calculate_angle(tx3_pos, centroid)
print(f"Angle from TX1 to Centroid: {angle_tx1_to_centroid:.2f} degrees")
print(f"Angle from TX2 to Centroid: {angle_tx2_to_centroid:.2f} degrees")
print(f"Angle from TX3 to Centroid: {angle_tx3_to_centroid:.2f} degrees")
