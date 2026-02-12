import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os

# load saved data
data_full = np.load('./full_scene_RANGE_DOPPLER_1bounce.npy')
data_lead = np.load('./lead_car_RANGE_DOPPLER_1bounce.npy')
data_semi = np.load('./semi_RANGE_DOPPLER_1bounce.npy')
data_terrain = np.load('./terrain_RANGE_DOPPLER_1bounce.npy')
# data is in format [time_idx,tx,rx,doppler_bin,range_bin]

export_gif = False

num_time_steps = data_full.shape[0]

# Dynamic range setting (in dB)
dynamic_range_db = 100

# Calculate vmin and vmax based on the maximum value across all data
all_data_db = [20*np.log10(np.abs(data_full)+1e-15),
               20*np.log10(np.abs(data_lead)+1e-15),
               20*np.log10(np.abs(data_semi)+1e-15),
               20*np.log10(np.abs(data_terrain)+1e-15)]
vmax = max([np.max(d) for d in all_data_db])
vmin = vmax - dynamic_range_db

# Create figure and axes
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Initialize plots with dynamic range
im0 = axs[0, 0].imshow(20*np.log10(np.abs(data_full[0,0,0])+1e-15), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
axs[0, 0].set_title('Full Scene (Time: 0)')
im1 = axs[0, 1].imshow(20*np.log10(np.abs(data_lead[0,0,0])+1e-15), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
axs[0, 1].set_title('Lead Car Only (Time: 0)')
im2 = axs[1, 0].imshow(20*np.log10(np.abs(data_semi[0,0,0])+1e-15), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
axs[1, 0].set_title('Semi Truck Only (Time: 0)')
im3 = axs[1, 1].imshow(20*np.log10(np.abs(data_terrain[0,0,0])+1e-15), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
axs[1, 1].set_title('Terrain Only (Time: 0)')

for ax in axs.flat:
    ax.set_xlabel('Range Bins')
    ax.set_ylabel('Doppler Bins')

plt.tight_layout()

# Animation update function
def update(time_idx):
    im0.set_array(20*np.log10(np.abs(data_full[time_idx,0,0])+1e-15))
    axs[0, 0].set_title(f'Full Scene (Time: {time_idx})')
    im1.set_array(20*np.log10(np.abs(data_lead[time_idx,0,0])+1e-15))
    axs[0, 1].set_title(f'Lead Car Only (Time: {time_idx})')
    im2.set_array(20*np.log10(np.abs(data_semi[time_idx,0,0])+1e-15))
    axs[1, 0].set_title(f'Semi Truck Only (Time: {time_idx})')
    im3.set_array(20*np.log10(np.abs(data_terrain[time_idx,0,0])+1e-15))
    axs[1, 1].set_title(f'Terrain Only (Time: {time_idx})')
    return [im0, im1, im2, im3]
ani = animation.FuncAnimation(fig, update, frames=num_time_steps, interval=100, blit=True)
# Create animation
if export_gif:
    

    # Save as GIF
    print("Saving animation as GIF...")
    ani.save('range_doppler_comparison.gif', writer='pillow', fps=10)
    print(f"Animation saved as 'range_doppler_comparison.gif' ({num_time_steps} frames)")

plt.show()


# create a new plot that shows data_full, sum of components, and their difference
fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))

# Calculate sum of components
data_sum = data_lead + data_semi + data_terrain

# Initialize plots
im_full2 = axs2[0].imshow(20*np.log10(np.abs(data_full[0,0,0])+1e-15), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
axs2[0].set_title('Full Scene (Time: 0)')
axs2[0].set_xlabel('Range Bins')
axs2[0].set_ylabel('Doppler Bins')

im_sum = axs2[1].imshow(20*np.log10(np.abs(data_sum[0,0,0])+1e-15), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
axs2[1].set_title('Sum of Components (Time: 0)')
axs2[1].set_xlabel('Range Bins')
axs2[1].set_ylabel('Doppler Bins')

# For difference, calculate in complex space first, then take magnitude
# This properly accounts for phase differences and constructive/destructive interference
complex_difference = data_full[0,0,0] - data_sum[0,0,0]
difference_db = 20*np.log10(np.abs(complex_difference)+1e-30)

# Calculate difference dynamic range separately for better visualization
diff_vmax = np.max(np.abs(complex_difference))
# diff_vmin = diff_vmax - dynamic_range_db

print(f"Time 0 - Max Full: {np.max(20*np.log10(np.abs(data_full[0,0,0])+1e-30)):.2f} dB")
print(f"Time 0 - Max Sum: {np.max(20*np.log10(np.abs(data_sum[0,0,0])+1e-30)):.2f} dB")
print(f"Time 0 - Max Difference: {diff_vmax:.2f} dB")
print(f"Time 0 - Relative difference: {diff_vmax - np.max(20*np.log10(np.abs(data_full[0,0,0])+1e-30)):.2f} dB")

im_diff = axs2[2].imshow(np.abs(complex_difference), cmap='jet', aspect='auto')
axs2[2].set_title('Difference (|Full - Sum|) (Time: 0)')
axs2[2].set_xlabel('Range Bins')
axs2[2].set_ylabel('Doppler Bins')
# add colorbars
fig2.colorbar(im_full2, ax=axs2[0], fraction=0.046, pad=0.04)
fig2.colorbar(im_sum, ax=axs2[1], fraction=0.046, pad=0.04)
fig2.colorbar(im_diff, ax=axs2[2], fraction=0.046, pad=0.04)

plt.tight_layout()

def update_compare(time_idx):
    im_full2.set_array(20*np.log10(np.abs(data_full[time_idx,0,0])+1e-15))
    axs2[0].set_title(f'Full Scene (Time: {time_idx})')
    
    im_sum.set_array(20*np.log10(np.abs(data_sum[time_idx,0,0])+1e-15))
    axs2[1].set_title(f'Sum of Components (Time: {time_idx})')
    
    # Compute difference in complex domain, then take magnitude
    complex_difference = data_full[time_idx,0,0] - data_sum[time_idx,0,0]
    difference_db = 20*np.log10(np.abs(complex_difference)+1e-30)
    im_diff.set_array(np.abs(complex_difference))
    axs2[2].set_title(f'Difference (|Full - Sum|) (Time: {time_idx})')

    # get the index where the peak value occurs in the full data.
    # max_idx = np.unravel_index(np.argmax(np.abs(data_full[time_idx,0,0])), data_full[time_idx,0,0].shape)
    # print(f"Time {time_idx}: Peak at Doppler bin {max_idx[0]}, Range bin {max_idx[1]} with value {data_full[time_idx,0,0][max_idx]}")
    # # print the same index and value for the sum data
    # print(f"Time {time_idx}: Sum Peak at Doppler bin {max_idx[0]}, Range bin {max_idx[1]} with value {data_sum[time_idx,0,0][max_idx]}")
    # # print the same data for the difference
    # print(f"Time {time_idx}: Difference at Peak: {difference_db[max_idx]} dB")
    
    return [im_full2, im_sum, im_diff]

ani2 = animation.FuncAnimation(fig2, update_compare, frames=num_time_steps, interval=100, blit=True)

if export_gif:
    print("Saving comparison animation as GIF...")
    ani2.save('range_doppler_full_vs_sum.gif', writer='pillow', fps=10)
    print(f"Comparison animation saved as 'range_doppler_full_vs_sum.gif' ({num_time_steps} frames)")
plt.show()


# Create a third plot: Full scene with segmented mask
fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))

# Create segmentation mask
# Assign each component to a color based on which has the highest magnitude
def create_segmentation_mask(time_idx):
    # Get magnitude of each component
    mag_lead = np.abs(data_lead[time_idx, 0, 0])
    mag_semi = np.abs(data_semi[time_idx, 0, 0])
    mag_terrain = np.abs(data_terrain[time_idx, 0, 0])
    mag_full = np.abs(data_full[time_idx, 0, 0])
    
    # Stack them and find which component has maximum value at each pixel
    stacked = np.stack([mag_lead, mag_semi, mag_terrain], axis=0)
    max_component = np.argmax(stacked, axis=0)
    
    # Set to 0 (background) where full scene is above threshold but all components are below
    threshold = 1e-5
    all_below_threshold = (mag_lead < threshold) & (mag_semi < threshold) & (mag_terrain < threshold)
    full_above_threshold = mag_full >= threshold
    
    # Create mask: 0=background (full scene signal not in any component), 1=lead, 2=semi, 3=terrain
    mask = max_component + 1
    mask[all_below_threshold & full_above_threshold] = 0
    mask[~full_above_threshold] = 0  # Also set to 0 where full scene is below threshold
    
    return mask

# Initialize plots
im_full3 = axs3[0].imshow(20*np.log10(np.abs(data_full[0,0,0])+1e-10), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
axs3[0].set_title('Full Scene (Time: 0)')
axs3[0].set_xlabel('Range Bins')
axs3[0].set_ylabel('Doppler Bins')

# Create custom colormap for segmentation
from matplotlib.colors import ListedColormap
colors = ['black', 'red', 'green', 'blue']  # background, lead, semi, terrain
cmap_segmentation = ListedColormap(colors)

mask_0 = create_segmentation_mask(0)
im_mask = axs3[1].imshow(mask_0, cmap=cmap_segmentation, aspect='auto', vmin=0, vmax=3)
axs3[1].set_title('Segmentation Mask (Time: 0)')
axs3[1].set_xlabel('Range Bins')
axs3[1].set_ylabel('Doppler Bins')

# Add colorbar with labels
cbar = fig3.colorbar(im_mask, ax=axs3[1], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(['Background', 'Lead Car', 'Semi Truck', 'Terrain'])

fig3.colorbar(im_full3, ax=axs3[0], fraction=0.046, pad=0.04)

plt.tight_layout()

def update_segmentation(time_idx):
    im_full3.set_array(20*np.log10(np.abs(data_full[time_idx,0,0])+1e-20))
    axs3[0].set_title(f'Full Scene (Time: {time_idx})')
    
    mask = create_segmentation_mask(time_idx)
    im_mask.set_array(mask)
    axs3[1].set_title(f'Segmentation Mask (Time: {time_idx})')
    
    return [im_full3, im_mask]

ani3 = animation.FuncAnimation(fig3, update_segmentation, frames=num_time_steps, interval=100, blit=True)

if export_gif:
    print("Saving segmentation animation as GIF...")
    ani3.save('range_doppler_segmentation.gif', writer='pillow', fps=10)
    print(f"Segmentation animation saved as 'range_doppler_segmentation.gif' ({num_time_steps} frames)")
plt.show()



