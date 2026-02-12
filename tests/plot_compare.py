import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


path = r'C:\Users\asligar\Documents\perceive_em\output\helsinki_sar'
# open results numpy file

# each = 'image_0.25_grid_000.npy'
# image = np.load(os.path.join(path,each))
# # plot image using imshow
# fig, ax = plt.subplots(figsize=(8, 8))
# im = ax.imshow(image, cmap='bone_r', extent=(0, image.shape[0], 0, image.shape[1]), aspect='auto')
# ax.set_title(f'Image for Ray Density: {each}')
# ax.set_xlabel('Cross-Range (m)')
# ax.set_ylabel('Down-Range (m)')
# fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
# plt.tight_layout()
# plt.show()

rayshoot = 'sbr'
# all_rd = ['0.001','0.005','0.01','0.05','0.1','0.25']
all_rd =['0.01','0.05','0.1','0.25','0.35']
all_time = {}
all_image = {}
all_images_db = {}

for each in all_rd:

    time = np.load(os.path.join(path,f'time_rd{each}_shoot{rayshoot}.npy'))
    image = np.load(os.path.join(path,f'image_rd{each}_shoot{rayshoot}.npy'))
    # remove values below -300 from image

    all_time[each] = time
    all_image[each] = image
    all_images_db[each] = 20*np.log10(np.fmax(np.abs(image),1e-20))

# create a 2x2 subplot
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
# plot each image using imshow
# set common vmin and vmax for all images

# compute global vmin/vmax from finite values only to avoid -inf/NaN
finite_mins = []
finite_maxs = []
for img in all_images_db.values():
    flat = img.flatten()
    fm = flat[np.isfinite(flat)]
    if fm.size > 0:
        finite_mins.append(np.min(fm))
        finite_maxs.append(np.max(fm))

vmin = min(finite_mins) if finite_mins else -100.0
vmax = max(finite_maxs) if finite_maxs else 0.0
vmin = vmax - 100  # set vmin to be 40 dB below max
for i, each in enumerate(all_rd):
    ax = axs[i//2, i%2]
    to_plot = 20*np.log10(np.abs(all_image[each]))
    im = ax.imshow(to_plot, cmap='bone_r', vmin=vmin, vmax=vmax, extent=(0, all_image[each].shape[0], 0, all_image[each].shape[1]), aspect='auto')
    ax.set_title(f'Ray Density: {each}\nComputation Time: {all_time[each]:.2f} s')
    ax.set_xlabel('Cross-Range (m)')
    ax.set_ylabel('Down-Range (m)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
    # add some statistics to a text box for each plot. Include min, max, mean, std

    # compute statistics excluding non-finite values
    finite_mask = np.isfinite(to_plot)
    if np.any(finite_mask):
        to_plot_finite = to_plot[finite_mask]
        tmin = np.min(to_plot_finite)
        tmax = np.max(to_plot_finite)
        tmean = np.mean(to_plot_finite)
        tstd = np.std(to_plot_finite)
        mean_str = f'{tmean:.2f}'
        std_str = f'{tstd:.2f}'
    else:
        tmin = np.nan
        tmax = np.nan
        mean_str = 'N/A'
        std_str = 'N/A'

    textstr = '\n'.join((
        f'Min: {tmin:.2f}' if np.isfinite(tmin) else 'Min: N/A',
        f'Max: {tmax:.2f}' if np.isfinite(tmax) else 'Max: N/A',
        f'Mean: {mean_str}',
        f'Std: {std_str}',
    ))  
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    print(textstr)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

plt.tight_layout()
# ensure plots directory exists and save the figure with a descriptive name
plots_dir = os.path.join(path, 'plots')
os.makedirs(plots_dir, exist_ok=True)
rd_list_str = '_'.join([str(r) for r in all_rd])
all_fig_name = os.path.join(plots_dir, f'all_rd_images_shoot{rayshoot}_rdlist_{rd_list_str}.png')
fig.savefig(all_fig_name, dpi=150, bbox_inches='tight')
print(f'Saved all RD images figure to: {all_fig_name}')
plt.show()



# create plot that is 3x1 that shows difference between images using '0.25' as reference
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
ref_image = all_image[all_rd[-1]]
for i, each in enumerate(all_rd[:-1]):
    ax = axs[i//2, i%2]

    diff_image = np.abs(np.abs(ref_image) - np.abs(all_image[each]))
    diff_image_db = 20*np.log10(np.abs(diff_image)+1e-12)
    # add some statistics to a text box for each plot. Include min, max, mean, std
    # compute statistics excluding non-finite values
    finite_mask = np.isfinite(diff_image)
    if np.any(finite_mask):
        d_finite = diff_image[finite_mask]
        dmin = np.min(d_finite)
        dmax = np.max(d_finite)
        dmean = np.mean(d_finite)
        dstd = np.std(d_finite)
        mean_str = f'{dmean:.2f}'
        std_str = f'{dstd:.2f}'
    else:
        dmin = np.nan
        dmax = np.nan
        mean_str = 'N/A'
        std_str = 'N/A'

    textstr = '\n'.join((
        f'Min: {dmin:.2f}' if np.isfinite(dmin) else 'Min: N/A',
        f'Max: {dmax:.2f}' if np.isfinite(dmax) else 'Max: N/A',
        f'Mean: {mean_str}',
        f'Std: {std_str}',
    ))  
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # print(textstr)


    # compute a safe vmax from finite values only
    finite_mask_v = np.isfinite(diff_image)
    vmax_safe = np.mean(diff_image[finite_mask_v]) if np.any(finite_mask_v) else 0.0
    im = ax.imshow(diff_image, cmap='bwr', extent=(0, diff_image.shape[0], 0, diff_image.shape[1]), aspect='auto', vmin=0, vmax=vmax_safe)
    ax.set_title(f'Difference Image: Ray Density {each} vs 0.25')
    ax.set_xlabel('Cross-Range (m)')
    ax.set_ylabel('Down-Range (m)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity Difference')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
plt.tight_layout()
# save difference figure
diff_fig_name = os.path.join(plots_dir, f'diff_images_ref_rd0.5_shoot{rayshoot}_rdlist_{rd_list_str}.png')
fig.savefig(diff_fig_name, dpi=150, bbox_inches='tight')
print(f'Saved diff images figure to: {diff_fig_name}')
plt.show()

# Create CDF plot for all images in all_images_db
fig, ax = plt.subplots(figsize=(12, 8))

# Store percentile values for each ray density
percentile_values = {each: {} for each in all_rd}

for each in all_rd:
    # Flatten the image to 1D array and keep finite values only
    raw_data = all_images_db[each].flatten()
    # keep only finite values and exclude values below -200 dB
    finite_mask = np.isfinite(raw_data)
    thresh_mask = raw_data >= -200.0
    keep_mask = finite_mask & thresh_mask
    if np.any(keep_mask):
        data = raw_data[keep_mask]
    else:
        data = np.array([])

    # Sort the data and calculate cumulative probability
    if data.size > 0:
        sorted_data = np.sort(data)
        cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # Find and store the x-values at 50%, 80%, and 100% percentiles
        p50 = np.percentile(data, 50)
        p80 = np.percentile(data, 80)
        p100 = np.percentile(data, 100)
    else:
        sorted_data = np.array([])
        cumulative_prob = np.array([])
        p50 = np.nan
        p80 = np.nan
        p100 = np.nan
    percentile_values[each]['50%'] = p50
    percentile_values[each]['80%'] = p80
    percentile_values[each]['100%'] = p100
    
    # Plot CDF with legend including percentile values (skip if no finite data)
    p50_label = f'{p50:.1f}' if np.isfinite(p50) else 'N/A'
    p80_label = f'{p80:.1f}' if np.isfinite(p80) else 'N/A'
    p100_label = f'{p100:.1f}' if np.isfinite(p100) else 'N/A'
    label = f'RD: {each} | 50%: {p50_label} dB, 80%: {p80_label} dB, 100%: {p100_label} dB'
    if sorted_data.size > 0:
        ax.plot(sorted_data, cumulative_prob, label=label, linewidth=2)
    else:
        # still add a legend entry indicating no finite data
        ax.plot([], [], label=label + ' (no finite data)', linewidth=0)

# Add horizontal lines for 50%, 80%, and 100% cumulative probability
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='50th Percentile')
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='80th Percentile')
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='100th Percentile')

ax.set_xlabel('Intensity (dB)', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('Cumulative Distribution Function (CDF) of Image Intensities', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
plt.tight_layout()
# save CDF plot
# build a short percentile summary for filename
def safe_pct_str(val):
    return f'{val:.0f}' if np.isfinite(val) else 'nan'

p50s = '_'.join([f'{each}-{safe_pct_str(percentile_values[each]["50%"])}' for each in all_rd])
cdf_fig_name = os.path.join(plots_dir, f'CDF_rdlist_{rd_list_str}_shoot{rayshoot}_p50s_{p50s}.png')
fig.savefig(cdf_fig_name, dpi=150, bbox_inches='tight')
print(f'Saved CDF plot to: {cdf_fig_name}')
plt.show()

rayshoot='sbr'
time = np.load(os.path.join(path,f'time_rd{all_rd[-1]}_shoot{rayshoot}.npy'))
image = np.load(os.path.join(path,f'image_rd{all_rd[-1]}_shoot{rayshoot}.npy'))

time_enhanced = np.load(os.path.join(path,f'time_rd{all_rd[-1]}_shoot{rayshoot}_enhanced.npy'))
image_enhanced = np.load(os.path.join(path,f'image_rd{all_rd[-1]}_shoot{rayshoot}_enhanced.npy'))

# create plot comparing normal and enhanced ray processing for last ray density
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Normal SBR
to_plot = 20*np.log10(np.abs(image)+1e-12)
im1 = axs[0].imshow(to_plot, cmap='bone_r', extent=(0, image.shape[0], 0, image.shape[1]), aspect='auto')
axs[0].set_title(f'Ray Density: {all_rd[-1]} SBR\nComputation Time: {time:.2f} s')
axs[0].set_xlabel('Cross-Range (m)')
axs[0].set_ylabel('Down-Range (m)')
fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label='Intensity')

# Plot 2: Enhanced SBR
to_plot_enhanced = 20*np.log10(np.abs(image_enhanced)+1e-12)
im2 = axs[1].imshow(to_plot_enhanced, cmap='bone_r', extent=(0, image_enhanced.shape[0], 0, image_enhanced.shape[1]), aspect='auto')
axs[1].set_title(f'Ray Density: {all_rd[-1]} SBR Enhanced Ray Processing\nComputation Time: {time_enhanced:.2f} s')
axs[1].set_xlabel('Cross-Range (m)')
axs[1].set_ylabel('Down-Range (m)')
fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, label='Intensity')

# Plot 3: Difference
diff_image = np.abs(np.abs(image) - np.abs(image_enhanced))
diff_image_db = 20*np.log10(np.abs(diff_image)+1e-12)
finite_mask_v = np.isfinite(diff_image)
vmax_safe = np.mean(diff_image[finite_mask_v]) if np.any(finite_mask_v) else 0.0
im3 = axs[2].imshow(diff_image, cmap='bwr', extent=(0, diff_image.shape[0], 0, diff_image.shape[1]), aspect='auto', vmin=0, vmax=vmax_safe)
axs[2].set_title(f'Difference Image: Ray Density {all_rd[-1]} SBR vs Enhanced Ray Processing')
axs[2].set_xlabel('Cross-Range (m)')
axs[2].set_ylabel('Down-Range (m)')
fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04, label='Intensity Difference')

plt.tight_layout()
# save enhanced comparison figure
enhanced_fig_name = os.path.join(plots_dir, f'enhanced_comparison_rd{all_rd[-1]}_shoot{rayshoot}.png')
fig.savefig(enhanced_fig_name, dpi=150, bbox_inches='tight')
print(f'Saved enhanced comparison figure to: {enhanced_fig_name}')
plt.show()



rayshoot='sbr'
time_sbr = np.load(os.path.join(path,f'time_rd{all_rd[-1]}_shoot{rayshoot}.npy'))
image_sbr = np.load(os.path.join(path,f'image_rd{all_rd[-1]}_shoot{rayshoot}.npy'))

rayshoot='grid'
time_grid = np.load(os.path.join(path,f'time_rd{all_rd[-1]}_shoot{rayshoot}.npy'))
image_grid = np.load(os.path.join(path,f'image_rd{all_rd[-1]}_shoot{rayshoot}.npy'))

# create plot comparing sbr and grid ray processing for last ray density
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Normal SBR
to_plot = 20*np.log10(np.abs(image_sbr)+1e-12)
im1 = axs[0].imshow(to_plot, cmap='bone_r', extent=(0, image_sbr.shape[0], 0, image_sbr.shape[1]), aspect='auto')
axs[0].set_title(f'Ray Density: {all_rd[-1]} SBR Rayshoot\nComputation Time: {time_sbr:.2f} s')
axs[0].set_xlabel('Cross-Range (m)')
axs[0].set_ylabel('Down-Range (m)')
fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label='Intensity')

# Plot 2: Enhanced SBR
to_plot_grid = 20*np.log10(np.abs(image_grid)+1e-12)
im2 = axs[1].imshow(to_plot_grid, cmap='bone_r', extent=(0, image_grid.shape[0], 0, image_grid.shape[1]), aspect='auto')
axs[1].set_title(f'Ray Density: {all_rd[-1]} Grid Rayshoot\nComputation Time: {time_grid:.2f} s')
axs[1].set_xlabel('Cross-Range (m)')
axs[1].set_ylabel('Down-Range (m)')
fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, label='Intensity')

# Plot 3: Difference
diff_image = np.abs(np.abs(image_sbr) - np.abs(image_grid))
diff_image_db = 20*np.log10(np.abs(diff_image)+1e-12)
finite_mask_v = np.isfinite(diff_image)
vmax_safe = np.mean(diff_image[finite_mask_v]) if np.any(finite_mask_v) else 0.0
im3 = axs[2].imshow(diff_image, cmap='bwr', extent=(0, diff_image.shape[0], 0, diff_image.shape[1]), aspect='auto', vmin=0, vmax=vmax_safe)
axs[2].set_title(f'Difference Image: Ray Density {all_rd[-1]} SBR vs Grid Rayshoot')
axs[2].set_xlabel('Cross-Range (m)')
axs[2].set_ylabel('Down-Range (m)')
fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04, label='Intensity Difference')

plt.tight_layout()
# save sbr vs grid comparison
sbr_grid_fig_name = os.path.join(plots_dir, f'sbr_vs_grid_rd{all_rd[-1]}.png')
fig.savefig(sbr_grid_fig_name, dpi=150, bbox_inches='tight')
print(f'Saved SBR vs Grid comparison to: {sbr_grid_fig_name}')
plt.show()


##### ZOOOOMED IN VERSIONS #####

# create plot comparing sbr and grid ray processing for last ray density
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Define zoom location (x_min, x_max, y_min, y_max)
zoom_location = (2000, 2500, 2000, 2500)
x_min, x_max, y_min, y_max = zoom_location

# Plot 1: Normal SBR (zoomed)
to_plot_full = 20*np.log10(np.abs(image_sbr)+1e-12)
to_plot = to_plot_full[y_min:y_max, x_min:x_max]
im1 = axs[0].imshow(to_plot, cmap='bone_r', extent=zoom_location, aspect='auto')
axs[0].set_title(f'Ray Density: {all_rd[-1]} SBR Rayshoot\nComputation Time: {time_sbr:.2f} s')
axs[0].set_xlabel('Cross-Range (m)')
axs[0].set_ylabel('Down-Range (m)')
fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label='Intensity')

# Plot 2: Grid Rayshoot (zoomed)
to_plot_grid_full = 20*np.log10(np.abs(image_grid)+1e-12)
to_plot_grid = to_plot_grid_full[y_min:y_max, x_min:x_max]
im2 = axs[1].imshow(to_plot_grid, cmap='bone_r', extent=zoom_location, aspect='auto')
axs[1].set_title(f'Ray Density: {all_rd[-1]} Grid Rayshoot\nComputation Time: {time_grid:.2f} s')
axs[1].set_xlabel('Cross-Range (m)')
axs[1].set_ylabel('Down-Range (m)')
fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, label='Intensity')

# Plot 3: Difference (zoomed)
diff_image_full = np.abs(np.abs(image_sbr) - np.abs(image_grid))
diff_image = diff_image_full[y_min:y_max, x_min:x_max]
finite_mask_v = np.isfinite(diff_image)
vmax_safe = np.mean(diff_image[finite_mask_v]) if np.any(finite_mask_v) else 0.0
im3 = axs[2].imshow(diff_image, cmap='bwr', extent=zoom_location, aspect='auto', vmin=0, vmax=vmax_safe)
axs[2].set_title(f'Difference Image: Ray Density {all_rd[-1]} SBR vs Grid Rayshoot')
axs[2].set_xlabel('Cross-Range (m)')
axs[2].set_ylabel('Down-Range (m)')
fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04, label='Intensity Difference')

plt.tight_layout()
# save zoomed comparison
zoom_fig_name = os.path.join(plots_dir, f'sbr_vs_grid_rd{all_rd[-1]}_zoom_{x_min}-{x_max}_{y_min}-{y_max}.png')
fig.savefig(zoom_fig_name, dpi=150, bbox_inches='tight')
print(f'Saved zoomed SBR vs Grid comparison to: {zoom_fig_name}')
plt.show()



##### ZOOOOMED IN VERSIONS #####

# create plot comparing sbr and grid ray processing for last ray density
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Define zoom location (x_min, x_max, y_min, y_max)
zoom_location = (2000, 2500, 2000, 2500)
x_min, x_max, y_min, y_max = zoom_location

# Plot 1: Normal SBR (zoomed)
to_plot_full = 20*np.log10(np.abs(image_sbr)+1e-12)
to_plot = to_plot_full[y_min:y_max, x_min:x_max]
im1 = axs[0].imshow(to_plot, cmap='bone_r', extent=zoom_location, aspect='auto')
axs[0].set_title(f'Ray Density: {all_rd[-1]} SBR Rayshoot\nComputation Time: {time_sbr:.2f} s')
axs[0].set_xlabel('Cross-Range (m)')
axs[0].set_ylabel('Down-Range (m)')
fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label='Intensity')

# Plot 2: Enhanced SBR- Enhanced (zoomed)
to_plot_grid_full = 20*np.log10(np.abs(image_enhanced)+1e-12)
to_plot_grid = to_plot_grid_full[y_min:y_max, x_min:x_max]
im2 = axs[1].imshow(to_plot_grid, cmap='bone_r', extent=zoom_location, aspect='auto')
axs[1].set_title(f'Ray Density: {all_rd[-1]} Enhanced Rayshoot\nComputation Time: {time_enhanced:.2f} s')
axs[1].set_xlabel('Cross-Range (m)')
axs[1].set_ylabel('Down-Range (m)')
fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, label='Intensity')

# Plot 3: Difference (zoomed)
diff_image_full = np.abs(np.abs(image_sbr) - np.abs(image_grid))
diff_image = diff_image_full[y_min:y_max, x_min:x_max]
finite_mask_v = np.isfinite(diff_image)
vmax_safe = np.mean(diff_image[finite_mask_v]) if np.any(finite_mask_v) else 0.0
im3 = axs[2].imshow(diff_image, cmap='bwr', extent=zoom_location, aspect='auto', vmin=0, vmax=vmax_safe)
axs[2].set_title(f'Difference Image: Ray Density {all_rd[-1]} SBR vs Enhanced Rayshoot')
axs[2].set_xlabel('Cross-Range (m)')
axs[2].set_ylabel('Down-Range (m)')
fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04, label='Intensity Difference')

plt.tight_layout()
# save zoomed comparison
zoom_fig_name = os.path.join(plots_dir, f'sbr_vs_enhanced_rd{all_rd[-1]}_zoom_{x_min}-{x_max}_{y_min}-{y_max}.png')
fig.savefig(zoom_fig_name, dpi=150, bbox_inches='tight')
print(f'Saved zoomed SBR vs Enhanced comparison to: {zoom_fig_name}')
plt.show()

##### INTERACTIVE PLOT WITH DYNAMIC MIN/MAX ADJUSTMENT #####

# Select which image to display interactively (using the last ray density SBR image)
selected_image = 20*np.log10(np.abs(image_sbr)+1e-12)

# Create figure and axis with space for sliders
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Calculate initial vmin and vmax
# compute interactive data_min/data_max from finite values only
sel_flat = selected_image.flatten()
sel_finite = sel_flat[np.isfinite(sel_flat)]
if sel_finite.size > 0:
    data_min = np.min(sel_finite)
    data_max = np.max(sel_finite)
else:
    data_min = 0.0
    data_max = 1.0
initial_vmin = data_max - 100  # 100 dB below max
initial_vmax = data_max

# Create initial plot
im = ax.imshow(selected_image, cmap='bone_r', vmin=initial_vmin, vmax=initial_vmax, 
               extent=(0, selected_image.shape[0], 0, selected_image.shape[1]), aspect='auto')
ax.set_title(f'Interactive SAR Image - Ray Density: {all_rd[-1]} SBR\nAdjust sliders to change dynamic range')
ax.set_xlabel('Cross-Range (m)')
ax.set_ylabel('Down-Range (m)')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity (dB)')

# Create slider axes
ax_vmin = plt.axes([0.1, 0.12, 0.8, 0.03])
ax_vmax = plt.axes([0.1, 0.07, 0.8, 0.03])

# Create sliders
slider_vmin = Slider(ax_vmin, 'Min (dB)', data_min, data_max, valinit=initial_vmin, valstep=0.5)
slider_vmax = Slider(ax_vmax, 'Max (dB)', data_min, data_max, valinit=initial_vmax, valstep=0.5)

# Add text box showing current dynamic range
text_box = ax.text(0.02, 0.98, f'Dynamic Range: {initial_vmax - initial_vmin:.1f} dB', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Update function for sliders
def update(val):
    vmin = slider_vmin.val
    vmax = slider_vmax.val
    
    # Ensure vmin < vmax
    if vmin >= vmax:
        vmin = vmax - 0.5
        slider_vmin.set_val(vmin)
    
    im.set_clim(vmin, vmax)
    text_box.set_text(f'Dynamic Range: {vmax - vmin:.1f} dB')
    fig.canvas.draw_idle()

# Register update function with sliders
slider_vmin.on_changed(update)
slider_vmax.on_changed(update)

# Add instruction text
instruction_text = f'Data Range: {data_min:.1f} to {data_max:.1f} dB'
plt.figtext(0.5, 0.02, instruction_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# save initial interactive view (before slider adjustments)
interactive_fig_name = os.path.join(plots_dir, f'interactive_initial_rd{all_rd[-1]}_shoot{rayshoot}.png')
fig.savefig(interactive_fig_name, dpi=150, bbox_inches='tight')
print(f'Saved initial interactive view to: {interactive_fig_name}')
plt.show()