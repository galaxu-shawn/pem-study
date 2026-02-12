import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipyio
import os
import matplotlib.animation as anim

dir_path = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Applications\DeepSense\DeepSense_beam_codebook\DeepSense_beam_codebook\codebook_beams'
beam_set = range(63)

num_of_beam = len(beam_set)
codebook_pattern = np.zeros((num_of_beam, 211))

for ii in beam_set:
    full_path = os.path.join(dir_path, f'built_in_beam_pattern_{ii}.mat')
    # load .mat file format using scipy.io.loadmat
    data = scipyio.loadmat(full_path)
    beam_pattern = np.squeeze(data['beam_pattern'])
    beam_pattern[0] = beam_pattern[1]
    codebook_pattern[ii] = beam_pattern

measurement_offset_angle = 4 * np.pi / 180
angle_start = 0 - measurement_offset_angle
angle_end = np.pi - measurement_offset_angle
num_of_angle = len(beam_pattern)
all_angs = np.linspace(angle_start * 180 / np.pi - 90, angle_end * 180 / np.pi - 90, num_of_angle)

test = np.flip(np.linspace(-45, 45, 64))
print(test[4])
print(test[29])
print(test[58])

# print(np.round(test, 1))

peak_beam_direction = [-45., -43.6, -42.1, -40.7, -39.3, -37.9, -36.4, -35., -33.6, -32.1, -30.7, -29.3,
                       -27.9, -26.4, -25., -23.6, -22.1, -20.7, -19.3, -17.9, -16.4, -15., -13.6 - 12.1,
                       -10.7, -9.3, -7.9, -6.4, -5., -3.6, -2.1, -0.7, 0.7, 2.1, 3.6, 5.,
                       6.4, 7.9, 9.3, 10.7, 12.1, 13.6, 15., 16.4, 17.9, 19.3, 20.7, 22.1,
                       23.6, 25., 26.4, 27.9, 29.3, 30.7, 32.1, 33.6, 35., 36.4, 37.9, 39.3,
                       40.7, 42.1, 43.6, 45.]

fig_count = 1

fig, ax = plt.subplots()

# for ii in [0,1,2]:
#     plt.plot(all_angs, 20*np.log10(codebook_pattern[ii]), label=f'Beam {ii}')
line, = ax.plot(all_angs, 20 * np.log10(codebook_pattern[0]), label=f'Beam {0}')
ax.set_ylim([-40, 0])


# q.xlabel('Angle (degree)')
# plt.ylabel('Magnitude (dB)')
# plt.title('Codebook Pattern')
# plt.legend()
# plt.grid()
# plt.show()

def animate(i):
    line.set_ydata(20 * np.log10(codebook_pattern[i]))  # update the data.
    ax.set_ylim([-40, 0])
    #
    #
    # idx_max = np.argmax(all_results_abs_norm[i])
    # antenna_beam_selected = peak_beam_direction[idx_max]
    #
    # frame_image = PIL.Image.open(ds_6g.frame_image[i])
    # ax2_handle.set_data(frame_image)
    # y = all_results_abs_norm[i]
    # y2 = all_results_abs_measure_norm[i]
    # for n, b in enumerate(barcollection):
    #     b.set_height(y[n])
    #
    # for n, b in enumerate(barcollection2):
    #     b.set_height(y2[n])

    ax.set_title(f"mmWave Beam Power Plot, Time Index:{i}", color='white')


animation = anim.FuncAnimation(fig, animate, 63, blit=False)

plt.show()

# peak_beam_direction = [133.71,131.43,128.86,128,126.29,125.43,124.57,123.71,121.14,119.43,117.71,116.86,115.14,115.14,113.43,112.57,
#                        110.00,109.14,107.43,105.71,103.14,103.14,102.29,100.57,98.857,98.,96.286,94.571,93.714,91.143,91.143,86.857,86.857,
#                        86.0,85.143,83.429,82.571,80.587,79.143,76.571,76.571,75.714,74.0,72.286,71.429,68.857,66.286,67.143,66.286,64.571,
#                        60.286,59.429,59.429,58.571,57.714,56.857,55.143,53.429,51.714,50.857,50.,49.143,46.571,46.714]
# peak_beam_direction = np.array(peak_beam_direction)-90
# print(peak_beam_direction)
# plt.plot(peak_beam_direction)
# plt.show()
