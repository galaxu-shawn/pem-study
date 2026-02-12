import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipyio
import os
import sys
sys.path.append("..")  # directory above this directory is where api_core exists

import matplotlib.animation as anim
from pem_utilities.load_deepsense_6g import DeepSense_6G

all_beam_angles = np.linspace(-45,45,64)

for sequence_index in range(1, 29):
    responses = []
    ds_6g = DeepSense_6G(csv_file='scenario1.csv',
                         scenario_folder='C:/Users/asligar/OneDrive - ANSYS, Inc/Documents/Applications/DeepSense/Scenario1/')
    ds_6g.load(seq_index=sequence_index)

    num_time_stamps = ds_6g.num_time_stamps
    time_idx = 0

    all_line_x = []
    all_line_y = []
    all_dist = []
    xy_to_use = []
    for time in ds_6g.time_stamps_original:

        beam_idx_at_max_meas = np.argmax(ds_6g.power_vs_index[time_idx])
        angle_meas = all_beam_angles[beam_idx_at_max_meas]
        pos = ds_6g.get_position_original(time_idx)

        x=np.linspace(-10,10,101)
        # x=np.tan(np.deg2rad(angle_meas))*y+pos[0]
        th = np.deg2rad(angle_meas)
        x0 = pos[0]
        y0 = pos[1]
        # x = np.tan()*y+(pos[1]+np.tan(np.deg2rad(angle_meas))*pos[0])
        y = np.tan(th) * x+ y0 + np.tan(th) * x0
        # y = np.tan(th)*x + y0 + np.tan(th)*x0
        all_line_x.append(x)
        all_line_y.append(y)
        # store value of x and y that is closest to the origin
        all_dist = np.sqrt(x**2+y**2)
        idx_at_min = np.argmin(np.abs(all_dist))
        xy_to_use.append([-1*x[idx_at_min], y[idx_at_min]])
        print(x[idx_at_min],y[idx_at_min])
        time_idx+=1
        # car_angle.append(np.rad2deg(np.arctan(pos[1] / pos[0])))
    xy = zip(all_line_x, all_line_y)
    for x,y in xy:
        plt.plot(x,y)
    # plt.plot(xy_to_use)
    plt.show()
    # all_results_abs_measure_norm = []
    # for n in range(len(ds_6g.power_vs_index)):
    #     all_results_abs_measure_norm.append(ds_6g.power_vs_index[n] / np.max(ds_6g.power_vs_index[n]))
    # all_results_abs_measure_norm = np.array(all_results_abs_measure_norm)

#
# for ii in beam_set:
#     full_path = os.path.join(dir_path, f'built_in_beam_pattern_{ii}.mat')
#     # load .mat file format using scipy.io.loadmat
#     data = scipyio.loadmat(full_path)
#     beam_pattern = np.squeeze(data['beam_pattern'])
#     beam_pattern[0] = beam_pattern[1]
#     codebook_pattern[ii] = beam_pattern
#
# measurement_offset_angle = 4 * np.pi / 180
# angle_start = 0 - measurement_offset_angle
# angle_end = np.pi - measurement_offset_angle
# num_of_angle = len(beam_pattern)
# all_angs = np.linspace(angle_start * 180 / np.pi - 90, angle_end * 180 / np.pi - 90, num_of_angle)
#
# beam_scan_angles = nnp.linspace(-45, 45, 64)
#
#
#
#
# fig_count = 1
#
# fig, ax = plt.subplots()
#
# # for ii in [0,1,2]:
# #     plt.plot(all_angs, 20*np.log10(codebook_pattern[ii]), label=f'Beam {ii}')
# line, = ax.plot(all_angs, 20 * np.log10(codebook_pattern[0]), label=f'Beam {0}')
# ax.set_ylim([-40, 0])
#
#
# # q.xlabel('Angle (degree)')
# # plt.ylabel('Magnitude (dB)')
# # plt.title('Codebook Pattern')
# # plt.legend()
# # plt.grid()
# # plt.show()
#
# def animate(i):
#     line.set_ydata(20 * np.log10(codebook_pattern[i]))  # update the data.
#     ax.set_ylim([-40, 0])
#     #
#     #
#     # idx_max = np.argmax(all_results_abs_norm[i])
#     # antenna_beam_selected = peak_beam_direction[idx_max]
#     #
#     # frame_image = PIL.Image.open(ds_6g.frame_image[i])
#     # ax2_handle.set_data(frame_image)
#     # y = all_results_abs_norm[i]
#     # y2 = all_results_abs_measure_norm[i]
#     # for n, b in enumerate(barcollection):
#     #     b.set_height(y[n])
#     #
#     # for n, b in enumerate(barcollection2):
#     #     b.set_height(y2[n])
#
#     ax.set_title(f"mmWave Beam Power Plot, Time Index:{i}", color='white')
#
#
# animation = anim.FuncAnimation(fig, animate, 63, blit=False)
#
# plt.show()
#
# # peak_beam_direction = [133.71,131.43,128.86,128,126.29,125.43,124.57,123.71,121.14,119.43,117.71,116.86,115.14,115.14,113.43,112.57,
# #                        110.00,109.14,107.43,105.71,103.14,103.14,102.29,100.57,98.857,98.,96.286,94.571,93.714,91.143,91.143,86.857,86.857,
# #                        86.0,85.143,83.429,82.571,80.587,79.143,76.571,76.571,75.714,74.0,72.286,71.429,68.857,66.286,67.143,66.286,64.571,
# #                        60.286,59.429,59.429,58.571,57.714,56.857,55.143,53.429,51.714,50.857,50.,49.143,46.571,46.714]
# # peak_beam_direction = np.array(peak_beam_direction)-90
# # print(peak_beam_direction)
# # plt.plot(peak_beam_direction)
# # plt.show()
