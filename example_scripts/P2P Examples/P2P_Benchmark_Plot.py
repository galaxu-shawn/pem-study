import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import json
from scipy import interpolate

##### IMPORTANT #####
# Set paths to file
output_summary_name_case1 = f'results_summary_case1_1gpu.json'
output_summary_name_case2a = f'results_summary_case2a_1gpu.json'
output_summary_name_case2a_2gpu = f'results_summary_case2a_2gpu.json'
output_summary_name_case2b = f'results_summary_case2b_1gpu.json'
output_summary_name_case2b_2gpu = f'results_summary_case2b_2gpu.json'
output_summary_name_case2c = f'results_summary_case2c_1gpu.json'
output_summary_name_case2c_2gpu = f'results_summary_case2c_2gpu.json'
output_summary_name_case3 = f'results_summary_case3_1gpu.json'


# CASE 1 - Rx Sweep
output_directory = '../output/benchmark/'  # output_directory: Directory to save the output files
with open(os.path.join(output_directory,output_summary_name_case1)) as f:
    results_dict = json.load(f)

fig, ax = plt.subplots()
x_val = []
y_val = []
for key, value in results_dict.items():
    x_val.append(value['num_rx'])
    y_val.append(value['fps_avg'])
x_val = np.array(x_val)
y_val = np.array(y_val)

ax.plot(x_val, y_val, label='Rx Sweep',color='blue',linestyle='-',marker='o',linewidth=2.0)

ax.set_xlabel('Number Rx')
ax.set_ylabel('Average Frames Per Second')
ax.set_yscale('log')
ax.set_title('Performance Benchmarking - 1 Tx vs Number of Rx')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(output_directory, 'benchmark_case1.png'))
plt.show()


# Case 2a: 1 GPU Tx Element Sweep

with open(os.path.join(output_directory,output_summary_name_case2a)) as f:
    results_dict_1gpu = json.load(f)

# Case 2a: 2 GPU Tx Element Sweep
file_name = 'results_summary_case2a_2gpu.json'
with open(os.path.join(output_directory,output_summary_name_case2a_2gpu)) as f:
    results_dict_2gpu = json.load(f)

fig, ax = plt.subplots()
x_val = []
y_val = []
x_val2 = []
y_val2 = []
for key, value in results_dict_1gpu.items():
    x_val.append(value['num_tx_per_tx'])
    y_val.append(value['fps_avg'])
for key, value in results_dict_2gpu.items():
    x_val2.append(value['num_tx_per_tx'])
    y_val2.append(value['fps_avg'])
ax.plot(x_val, y_val, label='1 GPU',color='blue',linestyle='-',marker='o',linewidth=2.0)
ax.plot(x_val2, y_val2, label='2 GPU',color='red',linestyle='-',marker='o',linewidth=2.0)
ax.set_xlabel('Number of Tx Elements')
ax.set_ylabel('Average Frames Per Second')
ax.set_yscale('log')
ax.set_title('Performance Benchmarking - Number of Tx Elements (1 Tx Device and 1 Rx)')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(output_directory, 'benchmark_case2a.png'))
plt.show()


# CASE 2b Tx Device Sweep - 1GPU
with open(os.path.join(output_directory,output_summary_name_case2b)) as f:
    results_dict_1gpu = json.load(f)

# Case 2b Tx Device Sweep - 2GPU
with open(os.path.join(output_directory,output_summary_name_case2b_2gpu)) as f:
    results_dict_2gpu = json.load(f)

fig, ax = plt.subplots()
x_val = []
y_val = []
x_val2 = []
y_val2 = []
for key, value in results_dict_1gpu.items():
    x_val.append(value['num_tx_units'])
    y_val.append(value['fps_avg'])
for key, value in results_dict_2gpu.items():
    x_val2.append(value['num_tx_units'])
    y_val2.append(value['fps_avg'])
ax.plot(x_val, y_val, label='1 GPU',color='blue',linestyle='-',marker='o',linewidth=2.0)
ax.plot(x_val2, y_val2, label='2 GPU',color='red',linestyle='-',marker='o',linewidth=2.0)
ax.set_xlabel('Number of Tx Devices')
ax.set_ylabel('Average Frames Per Second')
ax.set_yscale('log')
ax.set_title('Performance Benchmarking - Number of Tx Devices (1 Tx Element per Device and 1 Rx)')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(output_directory, 'benchmark_case2b.png'))
plt.show()




# Case 2c Tx Device and Element Sweep - 1GPU
with open(os.path.join(output_directory,output_summary_name_case2c)) as f:
    results_dict_1gpu = json.load(f)

# Tx Device and Element Sweep - 2GPU
with open(os.path.join(output_directory,output_summary_name_case2c_2gpu)) as f:
    results_dict_2gpu = json.load(f)





fig, ax = plt.subplots()
x_val = []
y_val = []
x_val2 = []
y_val2 = []
for key, value in results_dict_1gpu.items():
    if value['num_tx_per_tx']==32:
        x_val.append(value['num_tx_units'])
        y_val.append(value['fps_avg'])
for key, value in results_dict_2gpu.items():
    if value['num_tx_per_tx'] == 32:
        x_val2.append(value['num_tx_units'])
        y_val2.append(value['fps_avg'])
ax.plot(x_val, y_val, label='1 GPU',color='blue',linestyle='-',marker='o',linewidth=2.0)
ax.plot(x_val2, y_val2, label='2 GPU',color='red',linestyle='-',marker='o',linewidth=2.0)
ax.set_xlabel('Number of Tx Devices')
ax.set_ylabel('Average Frames Per Second')
ax.set_yscale('log')
ax.set_title('Performance Benchmarking - Number of Tx Devices (32 Tx Element per Device and 1 Rx)')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(output_directory, 'benchmark_case2c_32Tx.png'))
plt.show()


fig, ax = plt.subplots()
x_val = []
y_val = []
x_val2 = []
y_val2 = []
for key, value in results_dict_1gpu.items():
    if value['num_tx_per_tx']==128:
        x_val.append(value['num_tx_units'])
        y_val.append(value['fps_avg'])
for key, value in results_dict_2gpu.items():
    if value['num_tx_per_tx'] == 128:
        x_val2.append(value['num_tx_units'])
        y_val2.append(value['fps_avg'])
ax.plot(x_val, y_val, label='1 GPU',color='blue',linestyle='-',marker='o',linewidth=2.0)
ax.plot(x_val2, y_val2, label='2 GPU',color='red',linestyle='-',marker='o',linewidth=2.0)
ax.set_xlabel('Number of Tx Devices')
ax.set_ylabel('Average Frames Per Second')
ax.set_yscale('log')
ax.set_title('Performance Benchmarking - Number of Tx Devices (128 Tx Element per Device and 1 Rx)')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(output_directory, 'benchmark_case2c_128Tx.png'))
plt.show()




fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x_val = []
y_val = []
z_val = []
x_val2 = []
y_val2 = []
z_val2 = []
for key, value in results_dict_1gpu.items():
    if value['num_tx_per_tx']>32 and value['num_tx_units']>32:
        x_val.append(value['num_tx_per_tx'])
        y_val.append(value['num_tx_units'])
        z_val.append(value['fps_avg'])
for key, value in results_dict_2gpu.items():
    if value['num_tx_per_tx']>32 and value['num_tx_units']>32:
        x_val2.append(value['num_tx_per_tx'])
        y_val2.append(value['num_tx_units'])
        z_val2.append(value['fps_avg'])
x_val = np.array(x_val)
y_val = np.array(y_val)
z_val = np.array(z_val)
x_val2 = np.array(x_val2)
y_val2 = np.array(y_val2)
z_val2 = np.array(z_val2)

spline = interpolate.Rbf(x_val,y_val,z_val,function='thin-plate')

xi = np.linspace(x_val.min(), x_val.max(), 50)
yi = np.linspace(y_val.min(), y_val.max(), 50)
xi, yi = np.meshgrid(xi, yi)
zi = spline(xi,yi)
ax.set_xlabel('Number of Tx Elements')
ax.set_ylabel('Number of Tx Devices')
ax.set_zlabel('Average Frames Per Second')
ax.set_zscale('log')
ax.set_zticks([1e-1, 1, 2])
ax.plot_surface(xi,yi,zi,cmap=cm.jet)
plt.show()
plt.savefig(os.path.join(output_directory, 'benchmark_case2c.png'))


# CAse 3: Tx Device and Element and Rx Sweep - 1GPU
with open(os.path.join(output_directory,output_summary_name_case3)) as f:
    results_dict_1gpu = json.load(f)

fig, ax = plt.subplots()
x_val = []
y_val = []
x_val2 = []
y_val2 = []
x_val3 = []
y_val3 = []
for key, value in results_dict_1gpu.items():
    if value['num_rx']==32 and value['num_tx_per_tx']==32:
        x_val.append(value['num_tx_units'])
        y_val.append(value['fps_avg'])
    elif value['num_rx']==64 and value['num_tx_per_tx']==32:
        x_val2.append(value['num_tx_units'])
        y_val2.append(value['fps_avg'])
    elif value['num_rx']==128 and value['num_tx_per_tx']==32:
        x_val3.append(value['num_tx_units'])
        y_val3.append(value['fps_avg'])
ax.plot(x_val, y_val, label='32 Rx, 32 Tx Antenna Per Device',color='blue',linewidth=2.0)
ax.plot(x_val2, y_val2, label='64 Rx, 32 Tx Antenna Per Device',color='red',linewidth=2.0)
ax.plot(x_val3, y_val3, label='128 Rx, 32 Tx Antenna Per Device',color='green',linewidth=2.0)
ax.set_xlabel('Number of Tx Devices')
ax.set_ylabel('Average Frames Per Second')
ax.set_yscale('log')
ax.set_title('Performance Benchmarking - Number of Tx Devices (32 Tx Element per Device)')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(output_directory, 'benchmark_case3.png'))
plt.show()




