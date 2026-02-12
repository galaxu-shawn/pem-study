
import matplotlib.pyplot as plt
import numpy as np


# load our results, this is the output from our simulation, saved in the standard numpy format *.npy. The paths
# are realtive to this script. If you are running this script from a different location, you may need to change the paths
# to point to where your files are located. In python you need to use either "/" or "\\" to separate directories in the path
results = np.load('output.npy',)
x_pos = np.load('grid_x.npy')
y_pos = np.load('grid_y.npy')

# Shape of loaded results to help with understanding the data
# this is going to be all the samples organized in a 2D matrix with [x,y] positions
print(f'Loaded results with shape {results.shape}')

print(f'Loaded x_pos with shape {x_pos.shape}')
print(f'Loaded y_pos with shape {y_pos.shape}')

results_db= 20*np.log10(np.abs(results))

print(f'x_pos min: {x_pos.min()} max: {x_pos.max()}')
print(f'y_pos min: {y_pos.min()} max: {y_pos.max()}')
print(f'Results min: {results_db.min()} max: {results_db.max()}')


# print the results magnitude in dB at the x,y position (or closest to it)
x_y_pos = (5, 10)
x_idx = np.argmin(np.abs(x_pos - x_y_pos[0]))
y_idx = np.argmin(np.abs(y_pos - x_y_pos[1]))
print(f'Results at x={x_y_pos[0]} y={x_y_pos[1]} is {results_db[y_idx, x_idx]} dB')

dynamic_range = 60 #dB from peak
# create a 2d plot of all the results.
fig, ax = plt.subplots()

# rotate the results so that the plot is oriented correctly when usig imshow. imshow plots the matrix with the origin at the top left
# and the y axis increasing downwards. We want the origin at the bottom left and the y axis increasing upwards. our
# results are in the opposite orientation so we need to rotate it 90 degrees.
results_db_for_plot = np.rot90(results_db)
ax.imshow(results_db_for_plot,
            extent=[x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()],
            vmin=results.max()-dynamic_range,
            vmax=results.max(),
            origin='upper',
            cmap='jet',
            aspect = 'auto')
ax.set_title('2D plot of results in dB')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.show()

# plot the values at a specific x position
at_x = 5
x_idx = np.argmin(np.abs(x_pos - at_x))
fig, ax = plt.subplots()
ax.plot(y_pos, results_db[x_idx,:]) # the ':' means all values in the x_idx row
ax.set(xlabel='Y pos (m)', ylabel='Results (dB20)',
       title='Results at Single X position')
ax.grid()
plt.show()