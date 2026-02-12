import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

output_path = '../output/'
all_results = np.load(os.path.join(output_path,'all_results.npy'))
freq_domain = np.load(os.path.join(output_path,'freq_domain.npy'))
pulse_domain = np.load(os.path.join(output_path,'pulse_domain.npy'))
time_domain = np.load(os.path.join(output_path,'simulation_timestamps.npy'))

# Shape of loaded results to help with understanding the data
# this is going to be all the samples organized in a 2D matrix with [x,y] positions
print(f'Loaded results with shape {all_results.shape}')

print(f'Number of Simulations {all_results.shape[0]}')
print(f'Number of Tx {all_results.shape[1]}')
print(f'Number of Rx {all_results.shape[2]}')
print(f'Number of Channel Soundings {all_results.shape[3]}')
print(f'Number of Frequency Samples {all_results.shape[4]}')

print(f'Min. Freq. Domain {freq_domain[0]*1e-9} GHz')
print(f'Max. Freq. Domain {freq_domain[-1]*1e-9} GHz')

print(f'Min. Pulse Domain {pulse_domain[0]*1e6} us')
print(f'Max. Pulse Domain {pulse_domain[-1]*1e6} us')
cpi_length = pulse_domain[-1] - pulse_domain[0]

print(f'Min. Time Domain {time_domain[0]} s')
print(f'Max. Time Domain {time_domain[-1]} s')

results_db= 20*np.log10(np.abs(all_results))
print(f'Results min: {results_db.min()}dB max: {results_db.max()}dB')

# plot all the results for a single simulation at time step N
time_step_idx = 3
current_time = time_domain[time_step_idx]
print(f'Plotting results for time {time_domain[time_step_idx]} s to {time_domain[time_step_idx]+cpi_length} s ')


# create a 2d plot of all the results.
fig, ax = plt.subplots()

# rotate the results so that the plot is oriented correctly when using imshow. imshow plots the matrix with the origin at the top left
# and the y axis increasing downwards. We want the origin at the bottom left and the y axis increasing upwards. our
# results are in the opposite orientation so we need to rotate it 90 degrees.
results_db_for_plot = np.rot90(results_db[time_step_idx,0,0])
im = ax.imshow(results_db_for_plot,
                extent=[pulse_domain[0], pulse_domain[-1], freq_domain[0], freq_domain[-1]],
                vmin=results_db_for_plot.min(),
                vmax=results_db_for_plot.max(),
                origin='upper',
                cmap='jet',
                aspect = 'auto')
ax.set_title(f'CFR: One Simulation at {current_time} s to {time_domain[time_step_idx]+cpi_length} s')
ax.set_xlabel('Channel Soundings (Pulse Domain)')
ax.set_ylabel('Freq. Domain (Hz)')
fig.colorbar(im,ax=ax)
plt.show()





# Plot the CFR for a single channel sounding (using pulse index = 0)

results_db_for_plot = results_db[time_step_idx,0,0,0]
fig, ax = plt.subplots()
ax.plot(freq_domain, results_db_for_plot)
ax.set(xlabel='Freq (Hz)', ylabel='Results (dB20)',
       title='CFR for a single channel sounding')
ax.grid()
plt.show()




# create an animated plot of the CFR for all channel soundings
# flatten all the data so that every channel sounding and every simulation time step is stacked
results_db_flat = results_db.reshape(-1,results_db.shape[-1])
fig, ax = plt.subplots()
results_db_for_plot = results_db[0,0,0,0] # start with the first time step, index and pulse
line, = ax.plot(freq_domain, results_db_for_plot)
ax.set(xlabel='Freq (Hz)', ylabel='Results (dB20)',
       title='CFR for a single channel sounding',
       xlim=[freq_domain[0], freq_domain[-1]], ylim=[-100,-40])


def update(frame):
    line.set_ydata(results_db_flat[frame])
    # ax.set_title(f'CFR for a single channel sounding at time {time_domain[frame]} s')
    return line

ani = animation.FuncAnimation(fig=fig, func=update, frames=results_db_flat.shape[0], interval=30)
plt.show()