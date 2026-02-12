import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
sys.path.append("..")
from pem_utilities.post_processing import range_profile

# using the rx and tx positions from the AODT interface, defined as:
# Friss (dB) -80.85847751666549 LoS(dB) -80.85556885465671
# Tx: [-432.09566406    2.1259462   151.02806641]
# Rx: [-60.24069336 177.63636719   6.10316833]
# Distance: 435.9853818577855
# UE tilt: 0.0

# CFR results exported from AODT, using the AODT solver
from_aodt_solver = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\cfrs_new.txt'
# read this file line by line
with open(from_aodt_solver, 'r') as f:
    lines = f.readlines()
    all_results_aodt = []
    for line in lines:
        asdf = line.replace('(',"").replace(')',"").replace("\n","")

        all_results_aodt.append(complex(asdf))
all_results_aodt = np.array(all_results_aodt,dtype=complex)
# read this file to numpy array
# with open(from_aodt_solver, 'r') as f:
#        data = np.loadtxt(f,delimiter=',')
# all_results_aodt= np.vectorize(complex)(data[:,0], data[:,1])

output_path = '../output/'

# results created from using the proto file (same exact geometry) as what is used in the AODT interface
all_results_pem = np.load(os.path.join(output_path,'all_results_from_proto.npy'))
range_profiles_pem = np.load(os.path.join(output_path,'range_profiles_from_proto.npy'))
freq_domain = np.load(os.path.join(output_path,'freq_domain_from_proto.npy'))
rng_domain = np.load(os.path.join(output_path,'range_domain_from_proto.npy'))
time_domain = np.load(os.path.join(output_path,'simulation_timestamps_from_proto.npy'))
input_power_dbm = np.load(os.path.join(output_path,'incident_power_dbm_from_proto.npy'))

# results created from using the proto file (same exact geometry) as what is used in the AODT interface
all_results_pem_usd = np.load(os.path.join(output_path,'all_results_usd.npy'))
range_profiles_pem_usd = np.load(os.path.join(output_path,'range_profiles_usd.npy'))



# CFR results exported from AODT, but using the PEM solver
bin_file = r'C:\Users\asligar\Downloads\output (1).bin'
#read the binary file
with open(bin_file, 'rb') as f:
       # read the header
       # header = f.read(4)
       # read the data
       all_results_aodt_pem = np.fromfile(f, dtype=np.float32)
# convert the data to complex
all_results_aodt_pem = np.vectorize(complex)(all_results_aodt_pem[::2], all_results_aodt_pem[1::2])

# add input power to the results if we want to use this
input_power_watts = 10 ** ((input_power_dbm - 30) / 10)
input_power_watts_db = 10*np.log10(input_power_watts)


# plot CFR, real component
fig, ax = plt.subplots()
ax.plot(freq_domain, np.real(all_results_aodt_pem),label='AODT - PEM')
ax.plot(freq_domain, np.real(all_results_pem_usd[0,0,0,0]),label='PEM - USD')
ax.plot(freq_domain, np.real(all_results_aodt),label='AODT - Direct')
ax.plot(freq_domain, np.real(all_results_pem[0,0,0,0]),label='PEM')
ax.set(xlabel='Freq (Hz)', ylabel='Results (real)',title='CFR for a single channel sounding')
plt.legend()
ax.grid()
plt.show()

results_db_pem = 20*np.log10(np.abs(all_results_pem[0,0,0,0]))#- input_power_watts_db
results_db_aodt = 20*np.log10(np.abs(all_results_aodt))#- input_power_watts_db
results_db_aodt_pem= 20*np.log10(np.abs(all_results_aodt_pem))#- input_power_watts_db

print(f'Results min: {results_db_pem.min()}dB max: {results_db_pem.max()}dB')

# plot all the results for a single simulation at time step N
time_step_idx = 0
current_time = time_domain[time_step_idx]
print(f'Plotting results for time {time_domain[time_step_idx]} s to {time_domain[time_step_idx]} s ')

# plot CFR, dB
fig, ax = plt.subplots()
ax.plot(freq_domain, results_db_pem,label='Perceive EM Only')
ax.plot(freq_domain, results_db_aodt_pem,label='AODT - PEM')
ax.plot(freq_domain, results_db_aodt,label='AODT - Direct')
ax.set(xlabel='Freq (Hz)', ylabel='Results (dB20)',title='CFR for a single channel sounding')
plt.legend()
ax.grid()
plt.show()


# Plot the CIR for a single channel sounding (using pulse index = 0)
len_fft = 4096
results_td_db_for_plot_pem = 20*np.log10(np.abs(range_profile(all_results_pem[0,0,0,0], window= False, size=len_fft)))#- input_power_watts_db
results_td_db_for_plot_aodt_pem= 20*np.log10(np.abs(range_profile(all_results_aodt_pem, window= False, size=len_fft)))#- input_power_watts_db
results_td_db_for_plot_pem_usd= 20*np.log10(np.abs(range_profile(all_results_pem_usd[0,0,0,0], window= False, size=len_fft)))#- input_power_watts_db
results_td_db_for_plot_aodt = 20*np.log10(np.abs(np.fft.ifft(all_results_aodt,norm = 'ortho')))

fig, ax = plt.subplots()
new_rng_domain = np.linspace(rng_domain[0], rng_domain[-1], results_td_db_for_plot_pem.shape[0])
ax.plot(new_rng_domain, results_td_db_for_plot_pem,label='Perceive EM - only')
ax.plot(new_rng_domain, results_td_db_for_plot_aodt_pem,label='AODT - PEM')
ax.plot(new_rng_domain, results_td_db_for_plot_aodt,label='AODT - direct')
ax.set(xlabel='Range (m)', ylabel='Results (dB20)',title='CIR for a single channel sounding')
ax.grid()
plt.legend()
plt.show()

fig, ax = plt.subplots()
new_time_domain = new_rng_domain/3e8
ax.plot(new_time_domain*1e9, results_td_db_for_plot_pem,label='Perceive EM - Proto')
ax.plot(new_time_domain*1e9, results_td_db_for_plot_pem_usd,label='Perceive EM - USD')
ax.set(xlabel='Range (ns)', ylabel='Results (dB20)',title='CIR for a PEM from USD and from Proto')
ax.set_xlim([1e9*new_time_domain[128], 1e9*new_time_domain[1024]])
plt.legend()
ax.grid()
plt.show()



fig, ax = plt.subplots()
new_time_domain = new_rng_domain/3e8
ax.plot(new_time_domain*1e9, results_td_db_for_plot_pem,label='Perceive EM - only')
ax.plot(new_time_domain*1e9, results_td_db_for_plot_aodt_pem,label='AODT - PEM')
ax.plot(new_time_domain*1e9, results_td_db_for_plot_aodt,label='AODT - direct')
ax.set(xlabel='Range (ns)', ylabel='Results (dB20)',title='CIR for a single channel sounding')
ax.set_xlim([1e9*new_time_domain[128], 1e9*new_time_domain[1024]])
ax.grid()
plt.legend()
plt.show()

range_value = new_rng_domain[np.argmax(results_td_db_for_plot_pem)]
print(f'Range value at peak of range profile: {range_value}m')
print(f'Perceive EM (peak from range profile, dB): {np.abs(np.max(results_td_db_for_plot_aodt_pem))}')


fig, ax = plt.subplots()
new_time_domain = new_rng_domain/3e8
ax.plot(new_time_domain*1e9, results_td_db_for_plot_pem_usd,label='Perceive EM USD')
ax.plot(new_time_domain*1e9, results_td_db_for_plot_aodt,label='AODT - direct')
ax.set(xlabel='Range (ns)', ylabel='Results (dB20)',title='CIR for a single channel sounding')
ax.set_xlim([1e9*new_time_domain[128], 1e9*new_time_domain[1024]])
ax.grid()
plt.legend()
plt.show()

range_value = new_rng_domain[np.argmax(results_td_db_for_plot_pem)]
print(f'Range value at peak of range profile: {range_value}m')
print(f'Perceive EM (peak from range profile, dB): {np.abs(np.max(results_td_db_for_plot_aodt_pem))}')




#
# # sort range_profile from largest to smallest, keep on the top N points. Use the same index values to create a new array of time_domain with the corroponding values
# N=16
# idxs = np.argsort(results_td_db_for_plot, axis=0)[::-1]  # high to low
# results_td_db_for_plot = results_td_db_for_plot[idxs]
# results_td_db_for_plot = results_td_db_for_plot[:N] # keep only the top N points
# new_rng_domain = new_rng_domain[idxs] #ns
# new_rng_domain = new_rng_domain[:N]
# # results_td_db_for_plot = results_td_db_for_plot/np.max(results_td_db_for_plot) # normalize to 1
#
#
# plt.figure()
# plt.title("Channel impulse response realization")
# ax = plt.stem(new_rng_domain, results_td_db_for_plot,'g',markerfmt='gD',label='Perceive EM',bottom=np.min(results_td_db_for_plot))
#
#
# # plt.xlim([0, np.max(t)])
# # plt.ylim([-2e-6, a_max*1.1])
# # plt.xlabel(r"$\tau$ [ns]")
# # plt.ylabel(r"$|a|$")
# plt.legend()
# plt.show()



