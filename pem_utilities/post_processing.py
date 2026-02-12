import numpy as np
import time as walltime
import matplotlib.pyplot as plt
import scipy.signal.windows as windows

from pyargus import directionEstimation as de
from matplotlib.widgets import Slider
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max
from copy import deepcopy
from scipy.signal import savgol_filter

from pem_utilities.utils import apply_math_function

class DynamicImageViewer:
    """
    # ============================================================================
    # Interactive SAR Image Viewer
    # ============================================================================
    # Display the final SAR image with interactive dynamic range adjustment

    # Convert final image to dB scale
    """
    def __init__(self, image,math_function='db',cmap='Greys_r'):
        self.image = apply_math_function(image, math_function)


        # Create figure with space for control sliders
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(left=0.1, bottom=0.25)

        # Calculate data range from finite values only
        sel_flat = self.image.flatten()
        sel_finite = sel_flat[np.isfinite(sel_flat)]
        if sel_finite.size > 0:
            data_min = np.min(sel_finite)
            data_max = np.max(sel_finite)
        else:
            data_min = 0.0
            data_max = 1.0

        if math_function.lower() == 'db':
            # Set initial display range (100 dB dynamic range)
            initial_vmin = data_max - 100
            initial_vmax = data_max
            valstep=0.5
        elif math_function.lower() == 'abs':
            initial_vmin = data_max - 1e-5
            initial_vmax = data_max
            valstep=0.01
        else:
            initial_vmin = data_min
            initial_vmax = data_max
            valstep=0.01
        # Display SAR image
        im = self.ax.imshow(
            self.image,
            cmap=cmap,
            vmin=initial_vmin,
            vmax=initial_vmax,
            extent=(0, self.image.shape[1], 0, self.image.shape[0]),
            aspect='auto'
        )

        # Add labels and colorbar
        self.ax.set_title('Interactive SAR Image\nAdjust sliders to change dynamic range')
        self.ax.set_ylabel('Cross-Range (m)')
        self.ax.set_xlabel('Down-Range (m)')
        cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, label=f'Intensity {math_function}')

        # Create slider controls
        ax_vmin = plt.axes([0.1, 0.12, 0.8, 0.03])
        ax_vmax = plt.axes([0.1, 0.07, 0.8, 0.03])

        slider_vmin = Slider(
            ax_vmin, f'Min  {math_function}', 
            data_min, data_max, 
            valinit=initial_vmin, 
            valstep=valstep
        )
        slider_vmax = Slider(
            ax_vmax, f'Max  {math_function}', 
            data_min, data_max, 
            valinit=initial_vmax, 
            valstep=valstep
        )

        # Add dynamic range indicator
        text_box = self.ax.text(
            0.02, 0.98,
            f'Dynamic Range: {initial_vmax - initial_vmin:.1f} {math_function}',
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )


        def update_display(val):
            """
            Update image display when sliders are adjusted.
            Ensures vmin < vmax constraint is maintained.
            """
            vmin = slider_vmin.val
            vmax = slider_vmax.val
            
            # Enforce vmin < vmax
            if vmin >= vmax:
                vmin = vmax - 0.5
                slider_vmin.set_val(vmin)
            
            # Update image and dynamic range display
            im.set_clim(vmin, vmax)
            text_box.set_text(f'Dynamic Range: {vmax - vmin:.1f} dB')
            self.fig.canvas.draw_idle()


        # Connect sliders to update function
        slider_vmin.on_changed(update_display)
        slider_vmax.on_changed(update_display)

        # Add data range information
        instruction_text = f'Data Range: {data_min:.1f} to {data_max:.1f} {math_function}'
        plt.figtext(
            0.5, 0.02,
            instruction_text,
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )

        print('\nDisplaying interactive SAR image viewer...')
        plt.show()

    def save(self, filename):
        """
        Save the current figure to a file.
        """
        self.fig.savefig(filename, dpi=300,bbox_inches='tight')
        print(f'Figure saved to {filename}')



def crop_around_center(data, output_size=(None,None)):

    if output_size is None or output_size == (None, None):
        raise ValueError("Output size must be specified as a tuple (height, width)")
    if not isinstance(output_size, tuple) or len(output_size) != 2:
        raise ValueError("Output size must be a tuple of the form (height, width)")


    # Get the current image dimensions (should be square for radar imaging)
    current_height, current_width = data.shape
    
    # Calculate the target crop dimensions
    target_height = output_size[0]
    target_width = output_size[1]
    
    # Ensure we don't try to crop to a size larger than the current image
    if target_height > current_height or target_width > current_width:
        raise ValueError(f"Target crop size ({target_height}x{target_width}) is larger than current image size ({current_height}x{current_width})")
    
    # Calculate center indices for both dimensions
    # For even-sized arrays: center is at n//2 (e.g., for size 8, center at index 4)
    # For odd-sized arrays: center is at n//2 (e.g., for size 9, center at index 4)
    center_row = current_height // 2
    center_col = current_width // 2
    
    # Calculate how many pixels to take from each side of center
    # For even target size: take equal amounts from each side, but this creates slight asymmetry
    # For odd target size: center pixel is preserved exactly
    half_target_height = target_height // 2
    half_target_width = target_width // 2
    
    # Calculate crop boundaries
    # For odd target sizes: we take half_target pixels on each side of center
    # For even target sizes: we take half_target-1 pixels before center, half_target pixels after
    if target_height % 2 == 1:
        # Odd target height: symmetric cropping around center pixel
        row_start = center_row - half_target_height
        row_end = center_row + half_target_height + 1  # +1 because slice end is exclusive
    else:
        # Even target height: slightly asymmetric (one more pixel after center)
        row_start = center_row - half_target_height
        row_end = center_row + half_target_height
        
    if target_width % 2 == 1:
        # Odd target width: symmetric cropping around center pixel
        col_start = center_col - half_target_width
        col_end = center_col + half_target_width + 1  # +1 because slice end is exclusive
    else:
        # Even target width: slightly asymmetric (one more pixel after center)
        col_start = center_col - half_target_width
        col_end = center_col + half_target_width
    
    # Perform the crop
    data_cropped = data[row_start:row_end, col_start:col_end]
    
    # Verify the resulting dimensions match our target
    assert data_cropped.shape == (target_height, target_width), \
        f"Cropped image shape {data_cropped.shape} doesn't match target {(target_height, target_width)}"
 
    

    return data_cropped

def window_function(function='Flat', size=512, dbDown=30):
    if function.lower() == 'hann':
        win = np.hanning(size)
    elif function.lower() == 'hamming':
        win = np.hamming(size)
    else:
        win = np.ones(size)
    win_sum = np.sum(win)
    win *= size/win_sum
    return win, win_sum
def waterfall(data, chirpInterval, output_size=256, windowing=True):
    '''


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    chirpInterval : TYPE
        DESCRIPTION.
    output_size : TYPE, optional
        DESCRIPTION. The default is 256.
    windowing : TYPE, optional
        DESCRIPTION. The default is True.
    data_domain : TYPE, optional
        DESCRIPTION. The default is 'time' can be 'time' or 'freq'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    data = np.squeeze(data)
    num_frames = data.shape[0]
    num_chirps = data.shape[1]
    num_freq = data.shape[2]
    total_chrips = num_chirps * num_frames
    chirp_duration = chirpInterval

    # need to correct this in original simulation, since it included teh final time that is implied
    time_stamps_sim = np.linspace(0, chirp_duration * total_chrips, num=total_chrips)
    # I just happened to do everything in an order different from how RTR outputs data, so this is just rearrange
    data_chirp_freq = np.reshape(data, (total_chrips, num_freq))
    data_chirp_range = np.squeeze(
        convert_pulsefreq_to_pulserange(data_chirp_freq, output_size=output_size, windowing=windowing))
    return np.abs(data_chirp_range)


def spectrogram(data, time_stamps, velDomain, window_length=5, order=3):
    x, y = np.meshgrid(time_stamps, velDomain)

    all_results_dopp = np.max(np.abs(data), axis=2)

    spectrogram = all_results_dopp
    spectrogram = savgol_filter(spectrogram, window_length, order, mode='nearest', axis=0)
    return spectrogram


def range_profile(data, window: bool = False, size: int = 1024):
    """
    range profile calculation

    input: 1D array [freq_samples]

    returns: 1D array in original_lenth*upsample

    """
    data = np.squeeze(data)  # in case 1 x N is passed

    if data.ndim == 2:
        num_channels = data.shape[0]
        nfreq = int(data.shape[1])
    elif data.ndim == 1:
        num_channels = 1
        nfreq = int(len(data))
    else:
        print("ERROR: Input dimension must be at 1D or 2D")
        return np.zeros((num_channels, size), dtype='complex')

    out = np.zeros((num_channels, size), dtype='complex')


    # for ch in range(num_channels):
        # scale factors used for windowing function
    if window:
        win_range, _ = window_function(function='hamming', size=nfreq)
        pulse_f = data*win_range  # apply windowing
    else:
        pulse_f = data

    sf_upsample = size / nfreq
    # should probably upsample to the closest power of 2 for faster processing, but not going to for now
    if pulse_f.ndim == 1:
        out = np.fft.fftshift(sf_upsample*np.fft.ifft(pulse_f, n=size,axis=0),axes=0)
        out = sf_upsample * np.fft.ifft(pulse_f, n=size, axis=0)
    else:
        out = np.fft.fftshift(sf_upsample * np.fft.ifft(pulse_f, n=size, axis=1), axes=0)
        out = sf_upsample * np.fft.ifft(pulse_f, n=size, axis=1)
    return out


def convert_pulsefreq_to_pulserange(data, output_size=None, windowing=True):
    '''
    input: 3D array [channel][chirp][freq_samples], or 2D array [chirp][freq_samples] size is desired output in (ndoppler,nrange)
            output_size is up/down samping in range dimensions
            pulse=None, this is the pulse to use, if set to none it will extract from center pulse
    returns: 3D/2D array in [channel][chirp][range] or [chirp][range]
    '''

    if isinstance(data, list):
        data = np.array(data)

    if data.ndim == 3:
        num_channels = data.shape[0]
        num_chirps = data.shape[1]
        num_freq = data.shape[2]
    elif data.ndim == 2:
        num_channels = 1
        num_chirps = data.shape[0]
        num_freq = data.shape[1]
        data = np.moveaxis(np.atleast_3d(data), -1, 0)  # add channel as first dimension
    else:
        print("ERROR: Input dimension must be at least 2D")

    if output_size is not None:
        rPixels = output_size
    else:
        rPixels = num_freq

    # window
    converted_data = np.zeros(data.shape, dtype='complex')
    if windowing:
        for ch in range(num_channels):
            h_rng = np.hanning(num_freq)
            sf_rng = len(h_rng) / np.sum(h_rng)
            sf_upsample_rng = rPixels / num_freq
            h_rng = h_rng * sf_rng
            h_rng_2 = np.tile(h_rng, (num_chirps, 1))
            # apply windowing
            # ch_freq_win =sf_upsample_rng* np.multiply(ch_freq,h_rng_2)

            converted_data[ch] = sf_upsample_rng * np.multiply(data[ch], h_rng)
    else:
        converted_data = data

    # take fft
    out = np.fft.ifft(converted_data, n=rPixels)  # [ch][chirp][freq]fft across dop dimenions

    return np.fliplr(out)




def pulse_freq_to_doppler_range(pulse_freq_matrix, window_function = windows.hann, sidelobe_level=200, output_size_doppler=256, output_size_range=256):
    """

    # I have a few variations of this, but I think this is the most recent version of this function (08.01.25)
    # and should be the one tha tis used 

    Converts a 2D matrix from pulse vs. frequency to Doppler vs. range.

    This function applies Hann windowing, performs a 2D FFT to convert
    to Doppler-range, upsamples the data, and scales the result to
    compensate for windowing and FFT effects.

    Args:
        pulse_freq_matrix (np.ndarray): A complex-valued 2D NumPy array
                                        where axis 0 is the pulse dimension
                                        and axis 1 is the frequency dimension.
        upsample_factor_doppler (int): The factor by which to upsample the Doppler axis.
        upsample_factor_range (int): The factor by which to upsample the range axis.

    Returns:
        np.ndarray: A complex-valued 2D NumPy array representing the
                    Doppler vs. range map.
    """


    # Get the dimensions of the input matrix
    num_pulses, num_freq_bins = pulse_freq_matrix.shape

    # 1. Create and apply a 2D window
    # Create 1D windows for both dimensions
    
    if window_function is None: # no windowing
        windowed_data = pulse_freq_matrix
    else:
        freq_window = window_function(num_freq_bins)
        pulse_window = window_function(num_pulses)
        if window_function.__name__ == 'taylor':
            freq_window = window_function(num_freq_bins, nbar = 5, sll=sidelobe_level, sym=True)
            pulse_window = window_function(num_pulses, nbar = 5, sll=sidelobe_level, sym=True)

        # Create the 2D window by taking the outer product of the 1D windows
        window_2d = np.outer(pulse_window, freq_window)
        # Apply the window to the input data element-wise
        windowed_data = pulse_freq_matrix * window_2d

    # 2. Define upsampled FFT sizes
    n_fft_doppler = output_size_doppler
    n_fft_range = output_size_range

    # 3. Perform Range-FFT (Inverse FFT along the frequency axis)
    # The IFFT converts the frequency-domain data to the time domain, which corresponds to range.
    # Upsampling is done by specifying a larger number of FFT points (n_fft_range).
    range_processed_data = np.fft.ifft(windowed_data, n=n_fft_range, axis=1)

    # 4. Perform Doppler-FFT (FFT along the pulse axis)
    # The FFT along the slow-time (pulse) axis extracts Doppler frequency information.
    # Upsampling is done by specifying a larger number of FFT points (n_fft_doppler).
    doppler_range_matrix = np.fft.fft(range_processed_data, n=n_fft_doppler, axis=0)

    # 5. Center the zero-frequency/Doppler components
    # This shifts the DC component from the corner to the center of the matrix.
    doppler_range_matrix = np.fft.fftshift(doppler_range_matrix, axes=(0, 1))

    # 6. Scale the results
    # Compensate for the coherent gain of the window to preserve amplitude.
    # The numpy.fft.ifft function already scales the output by 1/N. We need to account
    # for the scaling of the window itself. The total scaling factor preserves the
    # amplitude of a signal component.
    window_sum = np.sum(window_2d)
    
    # The IFFT on the range axis scales by 1/n_fft_range.
    # The coherent gain of the window is its sum.
    # To restore the original amplitude, we must multiply by n_fft_range and divide by the window sum.
    # Since the 2D FFT is separable, we consider the scaling for each dimension.
    # Range scaling (due to window and IFFT): (1/sum(freq_window)) * n_fft_range / n_fft_range = 1/sum(freq_window)
    # Doppler scaling (due to window): 1/sum(pulse_window)
    # The `ifft` already applied the `1/n_fft_range` scaling.
    # The combined scaling factor is n_fft_range / window_sum.
    scaling_factor = n_fft_range / window_sum
    
    scaled_doppler_range_matrix = doppler_range_matrix * scaling_factor

    return scaled_doppler_range_matrix

def range_crossrange(data, window=False, size=(256, 256), data_order='freq_pulse'):
    """
    range doppler calculation

    input: 2D array [freq_samples][pulses], size is desired output in (ndoppler,nrange)

    data_order='freq_pulse' or 'freq_pulse' or 'pulse_freq'

    returns: 2D array in [range][doppler]

    """
    if data_order == 'freq_pulse':
        data = data.T
    time_before = walltime.time()
    # I think something is wrong with data being returned as opposte, freq and pulse are swaped
    nfreq = int(np.shape(data)[0])
    ntime = int(np.shape(data)[1])

    rPixels = size[0]
    dPixels = size[1]

    h_dop = np.hanning(ntime)
    sf_dop = len(h_dop) / np.sum(h_dop)
    sf_upsample_dop = dPixels / ntime

    h_rng = np.hanning(nfreq)
    sf_rng = len(h_rng) / np.sum(h_rng)
    sf_upsample_rng = rPixels / nfreq

    h_dop = h_dop * sf_rng
    h_rng = h_rng * sf_dop

    fp_win = sf_upsample_dop * np.multiply(data, h_dop)
    s1 = np.fft.ifft(fp_win, n=dPixels)
    s1 = np.rot90(s1)

    s1_win = sf_upsample_rng * np.multiply(h_rng, s1)
    s2 = np.fft.ifft(s1_win, n=rPixels)
    s2 = np.rot90(s2)
    s2_shift = np.fft.fftshift(s2, axes=1)
    # range_doppler = np.flipud(s2_shift)
    range_doppler = np.flipud(s2_shift)
    # range_doppler=s2_shift
    time_after = walltime.time()
    duration_time = time_after - time_before
    if duration_time == 0:
        duration_time = 1
    duration_fps = 1 / duration_time

    rp = 0
    return range_doppler


def range_crossrange2(data, window=False, size=(256, 256), data_order='freq_pulse'):
    """
    range doppler calculation

    input: 2D array [freq_samples][pulses], size is desired output in (ndoppler,nrange)

    data_order='freq_pulse' or 'freq_pulse' or 'pulse_freq'

    returns: 2D array in [range][doppler]

    """
    if data_order == 'freq_pulse':
        data = data.T
    time_before = walltime.time()
    # I think something is wrong with data being returned as opposte, freq and pulse are swaped
    nfreq = int(np.shape(data)[0])
    ntime = int(np.shape(data)[1])

    rPixels = size[0]
    dPixels = size[1]

    h_dop = np.hanning(ntime)
    sf_dop = len(h_dop) / np.sum(h_dop)
    sf_upsample_dop = dPixels / ntime

    h_rng = np.hanning(nfreq)
    sf_rng = len(h_rng) / np.sum(h_rng)
    sf_upsample_rng = rPixels / nfreq

    h_dop = h_dop * sf_rng
    h_rng = h_rng * sf_dop

    fp_win = sf_upsample_dop * np.multiply(data, h_dop)
    s1 = np.fft.ifft(fp_win, n=dPixels)
    s1 = np.rot90(s1)

    s1_win = sf_upsample_rng * np.multiply(h_rng, s1)
    s2 = np.fft.ifft(s1_win, n=rPixels)
    # s2 = np.rot90(s2)
    s2_shift = np.fft.fftshift(s2, axes=0)
    # range_doppler = np.flipud(s2_shift)
    range_doppler = np.flipud(s2_shift)
    # range_doppler=s2_shift
    time_after = walltime.time()
    duration_time = time_after - time_before
    if duration_time == 0:
        duration_time = 1
    duration_fps = 1 / duration_time

    rp = 0
    return range_doppler

def range_doppler_map(data, window=False, size=(256, 256)):
    """
    range doppler calculation

    input: 2D array [freq_samples][pulses], size is desired output in (ndoppler,nrange)

    returns: 2D array in [range][doppler]

    """

    time_before = walltime.time()
    # I think something is wrong with data being returned as opposte, freq and pulse are swaped
    nfreq = int(np.shape(data)[0])
    ntime = int(np.shape(data)[1])

    rPixels = size[0]
    dPixels = size[1]

    h_dop = np.hanning(ntime)
    sf_dop = len(h_dop) / np.sum(h_dop)
    sf_upsample_dop = dPixels / ntime

    h_rng = np.hanning(nfreq)
    sf_rng = len(h_rng) / np.sum(h_rng)
    sf_upsample_rng = rPixels / nfreq

    h_dop = h_dop * sf_rng
    h_rng = h_rng * sf_dop

    fp_win = sf_upsample_dop * np.multiply(data, h_dop)
    s1 = np.fft.ifft(fp_win, n=dPixels)
    s1 = np.rot90(s1)

    s1_win = sf_upsample_rng * np.multiply(h_rng, s1)
    s2 = np.fft.ifft(s1_win, n=rPixels)
    s2 = np.rot90(s2)
    s2_shift = np.fft.fftshift(s2, axes=0)
    # range_doppler = np.flipud(s2_shift)
    range_doppler = np.flipud(s2_shift)
    # range_doppler=s2_shift
    time_after = walltime.time()
    duration_time = time_after - time_before
    if duration_time == 0:
        duration_time = 1
    duration_fps = 1 / duration_time

    rp = 0
    return range_doppler, rp, duration_fps


def range_angle_map(data, antenna_spacing_wl=0.5, source_data='RangeDoppler', DoA_method='fft', fov=[-90, 90],
                    out_size=(256, 256), range_bin_idx=None, chirp_index=None,window=False):
    """
    range calculationg calculation

    input: 3D array [channel][pulses][freq], in case of FreqPulse mode
                or
           3D array [channel][doppler][range], in case of RangeDoppler mode
           source_data = 'RangeDoppler' or 'FreqPulse'

           DoA_method, 'fft', Bartlett, Capon, MEM, MUSIC

           out_size, output size in [range][xrange]
           range_bin=-1 do all range bins, or if specified do only specific range bin index

    returns: 2D array of size [range][xrange]

    """

    rPixels = out_size[0]
    xrPixels = out_size[1]

    xrng_dims = np.shape(data)[0]
    nchannel = xrng_dims

    DoA_method = DoA_method.lower()

    if source_data.lower() == 'freqpulse' or source_data == 'pulsefreq':
        if chirp_index is not None:
            data = data[:, chirp_index, :]
            ch_range = range_profile(data,size=rPixels,window=window)
        else:  # use the first chirp
            data = data[:, 0, :]
            ch_range = range_profile(data,size=rPixels,window=window)
        # import matplotlib.pyplot as plt
        # plt.plot(np.unwrap(np.angle(ch_range[:,44]),period=np.pi))
        # plt.show()
        if DoA_method == "fft":
            if window:
                h_xrng = np.hanning(xrng_dims)
                sf_xrng = len(h_xrng) / np.sum(h_xrng)
                sf_upsample_xrng = xrPixels / xrng_dims
                h_xrng = np.atleast_2d(h_xrng * sf_xrng)
                rng_ch_win = sf_upsample_xrng * np.multiply(ch_range, h_xrng.T)
            else:
                rng_ch_win = ch_range
            # rng_ch_win = rng_ch_win.T  # correct order after multiplication (same as swapaxes)
            rng_xrng = np.fft.fft(rng_ch_win, n=xrPixels,axis=0).T
            rng_xrng = np.fft.fftshift(rng_xrng, axes=1)
            rng_xrng = np.fliplr(rng_xrng)
        else:  # for DoA_method = bartlett, capon mem and music
            ang_stop = fov[1] + 90  # offset fov because beam serach is from 0 to 180
            ang_start = fov[0] + 90
            range_ch = ch_range.T
            array_alignment = np.arange(0, nchannel, 1) * antenna_spacing_wl
            incident_angles = np.linspace(ang_start, ang_stop, num=xrPixels)
            ula_scanning_vectors = de.gen_ula_scanning_vectors(array_alignment, incident_angles)
            sf = len(incident_angles) / xrng_dims
            if range_bin_idx is not None:  # do only specific range bin
                rPixels = 1
                range_ch = np.atleast_2d(range_ch[range_bin_idx])
            rng_xrng = np.zeros((rPixels, xrPixels), dtype=complex)  # (pulse,range)

            for n, rb in enumerate(range_ch):  # if range bin is speficied it will only go once
                ## R matrix calculation
                rb = np.reshape(rb, (1, nchannel))
                # R = de.corr_matrix_estimate(rb, imp="fast")
                R = np.outer(rb, rb.conj())
                # R = de.forward_backward_avg(R)
                if DoA_method == "bartlett":
                    range_bin = de.DOA_Bartlett(R, ula_scanning_vectors)
                elif DoA_method == "capon":
                    range_bin = de.DOA_Capon(R, ula_scanning_vectors)
                elif DoA_method == "mem":
                    range_bin = de.DOA_MEM(R, ula_scanning_vectors, column_select=0)
                elif DoA_method == "music":
                    range_bin = de.DOA_MUSIC(R, ula_scanning_vectors, signal_dimension=1)
                rng_xrng[n] = range_bin * sf


    elif source_data.lower() == 'rangedoppler' or source_data.lower() == 'dopplerrange':
        if DoA_method == 'fft':
            if out_size[1] != np.shape(data)[0]:
                print("WARNING: Range Angle Map Output size must be equal to the number of range cells")
                print(f"WARNING: Range Angle Map Output size is set to the number of range cells {np.shape(data)[2]}")

            # fft to get to range vs pulse
            # ch_rng_pulse = np.fft.fft(data, n=xrPixels,axis=1)
            # ch_rng_pulse = np.fft.fftshift(ch_rng_pulse, axes=-1)
            ch_rng_pulse = np.swapaxes(data, 1, 2)

            dop_dims = np.shape(ch_rng_pulse)[2]
            ch_range = ch_rng_pulse[:,:,dop_dims // 2]

            if window:
                h_xrng = np.hanning(xrng_dims)
                sf_xrng = len(h_xrng) / np.sum(h_xrng)
                sf_upsample_xrng = xrPixels / xrng_dims
                h_xrng = np.atleast_2d(h_xrng * sf_xrng)
                rng_ch_win = sf_upsample_xrng*np.multiply(ch_range, h_xrng.T)
                rng_ch_win = rng_ch_win.T  # correct order after multiplication (same as swapaxes)
            else:
                rng_ch_win = ch_range.T
            rng_xrng = np.fft.ifft(rng_ch_win, n=rPixels,axis=-1)
            rng_xrng = np.fft.fftshift(rng_xrng, axes=1)
        else:  # for DoA_method = bartlett, capon mem and music
            # just need to rearrange data, temporary
            ch_rng_pulse = np.swapaxes(data, 1, 2)
            rng_dims = np.shape(ch_rng_pulse)[1]
            dop_dims = np.shape(ch_rng_pulse)[2]
            xrng_dims = np.shape(ch_rng_pulse)[0]

            # ch_rng_pulse = np.fft.fft(data, n=xrPixels)
            # ch_rng_pulse = np.fft.fftshift(ch_rng_pulse, axes=2)
            # ch_rng_pulse = np.fliplr(ch_rng_pulse)

            range_ch = np.swapaxes(ch_rng_pulse, 2, 0)
            if chirp_index is None:
                range_ch = range_ch[int(dop_dims / 2)]
            else:
                range_ch = range_ch[chirp_index]

            ang_stop = fov[1] + 90  # offset fov because beam serach is from 0 to 180
            ang_start = fov[0] + 90
            array_alignment = np.arange(0, nchannel, 1) * antenna_spacing_wl
            incident_angles = np.linspace(ang_start, ang_stop, num=xrPixels)
            ula_scanning_vectors = de.gen_ula_scanning_vectors(array_alignment, incident_angles)

            sf = len(incident_angles) / xrng_dims
            if range_bin_idx is not None:  # do only specific range bin
                rng_dims = 1
                range_ch = np.atleast_2d(range_ch[range_bin_idx])
            rng_xrng = np.zeros((rng_dims, xrPixels), dtype=complex)  # (pulse,range)
            for n, rb in enumerate(range_ch):
                ## R matrix calculation
                rb = np.reshape(rb, (1, nchannel))
                # R = de.corr_matrix_estimate(rb, imp="fast")
                R = np.outer(rb, rb.conj())
                # R = de.forward_backward_avg(R)
                if DoA_method == "bartlett":
                    range_bin = de.DOA_Bartlett(R, ula_scanning_vectors)
                elif DoA_method == "capon":
                    range_bin = de.DOA_Capon(R, ula_scanning_vectors)
                elif DoA_method == "mem":
                    range_bin = de.DOA_MEM(R, ula_scanning_vectors, column_select=0)
                elif DoA_method == "music":
                    range_bin = de.DOA_MUSIC(R, ula_scanning_vectors, signal_dimension=1)
                rng_xrng[n] = range_bin * sf

    return rng_xrng


def peak_detector2(data, max_detections=20, threshold_rel=1e-2):
    '''
    passing data in as linear, but converting to dB seems to work
    '''
    time_before = walltime.time()
    size = np.shape(data)
    if len(size) > 2:
        data = data[0]

    data = np.abs(data)
    # data = 20*np.log10(np.abs(data))
    # threshold_rel*max_val of plot is the minimum threshold returned

    coordinates = peak_local_max(data, min_distance=5,
                                 threshold_rel=threshold_rel, num_peaks=max_detections,
                                 exclude_border=False)

    peak_mask = np.zeros_like(data, dtype=bool)
    peak_mask[tuple(coordinates.T)] = True

    time_after = walltime.time()
    duration_time = time_after - time_before
    if duration_time == 0:
        duration_time = 1
    duration_fps = 1 / duration_time

    # return as 1 or zero to be consistent with CFAR processing below
    return peak_mask.astype(int), duration_fps


def peak_detector(data, max_detections=20):
    '''
    passing data in as linear, but converting to dB seems to work
    '''
    time_before = walltime.time()
    size = np.shape(data)
    if len(size) > 2:
        data = data[0]
    data = np.abs(data)
    # data = 20*np.log10(np.abs(data))
    data[data > 1e-7] = 1
    data[data < 1e-7] = 0

    time_after = walltime.time()
    duration_time = time_after - time_before
    if duration_time == 0:
        duration_time = 1
    duration_fps = 1 / duration_time

    # return as 1 or zero to be consistent with CFAR processing below
    return data, duration_fps


######################################################################
"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                          Hit Processor Module


     Description:
     ------------
         Contains the implementation of the most common hit processing algorithms.

             - CA-CFAR processor: Implements an automatic detection with (Cell Averaging - Constant False Alarm Rate) detection.
             - Target DOA estimator: Estimates direction of arrival for the target reflection from the range-Doppler
                                     maps of the surveillance channels using phased array techniques.

     Notes:
     ------------

     Features:
     ------------

     Project: pyAPRIL
     Authors: Tamás Pető
     License: GNU General Public License v3 (GPLv3)

     Changelog :
         - Ver 1.0.0    : Initial version (2017 11 02)
         - Ver 1.0.1    : Faster CFAR implementation(2019 02 15)
         - Ver 1.1.0    : Target DOA estimation (2019 04 11)

 """


def CA_CFAR(rd_matrix, win_len=50, win_width=50, guard_len=10, guard_width=10, threshold=20):
    """
    Description:
    ------------
        Cell Averaging - Constant False Alarm Rate algorithm

        Performs an automatic detection on the input range-Doppler matrix with an adaptive thresholding.
        The threshold level is determined for each cell in the range-Doppler map with the estimation
        of the power level of its surrounding noise. The average power of the noise is estimated on a
        rectangular window, that is defined around the CUT (Cell Under Test). In order the mitigate the effect
        of the target reflection energy spreading some cells are left out from the calculation in the immediate
        vicinity of the CUT. These cells are the guard cells.
        The size of the estimation window and guard window can be set with the win_param parameter.

    Implementation notes:
    ---------------------

    Parameters:
    -----------

    :param rd_matrix: Range-Doppler map on which the automatic detection should be performed
    :param win_param: Parameters of the noise power estimation window
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param threshold: Threshold level above the estimated average noise power

    :type rd_matrix: R x D complex numpy array
    :type win_param: python list with 4 elements
    :type threshold: float

    Return values:
    --------------

    :return hit_matrix: Calculated hit matrix

    """

    time_before = walltime.time()

    norc = np.size(rd_matrix, 1)  # number of range cells
    noDc = np.size(rd_matrix, 0)  # number of Doppler cells
    hit_matrix = np.zeros((noDc, norc), dtype=float)

    # Convert range-Doppler map values to power
    rd_matrix = np.abs(rd_matrix) ** 2

    # Generate window mask
    rd_block = np.zeros((2 * win_width + 1, 2 * win_len + 1), dtype=float)
    mask = np.ones((2 * win_width + 1, 2 * win_len + 1))
    mask[win_width - guard_width:win_width + 1 + guard_width, win_len - guard_len:win_len + 1 + guard_len] = np.zeros(
        (guard_width * 2 + 1, guard_len * 2 + 1))

    cell_counter = np.sum(mask)

    # Convert threshold value
    threshold = 10 ** (threshold / 10)
    threshold /= cell_counter

    # -- Perform automatic detection --
    for j in np.arange(win_width, noDc - win_width, 1):  # Range loop
        for i in np.arange(win_len, norc - win_len, 1):  # Doppler loop
            rd_block = rd_matrix[j - win_width:j + win_width + 1, i - win_len:i + win_len + 1]
            rd_block = np.multiply(rd_block, mask)
            cell_SINR = rd_matrix[j, i] / np.sum(rd_block)  # esimtate CUT SINR

            # Hard decision
            if cell_SINR > threshold:
                hit_matrix[j, i] = 1
    time_after = walltime.time()
    duration_time = time_after - time_before
    duration_fps = 1 / duration_time
    return hit_matrix, duration_fps


def target_DOA_estimation(data, xrPixels, range_idx, doppler_idx, fov=[-90, 90], antenna_spacing_wl=0.5,
                          DOA_method="Bartlett"):
    """
        Performs DOA (Direction of Arrival) estimation for the given hits. To speed up the calculation for multiple
        hits this function requires the calculated range-Doppler maps from all the surveillance channels.

    Parameters:
    -----------
        :param: rd_maps: range-Doppler matrices from which the azimuth vector can be extracted
        :param: hit_list: Contains the delay and Doppler coordinates of the targets.
        :param: DOA_method: Name of the required algorithm to use for the estimation
        :param: array_alignment: One dimensional array, which describes the active antenna positions

        :type : rd_maps: complex valued numpy array with the size of  Μ x D x R , where R is equal to
                                the number of range cells, and D denotes the number of Doppler cells.
        :type: hit_list: Python list [[delay1, Doppler1],[delay2, Doppler2]...].
        :type: DOA_method: string
        :type: array_alignment: real valued numpy array with size of 1 x M, where M is the number of
                            surveillance antenna channels.

    Return values:
    --------------
        target_doa : Measured incident angles of the targets

    TODO: Extend with decorrelation support
    """

    size = np.shape(data)
    doa_list = []  # This list will contains the measured DOA values
    nchannel = int(size[0])

    ang_stop = fov[1] + 90  # offset fov because beam serach is from 0 to 180
    ang_start = fov[0] + 90

    array_alignment = np.arange(0, nchannel, 1) * antenna_spacing_wl

    incident_angles = np.linspace(ang_start, ang_stop, num=xrPixels)
    ula_scanning_vectors = de.gen_ula_scanning_vectors(array_alignment, incident_angles)
    DOA_method = DOA_method.lower()
    azimuth_vector = data[:, range_idx, doppler_idx]
    R = np.outer(azimuth_vector, azimuth_vector.conj())
    if DOA_method == "bartlett":
        doa_res = de.DOA_Bartlett(R, ula_scanning_vectors)
    elif DOA_method == "capon":
        doa_res = de.DOA_Capon(R, ula_scanning_vectors)
    elif DOA_method == "mem":
        doa_res = de.DOA_MEM(R, ula_scanning_vectors, column_select=0)
    elif DOA_method == "music":
        doa_res = de.DOA_MUSIC(R, ula_scanning_vectors, signal_dimension=1)

    doa_res_abs = np.abs(doa_res)
    max_location = np.argmax(doa_res_abs)
    # this is slowing down post processing and is not currently used
    # commenting out for now
    # max_value = np.max(doa_res_abs)
    # peaks_indices = find_peaks(doa_res_abs)
    # peaks_indices = peaks_indices[0]
    # peaks_values = doa_res_abs[peaks_indices]
    # #find_peaks does not indentify peaks and start or end of data set. I'll
    # #check if the max value is not in the peak dataset, if it isn't add it
    # if max_location not in peaks_indices:
    #     peaks_indices = np.append(peaks_indices,max_location)
    #     peaks_values = np.append(peaks_values,max_value)
    # peaks = list(zip(peaks_indices, peaks_values))
    # peaks = np.array(peaks)

    # threshold = 0.9 * max_value

    # filtered_peaks_indices = [int(index) for index, value in peaks if value > threshold]

    # minus 90 because orginal scan was 0 to 180,
    # coordinate sys for osi would mean these angles are reversed
    # assumes the Y axis is to the left if the vehicke is looking forward
    hit_doa = -1 * (incident_angles[max_location] - 90)
    # hit_doa_all = -1*(incident_angles[filtered_peaks_indices]-90)
    hit_doa_all = []
    return hit_doa, hit_doa_all


def create_target_list(rd_all_channels_az=None,
                       rd_all_channels_el=None,
                       rngDomain=None,
                       velDomain=None,
                       azPixels=256, elPixels=256,
                       antenna_spacing_wl=0.5,
                       radar_fov_az=[-90, 90],
                       radar_fov_el=[0, 90],
                       centerFreq=76.5e9,
                       rcs_min_detect=0,
                       min_detect_range=7.5,
                       rel_peak_threshold=1e-2,
                       max_detections=100,
                       exclude_velocities=None,
                       return_cfar=False,
                       incident_power=1):
    rd_all_channels_az = np.swapaxes(rd_all_channels_az, 1,
                                     2)  # i am re-using old code that expected format for [ch][range][doppler] but input form is actually [ch][doppler][range]
    if rd_all_channels_el is None:
        includes_elevation = False
    else:
        includes_elevation = True
        rd_all_channels_el = np.swapaxes(rd_all_channels_el, 1, 2)
    # set all rd maps to 0 where velocities are within the exclude range, this avoids adding these value to the peak detector
    if exclude_velocities is not None:
        v_min = exclude_velocities[0]
        v_max = exclude_velocities[1]
        # at what index is teh velocity closest to v_min in velDomain
        v_min_idx = np.argmin(np.abs(velDomain - v_min))
        v_max_idx = np.argmin(np.abs(velDomain - v_max))
        rd_all_channels_az[:, :, v_min_idx:v_max_idx] = 0
        if includes_elevation:
            rd_all_channels_el[:, :, v_min_idx:v_max_idx] = 0


    time_before_target_list = walltime.time()
    target_list = {}
    # this CA_CFAR is too slow, doing to just use local peak detection instead
    # rd_cfar, cfar_fps = pp.CA_CFAR(rd, win_len=50,win_width=50,guard_len=10,guard_width=10, threshold=20)
    rd_cfar, fps_cfar = peak_detector2(rd_all_channels_az[0], max_detections=max_detections,
                                       threshold_rel=rel_peak_threshold)
    target_index = np.where(rd_cfar == 1)  # any where there is a hit, get the index of tha tlocation
    num_targets = len(target_index[0])
    if num_targets == 0:
        print('no targets')
        target_list = None

    hit_idx = 0  # some hit targest may generate multiple hits (ie, multiple at same range, but different azimuth)
    for hit in range(num_targets):

        loc_dict = {}
        ddim_idx = target_index[1][hit]  # index  in doopper dimension
        rdim_idx = target_index[0][hit]  # index  in range dimension
        doa_az, all_doa_az_bins = target_DOA_estimation(rd_all_channels_az, azPixels,
                                                        rdim_idx, ddim_idx, antenna_spacing_wl=antenna_spacing_wl,
                                                        fov=radar_fov_az,
                                                        DOA_method="Bartlett")

        if includes_elevation == False:
            doa_el = 0
            all_doa_el_bins = [0]
        elif len(rd_all_channels_el) < 2:  # needs to have at least 2 channel to get elevation
            doa_el = 0
            all_doa_el_bins = [0]
        else:
            doa_el, all_doa_el_bins = target_DOA_estimation(rd_all_channels_el, elPixels,
                                                            rdim_idx, ddim_idx,
                                                            fov=radar_fov_el,
                                                            DOA_method="Bartlett")

        R_dist = rngDomain[rdim_idx]  # get range at index where peak/hit was detected
        loc_dict['range'] = R_dist
        # ignore hits that are closer than this distance and futher than 90%of max range
        # for doa_az_peak in all_doa_az_bins:
        #     for doa_el_peak in all_doa_el_bins:
        if ((loc_dict['range'] > min_detect_range) and (loc_dict['range'] < np.max(rngDomain) * .9)):
            loc_dict['azimuth'] = doa_az  # in degrees
            loc_dict['elevation'] = doa_el
            loc_dict['cross_range_dist'] = R_dist * np.sin(doa_az * np.pi / 180)
            loc_dict['xpos'] = R_dist * np.cos(
                doa_az * np.pi / 180)  # this is distnace as defined in +x in front of sensor
            loc_dict['ypos'] = R_dist * np.sin(doa_az * np.pi / 180)  # +y and -y is cross range dimenionson,
            loc_dict['zpos'] = R_dist * np.sin(doa_el * np.pi / 180)
            loc_dict['velocity'] = velDomain[ddim_idx]
            Pr = np.abs(rd_all_channels_az[0][rdim_idx][ddim_idx])
            loc_dict['p_received'] = np.float64(Pr)
            Pr_dB = 10 * np.log10(Pr)
            # TODO get transmit power from API
            Pt = incident_power  # 1Watt, input power, 0dBw is source power
            Pt_dB = 10 * np.log10(Pt)

            # user radar range equation to scale results by range to get relative rcs
            # is there a better way to do this? This will not work for objects in near field
            # gain used in dB, should probably use the actual antenna pattern gain, but this will be used for testing
            Gt = 10.67  # this is about the gain for hpbw =120deg
            Gr = 10.67
            # radar range equation in dB
            rcs_scaled_dB = Pr_dB + 30 * np.log10(4 * np.pi) + 40 * np.log10(R_dist) - (
                    Pt_dB + Gt + Gr + 20 * np.log10(3e8 / (centerFreq)))
            if rcs_scaled_dB > rcs_min_detect:  # only add if peak rcs is above min value specified
                loc_dict['rcs'] = rcs_scaled_dB
                target_list[hit_idx] = deepcopy(loc_dict)
                hit_idx += 1
                # target_list['original_time_index'] = time
    # if target recorded, add it to the list

    time_after_target_list = walltime.time()
    time_target_list = time_after_target_list - time_before_target_list
    if time_target_list == 0:
        time_target_list = 1
    fps_target_list = 1 / time_target_list

    if target_list is None:
        target_list = []
    if return_cfar:
        return target_list, fps_target_list, rd_cfar
    else:
        return target_list, fps_target_list


def plot_target_list(target_list, plot_2d=True, plot_3d=True, maximum_range=None, reference_points=None,figure_size=(12,8)):
    """
    Create professional visualizations of radar target detections.

    Parameters:
    -----------
    target_list : dict
        Dictionary containing detected targets with position and RCS information
    plot_2d : bool
        Whether to create 2D plots (top-down and elevation views)
    plot_3d : bool
        Whether to create a 3D scatter plot
    maximum_range : float
        Maximum range to display (for axis limits)
    reference_points : dict
        Dictionary of reference points to plot (e.g., corner reflectors)
    """
    if not target_list:
        print("No targets to display")
        return

    # Extract data for plotting
    x_values = [target_list[t]['xpos'] for t in target_list]
    y_values = [target_list[t]['ypos'] for t in target_list]
    z_values = [target_list[t]['zpos'] for t in target_list]
    rcs_values = [target_list[t]['rcs'] for t in target_list]

    # Determine axis limits dynamically if not specified
    if maximum_range is None:
        max_x = max(abs(max(x_values)), 1000)  # At least 1000m range
        max_y = max(abs(max(y_values)), abs(min(y_values)), 1000)
        max_z = max(abs(max(z_values)), abs(min(z_values)), 100)
    else:
        max_x = maximum_range
        max_y = maximum_range
        max_z = maximum_range / 2  # Typically elevation range is smaller

    # Set style for professional plotting
    plt.style.use('dark_background')

    # Custom colormap for RCS values
    cmap = plt.cm.plasma

    if plot_2d:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size, dpi=120)
        fig.suptitle('Radar Target Detection Results', fontsize=16, fontweight='bold', y=0.98)

        # Normalize RCS values for consistent coloring
        norm = plt.Normalize(min(rcs_values), max(rcs_values))

        # Top-down view (azimuth position)
        scatter1 = ax1.scatter(
            x_values, y_values,
            c=rcs_values,
            cmap=cmap,
            s=80,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            norm=norm
        )

        # Plot reference points if provided
        if reference_points:
            for label, pos in reference_points.items():
                ax1.scatter(pos[0], pos[1],
                            marker='X', s=100, color='lime',
                            edgecolors='black', linewidth=1.5,
                            label=label, zorder=10)

        # Configure top-down view
        ax1.set_xlim([0, max_x * 1.1])
        ax1.set_ylim([-max_y, max_y])
        ax1.set_xlabel('Range (m)', fontsize=12)
        ax1.set_ylabel('Cross-Range (m)', fontsize=12)
        ax1.set_title('Azimuth View (Top-Down)', fontsize=14, fontweight='bold')
        ax1.grid(True, color='gray', linestyle='--', alpha=0.5)

        # Add range rings (concentric circles)
        range_rings = [max_x / 4, max_x / 2, 3 * max_x / 4]
        for r in range_rings:
            circle = plt.Circle((0, 0), r, fill=False, color='gray',
                                linestyle='--', alpha=0.3)
            ax1.add_patch(circle)
            ax1.text(r * 0.7, r * 0.2, f"{int(r)}m", color='gray', alpha=0.7, fontsize=8)

        # Add radar position indicator
        ax1.scatter(0, 0, marker='o', s=150, color='white', edgecolors='black', zorder=10)
        ax1.text(0, 0, "RADAR", ha='center', va='center', fontsize=8, weight='bold')

        # Elevation view
        scatter2 = ax2.scatter(
            x_values, z_values,
            c=rcs_values,
            cmap=cmap,
            s=80,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            norm=norm
        )

        # Plot reference points in elevation view
        if reference_points:
            for label, pos in reference_points.items():
                if len(pos) > 2:  # If z-coordinate is provided
                    ax2.scatter(pos[0], pos[2],
                                marker='X', s=100, color='lime',
                                edgecolors='black', linewidth=1.5,
                                label=label, zorder=10)
                else:
                    ax2.scatter(pos[0], 0,
                                marker='X', s=100, color='lime',
                                edgecolors='black', linewidth=1.5,
                                label=label, zorder=10)

        # Configure elevation view
        ax2.set_xlim([0, max_x * 1.1])
        ax2.set_ylim([-max_z, max_z])
        ax2.set_xlabel('Range (m)', fontsize=12)
        ax2.set_ylabel('Height (m)', fontsize=12)
        ax2.set_title('Elevation View (Side)', fontsize=14, fontweight='bold')
        ax2.grid(True, color='gray', linestyle='--', alpha=0.5)

        # Add radar position indicator
        ax2.scatter(0, 0, marker='o', s=150, color='white', edgecolors='black', zorder=10)
        ax2.text(0, 0, "RADAR", ha='center', va='center', fontsize=8, weight='bold')

        # Add a colorbar for RCS values
        cbar = fig.colorbar(scatter1, ax=[ax1, ax2], pad=0.01, shrink=0.8)
        cbar.set_label('RCS (dBsm)', fontsize=12)

        # Add legend and ensure it doesn't have duplicate entries
        if reference_points:
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend = ax1.legend(by_label.values(), by_label.keys(),
                                title="Reference Points",
                                loc='upper right', framealpha=0.7)
            legend.get_title().set_fontweight('bold')

        # Tight layout for better spacing
        # plt.tight_layout(rect=[0, 0, 1, 0.96])

    if plot_3d:
        # Create 3D plot
        fig = plt.figure(figsize=(figure_size[1], figure_size[1]), dpi=120)
        ax3d = fig.add_subplot(111, projection='3d')

        # Plot targets with RCS-based coloring
        scatter3d = ax3d.scatter(
            x_values, y_values, z_values,
            c=rcs_values,
            cmap=cmap,
            s=70,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5
        )

        # Add reference points if provided
        if reference_points:
            for label, pos in reference_points.items():
                if len(pos) > 2:  # If z-coordinate is provided
                    ax3d.scatter(pos[0], pos[1], pos[2],
                                 marker='X', s=100, color='lime',
                                 edgecolors='black', linewidth=1.5,
                                 label=label)
                else:
                    ax3d.scatter(pos[0], pos[1], 0,
                                 marker='X', s=100, color='lime',
                                 edgecolors='black', linewidth=1.5,
                                 label=label)

        # Add radar position indicator
        ax3d.scatter(0, 0, 0, marker='o', s=150, color='white', edgecolors='black')
        ax3d.text(0, 0, 0, "RADAR", fontsize=10, weight='bold')

        # Set axis limits and labels
        ax3d.set_xlim([0, max_x * 1.1])
        ax3d.set_ylim([-max_y, max_y])
        ax3d.set_zlim([-max_z, max_z])
        ax3d.set_xlabel('Range (m)', fontsize=12, labelpad=10)
        ax3d.set_ylabel('Cross-Range (m)', fontsize=12, labelpad=10)
        ax3d.set_zlabel('Height (m)', fontsize=12, labelpad=10)

        # Add a title
        ax3d.set_title('3D Target Distribution', fontsize=16, fontweight='bold', y=1.02)

        # Add grid for better depth perception
        ax3d.grid(True, linestyle='--', alpha=0.3)

        # Add colorbar
        cbar = fig.colorbar(scatter3d, ax=ax3d, pad=0.1, shrink=0.8)
        cbar.set_label('RCS (dBsm)', fontsize=12)

        # Add legend if we have reference points
        if reference_points:
            handles, labels = ax3d.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend = ax3d.legend(by_label.values(), by_label.keys(),
                                 title="Reference Points",
                                 loc='upper right', framealpha=0.7)
            legend.get_title().set_fontweight('bold')

        # Set a good viewing angle
        ax3d.view_init(elev=25, azim=-35)

    plt.show()
def animate_target_list(target_lists, range_doppler=None, maximum_range=None, reference_points=None,
                       output_file=None, interval=200, fps=10,figure_size=(16,8)):
    """
    Create an animation of radar target detections over time.

    Parameters:
    -----------
    target_lists : list of dicts
        List of target dictionaries, where each dict represents targets at one time frame
    plot_2d : bool
        Whether to create 2D plots (top-down and elevation views)
    plot_3d : bool
        Whether to create a 3D scatter plot
    maximum_range : float
        Maximum range to display (for axis limits)
    reference_points : dict
        Dictionary of reference points to plot (e.g., corner reflectors)
    output_file : str
        Filename to save the animation (if None, animation is displayed but not saved)
    interval : int
        Interval between frames in milliseconds
    fps : int
        Frames per second for saved animation
    """
    import matplotlib.animation as animation

    if not target_lists or len(target_lists) == 0:
        print("No target data to animate")
        return

    # Set style for professional plotting
    plt.style.use('dark_background')

    # Determine overall max/min values across all frames for consistent scaling
    all_x = []
    all_y = []
    all_z = []
    all_rcs = []

    for frame_targets in target_lists:
        if frame_targets:
            x_values = [frame_targets[t]['xpos'] for t in frame_targets]
            y_values = [frame_targets[t]['ypos'] for t in frame_targets]
            z_values = [frame_targets[t]['zpos'] for t in frame_targets]
            rcs_values = [frame_targets[t]['rcs'] for t in frame_targets]

            all_x.extend(x_values)
            all_y.extend(y_values)
            all_z.extend(z_values)
            all_rcs.extend(rcs_values)

    if not all_x:  # No valid targets in any frame
        print("No valid targets found in any frame")
        return

    # Determine axis limits dynamically if not specified
    if maximum_range is None:
        max_x = max(max(all_x) if all_x else 0, 1000)  # At least 1000m range
        max_y = max(max(abs(y) for y in all_y) if all_y else 0, 1000)
        max_z = max(max(abs(z) for z in all_z) if all_z else 0, 100)
    else:
        max_x = maximum_range
        max_y = maximum_range
        max_z = maximum_range / 2  # Typically elevation range is smaller

    min_rcs = min(all_rcs) if all_rcs else 0
    max_rcs = max(all_rcs) if all_rcs else 1
    norm = plt.Normalize(min_rcs, max_rcs)
    cmap = plt.cm.plasma

    # Create figure and axes based on plot options
    if range_doppler is None:
        fig = plt.figure(figsize=figure_size, dpi=120)
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])  # Top-down view
        ax2 = fig.add_subplot(gs[0, 1])  # Elevation view
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')  # 3D view
        rd_ax = None
    else:
        fig = plt.figure(figsize=figure_size, dpi=120)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])  # Top-down view
        ax2 = fig.add_subplot(gs[0, 1])  # Elevation view
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')  # 3D view
        rd_ax = fig.add_subplot(gs[1, 1])  # Range-Doppler map


    fig.suptitle('Radar Target Detection Animation', fontsize=16, fontweight='bold', y=0.98)

    # Initialize scatter objects
    scatter1 = scatter2 = scatter3 = None
    rd_plot = None

    # Create empty scatter plots
    # Top-down view (azimuth position)
    scatter1 = ax1.scatter([], [], c=[], cmap=cmap, s=100, alpha=0.8, edgecolors='white', linewidth=0.5, norm=norm)

    # Configure top-down view
    ax1.set_xlim([0, max_x * 1.1])
    ax1.set_ylim([-max_y, max_y])
    ax1.set_xlabel('Range (m)', fontsize=12)
    ax1.set_ylabel('Cross-Range (m)', fontsize=12)
    ax1.set_title('Azimuth View (Top-Down)', fontsize=14, fontweight='bold')
    ax1.grid(True, color='gray', linestyle='--', alpha=0.5)

    # Add range rings (concentric circles)
    range_rings = [max_x / 4, max_x / 2, 3 * max_x / 4]
    for r in range_rings:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
        ax1.text(r * 0.7, r * 0.2, f"{int(r)}m", color='gray', alpha=0.7, fontsize=8)

    # Add radar position indicator
    ax1.scatter(0, 0, marker='o', s=150, color='white', edgecolors='black', zorder=10)
    ax1.text(0, 0, "RADAR", ha='center', va='center', fontsize=8, weight='bold')

    # Elevation view
    scatter2 = ax2.scatter([], [], c=[], cmap=cmap, s=100, alpha=0.8, edgecolors='white', linewidth=0.5, norm=norm)

    # Configure elevation view
    ax2.set_xlim([0, max_x * 1.1])
    ax2.set_ylim([-max_z, max_z])
    ax2.set_xlabel('Range (m)', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('Elevation View (Side)', fontsize=14, fontweight='bold')
    ax2.grid(True, color='gray', linestyle='--', alpha=0.5)

    # Add radar position indicator
    ax2.scatter(0, 0, marker='o', s=150, color='white', edgecolors='black', zorder=10)
    ax2.text(0, 0, "RADAR", ha='center', va='center', fontsize=8, weight='bold')


    # 3D scatter plot
    scatter3 = ax3.scatter([], [], [], c=[], cmap=cmap, s=100, alpha=0.8, edgecolors='white', linewidth=0.5, norm=norm)

    # Set axis limits and labels
    ax3.set_xlim([0, max_x * 1.1])
    ax3.set_ylim([-max_y, max_y])
    ax3.set_zlim([-max_z, max_z])
    ax3.set_xlabel('Range (m)', fontsize=12, labelpad=10)
    ax3.set_ylabel('Cross-Range (m)', fontsize=12, labelpad=10)
    ax3.set_zlabel('Height (m)', fontsize=12, labelpad=10)
    ax3.set_title('3D Target Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.scatter(0, 0, 0, marker='o', s=150, color='white', edgecolors='black')
    ax3.text(0, 0, 0, "RADAR", fontsize=10, weight='bold')
    ax3.view_init(elev=25, azim=-35)

    if rd_ax is not None and range_doppler is not None:
        # Get dimensions from first frame
        if len(range_doppler) > 0:
            rd_data = 20*np.log10(np.fmax(np.abs(range_doppler[0]),1e-30))
            rd_max = np.max(rd_data)
            vmin = rd_max - 80  # Show 50dB dynamic range
            vmax = rd_max

            # Create initial image
            rd_plot = rd_ax.imshow(rd_data,
                                   cmap='jet',
                                   aspect='auto',
                                   vmin=vmin,
                                   vmax=vmax)
            rd_ax.set_title('Range-Doppler Map', fontsize=14, fontweight='bold')
            rd_ax.set_ylabel('Doppler Bin', fontsize=12)
            rd_ax.set_xlabel('Range Bin', fontsize=12)

            # Add colorbar
            cbar = fig.colorbar(rd_plot, ax=rd_ax)
            cbar.set_label('Power (dB)', fontsize=12)

    # Add reference points if provided
    if reference_points:
        reference_point=reference_points[0]
        ref_label_text_plot1 = []
        ref_label_text_plot2 = []
        ref_label_text_plot3 = []
        for label, pos in reference_point.items():
            scatter1_ref = ax1.scatter([], [], marker='^', s=50, color='lime', edgecolors='white', label=label, zorder=5)
            ref_label_text_plot1.append(ax1.text(pos[0] + max_x * 0.02, pos[1] + max_y * 0.02, label, color='lime', fontsize=10))

            scatter2_ref = ax2.scatter([], [], marker='^', s=50, color='lime', edgecolors='white', zorder=5)
            ref_label_text_plot2.append(ax2.text(pos[0] + max_x * 0.02, (pos[2]) + max_z * 0.02, label, color='lime', fontsize=10))

            scatter3_ref = ax3.scatter([], [], [], marker='^', s=50, color='lime', edgecolors='white', label=label, zorder=5)
            ref_label_text_plot3.append(ax3.text(pos[0], pos[1], pos[2] if len(pos) > 2 else 0, label, color='lime', fontsize=10))

    # Add colorbar for RCS values

    cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter1 if scatter1 is not None else scatter3, cax=cbar_ax)
    cbar.set_label('RCS (dBsm)', fontsize=12)


    # Frame counter display
    frame_text = fig.text(0.01, 0.01, "", fontsize=10, color='white')
    num_detection_text = fig.text(0.01, 0.06, "", fontsize=10, color='white')
    num_actors_text = fig.text(0.01, 0.11, "", fontsize=10, color='white')

    # Create lists to store artists that will be returned by init and update functions
    artists_2d = []
    artists_3d = []

    artists_2d = [scatter1, scatter2]
    artists_3d = [scatter3]

    all_artists = artists_2d + artists_3d + [frame_text]+ [num_detection_text]+ [num_actors_text]

    def init():
        # Initialize function for animation
        frame_text.set_text("")
        num_detection_text.set_text("")
        num_actors_text.set_text("")
        scatter1.set_offsets(np.empty((0, 2)))
        scatter1.set_array(np.array([]))
        scatter2.set_offsets(np.empty((0, 2)))
        scatter2.set_array(np.array([]))

        scatter3._offsets3d = (np.array([]), np.array([]), np.array([]))
        scatter3.set_array(np.array([]))

        return all_artists

    def update(frame):

        # Update range-doppler map if provided
        if rd_ax is not None and range_doppler is not None and frame < len(range_doppler):
            rd_data = 20*np.log10(np.fmax(np.abs(range_doppler[frame]),1e-30))
            rd_plot.set_array(rd_data)


        # Update function for animation
        frame_text.set_text(f"Frame: {frame+1}/{len(target_lists)}")
        if target_lists is not None:
            num_detection_text.set_text(f"Number of detections: {len(target_lists[frame])}")
        if reference_points is not None:
            num_actors_text.set_text(f"Number of actors: {len(reference_points[frame])}")

        # Handle frame out of range
        if frame >= len(target_lists):
            scatter1.set_offsets(np.empty((0, 2)))
            scatter1.set_array(np.array([]))
            scatter2.set_offsets(np.empty((0, 2)))
            scatter2.set_array(np.array([]))

            scatter3._offsets3d = (np.array([]), np.array([]), np.array([]))
            scatter3.set_array(np.array([]))

            return all_artists

        targets = target_lists[frame]
        if not targets:
            # No targets in this frame
            scatter1.set_offsets(np.empty((0, 2)))
            scatter1.set_array(np.array([]))
            scatter2.set_offsets(np.empty((0, 2)))
            scatter2.set_array(np.array([]))


            scatter3._offsets3d = (np.array([]), np.array([]), np.array([]))
            scatter3.set_array(np.array([]))
        else:
            # Extract data for current frame
            x_values = [targets[t]['xpos'] for t in targets]
            y_values = [targets[t]['ypos'] for t in targets]
            z_values = [targets[t]['zpos'] for t in targets]
            rcs_values = [targets[t]['rcs'] for t in targets]


            # Update top-down view
            scatter1.set_offsets(np.column_stack((x_values, y_values)))
            scatter1.set_array(np.array(rcs_values))

            if reference_points:
                reference_point = reference_points[frame]
                pos_x = []
                pos_y = []
                pos_z = []
                n=0
                for label, pos in reference_point.items():
                    pos_x.append(pos[0])
                    pos_y.append(pos[1])
                    pos_z.append(pos[2])
                    ref_label_text_plot1[n].set_position([pos[0] + max_x * 0.02, pos[1] + max_y * 0.02])
                    ref_label_text_plot2[n].set_position([pos[0] + max_x * 0.02, (pos[2]) + max_z * 0.02])
                    n+=1
                scatter1_ref.set_offsets(np.column_stack((pos_x, pos_y)))
                scatter2_ref.set_offsets(np.column_stack((pos_x, pos_z)))



            # Update elevation view
            scatter2.set_offsets(np.column_stack((x_values, z_values)))
            scatter2.set_array(np.array(rcs_values))


            # Update 3D view
            scatter3._offsets3d = (np.array(x_values), np.array(y_values), np.array(z_values))
            scatter3.set_array(np.array(rcs_values))
            if reference_points:
                reference_point = reference_points[frame]
                pos_x = []
                pos_y = []
                pos_z = []
                n=0
                for label, pos in reference_point.items():
                    pos_x.append(pos[0])
                    pos_y.append(pos[1])
                    pos_z.append(pos[2])
                    ref_label_text_plot3[n].set_position(pos)
                    n+=1
                scatter3_ref._offsets3d = (np.array(pos_x), np.array(pos_y), np.array(pos_z))



        return all_artists

    # Create animation without blit for 3D plots (blit doesn't work well with 3D)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(target_lists),
        init_func=init,
        blit=False,  # Disable blit for 3D plots
        interval=interval
    )

    # Add timestamp to the figure
    # plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save animation if output file is specified
    if output_file:
        try:
            ani.save(output_file, writer='pillow', fps=fps)
            print(f"Animation saved to {output_file}")
        except Exception as e:
            print(f"Error saving animation: {e}")

    plt.show()
    return ani


def channel_capacity(H, BW_Hz=10e6, temperture=290):
    H = np.array(H, ndmin=3)
    # H  = H[:,:1]
    Nrx = H.shape[1]
    Ntx = H.shape[2]
    num_time_steps = H.shape[0]
    # SNR = np.power(10,SNRdb/10)
    all_cap = []

    kB = 1.3806488e-23
    N0 = kB * temperture
    Pb = N0

    for n in range(num_time_steps):
        H2 = H[n]  # /np.linalg.norm(H[n],ord=1)
        cap = np.identity(Nrx) + 1 / Pb / Ntx * np.dot(H2, H2.conj().T)
        cap2 = np.linalg.det(cap)
        cap3 = BW_Hz * np.log10(np.abs(cap2)) / np.log10(2)
        all_cap.append(cap3)

    return np.array(all_cap)






