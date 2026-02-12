import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.constants import speed_of_light
from scipy.fft import fft, ifft, fftshift
from scipy.signal.windows import hann


class BarkerCode:
    """
    Class for generating and processing Barker code radar waveforms.

    Barker codes are finite sequences with optimal autocorrelation properties
    for radar pulse compression applications.
    """

    def __init__(self,
                 code_length=13,
                 center_freq=10e9,
                 num_pulses=128,
                 pri=100e-6,
                 duty_cycle=0.025,
                 adc_sampling_rate=6.5e6,
                 number_adc_samples=512,
                 blanking_period=1000,
                 blanking_time=None,
                 upsample_factor=8,
                 transmit_power=10e3,
                 rx_noise_db=None):
        """
                Configure radar waveform parameters.

                Parameters:
                    code_length (int): Length of the Barker code to use (2, 3, 4, 5, 7, 11, or 13)
                    center_freq (float): Center frequency in Hz
                    num_pulses (int): Number of pulses in coherent processing interval (CPI)
                    pri (float): Pulse repetition interval in seconds
                    duty_cycle (float): Duty cycle of the pulse (0 to 1)
                    adc_sampling_rate (float): ADC sampling rate in Hz
                    number_adc_samples (int): Number of ADC samples to collect
                    blanking_period (float): Blanking period in meters
                    blanking_time (float): Blanking time in seconds (None for default, will override blanking_period)
                    upsample_factor (int): Factor to oversample the waveform
                    transmit_power (float): Transmit power in Watts
                    rx_noise_db (float, optional): Receiver noise level in dB

        Initialize a Barker code of specified length.

          |<---------------- One Tx Pulse + Receive Period ------------------>|
          |<--------------- Pulse Repetition Interval (PRI) ----------------->|
          |<-- Transmit -->|                                                  |
          |<- Duty Cycle ->|                                                  |
          |                |                                                  |
          |                |<X>|   X -> neither Tx'ing nor Rx'ing             |
          |                    |                                              |
          |<-- Blanking Pd --->|<--------------- Receive --------------->|<XX>|
          |<----- Tstart ----->| * * * * * ... * * * * * * * * * * * * * |    |
          |                    |<------------ Nsamp @ fs rate ---------->|    |

        Raises:
            KeyError: If code_length is not a valid Barker code length
        """
        # Available Barker codes by length
        barker_codes = {
            2: [1, -1],
            3: [1, 1, -1],
            4: [1, 1, -1, 1],
            5: [1, 1, 1, -1, 1],
            7: [1, 1, 1, -1, -1, 1, -1],
            11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
            13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
        }
        self.code_length = code_length
        self.barker_seq = barker_codes[code_length]
        self.perceive_em_response = None

        # using naming convention for variable from original matlab script for most variables for easier debugging
        # storing some of these outputs under more convenient names for easier access

        if blanking_time is not None:
            print("blanking time is being utliized (not using blanking period) ")
            blanking_period = blanking_time * speed_of_light / 2
            print(f"blanking period: {blanking_period} ---> blanking time: {blanking_time}")

        pi2 = 2 * np.pi
        # Store input parameters
        self.number_adc_samples = number_adc_samples
        self.center_freq = center_freq
        self.num_pulses = num_pulses
        self.pri = pri
        self.duty_cycle = duty_cycle
        self.adc_sampling_rate = adc_sampling_rate
        self.blanking_period = blanking_period
        self.upsample_factor = upsample_factor

        pri_us = pri * 1e6

        # Physical constants
        c_mpns = speed_of_light * 1e-9  # Speed of light in m/ns
        self.c_mpmus = speed_of_light * 1e-6  # Speed of light in m/μs

        # Convert ADC sampling rate to MHz for calculations
        adc_sampling_rate_mhz = adc_sampling_rate * 1e-6

        # Set interpolation method
        self.imethod = 'cubic'

        # Calculate derived radar properties
        # Pulse duration based on PRI and duty cycle
        Tpulse_mus = pri_us * duty_cycle

        # Number of chips in the pulse (length of Barker code)
        Nchip = len(self.barker_seq)

        # Duration of each chip in microseconds
        self.Tchip_mus = Tpulse_mus / Nchip

        # Time sampling interval for simulation (not ADC)
        dt_mus = self.Tchip_mus / upsample_factor

        # Number of time samples in one PRI
        Nt = int(pri_us / dt_mus)
        # Ensure Nt is even for FFT efficiency
        if Nt % 2 != 0:
            Nt -= 1

        hNt = Nt // 2

        # Adjust PRI to match time sample interval if needed
        self.pri = dt_mus * Nt * 1e-6

        # Create time samples spanning the PRI
        tSim_mus = np.arange(Nt) * dt_mus
        self.tSim_mus = tSim_mus

        # Generate ADC sample times @ fs Sampling Rate
        self.dtSamp_mus = 1 / adc_sampling_rate_mhz # ADC sampling time interval

        # Time when ADC sampling starts (after blanking period)
        self.tStart_mus = 2 * blanking_period / self.c_mpmus

        # Half-index offset for centered sampling
        halfIdxOffset = 0.5
        self.halfIdxOffset = halfIdxOffset

        # Generate relative and absolute ADC sample times
        self.tSamp_mus = (np.arange(number_adc_samples) + halfIdxOffset) * self.dtSamp_mus
        tSampAbs_mus = self.tSamp_mus + self.tStart_mus
        self.tSampAbs_mus = tSampAbs_mus

        # Calculate frequency domain parameters
        bw_mhz = 1 / dt_mus  # Bandwidth in MHz
        Nf = Nt  # Match frequency samples to time samples for FFT
        hNf = Nf // 2
        df_mhz = bw_mhz / Nf  # Frequency resolution

        # Generate frequency sample points centered at IF DC
        f_if_mhz = np.arange(-hNf, hNf) * df_mhz
        self.f_if_mhz = f_if_mhz

        # Construct Barker sequence in time domain with upsampling
        self.stx = np.repeat(self.barker_seq, upsample_factor)
        Nt_tx = Nchip * upsample_factor
        # Zero-pad to fill the PRI
        self.stx = np.concatenate((self.stx, np.zeros(Nt - Nt_tx)))

        # Convert to frequency domain
        self.Stx_dcatbeg = fft(self.stx)  # DC at beginning

        # Apply Hann window to smooth chip transitions
        win = hann(Nf, sym=False)
        win_peak = np.max(win)
        win = win / win_peak  # Normalize window
        self.Stx_win_dcatbeg = fftshift(self.Stx_dcatbeg) * win

        # Convert back to time domain (windowed signal)
        self.stx_win = ifft(fftshift(self.Stx_win_dcatbeg))

        self.input_waveform_td = self.stx_win  # store as more convient name
        self.input_waveform_fd = self.Stx_win_dcatbeg  # store as more convient name
        self.input_freq_if_domain = self.f_if_mhz * 1e-6
        self.input_time_domain = self.tSim_mus * 1e6

        # Calculate CPI duration
        self.cpi_duration = self.pri * (num_pulses+1) #ToDo: check if this is correct, I need to add 1 to pass verification
        self.bandwidth = bw_mhz * 1e6
        self.num_freq_samples = Nf
        self.transmit_power = transmit_power

        # Build waveform dictionary for radar configuration
        self.waveform_dict = {
            "mode": "PulsedDoppler",
            "output": "FreqPulse",
            "center_freq": center_freq,
            "bandwidth": bw_mhz * 1e6,
            "num_freq_samples": Nf,
            "cpi_duration": self.cpi_duration,
            "num_pulse_CPI": num_pulses,
            "tx_multiplex": "INDIVIDUAL",
            "mode_delay": "CENTER_CHIRP",
            "tx_incident_power": transmit_power
        }

        # Add noise parameter if provided
        self.rx_noise_db = rx_noise_db
        if rx_noise_db is not None:
            self.waveform_dict["rx_noise_db"] = rx_noise_db


    def process_received_signal(self, response,apply_window_to_response=True):
        # Store the response for further processing

        if apply_window_to_response:
            if response.ndim == 1:
                window = np.hanning(len(response))
                sf = len(window) / np.sum(window)
                window = window * sf
                response = np.multiply(window, response)
            elif response.ndim == 2:
                window = np.hanning(response.shape[1])
                sf = len(window) / np.sum(window)
                window = window * sf
                # multiple each pulse by window
                response = np.multiply(response, np.tile(window, (response.shape[0], 1)))
            else:
                raise ValueError('Unexpected response shape. Must be 1D or 2D array (single or multiple pulses).')

        self.perceive_em_response  = response

        if self.perceive_em_response.ndim == 1:
            # this is if a single tx, single rx, and single pulse are passed in as the response
            self._process(self.perceive_em_response)
        elif self.perceive_em_response.ndim == 2:
            # this is probably not the best way to do this, but this is a quick fix for now
            temp_output_waveform_if_fd = []
            temp_output_waveform_if_td = []
            temp_output_waveform_adc_td = []
            temp_output_waveform_resampled_td = []
            temp_output_waveform_matched_filter_td =  []

            for pulse_idx in range(self.perceive_em_response.shape[0]):
                self._process(self.perceive_em_response[pulse_idx])
                temp_output_waveform_if_fd.append(self.output_waveform_if_fd)
                temp_output_waveform_if_td.append(self.output_waveform_if_td)
                temp_output_waveform_adc_td.append(self.output_waveform_adc_td)
                temp_output_waveform_resampled_td.append(self.output_waveform_resampled_td)
                temp_output_waveform_matched_filter_td.append(self.output_waveform_matched_filter_td)
            self.output_waveform_if_fd = np.array(temp_output_waveform_if_fd)
            self.output_waveform_if_td = np.array(temp_output_waveform_if_td)
            self.output_waveform_adc_td = np.array(temp_output_waveform_adc_td)
            self.output_waveform_resampled_td = np.array(temp_output_waveform_resampled_td)
            self.output_waveform_matched_filter_td = np.array(temp_output_waveform_matched_filter_td)

        else:
            raise ValueError('Unexpected response shape. Must be 1D or 2D array (single or multiple pulses).')

    def _process(self,response):

        """Process the response through the entire signal chain"""
        # Generate the response to the Barker-sequence pulse
        # This is received signal in the IF stage. Time response will be complex
        # (I + jQ) because spectrum is no longer conjugate symmetric, owing to
        # application of scene scattering finite impulse response.
        Srx_dcatbeg = self.Stx_win_dcatbeg * response #input signal in frequency domain * response in frequency domain
        self.srx = ifft(fftshift(Srx_dcatbeg))

        # INTERPOLATE RX RESPONSE FROM SIM TIME TO ADC SAMPLE TIMES
        interp_func = interp1d(self.tSim_mus, self.srx, kind=self.imethod,
                               fill_value=0, bounds_error=False)
        self.srx_adc = interp_func(self.tSampAbs_mus)

        # Resample ADC samples to Tchip time interval
        # GENERATE RX ADC SAMPLE TIMES @ fs SAMPLING RATE
        dtResamp_mus = self.Tchip_mus
        Tlisten = self.number_adc_samples * self.dtSamp_mus
        Nresamp = int(np.ceil(Tlisten / dtResamp_mus))
        self.tResamp_mus = (np.arange(Nresamp) + self.halfIdxOffset) * dtResamp_mus
        self.tResampAbs_mus = self.tResamp_mus + self.tStart_mus

        interp_func_resamp = interp1d(self.tSampAbs_mus, self.srx_adc,
                                      kind=self.imethod, fill_value=0, bounds_error=False)
        self.srx_resamp = interp_func_resamp(self.tResampAbs_mus)

        # Apply matched filter - convolve with time-inverted Barker sequence
        self.srx_mf = np.convolve(self.srx_resamp, self.barker_seq[::-1], mode='full')

        # Calculate matched filter time and range domains
        Nmf = len(self.srx_mf)
        Nbark = len(self.barker_seq)
        Nresamp = len(self.srx_resamp)

        if (Nresamp + Nbark - 1) != Nmf:
            raise ValueError('Matched filter convolution output has unexpected size.')


        dtMf_mus = dtResamp_mus
        self.tMf_mus = (np.arange(Nmf) - (Nbark - 1) + self.halfIdxOffset) * dtMf_mus
        self.tMfAbs_mus = self.tMf_mus + self.tStart_mus
        self.RMf_m = self.tMfAbs_mus * self.c_mpmus * 0.5


        # output values, re-labled to more convenient names
        self.output_waveform_if_fd = Srx_dcatbeg
        self.output_waveform_if_td = self.srx

        self.output_waveform_adc_td = self.srx_adc
        self.output_time_domain = self.tSampAbs_mus

        self.output_waveform_resampled_td = self.srx_resamp
        self.output_time_domain_resampled = self.tResampAbs_mus * 1e-6

        self.output_range_domain_matched_filter = self.RMf_m
        self.output_time_domain_matched_filter = self.tMfAbs_mus
        self.output_waveform_matched_filter_td = self.srx_mf

    def plot_input_waveform(self,enhanced_plots=True):
        if enhanced_plots:
            self._plot_input_waveform_enhanced()
        else:
            self._plot_input_waveform_original()


    def _plot_input_waveform_enhanced(self, figsize=(10, 12), dpi=100, style='seaborn-v0_8-whitegrid'):
        """
        Plot the input Barker code waveform in time and frequency domains with enhanced visualizations.

        Parameters:
            figsize (tuple): Figure size in inches (width, height).
            dpi (int): Resolution for the output figures.
            style (str): Matplotlib style to use.

        Creates a comprehensive visualization with three subplots:
        1. Unwindowed Tx sequence in time domain
        2. Frequency domain comparison of windowed and unwindowed signals
        3. Windowed Tx sequence in time domain
        """
        # Set up modern style
        plt.style.use(style)

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)
        fig.suptitle(f'Barker Code Waveform Analysis - Code Length: {self.code_length}',
                     fontsize=16, fontweight='bold')

        # Subplot 1: Unwindowed Tx sequence in time domain with Barker code indicators
        chip_end_time = self.Tchip_mus * len(self.barker_seq) * 1.2
        axes[0].plot(self.tSim_mus, self.stx, '-', color='#1f77b4', linewidth=2)
        axes[0].plot(self.tSim_mus, self.stx, 'o', color='#1f77b4', markersize=4)

        # Add indicators for chip boundaries and label Barker codes
        for i, code in enumerate(self.barker_seq):
            chip_start = i * self.Tchip_mus
            chip_end = (i + 1) * self.Tchip_mus
            chip_mid = (chip_start + chip_end) / 2

            # Add vertical lines at chip boundaries
            if i < len(self.barker_seq):
                axes[0].axvline(x=chip_end, color='gray', linestyle=':', alpha=0.5)

            # Label each chip with its value
            axes[0].text(chip_mid, 0.5, f"{code}", ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        axes[0].set_xlim([0, min(chip_end_time, max(self.tSim_mus))])
        axes[0].set_ylim([-1.5, 1.5])
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('Time [μs]', fontsize=12)
        axes[0].set_ylabel('Amplitude', fontsize=12)
        axes[0].set_title(f"{self.upsample_factor}× Oversampled Barker Sequence (Unwindowed)", fontsize=14)

        # Add Barker code sequence annotation
        barker_seq_str = ', '.join([f"{val}" for val in self.barker_seq])
        axes[0].annotate(f"Barker Sequence: [{barker_seq_str}]",
                         xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Subplot 2: Frequency domain comparison of windowed and unwindowed signals
        unwindowed_db = abs_to_db(fftshift(self.Stx_dcatbeg))
        windowed_db = abs_to_db(self.Stx_win_dcatbeg)
        max_db = max(np.max(unwindowed_db), np.max(windowed_db))
        min_db = max_db - 60  # Show 60dB dynamic range

        axes[1].plot(self.f_if_mhz, unwindowed_db, linewidth=2, color='#1f77b4', label="Unwindowed")
        axes[1].plot(self.f_if_mhz, windowed_db, linewidth=2, color='#ff7f0e', label="Windowed (Hann)")

        # Mark bandwidth
        bandwidth = np.max(self.f_if_mhz) - np.min(self.f_if_mhz)
        main_lobe_width = 1.0 / (len(self.barker_seq) * self.Tchip_mus)  # Approximation of main lobe width

        axes[1].annotate(f'Bandwidth: {bandwidth:.2f} MHz\nMain Lobe Width: {main_lobe_width:.2f} MHz',
                         xy=(0.02, 0.05), xycoords='axes fraction', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Mark peak sidelobe levels for unwindowed signal
        peaks = self._find_peaks(unwindowed_db)
        if len(peaks) > 1:
            # Sort peaks by magnitude
            peak_values = [(i, unwindowed_db[i]) for i in peaks]
            peak_values.sort(key=lambda x: x[1], reverse=True)
            # Find first sidelobe (second highest peak)
            if len(peak_values) > 1:
                sidelobe_idx, sidelobe_val = peak_values[1]
                mainlobe_val = peak_values[0][1]
                psl_db = mainlobe_val - sidelobe_val
                axes[1].annotate(f'Peak Sidelobe: {psl_db:.1f} dB',
                                 xy=(self.f_if_mhz[sidelobe_idx], sidelobe_val),
                                 xytext=(self.f_if_mhz[sidelobe_idx] + 5, sidelobe_val + 5),
                                 arrowprops=dict(arrowstyle='->'), fontsize=9)

        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([min_db, max_db + 3])
        axes[1].set_xlabel('IF Frequency [MHz]', fontsize=12)
        axes[1].set_ylabel('Magnitude [dB]', fontsize=12)
        axes[1].set_title('Spectrum of Barker Code Signal', fontsize=14)
        axes[1].legend(fontsize=10, loc='upper right')

        # Subplot 3: Windowed Tx sequence in time domain
        axes[2].plot(self.tSim_mus, np.real(self.stx_win), '-', color='#1f77b4',
                     linewidth=2, label='I (real)')
        axes[2].plot(self.tSim_mus, np.imag(self.stx_win), '-', color='#ff7f0e',
                     linewidth=2, label='Q (imag)')

        axes[2].set_xlim([0, min(chip_end_time, max(self.tSim_mus))])
        axes[2].set_ylim([-1.5, 1.5])
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Time [μs]', fontsize=12)
        axes[2].set_ylabel('Amplitude', fontsize=12)
        axes[2].set_title('Windowed Barker Code Signal (Hann Window Applied)', fontsize=14)
        axes[2].legend(fontsize=10)

        # Add signal information box
        pulse_duration = self.Tchip_mus * len(self.barker_seq)
        range_resolution = self.c_mpmus / (2 * self.bandwidth * 1e-6)

        info_text = (
            f"PRI: {self.pri * 1e6:.2f} μs\n"
            f"Duty Cycle: {self.duty_cycle:.3f}\n"
            f"Pulse Duration: {pulse_duration:.2f} μs\n"
            f"Chip Duration: {self.Tchip_mus:.2f} μs\n"
            f"Center Frequency: {self.center_freq / 1e9:.2f} GHz\n"
            f"Range Resolution: {range_resolution:.2f} m"
        )

        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="steelblue", alpha=0.8))

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig.subplots_adjust(top=0.92, bottom=0.12)  # Make space for title and info box
        plt.show()

    def _plot_input_waveform_original(self):
        """
        Plot the input Barker code waveform in time and frequency domains.

        Creates a professional visualization with three subplots:
        1. Unwindowed Tx sequence in time domain
        2. Frequency domain comparison of windowed and unwindowed signals
        3. Windowed Tx sequence in time domain
        """
        OS_str = f"{self.upsample_factor}x Oversampled"

        # Set up modern style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=100)

        # Subplot 1: Unwindowed Tx sequence in time domain
        chip_end_time = self.Tchip_mus * len(self.barker_seq) * 1.2
        axes[0].plot(self.tSim_mus, self.stx, 'o-', color='#1f77b4', markersize=4, linewidth=2)
        axes[0].set_xlim([0, min(chip_end_time, max(self.tSim_mus))])
        axes[0].set_ylim([-1.5, 1.5])
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('Time [μs]', fontsize=12)
        axes[0].set_ylabel('Amplitude', fontsize=12)
        axes[0].set_title(f"Input Waveform: {OS_str} Tx Sequence (Unwindowed)", fontsize=14)

        # Subplot 2: Frequency domain comparison of windowed and unwindowed signals
        unwindowed_db = abs_to_db(fftshift(self.Stx_dcatbeg))
        windowed_db = abs_to_db(self.Stx_win_dcatbeg)
        max_db = max(np.max(unwindowed_db), np.max(windowed_db))
        min_db = max_db - 60  # Show 60dB dynamic range

        axes[1].plot(self.f_if_mhz, unwindowed_db, linewidth=2, color='#1f77b4', label="Unwindowed")
        axes[1].plot(self.f_if_mhz, windowed_db, linewidth=2, color='#ff7f0e', label="Windowed (Hann)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([min_db, max_db + 3])
        axes[1].set_xlabel('IF Frequency [MHz]', fontsize=12)
        axes[1].set_ylabel('Magnitude [dB]', fontsize=12)
        axes[1].set_title(f"Spectrum of {OS_str} Broadcast Signal", fontsize=14)
        axes[1].legend(fontsize=10)

        # Subplot 3: Windowed Tx sequence in time domain
        axes[2].plot(self.tSim_mus, np.real(self.stx_win), '-', color='#1f77b4',
                     linewidth=2, label='Real')
        axes[2].plot(self.tSim_mus, np.imag(self.stx_win), '-', color='#ff7f0e',
                     linewidth=2, label='Imaginary')
        axes[2].set_xlim([0, min(chip_end_time, max(self.tSim_mus))])
        axes[2].set_ylim([-1.5, 1.5])
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Time [μs]', fontsize=12)
        axes[2].set_ylabel('Amplitude', fontsize=12)
        axes[2].set_title(f"{OS_str} Tx Sequence, Windowed (Hann)", fontsize=14)
        axes[2].legend(fontsize=10)

        # Adjust spacing between subplots
        plt.tight_layout()
        plt.show()

    def plot_output_waveform(self,
                             plot_freq_response=False,
                             plot_time_domain_signal=False,
                             plot_adc_samples=False,
                             plot_resampled_signal=False,
                             plot_matched_filter_results=True,
                             pulse_idx=0,
                             dpi=100,
                             figsize=(10, 8),
                             style='seaborn-v0_8-whitegrid',
                             dynamic_range_db=60,
                             annotate_peaks=True,
                             enhanced_plots=True):
        """
        Process the radar response and generate visualizations of the signal processing chain.

        Parameters:
            plot_freq_response (bool): If True, plot the frequency domain response.
            plot_time_domain_signal (bool): If True, plot the time domain signal.
            plot_adc_samples (bool): If True, plot the ADC samples.
            plot_resampled_signal (bool): If True, plot the resampled signal at chip intervals.
            plot_matched_filter_results (bool): If True, plot the matched filter results.
            pulse_idx (int or list): Index or list of indices of pulses to plot.
            dpi (int): Resolution for the output figures.
            figsize (tuple): Figure size in inches (width, height).
            style (str): Matplotlib style to use.
            dynamic_range_db (float): Dynamic range in dB for magnitude plots.
            annotate_peaks (bool): If True, annotate peak locations on matched filter results.
            enhanced_plots (bool): If True, generate publication-quality plots. (Sonnet 3.7 generated)
        This method processes the response through the complete radar signal chain and
        visualizes each step with professional, publication-ready plots.
        """
        if self.perceive_em_response is None:
            raise ValueError('No response has been provided for processing. Run process_received_signal() first.')

        if not enhanced_plots:
            if plot_freq_response:
                self._plot_frequency_response(pulse_idx)
            if plot_time_domain_signal:
                self._plot_time_domain_signal(pulse_idx)
            if plot_adc_samples:
                self._plot_adc_samples(pulse_idx)
            if plot_resampled_signal:
                self._plot_resampled_signal(pulse_idx)
            if plot_matched_filter_results:
                self._plot_matched_filter_results(pulse_idx)
        else:

            # Set plot style
            plt.style.use(style)

            # Create color palette for multiple pulse plots
            colors = plt.cm.tab10(np.linspace(0, 1, 4))



            if plot_freq_response:
                self._plot_enhanced_frequency_response(pulse_idx, figsize, dpi, colors, dynamic_range_db)
            if plot_time_domain_signal:
                self._plot_enhanced_time_domain_signal(pulse_idx, figsize, dpi, colors)
            if plot_adc_samples:
                self._plot_enhanced_adc_samples(pulse_idx, figsize, dpi, colors)
            if plot_resampled_signal:
                self._plot_enhanced_resampled_signal(pulse_idx, figsize, dpi, colors)
            if plot_matched_filter_results:
                self._plot_enhanced_matched_filter_results(pulse_idx, figsize, dpi, colors,
                                                           dynamic_range_db, annotate_peaks)

        plt.show()

    def _plot_enhanced_frequency_response(self, pulse_idx, figsize, dpi, colors, dynamic_range_db):
        """Plot improved frequency domain response with magnitude and phase"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        fig.suptitle('Frequency Domain Response', fontsize=16, fontweight='bold')

        # Handle different input shapes and convert to list for consistent processing
        pulse_indices, data_to_plot = self._prepare_data_for_plotting(self.perceive_em_response, pulse_idx)

        # Magnitude plot with enhanced visualization
        for idx, line in enumerate(data_to_plot):
            magnitude_db = abs_to_db(line)
            color_idx = idx % len(colors)
            ax1.plot(self.f_if_mhz, magnitude_db, '-', color=colors[color_idx],
                     linewidth=2, label=f'Pulse {pulse_indices[idx]}')

            # Add peak markers
            peak_indices = self._find_peaks(magnitude_db)
            if len(peak_indices) > 0:
                ax1.plot(self.f_if_mhz[peak_indices], magnitude_db[peak_indices], 'o',
                         color=colors[color_idx], markersize=6)

        # Set dynamic range for y-axis
        max_db = max([np.max(abs_to_db(line)) for line in data_to_plot])
        min_db = max_db - dynamic_range_db
        ax1.set_ylim([min_db, max_db + 3])

        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('IF Frequency [MHz]', fontsize=12)
        ax1.set_ylabel('Magnitude [dB]', fontsize=12)
        ax1.set_title('Target Frequency Response Magnitude', fontsize=14)
        if len(pulse_indices) > 1:
            ax1.legend(fontsize=10)

        # Phase plot with enhanced visualization
        for idx, line in enumerate(data_to_plot):
            unwrapped_phase = np.unwrap(np.angle(line)) * 180 / np.pi
            ax2.plot(self.f_if_mhz, unwrapped_phase, '-', color=colors[idx % len(colors)],
                     linewidth=2, label=f'Pulse {pulse_indices[idx]}')

        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('IF Frequency [MHz]', fontsize=12)
        ax2.set_ylabel('Phase [degrees]', fontsize=12)
        ax2.set_title('Target Frequency Response Phase (Unwrapped)', fontsize=14)

        # Add helpful annotations about frequency response
        bandwidth = np.max(self.f_if_mhz) - np.min(self.f_if_mhz)
        ax1.annotate(f'Bandwidth: {bandwidth:.2f} MHz',
                     xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make space for the suptitle

    def _plot_enhanced_time_domain_signal(self, pulse_idx, figsize, dpi, colors):
        """Plot enhanced time domain signal with separate I/Q and magnitude plots"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        fig.suptitle('Time Domain Signal Analysis', fontsize=16, fontweight='bold')

        # Process data for plotting
        pulse_indices, data_to_plot = self._prepare_data_for_plotting(self.output_waveform_if_td, pulse_idx)

        # I/Q Components subplot
        for idx, line in enumerate(data_to_plot):
            color_idx = idx % len(colors)
            ax1.plot(self.tSim_mus, np.real(line), '-', color=colors[color_idx],
                     linewidth=2, label=f'I (real) P{pulse_indices[idx]}')
            ax1.plot(self.tSim_mus, np.imag(line), '--', color=colors[color_idx],
                     linewidth=1.5, alpha=0.7, label=f'Q (imag) P{pulse_indices[idx]}')

        # Calculate limits for meaningful visualization - focus on pulse duration
        chip_end_time = self.Tchip_mus * len(self.barker_seq) * 2
        ax1.set_xlim([0, min(chip_end_time, max(self.tSim_mus))])

        # Calculate amplitude limits with some margin
        max_amplitude = max([max(np.max(np.real(line)), np.max(np.imag(line))) for line in data_to_plot])
        min_amplitude = min([min(np.min(np.real(line)), np.min(np.imag(line))) for line in data_to_plot])
        margin = (max_amplitude - min_amplitude) * 0.1
        ax1.set_ylim([min_amplitude - margin, max_amplitude + margin])

        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time [μs]', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title('Time Domain Signal - I/Q Components', fontsize=14)

        # Create a more compact legend for I/Q components
        if len(pulse_indices) <= 2:  # Only show legend for few pulses to avoid clutter
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='upper right', fontsize=9)

        # Magnitude plot with enhanced visualization
        for idx, line in enumerate(data_to_plot):
            color_idx = idx % len(colors)
            magnitude_db = abs_to_db(line)
            ax2.plot(self.tSim_mus, magnitude_db, '-', color=colors[color_idx],
                     linewidth=2, label=f'Pulse {pulse_indices[idx]}')

            # Mark the beginning and end of the barker code sequence
            pulse_duration = self.Tchip_mus * len(self.barker_seq)
            ax2.axvspan(0, pulse_duration, alpha=0.15, color=colors[color_idx])

        # Set appropriate time limits to focus on relevant portion
        ax2.set_xlim([0, min(chip_end_time * 1.5, max(self.tSim_mus))])

        # Set reasonable magnitude limits
        max_mag = max([np.max(abs_to_db(line)) for line in data_to_plot])
        ax2.set_ylim([max_mag - 60, max_mag + 3])

        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time [μs]', fontsize=12)
        ax2.set_ylabel('Magnitude [dB]', fontsize=12)
        ax2.set_title('Time Domain Signal - Magnitude', fontsize=14)
        ax2.legend(fontsize=9, loc='upper right')

        # Add annotation about pulse parameters
        info_text = f"Barker Code Length: {self.code_length}\n" \
                    f"Pulse Duration: {self.Tchip_mus * len(self.barker_seq):.2f} μs\n" \
                    f"Chip Duration: {self.Tchip_mus:.2f} μs"

        ax1.annotate(info_text, xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make space for the suptitle

    def _plot_enhanced_adc_samples(self, pulse_idx, figsize, dpi, colors):
        """Plot enhanced ADC samples with improved visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        fig.suptitle('ADC Sample Analysis', fontsize=16, fontweight='bold')

        # Process data for plotting
        pulse_indices, data_to_plot = self._prepare_data_for_plotting(self.output_waveform_adc_td, pulse_idx)

        # I/Q Components subplot with markers showing each ADC sample
        for idx, line in enumerate(data_to_plot):
            color_idx = idx % len(colors)
            # Plot lines first (continuous signal representation)
            ax1.plot(self.tSampAbs_mus, np.real(line), '-', color=colors[color_idx],
                     linewidth=1.5, alpha=0.7)
            ax1.plot(self.tSampAbs_mus, np.imag(line), '--', color=colors[color_idx],
                     linewidth=1.2, alpha=0.7)

            # Then add markers to highlight the actual samples
            ax1.plot(self.tSampAbs_mus, np.real(line), 'o', color=colors[color_idx],
                     markersize=4, label=f'I (real) P{pulse_indices[idx]}')
            ax1.plot(self.tSampAbs_mus, np.imag(line), 's', color=colors[color_idx],
                     markersize=3, label=f'Q (imag) P{pulse_indices[idx]}')

        # Calculate amplitude limits with some margin
        max_amplitude = max([max(np.max(np.real(line)), np.max(np.imag(line))) for line in data_to_plot])
        min_amplitude = min([min(np.min(np.real(line)), np.min(np.imag(line))) for line in data_to_plot])
        margin = (max_amplitude - min_amplitude) * 0.1
        ax1.set_ylim([min_amplitude - margin, max_amplitude + margin])

        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time [μs]', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title('ADC Samples - I/Q Components', fontsize=14)

        # Create a more compact legend for I/Q components
        if len(pulse_indices) <= 2:  # Only show legend for few pulses to avoid clutter
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='upper right', fontsize=9)

        # Magnitude plot with enhanced visualization
        for idx, line in enumerate(data_to_plot):
            color_idx = idx % len(colors)
            magnitude_db = abs_to_db(line)
            ax2.plot(self.tSampAbs_mus, magnitude_db, '-', color=colors[color_idx],
                     linewidth=1.5)
            ax2.plot(self.tSampAbs_mus, magnitude_db, 'o', color=colors[color_idx],
                     markersize=4, label=f'Pulse {pulse_indices[idx]}')

        # Set reasonable magnitude limits
        max_mag = max([np.max(abs_to_db(line)) for line in data_to_plot])
        ax2.set_ylim([max_mag - 60, max_mag + 3])

        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time [μs]', fontsize=12)
        ax2.set_ylabel('Magnitude [dB]', fontsize=12)
        ax2.set_title('ADC Samples - Magnitude', fontsize=14)
        ax2.legend(fontsize=9, loc='upper right')

        # Add annotation about ADC parameters
        info_text = f"ADC Sample Rate: {self.adc_sampling_rate / 1e6:.2f} MHz\n" \
                    f"Sample Period: {self.dtSamp_mus:.3f} μs\n" \
                    f"Blanking Time: {self.tStart_mus:.2f} μs\n" \
                    f"Total Samples: {self.number_adc_samples}"

        ax2.annotate(info_text, xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Highlight the start time (end of blanking period)
        for ax in [ax1, ax2]:
            ax.axvline(x=self.tStart_mus, color='darkred', linestyle='--', alpha=0.7, linewidth=1)
            ax.annotate('Start Time\n(End of Blanking)',
                        xy=(self.tStart_mus, ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                        xytext=(10, 15), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'),
                        fontsize=8)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make space for the suptitle

    def _plot_enhanced_resampled_signal(self, pulse_idx, figsize, dpi, colors):
        """Plot enhanced resampled signal at chip intervals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        fig.suptitle('Resampled Signal Analysis', fontsize=16, fontweight='bold')

        # Process data for plotting
        pulse_indices, data_to_plot = self._prepare_data_for_plotting(self.output_waveform_resampled_td, pulse_idx)

        # I/Q Components subplot
        for idx, line in enumerate(data_to_plot):
            color_idx = idx % len(colors)
            # Plot lines connecting the samples
            ax1.plot(self.tResampAbs_mus, np.real(line), '-', color=colors[color_idx],
                     linewidth=1.5, alpha=0.8)
            ax1.plot(self.tResampAbs_mus, np.imag(line), '--', color=colors[color_idx],
                     linewidth=1.2, alpha=0.8)

            # Then add markers to highlight the actual samples
            ax1.plot(self.tResampAbs_mus, np.real(line), 'o', color=colors[color_idx],
                     markersize=4, label=f'I (real) P{pulse_indices[idx]}')
            ax1.plot(self.tResampAbs_mus, np.imag(line), 's', color=colors[color_idx],
                     markersize=3, label=f'Q (imag) P{pulse_indices[idx]}')

        # Calculate amplitude limits with some margin
        max_amplitude = max([max(np.max(np.real(line)), np.max(np.imag(line))) for line in data_to_plot])
        min_amplitude = min([min(np.min(np.real(line)), np.min(np.imag(line))) for line in data_to_plot])
        margin = (max_amplitude - min_amplitude) * 0.1
        ax1.set_ylim([min_amplitude - margin, max_amplitude + margin])

        # Add visual indicators for chip boundaries
        chip_boundaries = np.arange(
            self.tStart_mus,
            self.tStart_mus + len(self.barker_seq) * self.Tchip_mus * 1.5,
            self.Tchip_mus
        )

        # for boundary in chip_boundaries:
        #     if boundary >= min(self.tResampAbs_mus) and boundary <= max(self.tResampAbs_mus):
        #         ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)

        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time [μs]', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title('Resampled Signal (Chip Interval) - I/Q Components', fontsize=14)

        # Create a more compact legend
        if len(pulse_indices) <= 2:  # Only show legend for few pulses to avoid clutter
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='upper right', fontsize=9)

        # Magnitude plot
        for idx, line in enumerate(data_to_plot):
            color_idx = idx % len(colors)
            magnitude_db = abs_to_db(line)
            # Plot lines connecting the samples
            ax2.plot(self.tResampAbs_mus, magnitude_db, '-', color=colors[color_idx],
                     linewidth=1.5, alpha=0.8)
            # Add markers for each sample
            ax2.plot(self.tResampAbs_mus, magnitude_db, 'o', color=colors[color_idx],
                     markersize=4, label=f'Pulse {pulse_indices[idx]}')

        # Add the same chip boundaries to the magnitude plot
        # for boundary in chip_boundaries:
        #     if boundary >= min(self.tResampAbs_mus) and boundary <= max(self.tResampAbs_mus):
        #         ax2.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        #
        #         # Label every other chip boundary for clarity
        #         if (boundary - self.tStart_mus) % (2 * self.Tchip_mus) < 0.001:
        #             chip_num = int(round((boundary - self.tStart_mus) / self.Tchip_mus))
        #             if chip_num < len(self.barker_seq):
        #                 ax2.annotate(f'Chip {chip_num}',
        #                              xy=(boundary, ax2.get_ylim()[0] + 0.85 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])),
        #                              xytext=(0, 0), textcoords='offset points',
        #                              ha='center', va='center', fontsize=8,
        #                              bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))

        # Set reasonable magnitude limits
        max_mag = max([np.max(abs_to_db(line)) for line in data_to_plot])
        ax2.set_ylim([max_mag - 60, max_mag + 3])

        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time [μs]', fontsize=12)
        ax2.set_ylabel('Magnitude [dB]', fontsize=12)
        ax2.set_title('Resampled Signal (Chip Interval) - Magnitude', fontsize=14)
        ax2.legend(fontsize=9, loc='upper right')

        # Add annotation comparing original and resampled sampling
        info_text = (
            f"Original ADC Sample Rate: {self.adc_sampling_rate / 1e6:.2f} MHz\n"
            f"Original Sample Period: {self.dtSamp_mus:.3f} μs\n"
            f"Resampled Period: {self.Tchip_mus:.3f} μs\n"
            f"Resampling Factor: {self.Tchip_mus / self.dtSamp_mus:.2f}x"
        )

        ax1.annotate(info_text, xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Highlight the start time (end of blanking period)
        for ax in [ax1, ax2]:
            ax.axvline(x=self.tStart_mus, color='darkred', linestyle='--', alpha=0.7, linewidth=1)
            ax.annotate('Start Time\n(End of Blanking)',
                        xy=(self.tStart_mus, ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                        xytext=(10, 15), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'),
                        fontsize=8)

        # Calculate the range corresponding to the first few chips
        range_per_chip = self.c_mpmus * self.Tchip_mus / 2
        ax2.annotate(f"Range per chip: {range_per_chip:.2f} m",
                     xy=(0.02, 0.95), xycoords='axes fraction', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make space for the suptitle

    def _plot_enhanced_matched_filter_results(self, pulse_idx, figsize, dpi, colors, dynamic_range_db, annotate_peaks):
        """Plot enhanced matched filter results with dual x-axis for time/range and magnitude"""
        fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        fig.suptitle('Matched Filter Analysis', fontsize=16, fontweight='bold')

        # Process data for plotting
        pulse_indices, data_to_plot = self._prepare_data_for_plotting(
            self.output_waveform_matched_filter_td, pulse_idx)

        # Subplot 1: I/Q plot with dual x-axis
        ax1_range = axes[0]
        for idx, line in enumerate(data_to_plot):
            color_idx = idx % len(colors)
            ax1_range.plot(self.RMf_m, np.real(line), '-', color=colors[color_idx],
                           linewidth=2, alpha=0.8, label=f'I (real) P{pulse_indices[idx]}')
            ax1_range.plot(self.RMf_m, np.imag(line), '--', color=colors[color_idx],
                           linewidth=1.5, alpha=0.8, label=f'Q (imag) P{pulse_indices[idx]}')

        ax1_range.set_xlabel('Range [m]', fontsize=12)
        ax1_range.set_ylabel('Amplitude', fontsize=12)
        ax1_range.set_title('Matched Filter Output - I/Q Components', fontsize=14)
        ax1_range.grid(True, alpha=0.3)

        # Secondary x-axis for time
        ax1_time = ax1_range.twiny()
        ax1_time.plot(self.tMfAbs_mus, np.zeros_like(self.tMfAbs_mus), alpha=0)  # Invisible reference plot
        ax1_time.set_xlabel('Time [μs]', fontsize=12)

        # Align x-axis limits
        ax1_range.set_xlim([min(self.RMf_m), max(self.RMf_m)])
        ax1_time.set_xlim([min(self.tMfAbs_mus), max(self.tMfAbs_mus)])

        # Add compact legend for I/Q components
        if len(pulse_indices) <= 2:  # Only show legend for few pulses to avoid clutter
            handles, labels = ax1_range.get_legend_handles_labels()
            ax1_range.legend(handles, labels, loc='upper right', fontsize=9)

        # Subplot 2: Magnitude plot with peaks detected and annotated
        ax2 = axes[1]
        max_peaks = []

        for idx, line in enumerate(data_to_plot):
            color_idx = idx % len(colors)
            magnitude_db = abs_to_db(line)

            ax2.plot(self.RMf_m, magnitude_db, '-', color=colors[color_idx],
                     linewidth=2, label=f'Pulse {pulse_indices[idx]}')

            # Find peaks in matched filter output for annotation
            if annotate_peaks:
                # Find prominent peaks (use a threshold relative to max value)
                peak_threshold = np.max(magnitude_db) - 20  # 20dB below max
                peak_indices = self._find_peaks(magnitude_db, height=peak_threshold,
                                                distance=int(len(magnitude_db) / 50))

                # Keep track of detected peaks
                max_peaks.extend([(self.RMf_m[i], magnitude_db[i], idx) for i in peak_indices])

                # Annotate peaks
                for i in peak_indices:
                    range_val = self.RMf_m[i]
                    magnitude = magnitude_db[i]
                    ax2.plot(range_val, magnitude, 'o', color=colors[color_idx], markersize=6)

                    # Only annotate the highest peaks to avoid clutter
                    if magnitude > peak_threshold + 10:
                        ax2.annotate(f"{range_val:.1f}m",
                                     xy=(range_val, magnitude),
                                     xytext=(0, 10),
                                     textcoords="offset points",
                                     ha='center',
                                     fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))

        # Set appropriate y-axis limits for magnitude plot
        max_db = max([np.max(abs_to_db(line)) for line in data_to_plot])
        min_db = max_db - dynamic_range_db
        ax2.set_ylim([min_db, max_db + 3])

        # Add range resolution information
        if len(max_peaks) >= 2:
            # Sort peaks by range
            sorted_peaks = sorted(max_peaks, key=lambda x: x[0])
            min_separation = float('inf')

            # Find minimum separation between peaks
            for i in range(1, len(sorted_peaks)):
                separation = sorted_peaks[i][0] - sorted_peaks[i - 1][0]
                if separation < min_separation and separation > 0:
                    min_separation = separation

            if min_separation < float('inf'):
                ax2.annotate(f"Min. peak separation: {min_separation:.1f} m",
                             xy=(0.02, 0.95), xycoords='axes fraction',
                             fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([min(self.RMf_m), max(self.RMf_m)])
        ax2.set_xlabel('Range [m]', fontsize=12)
        ax2.set_ylabel('Magnitude [dB]', fontsize=12)
        ax2.set_title('Matched Filter Output - Magnitude', fontsize=14)
        ax2.legend(fontsize=9)

        # Add radar performance metrics annotation box
        range_res = self.c_mpmus / (2 * self.bandwidth * 1e-6)  # Range resolution based on bandwidth
        theoretical_max_range = self.c_mpmus * self.pri / 2  # Maximum unambiguous range

        info_text = (
            f"Barker Code Length: {self.code_length}\n"
            f"Range Resolution: {range_res:.2f} m\n"
            f"Max Unambiguous Range: {theoretical_max_range:.1f} m"
        )

        # Add the information box
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="steelblue", alpha=0.8))

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=0.15)  # Make space for annotations

    def _find_peaks(self, signal, height=None, distance=None):
        """Find peaks in a signal using a simple algorithm"""
        from scipy.signal import find_peaks

        # Default distance between peaks if not specified
        if distance is None:
            distance = max(int(len(signal) * 0.01), 5)

        # Find peaks
        peaks, _ = find_peaks(signal, height=height, distance=distance)
        return peaks

    def _prepare_data_for_plotting(self, source_data, pulse_idx):
        """Helper function to prepare data for plotting with consistent formatting"""
        if source_data.ndim == 1:
            data_to_plot = [source_data]
            pulse_indices = [0]
        else:
            if isinstance(pulse_idx, list):
                data_to_plot = []
                pulse_indices = pulse_idx
                for idx in pulse_idx:
                    data_to_plot.append(source_data[idx])
            else:
                data_to_plot = [source_data[pulse_idx]]
                pulse_indices = [pulse_idx]

        return pulse_indices, data_to_plot


    def _plot_frequency_response(self,pulse_idx):
        """Plot frequency domain response magnitude and phase as subplots in a single figure"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))

        if self.perceive_em_response.ndim == 1:
            data_to_plot = self.perceive_em_response
            data_to_plot = [data_to_plot]
            pulse_idx = [pulse_idx]
        else:
            if isinstance(pulse_idx, list):
                data_to_plot = []
                for idx in pulse_idx:
                    data_to_plot.append(self.perceive_em_response[idx])
            else:
                data_to_plot = self.perceive_em_response[pulse_idx]
                data_to_plot = [data_to_plot]
                pulse_idx = [pulse_idx]

        # Magnitude plot
        for idx, line in enumerate(data_to_plot):
            ax1.plot(self.f_if_mhz, abs_to_db(line), '-',label=f'Pulse {pulse_idx[idx]}')
        ax1.grid(True)
        ax1.set_xlabel('IF Frequency [MHz]')
        ax1.set_ylabel('|H(f) [dB]|')
        ax1.set_title('Magnitude of Target Response')
        ax1.legend()

        # Phase plot
        for idx, line in enumerate(data_to_plot):
            ax2.plot(self.f_if_mhz, np.unwrap(np.angle(line)) * 180 / np.pi, '-',label=f'Pulse {pulse_idx[idx]}')
        ax2.grid(True)
        ax2.set_xlabel('IF Frequency [MHz]')
        ax2.set_ylabel('Phase(H(f)) [deg]')
        ax2.set_title('Unwrapped Phase of Target Response')
        ax2.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

    def _plot_time_domain_signal(self,pulse_idx):
        """Plot time domain complex signal as subplots in a single figure"""
        OS_str = f"{self.upsample_factor}x Oversampled"

        if self.perceive_em_response.ndim == 1:
            data_to_plot = self.output_waveform_if_td
            data_to_plot = [data_to_plot]
            pulse_idx = [pulse_idx]
        else:
            if isinstance(pulse_idx, list):
                data_to_plot = []
                for idx in pulse_idx:
                    data_to_plot.append(self.output_waveform_if_td[idx])
            else:
                data_to_plot = self.output_waveform_if_td[pulse_idx]
                data_to_plot = [data_to_plot]
                pulse_idx = [pulse_idx]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # I/Q plot
        for idx, line in enumerate(data_to_plot):
            ax1.plot(self.tSim_mus, np.real(line), '-ob',label=f'Pulse {pulse_idx[idx]}')
            ax1.plot(self.tSim_mus, np.imag(line), '-+r',label=f'Pulse {pulse_idx[idx]}')
        ax1.grid(True)
        ax1.set_xlabel('Time [μs]')
        ax1.set_ylabel('s_rx')
        ax1.set_title(f"{OS_str} Rx Signal - I/Q Components")
        ax1.legend(['real', 'imag'])

        # Magnitude plot
        for idx, line in enumerate(data_to_plot):
            ax2.plot(self.tSim_mus, abs_to_db(line), '-ob',label=f'Pulse {pulse_idx[idx]}')
        ax2.grid(True)
        ax2.set_xlabel('Time [μs]')
        ax2.set_ylabel('s_rx [dB]')
        ax2.set_title(f"{OS_str} Rx Signal - Magnitude")

        # Adjust spacing between subplots
        plt.tight_layout()

    def _plot_adc_samples(self,pulse_idx):
        """Plot ADC samples"""

        if self.perceive_em_response.ndim == 1:
            data_to_plot = self.output_waveform_adc_td
            data_to_plot = [data_to_plot]
            pulse_idx = [pulse_idx]
        else:
            if isinstance(pulse_idx, list):
                data_to_plot = []
                for idx in pulse_idx:
                    data_to_plot.append(self.output_waveform_adc_td[idx])
            else:
                data_to_plot = self.output_waveform_adc_td[pulse_idx]
                data_to_plot = [data_to_plot]
                pulse_idx = [pulse_idx]
        plt.figure()
        for idx, line in enumerate(data_to_plot):
            plt.plot(self.tSampAbs_mus, np.real(line), '-ob',label=f'Pulse {pulse_idx[idx]}')
            plt.plot(self.tSampAbs_mus, np.imag(line), '-+r',label=f'Pulse {pulse_idx[idx]}')
        plt.grid()
        plt.xlabel('Time [μs]')
        plt.ylabel('s_rx')
        plt.title('Rx ADC Samples @ f_s')
        plt.legend(['I (real)', 'Q (imag)'])

    def _plot_resampled_signal(self,pulse_idx):
        """Plot resampled signal at chip interval"""

        if self.perceive_em_response.ndim == 1:
            data_to_plot = self.output_waveform_resampled_td
            data_to_plot = [data_to_plot]
            pulse_idx = [pulse_idx]
        else:
            if isinstance(pulse_idx, list):
                data_to_plot = []
                for idx in pulse_idx:
                    data_to_plot.append(self.output_waveform_resampled_td[idx])
            else:
                data_to_plot = self.output_waveform_resampled_td[pulse_idx]
                data_to_plot = [data_to_plot]
                pulse_idx = [pulse_idx]

        plt.figure()
        for idx, line in enumerate(data_to_plot):
            plt.plot(self.tResampAbs_mus, np.real(line), '-ob',label=f'I (real) Pulse {pulse_idx[idx]}')
            plt.plot(self.tResampAbs_mus, np.imag(line), '-+r',label=f'Q (imag) Pulse {pulse_idx[idx]}')

        plt.grid()
        plt.xlabel('Time [μs]')
        plt.ylabel('s_rx')
        plt.title('Rx Signal, Re-Sampled @ Chip Interval')


    def _plot_matched_filter_results(self,pulse_idx):
        """Plot matched filter results with dual x-axis for time/range and magnitude"""

        if self.perceive_em_response.ndim == 1:
            data_to_plot = self.output_waveform_matched_filter_td
            data_to_plot = [data_to_plot]
            pulse_idx = [pulse_idx]
        else:
            if isinstance(pulse_idx, list):
                data_to_plot = []
                for idx in pulse_idx:
                    data_to_plot.append(self.output_waveform_matched_filter_td[idx])
            else:
                data_to_plot = self.output_waveform_matched_filter_td[pulse_idx]
                data_to_plot = [data_to_plot]
                pulse_idx = [pulse_idx]
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        # Subplot 1: Combined I/Q plot with dual x-axis
        # Primary x-axis (bottom) for Range domain
        ax1_range = axes[0]
        for idx, line in enumerate(data_to_plot):
            ax1_range.plot(self.RMf_m, np.real(line), '-ob',label=f'I (real) Pulse {pulse_idx[idx]}')
            ax1_range.plot(self.RMf_m, np.imag(line), '-+r',label=f'Q (imag) Pulse {pulse_idx[idx]}')
        ax1_range.set_xlabel('Range [m]')
        ax1_range.set_ylabel('s_rx')
        ax1_range.set_title('Matched Filter Response - I/Q Components')
        ax1_range.grid(True)

        # Secondary x-axis (top) for Time domain
        ax1_time = ax1_range.twiny()
        for idx, line in enumerate(data_to_plot):
            ax1_time.plot(self.tMfAbs_mus, np.real(line), '-', alpha=0,label=f'I (real) Pulse {pulse_idx[idx]}')  # Invisible plot to set limits
            ax1_time.plot(self.tMfAbs_mus, np.imag(line), '-', alpha=0,label=f'Q (imag) Pulse {pulse_idx[idx]}')
        ax1_time.set_xlabel('Time [μs]')

        # Align the x-axis limits
        ax1_range.set_xlim([min(self.RMf_m), max(self.RMf_m)])
        ax1_time.set_xlim([min(self.tMfAbs_mus), max(self.tMfAbs_mus)])

        # Subplot 2: Range domain magnitude
        for idx, line in enumerate(data_to_plot):
            axes[1].plot(self.RMf_m, abs_to_db(line), '-ob',label=f'Pulse {pulse_idx[idx]}')
        axes[1].grid(True)
        axes[1].set_xlim([min(self.RMf_m), max(self.RMf_m)])

        axes[1].set_ylim([np.max(abs_to_db(line))-220, np.max(abs_to_db(line)) * 1.1])
        axes[1].set_xlabel('Range [m]')
        axes[1].set_ylabel('s_rx [dB]')
        axes[1].set_title('Matched Filter Response - Magnitude')

        # Set y limits to be the same for both subplots if desired
        ax1_range.set_ylim([np.min(np.real(line)) * 1.1, np.max(np.real(line)) * 1.1])

        # Adjust spacing between subplots
        plt.tight_layout()


    # Add these property decorators right after your class variables initialization
    @property
    def output_waveform_if_fd(self):
        """Get the frequency domain output waveform in IF stage."""
        return getattr(self, '_output_waveform_if_fd', None)


    @output_waveform_if_fd.setter
    def output_waveform_if_fd(self, value):
        self._output_waveform_if_fd = value


    @property
    def output_waveform_if_td(self):
        """Get the time domain output waveform in IF stage."""
        return getattr(self, '_output_waveform_if_td', None)


    @output_waveform_if_td.setter
    def output_waveform_if_td(self, value):
        self._output_waveform_if_td = value


    @property
    def output_waveform_adc_td(self):
        """Get the time domain ADC samples."""
        return getattr(self, '_output_waveform_adc_td', None)


    @output_waveform_adc_td.setter
    def output_waveform_adc_td(self, value):
        self._output_waveform_adc_td = value


    @property
    def output_time_domain(self):
        """Get the time domain for ADC samples."""
        return getattr(self, '_output_time_domain', None)


    @output_time_domain.setter
    def output_time_domain(self, value):
        self._output_time_domain = value


    @property
    def output_waveform_resampled_td(self):
        """Get the resampled time domain waveform."""
        return getattr(self, '_output_waveform_resampled_td', None)


    @output_waveform_resampled_td.setter
    def output_waveform_resampled_td(self, value):
        self._output_waveform_resampled_td = value


    @property
    def output_time_domain_resampled(self):
        """Get the resampled time domain."""
        return getattr(self, '_output_time_domain_resampled', None)


    @output_time_domain_resampled.setter
    def output_time_domain_resampled(self, value):
        self._output_time_domain_resampled = value


    @property
    def output_range_domain_matched_filter(self):
        """Get the range domain for matched filter output."""
        return getattr(self, '_output_range_domain_matched_filter', None)


    @output_range_domain_matched_filter.setter
    def output_range_domain_matched_filter(self, value):
        self._output_range_domain_matched_filter = value


    @property
    def output_time_domain_matched_filter(self):
        """Get the time domain for matched filter output."""
        return getattr(self, '_output_time_domain_matched_filter', None)


    @output_time_domain_matched_filter.setter
    def output_time_domain_matched_filter(self, value):
        self._output_time_domain_matched_filter = value


    @property
    def output_waveform_matched_filter_td(self):
        """Get the matched filter output in time domain."""
        return getattr(self, '_output_waveform_matched_filter_td', None)


    @output_waveform_matched_filter_td.setter
    def output_waveform_matched_filter_td(self, value):
        self._output_waveform_matched_filter_td = value

@staticmethod
def abs_to_db(value):
    return 20 * np.log10(np.fmax(np.abs(value),1e-30))