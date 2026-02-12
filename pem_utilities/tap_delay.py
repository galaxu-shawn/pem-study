import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal
from scipy import signal


class TapDelayLine:
    """
    A class for creating tap delay lines from 4D frequency domain data cubes.
    
    The data cube is organized as [tx][rx][pulse][freq] and this class provides
    methods to convert to time domain and filter based on power or tap count.
    
    Attributes:
        data_cube (np.ndarray): 4D array with shape [tx, rx, pulse, freq]
        frequencies (np.ndarray): Frequency samples corresponding to the last dimension
        tap_scale_factor (float): Scaling factor for tap values
        window_type (str): Type of window to apply ('hamming', 'hanning', 'blackman', 'none')
    """
    
    def __init__(
        self, 
        frequencies: np.ndarray,
        tap_scale_factor: float = 1.0,
        upsample_factor: int = 1,
        window_type: Literal['hamming', 'hann', 'blackman', 'none'] = 'hann'
    ):
        """
        Initialize the TapDelayLine processor.
        
        Args:
            frequencies: 1D array of frequency samples in Hz
            tap_scale_factor: Scale factor to apply to tap values (default: 1.0)
            window_type: Type of window for time domain windowing (default: 'hamming')
        """

        
        self.data_cube = None

        self.frequencies = frequencies
        self.tap_scale_factor = tap_scale_factor
        self.window_type = window_type
        self.upsample_factor = upsample_factor
        
        # Calculate time domain parameters
        self.freq_step = np.mean(np.diff(frequencies))
        self.time_step = 1.0 / (len(frequencies) * self.freq_step) # 1/BW, do I need to adjust for half frequency step sample
        self.time_samples = len(frequencies)
        self.times = np.arange(self.time_samples) * self.time_step

        if self.upsample_factor < 1:
            raise ValueError(f"upsample_factor must be >= 1, got {upsample_factor}")

        # Update time parameters for upsampled data
        if self.upsample_factor > 1:
            # Store upsampled time parameters
            self.time_step = self.time_step/ upsample_factor
            self.time_samples = len(self.frequencies) * upsample_factor
            self.times = np.arange(self.time_samples) * self.time_step


        self.figure = None
        self.axes = None

        self.accumulated_data_power = [] # Store output data for export, only populated if argument accumulate_data=True
        self.accumulated_data_power_dict = {}
        self.accumulated_data_count = [] # Store output data for export, only populated if argument accumulate_data=True
        self.accumulated_data_count_dict = {}
    
        self.accumulated_data_power_idx = 0
        self.accumulated_data_count_idx = 0

    def _intialize_data_cube(self, data_cube: np.ndarray):
        """
        Initialize the data cube after object creation.
        
        Args:
            data_cube: 4D array with shape [tx, rx, pulse, freq] containing frequency domain data
        """
        if data_cube.ndim > 4:
            raise ValueError(f"data_cube must be 4D or less, got shape {data_cube.shape}")
        elif data_cube.ndim < 4:
            # Expand dimensions to make it 4D
            data_cube = np.expand_dims(data_cube, axis=0)
            while data_cube.ndim < 4:
                data_cube = np.expand_dims(data_cube, axis=0)   
        
        if len(self.frequencies) != data_cube.shape[3]:
            raise ValueError(
                f"frequencies length {len(self.frequencies)} must match last dimension "
                f"of data_cube {data_cube.shape[3]}"
            )
        
        self.data_cube = data_cube

    def _get_window(self, length: int) -> np.ndarray:
        """
        Generate window function based on window_type.
        
        Args:
            length: Length of the window
            
        Returns:
            Window coefficients array
        """
        if self.window_type == 'hamming':
            win = np.hamming(length)
        elif self.window_type == 'hann':
            win = np.hanning(length)
        elif self.window_type == 'blackman':
            win = np.blackman(length)
        elif self.window_type == 'none':
            win = np.ones(length)
        else:
            raise ValueError(f"Unknown window type: {self.window_type}")
        
        win_sum = np.sum(win)
        win *= length/win_sum
        return win
        
    def convert_to_time_domain(
        self, 
        data_cube: np.ndarray = None,
        apply_window: bool = True,
    ) -> np.ndarray:
        """
        Convert frequency domain data to time domain using IFFT.
        
        Args:
            apply_window: Whether to apply windowing before IFFT (default: True)
            preserve_power: Whether to scale the result to preserve power after 
                          windowing and upsampling (default: True)
            upsample_factor: Factor by which to upsample the time domain (default: 1).
                           Must be >= 1. Upsampling is done via zero-padding in 
                           frequency domain, which increases time resolution.
            
        Returns:
            4D array with shape [tx, rx, pulse, time] containing time domain data
            where time dimension is original_freq_samples * upsample_factor
        """
        if data_cube is not None:
            self._intialize_data_cube(data_cube)
        elif self.data_cube is None:
            raise ValueError("data_cube must be provided")


        
        # Apply window in frequency domain if requested
        freq_data = self.data_cube.copy()
        window_scale = 1.0
        if apply_window and self.window_type != 'none':
            window = self._get_window(freq_data.shape[3])
            freq_data = freq_data * window[np.newaxis, np.newaxis, np.newaxis, :]
        
        # Apply upsampling via zero-padding in frequency domain
        if self.upsample_factor > 1:
            # Calculate new size
            new_freq_size = freq_data.shape[3] * self.upsample_factor
            
            sf_upsample =self.upsample_factor
            time_data = np.fft.ifft(freq_data, axis=3,n=new_freq_size)
        else:
            sf_upsample = 1.0
            time_data = np.fft.ifft(freq_data, axis=3)
       

        
        # Apply all scale factors: user scale * window scale * upsample scale
        total_scale = self.tap_scale_factor * sf_upsample 
        time_data = time_data * total_scale
        
        return time_data
    

    
    def filter_by_power_threshold(
        self,
        data_cube: np.ndarray, 
        power_percentage: float,
        channel_specific: bool = False,
        accumulate_data: bool = False,
        accumulation_idx: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter time domain data to keep only taps that contain specified power percentage.
        
        This method calculates the cumulative power in the time domain response and
        keeps only the taps needed to reach the specified power threshold.
        
        Args:
            power_percentage: Percentage of total power to retain (0-100)
            channel_specific: If False (default), averages power across tx/rx/pulse to create
                            a single tap mask for all channels. If True, creates individual
                            tap masks for each tx/rx/pulse combination.If true, the returned tap_mask
                            will have shape [tx, rx, pulse, time], otherwise it will have shape [time].
            
        Returns:
            Tuple of (filtered_time_data, tap_mask) where:
                - filtered_time_data: Time domain data with insignificant taps zeroed [tx, rx, pulse, time]
                - tap_mask: Boolean mask indicating which taps are kept
                           Shape [time] if channel_specific=False
                           Shape [tx, rx, pulse, time] if channel_specific=True
        """
        self.power_percentage = power_percentage
        self._intialize_data_cube(data_cube)

        if power_percentage < 0 or power_percentage > 100:
            raise ValueError("power_percentage must be between 0 and 100")
        
        # Get time domain data
        time_data = self.convert_to_time_domain()
        
        # Calculate power for each tap
        power = np.abs(time_data) ** 2
        
        if not channel_specific:
            # Original behavior: average across tx, rx, pulse
            avg_power = np.mean(power, axis=(0, 1, 2))  # Average over tx, rx, pulse
            
            # Sort taps by power (descending)
            sorted_indices = np.argsort(avg_power)[::-1]
            sorted_power = avg_power[sorted_indices]
            
            # Calculate cumulative power
            total_power = np.sum(sorted_power)
            cumulative_power = np.cumsum(sorted_power) / total_power * 100
            
            # Find number of taps needed to reach power threshold
            num_taps = np.searchsorted(cumulative_power, power_percentage) + 1
            num_taps = min(num_taps, len(sorted_indices))
            
            # Create mask for significant taps
            tap_mask = np.zeros(len(avg_power), dtype=bool)
            tap_mask[sorted_indices[:num_taps]] = True
            
            # Apply mask
            filtered_data = time_data.copy()
            filtered_data[:, :, :, ~tap_mask] = 0
        
        else:
            # Channel-specific behavior: separate mask for each tx/rx/pulse
            tx_size, rx_size, pulse_size, time_size = time_data.shape
            tap_mask = np.zeros((tx_size, rx_size, pulse_size, time_size), dtype=bool)
            filtered_data = time_data.copy()
            
            # Process each channel independently
            for tx in range(tx_size):
                for rx in range(rx_size):
                    for pulse in range(pulse_size):
                        channel_power = power[tx, rx, pulse, :]
                        
                        # Sort taps by power (descending)
                        sorted_indices = np.argsort(channel_power)[::-1]
                        sorted_power = channel_power[sorted_indices]
                        
                        # Calculate cumulative power
                        total_power = np.sum(sorted_power)
                        if total_power > 0:  # Avoid division by zero
                            cumulative_power = np.cumsum(sorted_power) / total_power * 100
                            
                            # Find number of taps needed to reach power threshold
                            num_taps = np.searchsorted(cumulative_power, power_percentage) + 1
                            num_taps = min(num_taps, len(sorted_indices))
                            
                            # Create mask for this channel's significant taps
                            tap_mask[tx, rx, pulse, sorted_indices[:num_taps]] = True
            
            # Apply channel-specific mask
            filtered_data[~tap_mask] = 0
        
        if accumulate_data:
            if accumulation_idx is None:
                self.accumulated_data_power_dict[self.accumulated_data_power_idx] = [filtered_data, tap_mask]
                self.accumulated_data_power_idx+=1
            else:
                self.accumulated_data_power_dict[accumulation_idx] = [filtered_data, tap_mask]
        return filtered_data, tap_mask
    
    def filter_by_tap_count(
        self,
        data_cube: np.ndarray,
        num_taps: int,
        method: Literal['uniform', 'power_based'] = 'power_based',
        channel_specific: bool = False,
        accumulate_data: bool = False,
        accumulation_idx: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter time domain data to keep only a specified number of taps.
        
        Args:
            num_taps: Number of taps to keep
            method: Selection method - 'uniform' for evenly spaced taps starting from 0,
                   'power_based' for taps with highest power (default: 'power_based')
            channel_specific: If False (default), averages power across tx/rx/pulse to create
                            a single tap mask for all channels. If True, creates individual
                            tap masks for each tx/rx/pulse combination. Only applies when
                            method='power_based'. If True, the returned tap_mask will have
                            shape [tx, rx, pulse, time], otherwise it will have shape [time].
            
        Returns:
            Tuple of (filtered_time_data, tap_mask) where:
                - filtered_time_data: Time domain data with only selected taps
                - tap_mask: Boolean mask indicating which taps are kept
                           Shape [time] if channel_specific=False
                           Shape [tx, rx, pulse, time] if channel_specific=True
        """
        self.num_taps = num_taps
        # Get time domain data
        self._intialize_data_cube(data_cube)
        # self.data_cube will be used in convert_to_time_domain
        time_data = self.convert_to_time_domain()

        if num_taps <= 0 or num_taps > time_data.shape[3]:
            raise ValueError(
                f"num_taps must be between 1 and {time_data.shape[3]}, got {num_taps}"
            )
        
        if method == 'uniform':
            # Uniform spacing doesn't support channel-specific mode
            if channel_specific:
                raise ValueError("channel_specific=True is only supported with method='power_based'")
            
            # Select uniformly spaced taps starting from time=0
            tap_mask = np.zeros(time_data.shape[3], dtype=bool)
            tap_indices = np.linspace(0, time_data.shape[3] - 1, num_taps, dtype=int)
            tap_mask[tap_indices] = True
            
            # Apply mask
            filtered_data = time_data.copy()
            filtered_data[:, :, :, ~tap_mask] = 0
        
        elif method == 'power_based':
            # Calculate power for each tap
            power = np.abs(time_data) ** 2
            
            if not channel_specific:
                # Original behavior: average across tx, rx, pulse
                avg_power = np.mean(power, axis=(0, 1, 2))
                sorted_indices = np.argsort(avg_power)[::-1]
                
                tap_mask = np.zeros(time_data.shape[3], dtype=bool)
                tap_mask[sorted_indices[:num_taps]] = True
                
                # Apply mask
                filtered_data = time_data.copy()
                filtered_data[:, :, :, ~tap_mask] = 0
            
            else:
                # Channel-specific behavior: separate mask for each tx/rx/pulse
                tx_size, rx_size, pulse_size, time_size = time_data.shape
                tap_mask = np.zeros((tx_size, rx_size, pulse_size, time_size), dtype=bool)
                filtered_data = time_data.copy()
                
                # Process each channel independently
                for tx in range(tx_size):
                    for rx in range(rx_size):
                        for pulse in range(pulse_size):
                            channel_power = power[tx, rx, pulse, :]
                            
                            # Sort taps by power (descending)
                            sorted_indices = np.argsort(channel_power)[::-1]
                            
                            # Select top num_taps
                            tap_mask[tx, rx, pulse, sorted_indices[:num_taps]] = True
                
                # Apply channel-specific mask
                filtered_data[~tap_mask] = 0
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if accumulate_data:
            if accumulation_idx is None:
                self.accumulated_data_count_dict[self.accumulated_data_count_idx] = [filtered_data, tap_mask]
                self.accumulated_data_count_idx+=1
            else:
                self.accumulated_data_count_dict[accumulation_idx] = [filtered_data, tap_mask]
        return filtered_data, tap_mask
    
    
    def get_tap_statistics(
        self, 
        tap_mask: np.ndarray,
        tx_idx: Optional[int] = None,
        rx_idx: Optional[int] = None,
        pulse_idx: Optional[int] = None
    ) -> dict:
        """
        Calculate statistics about the selected taps.
        
        Args:
            tap_mask: Boolean mask indicating which taps are selected.
                     Can be either:
                     - 1D array with shape [time] (averaged mask)
                     - 4D array with shape [tx, rx, pulse, time] (channel-specific mask)
            tx_idx: Transmit antenna index (required if tap_mask is 4D and you want 
                   channel-specific statistics, optional for overall statistics)
            rx_idx: Receive antenna index (required if tap_mask is 4D and you want
                   channel-specific statistics, optional for overall statistics)
            pulse_idx: Pulse index (required if tap_mask is 4D and you want
                      channel-specific statistics, optional for overall statistics)
            
        Returns:
            Dictionary with statistics including:
                - num_taps: Number of selected taps (average across channels if 4D mask without indices)
                - tap_indices: Indices of selected taps (only for 1D mask or 4D with indices)
                - tap_times: Time values of selected taps (only for 1D mask or 4D with indices)
                - power_retained: Percentage of power retained
                - total_taps: Total number of possible taps
                - channel_specific: Boolean indicating if statistics are for a specific channel
        """
        if self.data_cube is None:
            raise ValueError("Data cube is not initialized")
        time_data = self.convert_to_time_domain()
        
        # Calculate power statistics
        power = np.abs(time_data) ** 2
        total_power = np.sum(power)
        
        # Handle different tap_mask shapes
        if tap_mask.ndim == 1:
            # 1D mask: averaged across all channels 
            if tap_mask.shape[0] != time_data.shape[3]:
                raise ValueError(
                    f"tap_mask length {tap_mask.shape[0]} must match time dimension {time_data.shape[3]}"
                )
            
            retained_power = np.sum(power[:, :, :, tap_mask])
            power_percentage = (retained_power / total_power) * 100
            
            tap_indices = np.where(tap_mask)[0]
            tap_times = self.times[tap_indices]
            
            return {
                'num_taps': len(tap_indices),
                'tap_indices': tap_indices,
                'tap_times': tap_times,
                'power_retained': power_percentage,
                'total_taps': len(tap_mask),
                'channel_specific': False
            }
        
        elif tap_mask.ndim == 4:
            # 4D mask: channel-specific
            if tap_mask.shape != time_data.shape:
                raise ValueError(
                    f"tap_mask shape {tap_mask.shape} must match time_data shape {time_data.shape}"
                )
            
            # Check if user wants statistics for a specific channel
            if tx_idx is not None and rx_idx is not None and pulse_idx is not None:
                # Channel-specific statistics
                channel_mask = tap_mask[tx_idx, rx_idx, pulse_idx, :]
                channel_power = power[tx_idx, rx_idx, pulse_idx, :]
                
                channel_total_power = np.sum(channel_power)
                retained_power = np.sum(channel_power[channel_mask])
                power_percentage = (retained_power / channel_total_power) * 100 if channel_total_power > 0 else 0
                
                tap_indices = np.where(channel_mask)[0]
                tap_times = self.times[tap_indices]
                
                return {
                    'num_taps': len(tap_indices),
                    'tap_indices': tap_indices,
                    'tap_times': tap_times,
                    'power_retained': power_percentage,
                    'total_taps': time_data.shape[3],
                    'channel_specific': True,
                    'tx_idx': tx_idx,
                    'rx_idx': rx_idx,
                    'pulse_idx': pulse_idx
                }
            else:
                # Overall statistics across all channels with their respective masks
                print("Calculating average statistics across all channels with channel-specific masks.")
                retained_power = np.sum(power[tap_mask])
                power_percentage = (retained_power / total_power) * 100
                
                # Calculate average number of taps per channel
                num_taps_per_channel = np.sum(tap_mask, axis=3)  # Sum along time dimension
                avg_num_taps = np.mean(num_taps_per_channel)
                
                return {
                    'num_taps': avg_num_taps,
                    'num_taps_per_channel': num_taps_per_channel,
                    'power_retained': power_percentage,
                    'total_taps': time_data.shape[3],
                    'channel_specific': False,
                    'note': 'Statistics computed across all channels with channel-specific masks. '
                            'Provide tx_idx, rx_idx, and pulse_idx for specific channel statistics.'
                }
        
        else:
            raise ValueError(f"tap_mask must be 1D or 4D, got {tap_mask.ndim}D")
    
    def get_channel_impulse_response(self,filtered_data: np.ndarray,tx_idx: int,rx_idx: int,pulse_idx: int):
        """
        Retrieve the channel impulse response for given indices.
        
        Args:
            tx_idx: Transmit antenna index
            rx_idx: Receive antenna index
            pulse_idx: Pulse index
            filtered_data: Optional filtered time domain data. If None, uses unfiltered.
        Returns:
            times: Time values array
            cir: Channel impulse response array
        """

        cir = filtered_data[tx_idx, rx_idx, pulse_idx, :]
        
        return self.times, cir

    def plot_channel_impulse_response(
        self,
        filtered_data: np.ndarray,
        tx_idx: int,
        rx_idx: int,
        pulse_idx: int,
        show_magnitude: bool = True,
        show_phase: bool = False,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot the channel impulse response for visualization.
        
        Args:
            tx_idx: Transmit antenna index
            rx_idx: Receive antenna index
            pulse_idx: Pulse index
            filtered_data: Optional filtered time domain data
            show_magnitude: Whether to plot magnitude (default: True)
            show_phase: Whether to plot phase (default: False)
            figsize: Figure size tuple (default: (12, 6))
            
        Returns:
            Matplotlib figure object
        """
        times, cir = self.get_channel_impulse_response(filtered_data,tx_idx, rx_idx, pulse_idx)
        
        num_plots = sum([show_magnitude, show_phase])
        if num_plots == 0:
            raise ValueError("At least one of show_magnitude or show_phase must be True")
        
        # plt.ion()
        self.figure, self.axes = plt.subplots(num_plots, 1, figsize=figsize)
        if num_plots == 1:
            self.axes = [self.axes]

        plot_idx = 0
        
        if show_magnitude:
            line = self.axes[plot_idx].stem(times * 1e9, np.abs(cir), linefmt ='-',markerfmt='ro', basefmt='x')
            self.axes[plot_idx].set_xlabel('Time (ns)')
            self.axes[plot_idx].set_ylabel('Magnitude')
            self.axes[plot_idx].set_title(
                f'Channel Impulse Response Magnitude (TX={tx_idx}, RX={rx_idx}, Pulse={pulse_idx})'
            )
            self.axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        if show_phase:
            self.axes[plot_idx].plot(times * 1e9, np.angle(cir, deg=True))
            self.axes[plot_idx].set_xlabel('Time (ns)')
            self.axes[plot_idx].set_ylabel('Phase (degrees)')
            self.axes[plot_idx].set_title(
                f'Channel Impulse Response Phase (TX={tx_idx}, RX={rx_idx}, Pulse={pulse_idx})'
            )
            self.axes[plot_idx].grid(True, alpha=0.3)
        
        # plt.tight_layout()
        return self.figure
    
    def export_to_npy(
        self,
        filepath: str,
        filtered_data: Optional[np.ndarray] = None
    ):
        """
        Export tap delay line data to .npy file.
        
        Args:
            filepath: Output file path
            filtered_data: Optional filtered time domain data. If None, uses unfiltered.
        """
        if filtered_data is None:
            filtered_data = self.convert_to_time_domain()
        
        self.export_to_csv(filepath, filtered_data, format='npy')

    def export_to_csv(
        self,
        filepath: str,
        filtered_data: Optional[np.ndarray] = None,
        format: Literal['csv'] = 'csv'
    ):
        """
        Export tap delay line data to file.
        
        Args:
            filepath: Output file path
            filtered_data: Optional filtered time domain data. If None, uses unfiltered.
            format: Output format - 'csv' or 'npy' (default: 'csv')
        """
        if filtered_data is None:
            filtered_data = self.convert_to_time_domain()

        
        if format == 'npy':
            np.save(filepath, filtered_data)
        elif format == 'csv':
            # For CSV, flatten the data structure and save
            # Format: tx, rx, pulse, time_idx, real, imag, magnitude, phase
            tx_size, rx_size, pulse_size, time_size = filtered_data.shape
            
            with open(filepath, 'w') as f:
                f.write('tx,rx,pulse,time_idx,time_ns,real,imag,magnitude,phase_deg\n')
                
                for tx in range(tx_size):
                    for rx in range(rx_size):
                        for pulse in range(pulse_size):
                            for t_idx in range(time_size):
                                val = filtered_data[tx, rx, rx, t_idx]
                                if np.abs(val) > 1e-20:  # Only write non-zero values
                                    f.write(
                                        f'{tx},{rx},{pulse},{t_idx},'
                                        f'{self.times[t_idx]*1e9:.6f},'
                                        f'{val.real:.6e},{val.imag:.6e},'
                                        f'{np.abs(val):.6e},{np.angle(val, deg=True):.6f}\n'
                                    )
        else:
            raise ValueError(f"Unknown format: {format}")

    def _write_scenario_time_block(self, f, pulse_times: float, data):
        
        filtered_data = data[0][0,0]
        if data[1].ndim ==1:
            tap_mask = data[1]
        else:
            tap_mask = data[1][0,0]

        pulse_size,time_size = filtered_data.shape
        
        for pulse in range(pulse_size):
            tap_index=0
            for t_idx in range(time_size):
                if tap_mask.ndim == 1:
                    is_tap_kept = tap_mask[t_idx]
                else:
                    is_tap_kept = tap_mask[pulse, t_idx]
                
                if is_tap_kept:
                    val = filtered_data[pulse, t_idx]
                    f.write(
                        f'{pulse_times[pulse]},'
                        f'{tap_index+1},'
                        f'{self.times[t_idx]*1e9:.6f},'
                        f'{val.real:.6e},{val.imag:.6e}\n'
                    )
                    tap_index+=1

    def export_to_tdd(self, filepath: str, 
                      scenario_times: np.ndarray,
                      pulse_times: np.ndarray,
                      filter_type='power_threshold',
                      use_accumulated_with_time_idx=False,
                      summary_as_json: bool = False):
        
        # Channel File: BT-BT-BT1_to_UE-UE-UE1_Tx_0_Rx_0
        #   Tap Delay quantization: 4e-09(sec)
        #   Channel Sounding increment: 0.004(sec)
        # Fields: Scenario_Time(s), Tap#, Delay(nsec), Tap_Real, Tap_Imag
        #-------------------------------------------------------------------

        if scenario_times is None:
            raise ValueError("scenario_time must be provided for TDD export.")

        has_data_count_been_accumulated = False
        has_data_power_been_accumulated = False
        if len(self.accumulated_data_count_dict) > 0:
            has_data_count_been_accumulated = True
            if len(self.accumulated_data_count_dict) != len(scenario_times):
                raise ValueError("Length of scenario_time must match number of accumulated data sets.")
        if len(self.accumulated_data_power_dict) > 0:
            has_data_power_been_accumulated = True
            if len(self.accumulated_data_power_dict) != len(scenario_times):
                raise ValueError("Length of scenario_time must match number of accumulated data sets.")
        if not has_data_count_been_accumulated and not has_data_power_been_accumulated:
            raise ValueError("No accumulated data to export. Use accumulate_data=True in filtering methods.")
        
        if filter_type =='power_threshold' and not has_data_power_been_accumulated:
            raise ValueError("No accumulated data for power threshold filtering to export.")
        elif not has_data_count_been_accumulated:
            raise ValueError("No accumulated data for tap count filtering to export.")

        if filter_type =='power_threshold':
            data_to_write = self.accumulated_data_power_dict
        else:
            data_to_write = self.accumulated_data_count_dict
        
        # create a sub folder for tdd files

        # make a sub folder named 'TapDelayExports' if not already present
        export_folder = os.path.join(os.path.dirname(filepath), 'TapDelayExports')
        os.makedirs(export_folder, exist_ok=True)
        # get filename from original filepath, change its folder to export_folder
        filename = os.path.basename(filepath)
        filepath = os.path.join(export_folder, filename)

        # convert dict to list ordered by keys
        data_to_write = [data_to_write[idx] for idx in range(len(data_to_write))]

        tx_size, rx_size, pulse_size, time_size = data_to_write[0][0].shape

        all_exported_files = {}

        if data_to_write[0][1].ndim ==1:
            # single tap mask for all channels
            # create one file for all channels
            with open(filepath, 'w') as f:
                # get time step from scenario_times
                dt = scenario_times[1] - scenario_times[0]
                stop_time = scenario_times[-1]
                num_steps = len(scenario_times)
                pulse_time_step = pulse_times[1] - pulse_times[0]
                f.write(f'# Channel File: Taps Selected by Channel Averages\n')
                f.write(f'#   Tap Delay quantization: {self.time_step}(sec)\n')
                f.write(f'#   Filter Type {filter_type}\n')
                if filter_type =='fixed_count':
                    f.write(f'#   Number of Taps Selected per Channel: {self.num_taps}\n')
                elif filter_type =='power_threshold':
                    f.write(f'#   Power Percentage Threshold: {self.power_percentage}%\n')
                f.write(f'#   Channel Sounding increment: {pulse_time_step}(sec)\n')
                f.write('# Fields: Scenario_Time(s), Tap#, Delay(nsec), Tap_Real, Tap_Imag\n')
                f.write('# -------------------------------------------------------------------\n')   
                for idx, scenario_time in enumerate(scenario_times):
                    absolute_pulse_times = scenario_time+pulse_times
                    self._write_scenario_time_block(f, absolute_pulse_times,data=data_to_write[idx])
                # get filename only for summary
                filename = os.path.basename(filepath)
                all_exported_files['All Channels - Averaged'] = filename
        else:
            # create a new file for each channel combination
            for tx in range(tx_size):
                for rx in range(rx_size):
                        channel_filepath = filepath.replace('.tdd', f'_Tx{tx}_Rx{rx}.tdd')
                        with open(channel_filepath, 'w') as f:
                            # get time step from scenario_times
                            dt = scenario_times[1] - scenario_times[0]
                            stop_time = scenario_times[-1]
                            num_steps = len(scenario_times)
                            pulse_time_step = pulse_times[1] - pulse_times[0]
                            f.write(f'# Channel File: Tx {tx},Rx {rx}\n')
                            f.write(f'#   Tap Delay quantization: {self.time_step}(sec)\n')
                            f.write(f'#   Filter Type {filter_type}\n')
                            if filter_type =='fixed_count':
                                f.write(f'#   Number of Taps Selected per Channel: {self.num_taps}\n')
                            elif filter_type =='power_threshold':
                                f.write(f'#   Power Percentage Threshold: {self.power_percentage}%\n')
                            f.write(f'#   Channel Sounding increment: {pulse_time_step}(sec)\n')
                            f.write('# Fields: Scenario_Time(s), Tap#, Delay(nsec), Tap_Real, Tap_Imag\n')
                            f.write('# -------------------------------------------------------------------\n')   

                            for idx, scenario_time in enumerate(scenario_times):
                                absolute_pulse_times = scenario_time+pulse_times
                                # extract only the current channel data
                                channel_data = [
                                    data_to_write[idx][0][tx:tx+1, rx:rx+1],
                                    data_to_write[idx][1][tx:tx+1, rx:rx+1]
                                ]
                                self._write_scenario_time_block(f, absolute_pulse_times,data=channel_data)
                        filename = os.path.basename(channel_filepath)
                        all_exported_files[f'Tx{tx}_Rx{rx}'] = filename

        if summary_as_json:
            import json
            # write a summary file that describes all channels and files
            summary_filepath = filepath.replace('.tdd', '_summary.json')
            summary_dict = {
                'Total Scenario Time (sec)': stop_time,
                'Number of Scenario Steps': num_steps,
                'Scenario Time Steps (sec)': dt,
                'Pulse Time Step (sec)': pulse_time_step,
                'Filter Type': filter_type,
                'Tap Delay Quantization (sec)': self.time_step,
                'Number of Entities': 1, # export is always between 1 tx device and 1 rx device. Multiple antennas are handled within that.
                'Number of Transmit Antennas': tx_size,
                'Number of Receive Antennas': rx_size,
                'Number of Pulses': pulse_size,
            }
            if filter_type =='fixed_count':
                summary_dict['Number of Taps Selected per Channel'] = self.num_taps
            elif filter_type =='power_threshold':
                summary_dict['Power Percentage Threshold (%)'] = self.power_percentage
            summary_dict['Exported Files'] = all_exported_files

            with open(summary_filepath, 'w') as f:
                json.dump(summary_dict, f, indent=4)
            
        else:
            # write a summary file that describes all channels and files
            summary_filepath = filepath.replace('.tdd', '_summary.txt')
            with open(summary_filepath, 'w') as f:
                f.write('Tap Delay Line Export Summary\n')
                f.write('=============================\n\n')
                f.write(f'Total Scenario Time: {stop_time} sec\n')
                f.write(f'Number of Scenario Steps: {num_steps}\n')
                f.write(f'Scenario Time Steps: {dt} sec\n')
                f.write(f'Pulse Time Step: {pulse_time_step} sec\n')
                f.write(f'Filter Type: {filter_type}\n')
                f.write(f'Tap Delay Quantization: {self.time_step} sec\n')
                f.write(f'Number of Entities: 1\n') # export is always between 1 tx device and 1 rx device. Multiple antennas are handled within that.
                f.write(f'Number of Transmit Antennas: {tx_size}\n')
                f.write(f'Number of Receive Antennas: {rx_size}\n')
                f.write(f'Number of Pulses: {pulse_size}\n')
                if filter_type =='fixed_count':
                    f.write(f'Number of Taps Selected per Channel: {self.num_taps}\n\n')
                elif filter_type =='power_threshold':
                    f.write(f'Power Percentage Threshold: {self.power_percentage}%\n\n')
                f.write('Exported Files:\n')
                if data_to_write[0][1].ndim ==1:
                    # write out relative path only
                    f.write(f'- {os.path.basename(filepath)}: All channels combined\n')
                else:
                    for tx in range(tx_size):
                        for rx in range(rx_size):
                            channel_filepath = filepath.replace('.tdd', f'_Tx{tx}_Rx{rx}.tdd')
                            # write out relative path only
                            f.write(f'- {os.path.basename(channel_filepath)}: Tx {tx}, Rx {rx}\n')
                                    
        print(f'Wrote summary file: {summary_filepath}')
        return summary_filepath




