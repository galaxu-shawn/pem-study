# -*- coding: utf-8 -*-
# try importing cupy if available
# try:
#     import cupy as np
#     print("Using CuPy for accelerated computations.")
#     # If cupy is available, we can use it for accelerated computations
#     # set environment variable CUDA_PATH to the path of the CUDA toolkit
#     import os
#     if 'CUDA_PATH' not in os.environ:
#         os.environ['CUDA_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9'
#     CUPY_AVAILABLE = True
# except ImportError:
#     print("CuPy not available, falling back to NumPy.")
#     print(" For GPU accelartion of CFAR, see: https://docs.cupy.dev/en/stable/install.html")
#     # If cupy is not available, we use numpy
CUPY_AVAILABLE = False
import numpy as np
from scipy import ndimage
from typing import Tuple, Union, Literal
import warnings

class CFAR:
    """
    Constant False Alarm Rate (CFAR) detector for radar signal processing.
    
    This class implements various CFAR algorithms for target detection in radar data
    while maintaining a constant false alarm rate despite varying noise and clutter levels.
    
    Works with both raw radar data and normalized data (0-1 range).
    
    Supported CFAR algorithms:
    - CA-CFAR: Cell Averaging CFAR (default, good general purpose)
    - GO-CFAR: Greatest Of CFAR (better for multiple targets)
    - SO-CFAR: Smallest Of CFAR (better for clutter edges)
    - OS-CFAR: Ordered Statistic CFAR (robust to interfering targets)
    
    References:
    - Rohling, H. (1983). Radar CFAR thresholding in clutter and multiple target situations.
    - Richards, M. A. (2005). Fundamentals of radar signal processing.
    """
    
    def __init__(self, 
                 training_cells: int = 10,
                 guard_cells: int = 4,
                 threshold_factor: float = 3.0,
                 cfar_type: Literal['CA', 'GO', 'SO', 'OS'] = 'CA',
                 os_rank: int = None,
                 axis: int = 1,
                 edge_behavior: Literal['constant', 'nearest', 'wrap'] = 'constant',
                 normalized_data: bool = False):
        """
        Initialize CFAR detector.
        
        Parameters:
        -----------
        training_cells : int, default=10
            Number of training cells on each side of the guard cells.
            Total training window = 2 * training_cells
            
        guard_cells : int, default=4
            Number of guard cells on each side of the cell under test (CUT).
            Guards prevent target energy from leaking into noise estimation.
            
        threshold_factor : float, default=3.0
            Scaling factor applied to noise estimate for detection threshold.
            Higher values reduce false alarms but may miss weak targets.
            Typical range: 1.5-10.0 for raw data, may need adjustment for normalized data.
            
        cfar_type : str, default='CA'
            CFAR algorithm type:
            - 'CA': Cell Averaging - simple mean of training cells
            - 'GO': Greatest Of - max of left/right window averages
            - 'SO': Smallest Of - min of left/right window averages  
            - 'OS': Ordered Statistic - uses rank-order statistic
            
        os_rank : int, optional
            Rank for OS-CFAR (1-indexed). If None, defaults to 3/4 of training cells.
            Only used when cfar_type='OS'.
            
        axis : int, default=1
            Axis along which to apply CFAR (0=pulse, 1=range).
            For typical [pulse][range] data, use axis=1 for range CFAR.
            
        edge_behavior : str, default='constant'
            How to handle edges where full training window unavailable:
            - 'constant': pad with zeros
            - 'nearest': pad with edge values
            - 'wrap': wrap around to other side
            
        normalized_data : bool, default=False
            Set to True if input data is normalized to [0,1] range.
            This affects validation and PFA calculation warnings.
        """
        self.training_cells = training_cells
        self.guard_cells = guard_cells
        self.threshold_factor = threshold_factor
        self.cfar_type = cfar_type.upper()
        self.axis = axis
        self.edge_behavior = edge_behavior
        self.normalized_data = normalized_data
        
        # Validate inputs
        if self.cfar_type not in ['CA', 'GO', 'SO', 'OS']:
            raise ValueError("cfar_type must be one of: 'CA', 'GO', 'SO', 'OS'")
            
        if training_cells < 1:
            raise ValueError("training_cells must be >= 1")
            
        if guard_cells < 0:
            raise ValueError("guard_cells must be >= 0")
            
        if threshold_factor <= 0:
            raise ValueError("threshold_factor must be > 0")
        
        # Set OS-CFAR rank
        if self.cfar_type == 'OS':
            if os_rank is None:
                # Default to 3/4 quantile for robustness
                self.os_rank = max(1, int(0.75 * 2 * training_cells))
            else:
                self.os_rank = os_rank
                if not (1 <= self.os_rank <= 2 * training_cells):
                    raise ValueError(f"os_rank must be between 1 and {2 * training_cells}")
        else:
            self.os_rank = None
            
        # Calculate total window parameters
        self.half_window = training_cells + guard_cells
        self.total_window = 2 * self.half_window + 1
        
        # Issue warning about PFA calculation for normalized data
        if self.normalized_data:
            warnings.warn(
                "Theoretical PFA calculations may not be accurate for normalized data. "
                "Consider empirical PFA estimation or threshold tuning.",
                UserWarning
            )
        
    def detect(self, data: np.ndarray, return_threshold: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply CFAR detection to input data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input radar data, typically [pulse, range] or [range, pulse].
            Should be magnitude or power data (non-negative).
            Can be raw data or normalized to [0,1] range.
            
        return_threshold : bool, default=False
            If True, also return the computed threshold array.
            
        Returns:
        --------
        detections : np.ndarray (bool)
            Boolean array indicating detections (True = target detected).
            Same shape as input data.
            
        threshold : np.ndarray, optional
            Threshold values used for detection. Only returned if return_threshold=True.
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        if data.ndim != 2:
            raise ValueError("Input data must be 2D array")
            
        if np.any(data < 0):
            raise ValueError("Input data should be non-negative (magnitude or power)")
            
        # Check if data appears to be normalized
        data_min, data_max = np.min(data), np.max(data)
        if not self.normalized_data and 0 <= data_min and data_max <= 1:
            warnings.warn(
                "Input data appears to be normalized [0,1]. Consider setting normalized_data=True "
                "for better handling of edge cases and PFA calculations.",
                UserWarning
            )
        elif self.normalized_data and (data_min < 0 or data_max > 1):
            warnings.warn(
                "normalized_data=True but data range is not [0,1]. "
                f"Actual range: [{data_min:.3f}, {data_max:.3f}]",
                UserWarning
            )
        
        # Compute noise estimates
        noise_estimate = self._compute_noise_estimate(data)
        
        # Compute threshold
        threshold = self.threshold_factor * noise_estimate
        
        # Apply detection
        detections = data > threshold
        
        if return_threshold:
            return detections, threshold
        else:
            return detections
    
    def _compute_noise_estimate(self, data: np.ndarray) -> np.ndarray:
        """
        Compute noise estimate using selected CFAR algorithm.
        
        Parameters:
        -----------
        data : np.ndarray
            Input radar data [pulse, range] or [range, pulse].
            
        Returns:
        --------
        noise_estimate : np.ndarray
            Estimated noise level for each cell, same shape as input.
        """
        if self.cfar_type == 'CA':
            return self._ca_cfar(data)
        elif self.cfar_type == 'GO':
            return self._go_cfar(data)
        elif self.cfar_type == 'SO':
            return self._so_cfar(data)
        elif self.cfar_type == 'OS':
            return self._os_cfar(data)
    
    def _ca_cfar(self, data: np.ndarray) -> np.ndarray:
        """Cell Averaging CFAR - averages all training cells."""
        # Create training window mask (excludes CUT and guard cells)
        mask = self._create_training_mask()
        
        # Use scipy's uniform filter for efficient sliding window average
        # Need to handle the mask properly
        kernel_size = self.total_window
        
        if self.axis == 1:  # Range CFAR
            kernel = np.ones((1, kernel_size)) / np.sum(mask)
            kernel[0, self.half_window] = 0  # Zero out CUT
            # Zero out guard cells
            guard_start = self.half_window - self.guard_cells
            guard_end = self.half_window + self.guard_cells + 1
            kernel[0, guard_start:guard_end] = 0
            kernel[0, :] = kernel[0, :] * (2 * self.training_cells) / np.sum(kernel[0, :] != 0)
            
            noise_estimate = ndimage.convolve(data, kernel, mode=self.edge_behavior)
            
        else:  # Pulse CFAR (axis=0)
            kernel = np.ones((kernel_size, 1)) / np.sum(mask)
            kernel[self.half_window, 0] = 0  # Zero out CUT
            # Zero out guard cells
            guard_start = self.half_window - self.guard_cells
            guard_end = self.half_window + self.guard_cells + 1
            kernel[guard_start:guard_end, 0] = 0
            kernel[:, 0] = kernel[:, 0] * (2 * self.training_cells) / np.sum(kernel[:, 0] != 0)
            
            noise_estimate = ndimage.convolve(data, kernel, mode=self.edge_behavior)
            
        return noise_estimate
    
    def _go_cfar(self, data: np.ndarray) -> np.ndarray:
        """Greatest Of CFAR - takes maximum of left and right window averages."""
        # Compute left and right window averages separately
        left_avg = self._compute_window_average(data, 'left')
        right_avg = self._compute_window_average(data, 'right')
        
        # Take maximum
        return np.maximum(left_avg, right_avg)
    
    def _so_cfar(self, data: np.ndarray) -> np.ndarray:
        """Smallest Of CFAR - takes minimum of left and right window averages."""
        # Compute left and right window averages separately
        left_avg = self._compute_window_average(data, 'left')
        right_avg = self._compute_window_average(data, 'right')
        
        # Take minimum
        return np.minimum(left_avg, right_avg)
    
    def _os_cfar(self, data: np.ndarray) -> np.ndarray:
        """Ordered Statistic CFAR - uses rank-order statistic of training cells."""
        shape = data.shape
        noise_estimate = np.zeros_like(data)
        
        if self.axis == 1:  # Range CFAR
            for i in range(shape[0]):
                for j in range(shape[1]):
                    training_data = self._get_training_cells(data[i, :], j)
                    if len(training_data) > 0:
                        # Sort and take rank-order statistic
                        sorted_data = np.sort(training_data)
                        rank_idx = min(self.os_rank - 1, len(sorted_data) - 1)
                        noise_estimate[i, j] = sorted_data[rank_idx]
        else:  # Pulse CFAR
            for i in range(shape[0]):
                for j in range(shape[1]):
                    training_data = self._get_training_cells(data[:, j], i)
                    if len(training_data) > 0:
                        sorted_data = np.sort(training_data)
                        rank_idx = min(self.os_rank - 1, len(sorted_data) - 1)
                        noise_estimate[i, j] = sorted_data[rank_idx]
                        
        return noise_estimate
    
    def _compute_window_average(self, data: np.ndarray, side: str) -> np.ndarray:
        """Compute average of left or right training window."""
        if self.axis == 1:  # Range CFAR
            if side == 'left':
                kernel = np.zeros((1, self.total_window))
                start_idx = 0
                end_idx = self.half_window - self.guard_cells
                kernel[0, start_idx:end_idx] = 1.0 / self.training_cells
            else:  # right
                kernel = np.zeros((1, self.total_window))
                start_idx = self.half_window + self.guard_cells + 1
                end_idx = self.total_window
                kernel[0, start_idx:end_idx] = 1.0 / self.training_cells
        else:  # Pulse CFAR
            if side == 'left':
                kernel = np.zeros((self.total_window, 1))
                start_idx = 0
                end_idx = self.half_window - self.guard_cells
                kernel[start_idx:end_idx, 0] = 1.0 / self.training_cells
            else:  # right
                kernel = np.zeros((self.total_window, 1))
                start_idx = self.half_window + self.guard_cells + 1
                end_idx = self.total_window
                kernel[start_idx:end_idx, 0] = 1.0 / self.training_cells
                
        return ndimage.convolve(data, kernel, mode=self.edge_behavior)
    
    def _get_training_cells(self, data_slice: np.ndarray, center_idx: int) -> np.ndarray:
        """Extract training cells for OS-CFAR computation."""
        training_cells = []
        
        # Left training window
        left_start = max(0, center_idx - self.half_window)
        left_end = max(0, center_idx - self.guard_cells)
        if left_end > left_start:
            training_cells.extend(data_slice[left_start:left_end])
        
        # Right training window  
        right_start = min(len(data_slice), center_idx + self.guard_cells + 1)
        right_end = min(len(data_slice), center_idx + self.half_window + 1)
        if right_end > right_start:
            training_cells.extend(data_slice[right_start:right_end])
            
        return np.array(training_cells)
    
    def _create_training_mask(self) -> np.ndarray:
        """Create mask for training cells (True=training, False=guard/CUT)."""
        mask = np.ones(self.total_window, dtype=bool)
        
        # Mask out CUT and guard cells
        center = self.half_window
        guard_start = center - self.guard_cells
        guard_end = center + self.guard_cells + 1
        mask[guard_start:guard_end] = False
        
        return mask
    
    def get_pfa(self, threshold_factor: float = None) -> float:
        """
        Calculate theoretical probability of false alarm for CA-CFAR.
        
        Parameters:
        -----------
        threshold_factor : float, optional
            Threshold factor to use. If None, uses instance value.
            
        Returns:
        --------
        pfa : float
            Theoretical probability of false alarm.
            
        Note:
        -----
        This is an approximation valid for CA-CFAR with exponentially 
        distributed noise (Rayleigh clutter). May not be accurate for
        normalized data - consider empirical PFA estimation.
        """
        if self.normalized_data:
            warnings.warn(
                "Theoretical PFA calculation assumes exponentially distributed noise. "
                "This may not be accurate for normalized data. Consider empirical estimation.",
                UserWarning
            )
            
        if threshold_factor is None:
            threshold_factor = self.threshold_factor
            
        n_training = 2 * self.training_cells
        
        # For CA-CFAR with exponential noise
        pfa = (1 + threshold_factor / n_training) ** (-n_training)
        
        return pfa
    
    def set_pfa(self, target_pfa: float) -> float:
        """
        Set threshold factor to achieve target probability of false alarm.
        
        Parameters:
        -----------
        target_pfa : float
            Desired probability of false alarm (0 < pfa < 1).
            
        Returns:
        --------
        threshold_factor : float
            Required threshold factor to achieve target PFA.
            
        Note:
        -----
        This updates the instance threshold_factor and is valid for CA-CFAR
        with exponentially distributed noise. May not be accurate for normalized data.
        """
        if self.normalized_data:
            warnings.warn(
                "Automatic PFA setting assumes exponentially distributed noise. "
                "This may not be accurate for normalized data. Consider manual threshold tuning.",
                UserWarning
            )
            
        if not (0 < target_pfa < 1):
            raise ValueError("target_pfa must be between 0 and 1")
            
        n_training = 2 * self.training_cells
        
        # Solve: pfa = (1 + T/N)^(-N) for T
        threshold_factor = n_training * (target_pfa ** (-1/n_training) - 1)
        
        self.threshold_factor = threshold_factor
        return threshold_factor
    
    def estimate_empirical_pfa(self, test_data: np.ndarray, n_samples: int = 10000) -> float:
        """
        Estimate probability of false alarm empirically using test data.
        
        This is useful for normalized data where theoretical PFA may not be accurate.
        
        Parameters:
        -----------
        test_data : np.ndarray
            Representative noise-only data for PFA estimation.
            Should have same characteristics as operational data.
            
        n_samples : int, default=10000
            Number of random samples to use for estimation.
            
        Returns:
        --------
        empirical_pfa : float
            Empirically estimated probability of false alarm.
        """
        if test_data.ndim != 2:
            raise ValueError("Test data must be 2D array")
            
        # Randomly sample cells from the test data
        rows, cols = test_data.shape
        sample_rows = np.random.randint(0, rows, n_samples)
        sample_cols = np.random.randint(0, cols, n_samples)
        
        # Apply CFAR detection to samples
        detections = self.detect(test_data)
        
        # Count false alarms (detections in noise-only data)
        false_alarms = np.sum(detections[sample_rows, sample_cols])
        empirical_pfa = false_alarms / n_samples
        
        return empirical_pfa
    
    def tune_threshold_for_pfa(self, test_data: np.ndarray, target_pfa: float, 
                               tolerance: float = 0.01, max_iterations: int = 20) -> float:
        """
        Tune threshold factor to achieve target PFA using empirical estimation.
        
        Useful for normalized data where theoretical PFA calculations may be inaccurate.
        
        Parameters:
        -----------
        test_data : np.ndarray
            Representative noise-only data for tuning.
            
        target_pfa : float
            Desired probability of false alarm.
            
        tolerance : float, default=0.01
            Acceptable error in PFA.
            
        max_iterations : int, default=20
            Maximum tuning iterations.
            
        Returns:
        --------
        threshold_factor : float
            Tuned threshold factor.
        """
        # Binary search for optimal threshold
        low_threshold = 0.1
        high_threshold = 10.0
        
        for iteration in range(max_iterations):
            # Try middle threshold
            mid_threshold = (low_threshold + high_threshold) / 2
            original_threshold = self.threshold_factor
            self.threshold_factor = mid_threshold
            
            # Estimate PFA
            empirical_pfa = self.estimate_empirical_pfa(test_data)
            
            # Check if close enough
            if abs(empirical_pfa - target_pfa) < tolerance:
                return mid_threshold
            
            # Adjust search range
            if empirical_pfa > target_pfa:
                low_threshold = mid_threshold
            else:
                high_threshold = mid_threshold
        
        # Restore original threshold if tuning failed
        self.threshold_factor = original_threshold
        warnings.warn(f"Threshold tuning did not converge after {max_iterations} iterations")
        return self.threshold_factor
    
    def __str__(self) -> str:
        """String representation of CFAR parameters."""
        pfa_str = f"{self.get_pfa():.2e}" if not self.normalized_data else "N/A (normalized)"
        return (f"CFAR Detector:\n"
                f"  Type: {self.cfar_type}\n"
                f"  Training cells: {self.training_cells}\n"
                f"  Guard cells: {self.guard_cells}\n"
                f"  Threshold factor: {self.threshold_factor:.2f}\n"
                f"  Axis: {self.axis} ({'Range' if self.axis == 1 else 'Pulse'})\n"
                f"  Normalized data: {self.normalized_data}\n"
                f"  Total window: {self.total_window}\n"
                f"  Theoretical PFA (CA): {pfa_str}")


# Example usage and utility functions
def demo_cfar():
    """Demonstrate CFAR detection with synthetic radar data."""
    # Create synthetic radar data
    np.random.seed(42)
    n_pulses, n_range = 64, 256
    
    # Background noise/clutter
    noise_power = 1.0
    data = np.random.exponential(noise_power, (n_pulses, n_range))
    
    # Add some targets
    targets = [
        (20, 100, 10.0),  # (pulse, range, SNR_dB)
        (40, 150, 15.0),
        (30, 200, 8.0)
    ]
    
    for pulse, range_bin, snr_db in targets:
        signal_power = noise_power * (10 ** (snr_db / 10))
        data[pulse, range_bin] += signal_power
    
    # Apply different CFAR algorithms
    cfar_types = ['CA', 'GO', 'SO', 'OS']
    
    print("CFAR Detection Demo")
    print("=" * 50)
    
    for cfar_type in cfar_types:
        detector = CFAR(
            training_cells=12,
            guard_cells=3,
            threshold_factor=4.0,
            cfar_type=cfar_type
        )
        
        detections, threshold = detector.detect(data, return_threshold=True)
        n_detections = np.sum(detections)
        
        print(f"\n{cfar_type}-CFAR Results:")
        print(f"  Total detections: {n_detections}")
        print(f"  Detection rate: {n_detections / data.size * 100:.3f}%")
        print(f"  Theoretical PFA: {detector.get_pfa():.2e}")
        
        # Check if targets were detected
        for i, (pulse, range_bin, snr_db) in enumerate(targets):
            detected = detections[pulse, range_bin]
            print(f"  Target {i+1} (SNR={snr_db}dB): {'DETECTED' if detected else 'MISSED'}")


# Example usage for normalized data
def demo_normalized_cfar():
    """Demonstrate CFAR detection with normalized radar data."""
    # Create synthetic normalized radar data
    np.random.seed(42)
    n_pulses, n_range = 64, 256
    
    # Background noise normalized to [0,1]
    raw_noise = np.random.exponential(1.0, (n_pulses, n_range))
    noise_min, noise_max = np.min(raw_noise), np.max(raw_noise)
    data = (raw_noise - noise_min) / (noise_max - noise_min)
    
    # Add some normalized targets
    targets = [
        (20, 100, 0.85),  # (pulse, range, normalized_value)
        (40, 150, 0.95),
        (30, 200, 0.78)
    ]
    
    for pulse, range_bin, value in targets:
        data[pulse, range_bin] = value
    
    print("Normalized CFAR Detection Demo")
    print("=" * 50)
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    # Create CFAR detector for normalized data
    detector = CFAR(
        training_cells=12,
        guard_cells=3,
        threshold_factor=1.5,  # Lower threshold for normalized data
        cfar_type='CA',
        normalized_data=True
    )
    
    print(f"\n{detector}")
    
    # Apply detection
    detections, threshold = detector.detect(data, return_threshold=True)
    n_detections = np.sum(detections)
    
    print(f"\nDetection Results:")
    print(f"  Total detections: {n_detections}")
    print(f"  Detection rate: {n_detections / data.size * 100:.3f}%")
    
    # Check target detections
    for i, (pulse, range_bin, value) in enumerate(targets):
        detected = detections[pulse, range_bin]
        thresh_val = threshold[pulse, range_bin]
        print(f"  Target {i+1} (val={value:.3f}, thresh={thresh_val:.3f}): {'DETECTED' if detected else 'MISSED'}")
    
    # Demonstrate empirical PFA estimation
    empirical_pfa = detector.estimate_empirical_pfa(data)
    print(f"\nEmpirical PFA: {empirical_pfa:.2e}")


if __name__ == "__main__":
    demo_cfar()
    print("\n" + "="*50 + "\n")
    demo_normalized_cfar()