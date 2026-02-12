# Copyright (C) 2024 - 2025 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Domain transformation utilities for radar and electromagnetic simulations.

This module provides the DomainTransforms class for converting between frequency,
range, and aspect angle domains commonly used in radar signal processing and
electromagnetic simulations.
"""

import numpy as np


class DomainTransforms:
    """
    A utility class for transforming between frequency, range, and aspect angle domains.
    
    This class handles the mathematical relationships between different signal domains
    commonly used in radar and electromagnetic applications:
    - Frequency domain: Array of frequency samples
    - Range domain: Array of range bins (distance measurements)
    - Aspect domain: Array of aspect angles
    
    The class can be initialized with any one of the three domains and will automatically
    calculate the corresponding parameters for the other domains based on fundamental
    electromagnetic relationships.
    
    Attributes:
        c0 (float): Speed of light in m/s (299,792,458)
    
    Args:
        freq_domain (array-like, optional): Array of frequency samples in Hz
        range_domain (array-like, optional): Array of range samples in meters
        aspect_domain (array-like, optional): Array of aspect angles in degrees
        center_freq (float, optional): Center frequency in Hz (required for range/aspect domains)
    
    Raises:
        RuntimeError: If incorrect number of domains provided (must be exactly one)
        RuntimeError: If center_freq is missing when required for range/aspect domains
    
    Example:
        # Initialize with frequency domain
        freq_samples = np.linspace(28e9, 32e9, 1000)
        transforms = DomainTransforms(freq_domain=freq_samples)
        
        # Access derived range domain
        range_bins = transforms.range_domain
        
        # Initialize with range domain
        range_samples = np.linspace(0, 100, 512)
        transforms = DomainTransforms(range_domain=range_samples, center_freq=30e9)
    """
    
    def __init__(self, freq_domain=None, range_domain=None, aspect_domain=None, center_freq=None):
        # Physical constants
        self.c0 = 299792458  # Speed of light in m/s
        
        # Frequency domain parameters
        self.__freq_domain = None
        self.__bandwidth = None  # FFT bandwidth, including the extra df/2 tails on either side of the domain
        self.__num_freq = None
        self.__delta_freq = None
        self.__center_freq = center_freq
        self.__wavelength = None

        # Range domain parameters
        self.__range_resolution = None
        self.__range_period = None
        self.__range_upsample = 1  # Upsampling factor for range domain

        # Aspect domain parameters
        self.__num_aspect_angle = None
        self.__aspect_angle = None  # Angular span, not including the tails

        # Store input domains
        self.__aspect_domain = aspect_domain
        self.__range_domain = range_domain

        # Validate that exactly one domain is provided
        num_not_none = sum([freq_domain is not None, range_domain is not None, aspect_domain is not None])

        if num_not_none != 1:  # pragma: no cover
            raise RuntimeError("Incorrect number of domains passed to DomainTransforms.")
        elif freq_domain is not None:
            # Initialize from frequency domain
            self.__freq_domain = freq_domain
            self.compute_freq_domain_derived()
            self.calculate_range_domain()
        elif range_domain is not None:
            # Initialize from range domain
            if center_freq is None:
                raise RuntimeError("Missing center frequency")
            self.compute_range_domain_derived()
            self.calculate_freq_domain()
            self.calculate_aspect_domain()  # Always calculates even if it doesn't apply
        elif aspect_domain is not None:
            # Initialize from aspect domain
            if center_freq is None:
                raise RuntimeError("Missing center frequency")
            self.compute_aspect_domain_derived()
            self.calculate_range_domain_from_aspect()

    @property
    def freq_domain(self):
        """
        Array of frequency samples in Hz.
        
        Returns:
            numpy.ndarray: Frequency domain samples
        """
        return self.__freq_domain

    @property
    def bandwidth(self):
        """
        Total FFT bandwidth in Hz.
        
        Includes the extra df/2 tails on either side of the domain following
        ADP (Antenna Design Package) conventions.
        
        Returns:
            float: FFT bandwidth in Hz
        """
        return self.__bandwidth

    @property
    def num_freq(self):
        """
        Number of frequency samples.
        
        Returns:
            int: Number of frequency domain samples
        """
        return self.__num_freq

    @property
    def delta_freq(self):
        """
        Frequency step size in Hz.
        
        Returns:
            float: Frequency resolution in Hz
        """
        return self.__delta_freq

    @property
    def center_freq(self):
        """
        Center frequency in Hz.
        
        For odd-length frequency domains, this is the center sample.
        For even-length domains, this is the first sample in the right half.
        
        Returns:
            float: Center frequency in Hz
        """
        return self.__center_freq

    @property
    def wavelength(self):
        """
        Wavelength at center frequency in meters.
        
        Returns:
            float: Wavelength in meters
        """
        return self.__wavelength

    @property
    def range_resolution(self):
        """
        Range resolution in meters.
        
        The minimum resolvable distance difference in the range domain.
        
        Returns:
            float: Range resolution in meters
        """
        return self.__range_resolution

    @property
    def range_upsample(self):
        """
        Range upsampling factor.
        
        Returns:
            int: Upsampling factor for range domain
        """
        return self.__range_upsample

    @property
    def range_period(self):
        """
        Total range period in meters.
        
        The maximum unambiguous range that can be measured.
        
        Returns:
            float: Range period in meters
        """
        return self.__range_period

    @property
    def aspect_angle(self):
        """
        Total aspect angle span in degrees.
        
        The angular range covered by the aspect domain, not including tails.
        
        Returns:
            float: Aspect angle span in degrees
        """
        return self.__aspect_angle
    
    @aspect_angle.setter
    def aspect_angle(self, value):
        """
        Set the aspect angle span and recalculate range domain.
        
        Args:
            value (float): Aspect angle span in degrees
        """
        self.__aspect_angle = value
        self.calculate_range_domain_from_aspect()

    @property
    def num_aspect_angle(self):
        """
        Number of aspect angle samples.
        
        Returns:
            int: Number of aspect angle samples
        """
        return self.__num_aspect_angle
    
    @num_aspect_angle.setter
    def num_aspect_angle(self, value):
        """
        Set the number of aspect angle samples and recalculate aspect domain.
        
        Args:
            value (int): Number of aspect angle samples
        """
        self.__num_aspect_angle = value
        self.calculate_aspect_domain()

    @property
    def range_domain(self):
        """
        Array of range samples in meters.
        
        Returns:
            numpy.ndarray: Range domain samples
        """
        return self.__range_domain

    @property
    def aspect_domain(self):
        """
        Array of aspect angle samples in degrees.
        
        Returns:
            numpy.ndarray: Aspect domain samples
        """
        return self.__aspect_domain

    def compute_freq_domain_derived(self):
        """
        Calculate derived parameters from the frequency domain.
        
        Computes number of samples, frequency step size, bandwidth, and center frequency
        following ADP (Antenna Design Package) conventions.
        
        Raises:
            RuntimeError: If freq_domain is None
        """
        if self.freq_domain is None:
            raise RuntimeError("freq_domain is None")
            
        self.__num_freq = len(self.freq_domain)
        self.__delta_freq = self.freq_domain[1] - self.freq_domain[0]
        
        # Consistent with ADP definition, extra df/2 tails
        # Adopt ADP conventions that the center frequency is the center sample for odd-length frequency domains,
        # or the first sample in the right half for even-length domains
        self.__bandwidth = self.num_freq * self.delta_freq
        self.__center_freq = self.freq_domain[len(self.freq_domain) // 2]

    def compute_range_domain_derived(self):
        """
        Calculate derived parameters from the range domain.
        
        Computes range resolution and range period from the input range domain.
        
        Raises:
            RuntimeError: If range_domain is None
        """
        if self.range_domain is None:
            raise RuntimeError("range_domain is None")
            
        self.__range_resolution = self.range_domain[1] - self.range_domain[0]
        self.__range_period = self.range_domain[-1] + self.range_resolution

    def compute_aspect_domain_derived(self):
        """
        Calculate derived parameters from the aspect domain.
        
        Computes the total aspect angle span and number of aspect angle samples.
        
        Note:
            This calculation might not necessarily be correct for all cases
            and may need an additional angular frequency step (df).
        
        Raises:
            RuntimeError: If aspect_domain is None
        """
        if self.aspect_domain is None:
            raise RuntimeError("_aspect_domain is None")
            
        # This might not necessarily be correct - need an additional df
        self.__aspect_angle = self.aspect_domain[-1] - self.aspect_domain[0]
        self.__num_aspect_angle = len(self.aspect_domain)

    def calculate_range_domain(self):
        """
        Calculate the range domain from frequency domain parameters.
        
        Uses the fundamental relationship between bandwidth and range resolution:
        range_resolution = c / (2 * bandwidth)
        
        The bandwidth is consistent with ADP conventions.
        """
        # Calculate range resolution from bandwidth
        self.__range_resolution = self.c0 / self.bandwidth / 2  # The bandwidth is consistent with ADP
        self.__range_period = self.num_freq * self.range_resolution
        
        # Create range domain with optional upsampling
        num_range = self.range_upsample * self.num_freq
        self.__range_domain = np.linspace(0, self.range_period - self.range_resolution, num=num_range)

    def calculate_freq_domain(self):
        """
        Calculate the frequency domain from range domain parameters.
        
        Uses the inverse relationship: bandwidth = c / (2 * range_resolution)
        Creates a symmetric frequency domain centered around the center frequency.
        """
        # Calculate bandwidth from range resolution
        self.__bandwidth = self.c0 / (2 * self.range_resolution)
        self.__num_freq = len(self.range_domain)
        self.__delta_freq = self.bandwidth / self.num_freq
        
        # Create symmetric frequency domain around center frequency
        freqStart = self.center_freq - np.floor(0.5 * self.num_freq) * self.delta_freq
        freqStop = self.center_freq + (self.num_freq - 1) * self.delta_freq
        self.__freq_domain = np.linspace(freqStart, freqStop, num=self.num_freq)

    def calculate_aspect_domain(self):
        """
        Calculate the aspect domain parameters from range domain.
        
        Determines the number of aspect angle samples and total angular span
        based on the range period and center frequency using radar geometry relationships.
        """
        # Number of aspect angles is determined by range sampling
        self.__num_aspect_angle = int(np.ceil(self.range_period / self.range_resolution))
        
        if self.center_freq:
            # Calculate angular resolution based on wavelength and range period
            d_ang = self.c0 / self.center_freq / 2 / self.range_period * 180 / np.pi
            self.__aspect_angle = d_ang * (self.num_aspect_angle - 1)

    def calculate_range_domain_from_aspect(self):
        """
        Calculate the range domain parameters from aspect domain.
        
        Uses radar geometry relationships to determine range period and resolution
        from the aspect angle span and number of samples.
        
        Raises:
            RuntimeError: If center_freq is not defined
        """
        # Calculate angular step size
        d_ang = self.aspect_angle / (self.num_aspect_angle - 1)

        if self.center_freq:
            # Calculate range period from angular resolution and wavelength
            self.__range_period = self.c0 / self.center_freq / 2 / d_ang * 180 / np.pi
            self.__range_resolution = self.range_period / self.num_aspect_angle
        else:
            raise RuntimeError("Center Freq most be defined for domain from aspect")
