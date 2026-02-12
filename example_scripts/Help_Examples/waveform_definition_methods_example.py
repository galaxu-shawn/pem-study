"""
Waveform Definition Methods Example Script
=========================================

This script demonstrates the various ways to define Waveform() objects for electromagnetic
simulations. The Waveform class is used to configure radar and communication system
parameters including frequency, bandwidth, timing, and output specifications.

Main Waveform Types:
- PulsedDoppler waveforms: Traditional radar with pulse trains
- FMCW waveforms: Frequency Modulated Continuous Wave radar
- Communication waveforms: For P2P and wireless planning simulations
- Custom waveforms: Specialized configurations for specific applications

Author: Example Script
Date: June 2025
"""

import os

import sys
import numpy as np
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.actor import Actors
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 


def main():
    """
    Comprehensive demonstration of Waveform definition methods
    """
    
    print("=" * 65)
    print("WAVEFORM DEFINITION METHODS DEMONSTRATION")
    print("=" * 65)
    
    # ========================================================================
    # 1. BASIC PULSED DOPPLER WAVEFORMS
    # ========================================================================
    # Method 1A: Standard automotive radar configuration
    print("   a) Standard automotive radar (77 GHz):")
    automotive_waveform_dict = {
        "mode": "PulsedDoppler",
        "output": "FreqPulse",      # this can be FreqPulse or RangeDoppler, this will determine the output data type
        "center_freq": 77.0e9,      # 77 GHz
        "bandwidth": 600e6,         # 600 MHz bandwidth
        "num_freq_samples": 512,    # 512 frequency samples
        "cpi_duration": 4.0e-3,     # 4 ms coherent processing interval
        "num_pulse_CPI": 256,       # 256 pulses per CPI
        "tx_multiplex": "SIMULTANEOUS",
        "mode_delay": "CENTER_CHIRP"
    }
    automotive_waveform = Waveform(automotive_waveform_dict)
    print(f"     ✓ Automotive radar: {automotive_waveform.center_freq/1e9:.1f} GHz, "
          f"{automotive_waveform.bandwidth/1e6:.0f} MHz BW")
    
    # Method 1B: High-resolution radar
    print("   b) High-resolution radar:")
    high_res_waveform_dict = {
        "mode": "PulsedDoppler",
        "output": "RangeDoppler",    # Direct range-Doppler output
        "center_freq": 94.0e9,       # W-band radar
        "bandwidth": 2.0e9,          # 2 GHz for high range resolution
        "num_freq_samples": 1024,    # High frequency resolution
        "cpi_duration": 10.0e-3,     # Longer CPI for velocity resolution
        "num_pulse_CPI": 512,        # More pulses for Doppler resolution
        "tx_multiplex": "INDIVIDUAL",
        "mode_delay": "CENTER_CHIRP"
    }
    high_res_waveform = Waveform(high_res_waveform_dict)
    range_res = 3e8 / (2 * high_res_waveform.bandwidth)
    print(f"     ✓ High-res radar: {range_res:.2f}m range resolution, "
          f"{high_res_waveform.num_pulse_cpi} Doppler bins")
    
    # ========================================================================
    # 2. FMCW WAVEFORMS
    # ========================================================================
    print("\n2. FMCW (Frequency Modulated Continuous Wave) Waveforms")
    print("-" * 55)
    
    print("   FMCW waveforms use continuous frequency sweeps instead of pulses.")
    
    # Method 2A: Basic FMCW configuration
    print("   a) Basic FMCW waveform:")
    fmcw_basic_dict = {
        "mode": "FMCW",
        "output": "ADC_SAMPLES",
        "center_freq": 24.0e9,       # 24 GHz ISM band
        "bandwidth": 200e6,          # 200 MHz sweep
        "num_freq_samples": 256,     # ADC samples per chirp
        "cpi_duration": 2.0e-3,      # 2 ms frame time
        "num_pulse_CPI": 128,        # Number of chirps per frame
        "ADC_SampleRate": 10e6,      # 10 MHz ADC sampling rate
        "isIQChannel": True,         # I/Q sampling
        "tx_multiplex": "SIMULTANEOUS"
    }
    fmcw_basic = Waveform(fmcw_basic_dict)
    print(f"     ✓ FMCW: {fmcw_basic.center_freq/1e9:.1f} GHz, "
          f"{fmcw_basic.adc_sample_rate/1e6:.0f} MHz ADC rate")
    
    # Method 2B: Automotive FMCW (short range)
    print("   b) Automotive FMCW (short range):")
    fmcw_automotive_dict = {
        "mode": "FMCW",
        "output": "RangeDoppler",
        "center_freq": 77.0e9,
        "bandwidth": 4.0e9,          # 4 GHz for sub-meter resolution
        "num_freq_samples": 512,
        "cpi_duration": 50.0e-3,     # 50 ms for slow vehicle detection
        "num_pulse_CPI": 64,         # 64 chirps
        "ADC_SampleRate": 25e6,      # 25 MHz ADC
        "isIQChannel": True,
        "tx_multiplex": "INDIVIDUAL"
    }
    fmcw_automotive = Waveform(fmcw_automotive_dict)
    max_range = (3e8 * fmcw_automotive.adc_sample_rate) / (2 * fmcw_automotive.bandwidth * fmcw_automotive.num_freq_samples) * fmcw_automotive.num_freq_samples
    print(f"     ✓ Automotive FMCW: {max_range:.0f}m max range, "
          f"{3e8/(2*fmcw_automotive.bandwidth):.2f}m resolution")
    
    # ========================================================================
    # 3. COMMUNICATION WAVEFORMS
    # ========================================================================
    print("\n3. Communication Waveforms")
    print("-" * 30)
    
    print("   Waveforms optimized for wireless communication simulations.")
    
    # Method 3A: 5G sub-6 GHz communication
    print("   a) 5G sub-6 GHz communication:")
    comm_5g_sub6_dict = {
        "mode": "PulsedDoppler",
        "output": "FreqPulse",
        "center_freq": 3.5e9,        # 3.5 GHz 5G band
        "bandwidth": 100e6,          # 100 MHz channel
        "num_freq_samples": 1024,    # High spectral resolution
        "cpi_duration": 1.0e-3,      # 1 ms subframe
        "num_pulse_CPI": 14,         # 14 OFDM symbols per subframe
        "tx_multiplex": "INDIVIDUAL",
        "mode_delay": "CENTER_CHIRP"
    }
    comm_5g_sub6 = Waveform(comm_5g_sub6_dict)
    print(f"     ✓ 5G sub-6: {comm_5g_sub6.center_freq/1e9:.1f} GHz, "
          f"{comm_5g_sub6.bandwidth/1e6:.0f} MHz channel")
    
    # Method 3B: 5G mmWave communication
    print("   b) 5G mmWave communication:")
    comm_5g_mmwave_dict = {
        "mode": "PulsedDoppler", 
        "output": "FreqPulse",
        "center_freq": 28.0e9,       # 28 GHz mmWave
        "bandwidth": 800e6,          # 800 MHz wide channel
        "num_freq_samples": 2048,    # Very high resolution
        "cpi_duration": 0.5e-3,      # 0.5 ms slot
        "num_pulse_CPI": 14,         # 14 symbols
        "tx_multiplex": "SIMULTANEOUS",
        "tx_incident_power": 0.1,    # 100 mW transmit power
        "rx_noise_db": -90,          # -90 dBm noise floor
        "rx_gain_db": 20             # 20 dB receiver gain
    }
    comm_5g_mmwave = Waveform(comm_5g_mmwave_dict)
    print(f"     ✓ 5G mmWave: {comm_5g_mmwave.center_freq/1e9:.0f} GHz, "
          f"{comm_5g_mmwave.tx_incident_power:.1f}W Tx power")
    
    # Method 3C: WiFi 6 communication
    print("   c) WiFi 6 communication:")
    wifi6_dict = {
        "mode": "PulsedDoppler",
        "output": "FreqPulse", 
        "center_freq": 5.8e9,        # 5.8 GHz WiFi band
        "bandwidth": 80e6,           # 80 MHz channel
        "num_freq_samples": 256,     # OFDM subcarriers
        "cpi_duration": 4.0e-3,      # 4 ms frame
        "num_pulse_CPI": 100,        # Multiple OFDM symbols
        "tx_multiplex": "INDIVIDUAL",
        "tx_incident_power": 0.02,   # 20 mW (typical WiFi power)
        "mode_delay": "FIRST_CHIRP"
    }
    wifi6 = Waveform(wifi6_dict)
    print(f"     ✓ WiFi 6: {wifi6.center_freq/1e9:.1f} GHz, "
          f"{wifi6.bandwidth/1e6:.0f} MHz channel")
    
    # ========================================================================
    # 4. TIMING CONFIGURATION METHODS
    # ========================================================================
    print("\n4. Timing Configuration Methods")
    print("-" * 35)
    
    print("   Different ways to specify timing parameters.")
    
    # Method 4A: Using CPI duration (recommended)
    print("   a) Using CPI duration:")
    timing_cpi_dict = {
        "mode": "PulsedDoppler",
        "center_freq": 10.0e9,
        "bandwidth": 500e6,
        "cpi_duration": 8.0e-3,      # Specify total CPI time
        "num_pulse_CPI": 200,        # Number of pulses in CPI
        # pulse_interval calculated automatically as cpi_duration/num_pulse_CPI
    }
    timing_cpi = Waveform(timing_cpi_dict)
    calculated_pri = timing_cpi.cpi_duration / timing_cpi.num_pulse_cpi
    print(f"     ✓ CPI method: {timing_cpi.cpi_duration*1000:.1f}ms CPI, "
          f"{calculated_pri*1e6:.1f}μs PRI")
    
    # Method 4B: Using pulse interval
    print("   b) Using pulse interval:")
    timing_pri_dict = {
        "mode": "PulsedDoppler", 
        "center_freq": 10.0e9,
        "bandwidth": 500e6,
        "pulse_interval": 50e-6,     # 50 μs between pulses
        "num_pulse_CPI": 160,        # Number of pulses
        # cpi_duration calculated automatically
    }
    timing_pri = Waveform(timing_pri_dict)
    calculated_cpi = timing_pri.pulse_interval * timing_pri.num_pulse_cpi
    print(f"     ✓ PRI method: {timing_pri.pulse_interval*1e6:.0f}μs PRI, "
          f"{calculated_cpi*1000:.1f}ms total CPI")
    
    # Method 4C: Mode delay options
    print("   c) Mode delay configurations:")
    
    delay_center_dict = {
        "mode": "PulsedDoppler",
        "center_freq": 35.0e9,
        "bandwidth": 1.0e9,
        "cpi_duration": 5.0e-3,
        "num_pulse_CPI": 100,
        "mode_delay": "CENTER_CHIRP"  # Align to center of chirp
    }
    delay_center = Waveform(delay_center_dict)
    
    delay_first_dict = {
        "mode": "PulsedDoppler",
        "center_freq": 35.0e9, 
        "bandwidth": 1.0e9,
        "cpi_duration": 5.0e-3,
        "num_pulse_CPI": 100,
        "mode_delay": "FIRST_CHIRP"   # Align to start of chirp
    }
    delay_first = Waveform(delay_first_dict)
    
    print(f"     ✓ CENTER_CHIRP timing: {delay_center.mode_delay}")
    print(f"     ✓ FIRST_CHIRP timing: {delay_first.mode_delay}")
    
    # ========================================================================
    # 5. OUTPUT TYPE CONFIGURATIONS
    # ========================================================================
    print("\n5. Output Type Configurations")
    print("-" * 33)
    
    print("   Different output formats for different analysis needs.")
    
    # Method 5A: FreqPulse output (most common)
    print("   a) FreqPulse output:")
    output_freqpulse_dict = {
        "mode": "PulsedDoppler",
        "output": "FreqPulse",      # Raw frequency domain data
        "center_freq": 60.0e9,
        "bandwidth": 1.0e9,
        "num_freq_samples": 256,
        "cpi_duration": 3.0e-3,
        "num_pulse_CPI": 128
    }
    output_freqpulse = Waveform(output_freqpulse_dict)
    print(f"     ✓ FreqPulse: Returns [{output_freqpulse.num_pulse_cpi} x {output_freqpulse.num_freq_samples}] array")
    
    # Method 5B: RangeDoppler output (processed)
    print("   b) RangeDoppler output:")
    output_rangedoppler_dict = {
        "mode": "PulsedDoppler",
        "output": "RangeDoppler",   # Processed range-Doppler map
        "center_freq": 60.0e9,
        "bandwidth": 1.0e9,
        "num_freq_samples": 256,
        "cpi_duration": 3.0e-3,
        "num_pulse_CPI": 128
    }
    output_rangedoppler = Waveform(output_rangedoppler_dict)
    print(f"     ✓ RangeDoppler: Returns processed range-Doppler map")
    
    # Method 5C: ADC_SAMPLES output (FMCW)
    print("   c) ADC_SAMPLES output:")
    output_adc_dict = {
        "mode": "FMCW",
        "output": "ADC_SAMPLES",    # Raw ADC time-domain samples
        "center_freq": 24.0e9,
        "bandwidth": 200e6,
        "num_freq_samples": 512,
        "cpi_duration": 1.0e-3,
        "num_pulse_CPI": 64,
        "ADC_SampleRate": 20e6
    }
    output_adc = Waveform(output_adc_dict)
    print(f"     ✓ ADC_SAMPLES: Raw time-domain samples at {output_adc.adc_sample_rate/1e6:.0f} MHz")
    
    # ========================================================================
    # 6. MULTIPLEXING AND CHANNEL CONFIGURATIONS
    # ========================================================================
    print("\n6. Multiplexing and Channel Configurations")
    print("-" * 45)
    
    print("   Different transmission and channel configurations.")
    
    # Method 6A: Simultaneous transmission
    print("   a) Simultaneous transmission:")
    simultaneous_dict = {
        "mode": "PulsedDoppler",
        "center_freq": 77.0e9,
        "bandwidth": 1.0e9,
        "tx_multiplex": "SIMULTANEOUS",  # All TX antennas transmit together
        "cpi_duration": 4.0e-3,
        "num_pulse_CPI": 256,
        "isIQChannel": True              # I/Q sampling enabled
    }
    simultaneous_wf = Waveform(simultaneous_dict)
    print(f"     ✓ SIMULTANEOUS: All TX antennas transmit together")
    
    # Method 6B: Interleaved transmission
    print("   b) Interleaved transmission:")
    interleaved_dict = {
        "mode": "PulsedDoppler",
        "center_freq": 77.0e9, 
        "bandwidth": 1.0e9,
        "tx_multiplex": "INDIVIDUAL",    # TX antennas transmit individually/interleaved
        "cpi_duration": 4.0e-3,
        "num_pulse_CPI": 256,
        "isIQChannel": True
    }
    interleaved_wf = Waveform(interleaved_dict)
    print(f"     ✓ INDIVIDUAL: TX antennas transmit in sequence")
    
    # Method 6C: I-only channel (no Q)
    print("   c) I-only channel (no quadrature):")
    i_only_dict = {
        "mode": "PulsedDoppler",
        "center_freq": 77.0e9,
        "bandwidth": 1.0e9,
        "cpi_duration": 4.0e-3,
        "num_pulse_CPI": 256,
        "isIQChannel": False             # Only I channel, no Q
    }
    i_only_wf = Waveform(i_only_dict)
    print(f"     ✓ I-only: Real sampling without quadrature component")
    
    # ========================================================================
    # 7. POWER AND NOISE CONFIGURATIONS
    # ========================================================================
    print("\n7. Power and Noise Configurations")
    print("-" * 37)
    
    print("   RF system parameters for realistic simulations.")
    
    # Method 7A: High-power radar
    print("   a) High-power radar system:")
    high_power_dict = {
        "mode": "PulsedDoppler",
        "center_freq": 94.0e9,       # W-band
        "bandwidth": 2.0e9,
        "cpi_duration": 10.0e-3,
        "num_pulse_CPI": 512,
        "tx_incident_power": 100.0,  # 100 watts peak power
        "rx_noise_db": -110,         # -110 dBm noise floor
        "rx_gain_db": 40             # 40 dB receiver gain
    }
    high_power_wf = Waveform(high_power_dict)
    print(f"     ✓ High power: {high_power_wf.tx_incident_power:.0f}W Tx, "
          f"{high_power_wf.rx_gain_db}dB Rx gain")
    
    # Method 7B: Low-power sensor
    print("   b) Low-power sensor system:")
    low_power_dict = {
        "mode": "PulsedDoppler", 
        "center_freq": 24.0e9,       # ISM band
        "bandwidth": 200e6,
        "cpi_duration": 50.0e-3,     # Long integration time
        "num_pulse_CPI": 1000,       # Many pulses for sensitivity
        "tx_incident_power": 0.001,  # 1 mW (-30 dBm)
        "rx_noise_db": -95,          # -95 dBm noise floor
        "rx_gain_db": 15             # 15 dB receiver gain
    }
    low_power_wf = Waveform(low_power_dict)
    print(f"     ✓ Low power: {low_power_wf.tx_incident_power*1000:.0f}mW Tx, "
          f"long {low_power_wf.cpi_duration*1000:.0f}ms integration")
    
    # ========================================================================
    # 8. SPECIALIZED WAVEFORM CONFIGURATIONS
    # ========================================================================
    print("\n8. Specialized Waveform Configurations")
    print("-" * 42)
    
    print("   Waveforms for specific applications and scenarios.")
    
    # Method 8A: Weather radar
    print("   a) Weather radar (meteorological):")
    weather_radar_dict = {
        "mode": "PulsedDoppler",
        "output": "RangeDoppler",
        "center_freq": 5.6e9,        # C-band weather radar
        "bandwidth": 10e6,           # Narrow bandwidth for long range
        "num_freq_samples": 128,     # Sufficient for weather targets
        "cpi_duration": 100.0e-3,    # Long CPI for slow weather motion
        "num_pulse_CPI": 1000,       # Many pulses for averaging
        "tx_multiplex": "SIMULTANEOUS",
        "tx_incident_power": 1000.0  # 1 kW peak power
    }
    weather_radar_wf = Waveform(weather_radar_dict)
    max_weather_range = 3e8 * weather_radar_wf.num_freq_samples / (2 * weather_radar_wf.bandwidth)
    print(f"     ✓ Weather radar: {max_weather_range/1000:.0f}km range, "
          f"{weather_radar_wf.cpi_duration*1000:.0f}ms CPI")
    
    # Method 8B: SAR (Synthetic Aperture Radar)
    print("   b) SAR imaging radar:")
    sar_dict = {
        "mode": "PulsedDoppler",
        "output": "FreqPulse", 
        "center_freq": 9.6e9,        # X-band SAR
        "bandwidth": 600e6,          # Wide bandwidth for resolution
        "num_freq_samples": 2048,    # High range resolution
        "cpi_duration": 1.0e-3,      # Short CPI for motion compensation
        "num_pulse_CPI": 128,        # Moderate pulse count
        "tx_multiplex": "INDIVIDUAL",
        "mode_delay": "FIRST_CHIRP"  # Precise timing for SAR
    }
    sar_wf = Waveform(sar_dict)
    sar_resolution = 3e8 / (2 * sar_wf.bandwidth)
    print(f"     ✓ SAR radar: {sar_resolution:.2f}m resolution, "
          f"{sar_wf.center_freq/1e9:.1f} GHz X-band")
    
    # Method 8C: Through-wall radar
    print("   c) Through-wall sensing radar:")
    through_wall_dict = {
        "mode": "PulsedDoppler",
        "output": "RangeDoppler",
        "center_freq": 2.4e9,        # Lower frequency for penetration
        "bandwidth": 1.5e9,          # Ultra-wideband for resolution
        "num_freq_samples": 1024,    # High resolution
        "cpi_duration": 200.0e-3,    # Long observation for human motion
        "num_pulse_CPI": 2000,       # Many pulses for small motion detection
        "tx_multiplex": "SIMULTANEOUS",
        "tx_incident_power": 0.01    # Low power to meet regulations
    }
    through_wall_wf = Waveform(through_wall_dict)
    wall_penetration_range = 3e8 / (2 * through_wall_wf.bandwidth)
    print(f"     ✓ Through-wall: {wall_penetration_range:.2f}m resolution, "
          f"{through_wall_wf.tx_incident_power*1000:.0f}mW power")
    
    # ========================================================================
    # 9. WINDOWING AND POST-PROCESSING PARAMETERS
    # ========================================================================
    print("\n9. Windowing and Post-Processing Parameters")
    print("-" * 47)
    
    print("   Configure windowing for sidelobe control.")
    
    # Method 9A: Custom windowing specifications
    print("   a) Custom windowing specifications:")
    custom_window_dict = {
        "mode": "PulsedDoppler",
        "output": "RangeDoppler",
        "center_freq": 77.0e9,
        "bandwidth": 1.0e9,
        "cpi_duration": 4.0e-3,
        "num_pulse_CPI": 256
        # Note: r_specs and d_specs are set automatically in Waveform.__init__()
        # Default: "hann,50" (Hann window with 50dB sidelobe level)
    }
    custom_window_wf = Waveform(custom_window_dict)
    print(f"     ✓ Default windowing: Range={custom_window_wf.r_specs}, "
          f"Doppler={custom_window_wf.d_specs}")
    
    # Method 9B: Modify windowing after creation
    print("   b) Modified windowing after creation:")
    modified_window_wf = Waveform(custom_window_dict.copy())
    modified_window_wf.r_specs = "blackman,60"    # Blackman window, 60dB sidelobes
    modified_window_wf.d_specs = "kaiser,40"      # Kaiser window, 40dB sidelobes
    print(f"     ✓ Modified windowing: Range={modified_window_wf.r_specs}, "
          f"Doppler={modified_window_wf.d_specs}")
    
    # ========================================================================
    # 10. FREQUENCY BAND EXAMPLES
    # ========================================================================
    print("\n10. Common Frequency Band Examples")
    print("-" * 38)
    
    print("   Waveforms for different frequency bands and applications.")
    
    frequency_bands = [
        {
            'name': 'L-band (GPS)',
            'center_freq': 1.575e9,
            'bandwidth': 20e6,
            'application': 'Navigation and positioning'
        },
        {
            'name': 'S-band (WiFi)', 
            'center_freq': 2.4e9,
            'bandwidth': 80e6,
            'application': 'Wireless communication'
        },
        {
            'name': 'C-band (Weather)',
            'center_freq': 5.6e9,
            'bandwidth': 10e6,
            'application': 'Weather radar'
        },
        {
            'name': 'X-band (Marine)',
            'center_freq': 9.4e9,
            'bandwidth': 200e6,
            'application': 'Marine and SAR radar'
        },
        {
            'name': 'Ku-band (Satellite)',
            'center_freq': 15.0e9,
            'bandwidth': 500e6,
            'application': 'Satellite communication'
        },
        {
            'name': 'K-band (Automotive)',
            'center_freq': 24.0e9,
            'bandwidth': 250e6,
            'application': 'Short-range automotive radar'
        },
        {
            'name': 'W-band (Automotive)',
            'center_freq': 77.0e9,
            'bandwidth': 1.0e9,
            'application': 'Long-range automotive radar'
        },
        {
            'name': 'W-band (Imaging)',
            'center_freq': 94.0e9,
            'bandwidth': 2.0e9,
            'application': 'High-resolution imaging radar'
        }
    ]
    
    for band in frequency_bands:
        band_dict = {
            "mode": "PulsedDoppler",
            "center_freq": band['center_freq'],
            "bandwidth": band['bandwidth'],
            "cpi_duration": 5.0e-3,
            "num_pulse_CPI": 128
        }
        band_wf = Waveform(band_dict)
        wavelength = 3e8 / band_wf.center_freq
        print(f"     {band['name']:18}: {band_wf.center_freq/1e9:5.1f} GHz, "
              f"λ={wavelength*100:.1f}cm - {band['application']}")
    
    # ========================================================================
    # 11. EXTRACTING DOMAIN INFORMATION FROM WAVEFORMS
    # ========================================================================
    print("\n11. Extracting Domain Information from Waveforms")
    print("-" * 50)
    
    print("   Demonstrating how to extract range, frequency, velocity, and time domains.")
    print("   The get_response_domains() method computes domain arrays based on waveform parameters.")
    
    # Method 11A: Extract domains from PulsedDoppler waveform with FreqPulse output
    print("   a) FreqPulse output domain extraction:")

    all_actors = Actors()
    dummy_name = all_actors.add_actor('dummy')
    
    # we need to add a waveform to an antenna, then we can extract the domains
    tx_rx_device = add_single_tx_rx(
        all_actors=all_actors,
        mode_name='mode1',
        waveform=automotive_waveform,
        ffd_file='dipole.ffd',                 # Use FFD file for both TX and RX, this will look for a file in the antenna_device_library
        scale_pattern=4.0
    )

    which_mode = tx_rx_device.modes['mode1']  # tell it which mode we want to get respones from
    tx_rx_device.waveforms['mode1'].get_response_domains(which_mode)
    vel_domain = tx_rx_device.waveforms['mode1'].vel_domain
    # Extract response domains

    
    print(f"     ✓ Frequency domain: {len(tx_rx_device.waveforms['mode1'].freq_domain)} samples")
    print(f"       - Range: {tx_rx_device.waveforms['mode1'].freq_domain[0]/1e9:.3f} to {tx_rx_device.waveforms['mode1'].freq_domain[-1]/1e9:.3f} GHz")
    print(f"       - Center: {tx_rx_device.waveforms['mode1'].center_freq/1e9:.1f} GHz")
    
    print(f"     ✓ Pulse domain: {len(tx_rx_device.waveforms['mode1'].pulse_domain)} samples")
    print(f"       - Range: {tx_rx_device.waveforms['mode1'].pulse_domain[0]*1000:.2f} to {tx_rx_device.waveforms['mode1'].pulse_domain[-1]*1000:.2f} ms")
    print(f"       - Duration: {tx_rx_device.waveforms['mode1'].cpi_duration*1000:.1f} ms total")
    
    print(f"     ✓ Range domain: {len(tx_rx_device.waveforms['mode1'].rng_domain)} samples")
    print(f"       - Range: 0 to {tx_rx_device.waveforms['mode1'].max_range:.0f} m")
    print(f"       - Resolution: {tx_rx_device.waveforms['mode1'].max_range/len(tx_rx_device.waveforms['mode1'].rng_domain):.2f} m")
    
    print(f"     ✓ Velocity domain: {len(tx_rx_device.waveforms['mode1'].vel_domain)} samples")
    print(f"       - Range: {tx_rx_device.waveforms['mode1'].vel_domain[0]:.1f} to {tx_rx_device.waveforms['mode1'].vel_domain[-1]:.1f} m/s")
    print(f"       - Resolution: {tx_rx_device.waveforms['mode1'].vel_res:.2f} m/s")
    

    
    print("\n   Summary of domain extraction methods:")
    print("   • Use get_response_domains(h_mode) to compute all domain arrays")
    print("   • freq_domain: Frequency samples for each pulse/chirp")
    print("   • pulse_domain: Time samples for pulse train")
    print("   • rng_domain: Range bins for target distance")
    print("   • vel_domain: Velocity bins for Doppler processing")
    print("   • fast_time_domain: Fast time samples (range-related)")
    print("   • Domains depend on waveform parameters and output type")
    print("   • Use domains for creating analysis grids and bin calculations")
    print("   • Check unambiguous ranges and velocities for proper design")

    # ========================================================================
    # 12. SUMMARY AND BEST PRACTICES
    # ========================================================================
    print("\n12. Summary and Best Practices")
    print("-" * 35)
    
    print("   Key waveform configuration principles:")
    print("   • Required parameters: 'mode', 'center_freq', 'bandwidth'")
    print("   • Use 'PulsedDoppler' for most radar applications")
    print("   • Use 'FMCW' for continuous wave automotive radar")
    print("   • 'FreqPulse' output for raw data processing")
    print("   • 'RangeDoppler' output for processed results")
    print("   • Consider frequency-dependent material properties")
    print("   • Match bandwidth to required range resolution")
    print("   • Match CPI duration to target dynamics")
    print("   • Use 'INDIVIDUAL' tx_multiplex for MIMO processing")
    print("   • Include noise and power parameters for link budgets")
    
    print("\n   Common parameter relationships:")
    print("   • Range resolution = c / (2 × bandwidth)")
    print("   • Max unambiguous range = c × PRI / 2")
    print("   • Velocity resolution = λ / (2 × CPI_duration)")
    print("   • Doppler bandwidth = 1 / PRI")
    print("   • Wavelength λ = c / center_frequency")
    
    print("\n   Typical applications:")
    print("   • Automotive radar: 77 GHz, 1 GHz BW, 4ms CPI")
    print("   • Weather radar: 5.6 GHz, 10 MHz BW, 100ms CPI")
    print("   • 5G communications: 28 GHz, 800 MHz BW, 0.5ms slot")
    print("   • Through-wall sensing: 2.4 GHz, 1.5 GHz UWB, 200ms CPI")
    print("   • SAR imaging: 9.6 GHz, 600 MHz BW, 1ms CPI")
    
    # Count total waveforms created
    total_waveforms = 30  # Approximate count from all examples
    print(f"\n   Total waveforms demonstrated: {total_waveforms}")
    print("   Frequency range: 1.575 GHz to 94 GHz")
    print("   Applications: Radar, Communication, Sensing, Imaging")
    
    print("\n" + "=" * 65)
    print("WAVEFORM DEFINITION METHODS DEMONSTRATION COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    """
    Run the waveform definition methods demonstration
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()