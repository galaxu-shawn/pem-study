# convert mvg file format to ffd format
import os
import sys
import numpy as np
from pathlib import Path


def convert_mvg_to_ffd(mvg_file, ffd_file):
    """
    Convert a .mvg file to a .ffd file format.
    
    MVG format appears to contain:
    - First line: "3" (indicating 3D pattern data)
    - Second line: Header with column names
    - Data rows: Azimuth, Elevation, Frequency, ETheta_Real, ETheta_Imag, EPhi_Real, EPhi_Imag, ETotal
    
    FFD format contains:
    - Theta range: start end num_points
    - Phi range: start end num_points  
    - Frequencies count
    - Frequency value
    - Data matrix: ETheta_Real ETheta_Imag EPhi_Real EPhi_Imag (for each theta/phi combination)
    
    Parameters:
    mvg_file (str): Path to the input .mvg file.
    ffd_file (str): Path to the output .ffd file.
    """
    
    # Read and parse the MVG file
    with open(mvg_file, 'r') as file:
        lines = file.readlines()

    # Remove comments and empty lines
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    if len(lines) < 3:
        raise ValueError("MVG file appears to be empty or malformed")
    
    # Skip the first line (appears to be "3")
    # Skip the header line 
    data_lines = lines[2:]
    
    # Parse the data
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 8:  # Ensure we have all required columns
            # Convert to float: [Azimuth, Elevation, Frequency, ETheta_Real, ETheta_Imag, EPhi_Real, EPhi_Imag, ETotal]
            data.append([float(part) for part in parts[:8]])
    
    if not data:
        raise ValueError("No valid data found in MVG file")
        
    data = np.array(data)
    
    # Extract columns
    azimuth = np.rad2deg(data[:, 0])    # Azimuth (corresponds to phi in spherical coordinates)
    elevation = np.rad2deg(data[:, 1])  # Elevation (corresponds to theta in spherical coordinates) 
    frequency = data[:, 2]  # Frequency
    e_theta_real = data[:, 3]  # ETheta real part
    e_theta_imag = data[:, 4]  # ETheta imaginary part  
    e_phi_real = data[:, 5]    # EPhi real part
    e_phi_imag = data[:, 6]    # EPhi imaginary part
    
    # Get unique values and ranges
    unique_theta = np.unique(elevation)  # Elevation becomes theta
    unique_phi = np.unique(azimuth)      # Azimuth becomes phi
    unique_freq = np.unique(frequency)
    
    # Sort the angles
    unique_theta = np.sort(unique_theta)
    unique_phi = np.sort(unique_phi)
    
    print(f"Theta range: {unique_theta.min():.1f} to {unique_theta.max():.1f} degrees ({len(unique_theta)} points)")
    print(f"Phi range: {unique_phi.min():.1f} to {unique_phi.max():.1f} degrees ({len(unique_phi)} points)")
    print(f"Frequencies: {len(unique_freq)} ({unique_freq[0] if len(unique_freq) > 0 else 'N/A'} Hz)")
    
    # Write FFD file
    with open(ffd_file, 'w') as file:
        # Write theta range: min max num_points
        file.write(f"{unique_theta.min():.0f} {unique_theta.max():.0f} {len(unique_theta)}\n")
        
        # Write phi range: min max num_points  
        file.write(f"{unique_phi.min():.0f} {unique_phi.max():.0f} {len(unique_phi)}\n")
        
        # Write frequency information
        file.write(f"Frequencies {len(unique_freq)}\n")
        for freq in unique_freq:
            file.write(f"Frequency {freq:.15e}\n")
        
        # Write the field data
        # FFD format expects data organized by theta (elevation) then phi (azimuth)
        for freq_val in unique_freq:
            for theta_val in unique_theta:
                for phi_val in unique_phi:
                    # Find the data point that matches this theta, phi, frequency
                    mask = (np.abs(elevation - theta_val) < 1e-6) & \
                           (np.abs(azimuth - phi_val) < 1e-6) & \
                           (np.abs(frequency - freq_val) < 1e-6)
                    
                    if np.any(mask):
                        idx = np.where(mask)[0][0]
                        # Write: ETheta_Real ETheta_Imag EPhi_Real EPhi_Imag
                        file.write(f"{e_theta_real[idx]:.15e} {e_theta_imag[idx]:.15e} {e_phi_real[idx]:.15e} {e_phi_imag[idx]:.15e}\n")
                    else:
                        # If no data point found, write zeros
                        file.write("0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00\n")

    print(f"Successfully converted {mvg_file} to {ffd_file}")
    print(f"Output file contains {len(unique_theta) * len(unique_phi) * len(unique_freq)} data points")


if __name__ == "__main__":
    # Example usage
    mvg_file = "C:/Users/asligar/OneDrive - ANSYS, Inc/Documents/Scripting/mvg_dipole_2450.txt"
    ffd_file = "dipole_2450_converted.ffd"
    
    if not os.path.exists(mvg_file):
        print(f"Error: The file {mvg_file} does not exist.")
        print("Please update the mvg_file path to point to your actual MVG file.")
    else:
        try:
            convert_mvg_to_ffd(mvg_file, ffd_file)
        except Exception as e:
            print(f"Error during conversion: {e}")
            import traceback
            traceback.print_exc()