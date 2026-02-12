#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Script: Extract Scattering Centers from 3D RCS Data

This script demonstrates how to use the ScatteringCentersExtractor module
with the Perceive EM framework to extract and analyze scattering centers
from radar cross section simulations.

Usage:
    python example_scattering_centers.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the pem_utilities to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pem_utilities'))

from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API
from pem_utilities.actor import Actors
from pem_utilities.rcs import RCS
from pem_utilities.scattering_centers import ScatteringCentersExtractor
from pem_utilities.materials import MaterialManager
from pem_utilities.rotation import euler_to_rot

def run_rcs_simulation_example():
    """
    Run a complete example with RCS simulation and scattering centers extraction.
    """
    print("=== RCS Simulation & Scattering Centers Extraction Example ===")
    
    # Get repository paths
    paths = get_repo_paths()
    
    # Initialize Perceive EM API
    pem_api_manager = Perceive_EM_API()
    pem = pem_api_manager.pem
    RssPy = pem_api_manager.RssPy
    
    # Create actor collection
    all_actors = Actors()
    
    # Initialize material manager
    mat_manager = MaterialManager()
    
    print("Setting up RCS simulation...")
    
    # Add a target geometry (using an example STL file if available)
    target_files = ['t72.stl', 'explorer_ship_meter.stl', 'Corner_Reflector.stl']
    target_file = None
    
    for filename in target_files:
        full_path = paths.models / filename
        if full_path.exists():
            target_file = str(full_path)
            print(f"Using target geometry: {filename}")
            break
    
    if target_file:
        # Add the target to the scene
        actor_target_name = all_actors.add_actor(
            filename=target_file,
            target_ray_spacing=0.1,
            scale_mesh=1.0
        )
        
        all_actors.actors[actor_target_name].coord_sys.pos = [0, 0, 0]
        all_actors.actors[actor_target_name].coord_sys.rot = np.eye(3)
        all_actors.actors[actor_target_name].coord_sys.update()
    else:
        print("No target geometry found. Using synthetic data example instead.")
        run_synthetic_data_example()
        return
    
    # Initialize RCS simulation
    rcs = RCS(all_actors=all_actors, rcs_mode='monostatic', rayshoot_method='grid')
    
    # Configure RCS parameters
    rcs.center_freq = 10e9
    rcs.num_freqs = 3
    rcs.bandwidth = 300e6
    rcs.polarization = 'VV'
    
    # Set angular sampling (reduced for faster computation)
    rcs.phi_start = 0
    rcs.phi_stop = 360
    rcs.phi_step_deg = 10  # 10 degree steps for faster computation
    
    rcs.theta_start = 30
    rcs.theta_stop = 150
    rcs.theta_step_deg = 10  # 10 degree steps for faster computation
    
    # Simulation settings
    rcs.go_blockage = -1
    rcs.max_num_refl = 3
    rcs.max_num_trans = 0
    rcs.ray_density = 0.5
    
    rcs.output_path = str(paths.output)
    
    try:
        print("Running RCS simulation...")
        rcs.run_simulation(show_modeler=False)
        
        print("RCS simulation complete!")
        rcs.print_timing_summary()
        
        # Extract scattering centers
        print("\nExtracting scattering centers...")
        
        extractor = ScatteringCentersExtractor(frequency=rcs.center_freq)
        
        # Extract scattering centers from RCS results
        centers = extractor.extract_from_rcs_object(
            rcs,
            freq_idx=0,  # Use center frequency
            threshold_db=-15,  # 15 dB below peak
            min_separation_deg=15,  # Minimum 15 degree separation
            clustering=True
        )
        
        if centers.empty:
            print("No scattering centers found!")
            return
        
        # Characterize the centers
        centers_enhanced = extractor.characterize_scattering_centers(centers)
        
        # Generate summary report
        extractor.generate_summary_report(centers_enhanced)
        
        # Create visualizations
        print("Creating visualizations...")
        
        # 3D plot of scattering centers
        extractor.plot_scattering_centers_3d(centers_enhanced, show_sphere=True)
        
        # Spherical projection plot
        extractor.plot_scattering_centers_spherical(centers_enhanced)
        
        # Export results
        output_file = paths.output / 'scattering_centers_analysis.csv'
        extractor.export_to_csv(centers_enhanced, str(output_file))
        
        print(f"\nAnalysis complete! Results saved to: {output_file}")
        
        return centers_enhanced
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("Running synthetic data example instead...")
        run_synthetic_data_example()


def run_synthetic_data_example():
    """
    Run example with synthetic RCS data to demonstrate the extraction capabilities.
    """
    print("\n=== Synthetic Data Scattering Centers Example ===")
    
    # Create synthetic spherical grid with higher resolution
    theta_deg = np.linspace(10, 170, 81)  # Avoid poles for better visualization
    phi_deg = np.linspace(0, 360, 181)
    
    THETA, PHI = np.meshgrid(theta_deg, phi_deg, indexing='ij')
    
    # Create synthetic RCS pattern with multiple scattering centers
    rcs_linear = np.zeros_like(THETA)
    
    print("Creating synthetic target with multiple scattering centers...")
    
    # Add dominant scattering centers representing different parts of a complex target
    scattering_mechanisms = [
        # (theta, phi, RCS_m², width_deg, description)
        (45, 0, 1000, 8, "Nose/front face specular reflection"),
        (90, 90, 500, 12, "Side panel specular reflection"), 
        (135, 180, 200, 6, "Tail/rear corner reflector"),
        (60, 270, 800, 10, "Wing/fin edge diffraction"),
        (120, 45, 150, 15, "Hull/body surface creeping wave"),
        (75, 315, 300, 8, "Engine inlet/cavity resonance"),
    ]
    
    for theta_c, phi_c, rcs_val, width, desc in scattering_mechanisms:
        # Create realistic scattering pattern around each center
        angular_dist = np.sqrt((THETA - theta_c)**2 + (PHI - phi_c)**2)
        
        # Use different pattern shapes for different mechanisms
        if "specular" in desc:
            # Sharp specular reflection
            pattern = rcs_val * np.exp(-angular_dist**2 / (2 * (width/2)**2))
        elif "corner" in desc:
            # Corner reflector with some sidelobes
            pattern = rcs_val * np.exp(-angular_dist**2 / (2 * width**2))
            pattern += 0.1 * rcs_val * np.exp(-angular_dist**2 / (2 * (width*2)**2))
        elif "edge" in desc:
            # Edge diffraction with wider pattern
            pattern = rcs_val * np.exp(-angular_dist**1.5 / (2 * width**1.5))
        else:
            # General scattering
            pattern = rcs_val * np.exp(-angular_dist**2 / (2 * width**2))
        
        rcs_linear += pattern
    
    # Add background clutter and noise
    rcs_linear += np.random.exponential(2, rcs_linear.shape)  # Exponential noise
    rcs_linear += 10 * np.random.rayleigh(1, rcs_linear.shape)  # Rayleigh clutter
    
    print("Extracting scattering centers from synthetic data...")
    
    # Initialize extractor
    extractor = ScatteringCentersExtractor(frequency=10e9)
    
    # Extract scattering centers with different threshold settings
    print("\n--- Extracting with strict threshold (-10 dB) ---")
    centers_strict = extractor.extract_from_spherical_grid(
        theta_deg, phi_deg, rcs_linear,
        threshold_db=-10,  # Stricter threshold
        min_separation_deg=8,
        clustering=True
    )
    
    print("\n--- Extracting with relaxed threshold (-20 dB) ---")
    centers_relaxed = extractor.extract_from_spherical_grid(
        theta_deg, phi_deg, rcs_linear,
        threshold_db=-20,  # More relaxed threshold
        min_separation_deg=8,
        clustering=True
    )
    
    # Use the strict threshold results for detailed analysis
    if not centers_strict.empty:
        print("\n=== Analysis with Strict Threshold ===")
        
        # Characterize the centers
        centers_enhanced = extractor.characterize_scattering_centers(centers_strict)
        
        # Generate summary report
        extractor.generate_summary_report(centers_enhanced)
        
        print("\n=== Creating Visualizations ===")
        
        # Plot RCS pattern with overlaid scattering centers
        extractor.plot_rcs_pattern_with_centers(theta_deg, phi_deg, rcs_linear, centers_enhanced)
        
        # 3D visualization
        extractor.plot_scattering_centers_3d(centers_enhanced, show_sphere=True)
        
        # Spherical projection
        extractor.plot_scattering_centers_spherical(centers_enhanced)
        
        # Export results
        extractor.export_to_csv(centers_enhanced, 'synthetic_scattering_centers.csv')
        
        print(f"\nFound {len(centers_enhanced)} scattering centers")
        print("\nOriginal scattering mechanisms were:")
        for i, (theta_c, phi_c, rcs_val, width, desc) in enumerate(scattering_mechanisms):
            print(f"  {i+1}: θ={theta_c:3.0f}°, φ={phi_c:3.0f}°, "
                  f"RCS={10*np.log10(rcs_val):5.1f} dBsm - {desc}")
        
        print(f"\nExtracted centers:")
        for i, row in centers_enhanced.iterrows():
            print(f"  {i+1}: θ={row['theta_deg']:5.1f}°, φ={row['phi_deg']:5.1f}°, "
                  f"RCS={row['rcs_db']:5.1f} dBsm - {row['strength_category']}")
        
        return centers_enhanced
    
    else:
        print("No scattering centers found with strict threshold!")
        return None


def analyze_frequency_dependence():
    """
    Demonstrate frequency-dependent analysis of scattering centers.
    """
    print("\n=== Frequency Dependence Analysis ===")
    
    frequencies = [2e9, 5e9, 10e9, 15e9, 20e9]  # 2-20 GHz
    
    # Create synthetic multi-frequency RCS data
    theta_deg = np.linspace(30, 150, 61)
    phi_deg = np.linspace(0, 360, 91)
    
    THETA, PHI = np.meshgrid(theta_deg, phi_deg, indexing='ij')
    
    all_frequency_centers = []
    
    for freq in frequencies:
        print(f"\nAnalyzing at {freq/1e9:.1f} GHz...")
        
        # Create frequency-dependent RCS pattern
        wavelength = 3e8 / freq
        
        # Some scattering mechanisms are frequency dependent
        rcs_linear = np.zeros_like(THETA)
        
        # Large surface - frequency independent
        angular_dist = np.sqrt((THETA - 60)**2 + (PHI - 45)**2)
        rcs_linear += 500 * np.exp(-angular_dist**2 / (2 * 8**2))
        
        # Resonant cavity - frequency dependent peak at 10 GHz
        resonance_factor = np.exp(-((freq - 10e9) / 3e9)**2)
        angular_dist = np.sqrt((THETA - 90)**2 + (PHI - 180)**2)
        rcs_linear += 1000 * resonance_factor * np.exp(-angular_dist**2 / (2 * 6**2))
        
        # Edge diffraction - frequency dependent (higher freq = more directional)
        edge_sharpness = freq / 5e9  # Sharper at higher frequencies
        angular_dist = np.sqrt((THETA - 120)**2 + (PHI - 270)**2)
        rcs_linear += 300 * np.exp(-angular_dist**2 / (2 * (10/edge_sharpness)**2))
        
        # Add noise
        rcs_linear += np.random.exponential(1, rcs_linear.shape)
        
        # Extract centers
        extractor = ScatteringCentersExtractor(frequency=freq)
        centers = extractor.extract_from_spherical_grid(
            theta_deg, phi_deg, rcs_linear,
            threshold_db=-15,
            min_separation_deg=10,
            clustering=False
        )
        
        if not centers.empty:
            centers['frequency_ghz'] = freq / 1e9
            all_frequency_centers.append(centers)
            print(f"  Found {len(centers)} scattering centers")
    
    if all_frequency_centers:
        import pandas as pd
        
        # Combine all frequency results
        combined_centers = pd.concat(all_frequency_centers, ignore_index=True)
        
        # Plot frequency dependence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: RCS vs frequency for each angular location
        for i, freq in enumerate(frequencies):
            freq_data = combined_centers[combined_centers['frequency_ghz'] == freq/1e9]
            ax1.scatter([freq/1e9] * len(freq_data), freq_data['rcs_db'], 
                       alpha=0.7, label=f'{freq/1e9:.1f} GHz' if i < 3 else "")
        
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('RCS (dBsm)')
        ax1.set_title('Scattering Centers vs Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Angular distribution at different frequencies
        for freq in [2e9, 10e9, 20e9]:
            freq_data = combined_centers[combined_centers['frequency_ghz'] == freq/1e9]
            if not freq_data.empty:
                ax2.scatter(freq_data['phi_deg'], freq_data['theta_deg'], 
                           s=50, alpha=0.7, label=f'{freq/1e9:.1f} GHz')
        
        ax2.set_xlabel('Phi (degrees)')
        ax2.set_ylabel('Theta (degrees)')
        ax2.set_title('Angular Distribution by Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Export combined results
        combined_centers.to_csv('frequency_dependent_scattering_centers.csv', index=False)
        print(f"\nFrequency analysis complete! Combined results saved.")
        
        return combined_centers
    
    return None


def main():
    """Main function to run all examples."""
    print("Scattering Centers Extraction Examples")
    print("="*50)
    
    try:
        # Try to run with actual RCS simulation first
        centers = run_rcs_simulation_example()
        
    except Exception as e:
        print(f"RCS simulation failed: {e}")
        print("Running synthetic examples instead...")
        
        # Run synthetic data example
        centers = run_synthetic_data_example()
        
        # Run frequency dependence analysis
        freq_centers = analyze_frequency_dependence()
    
    print("\n" + "="*50)
    print("All examples completed!")
    print("Check the generated plots and CSV files for results.")


if __name__ == "__main__":
    main()