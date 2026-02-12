# -*- coding: utf-8 -*-
"""
Scattering Centers Extraction Module

This module provides tools for extracting and analyzing scattering centers from 3D RCS (Radar Cross Section) data.
It includes peak detection algorithms, spatial localization methods, clustering techniques, and visualization tools.

The module can work with RCS data from the Perceive EM framework or from external sources.

Example usage:
    # Extract scattering centers from RCS simulation results
    extractor = ScatteringCentersExtractor()
    centers = extractor.extract_from_rcs_data(rcs_data)
    
    # Visualize results
    extractor.plot_scattering_centers_3d(centers)
    
    # Export results
    extractor.export_to_csv(centers, 'scattering_centers.csv')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
import scipy.spatial
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
import pyvista as pv
from tqdm import tqdm
import warnings

class ScatteringCentersExtractor:
    """
    A class for extracting scattering centers from 3D RCS data.
    
    This class provides comprehensive tools for analyzing radar cross section data
    to identify and characterize dominant scattering mechanisms.
    """
    
    def __init__(self, wavelength=None, frequency=None):
        """
        Initialize the scattering centers extractor.
        
        Args:
            wavelength (float): Operating wavelength in meters
            frequency (float): Operating frequency in Hz (alternative to wavelength)
        """
        if frequency is not None:
            self.frequency = frequency
            self.wavelength = 3e8 / frequency
        elif wavelength is not None:
            self.wavelength = wavelength
            self.frequency = 3e8 / wavelength
        else:
            # Default to 10 GHz
            self.frequency = 10e9
            self.wavelength = 3e8 / self.frequency
            
        self.scattering_centers = None
        self.rcs_data = None
        
    def extract_from_rcs_object(self, rcs_obj, freq_idx=0, threshold_db=-20, 
                               min_separation_deg=5, clustering=True):
        """
        Extract scattering centers from an RCS object from pem_utilities.
        
        Args:
            rcs_obj: RCS object from pem_utilities.rcs
            freq_idx (int): Frequency index to analyze
            threshold_db (float): Minimum RCS threshold in dB relative to peak
            min_separation_deg (float): Minimum angular separation for peaks
            clustering (bool): Whether to apply spatial clustering
            
        Returns:
            pandas.DataFrame: Scattering centers with positions and properties
        """
        if not hasattr(rcs_obj, 'results_df') or rcs_obj.results_df.empty:
            raise ValueError("RCS object has no simulation results. Run simulation first.")
            
        # Get RCS data
        data = rcs_obj.get_rcs_data(freq_idx=freq_idx, function='dB', return_format='dataframe')
        
        return self.extract_from_dataframe(data, threshold_db, min_separation_deg, clustering)
    
    def extract_from_dataframe(self, rcs_df, threshold_db=-20, min_separation_deg=5, 
                              clustering=True):
        """
        Extract scattering centers from RCS DataFrame.
        
        Args:
            rcs_df (pandas.DataFrame): DataFrame with columns 'theta', 'phi', 'rcs_data'
            threshold_db (float): Minimum RCS threshold in dB relative to peak
            min_separation_deg (float): Minimum angular separation for peaks
            clustering (bool): Whether to apply spatial clustering
            
        Returns:
            pandas.DataFrame: Scattering centers with positions and properties
        """
        scattering_centers = []
        
        print("Extracting scattering centers from RCS data...")
        
        # Convert RCS data to dB if not already
        for idx, row in tqdm(rcs_df.iterrows(), total=len(rcs_df), desc="Processing RCS data"):
            theta = row['theta']
            phi = row['phi']
            rcs_data = row['rcs_data']
            
            # Ensure RCS data is in dB
            if np.max(rcs_data) > 100:  # Likely linear scale
                rcs_db = 10 * np.log10(np.maximum(rcs_data, 1e-10))
            else:
                rcs_db = rcs_data
            
            # Find peaks above threshold
            max_rcs = np.max(rcs_db)
            threshold = max_rcs + threshold_db
            
            if max_rcs > threshold:
                # Store the peak
                peak_idx = np.argmax(rcs_db)
                center = {
                    'theta_deg': theta,
                    'phi_deg': phi,
                    'rcs_db': max_rcs,
                    'freq_idx': peak_idx if isinstance(rcs_db, np.ndarray) else 0,
                    'x': np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                    'y': np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
                    'z': np.cos(np.deg2rad(theta))
                }
                scattering_centers.append(center)
        
        if not scattering_centers:
            print("No scattering centers found above threshold.")
            return pd.DataFrame()
        
        centers_df = pd.DataFrame(scattering_centers)
        
        # Apply angular separation filtering
        centers_df = self._filter_by_angular_separation(centers_df, min_separation_deg)
        
        # Apply clustering if requested
        if clustering and len(centers_df) > 1:
            centers_df = self._apply_clustering(centers_df)
        
        # Convert to Cartesian coordinates for easier processing
        centers_df = self._add_cartesian_coordinates(centers_df)
        
        self.scattering_centers = centers_df
        print(f"Extracted {len(centers_df)} scattering centers.")
        
        return centers_df
    
    def extract_from_spherical_grid(self, theta_deg, phi_deg, rcs_data, 
                                   threshold_db=-20, min_separation_deg=5, 
                                   clustering=True):
        """
        Extract scattering centers from spherical grid data.
        
        Args:
            theta_deg (array): Theta angles in degrees
            phi_deg (array): Phi angles in degrees  
            rcs_data (array): RCS data array (theta x phi) or (theta x phi x freq)
            threshold_db (float): Minimum RCS threshold in dB relative to peak
            min_separation_deg (float): Minimum angular separation for peaks
            clustering (bool): Whether to apply spatial clustering
            
        Returns:
            pandas.DataFrame: Scattering centers with positions and properties
        """
        print("Extracting scattering centers from spherical grid...")
        
        # Ensure RCS data is in dB
        if np.max(rcs_data) > 100:  # Likely linear scale
            rcs_db = 10 * np.log10(np.maximum(rcs_data, 1e-10))
        else:
            rcs_db = rcs_data
        
        # Handle different data dimensions
        if rcs_db.ndim == 3:
            # Use first frequency if 3D
            rcs_db = rcs_db[:, :, 0]
        elif rcs_db.ndim != 2:
            raise ValueError("RCS data must be 2D (theta x phi) or 3D (theta x phi x freq)")
        
        scattering_centers = []
        
        # Find local maxima
        max_rcs = np.max(rcs_db)
        threshold = max_rcs + threshold_db
        
        # Use scipy to find peaks
        from scipy.ndimage import maximum_filter
        local_maxima = maximum_filter(rcs_db, size=3) == rcs_db
        
        # Get coordinates of peaks above threshold
        peak_coords = np.where((local_maxima) & (rcs_db > threshold))
        
        for i in range(len(peak_coords[0])):
            theta_idx = peak_coords[0][i]
            phi_idx = peak_coords[1][i]
            
            center = {
                'theta_deg': theta_deg[theta_idx],
                'phi_deg': phi_deg[phi_idx],
                'rcs_db': rcs_db[theta_idx, phi_idx],
                'theta_idx': theta_idx,
                'phi_idx': phi_idx,
            }
            scattering_centers.append(center)
        
        if not scattering_centers:
            print("No scattering centers found above threshold.")
            return pd.DataFrame()
        
        centers_df = pd.DataFrame(scattering_centers)
        
        # Apply angular separation filtering
        centers_df = self._filter_by_angular_separation(centers_df, min_separation_deg)
        
        # Apply clustering if requested
        if clustering and len(centers_df) > 1:
            centers_df = self._apply_clustering(centers_df)
        
        # Add Cartesian coordinates
        centers_df = self._add_cartesian_coordinates(centers_df)
        
        self.scattering_centers = centers_df
        print(f"Extracted {len(centers_df)} scattering centers.")
        
        return centers_df
    
    def _filter_by_angular_separation(self, centers_df, min_separation_deg):
        """Filter scattering centers by minimum angular separation."""
        if len(centers_df) <= 1:
            return centers_df
        
        # Sort by RCS strength (highest first)
        centers_df = centers_df.sort_values('rcs_db', ascending=False).reset_index(drop=True)
        
        filtered_centers = []
        
        for idx, center in centers_df.iterrows():
            # Check angular separation from all previously accepted centers
            keep_center = True
            
            for accepted_center in filtered_centers:
                # Calculate angular separation
                theta1, phi1 = np.deg2rad([center['theta_deg'], center['phi_deg']])
                theta2, phi2 = np.deg2rad([accepted_center['theta_deg'], accepted_center['phi_deg']])
                
                # Spherical distance
                angular_sep = np.arccos(np.clip(
                    np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) + 
                    np.cos(theta1) * np.cos(theta2), -1, 1
                ))
                
                if np.rad2deg(angular_sep) < min_separation_deg:
                    keep_center = False
                    break
            
            if keep_center:
                filtered_centers.append(center)
        
        return pd.DataFrame(filtered_centers)
    
    def _apply_clustering(self, centers_df, eps_deg=10, min_samples=2):
        """Apply DBSCAN clustering to group nearby scattering centers."""
        if len(centers_df) < min_samples:
            centers_df['cluster'] = 0
            return centers_df
        
        # Convert to Cartesian coordinates for clustering
        positions = centers_df[['x', 'y', 'z']].values if 'x' in centers_df.columns else \
                   self._spherical_to_cartesian(centers_df['theta_deg'].values, 
                                               centers_df['phi_deg'].values)
        
        # Apply DBSCAN clustering
        eps_cartesian = 2 * np.sin(np.deg2rad(eps_deg / 2))  # Convert angular to Cartesian distance
        clustering = DBSCAN(eps=eps_cartesian, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(positions)
        
        centers_df['cluster'] = cluster_labels
        
        return centers_df
    
    def _add_cartesian_coordinates(self, centers_df):
        """Add Cartesian coordinates to the centers DataFrame."""
        if 'x' not in centers_df.columns:
            x, y, z = self._spherical_to_cartesian(centers_df['theta_deg'].values, 
                                                  centers_df['phi_deg'].values)
            centers_df['x'] = x
            centers_df['y'] = y
            centers_df['z'] = z
        
        return centers_df
    
    def _spherical_to_cartesian(self, theta_deg, phi_deg, r=1):
        """Convert spherical to Cartesian coordinates."""
        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)
        
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad)
        
        return x, y, z
    
    def characterize_scattering_centers(self, centers_df=None):
        """
        Characterize scattering centers with additional properties.
        
        Args:
            centers_df (pandas.DataFrame): Scattering centers data
            
        Returns:
            pandas.DataFrame: Enhanced scattering centers with characteristics
        """
        if centers_df is None:
            centers_df = self.scattering_centers
        
        if centers_df is None or centers_df.empty:
            raise ValueError("No scattering centers data available.")
        
        enhanced_centers = centers_df.copy()
        
        # Calculate relative strength
        max_rcs = enhanced_centers['rcs_db'].max()
        enhanced_centers['relative_strength_db'] = enhanced_centers['rcs_db'] - max_rcs
        
        # Classify by strength
        enhanced_centers['strength_category'] = pd.cut(
            enhanced_centers['relative_strength_db'],
            bins=[-np.inf, -20, -10, -3, 0],
            labels=['Weak', 'Moderate', 'Strong', 'Dominant']
        )
        
        # Estimate physical characteristics
        enhanced_centers['estimated_size_wavelengths'] = np.sqrt(
            10**(enhanced_centers['rcs_db']/10) / (4 * np.pi)
        ) / self.wavelength
        
        return enhanced_centers
    
    def plot_scattering_centers_3d(self, centers_df=None, show_sphere=True, 
                                  color_by='rcs_db', figsize=(12, 9)):
        """
        Create 3D visualization of scattering centers.
        
        Args:
            centers_df (pandas.DataFrame): Scattering centers data
            show_sphere (bool): Whether to show reference sphere
            color_by (str): Column to use for coloring
            figsize (tuple): Figure size
        """
        if centers_df is None:
            centers_df = self.scattering_centers
        
        if centers_df is None or centers_df.empty:
            raise ValueError("No scattering centers data available.")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot scattering centers
        scatter = ax.scatter(centers_df['x'], centers_df['y'], centers_df['z'], 
                           c=centers_df[color_by], s=50, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        plt.colorbar(scatter, label=color_by)
        
        # Show reference sphere if requested
        if show_sphere:
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Scattering Centers 3D View\n({len(centers_df)} centers)')
        
        # Equal aspect ratio
        max_range = 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        plt.tight_layout()
        plt.show()
    
    def plot_scattering_centers_spherical(self, centers_df=None, projection='mollweide'):
        """
        Create spherical projection plot of scattering centers.
        
        Args:
            centers_df (pandas.DataFrame): Scattering centers data
            projection (str): Map projection type
        """
        if centers_df is None:
            centers_df = self.scattering_centers
        
        if centers_df is None or centers_df.empty:
            raise ValueError("No scattering centers data available.")
        
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': projection})
        
        # Convert to longitude/latitude for plotting
        lon = np.deg2rad(centers_df['phi_deg'])
        lat = np.deg2rad(90 - centers_df['theta_deg'])  # Convert theta to latitude
        
        scatter = ax.scatter(lon, lat, c=centers_df['rcs_db'], s=50, 
                           cmap='viridis', alpha=0.8)
        
        plt.colorbar(scatter, label='RCS (dB)')
        ax.set_title(f'Scattering Centers Spherical View\n({len(centers_df)} centers)')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_rcs_pattern_with_centers(self, theta_deg, phi_deg, rcs_data, 
                                     centers_df=None, freq_idx=0):
        """
        Plot RCS pattern with scattering centers overlay.
        
        Args:
            theta_deg (array): Theta angles
            phi_deg (array): Phi angles
            rcs_data (array): RCS data
            centers_df (pandas.DataFrame): Scattering centers
            freq_idx (int): Frequency index for 3D data
        """
        if centers_df is None:
            centers_df = self.scattering_centers
        
        # Handle 3D data
        if rcs_data.ndim == 3:
            rcs_plot = rcs_data[:, :, freq_idx]
        else:
            rcs_plot = rcs_data
        
        # Convert to dB if needed
        if np.max(rcs_plot) > 100:
            rcs_plot = 10 * np.log10(np.maximum(rcs_plot, 1e-10))
        
        # Create meshgrid for plotting
        PHI, THETA = np.meshgrid(phi_deg, theta_deg)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot RCS pattern
        im = ax.contourf(PHI, THETA, rcs_plot, levels=50, cmap='viridis')
        plt.colorbar(im, label='RCS (dB)')
        
        # Overlay scattering centers
        if centers_df is not None and not centers_df.empty:
            ax.scatter(centers_df['phi_deg'], centers_df['theta_deg'], 
                      c='red', s=100, marker='x', linewidths=3, 
                      label=f'Scattering Centers ({len(centers_df)})')
            ax.legend()
        
        ax.set_xlabel('Phi (degrees)')
        ax.set_ylabel('Theta (degrees)')
        ax.set_title('RCS Pattern with Scattering Centers')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_to_csv(self, centers_df=None, filename='scattering_centers.csv'):
        """Export scattering centers to CSV file."""
        if centers_df is None:
            centers_df = self.scattering_centers
        
        if centers_df is None or centers_df.empty:
            raise ValueError("No scattering centers data available.")
        
        centers_df.to_csv(filename, index=False)
        print(f"Scattering centers exported to {filename}")
    
    def generate_summary_report(self, centers_df=None):
        """Generate a summary report of scattering centers."""
        if centers_df is None:
            centers_df = self.scattering_centers
        
        if centers_df is None or centers_df.empty:
            print("No scattering centers data available.")
            return
        
        print("=== SCATTERING CENTERS SUMMARY REPORT ===")
        print(f"Total number of scattering centers: {len(centers_df)}")
        print(f"Operating frequency: {self.frequency/1e9:.2f} GHz")
        print(f"Operating wavelength: {self.wavelength*1000:.2f} mm")
        print()
        
        print("RCS Statistics:")
        print(f"  Maximum RCS: {centers_df['rcs_db'].max():.2f} dBsm")
        print(f"  Minimum RCS: {centers_df['rcs_db'].min():.2f} dBsm")
        print(f"  Mean RCS: {centers_df['rcs_db'].mean():.2f} dBsm")
        print(f"  RCS Standard deviation: {centers_df['rcs_db'].std():.2f} dB")
        print()
        
        print("Angular Distribution:")
        print(f"  Theta range: {centers_df['theta_deg'].min():.1f}° to {centers_df['theta_deg'].max():.1f}°")
        print(f"  Phi range: {centers_df['phi_deg'].min():.1f}° to {centers_df['phi_deg'].max():.1f}°")
        print()
        
        if 'strength_category' in centers_df.columns:
            print("Strength Categories:")
            category_counts = centers_df['strength_category'].value_counts()
            for category, count in category_counts.items():
                print(f"  {category}: {count} centers")
            print()
        
        if 'cluster' in centers_df.columns:
            n_clusters = len(centers_df['cluster'].unique())
            print(f"Number of clusters: {n_clusters}")
            if n_clusters > 1:
                cluster_counts = centers_df['cluster'].value_counts()
                print("Cluster distribution:")
                for cluster_id, count in cluster_counts.items():
                    if cluster_id != -1:  # Exclude noise points
                        print(f"  Cluster {cluster_id}: {count} centers")
                if -1 in cluster_counts:
                    print(f"  Noise points: {cluster_counts[-1]} centers")
        
        print("="*50)


def example_usage_with_rcs_object():
    """
    Example of how to use the ScatteringCentersExtractor with an RCS simulation.
    """
    try:
        from pem_utilities.rcs import RCS
        from pem_utilities.path_helper import get_repo_paths
        from pem_utilities.actor import Actors
        
        print("Setting up RCS simulation for scattering centers extraction...")
        
        # Setup basic RCS simulation
        all_actors = Actors()
        rcs = RCS(all_actors=all_actors, rcs_mode='monostatic', rayshoot_method='grid')
        
        # Configure RCS parameters
        rcs.center_freq = 10e9
        rcs.phi_start = 0
        rcs.phi_stop = 360
        rcs.phi_step_deg = 5
        rcs.theta_start = 45
        rcs.theta_stop = 135
        rcs.theta_step_deg = 5
        
        # Add a simple target (would need actual geometry in practice)
        # This is just an example structure
        
        print("Running RCS simulation...")
        # rcs.run_simulation()  # Uncomment when you have actual geometry
        
        print("Extracting scattering centers...")
        extractor = ScatteringCentersExtractor(frequency=rcs.center_freq)
        
        # Extract scattering centers
        # centers = extractor.extract_from_rcs_object(rcs, threshold_db=-15)
        
        # Generate report and visualizations
        # extractor.generate_summary_report(centers)
        # extractor.plot_scattering_centers_3d(centers)
        
        print("Example setup complete. Uncomment simulation lines when geometry is available.")
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("This example requires the full Perceive EM framework.")


def example_usage_with_synthetic_data():
    """
    Example of how to use the ScatteringCentersExtractor with synthetic data.
    """
    print("Generating synthetic RCS data for demonstration...")
    
    # Create synthetic spherical grid
    theta_deg = np.linspace(0, 180, 91)
    phi_deg = np.linspace(0, 360, 181)
    
    THETA, PHI = np.meshgrid(theta_deg, phi_deg, indexing='ij')
    
    # Create synthetic RCS pattern with multiple scattering centers
    rcs_linear = np.zeros_like(THETA)
    
    # Add some dominant scattering centers
    centers_true = [
        (45, 0, 100),    # (theta, phi, RCS in m²)
        (90, 90, 50),
        (135, 180, 25),
        (60, 270, 75),
    ]
    
    for theta_c, phi_c, rcs_val in centers_true:
        # Create Gaussian-like pattern around each center
        angular_dist = np.sqrt((THETA - theta_c)**2 + (PHI - phi_c)**2)
        rcs_linear += rcs_val * np.exp(-angular_dist**2 / (2 * 15**2))
    
    # Add some noise
    rcs_linear += np.random.exponential(1, rcs_linear.shape)
    
    print("Extracting scattering centers...")
    
    # Initialize extractor
    extractor = ScatteringCentersExtractor(frequency=10e9)
    
    # Extract scattering centers
    centers = extractor.extract_from_spherical_grid(
        theta_deg, phi_deg, rcs_linear,
        threshold_db=-15,
        min_separation_deg=10,
        clustering=True
    )
    
    # Characterize the centers
    centers_enhanced = extractor.characterize_scattering_centers(centers)
    
    print("\nExtraction complete!")
    
    # Generate summary report
    extractor.generate_summary_report(centers_enhanced)
    
    # Create visualizations
    extractor.plot_rcs_pattern_with_centers(theta_deg, phi_deg, rcs_linear, centers_enhanced)
    extractor.plot_scattering_centers_3d(centers_enhanced)
    extractor.plot_scattering_centers_spherical(centers_enhanced)
    
    # Export results
    extractor.export_to_csv(centers_enhanced, 'example_scattering_centers.csv')
    
    print(f"\nFound {len(centers_enhanced)} scattering centers")
    print("True centers were at:")
    for i, (theta_c, phi_c, rcs_val) in enumerate(centers_true):
        print(f"  Center {i+1}: θ={theta_c}°, φ={phi_c}°, RCS={10*np.log10(rcs_val):.1f} dBsm")


if __name__ == "__main__":
    print("Scattering Centers Extraction Tool")
    print("="*50)
    
    # Run the synthetic data example
    example_usage_with_synthetic_data()
    
    print("\nFor RCS object integration:")
    example_usage_with_rcs_object()