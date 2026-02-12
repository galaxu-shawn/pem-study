import numpy as np
import pyvista as pv
from scipy.fftpack import fft2, ifft2
import os
import time as walltime

class OceanSurface:
    def __init__(self,
                 num_grid,
                 scene_length,
                 wind_speed,
                 wave_amplitude,
                 choppiness,
                 wind_direction=[1,0],
                 include_wake=False,
                 velocity_ship=10,
                 length_ship=110,
                 beam_ship=20.3,
                 draft_ship=3.5,
                 initial_wake_position=[0,0],
                 update_wake_position=True,
                 enable_swell=False,
                 swell_amplitude=0.0,
                 swell_wavelength=100.0,
                 swell_direction=[1,0],
                 swell_phase=0.0,
                 swell_frequency=None,
                 smooth=False):
        num_grid = int(num_grid)  # grid size
        self.num_grid = num_grid  # grid size
        self.scene_length = scene_length  # domain size
        self.wind_speed = wind_speed
        self.wind_direction = np.array(wind_direction) / np.linalg.norm(wind_direction)  # normalize wind direction
        self.wave_amplitude = wave_amplitude  # wave amplitude
        self.choppiness = choppiness  # choppiness factor
        self.smooth = smooth
        self.g = 9.81  # gravitational constant
        
        # Create frequency grid
        freq = np.fft.fftfreq(num_grid, scene_length / num_grid)
        self.kx, self.ky = np.meshgrid(freq * 2 * np.pi, freq * 2 * np.pi)
        self.k = np.sqrt(self.kx ** 2 + self.ky ** 2)
        self.k[0, 0] = 1e-8  # avoid division by zero at DC component

        # Generate initial random phases
        self.h0_plus, self.h0_minus = self.generate_h0()
        self.omega = np.sqrt(self.g * self.k)
        self.mesh = None
        self.time = 0

        self.include_wake = include_wake
        self.velocity_ship = velocity_ship
        self.length_ship = length_ship
        self.beam_ship = beam_ship
        self.draft_ship = draft_ship
        self.initial_wake_position = initial_wake_position
        self.update_wake_position = update_wake_position
        self.enable_swell = enable_swell
        self.swell_amplitude = swell_amplitude
        self.swell_wavelength = swell_wavelength
        self.swell_direction = np.array(swell_direction) / np.linalg.norm(swell_direction)
        self.swell_phase = swell_phase
        self.swell_frequency = swell_frequency  # If None, will be computed from wavelength

    def generate_h0(self):
        """Generate initial wave amplitudes with proper Phillips spectrum"""
        # Generate random Gaussian noise
        xi_r = np.random.normal(0, 1, (self.num_grid, self.num_grid))
        xi_i = np.random.normal(0, 1, (self.num_grid, self.num_grid))
        
        # Calculate Phillips spectrum
        phillips = self.phillips_spectrum()
        
        # Generate h0+ and h0- with proper normalization
        h0_plus = (xi_r + 1j * xi_i) * np.sqrt(phillips / 2)
        h0_minus = (xi_r - 1j * xi_i) * np.sqrt(phillips / 2)
        
        return h0_plus, h0_minus

    def phillips_spectrum(self):
        """Phillips spectrum for ocean waves"""
        # Avoid division by zero
        k_safe = np.where(self.k == 0, 1e-8, self.k)
        
        # Wind direction dot product with wave vector
        k_dot_w = (self.kx * self.wind_direction[0] + self.ky * self.wind_direction[1]) / k_safe
        k_dot_w = np.where(self.k == 0, 0, k_dot_w)
        
        # Phillips spectrum parameters
        L = self.wind_speed ** 2 / self.g  # largest possible wave
        l = L / 1000  # small wave cutoff
        
        # Phillips spectrum formula
        phillips = (self.wave_amplitude * np.exp(-1.0 / (k_safe * L) ** 2) / (k_safe ** 4) * 
                   np.abs(k_dot_w) ** 2 * np.exp(-k_safe ** 2 * l ** 2))
        
        # Set DC component to zero
        phillips[0, 0] = 0
        
        # Suppress waves going against the wind
        phillips = np.where(k_dot_w < 0, phillips * 0.07, phillips)
        
        return phillips

    def generate_height_field(self, t=None, sigma=1.0):
        """Generate height field at time t, normalized to match wave_amplitude peak-to-peak, with optional swell"""
        if t is None:
            t = self.time
        else:
            self.time = t
        # Time evolution of wave amplitudes
        h_tilde = (self.h0_plus * np.exp(1j * self.omega * t) + 
                   np.conj(self.h0_minus) * np.exp(-1j * self.omega * t))
        # Convert to real height field using inverse FFT
        height = np.real(np.fft.ifft2(h_tilde))
        # Normalize so that peak-to-peak matches wave_amplitude
        ptp = height.max() - height.min()
        if ptp > 0:
            height = (height - height.mean()) * (self.wave_amplitude / ptp)

        # Add swell component if enabled
        if self.enable_swell and self.swell_amplitude > 0.0:
            x = np.linspace(-self.scene_length/2, self.scene_length/2, self.num_grid)
            y = np.linspace(-self.scene_length/2, self.scene_length/2, self.num_grid)
            X, Y = np.meshgrid(x, y)
            swell_dir = self.swell_direction
            # Compute frequency if not set
            if self.swell_frequency is None:
                g = self.g
                swell_freq = np.sqrt(2 * np.pi * g / self.swell_wavelength)
            else:
                swell_freq = self.swell_frequency
            # Swell phase propagation
            phase = (2 * np.pi * (X * swell_dir[0] + Y * swell_dir[1]) / self.swell_wavelength)
            swell = self.swell_amplitude * np.sin(phase + self.swell_phase + swell_freq * t)
            height += swell
            
        # Apply Gaussian smoothing to the height data if requested
        if self.smooth:
            from scipy.ndimage import gaussian_filter
            height = gaussian_filter(height, sigma=sigma)
            
        return height

    def generate_displacement_field(self, t=None):
        """Generate horizontal displacement field for choppiness"""
        if t is None:
            t = self.time
        else:
            self.time = t
            
        # Time evolution of wave amplitudes
        h_tilde = (self.h0_plus * np.exp(1j * self.omega * t) + 
                   np.conj(self.h0_minus) * np.exp(-1j * self.omega * t))
        
        # Calculate displacement components
        k_safe = np.where(self.k == 0, 1e-8, self.k)
        dx_tilde = -1j * self.kx * h_tilde / k_safe
        dy_tilde = -1j * self.ky * h_tilde / k_safe
        
        # Set DC component to zero for displacements
        dx_tilde[0, 0] = 0
        dy_tilde[0, 0] = 0
        
        # Convert to real displacement fields
        dx = np.real(np.fft.ifft2(dx_tilde))
        dy = np.real(np.fft.ifft2(dy_tilde))

        return self.choppiness * dx, self.choppiness * dy

    def _generate_grid(self, t=None):
        if t is None:
            t = self.time
        else: # update time based on the input
            self.time = t
        # Fix: Use num_grid instead of num_grid+1 to match FFT array dimensions
        x = np.linspace(-self.scene_length/2, self.scene_length/2, self.num_grid)
        y = np.linspace(-self.scene_length/2, self.scene_length/2, self.num_grid)
        X, Y = np.meshgrid(x, y)

        height = self.generate_height_field(t)

        if self.include_wake:
            start_time = walltime.time()
            x_init = self.initial_wake_position[0]
            y_init = self.initial_wake_position[1]
            # currently only support ship moving in +Y direction
            wake_pos_x = x_init
            wake_pos_y = y_init
            if self.update_wake_position: #only update the position if True
                wake_pos_y = y_init + self.velocity_ship * t
            zship = Wake(velocity_ship=self.velocity_ship,
                         length_ship=self.length_ship,
                         beam_ship=self.beam_ship,
                         draft_ship=self.draft_ship,
                         scene_length=self.scene_length,
                         grid_spacing=self.scene_length/self.num_grid,
                         wake_position=[wake_pos_x,wake_pos_y],
                         num_grid=self.num_grid)  # Pass num_grid to Wake
            zship.calculate(time=t)
            height_wake = zship.z_ship
            end_time = walltime.time()
            # print(f"Time to calculate wake: {end_time - start_time}")
            # shift to the center (all wake starts at 0,0) for now
            # it will translate in Y direction with the ships velocity
            height += height_wake
        dx, dy = self.generate_displacement_field(t)

        X += dx
        Y += dy

        # Create the surface using PyVista
        grid = pv.StructuredGrid(X, Y, height)

        # Add the height data as a scalar field to the grid
        grid["Height"] = height.ravel('F')
        return grid

    def generate_mesh(self, t=None):
        
        grid = self._generate_grid(t=t)
        if self.smooth:
            grid = self.smooth_mesh(grid, smoothing_iterations=20)
        temp  = grid.extract_geometry()
        self.mesh = temp.triangulate()
        return self.mesh

    def save_mesh(self,output_path):
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)

    def smooth_mesh(self, mesh, smoothing_iterations=20):
            # Convert StructuredGrid to PolyData for smoothing
            surface = mesh.extract_geometry()
            # Apply Taubin smoothing (preserves features better than Laplacian)
            smoothed = surface.smooth_taubin(n_iter=smoothing_iterations, 
                                           pass_band=0.1, 
                                           feature_smoothing=True,
                                           boundary_smoothing=True,
                                           feature_angle=45.0,
                                           edge_angle=15.0,
                                           normalize_coordinates=True)
            return smoothed

    def create_ocean_surface_plot(self, t=None):
        grid = self._generate_grid(t=t)
        if self.smooth:
            return self.smooth_mesh(grid, smoothing_iterations=20)  
        return grid



class Wake:
    def __init__(self, velocity_ship=10, length_ship=110, beam_ship=20.3, draft_ship=3.5, 
                 scene_length=1000, grid_spacing=2.5, wake_position=[0,0], num_grid=None):
        """
        Simplified Gaussian Wake Model
        Creates a V-shaped wake pattern behind a moving ship using Gaussian distributions
        """
        self.velocity_ship = velocity_ship  # Ship velocity (m/s)
        self.length_ship = length_ship      # Ship length (m)
        self.beam_ship = beam_ship          # Ship beam/width (m)
        self.draft_ship = draft_ship        # Ship draft (m)
        self.scene_length = scene_length    # Scene size (m)
        self.grid_spacing = grid_spacing    # Grid spacing (m)
        self.wake_position = wake_position  # Ship position [x, y]
        self.num_grid = num_grid
        
        # Wake parameters
        self.kelvin_angle = 19.47 * np.pi / 180  # Kelvin wake half-angle in radians
        self.wake_length = min(scene_length * 0.8, velocity_ship * 30)  # Wake extends ~30 seconds behind
        self.wake_amplitude = self.calculate_wake_amplitude()
        
        # Initialize wake field
        self.z_ship = None

    def calculate_wake_amplitude(self):
        """Calculate wake amplitude based on ship parameters and Froude number"""
        g = 9.81
        froude_number = self.velocity_ship / np.sqrt(g * self.length_ship)
        
        # More physically accurate wake amplitude calculation
        # Based on ship displacement and energy considerations
        
        # Estimate ship displacement (simplified rectangular hull approximation)
        ship_displacement = self.length_ship * self.beam_ship * self.draft_ship * 0.7  # Block coefficient ~0.7
        
        # Wake amplitude scales with ship's waterplane area and velocity
        waterplane_area = self.length_ship * self.beam_ship
        
        # Base amplitude from ship size (typical values: 0.1-2.0m for large ships)
        # Scale with waterplane area relative to a reference ship (100m x 15m)
        reference_area = 100 * 15  # Reference ship waterplane area
        size_factor = np.sqrt(waterplane_area / reference_area)
        
        # Velocity contribution - wake energy scales with V²
        # Normalize by reference velocity of 10 m/s
        velocity_factor = (self.velocity_ship / 10.0) ** 2
        
        # Froude number effects
        if froude_number < 0.3:
            # Subcritical regime - smaller wakes
            froude_factor = froude_number / 0.3
        elif froude_number < 1.0:
            # Transitional regime - wake grows with Fr
            froude_factor = 1.0 + (froude_number - 0.3) * 2.0
        else:
            # Supercritical regime - wake amplitude plateaus
            froude_factor = 2.4 + 0.1 * (froude_number - 1.0)
        
        # Draft effect - deeper ships create larger wakes
        # But effect saturates for very deep drafts
        draft_factor = 1.0 + np.tanh(self.draft_ship / 5.0) * 0.5
        
        # Base amplitude for reference conditions (should give ~0.5m for typical ship)
        base_amplitude = 0.2  # meters
        
        # Combine all factors
        wake_amplitude = (base_amplitude * size_factor * velocity_factor * 
                         froude_factor * draft_factor)
        
        # Practical limits (real ship wakes rarely exceed 3-4 meters)
        wake_amplitude = min(wake_amplitude, 4.0)
        wake_amplitude = max(wake_amplitude, 0.01)  # Minimum wake
        
        return wake_amplitude

    def calculate(self, time=0):
        """Calculate the wake field using simplified Gaussian model"""
        # Create coordinate grids
        if self.num_grid is not None:
            x = np.linspace(-self.scene_length/2, self.scene_length/2, self.num_grid)
            y = np.linspace(-self.scene_length/2, self.scene_length/2, self.num_grid)
        else:
            dx = self.grid_spacing
            x = np.arange(-self.scene_length/2, self.scene_length/2 + dx, dx)
            y = np.arange(-self.scene_length/2, self.scene_length/2 + dx, dx)
        
        X, Y = np.meshgrid(x, y)
        
        # Ship position (ship moves in +Y direction)
        ship_x = self.wake_position[0]
        ship_y = self.wake_position[1]
        
        # Transform to ship-relative coordinates
        X_rel = X - ship_x
        Y_rel = Y - ship_y
        
        # Initialize wake field
        wake_field = np.zeros_like(X)
        
        # Create V-shaped wake pattern
        wake_field += self.create_kelvin_wake(X_rel, Y_rel)
        
        # Add bow wave (transverse wave at ship position)
        wake_field += self.create_bow_wave(X_rel, Y_rel)
        
        # Add turbulent wake (directly behind ship)
        wake_field += self.create_turbulent_wake(X_rel, Y_rel)
        
        self.z_ship = wake_field

    def create_kelvin_wake(self, X_rel, Y_rel):
        """Create the characteristic V-shaped Kelvin wake pattern"""
        wake = np.zeros_like(X_rel)
        
        # Only create wake behind the ship (Y_rel < 0)
        behind_ship = Y_rel < -self.length_ship/2
        
        if np.any(behind_ship):
            # Get coordinates for points behind the ship
            X_behind = X_rel[behind_ship]
            Y_behind = Y_rel[behind_ship]
            
            # Distance behind ship (positive values)
            y_distance = np.abs(Y_behind + self.length_ship/2)
            
            # Kelvin wake boundaries (V-shaped region)
            wake_half_width = y_distance * np.tan(self.kelvin_angle)
            
            # Create wake only within the V-shaped region
            in_wake_region = np.abs(X_behind) <= wake_half_width
            
            if np.any(in_wake_region):
                # Extract coordinates within wake region
                x_wake = X_behind[in_wake_region]
                y_wake = y_distance[in_wake_region]
                wake_width = wake_half_width[in_wake_region]
                
                # Transverse waves (perpendicular to ship motion)
                # Wave length scales with ship length and velocity
                transverse_wavelength = self.length_ship * 0.5
                transverse_frequency = 2 * np.pi / transverse_wavelength
                transverse_waves = np.sin(transverse_frequency * y_wake)
                
                # Divergent waves (along the V-pattern edges)
                # Higher frequency waves that form the wake boundaries
                divergent_wavelength = self.length_ship * 0.2
                divergent_frequency = 2 * np.pi / divergent_wavelength
                
                # Distance from wake centerline (normalized by wake width)
                normalized_x = np.abs(x_wake) / (wake_width + 1e-8)  # Avoid division by zero
                
                # Divergent waves are stronger near the wake edges
                edge_factor = np.exp(-((normalized_x - 0.7)**2) / (0.3**2))  # Peak at 70% of wake width
                divergent_waves = edge_factor * np.sin(divergent_frequency * y_wake)
                
                # Lateral decay profile (Gaussian across wake width)
                sigma_width = wake_width / 3.0  # 3-sigma contains most of the wake
                sigma_width = np.maximum(sigma_width, self.beam_ship/6)  # Minimum width
                lateral_decay = np.exp(-0.5 * (x_wake**2) / (sigma_width**2))
                
                # Longitudinal decay (exponential with distance behind ship)
                decay_length = self.wake_length / 3.0
                longitudinal_decay = np.exp(-y_wake / decay_length)
                
                # Combine wave components
                wave_pattern = (0.7 * transverse_waves +  # Dominant transverse component
                              0.3 * divergent_waves)      # Secondary divergent component
                
                # Apply amplitude and decay
                wake_height = (self.wake_amplitude * wave_pattern * 
                             lateral_decay * longitudinal_decay)
                
                # Fix broadcasting: Create array for all behind_ship points, then fill wake region
                wake_behind = np.zeros(np.sum(behind_ship))
                wake_behind[in_wake_region] = wake_height
                wake[behind_ship] = wake_behind
        
        return wake

    def create_bow_wave(self, X_rel, Y_rel):
        """Create bow wave at the front of the ship"""
        bow_wave = np.zeros_like(X_rel)
        
        # Bow wave region (near ship front)
        near_bow = (np.abs(Y_rel) < self.length_ship/2) & (np.abs(X_rel) < self.beam_ship)
        
        if np.any(near_bow):
            # Gaussian profile around ship
            sigma_x = self.beam_ship / 2
            sigma_y = self.length_ship / 4
            
            bow_amplitude = self.wake_amplitude * 1.5  # Bow wave is typically larger
            
            bow_wave[near_bow] = bow_amplitude * np.exp(
                -0.5 * ((X_rel[near_bow]**2) / sigma_x**2 + 
                       (Y_rel[near_bow]**2) / sigma_y**2)
            )
        
        return bow_wave

    def create_turbulent_wake(self, X_rel, Y_rel):
        """Create turbulent wake directly behind the ship"""
        turbulent_wake = np.zeros_like(X_rel)
        
        # Turbulent region directly behind ship
        behind_ship = (Y_rel < -self.length_ship/4) & (Y_rel > -self.length_ship*2)
        in_beam = np.abs(X_rel) < self.beam_ship/2
        turbulent_region = behind_ship & in_beam
        
        if np.any(turbulent_region):
            # Distance behind ship
            y_dist = np.abs(Y_rel[turbulent_region] + self.length_ship/4)
            
            # Gaussian profile across ship beam
            sigma_x = self.beam_ship / 4
            lateral_profile = np.exp(-0.5 * (X_rel[turbulent_region]**2) / sigma_x**2)
            
            # Exponential decay with distance
            decay_profile = np.exp(-y_dist / (self.length_ship * 2))
            
            # Turbulent amplitude (negative - creates depression)
            turbulent_amplitude = -self.wake_amplitude * 0.5
            
            turbulent_wake[turbulent_region] = (turbulent_amplitude * 
                                              lateral_profile * decay_profile)
        
        return turbulent_wake
    

if __name__ == "__main__":

    # Set up the ocean parameters
    num_grid = 400  # grid size
    scene_length = 200 # domain size
    wind_speed =1
    wind_direction = [1, 0]
    wave_amplitude = .1 # wave amplitude
    choppiness = .1  # choppiness factor
    dt = 0.1
    ocean = OceanSurface(num_grid, scene_length, wind_speed, wave_amplitude,choppiness,
                         wind_direction=wind_direction,
                         include_wake=True,
                         velocity_ship=10,
                         length_ship=110,
                         beam_ship=20.3,
                         draft_ship=6.5,
                         initial_wake_position=[0,0],
                         enable_swell=True,
                         swell_amplitude=0.025,
                         swell_wavelength=10.0,
                         swell_direction=[1,0],
                         smooth=True
                         )

    # Set up the plotter
    plotter = pv.Plotter()
    plotter.show_grid()
    output_path = '../output/seastate'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plotter.open_movie(os.path.join(output_path,"ocean_animation.mp4"))

    # Create the initial surface


    # Set up the colormap
    colormap = "bone"  # You can change this to other colormaps like "blues", "coolwarm", etc.

    # Get the height range for consistent color scaling
    height_range = None
    surf = None
    # Animate
    n_frames = 1000
    grid = ocean.create_ocean_surface_plot(0)
    for frame in range(n_frames):

        # plotter.clear_actors()
        # if surf is not None:
        #     plotter.remove_actor(surf)

        t = frame * dt
        grid = ocean.create_ocean_surface_plot(t)

        # Calculate and print peak-to-peak value for this frame
        peak_to_peak = grid['Height'].max() - grid['Height'].min()
        print(f"Frame {frame}: Peak-to-peak wave height = {peak_to_peak:.3f}")

        # Update the height range if it's not set
        if height_range is None:
            height_range = [grid["Height"].min(), grid["Height"].max()]

        if surf is None:
            # Add the surface with the colormap based on height
            surf = plotter.add_mesh(
                grid,
                scalars="Height",
                cmap=colormap,
                show_edges=False,
                clim=height_range,  # Use consistent color scaling
            )

        else:
            for ac in plotter.actors.keys():
                if hasattr(plotter.actors[ac], 'mapper'):
                    plotter.actors[ac].mapper.dataset = grid
        frame_zfill = str(frame).zfill(4)
        # ocean.save_mesh(os.path.join(output_path,f"seastate_{frame_zfill}.stl"))
        # Add a colorbar
        if frame == 0:  # Only add the colorbar once
            plotter.add_scalar_bar("Wave Height", vertical=True, title_font_size=10, label_font_size=8)
        plotter.write_frame()
    plotter.close()