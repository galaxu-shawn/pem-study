"""Modern sea state synthesis with directional spectrum and wake coupling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
from scipy.fft import ifft2
from scipy.ndimage import gaussian_filter


def estimate_ship_behavior(ocean, ship_dict,visualization_scale=1.0):
    """
    Calculate ship rocking parameters based on ocean conditions and ship characteristics.
    
    This function estimates the roll and pitch motion of a ship based on:
    - Ocean wave parameters (wave height, period, direction)
    - Ship geometry (length, width/beam)
    - Ship heading relative to wave direction
    
    The function uses simplified naval architecture principles to estimate angular
    motion amplitudes and rates for simulation purposes.
    
    Args:
        ocean: OceanSurface instance containing wave and wind parameters including:
            - swell_wavelength: Dominant wavelength in meters
            - wave_amplitude: Wave height in meters
            - wind_direction: Wind direction vector (x, y)
        ship_dict: Dictionary containing ship parameters:
            - orientation: Ship heading in degrees (0=east, 90=north)
            - length: Ship length in meters
            - width: Ship beam (width) in meters
        visualization_scale: Scaling factor to enhance visibility of ship motion in simulations.
    Returns:
        dict: Updated ship_dict with added rocking parameters:
            - roll_amplitude: Maximum roll angle in radians
            - pitch_amplitude: Maximum pitch angle in radians
            - wave_omega: Angular frequency of waves in rad/s
            - relative_wave_angle: Angle between waves and ship heading in radians
            
    Notes:
        - Roll motion is maximum when waves approach from the side (beam seas)
        - Pitch motion is maximum when waves approach from front/back (head/following seas)
        - Includes visualization scaling (default 2.5×) to make motion apparent in simulations
        - Wave steepness (H/λ) is factored in: steeper waves cause more violent motion
        - Smaller ships (relative to wavelength) experience more pronounced motion
        - Typical real-world values: 2-5° in moderate seas, 10-20° in heavy seas
    """
    # ===== Extract wave parameters from ocean configuration =====
    
    # Get dominant wavelength from swell component
    dominant_wavelength = ocean.swell_wavelength
    
    # Calculate wave speed using deep water dispersion relation: c = sqrt(g*λ / 2π)
    wave_speed = np.sqrt(9.81 * dominant_wavelength / (2 * np.pi))  # m/s
    
    # Calculate wave period: T = λ / c
    wave_period = dominant_wavelength / wave_speed  # seconds
    
    # Get significant wave height from ocean configuration
    estimated_wave_height = ocean.wave_amplitude  # meters 
    
    # ===== Calculate wind direction =====
    
    # Extract and normalize wind direction vector
    wind_dir = np.array(ocean.wind_direction)
    wind_dir = wind_dir / np.linalg.norm(wind_dir)
    
    # Convert to angle in radians (0=east, π/2=north)
    wind_angle = np.arctan2(wind_dir[1], wind_dir[0])
    
    # ===== Print diagnostic information =====
    
    print(f'\n{"="*70}')
    print(f'Ship Rocking Parameters')
    print(f'{"="*70}')
    print(f'Estimated Wave Height: {estimated_wave_height:.2f} m')
    print(f'Wave Period: {wave_period:.2f} s')
    print(f'Wind Direction: {np.rad2deg(wind_angle):.1f} deg')
    print(f'{"="*70}\n')
    
    # ===== Extract ship parameters =====

    ship_heading = ship_dict['orientation']  # Ship heading in degrees
    ship_length = ship_dict['length']  # Ship length in meters
    ship_width = ship_dict['width']  # Ship beam (width) in meters
    
    # Convert ship heading from degrees to radians
    ship_heading_rad = np.deg2rad(ship_heading)
    
    # ===== Calculate relative wave direction =====
    
    # Angle between incoming waves and ship's longitudinal axis
    # 0° = head seas (waves from bow), 90° = beam seas (waves from side)
    relative_angle = wind_angle - ship_heading_rad
    
    # ===== Calculate directional factors for roll and pitch =====
    
    # Roll motion: rocking along ship's longitudinal axis (X in ship frame)
    # Maximum when waves hit ship from the side (beam seas, relative_angle = ±90°)
    # Minimum when waves are head-on or following (relative_angle = 0° or 180°)
    roll_factor = np.abs(np.sin(relative_angle))
    
    # Pitch motion: rocking along ship's transverse axis (Y in ship frame)
    # Maximum when waves hit ship from front/back (head/following seas, relative_angle = 0° or 180°)
    # Minimum when waves are from the side (beam seas, relative_angle = ±90°)
    pitch_factor = np.abs(np.cos(relative_angle))
    
    # ===== Calculate roll amplitude =====
    
    # Roll amplitude depends on:
    # 1. Wave height (larger waves → more roll)
    # 2. Ship beam/width (narrower ships → more roll)
    # 3. Wave direction (beam seas → maximum roll)
    # 4. Wave steepness (steeper waves → more violent motion)
    
    # Calculate wave steepness (H/λ) - affects motion intensity
    wave_steepness = estimated_wave_height / dominant_wavelength
    steepness_factor = 1.0 + 2.0 * wave_steepness  # Steeper waves = more motion
    
    # Base roll calculation with enhanced empirical factor
    # Typical values: 2-10° for moderate seas, up to 20-30° for heavy seas
    base_roll = (estimated_wave_height / ship_width) * 20 * roll_factor
    
    # Apply steepness and size corrections
    # Smaller ships (relative to wavelength) respond more quickly
    size_ratio = ship_length / dominant_wavelength
    size_factor = 1.0 if size_ratio > 0.5 else (1.0 + (0.5 - size_ratio))
    
    roll_amplitude_deg = base_roll * steepness_factor * size_factor  # degrees
    
    # ===== Calculate pitch amplitude =====
    
    # Pitch amplitude depends on:
    # 1. Wave height (larger waves → more pitch)
    # 2. Ship length (shorter ships → more pitch)
    # 3. Wave direction (head/following seas → maximum pitch)
    # 4. Wave period matching ship natural period
    
    # Base pitch calculation with enhanced empirical factor
    # Pitch is typically less severe than roll (better stability about Y-axis)
    base_pitch = (estimated_wave_height / ship_length) * 15 * pitch_factor
    
    # Apply steepness and size corrections
    pitch_amplitude_deg = base_pitch * steepness_factor * size_factor  # degrees
    
    # ===== Calculate angular frequency =====
    
    # Wave angular frequency: ω = 2π/T (radians per second)
    # This is the rate at which the wave oscillates
    # Ship motion follows wave period: as waves pass, ship rocks at this frequency
    # For sinusoidal motion θ(t) = A·sin(ωt), angular velocity is θ'(t) = A·ω·cos(ωt)
    # Maximum angular velocity occurs when cos(ωt) = ±1, giving |θ'_max| = A·ω
    omega = 2 * np.pi / wave_period
    
    # ===== Convert amplitudes to radians and apply scaling =====
    
    # Convert degree amplitudes to radians for use in simulations
    # Apply visibility scaling factor to make ship motion more apparent in visualizations
    # Real ship motions in moderate seas are typically 2-5° which can be subtle to perceive
    
    roll_amplitude_rad = np.deg2rad(roll_amplitude_deg) * visualization_scale
    pitch_amplitude_rad = np.deg2rad(pitch_amplitude_deg) * visualization_scale 
    
    # ===== Store calculated parameters in ship dictionary =====
    
    # These parameters can be used to calculate ship motion at any time t:
    # roll(t) = roll_amplitude * sin(wave_omega * t)
    # pitch(t) = pitch_amplitude * sin(wave_omega * t)
    ship_dict['roll_amplitude'] = roll_amplitude_rad  # Maximum roll angle (radians)
    ship_dict['pitch_amplitude'] = pitch_amplitude_rad  # Maximum pitch angle (radians)
    ship_dict['wave_omega'] = omega  # Wave angular frequency (rad/s)
    ship_dict['relative_wave_angle'] = relative_angle  # Angle between waves and ship (radians)
    
    # ===== Print ship-specific rocking summary =====
    
    print(f'  Heading: {ship_heading:.1f} deg')
    print(f'  Relative Wave Angle: {np.rad2deg(relative_angle):.1f} deg')
    print(f'  Roll Amplitude: {roll_amplitude_deg:.2f} deg')
    print(f'  Pitch Amplitude: {pitch_amplitude_deg:.2f} deg')
    print(f'  Max Roll Rate: {roll_amplitude_rad * omega:.4f} rad/s')
    print(f'  Max Pitch Rate: {pitch_amplitude_rad * omega:.4f} rad/s\n')

    return ship_dict

class OceanPresets:
    """Library of standard ocean surface configurations."""
    
    BEAUFORT_SCALE = {
        0: {"wind_speed": 0.5, "wave_amplitude": 0.0, "name": "Calm"},
        1: {"wind_speed": 0.95, "wave_amplitude": 0.15, "name": "Light air"},
        2: {"wind_speed": 2.55, "wave_amplitude": 0.45, "name": "Light breeze"},
        3: {"wind_speed": 4.4, "wave_amplitude": 0.9, "name": "Gentle breeze"},
        4: {"wind_speed": 6.7, "wave_amplitude": 1.5, "name": "Moderate breeze"},
        5: {"wind_speed": 9.35, "wave_amplitude": 2.5, "name": "Fresh breeze"},
        6: {"wind_speed": 12.3, "wave_amplitude": 3.5, "name": "Strong breeze"},
        7: {"wind_speed": 15.5, "wave_amplitude": 4.75, "name": "Near gale"},
        8: {"wind_speed": 18.95, "wave_amplitude": 6.5, "name": "Gale"},
        9: {"wind_speed": 22.6, "wave_amplitude": 8.5, "name": "Strong gale"},
        10: {"wind_speed": 26.45, "wave_amplitude": 10.75, "name": "Storm"},
        11: {"wind_speed": 30.0, "wave_amplitude": 13.75, "name": "Violent storm"},
        12: {"wind_speed": 33.0, "wave_amplitude": 15.0, "name": "Hurricane"},
    }
    
    @classmethod
    def from_beaufort(cls, beaufort_number: int, **kwargs):
        """Create ocean surface from Beaufort scale number."""
        params = cls.BEAUFORT_SCALE.get(beaufort_number)
        if params is None:
            raise ValueError(f"Beaufort number {beaufort_number} not in preset library, valid value are {list(cls.BEAUFORT_SCALE.keys())}")
        
        ocean_config = OceanConfig(
            wind_speed=params["wind_speed"],
            wave_amplitude=params["wave_amplitude"],
            **kwargs
        )
        return OceanSurface(ocean_config)



@dataclass
class SpectrumParameters:
    """Container for defining the target sea state."""

    wave_amplitude: float
    wind_speed: float
    peak_period: Optional[float] = None
    significant_wave_height: Optional[float] = None
    gamma: float = 3.3  # JONSWAP peak enhancement
    directional_spread: float = 30.0  # degrees
    directional_exponent: Optional[int] = None


@dataclass
class OceanConfig:
    """Configuration for basic ocean surface parameters."""
    
    num_grid: int = 256
    scene_length: float = 1000.0
    wind_speed: float = 10.0
    wave_amplitude: float = 1.0
    choppiness: float = 0.5
    wind_direction: Tuple[float, float] = (1.0, 0.0)
    smooth: bool = False
    depth: Optional[float] = None
    spectrum: str = "jonswap"
    spectrum_gamma: float = 3.3
    directional_spread: float = 30.0
    directional_exponent: Optional[int] = None
    random_seed: Optional[int] = None


@dataclass
class WakeConfig:
    """Configuration for ship wake parameters."""
    
    enabled: bool = True
    velocity_ship: float = 10.0
    length_ship: float = 110.0
    beam_ship: float = 20.3
    draft_ship: float = 3.5
    initial_position: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    update_position: bool = True
    amplitude_scale: float = 1.0


@dataclass
class SwellConfig:
    """Configuration for swell wave parameters."""
    
    enabled: bool = False
    amplitude: float = 0.0
    wavelength: float = 100.0
    direction: Tuple[float, float] = (1.0, 0.0)
    phase: float = 0.0
    frequency: Optional[float] = None


class OceanSurface:
    """Generate ocean surface realisations using a directional JONSWAP spectrum."""

    def __init__(
        self,
        ocean_config: Optional[OceanConfig] = None,
        wake_config: Optional[WakeConfig] = None,
        swell_config: Optional[SwellConfig] = None,
        # Legacy parameters for backward compatibility (deprecated)
        num_grid: Optional[int] = None,
        scene_length: Optional[float] = None,
        wind_speed: Optional[float] = None,
        wave_amplitude: Optional[float] = None,
        choppiness: Optional[float] = None,
        wind_direction: Optional[Sequence[float] | np.ndarray] = None,
        include_wake: Optional[bool] = None,
        velocity_ship: Optional[float] = None,
        length_ship: Optional[float] = None,
        beam_ship: Optional[float] = None,
        draft_ship: Optional[float] = None,
        initial_wake_position: Optional[Sequence[float] | np.ndarray] = None,
        wake_rotation: Optional[float] = None,
        update_wake_position: Optional[bool] = None,
        wake_amplitude_scale: Optional[float] = None,
        enable_swell: Optional[bool] = None,
        swell_amplitude: Optional[float] = None,
        swell_wavelength: Optional[float] = None,
        swell_direction: Optional[Sequence[float] | np.ndarray] = None,
        swell_phase: Optional[float] = None,
        swell_frequency: Optional[float] = None,
        smooth: Optional[bool] = None,
        depth: Optional[float] = None,
        spectrum: Optional[str] = None,
        spectrum_gamma: Optional[float] = None,
        directional_spread: Optional[float] = None,
        directional_exponent: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        # Handle legacy parameter initialization vs new config-based initialization
        if ocean_config is None:
            # Legacy mode: use individual parameters
            if num_grid is None or scene_length is None or wind_speed is None or wave_amplitude is None or choppiness is None:
                raise ValueError(
                    "Either provide ocean_config or all required legacy parameters "
                    "(num_grid, scene_length, wind_speed, wave_amplitude, choppiness)"
                )
            ocean_config = OceanConfig(
                num_grid=num_grid,
                scene_length=scene_length,
                wind_speed=wind_speed,
                wave_amplitude=wave_amplitude,
                choppiness=choppiness,
                wind_direction=tuple(wind_direction) if wind_direction is not None else (1.0, 0.0),
                smooth=smooth if smooth is not None else False,
                depth=depth,
                spectrum=spectrum if spectrum is not None else "jonswap",
                spectrum_gamma=spectrum_gamma if spectrum_gamma is not None else 3.3,
                directional_spread=directional_spread if directional_spread is not None else 30.0,
                directional_exponent=directional_exponent,
                random_seed=random_seed,
            )
        
        if wake_config is None and include_wake:
            # Legacy wake parameters provided
            wake_config = WakeConfig(
                enabled=True,
                velocity_ship=velocity_ship if velocity_ship is not None else 10.0,
                length_ship=length_ship if length_ship is not None else 110.0,
                beam_ship=beam_ship if beam_ship is not None else 20.3,
                draft_ship=draft_ship if draft_ship is not None else 3.5,
                initial_position=tuple(initial_wake_position) if initial_wake_position is not None else (0.0, 0.0),
                rotation=wake_rotation if wake_rotation is not None else 0.0,
                update_position=update_wake_position if update_wake_position is not None else True,
                amplitude_scale=wake_amplitude_scale if wake_amplitude_scale is not None else 1.0,
            )
        elif wake_config is None:
            wake_config = WakeConfig(enabled=False)
        
        if swell_config is None and enable_swell:
            # Legacy swell parameters provided
            swell_config = SwellConfig(
                enabled=True,
                amplitude=swell_amplitude if swell_amplitude is not None else 0.0,
                wavelength=swell_wavelength if swell_wavelength is not None else 100.0,
                direction=tuple(swell_direction) if swell_direction is not None else (1.0, 0.0),
                phase=swell_phase if swell_phase is not None else 0.0,
                frequency=swell_frequency,
            )
        elif swell_config is None:
            swell_config = SwellConfig(enabled=False)
        
        # Store configs
        self.ocean_config = ocean_config
        self.wake_config = wake_config
        self.swell_config = swell_config
        
        # Extract ocean parameters
        self.num_grid = int(ocean_config.num_grid)
        if self.num_grid < 4:
            raise ValueError("num_grid must be >= 4 for spectral synthesis")

        self.scene_length = float(ocean_config.scene_length)
        self.wind_speed = float(ocean_config.wind_speed)
        self.wind_direction = self._normalize_vector(np.asarray(ocean_config.wind_direction, dtype=float))
        self.wave_amplitude = float(ocean_config.wave_amplitude)
        self.choppiness = float(ocean_config.choppiness)
        self.smooth = bool(ocean_config.smooth)
        self.depth = ocean_config.depth
        self.spectrum_type = ocean_config.spectrum.lower()
        self.gamma = float(ocean_config.spectrum_gamma)
        self.directional_spread = float(ocean_config.directional_spread)
        self.directional_exponent = ocean_config.directional_exponent
        self.random_seed = ocean_config.random_seed

        # Wake will be managed separately (support multiple wakes)
        self.wakes: list[Wake] = []
        
        # Extract swell parameters
        self.enable_swell = swell_config.enabled
        self.swell_amplitude = float(swell_config.amplitude)
        self.swell_wavelength = float(swell_config.wavelength)
        self.swell_direction = self._normalize_vector(np.asarray(swell_config.direction, dtype=float))
        self.swell_phase = float(swell_config.phase)
        self.swell_frequency = swell_config.frequency

        self.g = 9.81
        self.dx = self.scene_length / self.num_grid
        self.kx, self.ky = self._build_wavenumber_grid(self.num_grid, self.dx)
        self.k = np.hypot(self.kx, self.ky)
        self.k[0, 0] = 1e-12

        self.wind_angle = float(np.arctan2(self.wind_direction[1], self.wind_direction[0]))
        self.omega = self._dispersion_relation(self.k)
        self.delta_k = 2.0 * np.pi / self.scene_length

        rng = np.random.default_rng(self.random_seed)
        self._rng = rng

        spectrum_params = SpectrumParameters(
            wave_amplitude=self.wave_amplitude,
            wind_speed=self.wind_speed,
            peak_period=None,
            significant_wave_height=None,
            gamma=self.gamma,
            directional_spread=self.directional_spread,
            directional_exponent=self.directional_exponent,
        )
        self.spectrum_params = spectrum_params

        self.directional_spectrum = self._compute_directional_spectrum()
        self.h0 = self._generate_initial_amplitudes()
        self.h0_conj = np.conj(self.h0[::-1, ::-1])

        raw_height = self._height_from_spectrum(0.0)
        ptp = float(np.ptp(raw_height))
        self.height_scale = self.wave_amplitude / ptp if ptp > 1e-12 else 1.0

        self.mesh: Optional[pv.PolyData] = None
        self.time = 0.0
        
        # Initialize wake if configured
        if wake_config.enabled:
            wake_rotation_rad = np.deg2rad(wake_config.rotation)
            wake = Wake(
                velocity_ship=wake_config.velocity_ship,
                length_ship=wake_config.length_ship,
                beam_ship=wake_config.beam_ship,
                draft_ship=wake_config.draft_ship,
                scene_length=self.scene_length,
                grid_spacing=self.scene_length / self.num_grid,
                wake_position=wake_config.initial_position,
                wake_rotation=wake_rotation_rad,
                num_grid=self.num_grid,
                wake_amplitude_scale=wake_config.amplitude_scale,
            )
            self.wakes.append(wake)

    # Factory methods for common scenarios
    @classmethod
    def calm_sea(
        cls,
        num_grid: int = 256,
        scene_length: float = 1000.0,
        random_seed: Optional[int] = None,
    ) -> "OceanSurface":
        """Create a calm sea surface with gentle waves.
        
        Args:
            num_grid: Grid resolution
            scene_length: Physical size of simulation domain in meters
            random_seed: Random seed for reproducibility
            
        Returns:
            OceanSurface configured for calm conditions
        """
        ocean_config = OceanConfig(
            num_grid=num_grid,
            scene_length=scene_length,
            wind_speed=5.0,
            wave_amplitude=0.3,
            choppiness=0.3,
            random_seed=random_seed,
        )
        return cls(ocean_config)

    @classmethod
    def rough_sea(
        cls,
        num_grid: int = 256,
        scene_length: float = 1000.0,
        random_seed: Optional[int] = None,
    ) -> "OceanSurface":
        """Create a rough sea surface with large waves.
        
        Args:
            num_grid: Grid resolution
            scene_length: Physical size of simulation domain in meters
            random_seed: Random seed for reproducibility
            
        Returns:
            OceanSurface configured for rough conditions
        """
        ocean_config = OceanConfig(
            num_grid=num_grid,
            scene_length=scene_length,
            wind_speed=20.0,
            wave_amplitude=2.0,
            choppiness=0.8,
            random_seed=random_seed,
        )
        return cls(ocean_config)

    @classmethod
    def with_ship_wake(
        cls,
        ship_velocity: float,
        ship_length: float,
        num_grid: int = 256,
        scene_length: float = 1000.0,
        wind_speed: float = 10.0,
        wave_amplitude: float = 1.0,
        ship_position: Tuple[float, float] = (0.0, 0.0),
        ship_heading: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> "OceanSurface":
        """Create ocean surface with ship wake.
        
        Args:
            ship_velocity: Ship velocity in m/s
            ship_length: Ship length in meters
            num_grid: Grid resolution
            scene_length: Physical size of simulation domain in meters
            wind_speed: Wind speed in m/s
            wave_amplitude: Wave amplitude in meters
            ship_position: Initial ship position (x, y) in meters
            ship_heading: Ship heading in degrees (0 = east, 90 = north)
            random_seed: Random seed for reproducibility
            
        Returns:
            OceanSurface configured with wake
        """
        ocean_config = OceanConfig(
            num_grid=num_grid,
            scene_length=scene_length,
            wind_speed=wind_speed,
            wave_amplitude=wave_amplitude,
            choppiness=0.5,
            random_seed=random_seed,
        )
        
        # Auto-calculate beam and draft from length if typical values
        beam = ship_length * 0.15  # Typical beam-to-length ratio
        draft = ship_length * 0.05  # Typical draft-to-length ratio
        
        wake_config = WakeConfig(
            enabled=True,
            velocity_ship=ship_velocity,
            length_ship=ship_length,
            beam_ship=beam,
            draft_ship=draft,
            initial_position=ship_position,
            rotation=ship_heading,
            update_position=True,
            amplitude_scale=1.0,
        )
        
        return cls(ocean_config, wake_config)

    def add_wake(self, wake: "Wake") -> None:
        """Add a wake component to the ocean surface.
        
        Multiple wakes can be added and will be superimposed.
        
        Args:
            wake: Wake instance to add to the ocean
        """
        self.wakes.append(wake)

    def remove_wake(self, index: Optional[int] = None) -> None:
        """Remove wake component(s) from the ocean surface.
        
        Args:
            index: Index of wake to remove. If None, removes all wakes.
        """
        if index is None:
            self.wakes.clear()
        else:
            if 0 <= index < len(self.wakes):
                self.wakes.pop(index)
            else:
                raise IndexError(f"Wake index {index} out of range (0-{len(self.wakes)-1})")
    
    def get_num_wakes(self) -> int:
        """Get the number of wake components currently active.
        
        Returns:
            Number of wakes
        """
        return len(self.wakes)
    
    def add_swell(
        self,
        amplitude: float,
        wavelength: float,
        direction: Tuple[float, float] = (1.0, 0.0),
        phase: float = 0.0,
        frequency: Optional[float] = None,
    ) -> None:
        """Add swell waves to the ocean surface.
        
        Args:
            amplitude: Swell amplitude in meters
            wavelength: Swell wavelength in meters
            direction: Swell direction as (x, y) tuple
            phase: Initial phase in radians
            frequency: Swell frequency in rad/s (auto-calculated if None)
        """
        self.swell_config = SwellConfig(
            enabled=True,
            amplitude=amplitude,
            wavelength=wavelength,
            direction=direction,
            phase=phase,
            frequency=frequency,
        )
        self.enable_swell = True
        self.swell_amplitude = float(amplitude)
        self.swell_wavelength = float(wavelength)
        self.swell_direction = self._normalize_vector(np.asarray(direction, dtype=float))
        self.swell_phase = float(phase)
        self.swell_frequency = frequency

    def remove_swell(self) -> None:
        """Remove swell waves from the ocean surface."""
        self.swell_config.enabled = False
        self.enable_swell = False

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            raise ValueError("Zero-length direction vector")
        return vec / norm

    @staticmethod
    def _build_wavenumber_grid(num_grid: int, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        freq = 2.0 * np.pi * np.fft.fftfreq(num_grid, d=dx)
        return np.meshgrid(freq, freq)

    def _dispersion_relation(self, k: np.ndarray) -> np.ndarray:
        if self.depth is None:
            omega = np.sqrt(self.g * k)
        else:
            kd = k * self.depth
            omega = np.sqrt(self.g * k * np.tanh(kd))
        omega[k == 0] = 0.0
        return omega

    def _df_dk(self, k: np.ndarray, omega: np.ndarray) -> np.ndarray:
        k_safe = np.where(k < 1e-12, 1e-12, k)
        if self.depth is None:
            domega_dk = 0.5 * np.sqrt(self.g / k_safe)
        else:
            kd = k_safe * self.depth
            tanh_kd = np.tanh(kd)
            sech2_kd = 1.0 / np.cosh(kd) ** 2
            numerator = self.g * (tanh_kd + kd * sech2_kd)
            denom = np.where(omega < 1e-12, 1e-12, omega)
            domega_dk = 0.5 * numerator / denom
        return domega_dk / (2.0 * np.pi)

    def _compute_directional_spectrum(self) -> np.ndarray:
        frequency = self.omega / (2.0 * np.pi)
        spectra = self._jonswap_spectrum(frequency)
        directional = self._directional_distribution()
        df_dk = self._df_dk(self.k, self.omega)
        spectrum_2d = spectra * directional * df_dk

        cutoff = self.delta_k * 0.5
        spectrum_2d[self.k < cutoff] = 0.0
        spectrum_2d[0, 0] = 0.0
        spectrum_2d = np.nan_to_num(spectrum_2d, nan=0.0, posinf=0.0, neginf=0.0)
        return spectrum_2d

    def _jonswap_spectrum(self, frequency: np.ndarray) -> np.ndarray:
        if self.spectrum_type not in {"jonswap", "pm"}:
            raise ValueError(f"Unsupported spectrum type '{self.spectrum_type}'")

        f_safe = np.maximum(frequency, 1e-5)
        if self.spectrum_params.peak_period:
            fp = 1.0 / max(self.spectrum_params.peak_period, 1e-3)
        else:
            fp = 0.13 * self.g / max(self.wind_speed, 1e-3)
        sigma = np.where(f_safe <= fp, 0.07, 0.09)
        r = np.exp(-((f_safe - fp) ** 2) / (2.0 * sigma ** 2 * fp ** 2))

        alpha_pm = 0.0081
        base = (
            alpha_pm
            * self.g**2
            / (2.0 * np.pi) ** 4
            / f_safe**5
            * np.exp(-1.25 * (fp / f_safe) ** 4)
        )

        if self.spectrum_type == "jonswap":
            gamma = self.gamma
            enhancement = gamma**r
        else:
            enhancement = np.ones_like(base)

        spectrum = base * enhancement
        spectrum[frequency <= 0] = 0.0
        return spectrum

    def _directional_distribution(self) -> np.ndarray:
        theta = np.arctan2(self.ky, self.kx)
        theta_diff = np.arctan2(
            np.sin(theta - self.wind_angle), np.cos(theta - self.wind_angle)
        )
        spread_rad = np.deg2rad(max(self.directional_spread, 1.0))
        if self.directional_exponent is not None:
            exponent = max(1, int(self.directional_exponent))
        else:
            exponent = int(np.clip(np.round((2.0 / spread_rad) ** 2), 1, 50))
        spread = np.cos(0.5 * theta_diff)
        spread = np.clip(spread, 0.0, None) ** (2 * exponent)
        mean_spread = spread.mean()
        if mean_spread > 0:
            spread /= mean_spread
        spread[self.k == 0] = 0.0
        return spread

    def _generate_initial_amplitudes(self) -> np.ndarray:
        amplitude = np.sqrt(self.directional_spectrum) * self.delta_k
        xi_real = self._rng.normal(size=(self.num_grid, self.num_grid))
        xi_imag = self._rng.normal(size=(self.num_grid, self.num_grid))
        h0 = (xi_real + 1j * xi_imag) * amplitude / np.sqrt(2.0)
        h0[0, 0] = 0.0
        return h0

    def _height_from_spectrum(self, t: float) -> np.ndarray:
        phase_pos = np.exp(1j * self.omega * t)
        phase_neg = np.exp(-1j * self.omega * t)
        spectrum = self.h0 * phase_pos + self.h0_conj * phase_neg
        height = np.real(ifft2(spectrum))
        return height

    def generate_height_field(self, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time
        else:
            self.time = float(t)

        height = self._height_from_spectrum(t) * self.height_scale
        height -= height.mean()

        if self.enable_swell and self.swell_amplitude > 0.0:
            height += self._swell_field(t)

        if self.wakes:
            height += self._wake_height(t)

        return height

    def generate_displacement_field(self, t: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        if t is None:
            t = self.time
        else:
            self.time = float(t)

        phase_pos = np.exp(1j * self.omega * t)
        phase_neg = np.exp(-1j * self.omega * t)
        spectrum = self.h0 * phase_pos + self.h0_conj * phase_neg
        spectrum *= self.height_scale

        k_safe = np.where(self.k < 1e-12, 1e-12, self.k)
        dx_tilde = -1j * self.kx / k_safe * spectrum
        dy_tilde = -1j * self.ky / k_safe * spectrum
        dx_tilde[0, 0] = 0.0
        dy_tilde[0, 0] = 0.0

        dx = np.real(ifft2(dx_tilde)) * self.choppiness
        dy = np.real(ifft2(dy_tilde)) * self.choppiness
        return dx, dy

    def _wake_height(self, t: float) -> np.ndarray:
        if not self.wakes:
            return np.zeros((self.num_grid, self.num_grid))
        
        # Superimpose all wakes
        total_wake_height = np.zeros((self.num_grid, self.num_grid))
        
        for wake in self.wakes:
            # Get initial position from the wake object (stored during creation)
            # Check if wake has stored config parameters
            if hasattr(wake, '_initial_position'):
                x0, y0 = wake._initial_position
                velocity = wake._velocity_ship
                rotation_rad = wake._rotation_rad
                update_position = wake._update_position
            else:
                # Fallback to wake's current position
                x0, y0 = wake.wake_position
                velocity = wake.velocity_ship
                rotation_rad = wake.wake_rotation
                update_position = True
            
            # Update wake position if configured
            if update_position:
                # advance ship along its heading
                heading = np.array([
                    np.cos(rotation_rad),
                    np.sin(rotation_rad),
                ])
                displacement = velocity * t * heading
                wake.wake_position = np.array([x0 + displacement[0], y0 + displacement[1]])
            
            # Calculate this wake's contribution
            wake.calculate(time=t)
            total_wake_height += wake.z_ship
        
        return total_wake_height


    def _swell_field(self, t: float) -> np.ndarray:
        x = np.linspace(-self.scene_length / 2, self.scene_length / 2, self.num_grid)
        y = np.linspace(-self.scene_length / 2, self.scene_length / 2, self.num_grid)
        xx, yy = np.meshgrid(x, y)
        swell_dir = self.swell_direction
        phase = 2.0 * np.pi * (xx * swell_dir[0] + yy * swell_dir[1]) / self.swell_wavelength
        if self.swell_frequency is None:
            freq = np.sqrt(2.0 * np.pi * self.g / self.swell_wavelength)
        else:
            freq = float(self.swell_frequency)
        swell = self.swell_amplitude * np.sin(phase + self.swell_phase + freq * t)
        return swell

    def _generate_grid(self, t: Optional[float] = None) -> pv.StructuredGrid:
        if t is None:
            t = self.time
        else:
            self.time = float(t)

        x = np.linspace(-self.scene_length / 2, self.scene_length / 2, self.num_grid)
        y = np.linspace(-self.scene_length / 2, self.scene_length / 2, self.num_grid)
        xx, yy = np.meshgrid(x, y)

        height = self.generate_height_field(t)
        dx, dy = self.generate_displacement_field(t)

        xx += dx
        yy += dy

        grid = pv.StructuredGrid(xx, yy, height)
        grid["Height"] = height.ravel(order="F")
        return grid

    def generate_mesh(self, t: Optional[float] = None) -> pv.PolyData:
        grid = self._generate_grid(t)
        if self.smooth:
            mesh = self.smooth_mesh(grid, smoothing_iterations=20)
        else:
            mesh = grid.extract_geometry().triangulate()
        self.mesh = mesh
        return mesh

    def save_mesh(self, output_path: str) -> None:
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)

    def smooth_mesh(self, grid: pv.StructuredGrid | pv.PolyData, smoothing_iterations: int = 20) -> pv.PolyData:
        surface = grid.extract_geometry()
        smoothed = surface.smooth_taubin(
            n_iter=smoothing_iterations,
            pass_band=0.1,
            feature_smoothing=True,
            boundary_smoothing=True,
            feature_angle=45.0,
            edge_angle=15.0,
            normalize_coordinates=True,
        )
        return smoothed.triangulate()

    def create_ocean_surface_plot(self, t: Optional[float] = None) -> pv.PolyData | pv.StructuredGrid:
        grid = self._generate_grid(t)
        if self.smooth:
            return self.smooth_mesh(grid, smoothing_iterations=20)
        return grid


class Wake:
    """Composite ship wake model with Kelvin, bow and turbulent components."""

    def __init__(
        self,
        velocity_ship: float = 10.0,
        length_ship: float = 110.0,
        beam_ship: float = 20.3,
        draft_ship: float = 3.5,
        scene_length: float = 1000.0,
        grid_spacing: float = 2.5,
        wake_position: Sequence[float] | np.ndarray = (0.0, 0.0),
        wake_rotation: float = 0.0,
        num_grid: Optional[int] = None,
        wake_amplitude_scale: float = 1.0,
        reference_waterplane_area: Optional[float] = None,
    ) -> None:
        self.velocity_ship = float(velocity_ship)
        self.length_ship = float(length_ship)
        self.beam_ship = float(beam_ship)
        self.draft_ship = float(draft_ship)
        self.scene_length = float(scene_length)
        self.grid_spacing = float(grid_spacing)
        self.wake_position = np.asarray(wake_position, dtype=float)
        self.wake_rotation = float(wake_rotation)
        self.num_grid = num_grid
        self.wake_amplitude_scale = float(wake_amplitude_scale)
        self.reference_waterplane_area = reference_waterplane_area

        self.g = 9.81
        self.kelvin_angle = np.deg2rad(19.47)
        self.wake_length = min(self.scene_length * 0.8, self.velocity_ship * 30.0)
        self.wake_amplitude = self._estimate_wake_amplitude()
        self.z_ship: Optional[np.ndarray] = None
        
        # Store initial parameters for position tracking
        self._initial_position = np.asarray(wake_position, dtype=float).copy()
        self._velocity_ship = float(velocity_ship)
        self._rotation_rad = float(wake_rotation)
        self._update_position = True  # Default to updating position

    @classmethod
    def create(
        cls,
        ship_length: float,
        ship_velocity: float,
        scene_length: float = 1000.0,
        num_grid: int = 256,
        ship_position: Tuple[float, float] = (0.0, 0.0),
        ship_heading: float = 0.0,
        beam_ship: Optional[float] = None,
        draft_ship: Optional[float] = None,
    ) -> "Wake":
        """Create a wake with automatic parameter estimation.
        
        Args:
            ship_length: Ship length in meters
            ship_velocity: Ship velocity in m/s
            scene_length: Physical size of simulation domain in meters
            num_grid: Grid resolution
            ship_position: Ship position (x, y) in meters
            ship_heading: Ship heading in degrees (0 = east, 90 = north)
            beam_ship: Ship beam in meters (auto-calculated if None)
            draft_ship: Ship draft in meters (auto-calculated if None)
            
        Returns:
            Wake instance with sensible defaults
        """
        # Auto-calculate beam and draft from typical ratios if not provided
        if beam_ship is None:
            beam_ship = ship_length * 0.15  # Typical beam-to-length ratio
        if draft_ship is None:
            draft_ship = ship_length * 0.05  # Typical draft-to-length ratio
        
        grid_spacing = scene_length / num_grid
        wake_rotation_rad = np.deg2rad(ship_heading)
        
        wake = cls(
            velocity_ship=ship_velocity,
            length_ship=ship_length,
            beam_ship=beam_ship,
            draft_ship=draft_ship,
            scene_length=scene_length,
            grid_spacing=grid_spacing,
            wake_position=ship_position,
            wake_rotation=wake_rotation_rad,
            num_grid=num_grid,
            wake_amplitude_scale=1.0,
        )
        
        # Ensure tracking parameters are set correctly
        wake._initial_position = np.asarray(ship_position, dtype=float)
        wake._velocity_ship = float(ship_velocity)
        wake._rotation_rad = wake_rotation_rad
        wake._update_position = True
        
        return wake
    
    def set_static(self, static: bool = True) -> None:
        """Set whether the wake position should remain static or update with time.
        
        Args:
            static: If True, wake stays at initial position. If False, wake moves with ship velocity.
        """
        self._update_position = not static
    
    def is_static(self) -> bool:
        """Check if the wake is static (not moving with time).
        
        Returns:
            True if wake is static, False if it moves with time.
        """
        return not self._update_position

    def _estimate_wake_amplitude(self) -> float:
        """
        Estimate wake amplitude using empirical scaling relationships.
        
        Key factors:
        - Ship size (waterplane area)
        - Velocity (quadratic relationship with wave energy)
        - Froude number (regime-dependent scaling)
        - Draft (displacement proxy)
        
        Returns amplitude in meters, typically 0.1-8.0m for realistic vessels.
        """
        froude = self.velocity_ship / np.sqrt(self.g * self.length_ship)
        waterplane_area = self.length_ship * self.beam_ship
        
        # Reference area (100m × 15m = medium cargo vessel)
        if self.reference_waterplane_area is not None:
            reference_area = self.reference_waterplane_area
        else:
            reference_area = 100.0 * 15.0
        
        # Size scaling: use linear relationship (larger ships → proportionally larger wakes)
        # For very large ships, use a dampened power law to avoid unrealistic amplitudes
        area_ratio = waterplane_area / reference_area
        if area_ratio > 2.0:
            # Dampen for very large vessels: sqrt for the excess beyond 2×
            size_factor = 2.0 + np.sqrt(area_ratio - 2.0)
        else:
            size_factor = area_ratio
        
        # Velocity scaling: wake energy scales with V^2, but amplitude with sqrt(energy) ~ V
        # Use V^1.5 as compromise between linear and quadratic
        velocity_factor = (self.velocity_ship / 10.0) ** 1.5
        
        # Froude number regime effects
        if froude < 0.3:
            # Subcritical: weak wakes, minimum factor of 0.3
            froude_factor = np.clip(froude / 0.3, 0.3, 1.0)
        elif froude < 0.5:
            # Transitional regime: wake amplitude grows rapidly
            froude_factor = 1.0 + (froude - 0.3) * 3.0
        elif froude < 1.0:
            # Critical regime: maximum wave generation
            froude_factor = 1.6 + (froude - 0.5) * 2.0
        else:
            # Supercritical: amplitude plateaus but remains high
            froude_factor = 2.6 + 0.2 * (froude - 1.0)
        
        # Draft factor: deeper draft → more displacement → larger wake
        # Use tanh to saturate for very deep drafts
        draft_factor = 1.0 + np.tanh(self.draft_ship / 4.0) * 0.8
        
        # Base amplitude for reference vessel at reference speed
        # Increased from 0.2m to 0.5m for more realistic baseline
        base = 0.5
        
        # Compute raw amplitude
        amplitude = base * size_factor * velocity_factor * froude_factor * draft_factor
        
        # Apply user-defined scaling factor
        amplitude *= self.wake_amplitude_scale
        
        # Clip to physically reasonable range
        # Lower bound: very small wakes (0.05m)
        # Upper bound: extreme conditions (10m for large fast vessels)
        return float(np.clip(amplitude, 0.05, 10.0))

    def calculate(self, time: float = 0.0) -> None:
        if self.num_grid is not None:
            axis = np.linspace(-self.scene_length / 2, self.scene_length / 2, self.num_grid)
        else:
            axis = np.arange(-self.scene_length / 2, self.scene_length / 2 + self.grid_spacing, self.grid_spacing)
        xx, yy = np.meshgrid(axis, axis)

        ship_x, ship_y = self.wake_position
        # print(f"Calculating wake at time {time:.2f}s, ship position ({ship_x:.2f}, {ship_y:.2f})")
        yy = yy - (ship_y + 0.0 * time)
        xx = xx - ship_x

        cos_r = np.cos(-self.wake_rotation)
        sin_r = np.sin(-self.wake_rotation)
        x_ship = cos_r * yy + sin_r * xx
        y_ship = -sin_r * yy + cos_r * xx

        # Calculate individual wake components
        kelvin = self._kelvin_wake(x_ship, y_ship)
        bow = self._bow_wave(x_ship, y_ship)
        turbulent = self._turbulent_core(x_ship, y_ship)
        
        # Blend components with smooth transitions
        wake = self._blend_wake_components(x_ship, y_ship, kelvin, bow, turbulent)

        self.z_ship = wake
    
    def _blend_wake_components(
        self,
        x_rel: np.ndarray,
        y_rel: np.ndarray,
        kelvin: np.ndarray,
        bow: np.ndarray,
        turbulent: np.ndarray,
    ) -> np.ndarray:
        """
        Smoothly blend wake components to avoid discontinuities.
        
        Uses smooth transition functions (tanh) to blend between:
        - Bow wave (near ship)
        - Kelvin wake (behind ship)
        - Turbulent core (centerline behind ship)
        
        Key principle: weights always sum to maintain continuity.
        """
        # Longitudinal blending: smooth transition from bow to wake components
        # Use wider transition region for smoother blending
        transition_center = -self.length_ship / 2.0
        transition_width = self.length_ship / 3.0  # Wider for smoother blend
        
        # Smooth sigmoid transition (0 at bow, 1 well behind stern)
        bow_to_wake = 0.5 * (1.0 + np.tanh((-y_rel - transition_center) / transition_width))
        
        # Lateral blending for turbulent core: Gaussian centered on centerline
        # This defines where turbulent core has influence
        lateral_width = self.beam_ship / 2.5  # Slightly wider for smoother transition
        turbulent_lateral = np.exp(-0.5 * (x_rel / lateral_width) ** 2)
        
        # Longitudinal activation for turbulent: starts gradually behind stern
        turbulent_start_center = -self.length_ship / 3.0
        turbulent_start_width = self.length_ship / 5.0
        turbulent_longitudinal = 0.5 * (1.0 + np.tanh((-y_rel - turbulent_start_center) / turbulent_start_width))
        
        # Combined turbulent influence (both lateral and longitudinal)
        turbulent_weight = turbulent_lateral * turbulent_longitudinal * 0.6  # Scale to max 0.6
        
        # Behind stern: blend between kelvin and turbulent
        # Weight for kelvin in wake region (reduced where turbulent is strong)
        kelvin_weight = bow_to_wake * (1.0 - turbulent_weight)
        
        # Weight for bow (fades as we go behind)
        bow_weight = 1.0 - bow_to_wake
        
        # Ensure weights are properly normalized in wake region
        # This maintains smooth continuity
        wake_region_weight_sum = kelvin_weight + turbulent_weight
        
        # Composite wake: weighted sum with normalized weights
        wake = (
            bow * bow_weight +  # Bow wave (strong near ship, fades behind)
            kelvin * kelvin_weight +  # Kelvin wake (grows behind ship, reduced at centerline)
            turbulent * turbulent_weight  # Turbulent core (centerline behind ship)
        )
        
        return wake

    def _kelvin_wake(self, x_rel: np.ndarray, y_rel: np.ndarray) -> np.ndarray:
        """
        Kelvin wake pattern with smooth transitions.
        Computes wake for regions behind ship with gradual onset.
        """
        wake = np.zeros_like(x_rel)
        
        # Use soft threshold instead of hard cutoff for smoother blending
        # Compute for extended region and use smooth start ramp
        behind = y_rel < -self.length_ship / 3.0  # Start slightly earlier
        if not np.any(behind):
            return wake

        x_b = x_rel[behind]
        y_b = y_rel[behind]
        
        # Distance from wake origin (stern)
        wake_origin = -self.length_ship / 2.0
        distance = np.abs(y_b - wake_origin)
        half_width = distance * np.tan(self.kelvin_angle)
        
        # Instead of hard cutoff, use a soft transition beyond the Kelvin angle
        # Extend the calculation region to include gradual falloff
        extended_width = half_width * 1.5  # Allow 50% beyond theoretical Kelvin angle
        in_region = np.abs(x_b) <= extended_width
        
        if not np.any(in_region):
            return wake

        x_w = x_b[in_region]
        y_w = distance[in_region]
        width = half_width[in_region]

        # Wave patterns with smoother wavelengths
        transverse_lambda = self.length_ship * 0.6  # Longer wavelength = smoother
        divergent_lambda = self.length_ship * 0.25  # Slightly longer
        
        # Add phase smoothing to reduce sharp gradients
        transverse = np.sin(2.0 * np.pi * y_w / transverse_lambda)
        divergent = np.sin(2.0 * np.pi * y_w / divergent_lambda)

        # Smooth angular envelope: gradually fade beyond the Kelvin angle
        # This creates a realistic smooth transition rather than sharp cutoff
        normalized_x = np.abs(x_w) / (width + 1e-8)
        
        # Main angular envelope: smooth transition from 0.8 to 1.2 times Kelvin angle
        angular_envelope = np.where(
            normalized_x <= 0.8,
            1.0,  # Full strength inside 80% of Kelvin angle
            np.exp(-((normalized_x - 0.8) ** 2) / (0.3 ** 2))  # Smooth Gaussian falloff beyond
        )
        
        # Edge enhancement for divergent waves (more prominent near Kelvin angle)
        edge_factor = np.exp(-((normalized_x - 0.9) ** 2) / (0.2 ** 2))
        
        # Improved lateral decay: use wider spread for more realistic wake
        # Scale with distance to allow wake to naturally expand
        lateral_scale = np.maximum(width / 2.0, self.beam_ship / 3.0)
        lateral_decay = np.exp(-0.5 * (x_w ** 2) / (lateral_scale ** 2))
        
        # Longitudinal decay
        longitudinal_decay = np.exp(-y_w / (self.wake_length / 3.0))
        
        # Smooth start ramp: gradually increase wake strength from stern
        # This prevents discontinuity at the wake origin
        start_distance = self.length_ship / 6.0  # Ramp up over ~18m for 110m ship
        start_ramp = np.tanh(y_w / start_distance)  # Smooth 0->1 transition

        # Combine patterns with improved weighting
        # Transverse waves dominant in center, divergent waves more prominent at edges
        transverse_weight = 0.6 + 0.2 * (1.0 - normalized_x.clip(0, 1))
        divergent_weight = 0.4 + 0.4 * edge_factor
        
        pattern = transverse_weight * transverse + divergent_weight * divergent
        
        # Apply all envelope functions including smooth start
        wake_height = (
            self.wake_amplitude 
            * pattern 
            * angular_envelope  # Smooth angular boundary
            * lateral_decay 
            * longitudinal_decay
            * start_ramp  # Smooth longitudinal start
        )

        wake_segment = np.zeros(np.count_nonzero(behind))
        wake_segment[in_region] = wake_height
        wake[behind] = wake_segment
        return wake

    def _bow_wave(self, x_rel: np.ndarray, y_rel: np.ndarray) -> np.ndarray:
        """
        Bow wave with smooth Gaussian profile.
        Extends computation beyond hard boundaries to enable smooth blending.
        """
        bow = np.zeros_like(x_rel)
        
        # Compute everywhere with Gaussian falloff (no hard mask)
        # This allows natural smooth decay instead of abrupt cutoff
        sigma_x = self.beam_ship / 2.0
        sigma_y = self.length_ship / 3.5  # Slightly wider for smoother transition
        amplitude = 1.5 * self.wake_amplitude
        
        # Gaussian profile centered at ship
        bow = amplitude * np.exp(
            -0.5 * ((x_rel / sigma_x) ** 2 + (y_rel / sigma_y) ** 2)
        )
        
        # Optional: gentle forward cutoff to avoid unrealistic bow wave ahead of ship
        forward_cutoff = 0.5 * (1.0 - np.tanh((y_rel + self.length_ship / 3.0) / (self.length_ship / 8.0)))
        bow *= forward_cutoff
        
        return bow

    def _turbulent_core(self, x_rel: np.ndarray, y_rel: np.ndarray) -> np.ndarray:
        """
        Turbulent wake core with smooth Gaussian profile.
        Negative amplitude represents the depression behind the vessel.
        Uses very smooth transitions to avoid discontinuities.
        """
        turbulence = np.zeros_like(x_rel)
        
        # Lateral Gaussian profile centered on ship centerline
        # Wider sigma for smoother lateral transition
        sigma_x = self.beam_ship / 2.8
        lateral = np.exp(-0.5 * (x_rel / sigma_x) ** 2)
        
        # Longitudinal profile with multiple smooth components
        y_start = -self.length_ship / 3.0  # Start position
        
        # Smooth activation using tanh (smoother than step function)
        start_width = self.length_ship / 8.0
        activation = 0.5 * (1.0 + np.tanh((-y_rel - y_start) / start_width))
        
        # Gradual decay with distance behind ship
        y_behind = np.maximum(0, -y_rel - y_start)
        decay_length = self.length_ship * 3.0  # Longer for gentler decay
        longitudinal = np.exp(-y_behind / decay_length)
        
        # Additional smoothing near the start to prevent any sharp onset
        # Quadratic ramp over initial region
        ramp_length = self.length_ship / 4.0
        near_start = y_behind < ramp_length
        if np.any(near_start):
            ramp_factor = (y_behind[near_start] / ramp_length) ** 2
            longitudinal_array = longitudinal.copy()
            longitudinal_array[near_start] *= ramp_factor
            longitudinal = longitudinal_array
        
        # Combine all factors with reduced amplitude for gentler effect
        amplitude = -0.4 * self.wake_amplitude  # Reduced from -0.5 for smoother integration
        turbulence = amplitude * lateral * longitudinal * activation
        
        return turbulence

if __name__ == "__main__":
    # Example 1: Using factory method (recommended for most users)
    # print("Example 1: Factory method with ship wake")
    # ocean = OceanSurface.with_ship_wake(
    #     ship_velocity=12.0,
    #     ship_length=110.0,
    #     num_grid=400,
    #     scene_length=250.0,
    #     wind_speed=12.0,
    #     wave_amplitude=0.5,
    #     ship_position=(0.0, 0.0),
    #     ship_heading=0.0,
    #     random_seed=13,
    # )
    
    # # Add swell using the interface
    # ocean.add_swell(
    #     amplitude=0.05,
    #     wavelength=80.0,
    #     direction=(0.8, 0.2),
    # )
    
    # Example 2: Using config objects (for advanced users)
    # ocean_config = OceanConfig(
    #     num_grid=400,
    #     scene_length=250.0,
    #     wind_speed=12.0,
    #     wave_amplitude=0.5,
    #     choppiness=0.1,
    #     wind_direction=(1.0, 0.2),
    #     smooth=True,
    #     random_seed=12,
    # )
    
    # wake_config = WakeConfig(
    #     enabled=True,
    #     velocity_ship=12.0,
    #     length_ship=110.0,
    #     beam_ship=20.3,
    #     draft_ship=9.5,
    #     initial_position=(0.0, 0.0),
    #     rotation=0.0,
    #     amplitude_scale=1.5,
    # )
    
    # swell_config = SwellConfig(
    #     enabled=True,
    #     amplitude=0.05,
    #     wavelength=80.0,
    #     direction=(0.8, 0.2),
    # )
    
    # ocean = OceanSurface(ocean_config, wake_config, swell_config)
    
    # Example 3: Creating separate Wake and adding it
    print("Example 3: Creating separate Wake and adding it. Wakes are static, not moving with time.")
    scene_length = 250.0
    num_grid = 400
    ocean = OceanSurface.calm_sea(num_grid=num_grid, scene_length=scene_length)
    wake = Wake.create(
        ship_length=110.0,
        ship_velocity=12,
        scene_length=scene_length,
        num_grid=num_grid,
        ship_position=(0.0, 0.0),
        ship_heading=45,
    )
    wake.set_static(True) 
    ocean.add_wake(wake)

    wake2 = Wake.create(
        ship_length=160.0,
        ship_velocity=8,
        scene_length=scene_length,
        num_grid=num_grid,
        ship_position=(100, 0.0),
        ship_heading=-90,
    )
    wake2.set_static(True) 
    ocean.add_wake(wake2)
 
    ocean.update_wake_position = False  # Disable automatic position updates for static wakes

    # Example 4: Using preset sea states
    # ocean = OceanPresets.from_beaufort(9, num_grid=512, scene_length=2000)
    
    # ocean_config = OceanConfig(num_grid=400,scene_length=250.)
    # ocean_config.wind_speed = 12.0
    # ocean_config.wave_amplitude = 0.5
    # ocean_config.wind_direction = (0.5, 0.5)

    # ocean = OceanSurface(ocean_config=ocean_config)
    # ocean.add_swell(amplitude=0.1,wavelength= 80,direction =(1.0, 0.0))

    
    # Start with basic ocean
    # ocean = OceanSurface.rough_sea(num_grid=512, scene_length=2000.0)

    # # Simulate scenario where ships appear over time
    # ships = [
    #     {"length": 120.0, "velocity": 18.0, "position": (0, -600), "heading": 0},
    #     {"length": 80.0, "velocity": 12.0, "position": (400, 0), "heading": 90},
    #     {"length": 60.0, "velocity": 10.0, "position": (-300, 200), "heading": 180},
    # ]

    # for i, ship in enumerate(ships):
    #     wake = Wake.create(
    #         ship_length=ship["length"],
    #         ship_velocity=ship["velocity"],
    #         scene_length=2000.0,
    #         num_grid=512,
    #         ship_position=ship["position"],
    #         ship_heading=ship["heading"],
    #     )
    #     ocean.add_wake(wake)
    #     print(f"Added ship {i+1}: {ship['length']}m vessel at {ship['position']}")
    
    # # Add swell using the interface
    # ocean.add_swell(
    #     amplitude=0.05,
    #     wavelength=80.0,
    #     direction=(0.8, 0.2),
    # )

    dt = 0.1

    plotter = pv.Plotter()
    plotter.show_grid()
    output_dir = os.path.join("..", "output", "seastate3")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plotter.open_movie(os.path.join(output_dir, "ocean_animation.mp4"))
    grid = ocean.create_ocean_surface_plot(0.0)
    surf = plotter.add_mesh(grid, scalars="Height", cmap="bone", show_edges=False)
    # plot bounds of surf
    print(f'Ocean surface bounds: {surf.bounds}')
    plotter.add_scalar_bar("Wave Height")
    # add axis
    plotter.add_axes()

    for frame in range(300):
        t = frame * dt
        grid = ocean.create_ocean_surface_plot(t)
        surf.mapper.dataset = grid
        plotter.write_frame()

    plotter.close()