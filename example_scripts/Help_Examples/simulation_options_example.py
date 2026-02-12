"""
Simulation Options Example Script
================================

This script demonstrates the usage of the SimulationOptions class, which provides
a convenient interface for configuring simulation parameters for electromagnetic
field calculations. It acts as a wrapper around the api_core API calls.

The SimulationOptions class manages key simulation parameters including:
- Ray spacing and density settings
- Maximum reflections and transmissions
- GPU device configuration
- Geometrical Optics (GO) blockage settings
- Field of view and bounding box parameters

Author: Example Script
Date: June 2025
"""

import os
import sys
from pathlib import Path
import numpy as np

from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 



def main():
    """
    Comprehensive demonstration of SimulationOptions usage
    """
    
    print("=" * 70)
    print("SIMULATION OPTIONS DEMONSTRATION")
    print("=" * 70)
    
    # ========================================================================
    # 1. BASIC SIMULATION OPTIONS CREATION
    # ========================================================================
    print("\n1. Creating SimulationOptions Object")
    print("-" * 40)
    
    print("   Creating a new SimulationOptions instance...")
    sim_options = SimulationOptions()
    
    print("   ✓ SimulationOptions object created successfully")
    print("   ✓ Default values are automatically set")
    print("   ✓ API reference is automatically configured")
    
    # ========================================================================
    # 2. RAY SPACING AND DENSITY CONFIGURATION
    # ========================================================================
    print("\n2. Ray Spacing and Density Configuration")
    print("-" * 45)
    
    print("   Ray spacing and density control simulation accuracy and performance.")
    print("   Smaller spacing = higher accuracy but longer computation time.")
    
    # Method 2A: Setting ray spacing directly
    print("\n   a) Setting ray spacing directly:")
    ray_spacing_meters = 0.1  # 10 cm spacing between rays on bounding box
    sim_options.ray_spacing = ray_spacing_meters
    print(f"     ✓ Ray spacing set to: {sim_options.ray_spacing} meters")
    print(f"     ✓ Ray density cleared: {sim_options.ray_density}")
    print("     • Smaller values may increase accuracy but impact simulation speed")
    print("     • Typical range: 0.01m to 1.0m depending on frequency")
    
    # Method 2B: Setting ray density (overrides ray spacing)
    # the Perceive EM API only takes ray spacing, but here we are just calculating the spacing from the density
    print("\n   b) Setting ray density (calculated approach):")
    center_freq = 77e9  # 77 GHz automotive radar
    wavelength = 3e8 / center_freq  # Calculate wavelength
    ray_density = 2.0  # rays per wavelength
    calculated_ray_spacing = np.sqrt(2) * wavelength / ray_density
    sim_options.ray_spacing = calculated_ray_spacing
    print(f"     ✓ Center frequency: {center_freq/1e9:.1f} GHz")
    print(f"     ✓ Wavelength: {wavelength*1000:.2f} mm")
    print(f"     ✓ Ray density: {ray_density} rays/wavelength")
    print(f"     ✓ Calculated ray spacing: {calculated_ray_spacing*1000:.2f} mm")
    
    # Method 2C: Frequency-dependent ray spacing examples
    print("\n   c) Frequency-dependent ray spacing examples:")
    frequencies = [
        (2.4e9, "WiFi 2.4 GHz"),
        (5.8e9, "WiFi 5.8 GHz"), 
        (24e9, "K-band automotive"),
        (77e9, "W-band automotive"),
        (94e9, "W-band imaging")
    ]
    
    print(f"     {'Frequency':<20} {'Wavelength':<12} {'Ray Spacing':<15} {'Application'}")
    print(f"     {'-'*20} {'-'*12} {'-'*15} {'-'*20}")
    
    for freq, app in frequencies:
        wl = 3e8 / freq
        rs = np.sqrt(2) * wl / 2.0  # 2 rays per wavelength
        print(f"     {freq/1e9:6.1f} GHz         {wl*1000:8.2f} mm    {rs*1000:10.2f} mm    {app}")
    
    # ========================================================================
    # 2D. RAY SHOOT METHOD CONFIGURATION
    # ========================================================================
    print("\n2D. Ray Shoot Method Configuration")
    print("-" * 38)
    
    print("   Ray shoot method determines how rays are launched from antennas.")
    print("   Different methods optimize for different simulation scenarios.")
    
    # Method 2D-A: Grid-based ray shooting
    print("\n   a) Grid-based ray shooting:")
    sim_options.ray_shoot_method = "grid"
    print(f"     ✓ Ray shoot method set to: '{sim_options.ray_shoot_method}'")
    print("     • Uniform grid pattern of rays launched from antenna")
    print("     • Regular spacing ensures consistent sampling")
    print("     • Good for general-purpose electromagnetic simulations")
    print("     • Predictable ray distribution and computational load")
    print("     • Recommended for most antenna pattern analysis")
    print("     • No center frequency required")
    
    # Method 2D-B: Shooting and Bouncing Rays (SBR)
    print("\n   b) Shooting and Bouncing Rays (SBR):")
    print("     • IMPORTANT: SBR method requires center frequency to be set!")
    print("     • Must create SimulationOptions with center_freq parameter for SBR")
    
    # Create a new SimulationOptions with center frequency for SBR  shoot method demonstration
    center_freq_sbr = 77e9  # 77 GHz automotive radar
    sim_options_sbr = SimulationOptions(center_freq=center_freq_sbr)
    sim_options_sbr.ray_shoot_method = "sbr"
    print(f"     ✓ Created SimulationOptions with center_freq: {center_freq_sbr/1e9:.1f} GHz")
    print(f"     ✓ Ray shoot method set to: '{sim_options_sbr.ray_shoot_method}'")
    print("     • Optimized ray launching based on antenna pattern")
    print("     • Concentrates rays in high-gain antenna directions")
    print("     • More efficient for highly directional antennas")
    print("     • Can reduce computation time for focused beams")
    print("     • Particularly effective for radar and satellite applications")
    print("     • WARNING: SBR is a BETA feature in version 25.2")
    
    # Method 2D-C: Proper initialization examples
    print("\n   c) Proper initialization for each method:")
    print("     Grid method initialization:")
    print("       sim_options = SimulationOptions()  # No center_freq needed")
    print("       sim_options.ray_shoot_method = 'grid'")
    print("     ")
    print("     SBR method initialization:")
    print("       sim_options = SimulationOptions(center_freq=77e9)  # Must provide center_freq")
    print("       sim_options.ray_shoot_method = 'sbr'")
    print("     ")
    print("     Alternative SBR approach:")
    print("       sim_options = SimulationOptions()")
    print("       sim_options.center_freq = 77e9  # Set center frequency first")
    print("       sim_options.ray_shoot_method = 'sbr'  # Then set method")
    
    # Method 2D-D: Method selection recommendations
    print("\n   d) Ray shoot method recommendations:")
    ray_scenarios = [
        ("Omnidirectional antennas", "grid", "Uniform coverage needed"),
        ("Horn antennas", "sbr", "Directional beam concentration"),
        ("Patch arrays", "sbr", "High gain, focused radiation"),
        ("Dipole antennas", "grid", "Moderate directivity"),
        ("Radar systems", "sbr", "Highly directional beams"),
        ("WiFi/cellular base stations", "grid", "Sector coverage patterns"),
        ("Satellite dishes", "sbr", "Pencil beam antennas"),
        ("RFID readers", "grid", "Near-field applications")
    ]
    
    print(f"     {'Application':<25} {'Method':<6} {'Reasoning'}")
    print(f"     {'-'*25} {'-'*6} {'-'*25}")
    
    for application, method, reasoning in ray_scenarios:
        print(f"     {application:<25} {method:<6} {reasoning}")
    
    # Method 2D-E: Performance comparison
    print("\n   e) Performance characteristics:")
    print("     Grid method:")
    print("       + Consistent ray distribution")
    print("       + Predictable computational load")
    print("       + Good for low-to-moderate gain antennas")
    print("       + No frequency dependency for setup")
    print("       - May waste computation on low-gain directions")
    print("     ")
    print("     SBR method:")
    print("       + Efficient for high-gain antennas")
    print("       + Concentrates rays where antenna radiates most")
    print("       + Can significantly reduce computation time")
    print("       + Adaptive to antenna radiation patterns")
    print("       - Ray distribution varies with antenna pattern")
    print("       - Requires center frequency for initialization")
    print("       - BETA feature (may have limitations)")
    
    # Method 2D-F: Error handling example
    print("\n   f) Common initialization errors to avoid:")
    print("     ❌ INCORRECT - Will cause ValueError:")
    print("       sim_options = SimulationOptions()  # No center_freq provided")
    print("       sim_options.ray_shoot_method = 'sbr'  # ERROR: requires center frequency")
    print("     ")
    print("     ✓ CORRECT - Proper SBR initialization:")
    print("       sim_options = SimulationOptions(center_freq=77e9)")
    print("       sim_options.ray_shoot_method = 'sbr'  # Works correctly")
    
    # Continue using the original sim_options for the rest of the example (with grid method)
    sim_options.ray_shoot_method = "grid"  # Reset to grid for remaining examples
    
    # ========================================================================
    # 3. REFLECTION AND TRANSMISSION LIMITS
    # ========================================================================
    print("\n3. Reflection and Transmission Limits")
    print("-" * 42)
    
    print("   Control computational complexity vs. physical accuracy.")
    print("   Higher values capture more multipath but increase computation time.")
    
    # Method 3A: Maximum reflections
    print("\n   a) Maximum reflections:")
    max_reflections = 3
    sim_options.max_reflections = max_reflections
    print(f"     ✓ Max reflections set to: {sim_options.max_reflections}")
    print("     • Bounce limit for rays off surfaces")
    print("     • Typical values: 1-5 for most applications")
    print("     • Urban environments may need 3-5 for accuracy")
    print("     • Indoor scenarios often use 5-10")
    
    # Method 3B: Maximum transmissions
    print("\n   b) Maximum transmissions:")
    max_transmissions = 1
    sim_options.max_transmissions = max_transmissions
    print(f"     ✓ Max transmissions set to: {sim_options.max_transmissions}")
    print("     • Limit for rays passing through materials")
    print("     • Typical values: 0-2 for most scenarios")
    print("     • Set to 0 if no transparent materials")
    print("     • Building penetration studies may use 2-3")
    
    # Method 3C: Application-specific recommendations
    print("\n   c) Application-specific recommendations:")
    applications = [
        ("Automotive radar", 3, 1, "Moderate multipath, some glass"),
        ("Indoor WiFi", 5, 2, "Many reflections, wall penetration"),
        ("Outdoor 5G", 3, 0, "Building reflections, no penetration"),
        ("RCS measurement", 1, 0, "Direct scattering only"),
        ("Through-wall radar", 2, 3, "Limited reflections, wall penetration"),
        ("SAR imaging", 1, 0, "Single-bounce scattering")
    ]
    
    print(f"     {'Application':<18} {'Refl':<5} {'Trans':<6} {'Reasoning'}")
    print(f"     {'-'*18} {'-'*5} {'-'*6} {'-'*30}")
    
    for app, refl, trans, reason in applications:
        print(f"     {app:<18} {refl:<5} {trans:<6} {reason}")
    
    # ========================================================================
    # 4. GEOMETRICAL OPTICS (GO) BLOCKAGE
    # ========================================================================
    print("\n4. Geometrical Optics (GO) Blockage")
    print("-" * 39)
    
    print("   GO blockage determines when shadowing effects are applied.")
    print("   Controls whether rays are blocked by intervening objects.")
    
    # Method 4A: Disable GO blockage
    print("\n   a) Disable GO blockage:")
    sim_options.go_blockage = -1
    print(f"     ✓ GO blockage set to: {sim_options.go_blockage}")
    print("     • -1 = Disabled (no shadowing)")
    print("     • Rays pass through all objects")
    print("     • Fastest computation, least realistic")
    print("     • Use for initial testing or open environments")
    
    # Method 4B: Enable GO blockage at first bounce
    print("\n   b) Enable GO blockage at first bounce:")
    sim_options.go_blockage = 0
    print(f"     ✓ GO blockage set to: {sim_options.go_blockage}")
    print("     • 0 = Blockage starts at bounce 0 (direct path)")
    print("     • Most physically accurate")
    print("     • Direct rays blocked by obstacles")
    print("     • Use for realistic propagation modeling")
    
    # Method 4C: Enable GO blockage at higher bounces
    print("\n   c) Enable GO blockage at higher bounces:")
    sim_options.go_blockage = 1
    print(f"     ✓ GO blockage set to: {sim_options.go_blockage}")
    print("     • 1+ = Blockage starts at specified bounce number")
    print("     • Allows direct path and first reflections")
    print("     • Blocks higher-order multipath")
    print("     • Compromise between accuracy and speed")
    
    # Method 4D: GO blockage recommendations
    print("\n   d) GO blockage recommendations:")
    scenarios = [
        ("Free space/open field", -1, "No obstacles to block rays"),
        ("Urban outdoor", 0, "Buildings create realistic shadows"),
        ("Indoor environments", 1, "Walls block but allow some multipath"),
        ("Vehicle-to-vehicle", 0, "Cars create significant blockage"),
        ("Satellite communications", -1, "Minimal atmospheric blockage"),
        ("Radar cross-section", -1, "Clear line-of-sight measurements")
    ]
    
    print(f"     {'Scenario':<22} {'Setting':<8} {'Reasoning'}")
    print(f"     {'-'*22} {'-'*8} {'-'*30}")
    
    for scenario, setting, reason in scenarios:
        print(f"     {scenario:<22} {setting:<8} {reason}")
    
    # ========================================================================
    # 5. FIELD OF VIEW CONFIGURATION
    # ========================================================================
    print("\n5. Field of View Configuration")
    print("-" * 34)
    
    print("   Controls antenna pattern field of view for Tx simulation.")
    print("   Affects which directions antennas can radiate (not impacting receive).")
    print("   Can be set to 180° or 360° depending on application.")
    print("   180° = half-space (e.g., directional antennas)")
    print("   360° = full-space (e.g., omnidirectional antennas)")
    
    # Method 5A: 180-degree field of view
    print("\n   a) 180-degree field of view:")
    sim_options.field_of_view = 180
    print(f"     ✓ Field of view set to: {sim_options.field_of_view} degrees")
    print("     • Half-space radiation pattern")
    print("     • Antennas radiate only in +X direction")
    print("     • Faster computation, reduced ray count")
    print("     • Use for directional antennas or ground-plane scenarios")
    
    # Method 5B: 360-degree field of view
    print("\n   b) 360-degree field of view:")
    sim_options.field_of_view = 360
    print(f"     ✓ Field of view set to: {sim_options.field_of_view} degrees")
    print("     • Full-space radiation pattern")
    print("     • Antennas radiate in all directions")

    
    # ========================================================================
    # 6. GPU DEVICE CONFIGURATION
    # ========================================================================
    print("\n6. GPU Device Configuration")
    print("-" * 31)
    
    print("   Configure which GPU devices to use for simulation.")
    print("   Controls memory allocation and parallel processing.")
    
    # Method 6A: List available GPUs
    print("\n   a) Available GPU devices:")

    gpu_list = pem.listGPUs()
    print(f"     ✓ Available GPUs: {gpu_list}")
    if gpu_list:
        print("     • GPU acceleration available")
        print("     • Use gpu_device to select specific GPU")
    else:
        print("     • No GPUs detected")
        print("     • CPU-only simulation will be used")

    
    # Method 6B: Single GPU configuration
    print("\n   b) Single GPU configuration:")
    sim_options.gpu_device = 0  # Use GPU 0
    print(f"     ✓ GPU device set to: {sim_options.gpu_device}")
    print("     • Uses first available GPU")
    print("     • Default 90% memory quota")
    print("     • Automatic configuration on simulation start")
    
    # Method 6C: Multiple GPU configuration
    print("\n   c) Multiple GPU configuration:")
    # sim_options.gpu_device = [0, 1]  # Use GPUs 0 and 1
    print("     • Multi-GPU support: sim_options.gpu_device = [0, 1]")
    print("     • Distributes computation across GPUs")
    print("     • Requires compatible hardware")
    print("     • Can specify different quotas per GPU")
    
    # Method 6D: GPU quota configuration
    print("\n   d) GPU memory quota:")
    sim_options.gpu_quota = 0.8  # Use 80% of GPU memory
    print(f"     ✓ GPU quota set to: {sim_options.gpu_quota}")
    print("     • Controls percentage of GPU memory used")
    print("     • Range: 0.1 to 0.95 (10% to 95%)")
    print("     • Lower values leave memory for other applications")
    print("     • Higher values maximize simulation performance")
    
    # ========================================================================
    # 7. BOUNDING BOX CONFIGURATION
    # ========================================================================
    print("\n7. Bounding Box Configuration")
    print("-" * 33)
    
    print("   Controls simulation region limits for computational efficiency.")
    print("   Truncates geometry outside specified bounds.")
    
    # Method 7A: Disable bounding box
    print("\n   a) Disable bounding box (default):")
    sim_options.bounding_box = -1
    print(f"     ✓ Bounding box set to: {sim_options.bounding_box}")
    print("     • -1 = Disabled (no truncation)")
    print("     • All geometry is included in simulation")
    print("     • Use when entire scene is relevant")
    
    # Method 7B: Enable bounding box with size limit
    print("\n   b) Enable bounding box with size limit:")
    bounding_box_size = 1000.0  # 1000 meter cube
    sim_options.bounding_box = bounding_box_size
    print(f"     ✓ Bounding box set to: {sim_options.bounding_box} meters")
    print("     • Truncates geometry outside cube centered at origin")
    print("     • Reduces memory usage and computation time")
    print("     • Use for large scenes with distant irrelevant objects")
    
    # Method 7C: Bounding box recommendations
    print("\n   c) Bounding box size recommendations:")
    bb_scenarios = [
        ("Urban coverage", 5000, "City-scale propagation study"),
        ("Campus WiFi", 500, "University or corporate campus"),
        ("Indoor office", 100, "Single building floor"),
        ("Automotive radar", 300, "Local traffic environment"),
        ("Drone operations", 1000, "Flight corridor analysis"),
        ("Satellite coverage", -1, "Global or continental scale")
    ]
    
    print(f"     {'Scenario':<17} {'Size (m)':<10} {'Description'}")
    print(f"     {'-'*17} {'-'*10} {'-'*25}")
    
    for scenario, size, desc in bb_scenarios:
        size_str = "Disabled" if size == -1 else f"{size}"
        print(f"     {scenario:<17} {size_str:<10} {desc}")
    
    # ========================================================================
    # 8. BATCH PROCESSING CONFIGURATION
    # ========================================================================
    print("\n8. Batch Processing Configuration")
    print("-" * 37)
    
    print("   Controls memory management for large simulations.")
    print("   Splits ray processing into manageable chunks.")
    
    # Method 8A: Default batch configuration
    print("\n   a) Maximum ray batches:")
    max_batches = 25
    sim_options.max_batches = max_batches
    print(f"     ✓ Max batches set to: {sim_options.max_batches}")
    print("     • Controls memory usage vs. computation speed")
    print("     • Higher values = more memory, faster processing")
    print("     • Lower values = less memory, slower processing")
    print("     • Typical range: 10-100 depending on GPU memory")
    
    # Method 8B: Batch size recommendations
    print("\n   b) Batch size recommendations by GPU memory:")
    gpu_configs = [
        ("4 GB GPU", 10, "Conservative for older hardware"),
        ("8 GB GPU", 25, "Balanced for mid-range cards"),
        ("16 GB GPU", 50, "Aggressive for high-end cards"),
        ("24+ GB GPU", 100, "Maximum for workstation cards"),
        ("Multi-GPU", 200, "Distributed across devices")
    ]
    
    print(f"     {'Hardware':<12} {'Batches':<8} {'Usage Note'}")
    print(f"     {'-'*12} {'-'*8} {'-'*25}")
    
    for hardware, batches, note in gpu_configs:
        print(f"     {hardware:<12} {batches:<8} {note}")
    
    # ========================================================================
    # 9. AUTO-CONFIGURATION AND SETUP
    # ========================================================================
    print("\n9. Auto-Configuration and Setup")
    print("-" * 35)
    
    print("   Automatically applies all configured settings to the simulation.")
    print("   Must be called before running electromagnetic calculations.")
    
    # Method 9A: Manual configuration steps
    print("\n   a) Manual configuration approach:")
    print("     • Individual API calls for each setting")
    print("     • pem.setRaySpacing(value)")
    print("     • pem.setMaxNumRefl(value)")
    print("     • pem.setMaxNumTrans(value)")
    print("     • pem.setGPUDevices(devices, quotas)")
    print("     • More control but more complex")
    
    # Method 9B: Auto-configuration (recommended)
    print("\n   b) Auto-configuration (recommended):")
    print("     ✓ Calling sim_options.auto_configure_simulation()...")
    print("     • Applies all SimulationOptions settings")
    print("     • Configures GPU devices automatically")
    print("     • Sets up ray processing parameters")
    print("     • Validates configuration for consistency")
    
    # Note: We don't actually call this in the example to avoid affecting the pem state
    # sim_options.auto_configure_simulation()
    
    # Method 9C: Configuration verification
    print("\n   c) Configuration verification:")
    print("     • Use pem.isReady() to check configuration")
    print("     • Use pem.getLastWarnings() for issues")
    print("     • Use pem.reportSettings() for detailed status")
    
    # ========================================================================
    # 10. COMMON CONFIGURATION PATTERNS
    # ========================================================================
    print("\n10. Common Configuration Patterns")
    print("-" * 37)
    
    print("   Pre-configured settings for typical simulation scenarios.")
    
    # Method 10A: High accuracy configuration
    print("\n   a) High accuracy configuration:")
    high_accuracy = SimulationOptions()
    high_accuracy.ray_spacing = 0.01      # 1 cm ray spacing
    high_accuracy.max_reflections = 5     # Capture detailed multipath
    high_accuracy.max_transmissions = 2   # Include material penetration
    high_accuracy.go_blockage = -1         # Realistic shadowing
    high_accuracy.field_of_view = 360     # Full radiation pattern
    high_accuracy.max_batches = 100       # up to 100 batches (may use less if possible)
    
    print("     ✓ Ray spacing: 0.01m (1cm) - Very fine resolution")
    print("     ✓ Max reflections: 5 - Detailed multipath")
    print("     ✓ Max transmissions: 2 - Material penetration")
    print("     ✓ GO blockage: 1 - Realistic shadowing")
    print("     ✓ Field of view: 360° - Full pattern")
    print("     ✓ Max batches: 100 - Maximum memory usage")
    print("     • Use for: Research, validation, small scenes")
    
    # Method 10B: Balanced performance configuration
    print("\n   b) Balanced performance configuration:")
    balanced = SimulationOptions()
    balanced.ray_spacing = 0.1            # 10 cm ray spacing
    balanced.max_reflections = 3          # Moderate multipath
    balanced.max_transmissions = 1        # Limited penetration
    balanced.go_blockage = 1              # Selective shadowing
    balanced.field_of_view = 180          # Hemispherical pattern
    balanced.max_batches = 25             # Moderate memory usage
    
    print("     ✓ Ray spacing: 0.1m (10cm) - Good accuracy")
    print("     ✓ Max reflections: 3 - Essential multipath")
    print("     ✓ Max transmissions: 1 - Basic penetration")
    print("     ✓ GO blockage: 1 - Realistic shadowing")
    print("     ✓ Field of view: 180° - Directional pattern")
    print("     ✓ Max batches: 25 - Balanced memory")
    print("     • Use for: Engineering analysis, medium scenes")
    
    # Method 10C: Fast computation configuration
    print("\n   c) Fast computation configuration:")
    fast = SimulationOptions()
    fast.ray_spacing = 0.5                # 50 cm ray spacing
    fast.max_reflections = 1              # Direct + single bounce
    fast.max_transmissions = 0            # No penetration
    fast.go_blockage = -1                 # No shadowing
    fast.field_of_view = 180              # Hemispherical pattern
    fast.max_batches = 10                 # Minimal memory usage
    
    print("     ✓ Ray spacing: 0.5m (50cm) - Coarse resolution")
    print("     ✓ Max reflections: 1 - Direct + single bounce")
    print("     ✓ Max transmissions: 0 - No penetration")
    print("     ✓ GO blockage: -1 - disable GO Blockage shadowing")
    print("     ✓ Field of view: 180° - Directional pattern")
    print("     ✓ Max batches: 10 - Minimal memory")
    print("     • Use for: Initial testing, large scenes, real-time")
    
    
    
    # ========================================================================
    # 11. SUMMARY AND BEST PRACTICES
    # ========================================================================
    print("\n14. Summary and Best Practices")
    print("-" * 35)
    
    print("   Key principles for effective SimulationOptions usage:")
    
    best_practices = [
        "Always call auto_configure_simulation() before running",
        "Start with balanced settings, optimize based on results",
        "Scale ray spacing with frequency, you don't need exteremely small values",
        "Use appropriate reflections for environment complexity",
        "Enable GO blockage for realistic shadowing",
        "Monitor GPU memory usage and adjust batches accordingly",
        "Use bounding box for computational efficiency when large scenes don't impact scattering",
        "Validate settings with pem.isReady() before simulation",
        "Document configuration choices for reproducibility",
        "Profile different settings to find optimal balance"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        print(f"   {i:2d}. {practice}")
    
    print("\n   SimulationOptions parameter relationships:")
    print("   • Accuracy ∝ 1/ray_spacing (smaller = more accurate)")
    print("   • Computation time ∝ 1/ray_spacing² (quadratic scaling)")
    print("   • Multipath detail ∝ max_reflections (exponential growth)")
    print("   • Memory usage ∝ max_batches (linear scaling)")
    print("   • Realism ∝ go_blockage effectiveness")
    
    print("\n   Common simulation workflows:")
    print("   1. Create SimulationOptions()")
    print("   2. Set frequency-appropriate ray_spacing")
    print("   3. Configure reflections/transmissions for environment")
    print("   4. Enable appropriate go_blockage")
    print("   5. Set field_of_view for antenna type")
    print("   6. Configure GPU and memory settings")
    print("   7. Call auto_configure_simulation()")
    print("   8. Verify with pem.isReady()")
    print("   9. Run electromagnetic simulation")
    print("   10. Monitor performance and adjust if needed")
    
    print("\n" + "=" * 70)
    print("SIMULATION OPTIONS DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nAll SimulationOptions parameters demonstrated with examples.")
    print("Use these patterns as starting points for your simulations.")
    print("Remember to call auto_configure_simulation() before running!")


if __name__ == "__main__":
    """
    Run the simulation options demonstration
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()