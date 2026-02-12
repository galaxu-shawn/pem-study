"""
Materials Utilities Example Script
==================================

This script demonstrates comprehensive usage of the materials.py utilities module.
The Materials utilities provide a powerful framework for managing electromagnetic 
material properties, including predefined material libraries, ITU standard materials,
and custom material creation for electromagnetic simulations.

Main Components:
- MaterialManager class: Main interface for material management
- MatData class: Data structure for material properties
- ITU materials: Frequency-dependent materials based on ITU-R P.2040-3 standards
- Material library: JSON-based material definitions

Author: Example Script
Date: June 2025
"""

from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API
from pem_utilities.materials import MaterialManager, MatData
from pem_utilities.calculate_ITU_materials import calculate_properties, generate_itu_materials_dict

def main():
    """
    Comprehensive demonstration of Materials utilities functionality

    PEC is default material at index 0. Or if unassigned, simulation assumes PEC.
    Absorber is also a default material that is always availble (index -1 if no custom material is assigned).
    """
    
    print("=" * 60)
    print("MATERIALS UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # ========================================================================
    # 1. INITIALIZE MATERIAL MANAGER - BASIC
    # ========================================================================
    print("\n1. Basic Material Manager Initialization")
    print("-" * 45)
    
    # Method 1: Default initialization with standard material library
    print("   a) Default material manager:")
    mat_manager_basic = MaterialManager()
    print("     ✓ Material manager created with default material_library.json")
    
    # Show available materials from default library
    all_materials = mat_manager_basic.get_all_material_names()
    print(f"     ✓ Loaded {len(all_materials)} materials from library")
    print(f"     ✓ Sample materials: {all_materials[:5]}...")
    
    # ========================================================================
    # 2. INITIALIZE WITH ITU MATERIALS
    # ========================================================================
    print("\n2. ITU Materials Initialization")
    print("-" * 35)
    
    # Method 2: Generate ITU materials based on specific frequency
    print("   a) ITU materials generation:")
    frequency_ghz = 2.4  # 2.4 GHz frequency
    mat_manager_itu = MaterialManager(
        generate_itu_materials=True,
        itu_freq_ghz=frequency_ghz
    )
    print(f"     ✓ Generated ITU materials for {frequency_ghz} GHz")
    print(f" Use this library when intializing Actors, ie. all_actors = Actors(material_manager=mat_manager_itu) ")

    # Show ITU material properties calculation (just for demonstration), library has already been generated and passed into Actors()
    print("   b) ITU material properties calculation:")
    itu_properties = calculate_properties(frequency_ghz)
    
    # Display properties for a few key materials
    sample_materials = ['concrete', 'wood', 'glass', 'metal']
    for material in sample_materials:
        if material in itu_properties:
            props = itu_properties[material]
            print(f"     {material.capitalize():12}: εr'={props['relEpsReal']:.2f}, "
                  f"εr''={props['relEpsImag']:.2f}, σ={props['conductivity']:.3e} S/m")
    
    # 

    # ========================================================================
    # 3. GETTING MATERIAL INDICES AND PROPERTIES
    # ========================================================================
    print("\n3. Getting Material Indices and Properties")
    print("-" * 45)
    
    # Method 1: Get material index by name
    print("   a) Getting material indices:")
    
    common_materials = ['pec', 'concrete', 'wood', 'glass', 'metal']
    material_indices = {}
    
    for material in common_materials:
        try:
            idx = mat_manager_itu.get_index(material)
            material_indices[material] = idx
            print(f"     {material:10}: index {idx}")
        except Exception as e:
            print(f"     {material:10}: not found - {e}")
    
    # Method 2: Show material properties details
    print("   b) Material properties details:")
    for material_name in ['concrete', 'wood', 'pec']:
        if material_name in mat_manager_itu.all_materials:
            mat_data = mat_manager_itu.all_materials[material_name]
            print(f"     {material_name.capitalize():10}:")
            print(f"       Thickness: {mat_data.thickness}")
            print(f"       εr (real): {mat_data.rel_eps_real:.3f}")
            print(f"       εr (imag): {mat_data.rel_eps_imag:.3f}")
            print(f"       Conductivity: {mat_data.conductivity:.3e} S/m")
            print(f"       Coating idx: {mat_data.coating_idx}")
    

    print(f"Usage of this library within Actors would be like:")
    print(">>>> actor_name = all_actors.add_actor(filename='path/to/concrete_model.obj',mat_idx=mat_manager.get_index('concrete'))")

    # ========================================================================
    # 4. CREATING CUSTOM MATERIALS
    # ========================================================================
    print("\n4. Creating Custom Materials")
    print("-" * 30)
    
    # Method 1: Create simple dielectric material, add to material manager


    print("   a) Creating custom dielectric material:")
    custom_dielectric_props = MatData.from_dict({
        "thickness": -1,        # Infinite thickness
        "relEpsReal": 4.5,      # Relative permittivity (real)
        "relEpsImag": -0.1,      # Relative permittivity (imaginary)
        "relMuReal": 1.0,       # Relative permeability (real)
        "relMuImag": 0.0,       # Relative permeability (imaginary)
        "conductivity": 0.001  # Conductivity in S/m
    } )
    
    mat_manager_basic.create_material('custom_dielectric', custom_dielectric_props) # this will create a custom dielectric at index N, where N is the next available index in the material library
    custom_idx = mat_manager_basic.get_index('custom_dielectric')
    print(f"     ✓ Created custom_dielectric with index {custom_idx}")
    
    # Method 2: Create multi-layer material
    print("   b) Creating multi-layer material:")
    multilayer_props = MatData.from_dict({
        "thickness": [0.001, 0.002],           # Layer thicknesses in meters
        "relEpsReal": [2.2, 4.0],              # Permittivity for each layer
        "relEpsImag": [-0.01, -0.05],            # Loss for each layer
        "relMuReal": [1.0, 1.0],               # Permeability (non-magnetic)
        "relMuImag": [0.0, 0.0],               # Permeability (imaginary)
        "conductivity": [0.0001, 0.001]      # Conductivity for each layer
    }
    )
    mat_manager_basic.create_material('multilayer_composite', multilayer_props)
    multilayer_idx = mat_manager_basic.get_index('multilayer_composite')
    print(f"     ✓ Created multilayer_composite with index {multilayer_idx}")
    
    # Method 3: Create rough surface material
    print("   c) Creating rough surface material:")
    height_standard_dev = 17 # mm
    corr_length = 0.05 # meter
    roughness = height_standard_dev*1e-3/corr_length # rougness (unitless)
    rough_surface_props = MatData.from_dict({
        "thickness": -1,
        "relEpsReal": 3.0,
        "relEpsImag": -0.05,
        "relMuReal": 1.0,
        "relMuImag": 0.0,
        "conductivity": 0.01,
        "height_standard_dev": height_standard_dev,  # Surface roughness std dev in mm
        "roughness": roughness            # Roughness parameter
    })
    
    mat_manager_basic.create_material('my_rough_concrete', rough_surface_props)
    rough_idx = mat_manager_basic.get_index('my_rough_concrete')
    print(f"     ✓ Created rough_concrete with index {rough_idx}")
    
    
    # ========================================================================
    # 10. SUMMARY AND BEST PRACTICES
    # ========================================================================
    print("\n10. Summary and Best Practices")
    print("-" * 35)
    
    print("   Key points for using Materials utilities:")
    print("   • Use MaterialManager() to initialize material management")
    print("   • Generate ITU materials for frequency-specific simulations")
    print("   • Use get_index(material_name) to get material indices for assignments")
    print("   • Create custom materials with create_material() for special cases")
    print("   • Consider frequency dependence for accurate material modeling")
    print("   • Use 'pec' for perfect electric conductors (index 0)")
    print("   • Multi-layer materials support complex material structures")
    print("   • Surface roughness can be included for realistic scattering")
    print("   • Always validate material names before using in simulations")
    
    # Show final material count
    total_materials = len(mat_manager_itu.get_all_material_names())
    print(f"\n   Final material library size: {total_materials} materials")
    
    
    print("\n" + "=" * 60)
    print("MATERIALS UTILITIES DEMONSTRATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    """
    Run the materials utilities demonstration
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()