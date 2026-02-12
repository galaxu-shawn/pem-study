"""
ITU Material Properties Calculator

This module calculates electromagnetic material properties based on ITU-R Recommendation P.2040-3
for various building and ground materials. The calculations are frequency-dependent and follow
the ITU standard formulations for permittivity and conductivity.

The module provides functions to:
- Calculate frequency-dependent material properties (permittivity, conductivity)
- Generate material property dictionaries for electromagnetic simulations
- Export material properties to JSON format

Reference:
https://www.itu.int/rec/R-REC-P.2040-3-202308-I/en

Author: Arien Sligar
Date: June 10, 2025
"""
import os
import numpy as np
import json
from pem_utilities.path_helper import get_repo_paths # common paths to reference
paths = get_repo_paths() 

# ITU-R P.2040-3 Material Properties Table
# Each material has parameters a, b, c, d for calculating frequency-dependent properties:
# - Relative permittivity: εr = a * f^b
# - Conductivity: σ = c * f^d (S/m)
# - min_freq, max_freq: Valid frequency range in GHz
material_table = {}

# Reference materials with their ITU coefficients
material_table['vacuum'] = {'a':1., 'b': 0., 'c': 0.,'d': 0.,'min_freq':0.,'max_freq':1000.}
material_table['concrete'] = {'a':5.24, 'b': 0., 'c': 0.0462,'d': 0.7822,'min_freq':1.,'max_freq':100.}
material_table['brick'] = {'a':3.91, 'b': 0., 'c': 0.0238,'d': 0.16,'min_freq':1.,'max_freq':40.}
material_table['plasterboard'] = {'a':2.73, 'b': 0., 'c': 0.0085,'d': 0.9395,'min_freq':1.,'max_freq':100.}
material_table['wood'] = {'a':1.99, 'b': 0., 'c': 0.0047,'d': 1.0718,'min_freq':0.001,'max_freq':100.}
material_table['glass'] = {'a':6.31, 'b': 0., 'c': 0.0036,'d': 1.3394,'min_freq':0.1,'max_freq':100.}
material_table['ceiling_board'] = {'a':1.48, 'b': 0., 'c': 0.0011,'d': 1.075,'min_freq':1.,'max_freq':100.}
material_table['chipboard'] = {'a':2.58, 'b': 0., 'c': 0.0217,'d': 0.78,'min_freq':1.,'max_freq':100.}
material_table['plywood'] = {'a':2.71, 'b': 0., 'c': 0.33,'d': 1.075,'min_freq':1.,'max_freq':40.}
material_table['marble'] = {'a':7.074, 'b': 0., 'c': 0.0055,'d': 0.9262,'min_freq':1.,'max_freq':60.}
material_table['floorboard'] = {'a':3.66, 'b': 0., 'c': 0.0044,'d': 1.3515,'min_freq':50.,'max_freq':100.}
material_table['very_dry_ground'] = {'a':3.0, 'b': 0., 'c': 0.00015,'d': 2.52,'min_freq':1.,'max_freq':10.}
material_table['medium_dry_ground'] = {'a':15., 'b': -0.1, 'c': 0.035,'d': 1.63,'min_freq':1.,'max_freq':10}
material_table['wet_ground'] = {'a':30., 'b': -0.4, 'c': 0.15,'d': 1.3,'min_freq':1.,'max_freq':10}
# Metal approximated as high conductivity material
material_table['metal'] = {'a':1., 'b': 0, 'c': 10e7,'d': 0,'min_freq':1.,'max_freq':100}


def calculate_properties(freq):
    """
    Calculate electromagnetic material properties for all materials at a given frequency.
    
    This function computes frequency-dependent electromagnetic properties according to
    ITU-R P.2040-3 recommendations. The calculations include:
    - Real part of relative permittivity (εr')
    - Imaginary part of relative permittivity (εr'')
    - Conductivity (sigma)
    
    Args:
        freq (float): Frequency in GHz for which to calculate properties
        
    Returns:
        dict: Dictionary containing material properties for all materials.
              Each material entry includes:
              - thickness: Material thickness (-1 for infinite)
              - relEpsReal: Real part of relative permittivity
              - relEpsImag: Imaginary part of relative permittivity
              - relMuReal: Real part of relative permeability (always 1.0)
              - relMuImag: Imaginary part of relative permeability (always 0.0)
              - conductivity: Material conductivity in S/m
              - coating_idx: Material index for simulation
              
    Note:
        The function also includes a 'pec' (Perfect Electric Conductor) entry
        for simulation convenience.
    """
    # Initialize with Perfect Electric Conductor (PEC) reference
    properties = {'pec': {"coating_idx": 0}}
    
    # Calculate properties for each material in the ITU table
    for n, entry in enumerate(material_table):
        # Extract ITU coefficients for current material
        a = material_table[entry]['a']  # Permittivity coefficient
        b = material_table[entry]['b']  # Permittivity frequency exponent
        c = material_table[entry]['c']  # Conductivity coefficient
        d = material_table[entry]['d']  # Conductivity frequency exponent
        
        # Calculate frequency-dependent properties using ITU formulations
        er = a*np.power(freq,b)        # Real permittivity: εr = a * f^b
        sigma = c*np.power(freq,d)     # Conductivity: σ = c * f^d (S/m)
        
        # Calculate imaginary permittivity from conductivity
        # Formula: εr'' = -17.98 * σ / f (conversion factor accounts for units)
        er_imag = 17.98*sigma/freq*-1

        # Store complete material properties
        properties[entry] = {
            "thickness": -1,           # -1 indicates infinite thickness
            "relEpsReal": er,          # Real part of relative permittivity
            "relEpsImag": er_imag,     # Imaginary part of relative permittivity
            "relMuReal": 1.0,          # Real part of relative permeability (non-magnetic)
            "relMuImag": 0.0,          # Imaginary part of relative permeability
            "conductivity": sigma,      # Conductivity in S/m
            "coating_idx": n+1         # Unique material index (0 reserved for PEC)
        }

    return properties


def generate_itu_materials_dict(frequency):
    """
    Generate a formatted material properties dictionary for a specific frequency.
    
    This function creates a structured dictionary containing all ITU material
    properties calculated at the specified frequency. The output format is
    suitable for use in electromagnetic simulation software.
    
    Args:
        frequency (float): Frequency in GHz for material property calculations
        
    Returns:
        dict: Formatted dictionary with structure:
              - name: Descriptive name including frequency
              - version: Format version number
              - materials: Complete material properties dictionary
              
    Example:
        >>> mat_dict = generate_itu_materials_dict(2.4)
        >>> print(mat_dict['name'])
        'material_properties_ITU_2.4GHz'
    """
    # Calculate material properties at specified frequency
    properties = calculate_properties(frequency)

    # Create formatted dictionary with metadata
    mat_dict = {
        "name": f"material_properties_ITU_{frequency}GHz",
        "version": 1,
        "materials": properties
    }

    return mat_dict


def main():
    """
    Main function providing command-line interface for material property calculation.
    
    This function prompts the user for a frequency input, calculates the corresponding
    ITU material properties, and saves the results to a JSON file. The output file
    is named according to the specified frequency.
    
    The function handles:
    - User input for frequency in GHz
    - Material property calculation
    - JSON file output with formatted results
    - User feedback on completion
    
    Output:
        Creates a JSON file named 'material_properties_ITU_{frequency}GHz.json'
        containing all calculated material properties.
    """
    # Get frequency input from user
    frequency = float(input("Enter frequency (GHz): "))

    # Calculate material properties for the specified frequency
    properties = calculate_properties(frequency)

    # Create formatted output dictionary
    mat_dict = {
        "name": f"material_properties_ITU_{frequency}GHz",
        "version": 1,
        "materials": properties
    }

    # Generate output filename based on frequency
    output_file = os.path.join(paths.materials,f"material_properties_ITU_{frequency}GHz.json")
    
    # Save results to JSON file with readable formatting
    with open(output_file, 'w') as f:
        json.dump(mat_dict, f, indent=4)

    # Provide user feedback
    print(f"Material properties calculated for {frequency} GHz and saved in {output_file}.")


if __name__ == "__main__":
    main()
