#!/usr/bin/env python3
"""
Test script for the MVG to FFD converter.
This script will help you locate your MVG file and convert it to FFD format.
"""

import os
from convert_mvg_to_ffd import convert_mvg_to_ffd

def main():
    # Try to find the MVG file in common locations
    possible_paths = [
        "C:/Users/asligar/OneDrive - ANSYS, Inc/Documents/Scripting/mvg_dipole_2450.txt",
        "../../mvg_dipole_2450.txt",
        "./mvg_dipole_2450.txt",
        "../mvg_dipole_2450.txt"
    ]
    
    mvg_file = None
    for path in possible_paths:
        if os.path.exists(path):
            mvg_file = path
            print(f"Found MVG file at: {mvg_file}")
            break
    
    if mvg_file is None:
        print("MVG file not found in expected locations.")
        print("Please specify the full path to your mvg_dipole_2450.txt file:")
        mvg_file = input("Enter path: ").strip().strip('"')
        
        if not os.path.exists(mvg_file):
            print(f"Error: File {mvg_file} does not exist.")
            return
    
    # Set output file name
    base_name = os.path.splitext(os.path.basename(mvg_file))[0]
    ffd_file = f"{base_name}_converted.ffd"
    
    print(f"Converting {mvg_file} to {ffd_file}...")
    
    try:
        convert_mvg_to_ffd(mvg_file, ffd_file)
        print(f"\nConversion completed successfully!")
        print(f"Output file: {os.path.abspath(ffd_file)}")
        
        # Show first few lines of the output file
        print("\nFirst 10 lines of the output FFD file:")
        with open(ffd_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"{i+1:2d}: {line.rstrip()}")
                
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()