#!/usr/bin/env python3
"""
Example: Path Resolution using Path Helper

This script demonstrates how to path resolution helper

Usage:
    from utilities.path_helper import setup_api_core, get_repo_paths
    api_core, RssPy, api = setup_api_core()
    paths = get_repo_paths()


"""
from pem_utilities.path_helper import get_repo_paths
from pem_utilities.pem_core import Perceive_EM_API


def main():
    """Demonstrate the new path helper system"""
    
    # Get all repository paths
    paths = get_repo_paths()
    
    print("=== Path Helper Demo ===")
    print(f"Repository root: {paths.repo_root}")
    print(f"Example scripts: {paths.example_scripts}")
    print(f"Models: {paths.models}")
    print(f"Materials: {paths.materials}")
    print(f"Output: {paths.output}")
    print(f"Antenna library: {paths.antenna_device_library}")
    print(f"Cache: {paths.cache}")
    
    pem_api_manager = Perceive_EM_API()
    pem = pem_api_manager.pem  # The configured API object


    print("\n=== API Objects Available ===")
    print(f"api_core module: {pem_api_manager}")
    print(f"RssPy: {pem}")
    print(f"API version: {pem_api_manager.version}")

    
    # Now you can use paths in your code
    model_file = paths.models / "some_model.stl"
    output_file = paths.output / "results.json"
    antenna_config = paths.antenna_device_library / "example_1tx_1rx.json"
    
    print(f"\n=== Example File Paths ===")
    print(f"Model file: {model_file}")
    print(f"Output file: {output_file}")
    print(f"Antenna config: {antenna_config}")
    
    return paths

if __name__ == "__main__":
    main()