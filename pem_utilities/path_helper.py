"""
Perceive EM Repository Path Helper

This module provides centralized path resolution for all scripts in the repository.
It automatically finds the repository root and key directories regardless of where
scripts are located in the directory hierarchy.

Key Features:
- Automatic repository root detection using marker files
- Consistent path resolution for perceive_em_utilities, models, output, and antenna directories
- Works from any depth in the repository
- Thread-safe singleton pattern for performance
- Easy integration with existing scripts

Usage:
    from perceive_em_utilities.path_helper import get_repo_paths, setup_api_core
    
    # Get all paths
    paths = get_repo_paths()
    print(f"Repository root: {paths.repo_root}")
    print(f"Models directory: {paths.models}")
    
    # Setup API core and get ready-to-use objects
    api_core, RssPy, api = setup_api_core()

Author: Enhanced by GitHub Copilot
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import threading

# Thread lock for singleton initialization
_lock = threading.Lock()
_instance = None

@dataclass
class RepoPaths:
    """Container for all repository paths"""
    repo_root: Path
    example_scripts: Path
    materials: Path
    models: Path
    output: Path
    antenna_device_library: Path
    cache: Path
    
    def __post_init__(self):
        """Ensure all paths are Path objects and create directories if needed"""
        # Convert string paths to Path objects
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))
        
        # Create output and cache directories if they don't exist
        self.output.mkdir(exist_ok=True)
        self.cache.mkdir(exist_ok=True)

def _find_repo_root(start_path: Path) -> Optional[Path]:
    """
    Find the repository root by looking for marker files/directories.
    
    Args:
        start_path: Path to start searching from (typically __file__ location)
        
    Returns:
        Path to repository root, or None if not found
    """

    repo_root_ov_ext = Path(__file__).parent.parent.parent.parent / 'synopsys.perceive_em_extension/data/'
    if repo_root_ov_ext.exists():
        print(f'Found repo root (likeyly PEM used within Omniverse Extension): {repo_root_ov_ext.resolve()}')
        return repo_root_ov_ext.resolve()

    # Marker files/directories that indicate repository root
    markers = [
        'pyproject.toml',
        'requirements.txt', 
        'example_scripts',
        'pem_utilities',
        'antenna_device_library',
        'api_settings.json'
        '.git'
    ]
    
    current_path = Path(start_path).resolve()
    
    possible_path = None
    # Walk up the directory tree
    for parent in [current_path] + list(current_path.parents):
        # Check if multiple markers exist (higher confidence)
        marker_count = sum(1 for marker in markers if (parent / marker).exists())
        possible_path = parent
        # If we find multiple markers, we're likely at the repo root
        if marker_count >= 2:
            return parent

   
    if marker_count == 1 and possible_path is not None:
        return possible_path
    
    return None

def _create_repo_paths(repo_root: Path) -> RepoPaths:
    """Create RepoPaths object with all standard paths"""
    return RepoPaths(
        repo_root=repo_root,
        example_scripts=repo_root / 'example_scripts',
        models=repo_root / 'models',
        materials=repo_root / 'materials',
        output=repo_root / 'output', 
        antenna_device_library=repo_root / 'antenna_device_library',
        cache=repo_root / 'cache'
    )

def get_repo_paths(root_directory=None) -> RepoPaths:
    """
    Get repository paths using singleton pattern for performance.

    root_directory: Optional path to override automatic root detection
                    This is useful for packaged deployments where antenna_device_library, materials...etc may be in a different location.

    Returns:
        RepoPaths object containing all standard repository paths

    Raises:
        RuntimeError: If repository root cannot be found
    """
    global _instance

    if _instance is None:
        with _lock:
            if _instance is None:  # Double-check pattern
                # Find repo root starting from this file's location
                if root_directory is not None:
                    repo_root = Path(root_directory)
                else:
                    repo_root = _find_repo_root(Path(__file__))
                print('#' * 40)
                print(f"Current file path helper: {Path(__file__).resolve()}")
                print('#' * 40)
                # print(f"Current sys.path: {sys.path}")


                if repo_root is None:
                    raise RuntimeError(
                        "Cannot find repository root. Make sure this script is "
                        "located within the perceive_em repository structure."
                    )
                else:
                    print(f"Detected repo root: {repo_root}")

                _instance = _create_repo_paths(repo_root)

    return _instance



def get_path(path_name: str) -> Path:
    """
    Get a specific path by name.
    
    Args:
        path_name: Name of the path ('models', 'output', 'utilities', etc.)
        
    Returns:
        Path object for the requested path
        
    Raises:
        AttributeError: If path_name is not a valid path
    """
    paths = get_repo_paths()
    
    if not hasattr(paths, path_name):
        available_paths = [field.name for field in paths.__dataclass_fields__.values()]
        raise AttributeError(
            f"Unknown path '{path_name}'. Available paths: {available_paths}"
        )
    
    return getattr(paths, path_name)

