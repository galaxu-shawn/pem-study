# -*- coding: utf-8 -*-
"""
ANSYS Perceive EM API Core Module

This module provides core functionality for initializing and configuring the ANSYS Perceive EM API.
It handles API detection, licensing, and version management across multiple ANSYS versions.

Features:
- Automatic detection of installed ANSYS Perceive EM versions (241, 242, 251, 252)
- Cross-platform support (Windows and Linux)
- Flexible API path configuration via JSON settings
- License management for both standard and HPC licenses
- Support for both P2P and RADAR APIs (version dependent)
- Singleton pattern for RssPy to ensure single instance across all scripts

Author: asligar
Created: Fri Jan 26 15:00:00 2024

Usage:
    
Configuration:
    Create an 'api_settings.json' file in the same directory to customize settings:
    {
        "api_path": "path/to/api",
        "version": "252",
        "licensingclient_path": "default",
        "license_feature": "perceive_em",
        "hpc_pack_or_pool": "pack"
    }

    OR

        {
        "api_path": "default",
        "version": "default",
        "licensingclient_path": "default",
        "license_feature": "perceive_em",
        "hpc_pack_or_pool": "pack"
    }
"""

import sys
import os
import json
import glob
from pem_utilities.path_helper import get_repo_paths

# Global singleton variables for RssPy
_rss_py_module = None
_rss_api_instance = None
_api_initialized_version = None

    
class Perceive_EM_API:
    """
    Main class for managing ANSYS Perceive EM API initialization and configuration.
    
    This class encapsulates all functionality for detecting, configuring, and initializing
    the ANSYS Perceive EM API across different versions and platforms.
    
    Attributes:
        api: The initialized API object (RssApi or P2PApi depending on version/type)
        version (str): The API version being used
        api_path (str): Path to the API installation directory
        is_windows (bool): Whether running on Windows platform
        installed_versions (list): List of detected installed API versions
    """
    
    # Class-level constants
    POSSIBLE_VERSIONS = [251, 252, 261]


    def __init__(self, config_file=None,reset=False):
        """
        Initialize the Perceive EM API manager.
        
        Args:
            config_file (str, optional): Path to custom configuration JSON file.
                                       If None, looks for 'api_settings.json' in module directory.
        """


        
        if _rss_api_instance is None or reset:
            # Initialize platform detection
            self.is_windows = os.name == 'nt'
            self.python_version_major = sys.version_info[0]
            self.python_version_minor = sys.version_info[1]
            self.python_version = f"{self.python_version_major}.{self.python_version_minor}"
            # Initialize version tracking
            self.installed_versions = []
            self._scan_installed_versions()
            
            # Initialize configuration attributes
            self.api_path = None
            self.licensingclient_path = None
            self.license_feature = 'perceive_em'
            self.hpc_pack_or_pool = None
            self.version = '252'  # Default version
            self.RssPy = None

            # Load configuration and initialize API
            self._load_configuration(config_file)


            if int(self.version) <= int(252):
                support_python_version = ["3.10"]
            else:
                support_python_version = ["3.10", "3.11", "3.12"]

            if self.python_version not in support_python_version:
                print(f'WARNING: Python version {self.python_version} may not be supported with ANSYS Perceive EM API {self.version}')
                print(f'Supported Python versions are: 3.10 for Perceive EM 25.2 and earlier. Perceive EM 26.1 and higher supports {", ".join(["3.10", "3.11", "3.12"])}')
            
            self._initialize_api()
        else:
            # Reuse existing API instance if already initialized
            self.RssPy = _rss_py_module
            self.pem = _rss_api_instance
            self.version = _api_initialized_version
    
    def _scan_installed_versions(self):
        """Scan for installed API versions on the system."""
        for ver in self.POSSIBLE_VERSIONS:
            if self.is_windows:
                # Windows default installation location
                if ver<=252:
                    default_location = f'C:/Program Files/AnsysEM/Perceive_EM/v{ver}/Win64/lib/'
                else:
                    # default_location = f'C:/Program Files/AnsysEM/Perceive_EM/v{ver}/Win64/lib/RssPy/'
                    default_location = f'C:/Program Files/ANSYS Inc/v{ver}/PerceiveEM/lib/RssPy/'
                if os.path.exists(os.path.join(default_location, 'RssPy.pyd')):
                    self.installed_versions.append(ver)
                elif os.path.exists(os.path.join(default_location, f'RssPy{self.python_version_major}{self.python_version_minor}.pyd')):
                    self.installed_versions.append(ver)
            else:
                # Linux default installation location
                if ver<=252:
                    default_location = f'/opt/AnsysEM/Perceive_EM/v{ver}/Linux64/lib/'
                else:
                    default_location = f'/opt/ANSYS Inc/v{ver}/PerceiveEM/lib/RssPy/'
                if os.path.exists(os.path.join(default_location, 'RssPy.so')):
                    self.installed_versions.append(ver)
                elif os.path.exists(os.path.join(default_location, f'RssPy{self.python_version_major}{self.python_version_minor}.so')):
                    self.installed_versions.append(ver)
    
    def _load_configuration(self, config_file=None):
        """
        Load configuration from JSON file.
        
        Args:
            config_file (str, optional): Path to configuration file.
        """

        # Get all paths
        paths = get_repo_paths()
        print(f"Repository root: {paths.repo_root}")

        if config_file is None:
            config_file = os.path.abspath(os.path.join(paths.repo_root, 'api_settings.json'))
        
        if os.path.exists(config_file):
            with open(config_file) as f:
                api_settings_json = json.load(f)
            
            # Extract configuration values
            if 'api_path' in api_settings_json:
                self.api_path = api_settings_json['api_path']
            if 'version' in api_settings_json:
                self.version = self.convert_version_str(api_settings_json['version'])
            

            # Extract additional configuration
            if 'licensingclient_path' in api_settings_json:
                self.licensingclient_path = api_settings_json['licensingclient_path']
            if 'license_feature' in api_settings_json:
                self.license_feature = api_settings_json['license_feature']
            if 'hpc_pack_or_pool' in api_settings_json:
                self.hpc_pack_or_pool = api_settings_json['hpc_pack_or_pool']
    
    def _initialize_api(self):
        """Initialize the API object and configure licensing."""
        # Resolve API path
        self.api_path = self._resolve_api_path()
        if self.api_path is None:
            raise RuntimeError('API path could not be resolved')

        # Add API path to Python path for imports
        sys.path.append(self.api_path)
        
        # Configure licensing client
        self._configure_licensing_client()
        
        # Import and create API object
        self.pem = self._create_api_object()
        
        # Configure license settings
        self._configure_license_settings()
        
        # Display API information
        print(self.pem.copyright())
        print(self.pem.version(True))
    
    def _resolve_api_path(self):
        """Resolve the API path from configuration or auto-detection."""
        if self.api_path is None:
            # Path was not defined, try to find it automatically
            resolved_path = self.find_api_location()
            if resolved_path is None:
                print('API path not found in configuration')
                return None
            return resolved_path
        else:
            # Path was defined, validate it exists
            resolved_path = self.find_api_location(self.api_path)
            if resolved_path is None:
                print(f'API path: {self.api_path} not found in configuration')
                print(f'Configure api_settings.json to point to location of API lib')
                return None
            else:
                print(f'API path: {resolved_path}')
                return resolved_path
    
    def _configure_licensing_client(self):
        """Configure the licensing client."""
        if self.licensingclient_path is None or self.licensingclient_path.lower() == 'default':
            # Try default location first
            if self.is_windows:
                if int(self.version)<=252:
                    found = self.find_licensingclient(path=f'C:/Program Files/AnsysEM/Perceive_EM/v{self.version}/Win64/lib/')
                else:
                    found = self.find_licensingclient(path=f'C:/Program Files/ANSYS Inc./v{self.version}/PerceiveEM/lib/')
            else:
                if int(self.version)<=252:
                    found = self.find_licensingclient(path=f'/opt/AnsysEM/Perceive_EM/v{self.version}/Linux64/lib/')
                else:
                    found = self.find_licensingclient(path=f'/opt/ANSYS Inc./v{self.version}/PerceiveEM/lib/')
            
            if not found:
                # If no path is found, search common locations
                found = self.find_licensingclient()
                if not found:
                    print(f'Licensingclient not found, please specify path')
    
    def _create_api_object(self):
        """Create and return the appropriate API object based on version."""
        global _rss_py_module, _rss_api_instance, _api_initialized_version
        

        version_as_num = int(self.version)
        
        # Check if we already have a RssPy instance for this version
        if (_rss_api_instance is not None and 
            _api_initialized_version == self.version and
            _rss_py_module is not None):
            print(f"Reusing existing RssPy instance for version {self.version}")
            self.RssPy = _rss_py_module
            return _rss_api_instance
        
        # Import RssPy if not already imported or version changed
        if _rss_py_module is None or _api_initialized_version != self.version:
            pyversion_str = f'{self.python_version_major}{self.python_version_minor}'

            try:
                import RssPy
                print(f"Imported RssPy for version {self.version}")
            except ImportError as e:
                raise RuntimeError(f'Python version {self.python_version} is not supported with ANSYS Perceive EM API {self.version}')
                
            _rss_py_module = RssPy
        
        # Create new API instance only if needed
        if _rss_api_instance is None or _api_initialized_version != self.version:
            _rss_api_instance = _rss_py_module.RssApi()
            _api_initialized_version = self.version
            print(f"Created new RssPy.RssApi() instance for version {self.version}")
        
        # Unified API for version 251 and later
        self.RssPy = _rss_py_module  # Store reference for isOK method and RssPy types
        return _rss_api_instance
    
    def _configure_license_settings(self):
        """Configure API license settings."""
        # Configure API license mode
        if self.license_feature is not None:
            if self.license_feature == 'perceive_em':
                self.isOK(self.pem.selectApiLicenseMode(self.RssPy.ApiLicenseMode.PERCEIVE_EM))
        
        # Configure HPC license preference
        if self.hpc_pack_or_pool is not None:
            if self.hpc_pack_or_pool.lower() == 'pool':
                self.isOK(self.pem.selectPreferredHpcLicense(self.RssPy.HpcLicenseType.HPC_ANSYS))
                print(f'HPC Pool license selected')
            else:
                self.isOK(self.pem.selectPreferredHpcLicense(self.RssPy.HpcLicenseType.HPC_ANSYS_PACK))
                print(f'HPC Pack license selected')

    def isOK(self, r_gpu_call_stat):
        """
        Check the status of GPU API calls and handle errors/warnings.
        
        This function validates the return status from GPU API calls and provides
        appropriate error handling. It will print warnings for non-critical issues
        and exit the program for critical errors.
        
        Args:
            r_gpu_call_stat: The status returned from a GPU API call
            
        Returns:
            None: Function returns early for OK status
            
        Exits:
            Program exits if a critical error is encountered
        """
        if r_gpu_call_stat == self.RssPy.RGpuCallStat.OK:
            return
        elif r_gpu_call_stat == self.RssPy.RGpuCallStat.RGPU_WARNING:
            # Non-critical warning - print and continue
            print(self.pem.getLastWarnings())
        elif r_gpu_call_stat == True:
            return
        else:
            # Critical error - print error and exit
            print(self.pem.getLastError())
            sys.exit()
    
    def convert_version_str(self, version_str):
        """
        Convert version string to standardized format.
        
        Handles version string conversion and validation. If 'default' or None is passed,
        returns the latest installed version. Removes dots from version strings for
        consistency with API expectations.
        
        Args:
            version_str (str or None): Version string to convert (e.g., "25.1", "251", "default")
            
        Returns:
            str or None: Standardized version string without dots, or None if no versions found
            
        Examples:
            >>> convert_version_str("25.1")
            "251"
            >>> convert_version_str("default")
            "252"  # (assuming 252 is the latest installed)
        """
        # If version is default or None, use the latest installed version
        if version_str == 'default' or version_str is None:
            if len(self.installed_versions) == 0:
                print('No API versions found')
                return None
            version_str = str(self.installed_versions[-1])  # Get latest version
            version_str = version_str.replace(".", "")
            return version_str
        
        # Convert to string and remove dots for consistency
        version_str = str(version_str)
        version_str = version_str.replace(".", "")
        return version_str
    
    def find_api_location(self, path_to_check=None):
        """
        Locate the ANSYS Perceive EM API installation directory.
        
        Searches for the API files (RssPy.pyd on Windows, RssPy.so on Linux) in various
        locations including default installation paths and relative to the script location.
        
        Args:
            path_to_check (str, optional): Specific path to check for API files.
                                         If None or 'default', searches standard locations.
            
        Returns:
            str or None: Absolute path to the directory containing API files,
                        or None if not found
            
        Search Order:
            1. Default installation directory for current version
            2. Relative paths from script location
            3. User-specified path (if provided)
        """
        if path_to_check is None or path_to_check.lower() == 'default':
            # Check default installation location first
            if self.is_windows:
                if int(self.version)<=252:
                    default_location = f'C:/Program Files/AnsysEM/Perceive_EM/v{self.version}/Win64/lib/'
                    if os.path.exists(os.path.join(default_location, 'RssPy.pyd')):
                        return os.path.abspath(default_location)
                else:
                    default_location = f'C:/Program Files/ANSYS Inc./v{self.version}/PerceiveEM/lib/RssPy/'
                    if os.path.exists(os.path.join(default_location, f'RssPy{self.python_version_major}{self.python_version_minor}.pyd')):
                        return os.path.abspath(f'C:/Program Files/ANSYS Inc./v{self.version}/PerceiveEM/lib')
            else:
                if int(self.version)<=252:
                    default_location = f'/opt/AnsysEM/Perceive_EM/v{self.version}/Linux64/lib/'
                    if os.path.exists(os.path.join(default_location, 'RssPy.so')):
                        return os.path.abspath(default_location)
                else:
                    default_location = f'/opt/ANSYS Inc/v{self.version}/Perceive_EM/lib/RssPy/'
                    if os.path.exists(os.path.join(default_location, f'RssPy{self.python_version_major}{self.python_version_minor}.so')):
                        return os.path.abspath(default_location)
            
            # Check relative paths from script location
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

            if self.is_windows:
                all_files = glob.glob(os.path.join(root_dir, '*/**/RssPy.pyd'))
            else:
                all_files = glob.glob(os.path.join(root_dir, '*/**/RssPy.so'))
                
            if len(all_files) > 1:
                print(f'Multiple API files found, using {all_files[0]}')
            elif len(all_files) == 0:
                print('API file not found')
                return None
            return os.path.dirname(all_files[0])
        else:
            # Check the user-specified path
            if self.is_windows:
                if os.path.exists(os.path.join(path_to_check, 'RssPy.pyd')):
                    return os.path.abspath(path_to_check)
            else:
                if os.path.exists(os.path.join(path_to_check, 'RssPy.so')):
                    return os.path.abspath(path_to_check)

            # Check relative to script location if user didn't provide the full path
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
            root_dir = os.path.abspath(os.path.join(root_dir, path_to_check))
            
            if self.is_windows:
                if os.path.exists(os.path.join(root_dir, 'RssPy.pyd')):
                    return root_dir
                else:
                    print(f'API file not found at {root_dir}')
                    return None
            else:
                if os.path.exists(os.path.join(root_dir, 'RssPy.so')):
                    return root_dir
                else:
                    print(f'API file not found at {root_dir}')
                    return None
    
    def find_licensingclient(self, path=None):
        """
        Locate and configure the ANSYS licensing client.
        
        Searches for the licensing client in various standard locations and sets
        appropriate environment variables based on the API version. Different
        versions use different environment variable names.
        
        Args:
            path (str, optional): Specific path to check for licensing client.
                                 If None, searches standard locations.
            
        Returns:
            bool: True if licensing client found and configured, False otherwise
            
        Environment Variables Set:
            - RTR_LICENSE_DIR: For versions <= 251
            - ANSYSCL252_DIR: For versions > 251
        """
        if path is None:
            # Define possible licensing client locations
            if self.is_windows:
                possible_paths = [
                    f'C:/Program Files/AnsysEM/Perceive_EM/v{self.version}/Win64/lib/',
                    'C:/Program Files/AnsysEM/v261/Win64/',
                    'C:/Program Files/AnsysEM/v252/Win64/',
                    'C:/Program Files/AnsysEM/v251/Win64/',
                    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir + '/api/')),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir + '/api/'))
                ]
            else:
                possible_paths = [f'/opt/AnsysEM/Perceive_EM/v{self.version}/Linux64/lib/']

            # Search through possible paths
            for search_path in possible_paths:
                if os.path.exists(os.path.join(search_path, 'licensingclient')):
                    # Set appropriate environment variable based on version
                    if float(self.version) <= 251:
                        os.environ["RTR_LICENSE_DIR"] = search_path
                    else:
                        os.environ[f"ANSYSCL{self.version}_DIR"] = search_path
                    print(f'Found licensingclient at {search_path}')
                    return True
        elif os.path.exists(os.path.join(path, 'licensingclient')):
            # User-specified path contains licensing client
            if float(self.version) <= 251:
                os.environ["RTR_LICENSE_DIR"] = os.path.join(path, 'licensingclient')
            else:
                os.environ[f"ANSYSCL{self.version}_DIR"] = os.path.join(path, 'licensingclient')
            print(f'Found licensingclient at {path}')
            return True
        else:
            return False


# =============================================================================
# Backward Compatibility Support
# =============================================================================

# # Create a global instance for backward compatibility
# _global_api_manager = None
# pem = None

# def _ensure_global_api():
#     """Ensure the global API instance is initialized."""
#     global _global_api_manager, api
#     if _global_api_manager is None:
#         _global_api_manager = Perceive_EM_API()
#         pem = _global_api_manager.pem

# # Initialize global API for backward compatibility
# _ensure_global_api()

# # Export the isOK function for backward compatibility
# def isOK(r_gpu_call_stat):
#     """
#     Global isOK function for backward compatibility.
    
#     Args:
#         r_gpu_call_stat: The status returned from a GPU API call
#     """
#     _ensure_global_api()
#     return _global_api_manager.isOK(r_gpu_call_stat)





