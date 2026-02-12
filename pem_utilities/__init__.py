# """
# Perceive EM Utilities Package

# A comprehensive library for electromagnetic simulation utilities including:
# - Antenna device modeling and far field data processing
# - Beamforming algorithms
# - Material properties and ITU standards
# - Coordinate system transformations
# - Visualization tools
# - Post-processing utilities
# - Communication and radar signal processing
# """

# __version__ = "1.0.0"
# __author__ = "ANSYS"

# # Import commonly used modules for easier access
# from . import utils
# from . import antenna_device
# from . import materials
# from . import coordinate_system
# from . import beamformer
# from . import far_fields
# from . import heat_map
# from . import primitives
# from . import post_processing

# # Make key classes/functions available at package level
# try:
#     from .antenna_device import *
#     from .utils import *
#     from .materials import *
#     from .coordinate_system import *
# except ImportError:
#     # Handle cases where some dependencies might not be available
#     pass

# __all__ = [
#     'utils',
#     'antenna_device', 
#     'materials',
#     'coordinate_system',
#     'beamformer',
#     'far_fields',
#     'heat_map',
#     'primitives',
#     'post_processing'
# ]