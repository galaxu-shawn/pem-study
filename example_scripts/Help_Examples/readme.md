# Help Examples

This folder contains comprehensive example scripts that demonstrate how to use the core utilities and modules of Perceive EM. These examples are designed to help users understand the fundamental building blocks and common patterns for the simulation workflow.

## Overview

The Help Examples provide step-by-step demonstrations of key Perceive EM components with detailed explanations, best practices, and practical usage patterns. Each example is self-contained and includes extensive documentation to help users understand both basic usage and advanced features.

## Available Examples

### Basic Simulation Examples

#### `simple_radar_example.py`
The most basic radar simulation workflow demonstrating minimal setup requirements. Shows how to create a simple scene with one target and one radar, configure a pulsed Doppler waveform, run the simulation, and retrieve results. This is the ideal starting point for users new to radar simulations.

#### `simple_p2p_example.py`
The most basic platform-to-platform (communications) simulation workflow. Demonstrates how to set up separate transmitter and receiver antennas, enable coupling between them, and run P2P simulations for wireless communication analysis.

### Core Utilities Examples

#### `actors_utilities_example.py`
Comprehensive demonstration of the Actors utilities module for managing scene objects. Covers different methods of adding actors (from STL files, JSON configurations, primitives), setting positions and orientations, handling actor animations, and managing specialized actor types like vehicles, pedestrians, and quadcopters.
* Types of Actors:
    * Generators 
        * Any mesh that can be dynamically updated at each call. Seastate is an example of a goemetry that can be dynamically updated. Mesh is offset when time parameters is passed into generator object
        * Primitives
            * Utility to generate basic CAD objects. Some allow for generation based on RCS value, ie. Corner Reflector, Plate, Cube...etc.
    * CAD (STL/OBJ/GLTF/VTP/FACET)
        * VTP Note: This format can include embedded material properties, format is generated from Material Segmentation AI workflow.
        * Material properties must be defined during import, if not it will assume PEC.
    * DAE/DAEZ
        * Collada, blender legacy format for animated sequences. Format used for Carnegie Mellon Motion capture database
    * USD
        * Limited USD format input, animations not tested
    * JSON
        * Nested CAD files with material properties. Maintains parent/child relationship.
        * Supports special vehicle types, quadcopters, UAV, helicopters, vehicles...etc where motion of rotating objects is automatically updated with time.
        * Intial velocities and positions of children objects can be included
    * Other - a few other various example have not been formalized, but are possible, ie. geotiff, mitsuba, xml, ply, webcam


#### `materials_utilities_example.py`
Complete guide to the Materials utilities for managing electromagnetic material properties. Demonstrates the MaterialManager class, ITU standard materials, frequency-dependent materials, custom material creation, multi-layer materials, and material assignment workflows.

#### `antenna_insertion_methods_example.py`
Detailed demonstration of various methods for inserting antennas into simulations. Covers JSON-based antenna configurations, helper functions for single antennas, antenna arrays, multi-channel radar arrays, antenna positioning and orientation, and different antenna pattern types.

#### `waveform_definition_methods_example.py`
Comprehensive guide to defining Waveform objects for different simulation types. Demonstrates PulsedDoppler waveforms for radar, FMCW waveforms for continuous wave radar, communication waveforms for P2P simulations, and custom waveform configurations.

#### `simulation_options_example.py`
Complete demonstration of the SimulationOptions class for configuring simulation parameters. Covers ray spacing and density settings, reflection and transmission limits, GPU configuration, geometrical optics blockage, field of view settings, and performance optimization strategies.

#### `model_visualization_example.py`
Comprehensive guide to the ModelVisualization module for 3D scene visualization and animation. Demonstrates camera controls, data overlay capabilities, point cloud visualization for radar targets, video generation, and various visualization modes for both development and presentation purposes.

## Usage Patterns

Each example follows a consistent structure:

1. **Setup**: Import utilities and configure paths
2. **Demonstration**: Step-by-step feature demonstrations with explanations
3. **Best Practices**: Recommended usage patterns and common pitfalls
4. **Summary**: Key takeaways and workflow recommendations

## Getting Started

For users new to Perceive EM:

1. Start with `simple_radar_example.py` or `simple_p2p_example.py` depending on your application
2. Review `actors_utilities_example.py` to understand scene management
3. Explore `materials_utilities_example.py` for proper material handling
4. Use `antenna_insertion_methods_example.py` to configure your antenna systems
5. Optimize your simulation setup with `simulation_options_example.py`
6. Visualize results and dynamic scenes with `model_visualization_example.py`

## Running Examples

Each example can be run independently:

```bash
cd Help_Examples
python simple_radar_example.py
python actors_utilities_example.py
# ... etc
```

Examples will create output files in the `../output/` directory and provide detailed console output explaining each step.

## Educational Approach

These examples are designed as educational tools that:

- Explain the "why" behind each configuration choice
- Provide multiple approaches to accomplish the same task
- Include real-world usage scenarios and recommendations
- Demonstrate both basic and advanced features
- Show proper error handling and validation
- Include performance considerations and optimization tips

## Integration with Main Examples

The patterns and techniques demonstrated in these Help Examples are used throughout the main example scripts in other folders. Understanding these fundamental utilities will help you better understand and modify the more complex simulation scenarios in folders like `Radar Examples`, `P2P Examples`, and `Heatmap Examples`.