# Perceive EM

Perceive EM is a highly performant SBR solver engine, delivered as a lightweight API. Performance is achieved through utilization of GPU and algorithmic enhancements. Enabling simulation of large-scale radar and communications scenarios.

This repository contains example scripts and documentation to help users get started with the Perceive EM API. These example scripts are intended to only be used as a demonstration of how the Perceive EM API can be utilized, and not a replacement to using the API directly. This repository is USE AT YOUR OWN RISK.

The API is used to define the scene tree, assign materials, and define radar and communication platforms. The repository includes a comprehensive utilities package (`pem_utilities`) that provides streamlined functions for antenna modeling, beamforming, materials, coordinate systems, visualization, and result processing.


## Getting Started
Below is a quick-start guide for using the API to help familiarize with the basics of using the API. More detailed help can be found within the gpuradar_test_252_winx64.zip archive. Within this zip file, see the document manual. 

Download: [Quick Start Guide](https://ansys.sharepoint.com/:p:/s/PerceiveEM/ERDVAhvPhlBEvvzH2RhakOwBsg6yAqLqR6RIUSTmKyDbcg?e=EeTpzD)

Many of the examples provided use the `pem_utilities` package to help streamline the process of setting up and running simulations. This package provides professional-grade utilities for electromagnetic simulation workflows.

### Dependencies

* Python 3.10.x
* pem_utilities package (included in this repository)
* Perceive EM API >=25.1
  * Available Ansys Customer Portal
* Perceive EM License
  * HPC Pack or Pool license is required to run simulations on the GPU may be required depending on the number of Streaming Multiprocessors (SMs) on the GPU.

### Installing

#### 1. Download and Install Perceive EM
* Download Perceive EM from the Ansys Customer Portal
  * [https://download.ansys.com/](https://download.ansys.com/)
* Install the API
  * Once installed, the api files we be availble for use
    * For example, in Windows: "C:/Program Files/AnsysEM/Perceive_EMv251/Win64/lib"
  * Only the files located within this lib directory are needed for the python API used in this repository
* Configure api_settings.json to point to the location of the Perceive EM API.
  * Set api_path to the location of the Perceive EM API (or use "Default" if installed into the default location)
  * Set license_path to the location of the licensingclient folder (or use "Default" if installed into the default location)

```json
# example json setting configuration
{
    "version": 25.1,
    "api_path": "default",
    "licensingclient_path": "default",
    "license_feature": "perceive_em",
    "hpc_pack_or_pool": "pack"
}
```

#### 2. Licensing
The licensingsettingutility can be used to configure what server or web licenses to use. This is found in the Perceive EM installation directory, this utility can be run to configure license checkout location.

If there is an issue with licensing, please see error messages. Typically, this is caused by licesningclient not being found. If so, copy this folder from within the API directory into the directory that is reported during the error message.

#### 3. Creating Python Environment

There are serval ways to create a python environment, outlined here is one way, but any method is acceptable. 

##### Using Conda (miniforge)

Download and install miniforge from the [HERE](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe)

Download and install pyCharm Community Edition from [HERE](https://www.jetbrains.com/pycharm/download/#section=windows)
Note: PyCharm is not required, but is a good IDE to use for python development.

Create a new environment, from within the miniforge terminal, run the following command. Press Y to continue when prompted.
```commandline
conda create -n perceive_em python=3.10.18
```
Activate the newly created environment
```commandline
activate perceive_em
```

#### 4. Install the Perceive EM Utilities Package

**Recommended Method (Modern Packaging):**
Browse to the location of this repository and install the package in editable mode:
```commandline 
pip install -e .
```
This will install the `pem_utilities` package along with all required dependencies specified in `pyproject.toml`.

**Alternative Method (Legacy):**
If you prefer to install dependencies manually:
```commandline 
pip install -r requirements.txt
```

#### 5. Configure Python Environment in PyCharm of IDE of choice

Configure the newly installed python environment in PyCharm. Set Add New Interpreter > Add Local Interpreter...
![Configure Interpreter in PyCharm, settings.](/images/configure_interpreter_pycharm.png)
Browse to the location of the python.exe file in the newly created environment
![Configure Interpreter in PyCharm](/images/config_interp.png)

### Using the Utilities Package

After installation, you can import utilities from anywhere in your Python environment:

```python
# Import API
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# Import specific utilities
from pem_utilities.materials import MaterialManager
from pem_utilities.coordinate_system import CoordSys
from pem_utilities.load_mesh import MeshLoader
from pem_utilities.actor import Actors
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.debugging_utils import DebuggingCamera, DebuggingLogs
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.primitives import *

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy 

```

## Running an Example Project

* Example Projects are available in the examples folder
  * complex_intersection_microdoppler.py
    * An example is available to visualize the scene tree and simulation results for a radar
  * hello_p2p.py
    * A simple example to demonstrate how to set up a point to point communication link. Uses no utilities
    * This example is a good starting point to understand the basics of the API
  * hello_radar.py
    * A simple example to demonstrate how to set up a radar simulation. Uses no utilities
    * This example is a good starting point to understand the basics of the API
  * beamforming.py
    * A more complex P2P example, this is utilities to streamline the script setup, visualization and post-processing to show beamforming
  * basic_heatmap.py
    * Demonstration using utilities to set up a heatmap visualization of the fields in the scene

## Help - API QUICK START

### Scene Tree
* Perceive EM Simulation is defined using the concept of building a Scene Tree.
  * The tree is defined through a parent/child relationship of nodes, and properties assigned to each node
  * There are two types of Nodes, Generic and Radar Nodes 
  * API calls are used to add the nodes, tell them the relationship within the tree, and assign properties
* Every scene node can be defined with the following properties:
  * Scene Element (Geometry)
  * Position/Orientation and Velocities
  * Parent Node (if not root node)
  * Child Nodes inherit pos/orientation/velocities from parent
* Radar Node is a special type of node
  * This is the node that all antenna excitations are assigned
  * It cannot have children
  * Antenna platform/device/mode can only be assigned to this type of node
  * Tx/Rx antennas are assigned
  * No scene element can be assigned

![Example scenetree layout.](/images/scenetree.png)
```
h_node = rss_py.SceneNode()
api.addSceneNode(h_node)
api.setSceneElement(h_node, h_elem))
```
### Scene Nodes and Elements

The scene tree can consistent of any number or depth of scene nodes. 
* Nodes can be hierarchical, inheriting properties from parents. The position/oriention/velocity of a child node can be defined in the parent's coordinate system. Scene elements are geometry that is attached to the scene node. Any orientation/velocity will be applied to this geometry.
* Scene elements consist of a list of triangles and a material property assigned to these triangles. A helper function to loadTriangleMesh() is available, but ultimately only a list of triangles is required by the API.
![Example scenetree layout.](/images/scene_node.png)

### Radar Nodes

A radar node is a special type of node that is used to define the radar platform and antenna excitations.
* This node cannot have children that are generic nodes, and must be only RadarPlatform(), RadarDevice() and RadarAntenna().
* 
![Example scenetree layout.](/images/radar_node.png)

### Modes
Used to define Waveforms used in the simulation.
![Example scenetree layout.](/images/mode.png)

### Antennas
Antennas are assigned to Radar Devices. 
* Parametric Beam Antennas
* FFD Based Antennas
![Example scenetree layout.](/images/antennas.png)


## Maintainers

Arien Sligar
arien.sligar@ansys.com

## Version History

* 0.01
    * Initial Release

## License

This repository is licensed under the MIT License - see the LICENSE.md file for details. The Perceive EM API is licensed under the Ansys EULA.

