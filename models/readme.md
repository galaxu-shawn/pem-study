# Models

## Cad Files
Faceted CAD files in OBJ/STL format can be directly imported. Material will typically be imported with a single material property assigned to all triangles contained in one file.

## Specialized Formats

### Import from JSON
Multple CAD files can be imported as defined by a json file. This allows more actors to be imported from multiple cad files, with hierarchical relationships between parts, and special properties that can be applied to each part.
When defining the json file, the following structure needs to be defined:
```json
{
    "name": "actor_name",
    "version": 1,
    "class": "other",
    "parts": {
      "part-parent": {
        "file_name": "cad_name.stl",
        "properties": {
          "color": "blue",
          "material": "pec"
        },
        "part-child": {
            "file_name": "cad_name1.stl",
            "properties": {
                "color": "black",
                "material": "aluminium"
                }
            }
        }
    }
}
 
  
    
```
Parts must be prepended with "part-" and include a "file_name" field. The "properties" field is optional but can include color, material and various other properties (see examples).

#### Class: Vehicle
A vehicle is a special class of import that will duplicate tires/wheels and place those tires and wheels based on the wheelbased, diameter and wheel to wheel width properties defined at the top level of the json file. A vehicle must be defined as class="vehicle" in the json file in order for the model to be correctly setup up. See Audi_A1_2010.json for example.
#### Class: Other
The "other" class of imports will allow generic, multi-part actors to be imported. Additional properties, like rotation velocity/position can be assigned within these json files to account for how children parts behave with respect to the root object. See Wind_Turbine for an example.
### Class: DAE
A DAE file is an animated Collada file that currently only support a limited set of geometry, primarily human animations. This animated collada file contains the time depenenet pos and orientation of each joint of the human body. This time sequency of position/orientations is read from the dae file, and mapped to the cad files sotred in /pedestrian_bones_stl. This is general pose-able mannequin that can be used to simulate human movement.

### Class: DAE CMU
This database of 2500+ motion capture files is available from the Carnegie Mellon University Graphics Lab Motion Capture Database. The database contains free motion capture files organized by category. The database contains a large variety of human motion, including walking, running, boxing, and dancing. The motion capture data can be used to drive the animation of a pedestrian
Database originally converted from: http://mocap.cs.cmu.edu/

[Download Link](https://ansys.sharepoint.com/:u:/s/PerceiveEM/EYnkabtyrupPtjYX1E6_5jkBUC3lNNJlbNMvMMYmMXazyQ?e=bVZfPC)
