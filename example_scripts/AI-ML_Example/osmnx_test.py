# import osmnx as ox
# import geopandas as gpd
# place_name = "Edgewood Washington, DC, USA"
# # Get place boundary related to the place name as a geodataframe
# area = ox.geocode_to_gdf(place_name)
# import pyaedt
#
# pyaedt.import_from_openstreet_map([33,-155], env_name='default', terrain_radius=100, include_osm_buildings=True, including_osm_roads=True, import_in_aedt=False, plot_before_importing=True, z_offset=2, road_step=3, road_width=8, create_lightweigth_part=True)
#
#
import numpy as np
import os
import pyvista as pv
import sys
sys.path.append("..")
from pem_utilities.open_street_maps_geometry import BuildingsPrep, TerrainPrep, find_random_location

output_path = '../output/'
os.makedirs(output_path, exist_ok=True)
lat_lon = (40.739524, -73.990127)
max_radius = 500
buildings_prep = BuildingsPrep(output_path)
terrain_prep = TerrainPrep(output_path)
terrain = terrain_prep.get_terrain(lat_lon, max_radius=max_radius*1.75, flat_surface=True)
buildings = buildings_prep.generate_buildings(lat_lon, terrain['mesh'], max_radius=500)
all_mesh= terrain['mesh']+buildings['mesh']


xy = find_random_location(buildings['mesh'],outdoors=True)
if xy:
    all_mesh += pv.Sphere(center=[xy[0],xy[1],1.5], radius=5)
    all_mesh.plot()

asdf = 1