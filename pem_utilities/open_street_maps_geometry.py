import os
import numpy as np
import pyvista as pv
import vtk
import srtm
import utm
import re
from tqdm import tqdm
import imageio
from PIL import Image

try:
    import osmnx as ox
except ImportError as e:
    # create error message telling user to install osmnx
    print("osmnx is not installed. Please install it by running the following command in your terminal:")
    print("pip install osmnx")

from pem_utilities.utils import create_mesh_from_image2


def read_geotiff(filename, center_lat_lon,output_path='./',decimate_mesh_percentage=None):
    """
    Helper function retrieves the Ground Control
    Points of a GeoTIFF. Note that this requires gdal.
    """
    import rasterio
    utm_center = utm.from_latlon(center_lat_lon[0], center_lat_lon[1])

    # utm.from_latlon returns a tuple with easting and northing. I will assign easting to x and northing to y
    center_offset_x = utm_center[0]
    center_offset_y = utm_center[1]

    # Load a raster
    dataset = rasterio.open(filename)
    idxs = dataset.indexes
    all_z = dataset.read(idxs[0])

    # geo tiff might be defined a y for lat, x for long. Which is different from how my utm function return data. So I will correct it below
    # some inconsistent results might occure
    lon_minus = dataset.bounds[0]
    lon_plus = dataset.bounds[2]
    lat_minus = dataset.bounds[1]
    lat_plus = dataset.bounds[3]

    xy_minus = utm.from_latlon(lat_minus, lon_minus)
    xy_plus = utm.from_latlon(lat_plus, lon_plus)

    # center the image based on center_lat_lon
    x_minus = xy_minus[0] - center_offset_x
    y_minus = xy_minus[1] - center_offset_y
    x_plus = xy_plus[0] - center_offset_x
    y_plus = xy_plus[1] - center_offset_y

    all_x = np.linspace(x_minus, x_plus, dataset.width)
    all_y = np.linspace(y_minus, y_plus, dataset.height)
    x, y = np.meshgrid(all_x, all_y)



    grid = pv.StructuredGrid(x, y, np.flipud(all_z))
    temp = grid.extract_geometry()
    mesh = temp.triangulate()
    mesh["elevation"] = all_z.flatten(order="F")
    out_file = os.path.join(output_path, "terrain.stl")

    if decimate_mesh_percentage is not None:
        print(f'Decimating terrain mesh, reducing triangles by {decimate_mesh_percentage*100}%')
        print(f'Starting number of cells {mesh.number_of_cells}')
        mesh.decimate(decimate_mesh_percentage,inplace=True,progress_bar=True)
        print(f'Ending number of cells {mesh.number_of_cells}')
    mesh.save(out_file)

    return mesh, out_file

def get_z_elevation_from_mesh(xy,meshes,return_min_and_max=True):

    x_loc = xy[0]
    y_loc = xy[1]
    if not isinstance(meshes, list):
        meshes = [meshes]

    if len(meshes) == 0:
        if return_min_and_max:
            return [0,0]
        else:
            return 0

    intersection_points = []
    buffer = 10  # additional distance so intersection test is further away than directly on surface
    for mesh in meshes:
        bounds = mesh.bounds
        min_z = bounds[4]
        max_z = bounds[5]

        start_z = min_z - buffer
        stop_z = max_z + buffer

        # start from +Z and look down, intersection will be first point hit.
        stop_ray = [x_loc, y_loc, start_z]
        start_ray = [x_loc, y_loc, stop_z]
        intersection_point, _ = mesh.ray_trace(start_ray, stop_ray)

        # ray will intersect with a geometry and return a point (or list with size greater than 0)
        if len(intersection_point) > 0:
            intersection_points.append(intersection_point.flatten()[2])
    intersection_points = np.array(intersection_points)


    if return_min_and_max:
        if len(intersection_points) == 0:
            return [None, None]
        else:
            return [np.min(intersection_points), np.max(intersection_points)]
    elif len(intersection_points) == 0:
        return None
    else:
        return intersection_points[0]

def is_outdoor(mesh,x_loc,y_loc):
    bounds = mesh.bounds
    min_z = bounds[4]
    max_z = bounds[5]

    buffer = 100  # additional distance so intersection test is further away than directly on surface
    start_z = min_z + .5
    stop_z = max_z + buffer

    start_ray = [x_loc, y_loc, start_z]
    stop_ray = [x_loc, y_loc, stop_z]
    intersection_point, _ = mesh.ray_trace(start_ray, stop_ray)

    # ray will intersect with a geometry and return a point (or list with size greater than 0)
    if len(intersection_point) == 0:
        return True
    return False


def is_indoor(mesh, x_loc, y_loc):
    bounds = mesh.bounds
    min_z = bounds[4]
    max_z = bounds[5]
    is_location_outdoors = is_outdoor(mesh, x_loc, y_loc)
    # if it is outdoors, it is not indoors
    if is_location_outdoors:
        return False
    return True



def find_random_location(mesh,z_elevation_mesh = None, outdoors=True,return_highest_z=False,max_tries=100,
                         max_z_elevation=1e5):
    # if outdoors, this will find a random outside, nothing above it
    if isinstance(mesh, list):
        mesh_temp = pv.PolyData()
        for each in mesh:
            mesh_temp += each
        mesh = mesh_temp

    if isinstance(z_elevation_mesh, list):
        mesh_temp = pv.PolyData()
        for each in z_elevation_mesh:
            mesh_temp += each
        z_elevation_mesh = mesh_temp


    bounds = mesh.bounds
    min_x = bounds[0]
    max_x = bounds[1]
    min_y = bounds[2]
    max_y = bounds[3]
    min_z = bounds[4]
    max_z = bounds[5]


    if min_z == max_z:
        return False

    inset_percentage = 0.8
    size_x = (max_x-min_x)*(1-inset_percentage)/2
    size_y = (max_y-min_y)*(1-inset_percentage)/2

    min_x = min_x+size_x
    max_x = max_x - size_x
    min_y = min_y+size_y
    max_y = max_y - size_y

    valid_location = False
    tries = 0

    while not valid_location:
        x_loc = np.random.uniform(low=min_x, high=max_x)
        y_loc = np.random.uniform(low=min_y , high=max_y )
        buffer = 100  # additional distance so intersection test is further away than directly on surface

        start_z = min_z - 1
        stop_z = max_z + buffer

        start_ray = [x_loc, y_loc, start_z]
        stop_ray = [x_loc, y_loc, stop_z]
        intersection_point, _ = mesh.ray_trace(start_ray, stop_ray)

        if return_highest_z and len(intersection_point)>0:
            # don't care if is indoor or outdoor,just return highest Z value
            if intersection_point[0][2]>max_z_elevation:
                valid_location=False
            else:
                valid_location=True
        elif outdoors:
            valid_location=is_outdoor(mesh,x_loc,y_loc)
        else:
            # if input argument is telling this function  to look for indoors locations
            valid_location=is_indoor(mesh,x_loc,y_loc)

        if tries> max_tries:
            print('Could not find a valid location for antenna')
            return False
        tries += 1

    if z_elevation_mesh is None and return_highest_z:
        z_elevation_mesh=mesh

    if z_elevation_mesh is not None:
        min_z = z_elevation_mesh.bounds[4]
        max_z = z_elevation_mesh.bounds[5]
        # if there is a z elevation mesh, we need to find the elevation of the location
        start_ray = [x_loc, y_loc, min_z]
        stop_ray = [x_loc, y_loc, max_z]
        intersection_point, _ = z_elevation_mesh.ray_trace(start_ray, stop_ray)
        if len(intersection_point) != 0:
            z_surface_location = intersection_point.flatten()[2]
            return [x_loc, y_loc, z_surface_location]
        else:
            return [x_loc, y_loc]
    else:
        if len(intersection_point) != 0:
            z_surface_location = intersection_point.flatten()[2]
            return [x_loc, y_loc, z_surface_location]
        else:
            return [x_loc, y_loc]





def find_random_location_by_points(mesh,how_many_points = 2,max_z_elevation=None):

    random_points = []
    if not isinstance(mesh, list):
        mesh = [mesh]
    all_points = []
    for n, each in enumerate(mesh):
        mesh_points = np.array(each.points)
        all_points.extend(mesh_points)
    all_points = np.array(all_points)

    mask = all_points[:, 2] <= max_z_elevation  #
    all_points_filtered = all_points[mask]

    total_num_points =len(all_points_filtered)
    if total_num_points<how_many_points:
        print(f"Error, not enough points < max_z_elevation {max_z_elevation}")
        print(f"Requested {how_many_points}, only {total_num_points} below {max_z_elevation}")
        return False

    random_idxs = np.random.randint(total_num_points,size=how_many_points)
    random_points = all_points_filtered[random_idxs]

    return random_points


def convert_heatmap_to_vtk(data,xyz, terrain_mesh = None,building_mesh = None):
    # convert all points on the heatmap to a vtk mesh, where if the point is indoors (inside a building) it will be
    # set to 0. else it will be set to the value of the heatmap.
    # noramlize the data to a peak value of 1
    data = np.abs(data)
    data = data / np.max(data)
    terrain_mesh.clear_data()
    terrain_mesh['heatmap'] = np.ndarray.flatten(data, order='C')
    for n, point in enumerate(terrain_mesh.points):
        x = point[0]
        y = point[1]
        # field value inside have to be 0
        if is_indoor(building_mesh,x,y):
            terrain_mesh['heatmap'][n] = 0
    building_mesh['heatmap'] = np.zeros(len(building_mesh.points))
    building_mesh.active_scalars_name = 'heatmap'
    total_mesh = terrain_mesh.merge(building_mesh,merge_points=True)

    return terrain_mesh, building_mesh, total_mesh

class BuildingsPrep(object):
    """Contains all basic functions needed to generate buildings stl files."""

    def __init__(self, cad_path,use_cache=True):
        self.cad_path = cad_path
        self.gdf = None
        self.use_cache = use_cache

    @staticmethod
    def create_building_roof(all_pos):
        """Generate a filled in polygon from outline.
        Includes concave and convex shapes.

        Parameters
        ----------
        all_pos : list

        Returns
        -------
        :class:`pyvista.PolygonData`
        """
        points = vtk.vtkPoints()
        for each in all_pos:
            points.InsertNextPoint(each[0], each[1], each[2])

        # Create the polygon
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(all_pos))  # make a quad
        for n in range(len(all_pos)):
            polygon.GetPointIds().SetId(n, n)

        # Add the polygon to a list of polygons
        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)

        # Create a PolyData
        polygonPolyData = vtk.vtkPolyData()
        polygonPolyData.SetPoints(points)
        polygonPolyData.SetPolys(polygons)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()

        mapper.SetInputData(polygonPolyData)

        triFilter = vtk.vtkTriangleFilter()
        # let's filter the polydata
        triFilter.SetInputData(polygonPolyData)
        triFilter.Update()

        polygonPolyDataFiltered = triFilter.GetOutput()
        roof = pv.PolyData(polygonPolyDataFiltered)
        return roof

    def generate_buildings(self, center_lat_lon,
                           geo_data_frame=None,
                           terrain_mesh=None,
                           max_radius=500,
                           expand_radius_to_fit_buildings=False,
                           export_image_path=None,
                           max_buildings=None):
        """Generate the buildings stl file.

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        terrain_mesh : :class:`pyvista.PolygonData`
            Terrain mesh.
        max_radius : float, int
            Radius around latitude and longitude.

        Returns
        -------
        dict
            Info of generated stl file.
        """


        if self.use_cache:
            ox.settings.log_console = True
            ox.settings.use_cache = False
        ox.settings.requests_kwargs = {'verify': False}
        if geo_data_frame is not None:
            self.gdf = geo_data_frame
        else:
            self.gdf=None
            print('Downloading building footprints from Open Street Maps...')
            self.gdf = ox.features.features_from_point(center_lat_lon, tags={"building": True}, dist=max_radius)
            print('... Done.')
        if export_image_path is not None:
            bbox = ox.utils_geo.bbox_from_point(center_lat_lon, dist=max_radius)
            ox.plot.plot_footprints(self.gdf, ax=None, figsize=(8, 8), color='orange', edge_color='none', edge_linewidth=0,
                                       alpha=None, bgcolor='#000000', bbox=bbox, save=True, show=False, close=False,
                                       filepath=export_image_path, dpi=600)
        utm_center = utm.from_latlon(center_lat_lon[0], center_lat_lon[1])
        center_offset_x = utm_center[0]
        center_offset_y = utm_center[1]

        if len(self.gdf) == 0:
            return {"file_name": None, "mesh": None}
        else:
            # gdf_proj = ox.project_gdf(self.gdf)
            gdf_proj = ox.projection.project_gdf(self.gdf)

            geo = gdf_proj["geometry"]
            try:
                levels = gdf_proj["building:levels"]
                levels = levels.array
            except KeyError:
                levels = [1] * len(geo)
            try:
                height = gdf_proj["height"]
                height = height.array
            except KeyError:
                height = [10] * len(geo)

            temp = [levels, height]
            geo = geo.array

            building_meshes = pv.PolyData()  # empty location where all building meshses are stored


            last_displayed = -1
            if max_buildings is not None:
                geo = geo[:max_buildings]
            print(f'Processing buildings, total buildings: {len(geo)}')
            for n, _ in enumerate(tqdm(geo)):

                g = geo[n]
                if hasattr(g, "exterior"):
                    outer = g.exterior

                    xpos = np.array(outer.xy[0])
                    ypos = np.array(outer.xy[1])
                    l = levels[n]
                    h = height[n]
                    if hasattr(gdf_proj,'MaxHeight'):
                        if gdf_proj['MaxHeight'][n] != None:
                            h = gdf_proj['MaxHeight'][n]
                    # if hasattr(gdf_proj, 'Material'):
                    #     print(gdf_proj['Material'][n])

                    points = np.zeros((np.shape(outer.xy)[1], 3))
                    points[:, 0] = xpos
                    points[:, 1] = ypos
                    points[:, 0] -= center_offset_x
                    points[:, 1] -= center_offset_y

                    delta_elevation = 0
                    # if no terrain mesh, buildings we be set at z= 0
                    if terrain_mesh is not None:
                        buffer = 50  # additional distance so intersection test is further away than directly on surface
                        bb_terrain = terrain_mesh.bounds
                        start_z = bb_terrain[4] - buffer
                        stop_z = bb_terrain[5] + buffer

                        # The shape files do not have z/elevation position. So for them to align to the
                        # terrain we need to first get the position of the terrain at the xy position of shape file
                        # this will align the buildins so they sit on the terrain no matter the location
                        elevation_on_outline = []

                        # check every point on the building shape for z elevation location
                        for point in points:
                            # shoot ray to look for intersection point
                            start_ray = [point[0], point[1], start_z]
                            stop_ray = [point[0], point[1], stop_z]
                            intersection_point, _ = terrain_mesh.ray_trace(start_ray, stop_ray)
                            if len(intersection_point) != 0:
                                z_surface_location = intersection_point.flatten()[2]
                                elevation_on_outline.append(z_surface_location)
                        # find lowest point on building outline to align location
                        if elevation_on_outline:
                            min_elevation = np.min(elevation_on_outline)
                            max_elevation = np.max(elevation_on_outline)
                            delta_elevation = max_elevation - min_elevation

                            # change z position to minimum elevation of terrain
                            points[:, 2] = min_elevation
                        else:
                            points[:, 2] = start_z



                    # create closed and filled polygon from outline of building
                    roof = self.create_building_roof(points)
                    if isinstance(h, str):
                        h = h.replace("'", "")
                        # h = re.sub("[^\d\.]", "", h)
                        if 'km' in h:
                            h = h.replace('km', '')
                            h = float(h) * 1000
                        elif 'm' in h:
                            h = h.replace('m', '')
                            h = float(h)
                    if not np.isnan(float(h)):
                        extrude_h = float(h)# * 2
                    elif not np.isnan(float(l)):
                        extrude_h = float(l)# * 10
                    else:
                        extrude_h = 15.0

                    outline = pv.lines_from_points(points, close=True)

                    vert_walls = outline.extrude([0, 0, extrude_h + delta_elevation], inplace=False,capping=False)

                    roof_location = np.array([0, 0, extrude_h + delta_elevation])
                    roof.translate(roof_location, inplace=True)

                    building_meshes += vert_walls.triangulate()
                    building_meshes += roof.triangulate()

            el = building_meshes.points[:, 2]

            building_meshes["Elevation"] = el.ravel(order="F")

            if not expand_radius_to_fit_buildings:
                _, keep = building_meshes.clip(return_clipped=True, normal='x', origin=(-max_radius, 0, 0.0))
                keep, _ = keep.clip(return_clipped=True, normal='x', origin=(max_radius, 0, 0.0))
                _, keep = keep.clip(return_clipped=True, normal='y', origin=(0, -max_radius, 0.0))
                building_meshes, _ = keep.clip(return_clipped=True, normal='y', origin=(0, max_radius, 0.0))
            file_out = os.path.join(self.cad_path,"buildings.stl")
            building_meshes.save(file_out, binary=True)

            return {"file_name": file_out, "mesh": building_meshes, "temp": temp}

    def generate_buildings2(self, center_lat_lon,
                           geo_data_frame=None,
                           terrain_mesh=None,
                           max_radius=None,
                           expand_radius_to_fit_buildings=False,
                           export_image_path=None,
                           max_buildings=None):
        """Generate the buildings stl file. this new function will return seperate meshes for different materials

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        terrain_mesh : :class:`pyvista.PolygonData`
            Terrain mesh.
        max_radius : float, int
            Radius around latitude and longitude.

        Returns
        -------
        dict
            Info of generated stl file.
        """


        if self.use_cache:
            ox.settings.log_console = True
            ox.settings.use_cache = True
        ox.settings.requests_kwargs = {'verify': False}

        if geo_data_frame is not None:
            self.gdf = geo_data_frame
        else:
            self.gdf = ox.geometries.geometries_from_point(center_lat_lon, tags={"building": True}, dist=max_radius)
        if export_image_path is not None:
            bbox = ox.utils_geo.bbox_from_point(center_lat_lon, dist=max_radius)
            ox.plot.plot_footprints(self.gdf, ax=None, figsize=(8, 8), color='orange', edge_color='none', edge_linewidth=0,
                                       alpha=None, bgcolor='#000000', bbox=bbox, save=True, show=False, close=False,
                                       filepath=export_image_path, dpi=600)
        utm_center = utm.from_latlon(center_lat_lon[0], center_lat_lon[1])
        center_offset_x = utm_center[0]
        center_offset_y = utm_center[1]

        if len(self.gdf) == 0:
            return {"file_name": None, "mesh": None}
        else:
            # gdf_proj = ox.project_gdf(self.gdf)
            gdf_proj = ox.projection.project_gdf(self.gdf)
            geo = gdf_proj["geometry"]
            try:
                levels = gdf_proj["building:levels"]
                levels = levels.array
            except KeyError:
                levels = [1] * len(geo)
            try:
                height = gdf_proj["height"]
                height = height.array
            except KeyError:
                height = [10] * len(geo)

            temp = [levels, height]
            geo = geo.array

            num_materials = 1
            materials =['concrete']
            if hasattr(gdf_proj,'Material'):
                materials = list(np.unique(gdf_proj["Material"]))
                num_materials = len(materials)
            building_meshes = {}
            file_out = {}
            for i in range(num_materials):
                building_meshes[materials[i]] = pv.PolyData()
                file_out[materials[i]] = os.path.join(self.cad_path, f"\\buildings_{materials[i]}.stl")
            last_displayed = -1
            if max_buildings is not None:
                geo = geo[:max_buildings]
            print(f'Processing buildings, total buildings: {len(geo)}')
            for n, _ in enumerate(tqdm(geo)):
                g = geo[n]
                if hasattr(g, "exterior"):
                    outer = g.exterior

                    xpos = np.array(outer.xy[0])
                    ypos = np.array(outer.xy[1])
                    l = levels[n]
                    h = height[n]
                    if hasattr(gdf_proj,'MaxHeight'):
                        if gdf_proj['MaxHeight'][n] != None:
                            h = gdf_proj['MaxHeight'][n]
                    if hasattr(gdf_proj, 'Material'):
                        mat_name = gdf_proj['Material'][n]
                    else:
                        mat_name = 'concrete'

                    points = np.zeros((np.shape(outer.xy)[1], 3))
                    points[:, 0] = xpos
                    points[:, 1] = ypos
                    points[:, 0] -= center_offset_x
                    points[:, 1] -= center_offset_y

                    delta_elevation = 0
                    # if no terrain mesh, buildings we be set at z= 0
                    if terrain_mesh is not None:
                        buffer = 50  # additional distance so intersection test is further away than directly on surface
                        bb_terrain = terrain_mesh.bounds
                        start_z = bb_terrain[4] - buffer
                        stop_z = bb_terrain[5] + buffer

                        # The shape files do not have z/elevation position. So for them to align to the
                        # terrain we need to first get the position of the terrain at the xy position of shape file
                        # this will align the buildins so they sit on the terrain no matter the location
                        elevation_on_outline = []

                        # check every point on the building shape for z elevation location
                        # this is slow, commenting out for now, just going to use a single point in the future
                        # for point in points:
                        #     # shoot ray to look for intersection point
                        #     start_ray = [point[0], point[1], start_z]
                        #     stop_ray = [point[0], point[1], stop_z]
                        #     intersection_point, _ = terrain_mesh.ray_trace(start_ray, stop_ray)
                        #     if len(intersection_point) != 0:
                        #         z_surface_location = intersection_point.flatten()[2]
                        #         elevation_on_outline.append(z_surface_location)

                        # shoot ray to look for intersection point, only using a few points
                        # might not be accurate for hilly areas, but for primarily flat it should be fine
                        for idx in [0,len(points)//2]:
                            start_ray = [points[idx,0], points[idx,1], start_z]
                            stop_ray = [points[idx,0], points[idx,1], stop_z]
                            intersection_point, _ = terrain_mesh.ray_trace(start_ray, stop_ray)
                            if len(intersection_point) != 0:
                                z_surface_location = intersection_point.flatten()[2]
                                elevation_on_outline.append(z_surface_location)
                        # find lowest point on building outline to align location
                        if elevation_on_outline:
                            min_elevation = np.min(elevation_on_outline)
                            max_elevation = np.max(elevation_on_outline)
                            delta_elevation = max_elevation - min_elevation

                            # change z position to minimum elevation of terrain
                            points[:, 2] = min_elevation
                        else:
                            points[:, 2] = start_z



                    # create closed and filled polygon from outline of building
                    roof = self.create_building_roof(points)
                    if isinstance(h, str):
                        h = h.replace("'", "")
                        # h = re.sub("[^\d\.]", "", h)
                    if not np.isnan(float(h)):
                        extrude_h = float(h)# * 2
                    elif not np.isnan(float(l)):
                        extrude_h = float(l)# * 10
                    else:
                        extrude_h = 15.0

                    outline = pv.lines_from_points(points, close=True)

                    vert_walls = outline.extrude([0, 0, extrude_h + delta_elevation], inplace=False,capping=False)

                    roof_location = np.array([0, 0, extrude_h + delta_elevation])
                    roof.translate(roof_location, inplace=True)

                    building_meshes[mat_name] += vert_walls
                    building_meshes[mat_name] += roof
            all_materials = list(building_meshes.keys())
            for mat_name in all_materials:
                if building_meshes[mat_name].number_of_cells>0:
                    if not expand_radius_to_fit_buildings and max_radius is not None:
                        _, keep = building_meshes[mat_name].clip(return_clipped=True, normal='x', origin=(-max_radius, 0, 0.0))
                        keep, _ = keep.clip(return_clipped=True, normal='x', origin=(max_radius, 0, 0.0))
                        _, keep = keep.clip(return_clipped=True, normal='y', origin=(0, -max_radius, 0.0))
                        building_meshes[mat_name], _ = keep.clip(return_clipped=True, normal='y', origin=(0, max_radius, 0.0))
                    file_name = os.path.join(self.cad_path ,f"buildings_{mat_name}.stl")
                    building_meshes[mat_name].save(file_name, binary=True)
                    file_out[mat_name] = file_name
                else:
                    # remove the empty mesh
                    building_meshes.pop(mat_name)
                    file_out.pop(mat_name)

            return {"file_name": file_out, "mesh": building_meshes, "temp": temp}


class RoadPrep(object):
    """Contains all basic functions needed to generate road stl files."""

    def __init__(self, cad_path,use_cache=True):
        self.cad_path = cad_path
        self.use_cache = use_cache

    def create_roads(self, center_lat_lon, terrain_mesh, max_radius=1000, z_offset=0, road_step=10, road_width=5):
        """Generate the road stl file.

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        terrain_mesh : :class:`pyvista.PolygonData`
            Terrain mesh.
        max_radius : float, int
            Radius around latitude and longitude.
        z_offset : float, optional
            Elevation offset of the road.
        road_step : float, optional
            Road computation steps in meters.
        road_width : float, optional
            Road width in meter.

        Returns
        -------
        dict
            Info of generated stl file.
        """

        if self.use_cache:
            ox.settings.log_console = True
            ox.settings.use_cache = True

        graph = ox.graph_from_point(
            center_lat_lon, dist=max_radius, simplify=False, network_type="all", clean_periphery=True
        )

        g_projected = ox.project_graph(graph)

        utm_center = utm.from_latlon(center_lat_lon[0], center_lat_lon[1])
        center_offset_x = utm_center[0]
        center_offset_y = utm_center[1]

        _, edges = ox.graph_to_gdfs(g_projected)
        lines = []

        buffer = 10  # additional distance so intersection test is further away than directly on surface
        bb_terrain = terrain_mesh.bounds
        start_z = bb_terrain[4] - buffer
        stop_z = bb_terrain[5] + buffer

        line = pv.PolyData()
        road_ends = pv.PolyData()
        # convert each edge into a line
        count = 0
        last_displayed = -1
        for _, row in edges.iterrows():
            count += 1
            num_percent_bins = 40

            percent = np.round((count) / (len(edges)) * 100, decimals=1)
            if percent % 10 == 0 and percent != last_displayed:
                last_displayed = percent
                perc_done = int(num_percent_bins * percent / 100)
                perc_left = num_percent_bins - perc_done
                percent_symbol1 = "." * perc_left
                percent_symbol2 = "#" * perc_done

                i = percent_symbol2 + percent_symbol1 + " " + str(percent) + "% "


            x_pts = row["geometry"].xy[0]
            y_pts = row["geometry"].xy[1]
            z_pts = np.empty(len(x_pts))
            z_pts.fill(start_z + z_offset)
            for n in range(len(z_pts)):
                x_pts[n] = x_pts[n] - center_offset_x
                y_pts[n] = y_pts[n] - center_offset_y
                start_ray = [x_pts[n], y_pts[n], start_z]
                stop_ray = [x_pts[n], y_pts[n], stop_z]
                points, _ = terrain_mesh.ray_trace(start_ray, stop_ray)
                if len(points) != 0:
                    z_surface_location = points.flatten()[2]
                    z_pts[n] = z_surface_location + z_offset
            pts = np.column_stack((x_pts, y_pts, z_pts))
            # always 2 points, linear interpolate to higher number of points
            dist = np.sqrt(
                np.power(pts[0][0] - pts[1][0], 2)
                + np.power(pts[0][1] - pts[1][1], 2)
                + np.power(pts[0][2] - pts[1][2], 2)
            )
            if dist > road_step:
                num_steps = int(dist / road_step)
                xpos = np.linspace(pts[0][0], pts[1][0], num=num_steps)
                ypos = np.linspace(pts[0][1], pts[1][1], num=num_steps)
                zpos = np.linspace(pts[0][2], pts[1][2], num=num_steps)
                pts = np.column_stack((xpos, ypos, zpos))
            try:
                line += pv.lines_from_points(pts, close=True)
            except ValueError:
                pass
            end_shape = pv.Circle(road_width, resolution=16).delaunay_2d()
            road_ends += end_shape.translate(pts[0], inplace=False)
            end_shape = pv.Circle(road_width, resolution=16).delaunay_2d()
            road_ends += end_shape.translate(pts[-1], inplace=False)
            lines.append(line)

        roads = line.ribbon(width=road_width, normal=[0, 0, 1])
        roads += road_ends
        el = roads.points[:, 2]

        roads["Elevation"] = el.ravel(order="F")
        file_out = os.path.join(self.cad_path + "roads.stl")
        roads.save(file_out)
        return {"file_name": file_out, "mesh": roads, "graph": g_projected}


class TerrainPrep(object):
    """Contains all basic functions needed for creating a terrain stl mesh."""

    def __init__(self, cad_path="./",use_cache=True):
        self.cad_path = cad_path
        self.use_cache = use_cache

    def get_terrain(self, center_lat_lon=None, max_radius=500, grid_size=5,all_grid_pos=None, buffer_percent=0,flat_surface=False,shape='rectangle'):
        """Generate the terrain stl file.

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        max_radius : float, int
            Radius around latitude and longitude.
        grid_size : float, optional
            Grid size in meters.
        buffer_percent : float, optional
            Buffer extra size over the radius.


        Returns
        -------
        dict
            Info of generated stl file.
        """
        if self.use_cache:
            ox.settings.log_console = True
            ox.settings.use_cache = True

        if flat_surface:
            file_out = os.path.join(self.cad_path , "terrain.stl")
            if shape == 'rectangle':
                pointa = [max_radius, -max_radius, 0.0]
                pointb = [max_radius, max_radius, 0.0]
                pointc = [-max_radius, max_radius, 0.0]
                terrain_mesh = pv.Rectangle([pointa, pointb, pointc])
            elif shape=='grid':
                z = np.zeros((len(all_grid_pos), 1))
                xyz = np.hstack((all_grid_pos, z))
                terrain_mesh = pv.PolyData(xyz)
                terrain_mesh = terrain_mesh.delaunay_2d()

            elif shape == 'circle':
                terrain_mesh = pv.Circle(max_radius)
            terrain_mesh.save(file_out)
            return {"file_name": file_out, "mesh": terrain_mesh}
        utm_center = utm.from_latlon(center_lat_lon[0], center_lat_lon[1])

        if buffer_percent > 1:
            buffer_percent = buffer_percent / 100
        max_radius = max_radius * (buffer_percent + 1)
        all_data, _, all_utm = self.get_elevation(
            center_lat_lon,
            max_radius=max_radius,
            grid_size=grid_size,
        )

        all_data = np.nan_to_num(all_data, nan=-32768)

        xyz = []
        for lat_idx in range(all_data.shape[0]):
            for lon_idx in range(all_data.shape[1]):
                latlat_utm_centered = all_utm[lat_idx][lon_idx][0] - utm_center[0]
                lonlon_utm_centered = all_utm[lat_idx][lon_idx][1] - utm_center[1]

                if (
                    all_data[lat_idx][lon_idx] != -32768
                ):  # this is missing data from srtm, don't add if it doesn't exist
                    xyz.append([latlat_utm_centered, lonlon_utm_centered, all_data[lat_idx][lon_idx]])
        xyz = np.array(xyz)

        file_out = os.path.join(self.cad_path, "terrain.stl")
        # logger.info("saving STL as " + file_out)
        terrain_mesh = pv.PolyData(xyz)
        terrain_mesh = terrain_mesh.delaunay_2d(
            tol=10 / (2 * max_radius) / 2
        )  # tolerance, srtm is 30meter, so as a fraction of total size this would be 30/2/radius
        terrain_mesh = terrain_mesh.smooth(n_iter=100, relaxation_factor=0.04)

        el = terrain_mesh.points[:, 2]

        terrain_mesh["Elevation"] = el.ravel(order="F")

        terrain_mesh.save(file_out)
        return {"file_name": file_out, "mesh": terrain_mesh}

    @staticmethod
    def get_elevation(
        center_lat_lon,
        max_radius=500,
        grid_size=3,
    ):
        """Get Elevation map.

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        max_radius : float, int
            Radius around latitude and longitude.
        grid_size : float, optional
            Grid size in meters.

        Returns
        -------
        tuple
        """

        utm_center = utm.from_latlon(center_lat_lon[0], center_lat_lon[1])
        # assume never at boundary of zone number or letter
        zone_letter = utm.latitude_to_zone_letter(center_lat_lon[0])
        zone_number = utm.latlon_to_zone_number(center_lat_lon[0], center_lat_lon[1])
        # logger.info(zone_letter)
        # logger.info(zone_number)
        # logger.info(utm_center)
        utm_x_min = utm_center[0] - max_radius
        utm_x_max = utm_center[0] + max_radius

        utm_y_min = utm_center[1] - max_radius
        utm_y_max = utm_center[1] + max_radius

        sample_grid_size = grid_size  # meters
        num_samples = int(np.ceil(max_radius * 2 / sample_grid_size))
        x_samples = np.linspace(utm_x_min, utm_x_max, int(num_samples))
        y_samples = np.linspace(utm_y_min, utm_y_max, int(num_samples))
        elevation_data = srtm.get_data()

        all_data = np.zeros((num_samples, num_samples))
        all_utm = np.zeros((num_samples, num_samples, 2))
        all_lat_lon = np.zeros((num_samples, num_samples, 2))
        # logger.info("Terrain Points...")
        last_displayed = -1
        for n, x in enumerate(x_samples):
            for m, y in enumerate(y_samples):
                num_percent_bins = 40

                percent_complete = int((n * num_samples + m) / (num_samples * num_samples) * 100)
                if percent_complete % 10 == 0 and percent_complete != last_displayed:
                    last_displayed = percent_complete
                    perc_done = int(num_percent_bins * percent_complete / 100)
                    perc_left = num_percent_bins - perc_done
                    percent_symbol1 = "." * perc_left
                    percent_symbol2 = "#" * perc_done
                    i = percent_symbol2 + percent_symbol1 + " " + str(percent_complete) + "% "
                    # logger.info(f"\rPercent Complete:{i}")
                zone_letter = utm.latitude_to_zone_letter(center_lat_lon[0])
                zone_number = utm.latlon_to_zone_number(center_lat_lon[0], center_lat_lon[1])
                current_lat_lon = utm.to_latlon(x, y, zone_number, zone_letter)
                all_data[n, m] = elevation_data.get_elevation(current_lat_lon[0], current_lat_lon[1])
                all_lat_lon[n, m] = current_lat_lon
                all_utm[n, m] = [x, y]
        # logger.info(str(100) + "% - Done")
        return all_data, all_lat_lon, all_utm