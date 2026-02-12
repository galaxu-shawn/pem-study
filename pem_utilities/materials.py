"""
Created on Fri Jan 26 15:00:00 2024

@author: asligar
"""

import json
import os
from dataclasses import dataclass
from pem_utilities.pem_core import Perceive_EM_API
from pem_utilities.path_helper import get_repo_paths
from pem_utilities.calculate_ITU_materials import generate_itu_materials_dict

@dataclass
class MatData:
    """
    A data class that represents the properties of a material.

    Attributes:
    ------------
    thickness : float
        The thickness of the material. Defaults to -1.0.

    rel_eps_real : float
        The real part of the relative permittivity. Defaults to 1.0.

    rel_eps_imag : float
        The imaginary part of the relative permittivity. Defaults to 0.0.

    rel_mu_real : float
        The real part of the relative permeability. Defaults to 1.0.

    rel_mu_imag : float
        The imaginary part of the relative permeability. Defaults to 0.0.

    conductivity : float
        The conductivity of the material. Defaults to 0.0.

    height_standard_dev : float
        The standard deviation of the height. Defaults to None.

    roughness : float
        The roughness of the material. Defaults to None.

    backing : str
        The backing of the material. Defaults to None.

    coating_idx : int
        The index of the coating. Defaults to 1.
    """
    thickness: float = -1.0
    rel_eps_real: float = 1.0
    rel_eps_imag: float = 0.0
    rel_mu_real: float = 1.0
    rel_mu_imag: float = 0.0
    conductivity: float = 0.0
    height_standard_dev: float = None
    roughness: float = None
    backing: str = None
    coating_idx: int = 1

    @classmethod
    def from_dict(cls, data):
        """
        A class method that creates a MatData instance from a dictionary.

        Parameters:
        ------------
        data : dict
            The dictionary containing the material data.

        Returns:
        --------
        MatData
            The created MatData instance.
        """
        return cls(
            thickness=data.get('thickness', -1.0),
            rel_eps_real=data.get('relEpsReal', 1.0),
            rel_eps_imag=data.get('relEpsImag', 0.0),
            rel_mu_real=data.get('relMuReal', 1.0),
            rel_mu_imag=data.get('relMuImag', 0.0),
            conductivity=data.get('conductivity', 0.0),
            height_standard_dev=data.get('height_standard_dev', None),
            roughness=data.get('roughness', None),
            backing=data.get('backing', None),
            coating_idx=data.get('coating_idx', 1)
        )


class MaterialManager:
    """
    A class that manages materials.

    Attributes:
    ------------
    rss_py : RssPy
        The RssPy instance.

    api : api
        The api instance.

    cached_coatings : list
        The list of cached coatings.

    all_materials : dict
        The dictionary of all materials.
    """

    def __init__(self, material_library_name='material_library.json', generate_itu_materials=False, itu_freq_ghz=1.0):
        """
        Initialize a MaterialManager instance.

        This method initializes a MaterialManager instance by loading a JSON file that contains the material library.
        The material library is a collection of materials with their properties. Each material is represented as a MatData instance.
        The materials are stored in a dictionary where the key is the material name and the value is the corresponding MatData instance.
        The method also initializes an RssPy instance and an api instance which are used for managing the materials.
        The method also initializes a list of cached coatings which is used for storing the indices of the materials that have been added to the coating.

        Parameters:
        ------------
        material_library_name : str, optional
            The name of the JSON file that contains the material library. Defaults to 'material_library.json'.
        generate_itu_materials : bool, optional
            If True, generates ITU materials based on user defined frequency and ignores the material_library.json file. Defaults to False.
        itu_freq_ghz : float, optional
            The frequency in GHz at which to generate ITU materials if generate_itu_materials is True. Defaults to 1.0.
        """

        paths = get_repo_paths()
        # print(f"Materials Folder: {paths.materials}")

        
        self.pem_api_manager = Perceive_EM_API()
        self.pem = self.pem_api_manager.pem  # The configured API object
        self.rss_py = self.pem_api_manager.RssPy

        self.cached_coatings = []
        material_names_by_coating_idx = {}

        if isinstance(material_library_name, str):
            material_library_name = [material_library_name]

        # we can generate ITU material based on user defined frequency. If this is done, it will ignore
        # the material_library json file specified. You can't have both at the same time
        if generate_itu_materials:
            print(f'INFO: Generating ITU materials based on user defined frequency ({itu_freq_ghz}GHz. Appending with '
                  f'material_library.json file. Duplicate material names will always be overwritten with ITU '
                  f'Materials.')
            materials_json = generate_itu_materials_dict(itu_freq_ghz)
            all_materials_multiple_libs = materials_json['materials']
            # now append with default materials
            last_idx = 0
            for mat in all_materials_multiple_libs:
                idx = all_materials_multiple_libs[mat]['coating_idx']
                if idx > last_idx:
                    last_idx = idx

            default_mat_filename = os.path.join(paths.materials, 'material_library.json')
            with open(default_mat_filename) as f:
                materials_json = json.load(f)

            for mat in materials_json['materials']:
                if mat not in all_materials_multiple_libs:
                    last_idx += 1
                    all_materials_multiple_libs[mat] = materials_json['materials'][mat]
                    all_materials_multiple_libs[mat]['coating_idx'] = last_idx

        else:
            # load multiple material libraries if more than 1 is provided
            all_materials_multiple_libs = {}
            for mat_file in material_library_name:
                filename = os.path.join(paths.materials, mat_file)

                with open(filename) as f:
                    materials_json = json.load(f)
                all_materials_multiple_libs.update(materials_json['materials'])

            # correct indexing if more than one library
            if len(material_library_name) >1:
                for n, each in enumerate(all_materials_multiple_libs):
                    all_materials_multiple_libs[each]['coating_idx'] = n+1
                if 'pec' in all_materials_multiple_libs:
                    all_materials_multiple_libs['pec']['coating_idx'] = 0


        self.all_materials = {}
        for mat in all_materials_multiple_libs:
            self.all_materials[mat.lower()] = MatData.from_dict(all_materials_multiple_libs[mat])
        for mat in self.all_materials:
            if self.all_materials[mat].coating_idx not in material_names_by_coating_idx:
                material_names_by_coating_idx[self.all_materials[mat].coating_idx] = mat

        self.material_names_by_coating_idx = material_names_by_coating_idx

    def load_material(self, index):
        if index in self.material_names_by_coating_idx:
            if index not in self.cached_coatings:
                # this will add the material to the API based on the selected index
                ret = self.get_index(self.material_names_by_coating_idx[index])

    def get_index(self, material):
        """
        Get the index of a material. Also adds the material to the API if it is not already added.

        Parameters:
        ------------
        material : str
            The name of the material.

        Returns:
        --------
        int
            The index of the material.

        Raises:
        --------
        ValueError
            If the material is not found in the material library.
        """

        material = material.lower()
        if material in self.all_materials.keys():
            t = self.all_materials[material].thickness
            er_real = self.all_materials[material].rel_eps_real
            er_im = self.all_materials[material].rel_eps_imag
            mu_real = self.all_materials[material].rel_mu_real
            mu_imag = self.all_materials[material].rel_mu_imag
            cond = self.all_materials[material].conductivity
            mat_idx = self.all_materials[material].coating_idx
            material_is_rough = False
            if isinstance(t,list):
                if not len(t) == len(er_real) == len(er_im) == len(mu_real) == len(mu_imag) == len(cond):
                    raise ValueError(f'MULTI LAYER MATERIAL ERROR: {material} has different lengths of properties, must be the same')
                material_str = 'DielectricLayers '
                for each in range(len(t)):
                    material_str_each = f' {t[each]},{er_real[each]},{er_im[each]},{mu_real[each]},{mu_imag[each]},{cond[each]}'
                    material_str = material_str + material_str_each
            else:
                material_str = f'DielectricLayers {t},{er_real},{er_im},{mu_real},{mu_imag},{cond}'
            if material == 'pec':
                mat_idx = 0
                material_str = 'PEC'
                return mat_idx
            elif material == 'absorber':
                mat_idx = self.all_materials[material].coating_idx
                material_str = 'Absorber'
            else:
                if self.all_materials[material].backing:
                    backing_mat = self.all_materials[material].backing
                    material_str = f'{material_str}  {backing_mat}'
                if self.all_materials[material].roughness and self.all_materials[material].height_standard_dev:
                    material_is_rough = True
                    roughness = self.all_materials[material].roughness
                    std_dev = self.all_materials[material].height_standard_dev

            if mat_idx not in self.cached_coatings:
                self.cached_coatings.append(mat_idx)
                h_mat = self.rss_py.Coating()
                self.pem_api_manager.isOK(self.pem.addCoating(h_mat, material_str))
                self.pem_api_manager.isOK(self.pem.mapCoatingToIndex(h_mat, mat_idx))  # using plus 1 becuase 0 will always be pec
                if material_is_rough:
                    self.pem_api_manager.isOK(self.pem.setCoatingRoughness(h_mat, std_dev, roughness))
            return mat_idx
        else:
            print(f'ERROR: {material} not found in material library, assigning pec')
            return 0

    def create_material(self, material_name,mat_properties, overwrite=False):
        """
        Add a material to the material library.

        Parameters:
        ------------
        material_name : str
            The name of the material.

        overwrite : bool, optional
            If True, the existing material will be overwritten. Defaults to False.

        Raises:
        --------
        ValueError
            If the material already exists in the material library and overwrite is False.
        """
        if material_name in self.all_materials.keys() and not overwrite:
            raise ValueError(f'{material_name} already exists in the material library.  Set overwrite=True to overwrite')
        else:
            self.all_materials[material_name] = mat_properties
            self.all_materials[material_name].coating_idx = len(self.all_materials.keys())

    def get_all_material_names(self):
        """
        Get the names of all materials in the material library.

        Returns:
        --------
        list
            The list of all material names.
        """
        return list(self.all_materials.keys())