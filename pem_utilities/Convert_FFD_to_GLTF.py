from dataclasses import dataclass
import numpy as np
import pyvista as pv
import os
@dataclass
class FF_Data:
    Num_Freq: int = 1
    rETheta: np.ndarray = np.empty(0)
    rEPhi: np.ndarray = np.empty(0)
    Misc_FarField: np.ndarray = np.empty(0) #So we can store anything we want in here, for example abs(rETheta)
    Theta: np.ndarray = np.empty(0)
    Phi: np.ndarray = np.empty(0)
    Frequencies: np.ndarray = np.empty(0)
    Delta_Theta: float = 1
    Delta_Phi: float = 1
    Diff_Area: float = 1
    Total_Samples: int = 1
    Position: np.ndarray = np.zeros(3)
    Lattice_Vector: np.ndarray = np.zeros(6)
    Is_Component_Array: bool = False
    Incident_Power: float = 1

    def calc_rETotal(self):
        return np.sqrt(np.power(np.abs(self.rETheta), 2) + np.power(np.abs(self.rEPhi), 2))

    def calc_RealizedGainTheta(self, incident_power=None):
        if incident_power is None:
            incident_power = self.Incident_Power
        return 2 * np.pi * np.abs(np.power(self.rETheta, 2)) / incident_power / 377

    def calc_RealizedGainPhi(self, incident_power=None):
        if incident_power is None:
            incident_power = self.Incident_Power
        return 2 * np.pi * np.abs(np.power(self.rEPhi, 2)) / incident_power / 377

    def calc_RealizedGainTotal(self, incident_power=None):
        if incident_power is None:
            incident_power = self.Incident_Power
        data_total = np.sqrt(np.power(np.abs(self.rETheta), 2) + np.power(np.abs(self.rEPhi), 2))
        return 2 * np.pi * np.abs(np.power(data_total, 2)) / incident_power / 377


def read_ascii_ffd(filepath):
    data = FF_Data()
    all_freq = []
    temp_dict = {}
    base_path, filename_only = os.path.split(filepath)
    with open(filepath, 'r') as reader:
        theta = [int(i) for i in reader.readline().split()]
        phi = [int(i) for i in reader.readline().split()]
        num_freq = int(reader.readline().split()[1])
        theta_range = np.linspace(*theta)
        phi_range = np.linspace(*phi)
        ntheta = len(theta_range)
        nphi = len(phi_range)
        samples_per_freq = ntheta * nphi

        freq_index = -1
        field_index = 0
        for line in reader:
            if 'Frequency' in line:
                freq = float(line.split()[1])
                all_freq.append(freq)
                freq_index += 1
                field_index = 0
    reader.close()

    # I need a more general way to do this, but for now I will just catch the special cases that I know exist
    # if theta_range[0]==-180 and phi[-1]==360:
    #     phi_range = np.linspace(0,180)

    eep_txt = np.loadtxt(filepath, skiprows=4, comments='Frequency')
    Etheta = np.vectorize(complex)(eep_txt[:, 0], eep_txt[:, 1])
    Ephi = np.vectorize(complex)(eep_txt[:, 2], eep_txt[:, 3])

    data.rETheta = Etheta.reshape((num_freq, samples_per_freq))
    data.rEPhi = Ephi.reshape((num_freq, samples_per_freq))
    data.Theta = theta_range
    data.Phi = phi_range
    data.Frequencies = np.array(all_freq)
    data.Delta_Theta = np.abs(theta_range[1] - theta_range[0])
    data.Delta_Phi = np.abs(phi_range[1] - phi_range[0])
    data.Diff_Area =np.abs(np.radians(data.Delta_Theta) * np.radians(data.Delta_Phi) * np.sin(np.radians(theta_range)))
    data.Total_Samples = len(data.rETheta[0])
    data.Num_Freq = len(data.Frequencies)
    return data


def conversion_function(data, function_str=None):
    if function_str == 'dB10':
        data = 10 * np.log10(np.abs(data))
    elif function_str == 'dB20':
        data = 20 * np.log10(np.abs(data))
    elif function_str == 'abs':
        data = np.abs(data)
    elif function_str == 'real':
        data = np.real(data)
    elif function_str == 'imag':
        data = np.imag(data)
    elif function_str == 'norm':
        data = np.abs(data) / np.max(np.abs(data))
    elif function_str == 'ang':
        data = np.angle(data)
    elif function_str == 'ang_deg':
        data = np.angle(data, deg=True)
    else:
        data = data

    return data


def gen_ff_mesh(ff_data,theta,phi, mesh_limits=[0, 1]):
    # threshold values below certain value
    # ff_data = np.where(ff_data < mesh_limits[0], mesh_limits[0], ff_data)
    # shift data so all points are positive. Needed to correctly change plot
    # scale as magnitude changes. This keeps plot size consistent for different
    # magintudes
    if ff_data.min() < 0:
        ff_data_renorm = ff_data + np.abs(ff_data.min())
    else:
        ff_data_renorm = ff_data
    # ff_data_renorm = np.interp(ff_data, (ff_data.min(), ff_data.max()),
    #                                (mesh_limits[0], ff_data.max()))

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    r_no_renorm = np.reshape(ff_data, (len(theta), len(phi)))
    r = np.reshape(ff_data_renorm, (len(theta), len(phi)))
    r_max = np.max(r)
    x = mesh_limits[1] * r / r_max * np.sin(theta_grid) * np.cos(phi_grid)
    y = mesh_limits[1] * r / r_max * np.sin(theta_grid) * np.sin(phi_grid)
    z = mesh_limits[1] * r / r_max * np.cos(theta_grid)

    # for color display
    mag = np.ndarray.flatten(r_no_renorm, order='F')

    # create a mesh that can be displayed
    ff_mesh = pv.StructuredGrid(x, y, z)
    # ff_mesh.scale(ff_scale)
    # ff_mesh.translate([float(position[0]),float(position[1]),float(position[2])])
    # this store the actual values of gain, that are used for color coating
    ff_mesh['FarFieldData'] = mag
    return ff_mesh

# Input path to the ffd you want to convert to ffd.
ffd_path = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\dipole_HFSS_Y_axis_1.ffd'

# Output path where you want to save as gltf or glb
output_path = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\delete\asdf.gltf'

# read FFD file (ascii format is the only format officially exported by AEDT)
ff_data = read_ascii_ffd(ffd_path)

# Original data is complex quanitites for rETheta, and rEPhi. Which can be accessed by using the property ff_data.rETheta
# and ff_data.rEPhi. If you want to convert these to rETotal, RealizedGainTheta, RealizedGainPhi, RealizedGainTotal, use
# the corropsoing calc_*() function
data_to_plot = ff_data.calc_RealizedGainTotal()

# convert this data to the desired results
data_to_plot = conversion_function(data_to_plot, function_str='dB10')

# generate a pyvista mesh using the data form the previous step
ff_mesh = gen_ff_mesh(data_to_plot,ff_data.Theta,ff_data.Phi, mesh_limits=[0, 1])

# visualize the data using pyvista
pl = pv.Plotter()
pl.add_mesh(ff_mesh)
pl.export_gltf(output_path) # export to gltf
pl.show()