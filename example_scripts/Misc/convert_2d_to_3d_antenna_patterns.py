
import os
import sys
from pathlib import Path
# Dynamically resolve the path to the api_core module
script_path = Path(__file__).resolve()
api_core_path = script_path.parent.parent  # Adjusted to point to the correct parent directory
if api_core_path not in sys.path:
    sys.path.insert(0, str(api_core_path))

from pem_utilities.far_fields import FarFields, summing

test_pattern = 'C:\\Users\\asligar\\OneDrive - ANSYS, Inc\\Documents\\Scripting\\github\\perceive_em\\antenna_device_library\\AWR_1642\\rx1.ffd'

ff = FarFields()

ff.read_ffd(test_pattern,create_farfield_mesh=True)

port_name = ff.all_port_names[0]
ff_data = ff.data_dict[port_name]
re_thehta = ff_data.rETheta
re_phi = ff_data.rEPhi

theta_phi = []
for t in ff_data.Theta:
    for p in ff_data.Phi:
        theta_phi.append((t, p))

# get indices where phi is 0
indices_phi_0 = [i for i, (t, p) in enumerate(theta_phi) if p == 0]
# get re_theta and re_phi values at phi = 0
re_theta_phi_0 = re_thehta[indices_phi_0]
re_phi_phi_0 = re_phi[indices_phi_0]
# get indices where phi is 90
indices_phi_90 = [i for i, (t, p) in enumerate(theta_phi) if p == 90]
# get re_theta and re_phi values at phi = 90
re_theta_phi_90 = re_thehta[indices_phi_90]
re_phi_phi_90 = re_phi[indices_phi_90]


# np.reshape(ff/, (len(theta), len(phi)))

threed_pattern = summing(vertical_slice=re_theta_phi_0, horizontal_slice=re_theta_phi_90)

ff.plot()