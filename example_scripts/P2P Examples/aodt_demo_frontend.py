#
# Copyright ANSYS. All rights reserved.
import os
import numpy as np
import matplotlib.pyplot as plt
from aodt_demo_backend import AODT_DEMO
from pem_utilities.rotation import euler_to_rot
from pem_utilities.path_helper import get_repo_paths # common paths to reference

paths = get_repo_paths() 


backend = AODT_DEMO(show_modeler=True)

# different file types can be iported, including different ways to define material properties (not shown)
# file_to_import = r"C:\Users\asligar\OneDrive - ANSYS, Inc\Desktop\OV_Assets\ArchVis\Industrial\Pallets\Pallet_A1.usd"
# file_to_import = os.path.join(paths.models,'tokyo.usd')
file_to_import = os.path.join(paths.models,"whole-scene-static.stl")
backend.add_scene_element(filename=file_to_import)

# hard coded ru positions and orientations
# initialize the RU's and UE's in the scene
backend.add_ru(pos=[21.24,116.09,19.78],rot=np.eye(3),name='ru',scale_pattern=10)
# backend.add_ru(pos=[100,0,100],rot=np.eye(3),name='ru2',num_ant = [2,2])

# initial position for UE's. This coudl come from AODT, but I am just
# using an intialial value and creating 2 UE's
backend.add_ue(pos=[3.0, 4.0, 5.0],rot=np.eye(3),lin=[0,0,0],name='ue',scale_pattern=10)
# backend.add_ue(pos=[0,0,200],rot=np.eye(3),lin=[0,0,0],name='ue2')

# configure sim gets everything ready to run, includes setting like number of bounces, ray density
backend.configure_sim()

# this is what is called each time you want to run the simulation
backend.run_simulation()


# after the simulation is run, we can request the CFR
# we can reorganize this data as required by AODT, the current format is:
# [UE_idx][RU_idx][RU_rx_idx][pulse_idx][freq_idx]
cfr = backend.retrive_results()

# now if we want to update the UE's to a new position we can do it like this:
# this could be put into a function/callback whatever
backend.update_ue(pos=[3.0, 4.0, 5.0],rot=np.eye(3),lin=[10,0,0],name='ue')
backend.run_simulation()
cfr = backend.retrive_results()


backend.modeler.close()

fig, ax = plt.subplots(1, 1, layout='constrained')
ax.plot(20*np.log10(np.abs(cfr[0][0,0,0,0,:])))
ax.grid(True)
plt.show()



