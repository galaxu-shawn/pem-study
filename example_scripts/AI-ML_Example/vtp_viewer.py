import pyvista as pv
import os

folder = r'C:\Users\asligar\OneDrive - ANSYS, Inc\Documents\Scripting\github\perceive_em\example_scripts\output\Gen_56c1e2\Result_0002_0001'

# Load the VTP file
filename = os.path.join(folder, 'total.vtp')

# using pyvista load the vtp file
mesh = pv.read(filename)
plotter = pv.Plotter()
plotter.add_mesh(mesh,log_scale=True,clim=[-100,0])
plotter.show()

