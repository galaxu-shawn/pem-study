import numpy as np
import pyvista as pv
from scipy.fftpack import fft2, ifft2
import os
import time as walltime

class OceanSurface:
    def __init__(self,
                 num_grid,
                 scene_length,
                 wind_speed,
                 wave_amplitude,
                 choppiness,
                 wind_direction=[1,0],
                 include_wake=False,
                 velocity_ship=10,
                 length_ship=110,
                 beam_ship=20.3,
                 draft_ship=3.5,
                 initial_wake_position=[0,0],
                 update_wake_position=True):
        num_grid = int(num_grid)  # grid size
        self.num_grid = num_grid  # grid size
        self.scene_length = scene_length  # domain size
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.wave_amplitude = wave_amplitude  # wave amplitude
        self.choppiness = choppiness  # choppiness factor

        self.g = 9.81  # gravitational constant
        self.kx, self.ky = np.meshgrid(np.fft.fftfreq(num_grid+1) * (num_grid+1) / scene_length, np.fft.fftfreq(num_grid+1) * (num_grid+1) / scene_length)
        self.k = np.sqrt(self.kx ** 2 + self.ky ** 2)
        self.k[self.k == 0] = 1e-8  # avoid division by zero

        self.h0 = self.generate_h0()
        self.omega = np.sqrt(self.g * self.k)
        self.mesh = None
        self.time = 0

        self.include_wake = include_wake
        self.velocity_ship = velocity_ship
        self.length_ship = length_ship
        self.beam_ship = beam_ship
        self.draft_ship = draft_ship
        self.initial_wake_position = initial_wake_position
        self.update_wake_position = update_wake_position


    def generate_h0(self):
        r1, r2 = np.random.normal(0, 1, (2, self.num_grid+1, self.num_grid+1))
        return r1 + 1j * r2

    def phillips_spectrum(self):
        wind_direction = np.array(self.wind_direction)
        k_dot_w = self.kx * wind_direction[0] + self.ky * wind_direction[1]
        L = self.wind_speed ** 2 / self.g
        damping = 0.001
        return self.wave_amplitude * np.exp(-1 / (self.k * L) ** 2) / self.k ** 4 * np.abs(k_dot_w / self.k) ** 2 * np.exp(
            -self.k ** 2 * L ** 2 * damping)

    def generate_height_field(self, t=None):
        if t is None:
            t = self.time
        else: # update time based on the input
            self.time = t
        h_tilde = self.phillips_spectrum() ** 0.5 * self.h0 * np.exp(1j * self.omega * t)
        h_tilde += np.conj(h_tilde[::-1, ::-1])
        return np.real(ifft2(h_tilde))

    def generate_displacement_field(self, t=None):
        if t is None:
            t = self.time
        else: # update time based on the input
            self.time = t
        h_tilde = self.phillips_spectrum() ** 0.5 * self.h0 * np.exp(1j * self.omega * t)
        h_tilde += np.conj(h_tilde[::-1, ::-1])

        dx = np.real(ifft2(1j * self.kx / self.k * h_tilde))
        dy = np.real(ifft2(1j * self.ky / self.k * h_tilde))

        return self.choppiness * dx, self.choppiness * dy

    def _generate_grid(self, t=None):
        if t is None:
            t = self.time
        else: # update time based on the input
            self.time = t
        x = np.linspace(-self.scene_length/2, self.scene_length/2, self.num_grid+1)
        y = np.linspace(-self.scene_length/2, self.scene_length/2, self.num_grid+1)
        X, Y = np.meshgrid(x, y)

        height = self.generate_height_field(t)

        if self.include_wake:
            start_time = walltime.time()
            x_init = self.initial_wake_position[0]
            y_init = self.initial_wake_position[1]
            # currently only support ship moving in +Y direction
            wake_pos_x = x_init
            wake_pos_y = y_init
            if self.update_wake_position: #only update the position if True
                wake_pos_y = y_init + self.velocity_ship * t
            zship = Wake(velocity_ship=self.velocity_ship,
                         length_ship=self.length_ship,
                         beam_ship=self.beam_ship,
                         draft_ship=self.draft_ship,
                         scene_length=self.scene_length,
                         grid_spacing=self.scene_length/self.num_grid,
                         wake_position=[wake_pos_x,wake_pos_y])
            zship.calculate(time=t)
            height_wake = zship.z_ship
            end_time = walltime.time()
            # print(f"Time to calculate wake: {end_time - start_time}")
            # shift to the center (all wake starts at 0,0) for now
            # it will translate in Y direction with the ships velocity
            height += height_wake
        dx, dy = self.generate_displacement_field(t)

        X += dx
        Y += dy

        # Create the surface using PyVista
        grid = pv.StructuredGrid(X, Y, height)

        # Add the height data as a scalar field to the grid
        grid["Height"] = height.ravel('F')
        return grid

    def generate_mesh(self, t=None):
        grid = self._generate_grid(t=t)
        temp  = grid.extract_geometry()
        self.mesh = temp.triangulate()
        return self.mesh

    def save_mesh(self,output_path):
        if self.mesh is None:
            self.generate_mesh()
        self.mesh.save(output_path)

    def create_ocean_surface_plot(self, t=None):
        grid = self._generate_grid(t=t)
        return grid



class Wake:
    def __init__(self,velocity_ship=10,length_ship=110,beam_ship=20.3,draft_ship=3.5,scene_length=1000,grid_spacing=2.5,wake_position=[0,0]):
        # wake is only for ship traveling in +Y direction
        self.velocity_ship = velocity_ship  # Velocity of the ship (m/s)
        self.length_ship = length_ship  # Length at the waterline (meters)
        self.beam_ship = beam_ship  # Beam (meters)
        self.draft_ship = draft_ship  # Draft (meters)
        self.scene_length = scene_length  # Scene size (meters)
        self.grid_spacing = grid_spacing
        self.wake_position = wake_position

    def calculate(self,time=0):
        # time is just tranlating position of the wake along y axis.
        velocity_ship = self.velocity_ship
        length_ship = self.length_ship
        beam_ship = self.beam_ship
        draft_ship = self.draft_ship
        scene_length = self.scene_length

        dx = self.grid_spacing
        dy = dx

        g = 9.81  # Gravitational acceleration (m/s^2)
        scene_length_y = scene_length / 2
        x = np.arange(-scene_length, 0 + dx, dx)
        y = np.arange(-scene_length_y, scene_length_y + dy, dy)[::-1]
        z = 0
        v = g / (velocity_ship ** 2)
        Ns = len(x)  # Number of samples
        theta = np.linspace(-np.pi/2 + np.pi/25, np.pi/2 - np.pi/25, Ns)
        tau = v * np.sqrt(1/np.cos(theta) ** 2 - 1/np.cos(theta))
        t = np.linspace(np.min(tau), np.max(tau), Ns)
        dt = t[1] - t[0]
        Fr = velocity_ship / (np.sqrt(g * length_ship))  # Froude number
        a = (1 + np.sqrt(1 + (4 * t ** 2) / (v ** 2))) / 2
        b = np.sqrt(a) / (2 * Fr ** 2)

        def calculate_components(func, exp_func):
            C = np.zeros((len(t), len(x)))
            EXP = np.zeros((len(t), len(x)), dtype=complex)
            for i in range(len(t)):
                for j in range(len(x//2)):
                    xval = x[j] - self.wake_position[1] + scene_length / 2
                    if xval > 0: # get rid of mirror values
                        xval = 0
                    C[i, j] = func(t[i], xval, z, v, draft_ship, Fr)
                    EXP[i, j] = exp_func(t[i], y[j]+self.wake_position[0])
            INT = np.trapz(C[:, :, np.newaxis] * EXP[:, np.newaxis, :], t, axis=0)
            return -(16 * beam_ship * length_ship / np.pi) * velocity_ship * (Fr ** 6) * np.real(INT)

        # Define component functions
        def Cx_func(t, x, z, v, draft_ship, Fr):
            a = ((1 + np.sqrt(1 + (4 * t ** 2) / (v ** 2))) / 2)
            return (v * np.exp(v * z * a) * np.sin(v * x * np.sqrt(a)) * (np.exp(-draft_ship * v * a) - 1) *
                    np.sqrt(a) * (np.sin(np.sqrt(a) / (2 * Fr ** 2)) - (np.cos(np.sqrt(a) / (2 * Fr ** 2)) * np.sqrt(a)) / (2 * Fr ** 2))) / (
                    np.sqrt(t ** 2 / v ** 2 + 1/4) * np.sqrt(a ** 3))


        def calculate_components2(func, exp_func, t, x, y, z, v, draft_ship, Fr, beam_ship, length_ship, velocity_ship):
            # Vectorize the functions
            vectorized_func = np.vectorize(func)
            vectorized_exp_func = np.vectorize(exp_func)

            # Create meshgrid for t and x
            T, X = np.meshgrid(t, x, indexing='ij')

            # Calculate C and EXP in a vectorized manner
            C = vectorized_func(T, X, z, v, draft_ship, Fr)
            EXP = vectorized_exp_func(T, y)

            # Perform the integration
            INT = np.trapz(C[:, :, np.newaxis] * EXP[:, np.newaxis, :], t, axis=0)

            # Calculate the final result
            return -(16 * beam_ship * length_ship / np.pi) * velocity_ship * (Fr ** 6) * np.real(INT)

        def Cx_func2(t, x, z, v, draft_ship, Fr):
            a = ((1 + np.sqrt(1 + (4 * t ** 2) / (v ** 2))) / 2)
            sqrt_a = np.sqrt(a)
            exp_term = np.exp(v * z * a) * np.exp(-draft_ship * v * a)
            sin_term = np.sin(v * x * sqrt_a)
            trig_term = np.sin(sqrt_a / (2 * Fr ** 2)) - (np.cos(sqrt_a / (2 * Fr ** 2)) * sqrt_a) / (2 * Fr ** 2)
            denominator = np.sqrt(t ** 2 / v ** 2 + 1 / 4) * np.sqrt(a ** 3)

            return (v * exp_term * sin_term * (1 - np.exp(-draft_ship * v * a)) * sqrt_a * trig_term) / denominator

        def exp_func(t, y):
            return np.exp(1j * y * t)

        # two different ways of doing this, I am not sure which one is correct
        INT_Ux = calculate_components(Cx_func,lambda t, y: np.exp(1j * y * t))
        # INT_Ux = calculate_components2(Cx_func2, exp_func, t, x, y, z, v, draft_ship, Fr, beam_ship, length_ship,velocity_ship)
        self.z_ship =  (velocity_ship / g) * INT_Ux


if __name__ == "__main__":

    # Set up the ocean parameters
    num_grid = 400  # grid size
    scene_length = 500 # domain size
    wind_speed =10
    wind_direction = [1, 0]
    wave_amplitude = 1 # wave amplitude
    choppiness = .5  # choppiness factor
    dt = 0.1
    ocean = OceanSurface(num_grid, scene_length, wind_speed, wave_amplitude,choppiness,
                         wind_direction=wind_direction,
                         include_wake=True,
                         velocity_ship=10,
                         length_ship=110,
                         beam_ship=20.3,
                         draft_ship=6.5,
                         initial_wake_position=[0,0]
                         )

    # Set up the plotter
    plotter = pv.Plotter()
    plotter.show_grid()
    output_path = '../output/seastate'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plotter.open_movie(os.path.join(output_path,"ocean_animation.mp4"))

    # Create the initial surface


    # Set up the colormap
    colormap = "bone"  # You can change this to other colormaps like "blues", "coolwarm", etc.

    # Get the height range for consistent color scaling
    height_range = None
    surf = None
    # Animate
    n_frames = 1000
    grid = ocean.create_ocean_surface_plot(0)
    for frame in range(n_frames):

        # plotter.clear_actors()
        # if surf is not None:
        #     plotter.remove_actor(surf)

        t = frame * dt
        grid = ocean.create_ocean_surface_plot(t)

        # Update the height range if it's not set
        if height_range is None:
            height_range = [grid["Height"].min(), grid["Height"].max()]

        if surf is None:
            # Add the surface with the colormap based on height
            surf = plotter.add_mesh(
                grid,
                scalars="Height",
                cmap=colormap,
                show_edges=False,
                clim=height_range,  # Use consistent color scaling
            )

        else:
            # ac = list(plotter.actors.keys())[0]
            for ac in plotter.actors:
                if hasattr(plotter.actors[ac], 'mapper'):
                    plotter.actors[ac].mapper.dataset = grid
        frame_zfill = str(frame).zfill(4)
        # ocean.save_mesh(os.path.join(output_path,f"seastate_{frame_zfill}.stl"))
        # Add a colorbar
        # if frame == 0:  # Only add the colorbar once
        plotter.add_scalar_bar("Wave Height", vertical=True, title_font_size=10, label_font_size=8)

        plotter.write_frame()

    plotter.close()