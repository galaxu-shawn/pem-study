# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:19:16 2022

@author: asligar
"""

import numpy as np
import pyvista as pv
import time as walltime
from pem_utilities.FarFieldData import FF_Data
from pem_utilities.utils import apply_math_function

class FF_Fields_Beamformer():
    '''
    Load and compute far field data as defined by weights through superpostion
    of the individual far field values for each port. Scaled by mag/phase
    '''

    def __init__(self, ant_fields_dict):
        '''

        Loads all the far fields into memory.

        Parameters
        ----------

        adjust_phase_center : bol
            adjust antennas that have been exported to a phase center per port to one common
            phase center. If all ports use a common phase center, like exported from HFSS with all using
            Global CS as the common reference, then this should be set to False. If they have been exported
            Using a coordinate system for each port, this should be set to True.


        Returns
        -------
        None.
        creates a dictionary of (data_dict) of near field values and positions
        for each port

        '''

        time_before = walltime.time()
        print('Loading Embedded Element Patterns...')
        self.ffd_dict = ant_fields_dict

        all_port_names = list(ant_fields_dict.keys())
        array_positions = {}
        for port in all_port_names:
            array_positions[port] = ant_fields_dict[port].Position

        self.all_port_names = all_port_names

        self.beamform_results = None
        self.mesh = None
    def beamform(self,source_weights, generate_mesh=False, quantity='RealizedGainTotal',function='dB',freq_idx=0,freq_override=None,scale_pattern=1):
        start_time = walltime.time()
        data = FF_Data()
        all_port_names = self.all_port_names
        num_ports = len(all_port_names)

        if str(freq_idx) == 'All':
            num_freq = self.ffd_dict[all_port_names[0]].Num_Freq
            freqs = self.ffd_dict[all_port_names[0]].Frequencies
        else:
            num_freq = 1
            freqs = [self.ffd_dict[all_port_names[0]].Frequencies[freq_idx]]

        if freq_override is not None:
            freqs = []
            for n in range(num_freq):
                freqs.append(freq_override)

        phi = self.ffd_dict[all_port_names[0]].Phi
        theta = self.ffd_dict[all_port_names[0]].Theta

        rEtheta_total = np.zeros((num_freq, self.ffd_dict[all_port_names[0]].Total_Samples), dtype=complex)
        rEphi_total = np.zeros((num_freq, self.ffd_dict[all_port_names[0]].Total_Samples), dtype=complex)

        # for each frequency, in most cases this will be a single frequency
        for idx, freq in enumerate(freqs):

            phi_grid, theta_grid = np.meshgrid(np.deg2rad(phi), np.deg2rad(theta))
            ko = 2 * np.pi * freq / 3e8
            kx_grid = ko * np.sin(theta_grid) * np.cos(phi_grid)
            ky_grid = ko * np.sin(theta_grid) * np.sin(phi_grid)

            kx_flat = kx_grid.ravel()
            ky_flat = ky_grid.ravel()

            weights = np.zeros(num_ports, dtype=complex)
            incident_power = 0

            for n, port in enumerate(all_port_names):
                # assuming source weights are in order of port names, which I think is correct?
                # w_mag = source_weights[port]['mag']
                incident_power += np.abs(source_weights[n])
                # w_phase = np.deg2rad(source_weights[port]['phase'])
                # weights[n] = np.sqrt(w_mag) * np.exp(1j * w_phase)

                if hasattr(self.ffd_dict[port], 'Position'):
                    xyz_pos = self.ffd_dict[port].Position
                else:
                    xyz_pos = np.zeros(3)
                    print('No position information found for port, assuming 0,0,0')


                array_factor = np.exp(1j * (xyz_pos[0] * kx_flat + xyz_pos[1] * ky_flat)) * source_weights[n]
                if str(freq_idx) == 'All':
                    current_idx = idx
                else:
                    current_idx = freq_idx
                rEtheta_total[idx] += array_factor * self.ffd_dict[port].rETheta[current_idx]
                rEphi_total[idx] += array_factor * self.ffd_dict[port].rEPhi[current_idx]

        # populate data class
        data.rETheta = rEtheta_total
        data.rEPhi = rEphi_total
        data.Theta = self.ffd_dict[all_port_names[0]].Theta
        data.Phi = self.ffd_dict[all_port_names[0]].Phi
        data.Frequencies = self.ffd_dict[all_port_names[0]].Frequencies
        data.Delta_Theta = self.ffd_dict[all_port_names[0]].Delta_Theta
        data.Delta_Phi = self.ffd_dict[all_port_names[0]].Delta_Phi
        data.Diff_Area = self.ffd_dict[all_port_names[0]].Diff_Area
        data.Total_Samples = len(data.rETheta[0])
        data.Num_Freq = len(data.Frequencies)
        data.Lattice_Vector = self.ffd_dict[all_port_names[0]].Lattice_Vector
        data.Lattice_Vector = self.ffd_dict[all_port_names[0]].Lattice_Vector
        data.Is_Component_Array = self.ffd_dict[all_port_names[0]].Is_Component_Array
        data.Incident_Power = incident_power

        stop_time = walltime.time()
        elapsed_time = stop_time - start_time
        # print(f'combined Fields: {elapsed_time}')

        # if quantity.lower() == "calc_realizedgaintotal":

        ff_data = apply_math_function(data.calc_RealizedGainTotal(), function)
        self.gen_farfield_mesh_from_ffd(ff_data, data.Theta, data.Phi, mesh_limits=[0, 1], scale_pattern=scale_pattern)

        self.beamform_results = data

    def wMRT(self, H, D=None):

        H = np.atleast_2d(H)

        num_users = H.shape[0]
        num_antenna = H.shape[1]

        if D is None:
            D = np.eye(num_antenna)

        channel_vector = np.dot(H, D)
        # which norm is approtiate
        # w_mrt = channel_vector/np.linalg.norm(channel_vector,1)
        # I think 2 is correct, is it the sqrt(sum(abs(channel_vector))^2)
        channel_vector_norms = np.zeros(num_users)
        w_mrt_pwr = np.zeros((num_users, num_antenna), dtype='complex')
        w_mrt_voltage = np.zeros((num_users, num_antenna), dtype='complex')
        for n in range(num_users):
            channel_vector_norms[n] = np.linalg.norm(channel_vector[n], 2)

        min_ch = np.min(channel_vector_norms)
        max_ch = np.min(channel_vector_norms)
        for n in range(num_users):
            w = channel_vector[n] / channel_vector_norms[n]
            w_mrt_pwr[n] = np.power(np.abs(w), 2) * np.exp(1j * np.angle(w))
            w_mrt_voltage[n] = np.abs(w) * np.exp(1j * np.angle(w))

        channel_vector_norm2 = np.linalg.norm(channel_vector, 2)
        w_mrt2 = channel_vector / channel_vector_norm2

        composite_sparam = np.zeros(num_users, dtype='complex')
        for n in range(num_users):
            composite_sparam[n] = np.dot(H[n], w_mrt_voltage[n].conj().T)
            # composite_sparam[n] = np.dot(H[n],np.conj(w_mrt_voltage[n].T))
        composite_weights = np.zeros(num_antenna, dtype='complex')
        for user in range(1):
            # composite_weights += np.sqrt(max_ch/np.abs(w_mrt[user]))*np.exp(1j*np.angle(w_mrt[user]*180/np.pi))
            composite_weights += w_mrt_voltage[user]
        test = 1
        return w_mrt_voltage, composite_sparam, composite_weights

    def wZFBF(self, H, D=None):

        H = np.atleast_2d(H)

        num_users = H.shape[0]
        num_antenna = H.shape[1]

        # if D is None:
        D = np.moveaxis(np.tile(np.eye(num_antenna), (num_users, 1, 1)), 0, -1)

        channel_vector = H
        # which norm is approtiate
        # w_mrt = channel_vector/np.linalg.norm(channel_vector,1)
        # I think 2 is correct, is it the sqrt(sum(abs(channel_vector))^2)
        channel_vector_norms = np.zeros(num_users)
        w_mrt_pwr = np.zeros((num_users, num_antenna), dtype='complex')
        w_mrt_voltage = np.zeros((num_users, num_antenna), dtype='complex')
        for n in range(num_users):
            effective_channel = np.dot(H, D[:, :, n]).conj().T
            ci_denom = np.dot(effective_channel.conj().T, effective_channel)
            ci_num = effective_channel
            channel_inversion = np.linalg.lstsq(ci_denom.T, ci_num.T, rcond=None)[0].T

            channel_inversion_norm = np.linalg.norm(channel_inversion[:, n], 2)
            w = np.conj(channel_inversion[:, n] / channel_inversion_norm)
            w_mrt_pwr[n] = np.power(np.abs(w), 2) * np.exp(1j * np.angle(w))
            w_mrt_voltage[n] = np.abs(w) * np.exp(1j * np.angle(w))

        min_ch = np.min(channel_inversion_norm)
        max_ch = np.min(channel_inversion_norm)
        composite_sparam = np.zeros(num_users, dtype='complex')
        for n in range(num_users):
            composite_sparam[n] = np.dot(H[n], np.conj(w_mrt_voltage[n].T))

        test = 20 * np.log10(np.abs(composite_sparam))
        test2 = 20 * np.log10(np.abs(H[0]))
        composite_weights = np.zeros(num_antenna, dtype='complex')
        # for user in range(1):
        #     composite_weights += np.sqrt(max_ch/np.abs(w_mrt[user]))*np.exp(1j*np.angle(w_mrt[user]*180/np.pi))
        composite_weights = w_mrt_voltage[0]
        test = 1
        return w_mrt_voltage, composite_sparam, composite_weights

    def wBeamSelect(self, H, beam_idx=0, D=None):
        beam_idx = int(beam_idx)
        # this is just re-using other functions, lots of un-needed stuff in here, need to clean up
        H = np.atleast_2d(H)

        num_users = H.shape[0]
        num_antenna = H.shape[1]

        if D is None:
            D = np.eye(num_antenna)
        # if D is None:
        #    D = np.eye(num_antenna)*0
        #    D[beam_idx][beam_idx]=1

        channel_vector = np.dot(H, D)
        # which norm is approtiate
        # w_mrt = channel_vector/np.linalg.norm(channel_vector,1)
        # I think 2 is correct, is it the sqrt(sum(abs(channel_vector))^2)
        channel_vector_norms = np.zeros(num_users)
        w_mrt_pwr = np.zeros((num_users, num_antenna), dtype='complex')
        w_mrt_voltage = np.zeros((num_users, num_antenna), dtype='complex')
        for n in range(num_users):
            channel_vector_norms[n] = np.linalg.norm(channel_vector[n], 2)

        min_ch = np.min(channel_vector_norms)
        max_ch = np.min(channel_vector_norms)
        for n in range(num_users):
            w = channel_vector[n] / channel_vector_norms[n]
            w = np.zeros(len(w))
            w[beam_idx] = 1
            w_mrt_pwr[n] = np.power(np.abs(w), 2) * np.exp(1j * np.angle(w))
            w_mrt_voltage[n] = np.abs(w) * np.exp(1j * np.angle(w))

        channel_vector_norm2 = np.linalg.norm(channel_vector, 2)
        w_mrt2 = channel_vector / channel_vector_norm2

        composite_sparam = np.zeros(num_users, dtype='complex')
        for n in range(num_users):
            composite_sparam[n] = np.dot(H[n], w_mrt_voltage[n].conj().T)
            # composite_sparam[n] = np.dot(H[n],np.conj(w_mrt_voltage[n].T))
        composite_weights = np.zeros(num_antenna, dtype='complex')
        for user in range(1):
            # composite_weights += np.sqrt(max_ch/np.abs(w_mrt[user]))*np.exp(1j*np.angle(w_mrt[user]*180/np.pi))
            composite_weights += w_mrt_voltage[user]
        test = 1
        return w_mrt_voltage, composite_sparam, composite_weights

    def _atleast_4d(self, x):
        if x.ndim < 4:
            y = np.expand_dims(np.atleast_3d(x), axis=0)
        else:
            y = x
        return y

    def weighting_multi_chirp_freq(self, H, D=None, chirp_idx=None, freq_idx=None, method='MRT', beam_index=0):

        '''


        Parameters
        ----------
        H : array coupling array with dimensions of  [num_users][num_antenna][chirps][freq] or [num_users][num_antenna]
            DESCRIPTION. If num_users is not defined, it is assumed to be 1
        D : array, optional
            diagnoal matrix with diagnol component of 1 if we want to transmit ot that users, 0 if we don't
        chirp_idx : TYPE, optional
            DESCRIPTION. The default is None.
        freq_idx : TYPE, optional
            DESCRIPTION. The default is None.
        method : STR, optional
            'MRT', 'ZFBF', 'Beam_Select'
        beam_index: int, optional
            use to select which beam, if 'Beam_Select' is chosen for method

        Returns
        -------
        Array
        normalized maximum ratio transmition weigting matrix.

        '''

        H = self._atleast_4d(H)

        num_users = H.shape[0]
        num_antenna = H.shape[1]
        num_chirps = H.shape[2]
        num_freq = H.shape[3]

        if freq_idx is not None:
            num_freq = 1
        if chirp_idx is not None:
            num_chirps = 1

        if D is None:
            D = np.eye(num_antenna)

        w_all = np.zeros((num_users, num_chirps, num_freq, num_antenna), dtype='complex')
        composite_sparams_all = np.zeros((num_users, num_chirps, num_freq), dtype='complex')
        composite_weights_all = np.zeros((num_antenna, num_chirps, num_freq), dtype='complex')
        for c_idx in range(num_chirps):
            if chirp_idx is not None:
                chirp_to_use = chirp_idx
            else:
                chirp_to_use = c_idx
            for f_idx in range(num_freq):
                if freq_idx is not None:
                    freq_to_use = freq_idx
                else:
                    freq_to_use = f_idx
                H_temp = H[:, :, chirp_to_use, freq_to_use]
                if method.lower() == 'mrt':
                    w, s, composite_weights = self.wMRT(H_temp, D)
                elif method.lower() == 'zfbf':
                    w, s, composite_weights = self.wZFBF(H_temp, D)
                elif method.lower() == 'beam_select':
                    w, s, composite_weights = self.wBeamSelect(H_temp, beam_idx=beam_index)
                w_all[:, c_idx, f_idx] = w
                composite_sparams_all[:, c_idx, f_idx] = s
                composite_weights_all[:, c_idx, f_idx] = composite_weights
        return w_all, np.array(composite_sparams_all), np.array(composite_weights_all)


    def gen_farfield_mesh_from_ffd(self, ff_data, theta, phi, mesh_limits=[0, 1],scale_pattern=1):
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
        ff_mesh.scale(scale_pattern,inplace=True)
        # ff_mesh.translate([float(position[0]),float(position[1]),float(position[2])])
        # this store the actual values of gain, that are used for color coating
        ff_mesh['FarFieldData'] = mag
        self.mesh = ff_mesh
