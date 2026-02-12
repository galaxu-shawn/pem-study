import numpy as np
import pem_utilities.pem_core as pem_core
import scipy.interpolate

class Movement:
    '''
    classs to store bulk movment and positions of actors
    '''

    def __init__(self,
                 time_stamps=None,
                 pos_all=np.zeros((1, 3)),
                 rot_all=np.eye(3),
                 lin_all=np.zeros((1, 3)),
                 ang_all=np.zeros((1, 3)),
                 offset_xyz=np.zeros((1, 3))):

        rss_py = pem_core.RssPy
        api = pem_core.api
        self.rss_py = rss_py
        self.api = api
        self.rot_all = rot_all
        if self.rot_all.ndim == 2:  # when passing in only a 3x3 matrix, we need to extend to a nx3x3
            self.rot_all = np.atleast_3d(rot_all)
            self.rot_all = np.moveaxis(self.rot_all, -1, 0)

        self.pos_all = np.atleast_2d(pos_all)  # -offset_xyz
        self.lin_all = np.atleast_2d(lin_all)
        self.ang_all = np.atleast_2d(ang_all)
        # offset_xyz[2] = 0 #only offset in xy direction for now
        self.offset_xyz = offset_xyz
        self.time_stamps = time_stamps

    @property
    def pos_all(self):
        return self._pos_all  # -self.offset_xyz

    @pos_all.setter
    def pos_all(self, value):
        self._pos_all = np.atleast_2d(value)

    @property
    def rot_all(self):
        return self._rot_all

    @rot_all.setter
    def rot_all(self, value):
        if value.ndim == 2:
            temp = np.atleast_3d(value)
            temp = np.moveaxis(temp, -1, 0)
            self._rot_all = np.atleast_3d(temp)
        else:
            self._rot_all = np.atleast_3d(value)

    @property
    def lin_all(self):
        return self._lin_all

    @lin_all.setter
    def lin_all(self, value):
        self._lin_all = np.atleast_2d(value)

    @property
    def ang_all(self):
        return self._ang_all

    @ang_all.setter
    def ang_all(self, value):
        self._ang_all = np.atleast_2d(value)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, value):
        self._rot = value

    @property
    def lin(self):
        return self._lin

    @lin.setter
    def lin(self, value):
        self._lin = value

    @property
    def ang(self):
        return self._ang

    @ang.setter
    def ang(self, value):
        self._ang = value

    def pos_interp(self, times):
        pos_interp_func = scipy.interpolate.interp1d(self.time_stamps, self._pos, axis=0, assume_sorted=True,
                                                     bounds_error=False, fill_value='extrapolate')
        pos = pos_interp_func(times)
        return pos

    def rot_interp(self, times):
        rot_interp_func = scipy.interpolate.interp1d(self.time_stamps, self._rot, axis=0, assume_sorted=True,
                                                     bounds_error=False, fill_value='extrapolate')
        rot = rot_interp_func(times)
        return rot


    def lin_interp(self, times):
        lin_interp_func = scipy.interpolate.interp1d(self.time_stamps, self._lin, axis=0, assume_sorted=True,
                                                     bounds_error=False, fill_value='extrapolate')
        lin = lin_interp_func(times)
        return lin


    def ang_interp(self, times):
        ang_interp_func = scipy.interpolate.interp1d(self.time_stamps, self._ang, axis=0, assume_sorted=True,
                                                     bounds_error=False, fill_value='extrapolate')
        ang = ang_interp_func(times)
        return ang



    def estimate_velocity(self, timeStamps, correct_first_point=True, order=3):
        '''
        Make any actor with only a single time input, an array of values equal to lenght of time stamps

        Parameters
        ----------
        allActors : TYPE
            DESCRIPTION.
        timeStamps : TYPE
            DESCRIPTION.

        Returns
        -------
        allActors : TYPE
            DESCRIPTION.

        '''

        num_time = len(timeStamps)
        print("INFO: Velocity estimates are being performed")
        dt = 0
        last_time = 0
        velEst = None
        lin_all = np.zeros((len(timeStamps), 3))
        ang_all = np.zeros((len(timeStamps), 3))
        for idx in range(len(timeStamps)):
            current_time = float(timeStamps[idx])
            dt = current_time - last_time
            if self.rot_all.shape[0] != num_time:
                rot = self.rot_all[0]
            else:
                rot = self.rot_all[idx]
            if self.pos_all.shape[0] != num_time:
                pos = self.pos_all[0]
            else:
                pos = self.pos_all[idx]
            if (velEst is None) or (dt <= 0):
                velEst = self.RssPy.VelocityEstimate()
                velEst.setApproximationOrder(order)
            ret = velEst.push(
                current_time,
                np.ascontiguousarray(rot, dtype=np.float64),
                np.ascontiguousarray(pos, dtype=np.float64))
            if ret == False:
                # print(current_time)
                raise RuntimeError("error pushing velocity estimate")
            (_, lin, ang) = velEst.get()

            last_time = current_time
            lin_all[idx] = lin
            ang_all[idx] = ang

        if correct_first_point:
            if order == 2:
                if len(lin_all) >= 2:
                    print(
                        "INFO: First point in velocity estimate is not known, using second data point for time step 0")
                    lin_all[0] = lin_all[1]
                    ang_all[0] = ang_all[1]
            else:
                if len(lin_all) >= 3:
                    print(
                        "INFO: First 2 points in velocity estimate is not known, using third data point for time step 0 and 1")
                    lin_all[0] = lin_all[2]
                    ang_all[0] = ang_all[2]
                    lin_all[1] = lin_all[2]
                    ang_all[1] = ang_all[2]
        return lin_all, ang_all