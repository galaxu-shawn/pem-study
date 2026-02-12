# -*- coding: utf-8 -*-
"""
Generate theta/phi values from start/stop/step(deg/num).

useful for RCS simulations, ISAR imaging, and other radar applications where we are sweeping angles

"""


import numpy as np


class ThetaPhiSweepGenerator:
    """

    """
    
    def __init__(self):
        """

        """

        self._theta_start = 0
        self._theta_stop = 180
        self._theta_step_deg = 1
        self._theta_step_num = 181 # these deg/num values will be used to calculate the other one

        self._theta_domain = np.linspace(self._theta_start, self._theta_stop, self._theta_step_num)

        self._phi_start = 0
        self._phi_stop = 360
        self._phi_step_deg = 1
        self._phi_step_num = 361 # these deg/num values will be used to calculate the other one

        self._phi_domain = np.linspace(self._phi_start, self._phi_stop, self._phi_step_num)   

    def get_all_theta_phi_vals(self):
        """
        Generate all theta and phi values based on the configured ranges and steps.
        
        Returns:
            np.ndarray: Array of shape (num_theta, num_phi, 2) containing theta and phi values.
        """

        theta_grid, phi_grid = np.meshgrid(self.theta_domain, self.phi_domain, indexing='ij')
        all_vals = np.stack((theta_grid, phi_grid), axis=-1)    
        # flatten the array to return a 2D array of shape (num_theta * num_phi, 2)
        self.all_theta_phi_vals = np.reshape(all_vals, (-1, 2))
        return self.all_theta_phi_vals


    @property
    def theta_domain(self):
        """Get the theta domain values."""
        if self._theta_start > self._theta_stop:
            raise ValueError("Theta start must be less than or equal to theta stop")
        if self._theta_step_deg < 0:
            raise ValueError("Theta step number must be positive")
        if self._theta_step_num <= 0:
            raise ValueError("Theta step number must be positive")
        self._theta_domain = np.linspace(self._theta_start, self._theta_stop, self._theta_step_num)
        return self._theta_domain
    @theta_domain.setter
    def theta_domain(self, value):
        """Set the theta domain values with validation."""
        if len(value) < 1:
            raise ValueError("Theta domain must have at least 1 value")
        self._theta_domain = np.array(value)
        self._theta_step_num = len(self._theta_domain)
        if len(value)==1:
            self._theta_step_deg = 0
        else:
            self._theta_step_deg = value[1] - value[0]

    @property
    def phi_domain(self):
        """Get the phi domain values."""

        if self._phi_start > self._phi_stop:
            raise ValueError("Phi start must be less than or equal to phi stop")
        if self._phi_step_deg < 0:
            raise ValueError("Phi step must be positive")
        if self._phi_step_num <= 0:
            raise ValueError("Phi step number must be positive")
        self._phi_domain = np.linspace(self._phi_start, self._phi_stop, self._phi_step_num)
        return self._phi_domain
    @phi_domain.setter
    def phi_domain(self, value):
        """Set the phi domain values with validation."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError("Phi domain must be a list, tuple, or numpy array")
        if len(value) < 2:
            raise ValueError("Phi domain must have at least two values")
        self._phi_domain = np.array(value)
        self._phi_step_num = len(self._phi_domain)
        self._phi_step_deg = (self.phi_stop - self.phi_start) / (self._phi_step_num - 1)

    @property
    def theta_start(self):
        """Get the starting angle for theta."""
        return self._theta_start
    @theta_start.setter
    def theta_start(self, value):
        """Set the starting angle for theta with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Theta start must be a number")
        if value < 0 or value > 180:
            raise ValueError("Theta start must be between 0 and 180 degrees")
        self._theta_start = float(value)

    @property
    def theta_stop(self):
        """Get the stopping angle for theta."""
        return self._theta_stop
    @theta_stop.setter
    def theta_stop(self, value):
        """Set the stopping angle for theta with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Theta stop must be a number")
        if value < 0 or value > 180:
            raise ValueError("Theta stop must be between 0 and 180 degrees")
        self._theta_stop = float(value)

    @property
    def theta_step_deg(self):
        """Get the step size for theta."""
        return self._theta_step_deg
    @theta_step_deg.setter
    def theta_step_deg(self, value):
        """Set the step size for theta with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Theta step must be a number")
        if value < 0:
            raise ValueError("Theta step must be positive")
        if value == 0:
            if self.phi_start != self.phi_stop:
                raise ValueError("Phi step must be positive")
        self._theta_step_deg = float(value)
        if self.theta_start <= self.theta_stop :
            self._theta_step_num = int((self.theta_stop - self.theta_start) / self._theta_step_deg) + 1
        else:
            self._theta_step_num = 1

    @property
    def theta_step_num(self):
        """Get the number of theta steps."""
        return self._theta_step_num
    @theta_step_num.setter
    def theta_step_num(self, value):
        """Set the number of theta steps with validation."""
        if not isinstance(value, int):
            raise TypeError("Theta step number must be an integer")
        if value <= 0:
            raise ValueError("Theta step number must be positive")
        self._theta_step_num = value
        if self._theta_step_num == 1:
            self._theta_step_deg = 0
        else:
            self._theta_step_deg = (self.theta_stop - self.theta_start) / (self._theta_step_num-1)

    @property
    def phi_start(self):
        """Get the starting angle for phi."""
        return self._phi_start
    @phi_start.setter
    def phi_start(self, value):
        """Set the starting angle for phi with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Phi start must be a number")
        if np.abs(value) > 360:
            raise ValueError("Phi start must be equal to or between -360 and 360 degrees")
        self._phi_start = float(value)

    @property
    def phi_stop(self):
        """Get the stopping angle for phi."""
        return self._phi_stop
    @phi_stop.setter
    def phi_stop(self, value):
        """Set the stopping angle for phi with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Phi stop must be a number")
        if np.abs(value) > 360:
            raise ValueError("Phi stop must be equal to or between -360 and 360 degrees")
        self._phi_stop = float(value)

    @property
    def phi_step_deg(self):
        """Get the step size for phi."""
        return self._phi_step_deg
    @phi_step_deg.setter
    def phi_step_deg(self, value):
        """Set the step size for phi with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Phi step must be a number")
        if value == 0:
            if self.phi_start != self.phi_stop:
                raise ValueError("Phi step must be positive")

        self._phi_step_deg = float(value)
        self._phi_step_num = int((self.phi_stop - self.phi_start) / self._phi_step_deg) + 1 if self.phi_start < self.phi_stop else 1


    @property
    def phi_step_num(self):
        """Get the number of phi steps."""
        return self._phi_step_num
    @phi_step_num.setter
    def phi_step_num(self, value):
        """Set the number of phi steps with validation."""
        if not isinstance(value, int):
            raise TypeError("Phi step number must be an integer")
        if value <= 0:
            raise ValueError("Phi step number must be positive")
        self._phi_step_num = value
        if self._phi_step_num == 1:
            self._phi_step_deg = 0
        else:
            self._phi_step_deg = (self.phi_stop - self.phi_start) / (self._phi_step_num -1)