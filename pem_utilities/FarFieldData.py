from dataclasses import dataclass, field
import numpy as np

@dataclass
class FF_Data:
    Num_Freq: int = 1
    rETheta: np.ndarray = field(default_factory=lambda: np.empty(0))
    rEPhi: np.ndarray = field(default_factory=lambda: np.empty(0))
    Misc_FarField: np.ndarray = field(default_factory=lambda: np.empty(0))  # For storing additional data
    Theta: np.ndarray = field(default_factory=lambda: np.empty(0))
    Phi: np.ndarray = field(default_factory=lambda: np.empty(0))
    Frequencies: np.ndarray = field(default_factory=lambda: np.empty(0))
    Delta_Theta: float = 1
    Delta_Phi: float = 1
    Diff_Area: float = 1
    Total_Samples: int = 1
    Position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Lattice_Vector: np.ndarray = field(default_factory=lambda: np.zeros(6))
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