"""
config.py
Physical constants, cryogenic link parameters, and hardware constraints
for the TemporalMode-QNet simulation engine.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HardwareConfig:
    # Carrier & Qubit Parameters (in rad/ns, i.e., 2*pi * GHz)
    omega_0: float = 2 * np.pi * 6.0          # Carrier frequency: 6.0 GHz
    alpha_q: float = -2 * np.pi * 0.220       # Transmon anharmonicity: -220 MHz

    # 30-Meter Cryogenic Link Parameters
    link_length_m: float = 30.0               # Physical waveguide/coax length (m)
    v_group: float = 2.0e8                    # Group velocity in NbTi coax (~0.67c, m/s)
    beta_2: float = 1.5e-22                   # Group velocity dispersion (s^2/m)
    loss_db_per_m: float = 0.005              # Cryogenic attenuation (dB/m)

    # Tunable Coupler & Buffer Constraints (in rad/ns)
    gamma_max: float = 2 * np.pi * 0.050      # Max coupler decay rate: 50 MHz
    kappa_buffer: float = 2 * np.pi * 0.150   # Buffer resonator linewidth: 150 MHz
    stark_coeff: float = -0.15                # Ratio of AC Stark shift to instantaneous gamma(t)

    # Temporal Grid & Pulse Envelope Parameters (in nanoseconds)
    t_min: float = -250.0                     # Window start (ns)
    t_max: float = 250.0                      # Window end (ns)
    n_steps: int = 2001                       # Time grid points
    tau_pulse: float = 45.0                   # Characteristic wave packet width (ns)
    eps_reg: float = 1e-12                    # Numerical regularization for gamma(t) denominator

    @property
    def dt(self) -> float:
        return (self.t_max - self.t_min) / (self.n_steps - 1)

    @property
    def time_grid(self) -> np.ndarray:
        return np.linspace(self.t_min, self.t_max, self.n_steps)

    @property
    def link_transmission(self) -> float:
        """Power transmission efficiency eta = 10^(-loss_dB / 10)."""
        total_loss_db = self.loss_db_per_m * self.link_length_m
        return 10.0 ** (-total_loss_db / 10.0)
