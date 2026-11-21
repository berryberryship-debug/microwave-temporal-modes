"""
config.py
Physical constants, cryogenic link parameters, and hardware constraints
for the TemporalMode-QNet simulation engine.

NB : les valeurs ci-dessous sont ILLUSTRATIVES. Elles ne sont pas celles de
arXiv:2604.12947 (kappa_A/2pi = 26.7 MHz, kappa_B/2pi = 30.7 MHz, Gamma/2pi = 14 ou 24 MHz).
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HardwareConfig:
    # Carrier & Qubit Parameters (in rad/ns, i.e., 2*pi * GHz)
    omega_0: float = 2 * np.pi * 6.0          # Carrier frequency: 6.0 GHz [non verifie]
    alpha_q: float = -2 * np.pi * 0.220       # Transmon anharmonicity: -220 MHz [non verifie]

    # 30-Meter Cryogenic Link Parameters
    link_length_m: float = 30.0               # Physical waveguide length (m)
    v_group: float = 2.0e8                    # [A VERIFIER] arXiv:2604.12947 : ~0.72c ; non utilise dans wavepackets.py
    beta_2: float = 1.5e-22                   # [SOURCE INCONNUE] s^2/m ; effet negligeable (~1e-16) avec tau = 45 ns
    loss_db_per_m: float = 0.005              # [A VERIFIER] arXiv:2604.12947 : guide < 1 dB/km (0.001 dB/m)

    # Tunable Coupler & Buffer Constraints (in rad/ns)
    gamma_max: float = 2 * np.pi * 0.050      # Max coupler decay rate: 50 MHz
    kappa_buffer: float = 2 * np.pi * 0.150   # Buffer resonator linewidth: 150 MHz
    stark_coeff: float = -0.15                # Ratio of AC Stark shift to instantaneous gamma(t) [modele, non source]

    # Temporal Grid & Pulse Envelope Parameters (in nanoseconds)
    t_min: float = -250.0                     # Window start (ns)
    t_max: float = 250.0                      # Window end (ns)
    n_steps: int = 2001                       # Time grid points
    tau_pulse: float = 45.0                   # Characteristic wave packet width (ns)
    eps_reg: float = 1e-12                    # Numerical regularization for gamma(t) denominator

    # Seuils de regularisation (fractions de l'energie cumulee du mode).
    # Valeurs historiques : cum_lo = 1e-5, cum_hi = 0.9995 (perte de capture >= 5e-4).
    cum_lo: float = 1e-5                      # borne la queue precoce (diverge pour un mode gaussien)
    cum_hi: float = 1.0 - 1e-6                # |g_v|^2 reste fini en fin de mode : seuil quasi nul
    emit_remaining_min: float = 1e-6          # cote emetteur : 1 - int|u|^2 > seuil

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
