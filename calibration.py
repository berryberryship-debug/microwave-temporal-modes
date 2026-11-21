"""
calibration.py
Calibration en boucle fermee du couplage recepteur g_rec(t).

REMPLACEMENT : l'ancienne regle de mise a jour heuristique (correction point par point
a partir de Re/Im de <L_tot>/g_rec) a ete retiree. Mesuree avec le solveur corrige, elle
degradait la capture (97,75 % -> 73,39 %) pour toutes les valeurs de pas testees.

Nouvelle approche : optimisation sans gradient (Nelder-Mead) d'un petit nombre de
parametres physiques, avec pour objectif l'energie sortante mesuree pour l'etat sonde
(|0>+|1>)/sqrt(2) :
    g_rec(t) = a * |g0(t)| * exp( i [ arg g0(t) + s * Theta0(t) ] ),   Theta0(t) = int_{-inf}^t |g0|^2 d tau
  - s : phase proportionnelle a l'integrale du taux, qui compense un decalage Stark
        delta(t) = stark_coeff * |g(t)|^2 (la phase accumulee vaut stark_coeff * Theta0),
  - a : facteur d'amplitude global (borne par sqrt(gamma_max)).
Ce modele suppose que le decalage Stark suit la forme ci-dessus ; il ne compense pas
d'autres imperfections. Aucune comparaison a une mesure reelle n'a ete faite.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    propagate_dispersive_link,
    compute_impedance_matched_coupling,
)
from slh_solver import simulate_cascaded_scattering


@dataclass
class CalibrationReport:
    g_rec_calibrated: np.ndarray        # Profil de couplage complexe optimise g_rec(t)
    reflection_history: list            # Objectif (energie sortante, etat sonde) a chaque evaluation
    capture_before: float               # P_e avant calibration
    capture_after: float                # P_e apres calibration
    crosstalk_before: float             # P_e du mode orthogonal avant calibration
    crosstalk_after: float              # P_e du mode orthogonal apres calibration
    amp_scale: float                    # parametre a optimise
    stark_phase_coeff: float            # parametre s optimise


def predistort_emitter_dispersion(u_target: np.ndarray, cfg: HardwareConfig) -> np.ndarray:
    """
    Phase conjuguee de la dispersion beta_2 * L (inverse de propagate_dispersive_link).
    Avec beta_2 = 1.5e-22 s^2/m l'effet est negligeable (~1e-16).
    """
    dt_sec = cfg.dt * 1e-9
    freqs_hz = np.fft.fftfreq(len(u_target), d=dt_sec)
    delta_omega = 2.0 * np.pi * freqs_hz

    u_omega = np.fft.fft(u_target)
    inverse_dispersion = np.exp(+0.5j * cfg.beta_2 * cfg.link_length_m * (delta_omega**2))
    u_predistorted = np.fft.ifft(u_omega * inverse_dispersion)

    norm = np.sqrt(np.trapezoid(np.abs(u_predistorted) ** 2, dx=cfg.dt))
    return u_predistorted / norm


def _build_g_rec(g0: np.ndarray, theta0: np.ndarray, a: float, s: float, cfg: HardwareConfig) -> np.ndarray:
    amp = np.clip(a * np.abs(g0), 0.0, np.sqrt(cfg.gamma_max))
    return amp * np.exp(1j * (np.angle(g0) + s * theta0))


def calibrate_receiver_in_situ(
    u_arriving: np.ndarray,
    u_orthogonal: np.ndarray,
    cfg: HardwareConfig,
    n_iterations: int = 60,
) -> CalibrationReport:
    """
    n_iterations : nombre maximal d'iterations de Nelder-Mead.
    """
    _, g0 = compute_impedance_matched_coupling(u_arriving, cfg, apply_hardware_limits=True)
    theta0 = cumulative_trapezoid(np.abs(g0) ** 2, dx=cfg.dt, initial=0.0)

    res_init_match = simulate_cascaded_scattering(
        u_arriving, g0, cfg, initial_photon_state="single", include_stark_shift=True
    )
    res_init_ortho = simulate_cascaded_scattering(
        u_orthogonal, g0, cfg, initial_photon_state="single", include_stark_shift=True
    )

    history: list[float] = []

    def objective(params: np.ndarray) -> float:
        a, s = float(params[0]), float(params[1])
        g = _build_g_rec(g0, theta0, a, s, cfg)
        probe = simulate_cascaded_scattering(
            u_arriving, g, cfg, initial_photon_state="superposition", include_stark_shift=True
        )
        history.append(probe.total_reflected)
        return probe.total_reflected

    opt = minimize(
        objective,
        x0=np.array([1.0, 0.0]),
        method="Nelder-Mead",
        options={"maxiter": n_iterations, "xatol": 1e-4, "fatol": 1e-9},
    )
    a_opt, s_opt = float(opt.x[0]), float(opt.x[1])
    g_rec = _build_g_rec(g0, theta0, a_opt, s_opt, cfg)

    res_final_match = simulate_cascaded_scattering(
        u_arriving, g_rec, cfg, initial_photon_state="single", include_stark_shift=True
    )
    res_final_ortho = simulate_cascaded_scattering(
        u_orthogonal, g_rec, cfg, initial_photon_state="single", include_stark_shift=True
    )

    return CalibrationReport(
        g_rec_calibrated=g_rec,
        reflection_history=history,
        capture_before=float(res_init_match.p_e[-1]),
        capture_after=float(res_final_match.p_e[-1]),
        crosstalk_before=float(res_init_ortho.p_e[-1]),
        crosstalk_after=float(res_final_ortho.p_e[-1]),
        amp_scale=a_opt,
        stark_phase_coeff=s_opt,
    )


if __name__ == "__main__":
    cfg = HardwareConfig()
    t = cfg.time_grid
    modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)

    u0_pre = predistort_emitter_dispersion(modes[0], cfg)
    u1_pre = predistort_emitter_dispersion(modes[1], cfg)

    u0_arriving = propagate_dispersive_link(u0_pre, cfg, apply_loss=False)
    u1_arriving = propagate_dispersive_link(u1_pre, cfg, apply_loss=False)

    report = calibrate_receiver_in_situ(u0_arriving, u1_arriving, cfg)

    print(f"[INFO] Objectif (sortie, etat sonde) : {report.reflection_history[0]:.4f} -> {report.reflection_history[-1]:.2e} "
          f"({len(report.reflection_history)} evaluations)")
    print(f"[INFO] Parametres : a = {report.amp_scale:.4f}, s = {report.stark_phase_coeff:.4f} (stark_coeff = {cfg.stark_coeff})")
    print(f"[INFO] Capture mode cible HG0 : {report.capture_before*100:.2f}% -> {report.capture_after*100:.2f}%")
    print(f"[INFO] Diaphonie mode orthogonal HG1 : {report.crosstalk_before*100:.2f}% -> {report.crosstalk_after*100:.2f}%")
