"""
benchmarks.py
Évaluation des performances du réseau quantique en modes temporels orthogonaux :
1. Matrice N x N de sélectivité d'absorption et de diaphonie inter-modes (eta_mn).
2. Courbe de visibilité Hong-Ou-Mandel (HOM) en fonction du chevauchement temporel J.
3. Tomographie de séparateur de faisceau temporel (rotations de base dual-rail).
"""

from dataclasses import dataclass
import numpy as np
from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    generate_gram_schmidt_exponential_basis,
    compute_impedance_matched_coupling,
)
from slh_solver import simulate_cascaded_scattering
from calibration import predistort_emitter_dispersion, calibrate_receiver_in_situ
from wavepackets import propagate_dispersive_link


@dataclass
class BenchmarkResults:
    selectivity_matrix: np.ndarray      # Matrice eta[m, n] = P_e(récepteur m | mode incident n)
    selectivity_contrast_db: float      # Rapport diagonal / hors-diagonale moyen en dB
    mixing_angles: np.ndarray           # Angles theta du séparateur de faisceau temporel
    dual_rail_capture: np.ndarray       # Probabilité de capture P_e(theta) pour w_+(theta)
    hom_overlaps: np.ndarray            # Chevauchements |J|^2 = |<u_0 | u_probe>|^2
    hom_two_photon_reflected: np.ndarray  # Flux total réfléchi à 2 photons vs |J|^2


def compute_selectivity_matrix(
    cfg: HardwareConfig,
    n_modes: int = 3,
    basis_type: str = "hermite_gauss",
    calibrate: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Calcule la matrice N x N de routage sélectif :
        eta[m, n] = probabilité d'excitation P_e du récepteur programmé sur le mode m
                    lorsque le mode incident n traverse le lien de 30 m.
    """
    t = cfg.time_grid
    if basis_type == "hermite_gauss":
        modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=n_modes)
    elif basis_type == "gram_schmidt":
        modes = generate_gram_schmidt_exponential_basis(t, cfg.tau_pulse, n_modes=n_modes)
    else:
        raise ValueError(f"Base inconnue : {basis_type}")

    # Pré-distorsion à l'émission et propagation sur 30 m
    arriving_modes = np.zeros_like(modes)
    for n in range(n_modes):
        u_pre = predistort_emitter_dispersion(modes[n], cfg)
        arriving_modes[n] = propagate_dispersive_link(u_pre, cfg, apply_loss=False)

    eta_matrix = np.zeros((n_modes, n_modes))

    for m in range(n_modes):
        ortho_idx = (m + 1) % n_modes
        if calibrate:
            report = calibrate_receiver_in_situ(
                arriving_modes[m], arriving_modes[ortho_idx], cfg, n_iterations=10
            )
            g_rec_m = report.g_rec_calibrated
        else:
            _, g_rec_m = compute_impedance_matched_coupling(
                arriving_modes[m], cfg, apply_hardware_limits=True
            )

        for n in range(n_modes):
            res = simulate_cascaded_scattering(
                arriving_modes[n],
                g_rec_m,
                cfg,
                initial_photon_state="single",
                include_stark_shift=True,
            )
            eta_matrix[m, n] = res.p_e[-1]

    diag_mean = float(np.mean(np.diag(eta_matrix)))
    off_diag_mask = ~np.eye(n_modes, dtype=bool)
    off_diag_mean = float(np.mean(eta_matrix[off_diag_mask]))
    contrast_db = 10.0 * np.log10((diag_mean + 1e-12) / (off_diag_mean + 1e-12))

    return eta_matrix, contrast_db


def sweep_temporal_beam_splitter(
    cfg: HardwareConfig, n_angles: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vérifie le comportement de séparateur de faisceau temporel programmable :
    un photon dans u_0 est analysé par un récepteur programmé sur la superposition
        w_+(theta) = cos(theta) * u_0 + sin(theta) * u_1.
    Théoriquement : P_e(theta) = cos^2(theta).
    """
    t = cfg.time_grid
    modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)
    thetas = np.linspace(0.0, np.pi / 2.0, n_angles)
    p_capture = np.zeros(n_angles)

    for idx, theta in enumerate(thetas):
        w_plus = np.cos(theta) * modes[0] + np.sin(theta) * modes[1]
        _, g_rec = compute_impedance_matched_coupling(
            w_plus, cfg, apply_hardware_limits=True
        )
        res = simulate_cascaded_scattering(
            modes[0],
            g_rec,
            cfg,
            initial_photon_state="single",
            include_stark_shift=False,
        )
        p_capture[idx] = res.p_e[-1]

    return thetas, p_capture


def sweep_two_photon_saturation(
    cfg: HardwareConfig, n_points: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sonde la saturation non linéaire à deux photons |2_{w(theta)}> sur un récepteur
    accordé à u_0, en faisant varier le chevauchement modal |J|^2 = cos^2(theta).
    """
    t = cfg.time_grid
    modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)
    _, g_rec_0 = compute_impedance_matched_coupling(
        modes[0], cfg, apply_hardware_limits=True
    )

    thetas = np.linspace(0.0, np.pi / 2.0, n_points)
    overlaps_j2 = np.cos(thetas) ** 2
    reflected_photons = np.zeros(n_points)

    for idx, theta in enumerate(thetas):
        u_in = np.cos(theta) * modes[0] + np.sin(theta) * modes[1]
        res = simulate_cascaded_scattering(
            u_in,
            g_rec_0,
            cfg,
            initial_photon_state="two_photon",
            include_stark_shift=False,
        )
        reflected_photons[idx] = res.total_reflected

    return overlaps_j2, reflected_photons


def run_all_benchmarks(cfg: HardwareConfig, n_modes: int = 3) -> BenchmarkResults:
    eta_mat, contrast_db = compute_selectivity_matrix(
        cfg, n_modes=n_modes, basis_type="hermite_gauss", calibrate=True
    )
    thetas, dual_rail_p = sweep_temporal_beam_splitter(cfg, n_angles=7)
    overlaps, two_ph_ref = sweep_two_photon_saturation(cfg, n_points=6)

    return BenchmarkResults(
        selectivity_matrix=eta_mat,
        selectivity_contrast_db=contrast_db,
        mixing_angles=thetas,
        dual_rail_capture=dual_rail_p,
        hom_overlaps=overlaps,
        hom_two_photon_reflected=two_ph_ref,
    )
