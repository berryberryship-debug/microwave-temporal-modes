"""
benchmarks.py
Evaluation du routage en modes temporels orthogonaux, dans le modele ideal du solveur
(sans perte de photon sur le lien, sans erreurs de controle).

Contenu REEL de ce module (revision) :
1. Matrice N x N de selectivite d'absorption eta[m, n] et contraste diagonal / hors-diagonale
   (meme definition que le rapport de selectivite de arXiv:2604.12947 : moyenne de la
   diagonale sur moyenne hors-diagonale ; 10 log10 pour les dB).
2. Balayage "separateur de faisceau temporel" : capture d'un photon u_0 par un recepteur
   programme sur w(theta) = cos(theta) u_0 + sin(theta) u_1 ; attendu P_e = cos^2(theta)
   (c'est un recouvrement lineaire a un photon, pas une tomographie).
3. Balayage a deux photons : photons sortants en fonction du recouvrement modal |J|^2.
   Ce n'est PAS une courbe de visibilite Hong-Ou-Mandel (aucune interference sur un
   separateur de faisceau n'est calculee), et aucune fidelite d'intrication dual-rail
   n'est calculee dans ce depot.
"""

from dataclasses import dataclass
import numpy as np
from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    generate_gram_schmidt_exponential_basis,
    generate_sech_polynomial_basis,
    compute_impedance_matched_coupling,
    propagate_dispersive_link,
)
from slh_solver import simulate_cascaded_scattering
from calibration import predistort_emitter_dispersion, calibrate_receiver_in_situ

SECH_GAMMA_BW = 2.0 * np.pi * 0.014   # Gamma/2pi = 14 MHz (arXiv:2604.12947), en rad/ns


@dataclass
class BenchmarkResults:
    selectivity_matrix: np.ndarray        # eta[m, n] = P_e(recepteur m | mode incident n)
    selectivity_contrast_db: float        # 10 log10(moyenne diagonale / moyenne hors-diagonale)
    mixing_angles: np.ndarray             # angles theta (balayage separateur de faisceau)
    beam_splitter_capture: np.ndarray     # P_e(theta) pour un recepteur programme sur w(theta)
    two_photon_overlaps: np.ndarray       # |J|^2 = |<u_0 | u_in>|^2
    two_photon_outgoing: np.ndarray       # photons sortants (sur 2 incidents) vs |J|^2


def _make_basis(cfg: HardwareConfig, n_modes: int, basis_type: str) -> np.ndarray:
    t = cfg.time_grid
    if basis_type == "hermite_gauss":
        return generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=n_modes)
    if basis_type == "gram_schmidt":
        return generate_gram_schmidt_exponential_basis(t, cfg.tau_pulse, n_modes=n_modes)
    if basis_type == "sech":
        if n_modes > 3:
            raise ValueError("La base sech implementee est limitee a 3 modes.")
        return generate_sech_polynomial_basis(t, SECH_GAMMA_BW)[:n_modes]
    raise ValueError(f"Base inconnue : {basis_type}")


def compute_selectivity_matrix(
    cfg: HardwareConfig,
    n_modes: int = 3,
    basis_type: str = "hermite_gauss",
    calibrate: bool = True,
    calibration_iterations: int = 60,
) -> tuple[np.ndarray, float]:
    """
    eta[m, n] = P_e du recepteur programme sur le mode m quand le mode n a traverse le lien.
    Le contraste utilise un plancher 1e-12 : si les termes hors-diagonale sont ~0, la valeur
    en dB est dominee par ce plancher et n'a pas de sens physique.
    NB : 10 iterations (ancienne valeur) ne suffisent pas a Nelder-Mead pour converger ;
    defaut porte a 60.
    """
    modes = _make_basis(cfg, n_modes, basis_type)

    arriving_modes = np.zeros_like(modes)
    for n in range(n_modes):
        u_pre = predistort_emitter_dispersion(modes[n], cfg)
        arriving_modes[n] = propagate_dispersive_link(u_pre, cfg, apply_loss=False)

    eta_matrix = np.zeros((n_modes, n_modes))

    for m in range(n_modes):
        ortho_idx = (m + 1) % n_modes
        if calibrate:
            report = calibrate_receiver_in_situ(
                arriving_modes[m], arriving_modes[ortho_idx], cfg,
                n_iterations=calibration_iterations,
            )
            g_rec_m = report.g_rec_calibrated
        else:
            _, g_rec_m = compute_impedance_matched_coupling(
                arriving_modes[m], cfg, apply_hardware_limits=True
            )

        for n in range(n_modes):
            res = simulate_cascaded_scattering(
                arriving_modes[n], g_rec_m, cfg,
                initial_photon_state="single", include_stark_shift=True,
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
    Photon dans u_0, recepteur programme sur w(theta) = cos(theta) u_0 + sin(theta) u_1.
    Attendu : P_e(theta) = cos^2(theta) (verifie a ~2e-4 pres, sans Stark).
    """
    t = cfg.time_grid
    modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)
    thetas = np.linspace(0.0, np.pi / 2.0, n_angles)
    p_capture = np.zeros(n_angles)

    for idx, theta in enumerate(thetas):
        w_plus = np.cos(theta) * modes[0] + np.sin(theta) * modes[1]
        _, g_rec = compute_impedance_matched_coupling(w_plus, cfg, apply_hardware_limits=True)
        res = simulate_cascaded_scattering(
            modes[0], g_rec, cfg, initial_photon_state="single", include_stark_shift=False,
        )
        p_capture[idx] = res.p_e[-1]

    return thetas, p_capture


def sweep_two_photon_saturation(
    cfg: HardwareConfig, n_points: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deux photons dans w(theta) sur un recepteur accorde a u_0, |J|^2 = cos^2(theta).
    Sortie = photons sortants (flux integre). Controle de coherence : avec alpha_q = 0
    (recepteur lineaire) on obtient 0 photon sortant pour |J|^2 = 1 et 2,000 pour |J|^2 = 0 ;
    l'ecart observe avec alpha_q != 0 est un effet non lineaire du transmon.
    """
    t = cfg.time_grid
    modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)
    _, g_rec_0 = compute_impedance_matched_coupling(modes[0], cfg, apply_hardware_limits=True)

    thetas = np.linspace(0.0, np.pi / 2.0, n_points)
    overlaps_j2 = np.cos(thetas) ** 2
    outgoing = np.zeros(n_points)

    for idx, theta in enumerate(thetas):
        u_in = np.cos(theta) * modes[0] + np.sin(theta) * modes[1]
        res = simulate_cascaded_scattering(
            u_in, g_rec_0, cfg, initial_photon_state="two_photon", include_stark_shift=False,
        )
        outgoing[idx] = res.total_reflected

    return overlaps_j2, outgoing


def run_all_benchmarks(
    cfg: HardwareConfig,
    n_modes: int = 3,
    basis_type: str = "hermite_gauss",
    calibrate: bool = True,
) -> BenchmarkResults:
    eta_mat, contrast_db = compute_selectivity_matrix(
        cfg, n_modes=n_modes, basis_type=basis_type, calibrate=calibrate
    )
    thetas, bs_p = sweep_temporal_beam_splitter(cfg, n_angles=7)
    overlaps, outgoing = sweep_two_photon_saturation(cfg, n_points=6)

    return BenchmarkResults(
        selectivity_matrix=eta_mat,
        selectivity_contrast_db=contrast_db,
        mixing_angles=thetas,
        beam_splitter_capture=bs_p,
        two_photon_overlaps=overlaps,
        two_photon_outgoing=outgoing,
    )
