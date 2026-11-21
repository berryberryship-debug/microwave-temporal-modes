"""
calibration.py
Calibration in situ par tomographie hétérodyne IQ et pré-distorsion itérative
des enveloppes d'émission et de réception (dispersion 30m + décalage AC Stark).
"""

from dataclasses import dataclass
import numpy as np
from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    propagate_dispersive_link,
    compute_impedance_matched_coupling,
)
from slh_solver import simulate_cascaded_scattering


@dataclass
class CalibrationReport:
    g_rec_calibrated: np.ndarray        # Profil de couplage complexe optimisé g_rec(t)
    reflection_history: list[float]     # Énergie réfléchie résiduelle à chaque itération
    capture_before: float               # Probabilité de capture P_e avant calibration
    capture_after: float                # Probabilité de capture P_e après calibration
    crosstalk_before: float             # Diaphonie du mode orthogonal avant calibration
    crosstalk_after: float              # Diaphonie du mode orthogonal après calibration


def predistort_emitter_dispersion(u_target: np.ndarray, cfg: HardwareConfig) -> np.ndarray:
    """
    Applique la pré-distorsion de phase conjuguée au niveau de l'AWG émetteur
    pour annuler exactement la dispersion de vitesse de groupe (beta_2 * L)
    accumulée sur le lien de 30 mètres.
    """
    dt_sec = cfg.dt * 1e-9
    freqs_hz = np.fft.fftfreq(len(u_target), d=dt_sec)
    delta_omega = 2.0 * np.pi * freqs_hz

    u_omega = np.fft.fft(u_target)
    inverse_dispersion = np.exp(+0.5j * cfg.beta_2 * cfg.link_length_m * (delta_omega**2))
    u_predistorted = np.fft.ifft(u_omega * inverse_dispersion)

    # Re-normalisation pour conserver l'énergie unitaire de l'enveloppe d'émission
    norm = np.sqrt(np.trapezoid(np.abs(u_predistorted) ** 2, dx=cfg.dt))
    return u_predistorted / norm


def calibrate_receiver_in_situ(
    u_arriving: np.ndarray,
    u_orthogonal: np.ndarray,
    cfg: HardwareConfig,
    n_iterations: int = 12,
    lr_phase: float = 0.85,
    lr_amp: float = 0.35,
) -> CalibrationReport:
    """
    Optimise le couplage complexe g_rec(t) du récepteur en boucle fermée
    par minimisation du champ hétérodyne réfléchi <L_tot(t)> sous superposition (|0>+|1>)/sqrt(2).
    """
    # 1. Estimation analytique initiale (sans connaissance du Stark shift dynamique)
    _, g_rec = compute_impedance_matched_coupling(
        u_arriving, cfg, apply_hardware_limits=True
    )

    # Mesures de référence avant calibration (avec AC Stark shift actif)
    res_init_match = simulate_cascaded_scattering(
        u_arriving, g_rec, cfg, initial_photon_state="single", include_stark_shift=True
    )
    res_init_ortho = simulate_cascaded_scattering(
        u_orthogonal, g_rec, cfg, initial_photon_state="single", include_stark_shift=True
    )

    reflection_history: list[float] = []

    # 2. Boucle de rétroaction hétérodyne IQ
    for _ in range(n_iterations):
        # Sondage avec état superposition (|0> + |1>)/sqrt(2) pour accéder à <L_tot(t)> = I(t) + iQ(t)
        res_probe = simulate_cascaded_scattering(
            u_arriving,
            g_rec,
            cfg,
            initial_photon_state="superposition",
            include_stark_shift=True,
        )
        reflection_history.append(res_probe.total_reflected)

        # Champ cohérent réfléchi : L_tot(t) = g_u(t) <a(t)> + g_rec(t) <b(t)>
        a_ref = res_probe.a_ref_mean
        mask_active = np.abs(g_rec) > 1e-4 * np.max(np.abs(g_rec))

        # Correction de phase : le déphasage de a_ref par rapport à g_rec révèle le Stark shift Delta(t)
        phase_err = np.zeros_like(cfg.time_grid)
        phase_err[mask_active] = np.imag(
            a_ref[mask_active] / (g_rec[mask_active] + cfg.eps_reg)
        )

        # Correction d'amplitude : la composante en phase indique une sous- ou sur-absorption
        amp_err = np.zeros_like(cfg.time_grid)
        amp_err[mask_active] = np.real(
            a_ref[mask_active] / (g_rec[mask_active] + cfg.eps_reg)
        )

        # Mise à jour de l'enveloppe complexe du coupleur
        amp_curr = np.abs(g_rec)
        phase_curr = np.angle(g_rec)

        amp_new = np.clip(
            amp_curr * (1.0 - lr_amp * amp_err),
            0.0,
            np.sqrt(cfg.gamma_max),
        )
        phase_new = phase_curr - lr_phase * phase_err
        g_rec = amp_new * np.exp(1j * phase_new)

    # 3. Évaluation finale après calibration
    res_final_match = simulate_cascaded_scattering(
        u_arriving, g_rec, cfg, initial_photon_state="single", include_stark_shift=True
    )
    res_final_ortho = simulate_cascaded_scattering(
        u_orthogonal, g_rec, cfg, initial_photon_state="single", include_stark_shift=True
    )

    return CalibrationReport(
        g_rec_calibrated=g_rec,
        reflection_history=reflection_history,
        capture_before=float(res_init_match.p_e[-1]),
        capture_after=float(res_final_match.p_e[-1]),
        crosstalk_before=float(res_init_ortho.p_e[-1]),
        crosstalk_after=float(res_final_ortho.p_e[-1]),
    )


if __name__ == "__main__":
    cfg = HardwareConfig()
    t = cfg.time_grid
    modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)

    # Pré-distorsion de dispersion à l'émission + propagation sur 30 m
    u0_pre = predistort_emitter_dispersion(modes[0], cfg)
    u1_pre = predistort_emitter_dispersion(modes[1], cfg)

    u0_arriving = propagate_dispersive_link(u0_pre, cfg, apply_loss=False)
    u1_arriving = propagate_dispersive_link(u1_pre, cfg, apply_loss=False)

    report = calibrate_receiver_in_situ(
        u0_arriving, u1_arriving, cfg, n_iterations=12
    )

    print(f"[INFO] Réflexion résiduelle (it. 0 -> it. 12) : {report.reflection_history[0]:.4f} -> {report.reflection_history[-1]:.4f}")
    print(f"[INFO] Capture mode cible HG0 : {report.capture_before*100:.2f}% -> {report.capture_after*100:.2f}%")
    print(f"[INFO] Diaphonie mode orthogonal HG1 : {report.crosstalk_before*100:.2f}% -> {report.crosstalk_after*100:.2f}%")
