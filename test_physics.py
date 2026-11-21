"""
Tests de non-regression physique. Lancer : python -m pytest tests/  (ou python tests/test_physics.py)
Ils auraient detecte l'inversion de signe du Hamiltonien de cascade, que le seul
test de trace ne detecte pas.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    generate_sech_polynomial_basis,
    compute_impedance_matched_coupling,
    verify_orthonormality,
)
from slh_solver import simulate_cascaded_scattering
from calibration import calibrate_receiver_in_situ

cfg = HardwareConfig()
t = cfg.time_grid
HG = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)
_, G0 = compute_impedance_matched_coupling(HG[0], cfg, apply_hardware_limits=True)


def test_orthonormality():
    verify_orthonormality(HG, cfg.dt)
    verify_orthonormality(generate_sech_polynomial_basis(t, 2 * np.pi * 0.014), cfg.dt, tol=1e-4)


def test_matched_mode_is_captured():
    r = simulate_cascaded_scattering(HG[0], G0, cfg, include_stark_shift=False)
    assert r.p_e[-1] > 0.999, r.p_e[-1]
    assert r.total_reflected < 1e-3


def test_orthogonal_mode_is_rejected():
    r = simulate_cascaded_scattering(HG[1], G0, cfg, include_stark_shift=False)
    assert r.p_e[-1] < 1e-3
    assert r.total_reflected > 0.999


def test_two_photon_excitation_balance():
    r = simulate_cascaded_scattering(HG[0], G0, cfg, initial_photon_state="two_photon",
                                     include_stark_shift=False)
    total = r.p_e[-1] + 2 * r.p_f[-1] + r.n_cav[-1] + r.total_reflected
    assert abs(total - 2.0) < 1e-2, total


def test_trace_is_preserved():
    r = simulate_cascaded_scattering(HG[0], G0, cfg, include_stark_shift=False)
    assert r.trace_drift < 1e-9  # necessaire mais NON suffisant


def test_calibration_recovers_stark_phase():
    rep = calibrate_receiver_in_situ(HG[0], HG[1], cfg)
    assert abs(rep.stark_phase_coeff - cfg.stark_coeff) < 1e-2
    assert rep.capture_after > 0.999
    assert rep.capture_after > rep.capture_before


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("[PASS]", name)
