import numpy as np
from scipy.special import hermite
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from config import HardwareConfig


def generate_hermite_gauss_basis(t: np.ndarray, tau: float, n_modes: int) -> np.ndarray:
    dt = t[1] - t[0]
    modes = np.zeros((n_modes, len(t)), dtype=np.complex128)
    x = t / tau
    gauss = np.exp(-0.5 * x**2)

    for n in range(n_modes):
        Hn = hermite(n)(x)
        raw = Hn * gauss
        norm = np.sqrt(np.trapezoid(np.abs(raw) ** 2, dx=dt))
        modes[n] = raw / norm

    return modes


def generate_gram_schmidt_exponential_basis(
    t: np.ndarray, tau: float, n_modes: int, rising: bool = True
) -> np.ndarray:
    dt = t[1] - t[0]
    raw_candidates = np.zeros((n_modes, len(t)), dtype=np.complex128)

    for k in range(n_modes):
        rate = (1.0 + 0.6 * k) / tau
        if rising:
            env = np.where(t <= 0, np.exp(0.5 * rate * t), 0.0)
        else:
            env = np.where(t >= 0, np.exp(-0.5 * rate * t), 0.0)
        raw_candidates[k] = env

    basis = np.zeros_like(raw_candidates)
    for n in range(n_modes):
        v = raw_candidates[n].copy()
        for k in range(n):
            proj = np.trapezoid(np.conj(basis[k]) * v, dx=dt)
            v -= proj * basis[k]
        norm = np.sqrt(np.trapezoid(np.abs(v) ** 2, dx=dt))
        if norm < 1e-14:
            raise ValueError(f"Linear dependence encountered at mode n={n}.")
        basis[n] = v / norm

    return basis


def propagate_dispersive_link(
    u_t: np.ndarray, cfg: HardwareConfig, apply_loss: bool = True
) -> np.ndarray:
    dt_sec = cfg.dt * 1e-9
    freqs_hz = np.fft.fftfreq(len(u_t), d=dt_sec)
    delta_omega = 2.0 * np.pi * freqs_hz

    u_omega = np.fft.fft(u_t)
    dispersion_phase = np.exp(-0.5j * cfg.beta_2 * cfg.link_length_m * (delta_omega**2))
    u_prop = np.fft.ifft(u_omega * dispersion_phase)

    if apply_loss:
        u_prop *= np.sqrt(cfg.link_transmission)

    return u_prop


def compute_impedance_matched_coupling(
    u_target: np.ndarray, cfg: HardwareConfig, apply_hardware_limits: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    dt = cfg.dt
    power = np.abs(u_target) ** 2
    cum_energy = cumulative_trapezoid(power, dx=dt, initial=0.0)

    gamma_ideal = power / (cum_energy + cfg.eps_reg)

    total_energy = np.trapezoid(power, dx=dt)
    mask_active = (cum_energy > 1e-5 * total_energy) & (cum_energy < 0.9995 * total_energy)
    gamma_ideal = np.where(mask_active, gamma_ideal, 0.0)

    phase_target = np.angle(u_target)
    g_complex = np.sqrt(gamma_ideal) * np.exp(1j * phase_target)

    if apply_hardware_limits:
        sigma_steps = max(0.5, (1.0 / cfg.kappa_buffer) / dt)
        g_real = gaussian_filter1d(np.real(g_complex), sigma=sigma_steps)
        g_imag = gaussian_filter1d(np.imag(g_complex), sigma=sigma_steps)
        g_complex = g_real + 1j * g_imag

        gamma_limited = np.abs(g_complex) ** 2
        over_limit = gamma_limited > cfg.gamma_max
        ratio = np.ones_like(gamma_limited)
        np.divide(cfg.gamma_max, gamma_limited, out=ratio, where=over_limit)
        g_complex *= np.sqrt(ratio)

    gamma_final = np.abs(g_complex) ** 2
    return gamma_final, g_complex


def verify_orthonormality(modes: np.ndarray, dt: float, tol: float = 1e-6) -> np.ndarray:
    n_modes = modes.shape[0]
    gram = np.zeros((n_modes, n_modes), dtype=np.complex128)
    for m in range(n_modes):
        for n in range(n_modes):
            gram[m, n] = np.trapezoid(np.conj(modes[m]) * modes[n], dx=dt)

    error = np.max(np.abs(gram - np.eye(n_modes)))
    if error > tol:
        raise AssertionError(f"Orthonormality check failed: max deviation {error:.2e} > {tol:.2e}")
    return gram


if __name__ == "__main__":
    cfg = HardwareConfig()
    t = cfg.time_grid

    hg_modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=4)
    gram_hg = verify_orthonormality(hg_modes, cfg.dt)

    gs_modes = generate_gram_schmidt_exponential_basis(t, cfg.tau_pulse, n_modes=4)
    gram_gs = verify_orthonormality(gs_modes, cfg.dt)

    u0_prop = propagate_dispersive_link(hg_modes[0], cfg, apply_loss=False)
    u1_prop = propagate_dispersive_link(hg_modes[1], cfg, apply_loss=False)
    crosstalk_01 = np.abs(np.trapezoid(np.conj(hg_modes[0]) * u1_prop, dx=cfg.dt)) ** 2

    gamma_0, _ = compute_impedance_matched_coupling(hg_modes[0], cfg, apply_hardware_limits=True)

    print(f"[PASS] Hermite-Gauss Gram max error: {np.max(np.abs(gram_hg - np.eye(4))):.2e}")
    print(f"[PASS] Gram-Schmidt Exp Gram max error: {np.max(np.abs(gram_gs - np.eye(4))):.2e}")
    print(f"[INFO] 30m Link Power Transmission: {cfg.link_transmission * 100:.2f}%")
    print(f"[INFO] Peak Coupler Rate for HG0: {np.max(gamma_0) / (2 * np.pi) * 1e3:.2f} MHz")
    print(f"[INFO] Dispersion-induced |<u0 | u1_prop>|^2 crosstalk: {crosstalk_01:.2e}")
