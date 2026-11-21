"""
slh_solver.py
Solveur d'equation maitresse en cascade (SLH / Kiilerich-Molmer, PRL 123, 123604)
entre un paquet d'ondes itinerant (1 ou 2 photons) et un noeud recepteur transmon.

CORRECTION : le Hamiltonien de cascade a le signe de l'Eq. (5) de la reference,
    H_casc = (i/2) (g_u* g_rec a^dag b - g_u g_rec* b^dag a),
c'est-a-dire le produit en serie SLH (1/2i)(L2^dag L1 - L1^dag L2) avec L1 = g_u a
en amont. L'ancien signe inversait le sens de la cascade : capture nulle (voir tests).
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    propagate_dispersive_link,
    compute_impedance_matched_coupling,
)


@dataclass
class SimulationResult:
    t: np.ndarray
    p_g: np.ndarray                 # Population de |g> du recepteur
    p_e: np.ndarray                 # Population de |e> (capture 1 photon)
    p_f: np.ndarray                 # Population de |f> (fuite transmon)
    n_cav: np.ndarray               # Population residuelle dans la cavite virtuelle incidente
    a_ref_mean: np.ndarray          # Amplitude coherente sortante <L_tot(t)> (tomographie IQ)
    reflected_flux: np.ndarray      # Flux sortant <L_tot^dag L_tot(t)>
    total_reflected: float          # Integrale du flux sortant
    trace_drift: float              # |Tr(rho) - 1| max (ne detecte PAS une erreur de signe)


def compute_emitter_virtual_coupling(u_in: np.ndarray, cfg: HardwareConfig) -> np.ndarray:
    """
    g_u(t) = u*(t) / sqrt(1 - int_{-inf}^t |u|^2)   (Eq. 2 de PRL 123, 123604)
    """
    dt = cfg.dt
    power = np.abs(u_in) ** 2
    total_norm = np.trapezoid(power, dx=dt)
    u_norm = u_in / np.sqrt(total_norm + cfg.eps_reg)
    power_norm = np.abs(u_norm) ** 2

    cum_emitted = cumulative_trapezoid(power_norm, dx=dt, initial=0.0)
    remaining = np.maximum(1.0 - cum_emitted, 0.0)

    gamma_u = np.where(
        remaining > cfg.emit_remaining_min, power_norm / (remaining + cfg.eps_reg), 0.0
    )
    phase_u = np.angle(u_norm)
    return np.sqrt(gamma_u) * np.exp(-1j * phase_u)  # u*/sqrt(...)


def build_operators(dim_cav: int = 3, dim_q: int = 3) -> dict[str, np.ndarray]:
    a_c = np.diag(np.sqrt(np.arange(1, dim_cav)), k=1).astype(np.complex128)
    adag_c = a_c.conj().T
    I_c = np.eye(dim_cav, dtype=np.complex128)

    b_q = np.diag(np.sqrt(np.arange(1, dim_q)), k=1).astype(np.complex128)
    bdag_q = b_q.conj().T
    I_q = np.eye(dim_q, dtype=np.complex128)

    proj_g = np.zeros((dim_q, dim_q), dtype=np.complex128)
    proj_g[0, 0] = 1.0
    proj_e = np.zeros((dim_q, dim_q), dtype=np.complex128)
    proj_e[1, 1] = 1.0
    proj_f = np.zeros((dim_q, dim_q), dtype=np.complex128)
    if dim_q > 2:
        proj_f[2, 2] = 1.0

    n_q_op = bdag_q @ b_q
    anh_op = 0.5 * (bdag_q @ bdag_q @ b_q @ b_q)  # (1/2) b^dag b^dag b b

    return {
        "a": np.kron(a_c, I_q),
        "adag": np.kron(adag_c, I_q),
        "n_cav": np.kron(adag_c @ a_c, I_q),
        "b": np.kron(I_c, b_q),
        "bdag": np.kron(I_c, bdag_q),
        "n_q": np.kron(I_c, n_q_op),
        "anh": np.kron(I_c, anh_op),
        "P_g": np.kron(I_c, proj_g),
        "P_e": np.kron(I_c, proj_e),
        "P_f": np.kron(I_c, proj_f),
    }


def simulate_cascaded_scattering(
    u_in: np.ndarray,
    g_rec_t: np.ndarray,
    cfg: HardwareConfig,
    initial_photon_state: str = "single",
    dim_cav: int = 3,
    dim_q: int = 3,
    include_stark_shift: bool = True,
    detuning_extra: float = 0.0,
) -> SimulationResult:
    """
    d(rho)/dt = -i [H(t), rho] + D[L_tot(t)](rho) avec
        H(t)   = delta(t) n_q + alpha_q/2 b^dag b^dag b b
                 + (i/2) (g_u* g_rec a^dag b - g_u g_rec* b^dag a)
        L_tot  = g_u a + g_rec b
    """
    t_grid = cfg.time_grid
    g_u_t = compute_emitter_virtual_coupling(u_in, cfg)

    gu_interp = interp1d(t_grid, g_u_t, kind="linear", fill_value=0.0, bounds_error=False)
    grec_interp = interp1d(t_grid, g_rec_t, kind="linear", fill_value=0.0, bounds_error=False)

    ops = build_operators(dim_cav=dim_cav, dim_q=dim_q)
    a, adag = ops["a"], ops["adag"]
    b, bdag = ops["b"], ops["bdag"]
    n_q, anh = ops["n_q"], ops["anh"]

    psi_cav = np.zeros(dim_cav, dtype=np.complex128)
    if initial_photon_state == "single":
        psi_cav[1] = 1.0
    elif initial_photon_state == "two_photon":
        if dim_cav < 3:
            raise ValueError("dim_cav doit être >= 3 pour l'état de Fock |2>.")
        psi_cav[2] = 1.0
    elif initial_photon_state == "superposition":
        psi_cav[0] = 1.0 / np.sqrt(2.0)
        psi_cav[1] = 1.0 / np.sqrt(2.0)
    else:
        raise ValueError(f"État initial inconnu : {initial_photon_state}")

    psi_q = np.zeros(dim_q, dtype=np.complex128)
    psi_q[0] = 1.0
    psi_0 = np.kron(psi_cav, psi_q)
    rho_0 = np.outer(psi_0, psi_0.conj())
    dim_tot = dim_cav * dim_q

    def lindblad_rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
        rho = rho_vec.reshape((dim_tot, dim_tot))
        gu = complex(gu_interp(t))
        grec = complex(grec_interp(t))

        gamma_inst = abs(grec) ** 2
        delta_t = detuning_extra + (cfg.stark_coeff * gamma_inst if include_stark_shift else 0.0)

        H_node = delta_t * n_q + cfg.alpha_q * anh
        # Signe corrige : (i/2)(g_u* g_rec a^dag b - g_u g_rec* b^dag a)
        H_casc = 0.5j * (np.conj(gu) * grec * (adag @ b) - gu * np.conj(grec) * (a @ bdag))
        H_tot = H_node + H_casc

        L_tot = gu * a + grec * b
        Ldag_L = L_tot.conj().T @ L_tot

        comm = -1j * (H_tot @ rho - rho @ H_tot)
        diss = L_tot @ rho @ L_tot.conj().T - 0.5 * (Ldag_L @ rho + rho @ Ldag_L)
        return (comm + diss).reshape(-1)

    sol = solve_ivp(
        lindblad_rhs,
        t_span=(t_grid[0], t_grid[-1]),
        y0=rho_0.reshape(-1),
        t_eval=t_grid,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"Échec de l'intégration de l'équation maîtresse : {sol.message}")

    n_pts = len(t_grid)
    p_g = np.zeros(n_pts)
    p_e = np.zeros(n_pts)
    p_f = np.zeros(n_pts)
    n_cav = np.zeros(n_pts)
    a_ref_mean = np.zeros(n_pts, dtype=np.complex128)
    reflected_flux = np.zeros(n_pts)
    max_trace_err = 0.0

    for idx in range(n_pts):
        rho_t = sol.y[:, idx].reshape((dim_tot, dim_tot))
        tr = np.real(np.trace(rho_t))
        max_trace_err = max(max_trace_err, abs(tr - 1.0))

        p_g[idx] = np.real(np.trace(ops["P_g"] @ rho_t))
        p_e[idx] = np.real(np.trace(ops["P_e"] @ rho_t))
        p_f[idx] = np.real(np.trace(ops["P_f"] @ rho_t))
        n_cav[idx] = np.real(np.trace(ops["n_cav"] @ rho_t))

        L_t = g_u_t[idx] * a + g_rec_t[idx] * b
        a_ref_mean[idx] = np.trace(L_t @ rho_t)
        reflected_flux[idx] = np.real(np.trace(L_t.conj().T @ L_t @ rho_t))

    total_reflected = float(np.trapezoid(reflected_flux, dx=cfg.dt))

    return SimulationResult(
        t=t_grid, p_g=p_g, p_e=p_e, p_f=p_f, n_cav=n_cav,
        a_ref_mean=a_ref_mean, reflected_flux=reflected_flux,
        total_reflected=total_reflected, trace_drift=max_trace_err,
    )


if __name__ == "__main__":
    cfg = HardwareConfig()
    t = cfg.time_grid
    modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)

    _, g_rec_0 = compute_impedance_matched_coupling(modes[0], cfg, apply_hardware_limits=True)

    res_match = simulate_cascaded_scattering(
        modes[0], g_rec_0, cfg, initial_photon_state="single", include_stark_shift=False
    )
    res_ortho = simulate_cascaded_scattering(
        modes[1], g_rec_0, cfg, initial_photon_state="single", include_stark_shift=False
    )
    res_two_ph = simulate_cascaded_scattering(
        modes[0], g_rec_0, cfg, initial_photon_state="two_photon", include_stark_shift=False
    )

    print(f"[INFO] Dérive de trace Tr(rho) : {res_match.trace_drift:.2e} (ne valide pas la physique)")
    print(f"[INFO] Mode cible HG0 -> Capture P_e(fin) = {res_match.p_e[-1]*100:.2f}%, Sortie = {res_match.total_reflected*100:.2f}%")
    print(f"[INFO] Mode orthogonal HG1 -> P_e(fin) = {res_ortho.p_e[-1]*100:.2f}%, Sortie = {res_ortho.total_reflected*100:.2f}%")
    print(f"[INFO] |2_HG0> -> P_e = {res_two_ph.p_e[-1]*100:.2f}%, P_f = {res_two_ph.p_f[-1]*100:.2f}%, Photons sortants = {res_two_ph.total_reflected:.3f}")
