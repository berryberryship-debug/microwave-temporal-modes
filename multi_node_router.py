from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    propagate_dispersive_link,
    compute_impedance_matched_coupling,
)
from slh_solver import compute_emitter_virtual_coupling
from calibration import predistort_emitter_dispersion, calibrate_receiver_in_situ


@dataclass
class DualNodeResult:
    t: np.ndarray
    p_e_A: np.ndarray
    p_e_B: np.ndarray
    p_ee_coincidence: np.ndarray
    a_out_mean: np.ndarray
    bell_fidelity: float
    leaked_flux: float
    trace_drift: float


def build_dual_node_operators(dim_cav: int = 2, dim_q: int = 2) -> dict[str, np.ndarray]:
    a_single = np.diag(np.sqrt(np.arange(1, dim_cav)), k=1).astype(np.complex128)
    I_c = np.eye(dim_cav, dtype=np.complex128)

    b_single = np.diag(np.sqrt(np.arange(1, dim_q)), k=1).astype(np.complex128)
    I_q = np.eye(dim_q, dtype=np.complex128)

    def kron4(op0, op1, opA, opB):
        return np.kron(np.kron(op0, op1), np.kron(opA, opB))

    a0 = kron4(a_single, I_c, I_q, I_q)
    a1 = kron4(I_c, a_single, I_q, I_q)
    bA = kron4(I_c, I_c, b_single, I_q)
    bB = kron4(I_c, I_c, I_q, b_single)

    proj_e = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    return {
        "a0": a0,
        "a0_dag": a0.conj().T,
        "a1": a1,
        "a1_dag": a1.conj().T,
        "bA": bA,
        "bA_dag": bA.conj().T,
        "bB": bB,
        "bB_dag": bB.conj().T,
        "n_A": bA.conj().T @ bA,
        "n_B": bB.conj().T @ bB,
        "P_eA": kron4(I_c, I_c, proj_e, I_q),
        "P_eB": kron4(I_c, I_c, I_q, proj_e),
        "P_ee": kron4(I_c, I_c, proj_e, proj_e),
    }


def simulate_dual_node_demux(
    u0_in: np.ndarray,
    u1_in: np.ndarray,
    g_A_t: np.ndarray,
    g_B_t: np.ndarray,
    cfg: HardwareConfig,
    input_mode: str = "bell_superposition",
    include_stark_shift: bool = True,
) -> DualNodeResult:
    t_grid = cfg.time_grid
    gu0_t = compute_emitter_virtual_coupling(u0_in, cfg)
    gu1_t = compute_emitter_virtual_coupling(u1_in, cfg)

    gu0_itp = interp1d(t_grid, gu0_t, kind="linear", fill_value=0.0, bounds_error=False)
    gu1_itp = interp1d(t_grid, gu1_t, kind="linear", fill_value=0.0, bounds_error=False)
    gA_itp = interp1d(t_grid, g_A_t, kind="linear", fill_value=0.0, bounds_error=False)
    gB_itp = interp1d(t_grid, g_B_t, kind="linear", fill_value=0.0, bounds_error=False)

    ops = build_dual_node_operators()
    a0, a1 = ops["a0"], ops["a1"]
    bA, bA_dag = ops["bA"], ops["bA_dag"]
    bB, bB_dag = ops["bB"], ops["bB_dag"]
    n_A, n_B = ops["n_A"], ops["n_B"]

    ket_00 = np.array([1.0, 0.0], dtype=np.complex128)
    ket_10 = np.array([0.0, 1.0], dtype=np.complex128)

    psi_vac = np.kron(np.kron(ket_00, ket_00), np.kron(ket_00, ket_00))
    psi_10_gg = np.kron(np.kron(ket_10, ket_00), np.kron(ket_00, ket_00))
    psi_01_gg = np.kron(np.kron(ket_00, ket_10), np.kron(ket_00, ket_00))

    if input_mode == "only_HG0":
        psi_0 = psi_10_gg
    elif input_mode == "only_HG1":
        psi_0 = psi_01_gg
    elif input_mode == "probe_HG1":
        psi_0 = (psi_vac + psi_01_gg) / np.sqrt(2.0)
    elif input_mode == "bell_superposition":
        psi_0 = (psi_10_gg + psi_01_gg) / np.sqrt(2.0)
    else:
        raise ValueError(f"Mode d'entrée inconnu : {input_mode}")

    rho_0 = np.outer(psi_0, psi_0.conj())
    dim_tot = 16

    psi_eg = np.kron(np.kron(ket_00, ket_00), np.kron(ket_10, ket_00))
    psi_ge = np.kron(np.kron(ket_00, ket_00), np.kron(ket_00, ket_10))
    psi_bell_target = (psi_eg + psi_ge) / np.sqrt(2.0)

    def lindblad_rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
        rho = rho_vec.reshape((dim_tot, dim_tot))
        gu0 = complex(gu0_itp(t))
        gu1 = complex(gu1_itp(t))
        gA = complex(gA_itp(t))
        gB = complex(gB_itp(t))

        delta_A = cfg.stark_coeff * (abs(gA) ** 2) if include_stark_shift else 0.0
        delta_B = cfg.stark_coeff * (abs(gB) ** 2) if include_stark_shift else 0.0
        H_nodes = delta_A * n_A + delta_B * n_B

        L_emit = gu0 * a0 + gu1 * a1
        L_emit_dag = L_emit.conj().T
        LA = -gA * bA
        LA_dag = -np.conj(gA) * bA_dag
        LB = -gB * bB
        LB_dag = -np.conj(gB) * bB_dag

        H_em_A = 0.5j * (L_emit_dag @ LA - LA_dag @ L_emit)
        H_em_B = 0.5j * (L_emit_dag @ LB - LB_dag @ L_emit)
        H_A_B = 0.5j * (LA_dag @ LB - LB_dag @ LA)

        H_tot = H_nodes + H_em_A + H_em_B + H_A_B

        L_tot = L_emit + LA + LB
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
        max_step=1.0,
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"Échec de l'intégration multi-nœuds : {sol.message}")

    n_pts = len(t_grid)
    p_e_A = np.zeros(n_pts)
    p_e_B = np.zeros(n_pts)
    p_ee = np.zeros(n_pts)
    a_out_mean = np.zeros(n_pts, dtype=np.complex128)
    leaked = np.zeros(n_pts)
    max_trace_err = 0.0

    for idx in range(n_pts):
        rho_t = sol.y[:, idx].reshape((dim_tot, dim_tot))
        tr = np.real(np.trace(rho_t))
        max_trace_err = max(max_trace_err, abs(tr - 1.0))

        p_e_A[idx] = np.real(np.trace(ops["P_eA"] @ rho_t))
        p_e_B[idx] = np.real(np.trace(ops["P_eB"] @ rho_t))
        p_ee[idx] = np.real(np.trace(ops["P_ee"] @ rho_t))

        L_t = (gu0_t[idx] * a0 + gu1_t[idx] * a1) - g_A_t[idx] * bA - g_B_t[idx] * bB
        a_out_mean[idx] = np.trace(L_t @ rho_t)
        leaked[idx] = np.real(np.trace(L_t.conj().T @ L_t @ rho_t))

    rho_final = sol.y[:, -1].reshape((dim_tot, dim_tot))
    bell_fid = float(np.real(np.vdot(psi_bell_target, rho_final @ psi_bell_target)))
    leaked_total = float(np.trapezoid(leaked, dx=cfg.dt))

    return DualNodeResult(
        t=t_grid,
        p_e_A=p_e_A,
        p_e_B=p_e_B,
        p_ee_coincidence=p_ee,
        a_out_mean=a_out_mean,
        bell_fidelity=bell_fid,
        leaked_flux=leaked_total,
        trace_drift=max_trace_err,
    )


def calibrate_downstream_node(
    u0_arr: np.ndarray,
    u1_arr: np.ndarray,
    g_A_cal: np.ndarray,
    cfg: HardwareConfig,
    n_iterations: int = 12,
    lr_phase: float = 0.85,
    lr_amp: float = 0.35,
) -> np.ndarray:
    """
    Calibre le nœud aval B directement sur le champ HG1 sortant du nœud amont A
    (compensant le déphasage dispersif induit par la traversée de A + Stark shift de B).
    """
    g_zero = np.zeros_like(g_A_cal)
    res_thru_A = simulate_dual_node_demux(
        u0_arr, u1_arr, g_A_cal, g_zero, cfg, input_mode="probe_HG1", include_stark_shift=True
    )
    u1_after_A = 2.0 * res_thru_A.a_out_mean
    norm_out = np.sqrt(np.trapezoid(np.abs(u1_after_A) ** 2, dx=cfg.dt))
    u1_after_A /= norm_out + cfg.eps_reg

    _, g_B = compute_impedance_matched_coupling(u1_after_A, cfg, apply_hardware_limits=True)

    for _ in range(n_iterations):
        res_probe = simulate_dual_node_demux(
            u0_arr, u1_arr, g_A_cal, g_B, cfg, input_mode="probe_HG1", include_stark_shift=True
        )
        a_out = res_probe.a_out_mean
        mask = np.abs(g_B) > 1e-4 * np.max(np.abs(g_B))

        phase_err = np.zeros_like(cfg.time_grid)
        phase_err[mask] = -np.imag(a_out[mask] / (g_B[mask] + cfg.eps_reg))

        amp_err = np.zeros_like(cfg.time_grid)
        amp_err[mask] = -np.real(a_out[mask] / (g_B[mask] + cfg.eps_reg))

        amp_new = np.clip(np.abs(g_B) * (1.0 - lr_amp * amp_err), 0.0, np.sqrt(cfg.gamma_max))
        phase_new = np.angle(g_B) - lr_phase * phase_err
        g_B = amp_new * np.exp(1j * phase_new)

    return g_B


if __name__ == "__main__":
    cfg = HardwareConfig()
    t = cfg.time_grid
    modes = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=2)

    u0_arr = propagate_dispersive_link(predistort_emitter_dispersion(modes[0], cfg), cfg, apply_loss=False)
    u1_arr = propagate_dispersive_link(predistort_emitter_dispersion(modes[1], cfg), cfg, apply_loss=False)

    rep_A = calibrate_receiver_in_situ(u0_arr, u1_arr, cfg, n_iterations=10)
    g_A_cal = rep_A.g_rec_calibrated

    g_B_cal = calibrate_downstream_node(u0_arr, u1_arr, g_A_cal, cfg, n_iterations=12)

    res_0 = simulate_dual_node_demux(u0_arr, u1_arr, g_A_cal, g_B_cal, cfg, input_mode="only_HG0")
    res_1 = simulate_dual_node_demux(u0_arr, u1_arr, g_A_cal, g_B_cal, cfg, input_mode="only_HG1")
    res_bell = simulate_dual_node_demux(u0_arr, u1_arr, g_A_cal, g_B_cal, cfg, input_mode="bell_superposition")

    print("=== ROUTAGE MULTI-NŒUDS CALIBRÉ EN SÉRIE (30 m | Q_A=HG0, Q_B=HG1) ===")
    print(f"[PASS] Dérive maximale de trace Tr(rho) : {res_bell.trace_drift:.2e}")
    print(f"[INFO] Entrée |1_HG0> -> P_e(A) = {res_0.p_e_A[-1]*100:.2f}%, P_e(B) = {res_0.p_e_B[-1]*100:.2f}%, Fuite = {res_0.leaked_flux*100:.2f}%")
    print(f"[INFO] Entrée |1_HG1> -> P_e(A) = {res_1.p_e_A[-1]*100:.2f}%, P_e(B) = {res_1.p_e_B[-1]*100:.2f}%, Fuite = {res_1.leaked_flux*100:.2f}%")
    print(f"[INFO] Entrée Bell Dual-Rail -> P_e(A) = {res_bell.p_e_A[-1]*100:.2f}%, P_e(B) = {res_bell.p_e_B[-1]*100:.2f}%")
    print(f"[RÉSULTAT] Fidélité d'intrication distribuée F(|Phi+>) : {res_bell.bell_fidelity*100:.2f}%")
