"""
run_experiment.py
Point d'entree CLI. Verifie l'orthonormalite des bases et la transmission du lien,
puis execute les benchmarks. (Ce module NE verifie PAS la conservation de la trace ;
voir tests/test_physics.py pour les tests physiques.)
"""

import argparse
import numpy as np
from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    generate_gram_schmidt_exponential_basis,
    generate_sech_polynomial_basis,
    verify_orthonormality,
)
from benchmarks import run_all_benchmarks, SECH_GAMMA_BW


def run_analytical_verification(cfg: HardwareConfig) -> None:
    """Verifie les invariants mathematiques avant l'execution des simulations."""
    t = cfg.time_grid
    hg = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=4)
    gs = generate_gram_schmidt_exponential_basis(t, cfg.tau_pulse, n_modes=4)
    sech = generate_sech_polynomial_basis(t, SECH_GAMMA_BW)

    gram_hg = verify_orthonormality(hg, cfg.dt)
    gram_gs = verify_orthonormality(gs, cfg.dt)           # tautologique : base construite par Gram-Schmidt
    gram_sech = verify_orthonormality(sech, cfg.dt, tol=1e-4)

    print("=== 1. VERIFICATION ANALYTIQUE DES INVARIANTS ===")
    print(f"[PASS] Orthonormalite Hermite-Gauss (4 modes)    : erreur max = {np.max(np.abs(gram_hg - np.eye(4))):.2e}")
    print(f"[PASS] Orthonormalite Gram-Schmidt Exp (4 modes) : erreur max = {np.max(np.abs(gram_gs - np.eye(4))):.2e}")
    print(f"[PASS] Orthonormalite famille sech (3 modes)     : erreur max = {np.max(np.abs(gram_sech - np.eye(3))):.2e}")
    print(f"[INFO] Transmission lien cryogenique ({cfg.link_length_m:.0f} m) : {cfg.link_transmission * 100:.2f}%\n")


def save_diagnostic_plots(results, filename: str = "temporal_mode_benchmarks.png") -> None:
    """Figure recapitulative (matrice de selectivite, separateur de faisceau, deux photons)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im = axes[0].imshow(results.selectivity_matrix * 100.0, cmap="viridis", vmin=0, vmax=100)
    n_m = results.selectivity_matrix.shape[0]
    axes[0].set_xticks(range(n_m))
    axes[0].set_yticks(range(n_m))
    axes[0].set_xticklabels([f"mode {n}" for n in range(n_m)])
    axes[0].set_yticklabels([f"Rx {m}" for m in range(n_m)])
    axes[0].set_title(f"Selectivite (%) | Contraste: {results.selectivity_contrast_db:.1f} dB")
    for m in range(n_m):
        for n in range(n_m):
            val = results.selectivity_matrix[m, n] * 100.0
            axes[0].text(n, m, f"{val:.1f}%", ha="center", va="center", color="white" if val < 55 else "black")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    deg = np.degrees(results.mixing_angles)
    axes[1].plot(deg, results.beam_splitter_capture, "o-", label="Simulation SLH")
    axes[1].plot(deg, np.cos(results.mixing_angles) ** 2, "--", label=r"Theorie $\cos^2(\theta)$")
    axes[1].set_xlabel(r"Angle de melange $\theta$ (degres)")
    axes[1].set_ylabel(r"Probabilite de capture $P_e$")
    axes[1].set_title("Balayage separateur de faisceau temporel")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(results.two_photon_overlaps, results.two_photon_outgoing, "s-", color="crimson")
    axes[2].set_xlabel(r"Recouvrement modal $|\mathcal{J}|^2$")
    axes[2].set_ylabel("Photons sortants (sur 2 incidents)")
    axes[2].set_title("Deux photons sur un recepteur non lineaire")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    print(f"[INFO] Graphiques sauvegardes dans : {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TemporalMode-QNet : simulation de routage quantique en modes temporels orthogonaux."
    )
    parser.add_argument("--modes", type=int, default=3, help="Nombre de modes orthogonaux (defaut: 3 ; sech : max 3)")
    parser.add_argument("--basis", choices=["hermite_gauss", "gram_schmidt", "sech"], default="hermite_gauss",
                        help="Base de modes (defaut: hermite_gauss ; sech = famille de arXiv:2604.12947)")
    parser.add_argument("--length", type=float, default=30.0, help="Longueur du lien en metres (defaut: 30.0)")
    parser.add_argument("--tau", type=float, default=45.0, help="Largeur temporelle tau en ns (defaut: 45.0)")
    parser.add_argument("--no-calibrate", action="store_true", help="Desactive la calibration du recepteur")
    parser.add_argument("--plot", action="store_true", help="Genere la figure PNG des resultats")
    args = parser.parse_args()

    cfg = HardwareConfig(link_length_m=args.length, tau_pulse=args.tau)

    run_analytical_verification(cfg)

    print("=== 2. EXECUTION DES BENCHMARKS (modele ideal, sans perte de photon) ===")
    results = run_all_benchmarks(
        cfg, n_modes=args.modes, basis_type=args.basis, calibrate=not args.no_calibrate
    )

    print("\nMatrice de selectivite eta[m, n] (% d'absorption du mode n par le filtre m) :")
    for m in range(args.modes):
        row_str = "  ".join(f"{results.selectivity_matrix[m, n]*100:6.2f}%" for n in range(args.modes))
        print(f"  Filtre {m} : {row_str}")

    print(f"\n[RESULTAT] Contraste de selectivite (diag / hors-diag) : {results.selectivity_contrast_db:.2f} dB")
    print("           (arXiv:2604.12947 rapporte ~40, soit ~16 dB, sur materiel reel avec pertes ;")
    print("            non comparable a ce modele ideal)")
    print(f"[RESULTAT] Photons sortants a 2 photons (|J|^2 = 0 -> 1) : "
          f"{results.two_photon_outgoing[-1]:.3f} -> {results.two_photon_outgoing[0]:.3f}")

    if args.plot:
        save_diagnostic_plots(results)


if __name__ == "__main__":
    main()
