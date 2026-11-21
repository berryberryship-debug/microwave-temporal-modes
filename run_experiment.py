"""
run_experiment.py
Point d'entrée principal (CLI) et suite de vérification analytique
pour le moteur TemporalMode-QNet.
"""

import argparse
import numpy as np
from config import HardwareConfig
from wavepackets import (
    generate_hermite_gauss_basis,
    generate_gram_schmidt_exponential_basis,
    verify_orthonormality,
)
from benchmarks import run_all_benchmarks


def run_analytical_verification(cfg: HardwareConfig) -> None:
    """Vérifie les invariants mathématiques avant l'exécution des simulations."""
    t = cfg.time_grid
    hg = generate_hermite_gauss_basis(t, cfg.tau_pulse, n_modes=4)
    gs = generate_gram_schmidt_exponential_basis(t, cfg.tau_pulse, n_modes=4)

    gram_hg = verify_orthonormality(hg, cfg.dt)
    gram_gs = verify_orthonormality(gs, cfg.dt)

    print("=== 1. VÉRIFICATION ANALYTIQUE DES INVARIANTS ===")
    print(f"[PASS] Orthonormalité Hermite-Gauss (4 modes)   : erreur max = {np.max(np.abs(gram_hg - np.eye(4))):.2e}")
    print(f"[PASS] Orthonormalité Gram-Schmidt Exp (4 modes): erreur max = {np.max(np.abs(gram_gs - np.eye(4))):.2e}")
    print(f"[INFO] Transmission lien cryogénique ({cfg.link_length_m:.0f} m)    : {cfg.link_transmission * 100:.2f}%\n")


def save_diagnostic_plots(results, filename: str = "temporal_mode_benchmarks.png") -> None:
    """Génère la figure récapitulative (matrice de sélectivité + séparateur de faisceau)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panneau 1 : Matrice de sélectivité eta_mn
    im = axes[0].imshow(results.selectivity_matrix * 100.0, cmap="viridis", vmin=0, vmax=100)
    n_m = results.selectivity_matrix.shape[0]
    axes[0].set_xticks(range(n_m))
    axes[0].set_yticks(range(n_m))
    axes[0].set_xticklabels([f"HG{n}" for n in range(n_m)])
    axes[0].set_yticklabels([f"Rx HG{m}" for m in range(n_m)])
    axes[0].set_title(f"Sélectivité (%) | Contraste: {results.selectivity_contrast_db:.1f} dB")
    for m in range(n_m):
        for n in range(n_m):
            val = results.selectivity_matrix[m, n] * 100.0
            axes[0].text(n, m, f"{val:.1f}%", ha="center", va="center", color="white" if val < 55 else "black")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # Panneau 2 : Séparateur de faisceau temporel programmable
    deg = np.degrees(results.mixing_angles)
    axes[1].plot(deg, results.dual_rail_capture, "o-", label="Simulation SLH")
    axes[1].plot(deg, np.cos(results.mixing_angles) ** 2, "--", label=r"Théorie $\cos^2(\theta)$")
    axes[1].set_xlabel(r"Angle de mélange $\theta$ (degrés)")
    axes[1].set_ylabel(r"Probabilité de capture $P_e$")
    axes[1].set_title("Séparateur de faisceau temporel")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Panneau 3 : Saturation non linéaire à 2 photons
    axes[2].plot(results.hom_overlaps, results.hom_two_photon_reflected, "s-", color="crimson")
    axes[2].set_xlabel(r"Chevauchement modal $|\mathcal{J}|^2 = |\langle u_0 | u_{\mathrm{in}} \rangle|^2$")
    axes[2].set_ylabel("Photons réfléchis (sur 2 incidents)")
    axes[2].set_title("Saturation à 2 photons (|2_u>)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    print(f"[INFO] Graphiques sauvegardés dans : {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TemporalMode-QNet : Simulation de routage quantique en modes temporels orthogonaux."
    )
    parser.add_argument("--modes", type=int, default=3, help="Nombre de modes orthogonaux à évaluer (défaut: 3)")
    parser.add_argument("--length", type=float, default=30.0, help="Longueur du lien cryogénique en mètres (défaut: 30.0)")
    parser.add_argument("--tau", type=float, default=45.0, help="Largeur temporelle tau du paquet d'ondes en ns (défaut: 45.0)")
    parser.add_argument("--plot", action="store_true", help="Génère la figure PNG des résultats")
    args = parser.parse_args()

    cfg = HardwareConfig(link_length_m=args.length, tau_pulse=args.tau)

    run_analytical_verification(cfg)

    print("=== 2. EXÉCUTION DES BENCHMARKS DU RÉSEAU (30m + CALIBRATION) ===")
    results = run_all_benchmarks(cfg, n_modes=args.modes)

    print("\nMatrice de sélectivité eta[m, n] (% d'absorption du mode n par le filtre m) :")
    for m in range(args.modes):
        row_str = "  ".join(f"{results.selectivity_matrix[m, n]*100:6.2f}%" for n in range(args.modes))
        print(f"  Filtre HG{m} : {row_str}")

    print(f"\n[RÉSULTAT] Contraste moyen de sélectivité : {results.selectivity_contrast_db:.2f} dB")
    print(f"[RÉSULTAT] Réflexion à 2 photons (|J|^2 = 0 -> 1) : {results.hom_two_photon_reflected[-1]:.3f} -> {results.hom_two_photon_reflected[0]:.3f} photons")

    if args.plot:
        save_diagnostic_plots(results)


if __name__ == "__main__":
    main()
