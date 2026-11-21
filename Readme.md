# TemporalMode-QNet

**Moteur de simulation de systèmes quantiques ouverts pour l'émission, la propagation et l'absorption sélective de photons micro-ondes en modes temporels orthogonaux.**

Ce projet modélise les dynamiques d'électrodynamique quantique en circuit (cQED) sur un lien cryogénique de 30 mètres, inspiré des travaux expérimentaux et théoriques sur le routage quantique par enveloppes temporelles orthogonales (*Phys. Rev. Lett.* **137**, 210801).

---

## Architecture Physique & Mathématique

Plutôt que de discrétiser spatialement un guide d'ondes de 30 mètres en milliers de points, **TemporalMode-QNet** utilise le formalisme **Entrée–Sortie (Input–Output) avec cavités virtuelles** (Kiilerich & Mølmer, *Phys. Rev. Lett.* **123**, 123604) combiné à la théorie des systèmes quantiques ouverts en cascade (**SLH — Scattering, Lindblad, Hamiltonian**) :

1. **Bases de modes temporels orthogonaux (`wavepackets.py`) :**
   * Génération de bases orthonormées $\langle u_m | u_n \rangle = \delta_{mn}$ : modes d'**Hermite–Gauss** et exponentielles orthogonalisées par **Gram–Schmidt**.
   * Synthèse du profil de couplage dynamique d'adaptation d'impédance :
     $$\gamma(t) = \frac{|u_{\text{cible}}(t)|^2}{\int_{-\infty}^{t} |u_{\text{cible}}(\tau)|^2 \, d\tau}$$
   * Modélisation des contraintes matérielles réelles : plafond de couplage $\gamma_{\max}$, filtrage passe-bas par la largeur de raie du résonateur tampon ($\kappa_{\text{buffer}}$), dispersion de vitesse de groupe ($\beta_2 L$) et atténuation sur 30 m.

2. **Solveur d'équation maîtresse en cascade (`slh_solver.py`) :**
   * Évolution exacte de la matrice densité $\rho(t)$ sous équation maîtresse de Lindblad.
   * Simulation de l'absorption déterministe d'un mode cible, de la réflexion élastique des modes orthogonaux, de la saturation non linéaire à deux photons (effet Hong–Ou–Mandel) et des décalages AC Stark dynamiques $\Delta(t)$.

3. **Calibration in situ & Pré-distorsion (`calibration.py`) :**
   * Tomographie hétérodyne simulée $\langle a_{\text{ref}}(t) \rangle$ à la frontière du nœud récepteur.
   * Boucle de pré-distorsion itérative pour annuler la réflexion résiduelle et compenser le chirp de phase induit par la dispersion du câble coaxial.

---

## Structure du Dépôt

| Fichier | Rôle |
| :--- | :--- |
| `config.py` | Constantes physiques, paramètres du lien cryogénique (30 m) et limites matérielles du coupleur accordable. |
| `wavepackets.py` | Construction des bases temporelles orthogonales, propagation dispersive de Fourier et synthèse de $\gamma(t)$. |
| `slh_solver.py` | Solveur quantique ouvert en cascade (cavités virtuelles + équation de Lindblad). |
| `calibration.py` | Tomographie hétérodyne IQ et optimisation itérative par pré-distorsion d'enveloppe. |
| `benchmarks.py` | Matrices de sélectivité/diaphonie $\eta_{mn}$, fidélité d'intrication *dual-rail* et visibilité HOM. |
| `run_experiment.py` | Point d'entrée CLI et suite de vérification analytique (conservation de la trace et orthonormalité). |

---

## Installation & Vérification

### Prérequis
Le moteur repose uniquement sur des bibliothèques scientifiques standards :
* Python $\ge 3.10$
* `numpy`, `scipy`, `matplotlib`

```bash
git clone [https://github.com/berryberryship-debug/TemporalMode-QNet.git](https://github.com/berryberryship-debug/TemporalMode-QNet.git)
cd TemporalMode-QNet
pip install numpy scipy matplotlib
