# TemporalMode-QNet

**Simulateur de systèmes quantiques ouverts pour l'émission, la propagation et l'absorption sélective de photons micro-ondes en modes temporels orthogonaux.**

> **Statut : projet de recherche, non validé.** Ce code n'a pas encore été comparé quantitativement aux données expérimentales publiées. Les éléments marqués **[À VÉRIFIER]** sont des points que l'auteur doit confirmer avant publication.

Ce projet modélise la dynamique d'électrodynamique quantique en circuit (cQED) sur un lien cryogénique de 30 m, en s'inspirant de l'expérience décrite dans la référence [1].

---

## Références

1. A. Hernández-Antón *et al.*, « Emission and Absorption of Microwave Photons in Orthogonal Temporal Modes across a 30-Meter Two-Node Network », arXiv:2604.12947 (14 avril 2026). **Preprint, non certifié par évaluation par les pairs.** https://arxiv.org/abs/2604.12947
   *[À VÉRIFIER : l'ancienne référence « Phys. Rev. Lett. 137, 210801 » n'a pu être confirmée ; ne la rétablir que si la source est documentée.]*
2. A. H. Kiilerich et K. Mølmer, « Input-Output Theory with Quantum Pulses », *Phys. Rev. Lett.* **123**, 123604 (2019). arXiv:1902.09833. https://doi.org/10.1103/PhysRevLett.123.123604
3. A. H. Kiilerich et K. Mølmer, « Quantum interactions with pulses of radiation », *Phys. Rev. A* (2020), arXiv:2003.04573. *[À VÉRIFIER : référence de revue exacte.]*
4. G. F. Peñas *et al.*, arXiv:2403.12222 (construction des modes temporels orthogonaux, citée par [1]).
5. K. Sunada *et al.*, « Temporal-Mode Engineering for Multiplexed Microwave Photons and Mode-Selective Quantum State Transfer », arXiv:2603.10506 (11 mars 2026). Preprint ; travail parallèle sur quatre modes.

---

## Approche physique et mathématique

Plutôt que de discrétiser spatialement un guide d'ondes de 30 m, TemporalMode-QNet utilise le formalisme entrée–sortie avec **cavités virtuelles** [2, 3], combiné à la description en cascade des systèmes quantiques ouverts (formalisme SLH : Scattering, Lindblad, Hamiltonian).

**Hypothèses du formalisme de référence** [2] : couplage markovien du système aux champs d'entrée et de sortie, et propagation asymptotique non dispersive des impulsions. Toute extension au-delà de ces hypothèses (voir « Limites ») est une contribution de ce dépôt et doit être validée séparément.

### 1. Bases de modes temporels orthogonaux (`wavepackets.py`)

- Génération de bases orthonormées : ⟨uₘ | uₙ⟩ = δₘₙ.
- Dans [1], les modes d'ordre supérieur sont obtenus par orthogonalité polynomiale sous la fonction de poids sech²(Γt/2), via Gram–Schmidt. *[À VÉRIFIER : le code implémente-t-il cette base ? Les modes d'Hermite–Gauss et les exponentielles orthogonalisées mentionnés auparavant ne sont pas ceux de [1] ; préciser lesquels sont réellement utilisés.]*
- Synthèse du profil de couplage dynamique pour l'adaptation d'impédance :

$$\gamma(t) = \frac{|u_{\text{cible}}(t)|^2}{\int_{-\infty}^{t} |u_{\text{cible}}(\tau)|^2 \, d\tau}$$

  *[À VÉRIFIER : forme et normalisation de γ(t) par rapport à [2, 3].]*
- Contraintes matérielles modélisées : plafond de couplage γ_max, filtrage passe-bas par la largeur de raie du résonateur tampon κ_buffer, dispersion de vitesse de groupe (β₂L) et atténuation sur 30 m.

### 2. Solveur d'équation maîtresse en cascade (`slh_solver.py`)

- Intégration numérique de la matrice densité ρ(t) sous équation maîtresse de Lindblad, dans l'espace de Hilbert tronqué des cavités virtuelles.
- Cas simulés : absorption d'un mode cible, réflexion des modes orthogonaux, saturation non linéaire à deux photons, décalages AC Stark dynamiques Δ(t).
  *[À VÉRIFIER : l'étiquette « effet Hong–Ou–Mandel » pour la saturation à deux photons. HOM désigne une interférence à deux photons sur un séparateur ; ne conserver le terme que s'il décrit réellement ce qui est calculé.]*

### 3. Calibration et pré-distorsion (`calibration.py`)

- Tomographie hétérodyne simulée de ⟨a_ref(t)⟩ à la frontière du nœud récepteur.
- Boucle itérative de pré-distorsion pour réduire la réflexion résiduelle et compenser le chirp de phase dû à la dispersion du câble coaxial.

---

## Limites connues

- Le modèle repose sur des approximations de Born–Markov ; le terme « exact » ne s'applique qu'à l'intérieur de ces hypothèses et de la troncature de l'espace de Hilbert.
- La prise en compte de la dispersion (β₂L) dépasse le cadre non dispersif de [2] et n'est pas validée par la littérature citée.
- Aucune comparaison quantitative avec [1] ou [5] n'est fournie à ce stade (par exemple le rapport de sélectivité de 40 rapporté dans [1], ou les efficacités d'absorption de [5]).

---

## Structure du dépôt

| Fichier             | Rôle                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| `config.py`         | Constantes physiques, paramètres du lien cryogénique (30 m), limites du coupleur accordable.          |
| `wavepackets.py`    | Bases temporelles orthogonales, propagation dispersive de Fourier, synthèse de γ(t).                  |
| `slh_solver.py`     | Solveur quantique ouvert en cascade (cavités virtuelles + équation de Lindblad).                      |
| `calibration.py`    | Tomographie hétérodyne IQ et pré-distorsion itérative de l'enveloppe.                                 |
| `benchmarks.py`     | Matrices de sélectivité/diaphonie ηₘₙ, fidélité d'intrication *dual-rail*, visibilité HOM.            |
| `run_experiment.py` | Point d'entrée CLI et vérifications analytiques (conservation de la trace, orthonormalité).           |

---

## Installation

### Prérequis

- Python ≥ 3.10
- `numpy`, `scipy`, `matplotlib`

```bash
git clone https://github.com/berryberryship-debug/microwave-temporal-modes.git
cd microwave-temporal-modes
pip install numpy scipy matplotlib
```

*[À VÉRIFIER : l'ancien dépôt cité dans la commande de clonage, `TemporalMode-QNet.git`, ne correspond pas à ce dépôt.]*

## Utilisation et vérification

```bash
python run_experiment.py
```

*[À COMPLÉTER : options CLI réelles, sortie attendue, durée d'exécution.]*

## Licence

*[À COMPLÉTER : aucune licence n'est indiquée.]*

## Citer ce travail

*[À COMPLÉTER : format de citation du dépôt.]*
