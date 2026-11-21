# TemporalMode-QNet

**Simulateur de systèmes quantiques ouverts pour l'émission, la propagation et l'absorption sélective de photons micro-ondes en modes temporels orthogonaux.**

> **Statut : projet de recherche, partiellement validé.** Le solveur a été corrigé et reproduit désormais les propriétés attendues d'un absorbeur idéal (voir « Résultats vérifiés »). Il n'a **pas** été comparé quantitativement aux données expérimentales de [1]. Les éléments marqués **[À VÉRIFIER]** restent à confirmer par l'auteur.

---

## Références

1. A. Hernández-Antón *et al.*, « Emission and Absorption of Microwave Photons in Orthogonal Temporal Modes across a 30-Meter Two-Node Network », arXiv:2604.12947 (14 avril 2026). **Preprint, non certifié par évaluation par les pairs.** https://arxiv.org/abs/2604.12947
   *[À VÉRIFIER : l'ancienne référence « Phys. Rev. Lett. 137, 210801 » n'a pu être confirmée ; ne la rétablir que si la source est documentée.]*
2. A. H. Kiilerich et K. Mølmer, « Input-Output Theory with Quantum Pulses », *Phys. Rev. Lett.* **123**, 123604 (2019). arXiv:1902.09833. https://doi.org/10.1103/PhysRevLett.123.123604
3. A. H. Kiilerich et K. Mølmer, « Quantum interactions with pulses of radiation », arXiv:2003.04573 (2020). *[À VÉRIFIER : référence de revue.]*
4. G. F. Peñas, R. Puebla, J. J. García-Ripoll, « Multiplexed quantum state transfer in waveguides », *Phys. Rev. Research* **6**, 033294 (2024) (base de modes sech, citée par [1] comme réf. 26 ; arXiv:2403.12222).
5. K. Sunada *et al.*, arXiv:2603.10506 (11 mars 2026). Preprint ; travail parallèle sur quatre modes.

---

## Approche

Formalisme entrée–sortie avec **cavités virtuelles** [2, 3], en cascade (SLH). Conventions de [2] :

- cavité émettrice : g_u(t) = u*(t) / √(1 − ∫₋∞ᵗ |u|²)  (Éq. 2)
- cavité absorbante : g_v(t) = −v*(t) / √(∫₋∞ᵗ |v|²)  (Éq. 3)
- Hamiltonien de cascade : H = (i/2)(g_u* g_rec a†b − g_u g_rec* b†a), avec L_tot = g_u a + g_rec b (Éqs. 5–6)

**Hypothèses de [2]** : couplage markovien et propagation non dispersive. La dispersion du lien est ici traitée en amont comme une déformation de la fonction de mode : c'est une extension propre à ce dépôt, non validée par [2].

---

## Modules

| Fichier | Rôle |
| --- | --- |
| `config.py` | Paramètres du lien et du récepteur. **Valeurs illustratives**, pas celles de [1]. |
| `wavepackets.py` | Bases Hermite–Gauss, exponentielles Gram–Schmidt et **famille sech de [1]** (`generate_sech_polynomial_basis`) ; propagation dispersive ; couplage d'absorption. |
| `slh_solver.py` | Équation maîtresse de Lindblad en cascade (cavité virtuelle + transmon à 3 niveaux). |
| `calibration.py` | Calibration en boucle fermée par optimisation Nelder–Mead (a, s). |
| `tests/test_physics.py` | Tests de non-régression physique. |
| `benchmarks.py`, `run_experiment.py` | **Non audités** dans cette révision. |

---

## Corrections de cette révision

1. **Signe du Hamiltonien de cascade (critique).** L'ancien `H_casc` avait le signe opposé à l'Éq. (5) de [2] et au produit en série SLH : la cascade était inversée. Résultat avant correction : capture du mode cible = 0,00 %, sortie = 100 %. Le test de trace (dérive ~10⁻¹⁶) ne détectait pas l'erreur. Changer le signe de g_rec ne la corrige pas (équivalence de jauge).
2. **Calibration remplacée.** L'ancienne règle heuristique dégradait la capture (97,75 % → 73,39 %) pour tous les pas d'apprentissage testés. Elle est remplacée par une optimisation de deux paramètres physiques : un facteur d'amplitude a et une phase s·∫|g|² compensant un décalage Stark proportionnel au taux.
3. **Seuils de régularisation** `cum_lo`, `cum_hi`, `emit_remaining_min` désormais dans `config.py`. Les anciennes valeurs (10⁻⁵ et 0,9995) bornaient la capture à ~99,95 % et fixaient le pic de γ(t).
4. **Convention de phase** alignée sur [2] (conjugaison, signe de g_v).
5. Annotations **[À VÉRIFIER]** dans `config.py` ; suppression d'une variable inutilisée ; ajout des tests.

---

## Résultats vérifiés

Exécutés dans un environnement NumPy 2.5.4 / SciPy 1.18.1, paramètres par défaut de `config.py` (à reproduire par l'auteur).

| Test | Résultat |
| --- | --- |
| Mode cible HG0 sur filtre HG0 (sans Stark) | capture 100,00 %, sortie 0,00 % |
| Mode orthogonal HG1 sur filtre HG0 | capture 0,00 %, sortie 100 % |
| Avec décalage Stark, avant calibration | capture 97,80 %, diaphonie 1,77 % |
| Après calibration | capture 100,00 %, diaphonie 0,00 %, s = −0,1500 (= `stark_coeff`) |
| Famille sech de [1], Γ/2π = 14 MHz, matrice de transfert idéale 3×3 | diagonale 1, hors-diagonale 0 (à 10⁻⁴), pics de γ entre 12 et 14 MHz |
| Deux photons dans HG0 | P_e = 66,69 %, P_f = 0,00 %, 1,333 photon sortant |

**Lecture prudente :**
- La calibration retrouve exactement le paramètre Stark parce que le modèle simulé contient la même forme : le test est circulaire et ne valide rien sur du matériel réel.
- La matrice de transfert idéale n'est pas comparable à la sélectivité de ≈ 40 mesurée dans [1], qui inclut pertes et imperfections.
- Le résultat à deux photons (66,69 %) n'est confronté à aucune référence : **je ne sais pas** s'il est physiquement correct.

---

## Limites connues

- Modèle simplifié du récepteur : le transmon est couplé directement au guide avec un taux dépendant du temps et la capture est comptée en |e⟩. Dans [1], l'absorption se fait dans |f⟩ via la transition f0–g1 et un résonateur de largeur κ ; leur couplage physique g̃(t) (Éq. 4 de [1]) **n'est pas implémenté**.
- Valeurs de `config.py` illustratives : atténuation 0,005 dB/m (vs < 1 dB/km pour le guide dans [1]), v_g = 2,0×10⁸ m/s (vs ≈ 0,72 c), κ_buffer = 150 MHz et γ_max = 50 MHz (vs κ ≈ 27–31 MHz dans [1]). β₂ sans source ; son effet est négligeable (~10⁻¹⁶).
- Le lissage gaussien (σ = 1/κ_buffer) est heuristique ; un résonateur a une réponse exponentielle.
- Propagation par FFT périodique sans remplissage de zéros.
- Le pic de couplage de HG0 (≈ 21,8 MHz) est fixé par le seuil `cum_lo`, pas par la physique.
- Aucune étiquette « Hong–Ou–Mandel » n'est justifiée dans ces fichiers. **[À VÉRIFIER dans `benchmarks.py`.]**
- Le terme « exact » ne vaut que dans le cadre markovien et la troncature de l'espace de Hilbert (3 × 3).

---

## Installation et vérification

Prérequis : Python ≥ 3.10, **NumPy ≥ 2.0** (usage de `np.trapezoid`), SciPy ; `pytest` optionnel.

```bash
git clone https://github.com/berryberryship-debug/microwave-temporal-modes.git
cd microwave-temporal-modes
pip install numpy scipy
python tests/test_physics.py      # ou : python -m pytest tests/
python slh_solver.py
python calibration.py
```

## Licence

*[À COMPLÉTER : aucune licence n'est indiquée.]*

## Citer ce travail

*[À COMPLÉTER.]*
