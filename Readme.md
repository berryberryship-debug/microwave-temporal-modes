TemporalMode-QNet

Simulateur de systèmes quantiques ouverts pour l'émission, la propagation et l'absorption sélective de photons micro-ondes en modes temporels orthogonaux.

«Statut : projet de recherche, partiellement validé. Le solveur a été corrigé et reproduit désormais les propriétés attendues d'un absorbeur idéal (voir « Résultats vérifiés »). Il n'a pas été comparé quantitativement aux données expérimentales de [1]. Les éléments marqués [À VÉRIFIER] restent à confirmer par l'auteur.»

Le dépôt comprend également un outil indépendant d'analyse structurelle et de recherche dans le code source ("hybrid_graph_rag.py"). Cet outil facilite l'exploration de l'architecture logicielle. Il ne participe pas aux simulations quantiques et ne constitue pas une validation physique des résultats.

---

Références

1. A. Hernández-Antón et al., « Emission and Absorption of Microwave Photons in Orthogonal Temporal Modes across a 30-Meter Two-Node Network », arXiv:2604.12947 (14 avril 2026). Preprint, non certifié par évaluation par les pairs. https://arxiv.org/abs/2604.12947
   [À VÉRIFIER : l'ancienne référence « Phys. Rev. Lett. 137, 210801 » n'a pu être confirmée ; ne la rétablir que si la source est documentée.]
2. A. H. Kiilerich et K. Mølmer, « Input-Output Theory with Quantum Pulses », Phys. Rev. Lett. 123, 123604 (2019). arXiv:1902.09833. https://doi.org/10.1103/PhysRevLett.123.123604
3. A. H. Kiilerich et K. Mølmer, « Quantum interactions with pulses of radiation », arXiv:2003.04573 (2020). [À VÉRIFIER : référence de revue.]
4. G. F. Peñas, R. Puebla, J. J. García-Ripoll, « Multiplexed quantum state transfer in waveguides », Phys. Rev. Research 6, 033294 (2024) (base de modes sech, citée par [1] comme réf. 26 ; arXiv:2403.12222).
5. K. Sunada et al., arXiv:2603.10506 (11 mars 2026). Preprint ; travail parallèle sur quatre modes.

---

Approche physique

Formalisme entrée–sortie avec cavités virtuelles [2, 3], en cascade (SLH). Conventions de [2] :

- Cavité émettrice :
  (g_u(t) = u^*(t) / \sqrt{1 - \int_{-\infty}^{t} |u|^2}) (Éq. 2)
- Cavité absorbante :
  (g_v(t) = -v^*(t) / \sqrt{\int_{-\infty}^{t} |v|^2}) (Éq. 3)
- Hamiltonien de cascade :
  (H = (i/2)(g_u^* g_{\mathrm{rec}} a^\dagger b - g_u g_{\mathrm{rec}}^* b^\dagger a)), avec (L_{\mathrm{tot}} = g_u a + g_{\mathrm{rec}} b) (Éqs. 5–6).

Hypothèses de [2] : couplage markovien et propagation non dispersive. La dispersion du lien est ici traitée en amont comme une déformation de la fonction de mode : c'est une extension propre à ce dépôt, non validée par [2].

---

Architecture du dépôt

Le projet est organisé autour de deux ensembles fonctionnels distincts :

1. Le moteur de simulation quantique, consacré aux modes temporels, à leur propagation et à leur absorption.
2. Le moteur hybride d'analyse du code, consacré à l'exploration des relations entre modules, fonctions, classes et dépendances du dépôt.

Ces deux ensembles sont complémentaires sur le plan logiciel, mais leurs résultats doivent être interprétés séparément.

Modules de simulation

Fichier| Rôle
"config.py"| Paramètres du lien et du récepteur. Valeurs illustratives, pas celles de [1].
"wavepackets.py"| Bases Hermite–Gauss, exponentielles Gram–Schmidt et famille sech de [1] ("generate_sech_polynomial_basis") ; propagation dispersive ; couplage d'absorption.
"slh_solver.py"| Équation maîtresse de Lindblad en cascade (cavité virtuelle + transmon à 3 niveaux).
"calibration.py"| Calibration en boucle fermée par optimisation Nelder–Mead (a, s).
"test_physics.py"| Tests de non-régression physique.
"benchmarks.py"| Benchmarks numériques. Non audité dans cette révision.
"run_experiment.py"| Point d'entrée pour les vérifications analytiques et les benchmarks. Non audité dans cette révision.

Outil d'analyse du code

Fichier| Rôle
"hybrid_graph_rag.py"| Outil indépendant de cartographie du dépôt, d'analyse AST, de résolution des relations entre modules et de recherche hybride lexicale, structurelle et sémantique.

"hybrid_graph_rag.py" exploite notamment :

- L'analyse syntaxique Python avec "ast" pour extraire les structures du code.
- La représentation des relations entre les composants sous forme de graphe avec NetworkX.
- La recherche lexicale avec BM25.
- La recherche sémantique avec Sentence Transformers lorsque cette fonctionnalité est activée et que le modèle correspondant est disponible.
- La combinaison des résultats de recherche et du contexte structurel pour faciliter l'exploration du dépôt.

Périmètre : cet outil sert à comprendre, parcourir et auditer l'organisation du code. Il ne résout pas l'équation maîtresse, ne calcule pas les propriétés des photons et ne vérifie pas la validité des modèles physiques.

---

Corrections de cette révision physique

1. Signe du Hamiltonien de cascade (critique). L'ancien "H_casc" avait le signe opposé à l'Éq. (5) de [2] et au produit en série SLH : la cascade était inversée. Résultat avant correction : capture du mode cible = 0,00 %, sortie = 100 %. Le test de trace (dérive ~10⁻¹⁶) ne détectait pas l'erreur. Changer le signe de "g_rec" ne la corrige pas (équivalence de jauge).

2. Calibration remplacée. L'ancienne règle heuristique dégradait la capture (97,75 % → 73,39 %) pour tous les pas d'apprentissage testés. Elle est remplacée par une optimisation de deux paramètres physiques : un facteur d'amplitude "a" et une phase "s·∫|g|²" compensant un décalage Stark proportionnel au taux.

3. Seuils de régularisation "cum_lo", "cum_hi", "emit_remaining_min" désormais dans "config.py". Les anciennes valeurs (10⁻⁵ et 0,9995) bornaient la capture à ~99,95 % et fixaient le pic de γ(t).

4. Convention de phase alignée sur [2] (conjugaison, signe de "g_v").

5. Annotations [À VÉRIFIER] dans "config.py" ; suppression d'une variable inutilisée ; ajout des tests.

Ces corrections concernent le moteur physique. Elles ne constituent pas une validation de l'outil d'analyse du code.

---

Résultats vérifiés

Exécutés dans un environnement NumPy 2.5.4 / SciPy 1.18.1, paramètres par défaut de "config.py" (à reproduire par l'auteur).

Test| Résultat
Mode cible HG0 sur filtre HG0 (sans Stark)| capture 100,00 %, sortie 0,00 %
Mode orthogonal HG1 sur filtre HG0| capture 0,00 %, sortie 100 %
Avec décalage Stark, avant calibration| capture 97,80 %, diaphonie 1,77 %
Après calibration| capture 100,00 %, diaphonie 0,00 %, s = −0,1500 (= "stark_coeff")
Famille sech de [1], Γ/2π = 14 MHz, matrice de transfert idéale 3×3| diagonale 1, hors-diagonale 0 (à 10⁻⁴), pics de γ entre 12 et 14 MHz
Deux photons dans HG0| P_e = 66,69 %, P_f = 0,00 %, 1,333 photon sortant

Lecture prudente

- La calibration retrouve exactement le paramètre Stark parce que le modèle simulé contient la même forme : le test est circulaire et ne valide rien sur du matériel réel.
- La matrice de transfert idéale n'est pas comparable à la sélectivité de ≈ 40 mesurée dans [1], qui inclut pertes et imperfections.
- Le résultat à deux photons (66,69 %) n'est confronté à aucune référence : je ne sais pas s'il est physiquement correct.
- Les résultats ci-dessus concernent le modèle de simulation et les tests indiqués. Ils ne sont pas des résultats expérimentaux.
- L'analyse structurelle ou sémantique produite par "hybrid_graph_rag.py" ne constitue pas une preuve de correction mathématique ou physique.

---

Limites connues

- Modèle simplifié du récepteur : le transmon est couplé directement au guide avec un taux dépendant du temps et la capture est comptée en |e⟩. Dans [1], l'absorption se fait dans |f⟩ via la transition f0–g1 et un résonateur de largeur κ ; leur couplage physique g̃(t) (Éq. 4 de [1]) n'est pas implémenté.
- Valeurs de "config.py" illustratives : atténuation 0,005 dB/m (vs < 1 dB/km pour le guide dans [1]), v_g = 2,0×10⁸ m/s (vs ≈ 0,72 c), κ_buffer = 150 MHz et γ_max = 50 MHz (vs κ ≈ 27–31 MHz dans [1]). β₂ sans source ; son effet est négligeable (~10⁻¹⁶).
- Le lissage gaussien (σ = 1/κ_buffer) est heuristique ; un résonateur a une réponse exponentielle.
- Propagation par FFT périodique sans remplissage de zéros.
- Le pic de couplage de HG0 (≈ 21,8 MHz) est fixé par le seuil "cum_lo", pas par la physique.
- Aucune étiquette « Hong–Ou–Mandel » n'est justifiée dans ces fichiers. [À VÉRIFIER dans "benchmarks.py".]
- Le terme « exact » ne vaut que dans le cadre markovien et la troncature de l'espace de Hilbert (3 × 3).

Limites de l'outil hybride

- La qualité de la cartographie dépend de la résolution des imports et des appels par analyse statique.
- Les appels construits dynamiquement, les dépendances résolues à l'exécution et certaines formes de métaprogrammation peuvent échapper à l'analyse AST.
- La recherche sémantique dépend du modèle d'embedding disponible et de son environnement d'exécution.
- Un résultat de recherche indique une proximité lexicale, structurelle ou sémantique ; il ne démontre pas que deux fonctions sont mathématiquement équivalentes.
- L'outil ne remplace ni les tests unitaires, ni les tests physiques, ni une revue scientifique du modèle.

---

Installation

Prérequis :

- Python ≥ 3.10
- NumPy ≥ 2.0 (usage de "np.trapezoid")
- SciPy
- NetworkX et "rank-bm25" pour l'outil hybride
- Sentence Transformers pour activer la recherche sémantique
- "pytest" facultatif pour exécuter les tests avec pytest

Clonage du dépôt :

git clone https://github.com/berryberryship-debug/microwave-temporal-modes.git
cd microwave-temporal-modes

Installation des dépendances du simulateur :

pip install numpy scipy

Installation des dépendances de l'analyse structurelle et lexicale :

pip install networkx rank-bm25

Dépendance facultative pour la recherche sémantique :

pip install sentence-transformers

---

Utilisation

Simulations et vérifications physiques

Tests de non-régression :

python test_physics.py

Ou, si pytest est installé :

python -m pytest

Exécution du solveur :

python slh_solver.py

Exécution de la calibration :

python calibration.py

Point d'entrée des vérifications analytiques et des benchmarks :

python run_experiment.py

Les options précises de "run_experiment.py" sont consultables avec :

python run_experiment.py --help

Analyse du dépôt avec le moteur hybride

L'outil d'analyse est exécuté indépendamment du simulateur :

python hybrid_graph_rag.py --help

Les fonctions de diagnostic et de test intégrées à la version du script peuvent être invoquées séparément :

python hybrid_graph_rag.py --diagnostic
python hybrid_graph_rag.py --test

La construction de l'index et la recherche peuvent être lancées avec les options correspondantes :

python hybrid_graph_rag.py --build-only
python hybrid_graph_rag.py --query "nom du module ou concept recherché"

La recherche sémantique dépend de l'installation de Sentence Transformers et de la disponibilité du modèle configuré. Le mode lexical et structurel doit rester utilisable indépendamment de cette dépendance.

Note : ces commandes décrivent l'interface prévue pour le script hybride. Elles doivent être confirmées par l'exécution de "python hybrid_graph_rag.py --help" et des tests du fichier dans la version effectivement publiée. Leur présence dans ce README ne signifie pas que ces tests ont déjà réussi.

---

Séparation des validations

Le dépôt comporte deux catégories de vérification qui ne doivent pas être confondues :

Niveau| Objet| Ce que la vérification permet d'établir
Tests logiciels| Structures, imports, fonctions et exécution du code| Le comportement logiciel attendu dans les cas testés
Tests numériques et physiques| Équations, propagation, absorption, conservation et observables| La cohérence du modèle dans les hypothèses et limites testées
Analyse hybride du code| Graphe, relations et recherche dans les sources| Une aide à l'exploration et à l'audit de l'architecture

La réussite d'un test logiciel ne valide pas automatiquement le modèle physique. De même, une cartographie correcte des dépendances ne prouve pas la validité des équations simulées.

---

Licence

[À COMPLÉTER : aucune licence n'est indiquée.]

Citer ce travail

[À COMPLÉTER.]
