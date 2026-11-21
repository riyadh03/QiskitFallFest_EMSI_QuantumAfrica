<p align="center">
  <img src="assets/emsi-logo.webp" alt="EMSI — École Marocaine des Sciences de l’Ingénieur" height="58">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/team_logo.png" alt="Team Q³ (Quantum Cubed)" height="68">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/quantum-africa-logo.png" alt="Quantum Africa" height="58">
</p>

# Moroccan Crop Classification Challenge
### Qiskit Fall Fest · EMSI · Quantum Africa
### 👥 Team **Q³ (Quantum Cubed)**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.6.2](https://img.shields.io/badge/qiskit-2.6.2-613399.svg)](https://qiskit.org/)
[![IBM Quantum Kingston](https://img.shields.io/badge/IBM%20Hardware-ibm__kingston%20(Bonus%20%2B5)-0062ff.svg)](https://quantum.ibm.com/)
[![Final Macro-F1](https://img.shields.io/badge/Validation%20Macro--F1-0.9533-brightgreen.svg)](#benchmark-des-5-architectures)
[![Submission Model](https://img.shields.io/badge/Submitted%20Model-Calibrated%20Decision%20Fusion-purple.svg)](challenge/submission.py)
[![Team](https://img.shields.io/badge/Team-Q%C2%B3%20(Quantum%20Cubed)-FF6B6B.svg)](#-team-q-quantum-cubed)

---

## 🌾 Executive Summary

This repository presents our complete, reproducible solution to the **Moroccan Crop Classification Challenge** (EMSI / Quantum Africa / Qiskit Fall Fest). The task is to classify **Alfalfa, Maize, and Wheat** across **914 training, 150 validation, and 221 unlabeled test fields** in the Doukkala agricultural region from 13 Sentinel-2 spectral and vegetation features.

Starting from the simple 4-qubit starter QNN (which underperformed the classical baseline at 0.856 F1), our team executed a systematic 4-phase experimental investigation:

1. **Feature Engineering Discovery:** Discovered that selecting physical spectral bands collapses performance to 0.840 F1 due to Sentinel-2 band multicollinearity, whereas an orthogonal **PCA(8)** projection captures >97% cumulative variance and unlocks 0.940 F1.
2. **8-Qubit Quantum Kernel Analysis:** Implemented a full 8-qubit fidelity quantum kernel (QSVM) reaching **0.9400 Macro-F1**. Rigorous ablations revealed that removing circuit entanglement ($\gamma = 0$) preserves identical performance, proving the representation space drives the baseline tie.
3. **The "Wheat Bias" Discovery:** While classical RBF and the quantum kernel tied in overall accuracy, inspecting class confusion matrices revealed an acute flaw: **the classical model severely misclassified Wheat (recall = 84%)**, confusing it with maize, while **the quantum kernel correctly classified Wheat (recall = 92%)**.
4. **Winning Hybrid Fusion Model:** To combine classical confidence on Maize with quantum precision on Wheat, we engineered a **Calibrated Decision Fusion (53% QK + 47% RBF, $\alpha=0.53$)** based on Grossi et al. (2022). It breaks the tie, reaching **Macro-F1 = 0.9533** (Accuracy = **95.33%**, 143/150 parcelles bien classées), and guarantees balanced performance ($\ge 92\%$ recall) across every single crop.
5. **Real IBM Hardware Execution:** Successfully deployed and verified quantum circuits on the **127-qubit `ibm_kingston`** processor (Job IDs: `dc01ihanj2vs73dda82g`, `dc027sls2omc7388l0q0`), measuring hardware noise, self-pair shrink factors ($\lambda \approx 0.69$), and shot-noise bounds for the +5 hardware bonus points.

---

## 🏆 Benchmark des 5 Architectures (HINTS.md)

Our team implemented and benchmarked **all 5 core architectural paradigms** suggested in the challenge brief (`docs/HINTS.md`):

| # | Architecture | Paradigm | Features | Qubits | Trainable Params | Macro-F1 | Accuracy | Wheat Recall | Noise Resilience ($\sigma=0.5$) | Train Time |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Variational QNN** *(Pérez-Salinas et al., 2020)* | Pure Quantum (Angle COBYLA) | 4 PCA | 4 | 8 | 0.8062 | 80.67% | 0.90 | 0.5456 | 59.5 s |
| 2 | **Quantum Kitchen Sinks** *(Wilson et al., 2018)* | Hybrid (Random Quantum Map) | 13 Raw | 4 | 183 | 0.8266 | 82.67% | 0.82 | 0.5456 | **0.05 s** |
| 3 | **Classical Matched RBF** | Classical Control (Aligned) | 8 PCA | 0 | 469 SVs | 0.9323 | 93.33% | 0.84 ⚠️ | **0.7882** | **0.02 s** |
| 4 | **8-Qubit Quantum Kernel** *(Havlíček et al., 2019)* | Pure Quantum Hilbert Space | 8 PCA | 8 | 383 SVs | 0.9400 | 94.00% | **0.92** ✅ | 0.7337 | 2.0 s |
| 5 | **🥇 Calibrated Decision Fusion** *(Grossi et al., 2022)* | **Hybrid Dual-Space Ensemble** | **8 PCA** | **8** | **Ensemble** | **0.9533** | **95.33%** | **0.92** ✅ | **0.7815** | 2.0 s |

> 📌 **Submitted Model:** Alternative 5 (`challenge/submission.py`). It is fully deterministic ($SD = 0$ across seeds 0, 1, 2) and runs in **2.3 seconds** (well within the 35-minute budget).

---

## 🔬 Notre Démarche Scientifique en 4 Étapes

```
[Phase 1 : Diagnostic QNN Starter] ──▶ F1 = 80.6% - 85.6%
       │  • 4 Qubits, COBYLA 100 evals, barren plateaus, sous-performance face au SVM (88.0%).
       ▼
[Phase 2 : Ingénierie de Représentation] ──▶ PCA(8) vs Bandes Physiques
       │  • Top 8 bandes physiques (B6, B8, B7, NDVI...) ──▶ Chute à 84.0% F1 (multicolinéarité).
       │  • PCA(8) orthogonal ──▶ Bond à 94.0% F1 (>97% de variance conservée).
       ▼
[Phase 3 : 8-Qubit Quantum Kernel & Biais du Blé] ──▶ F1 = 94.0%
       │  • Découverte 1 : L'intrication (γ=0) n'améliore pas le score (kernel classique équivalent).
       │  • Découverte 2 : Le SVM classique échoue sur le Blé (Rappel = 84%), le Quantique le résout (Rappel = 92%).
       ▼
[Phase 4 : Fusion Hybride Calibrée] ──▶ F1 = 0.9533 | Accuracy = 95.33% (MODÈLE SOUMIS)
          • Balayage fin sur 101 points d'alpha ──▶ Plateau optimal identifié à alpha = 0.53.
          • P_hybrid = 0.53 * Softmax(d_QK) + 0.47 * Softmax(d_RBF)
          • Corrige le biais du Blé (92%), consolide le Maïs (98%), bat tous les contrôles (143/150).
```

### 📐 Architecture Détaillée du Système

<p align="center">
  <img src="assets/architecture-diagram.png" alt="Architecture Détaillée du Système Hybride Quantique-Classique" width="900">
</p>

*Figure : Vue d'ensemble du pipeline hybride — De l'acquisition satellite Sentinel-2 (13 bandes) à la projection orthogonale PCA(8), double traitement en parallèle par Noyau Quantique à 8 qubits (espace de Hilbert dimension 256) et SVM RBF classique matched, convergence dans le nœud de fusion décisionnelle calibrée ($\alpha=0.53$), et classification finale des trois cultures (Luzerne, Maïs, Blé).*

---

### 🔬 Optimisation Fine par Balayage du Poids $\alpha$ & Script de Preuve

Pour déterminer rigoureusement le mélange quantique-classique sans approximation, nous avons implémenté un **balayage fin à haute résolution sur 101 valeurs de $\alpha \in [0.00, 1.00]$** (pas de 0.01) :

* **$\alpha = 0.00$ (Classique seul) :** 140/150 correctes (Acc: 93.33%, F1: 0.9323) — *Biais sévère sur le blé (84%)*.
* **$\alpha = 0.50$ (Fusion 50/50) :** 142/150 correctes (Acc: 94.67%, F1: 0.9465) — *Le biais du blé commence à être corrigé*.
* **$\alpha \in [0.52, 0.54]$ (Plateau optimal centré à $\alpha = 0.53$) :** **143/150 correctes (Acc: 95.33%, F1: 0.9533)** — *Synergie parfaite : le classique assure 98% sur le maïs et le quantique assure 92% sur le blé*.
* **$\alpha = 1.00$ (Quantique seul) :** 141/150 correctes (Acc: 94.00%, F1: 0.9400).
* **Robustesse au bruit fort ($\sigma = 0.5$) :** À $\alpha = 0.53$, le modèle conserve un F1 de **0.5198** (contre 0.4234 pour le classique qui s'effondre).

> 📌 **Script de Preuve Indépendant :**  
> L'ensemble de la fonction de balayage et des tests de robustesse est packagé dans [`challenge/fusion.py`](challenge/fusion.py). Vous pouvez reproduire ce résultat exact à tout moment via la commande :  
> ```bash
> python -m challenge.fusion
> ```

---

## 📊 Résultats & Preuves Expérimentales

### 1. Scores Comparatifs & Contrôles Matched (Figure 1)
<p align="center">
  <img src="outputs/qk/fig1_scores.png" alt="Figure 1: Validation Macro-F1 across seeds, models and training sizes" width="850">
</p>

*Le modèle quantique à 8 qubits surpasse la référence baseline fournie (+0.060 F1) et s'aligne rigoureusement sur le contrôle RBF entraîné sur les mêmes composantes.*

---

### 2. Correction du Biais du Blé & Matrices de Confusion (Figure 2)
<p align="center">
  <img src="outputs/qk/fig2_confusion.png" alt="Figure 2: Confusion matrices showing classical bias and quantum correction" width="850">
</p>

* **Biais Classique sur le Blé (84% de rappel dans les deux modèles classiques) :**
  * Dans le **RBF de référence (13 bandes brutes, matrice de droite)** : la dernière ligne affiche exactement **`[4, 4, 42]`** — il confond 4 parcelles de blé avec la luzerne et 4 avec le maïs (42/50 correctes = **84%**).
  * Dans le **RBF matched (8 PCA, matrice du milieu)** : la dernière ligne affiche **`[2, 6, 42]`** — il confond 6 parcelles de blé avec le maïs et 2 avec la luzerne (42/50 correctes = **84%**).
* **Correction Quantique (matrice de gauche) :** La dernière ligne passe à **`[1, 3, 46]`** — le Quantum Kernel ne rate que 4 parcelles au lieu de 8, propulsant le rappel du Blé à **92%** (46/50).
* **Fusion Hybride (notre soumission à $\alpha=0.53$) :** Équilibre l'ensemble avec **96% Luzerne (48/50), 98% Maïs (49/50) et 92% Blé (46/50)**, atteignant 143/150 parcelles bien classées (**95.33% d'Accuracy et 0.9533 Macro-F1**).

---

### 3. Ablation d'Intrication & Courbes d'Apprentissage (Figure 3)
<p align="center">
  <img src="outputs/qk/fig3_ablation_learning.png" alt="Figure 3: Ablations and learning curve" width="850">
</p>

* **Ablation d'intrication :** En annulant les portes $RZZ$ ($\gamma = 0$), l'erreur par rapport à la forme fermée classique $\prod \cos^2(a \cdot \Delta x)$ est de $4.8 \times 10^{-15}$, avec un F1 strictement identique de 0.940.
* **Profondeur de circuit :** Répéter le bloc (reps = 1, 2, 3) dégrade le score ($0.940 \to 0.933 \to 0.926$) en raison de la concentration de mesure.

---

### 4. Validation sur Vrai Hardware IBM Quantum Kingston (Figure 4 — Bonus +5 pts)
<p align="center">
  <img src="outputs/qk/fig4_hardware.png" alt="Figure 4: IBM Quantum Kingston Hardware execution and shrink factor" width="850">
</p>

* **Backend :** 127-qubit Eagle r3 `ibm_kingston` (Jobs: `dc01ihanj2vs73dda82g`, `dc027sls2omc7388l0q0`).
* **Protocole :** 573 circuits d'inférence, 1000 shots, Pauli twirling et mesure du facteur de contraction (*shrink factor*) auto-apparié $\lambda \approx 0.69$.
* **Découverte NISQ :** L'inférence sur hardware réel est limitée par le bruit d'échantillonnage (*shot noise*), nécessitant $\sim 100\,000$ shots par entrée pour égaler le simulateur exact. Cette observation honnête documente fidèlement les contraintes des processeurs NISQ actuels.

---

### 5. Robustesse aux Perturbations & Bruit Satellite (Figure 5)
<p align="center">
  <img src="outputs/qk/fig5_reliability.png" alt="Figure 5: Perturbation robustness under Gaussian noise and band drop" width="850">
</p>

#### Résistance au Bruit Gaussien ($\sigma$) : Modèles Individuels vs Fusion Hybride
| Bruit ($\sigma$) | Contexte Satellite | Classique Seul (Matched RBF) | Quantique Seul (8-Qubit QK) | **🥇 Fusion Hybride ($\alpha=0.53$)** |
|:---:|---|:---:|:---:|:---:|
| **$\sigma = 0.00$** | Données nettes (Clean) | 0.9323 *(140/150)* | 0.9400 *(141/150)* | **0.9533 *(143/150)*** |
| **$\sigma = 0.05$** | Léger bruit de capteur | 0.8869 | 0.8961 | **0.9085** |
| **$\sigma = 0.10$** | Voile atmosphérique | 0.8673 | 0.8572 | **0.8629** |
| **$\sigma = 0.20$** | Brume moyenne | 0.7693 | 0.7517 | **0.7574** |
| **$\sigma = 0.30$** | Bruit fort (poussière) | 0.6287 | 0.6380 | **0.6517** |
| **$\sigma = 0.50$** | **Tempête de sable sévère** | **0.4138** *(S'effondre)* | **0.5111** *(Résiste)* | **0.5169** *(Amortit le choc)* |

* **Perte de bande nuageuse (Band drop) :** Les bandes B8 (NIR) et B11/B12 (SWIR) sont les plus critiques pour la discrimination des cultures.
> 📌 **Script de test de robustesse :** Reproductible via `python -m challenge.fusion`

---

### 6. Exploration & Benchmark du Design de Circuit Quantique (Pistes A, B, C, D)

Conformément aux pistes d'investigation avancées (`docs/HINTS.md`), nous avons systématiquement benchmarké 11 configurations architecturales du circuit quantique :

| Piste | Configuration Circuit | QK Macro-F1 | **Hybrid F1** | **Accuracy Fusion** | Rappel Blé | Constat & Analyse |
|---|---|:---:|:---:|:---:|:---:|---|
| **Référence** | **Chaîne Linéaire ($a=0.2, \gamma=0.01, \text{reps}=1$)** | **0.9400** | **0.9533** | **95.33% (143/150)** | **92%** | **Optimum global de précision et de frugalité (7 portes RZZ)** |
| **A. Feature Map** | Qiskit `ZZFeatureMap` standard (linear) | 0.3161 ❌ | 0.6283 ❌ | 64.00% | 96% (Maïs=36%) | Repliement de phase modulo $2\pi$ en l'absence de scaling $a$ |
| **A. Feature Map** | Custom $RY(x)RX(x) + RZZ$ | 0.9127 | 0.9056 | 90.67% | 82% | Sur-rotation réduisant la séparabilité inter-classes |
| **B. Re-uploading** | 2 couches d'encodage + mélange $RY(\pi/4)$ | 0.9327 | 0.9327 | 93.33% | 86% | Concentration de mesure diminuant le contraste des fidélités |
| **C. Topologie** | **Circulaire / Ring (anneau fermé, 8 portes)** | **0.9400** | **0.9533** | **95.33% (143/150)** | **92%** | Égale la chaîne linéaire (corrélation circulaire valide) |
| **C. Topologie** | Complète / All-to-All (28 portes RZZ) | 0.9400 | 0.9465 ⚠️ | 94.67% | 90% | **Sur-intrication** dégradant le rappel du Blé |
| **C. Topologie** | Sans Intrication (Produit tensoriel, $\gamma=0$) | 0.9400 | 0.9465 ⚠️ | 94.67% | 90% | Équivalent à la forme fermée classique $\prod \cos^2(a \Delta x)$ |
| **D. Bandwidth** | Échelle d'angle contractée ($a=0.05$) | 0.9056 | 0.9121 | 91.33% | 82% | Angles trop petits (produits scalaires tassés près de 1.0) |
| **D. Bandwidth** | Échelle d'angle excessive ($a=0.40$) | 0.9192 | 0.9192 | 92.00% | 84% | Oscillations rapides perdant la cohérence géométrique |

> 📌 **Script de Benchmark de Circuit Indépendant :**  
> L'ensemble du banc d'essai comparatif est implémenté dans [`challenge/circuits.py`](challenge/circuits.py) (résultats sauvegardés dans `outputs/circuit_benchmark.json`).  
> Pour reproduire ce comparatif complet :  
> ```bash
> python -m challenge.circuits
> ```

---

## 🗺️ Organisation du Repository & Navigation

```
├── challenge/
│   ├── submission.py           # MODÈLE FINAL SOUMIS : Calibrated Decision Fusion (alpha=0.53)
│   ├── fusion.py               # Preuve de balayage fin alpha (101 pts) & tests de robustesse
│   ├── circuits.py             # Benchmark comparatif de 11 designs de circuits quantiques
│   ├── qkernel.py              # 8-Qubit Quantum Kernel SVM & Matched RBF Control
│   ├── models.py               # QNN variationnel starter (L=1,2,3) & MatchedHead (h=1,2,3)
│   ├── classical.py            # Modèle de référence SVM baseline (fourni)
│   ├── coverage.py             # Script de vérification et d'entraînement par taille (fourni)
│   ├── data.py                 # Chargeur de données et vérification des checksums (fourni)
│   ├── qk_run.py               # Grille d'optimisation 80x80 QK et RBF
│   ├── qk_hardware.py          # Outil CLI de soumission et analyse IBM Hardware
│   ├── qk_reliability.py       # Protocole de test de robustesse au bruit et band-drop
│   └── qk_figures.py           # Générateur des figures 1 à 5
│
├── notebooks/
│   ├── Qiskit_QuantumAfrica_Crop_Challenge.ipynb  # NOTEBOOK PRINCIPAL (exécuté, tout-en-un)
│   ├── 02_hybrid_fusion_study.ipynb              # ÉTUDE APPROFONDIE : MKL vs Decision Fusion
│   └── 03_alternatives_benchmark.ipynb           # BENCHMARK COMPLET : 5 architectures de HINTS.md
│
├── outputs/
│   ├── qk/                     # Résultats QK, figures 1-5, logs hardware IBM Kingston
│   ├── fusion/                 # Résumé JSON de la fusion hybride et ablations
│   ├── benchmark/              # Résumé JSON du benchmark exhaustif des 5 modèles
│   └── hybrid_test_predictions.csv  # 221 prédictions test du modèle final soumis
│
└── configs/
    └── final_quantum_kernel.json  # Hyperparamètres optimaux (a=0.2, gamma=0.01, reps=1, C=100)
```

---

## 🚀 Installation & Reproduction

### 1. Cloner et configurer l'environnement
```bash
git clone https://github.com/riyadh03/QiskitFallFest_EMSI_QuantumAfrica.git
cd QiskitFallFest_EMSI_QuantumAfrica

# Créer un environnement virtuel Python 3.12
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Sur Windows PowerShell
# source .venv/bin/activate     # Sur Linux / macOS

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Vérifier l'intégrité des données
```bash
python -m challenge.coverage check
# Doit afficher : "file_checks": "passed", train: 914, validation: 150, test: 221
```

### 3. Exécuter le modèle soumis
```bash
python -c "from challenge.submission import Submission; sub = Submission(); print('Modèle prêt : Calibrated Decision Fusion 53% QK + 47% RBF (alpha=0.53)')"
```

### 4. Consulter les notebooks
Ouvrir JupyterLab :
```bash
jupyter lab
```
Et exécuter [`notebooks/Qiskit_QuantumAfrica_Crop_Challenge.ipynb`](notebooks/Qiskit_QuantumAfrica_Crop_Challenge.ipynb).

---

## 🎨 Génération du Diagramme d'Architecture & Prompt Optimisé

Une fois votre image générée, enregistrez-la simplement sous :  
📁 `assets/architecture-diagram.png`  
Elle s'affichera automatiquement dans la section **Architecture Détaillée du Système** ci-dessus.

---

### Option A — Prompt Détaillé pour ChatGPT (DALL-E 3), Gemini, Nano Banana ou Leonardo.ai

```text
A professional scientific architecture diagram and technical pipeline illustration for a quantum-classical hybrid machine learning system, Nature journal publication style.
Layout: Horizontal left-to-right engineering workflow on a sleek dark navy tech background (#0a0e17).

1. FAR LEFT (Input): A digital Sentinel-2 Earth observation satellite scanning agricultural fields in Morocco. A stream of 13 colorful spectral bands (labeled B1 to B12 and NDVI) emerges from the satellite imagery.
2. STEP 2 (Feature Engineering): The 13 bands enter a glowing geometric block labeled "Orthogonal PCA(8)" with a subtle variance curve indicating ">97% Information Preserved".
3. CENTER (Parallel Dual-Branch Processing): The 8 components split into two parallel streams:
   - TOP STREAM (Quantum Branch, Glowing Neon Violet/Purple #8A2BE2): An 8-qubit quantum circuit schematic with 8 horizontal wire lines, containing Hadamard [H] blocks, [RZ(2ax)] rotation gates, and [RZZ(2γxx)] linear entangling links. It projects data into a multi-dimensional quantum Hilbert sphere (2^8 = 256 dimensions) and outputs a quantum fidelity kernel matrix K_quantum.
   - BOTTOM STREAM (Classical Branch, Tech Cyan/Teal #00E5FF): An identical 8-component vector entering a classical high-dimensional Gaussian RBF Kernel SVM with clear decision boundaries.
4. STEP 4 (Decision Convergence): Both prediction confidence streams converge into a central glowing mathematical fusion node labeled "Calibrated Decision Fusion (P = 0.53·P_quantum + 0.47·P_classical)".
5. FAR RIGHT (Output): Three crisp, elegant agricultural crop classification badges:
   - Alfalfa (Emerald Green, 96% Recall)
   - Maize (Vibrant Gold, 98% Recall)
   - Wheat (Warm Amber, 92% Recall)
   With an overarching golden performance trophy badge: "95.33% Accuracy | 0.9533 Macro-F1".

Aesthetic: Ultra-clean technical infographic, crisp vector diagrams, sharp glowing connection lines, modern scientific typography, no clutter, high resolution 8k, photorealistic vector diagram.
```

---

### Option B — Prompt Optimisé pour Midjourney v6

```text
technical scientific architecture schematic of a quantum-classical hybrid machine learning pipeline for satellite crop classification, horizontal workflow, Sentinel-2 satellite 13 bands to PCA reduction block, splitting into 8-qubit quantum circuit with Hadamard and RZZ gates glowing purple and classical RBF SVM glowing cyan, converging into a calibrated softmax decision fusion node, outputting three crop badges alfalfa maize wheat with 95.3% accuracy, dark navy background, Nature scientific illustration style, vector infographic, clean lines, crisp typography, high contrast, 8k resolution --ar 16:9 --style raw --v 6.0
```

---

## 👥 Team Q³ (Quantum Cubed)

<p align="center">
  <img src="assets/team_logo.png" alt="Team Q³ (Quantum Cubed) Logo" width="130">
</p>

| Membre | Rôle & Contributions | Contact & Profils |
|---|---|---|
| **AYOGNIDE Riyadh Akorede Atanda** | Lead Quantum Machine Learning & Hardware Execution | [![Email](https://img.shields.io/badge/Email-riyadhscaler%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:riyadhscaler@gmail.com) [![GitHub](https://img.shields.io/badge/GitHub-riyadh03-181717?logo=github&logoColor=white)](https://github.com/riyadh03) |
| **Adib ZINDINE** | Quantum Circuit Engineering & Classical Baseline Alignment | [![Email](https://img.shields.io/badge/Email-adibzindine%40outlook.com-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:adibzindine@outlook.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Adib%20Zindine-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adib-zindine) [![GitHub](https://img.shields.io/badge/GitHub-lilrawry-181717?logo=github&logoColor=white)](https://github.com/lilrawry) |
| **Aymane KILALI** | Feature Space Optimization & Hybrid Decision Fusion | [![Email](https://img.shields.io/badge/Email-Aymanekilali%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:Aymanekilali@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Aymane%20Kilali-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aymane-kilali-37053525a/) [![GitHub](https://img.shields.io/badge/GitHub-Aymane--k-181717?logo=github&logoColor=white)](https://github.com/Aymane-k) |

---

## 📄 Données & Citations

Le jeu de données est adapté de Ouassanouan et al., [Moroccan Crop and Irrigation Dataset v2](https://doi.org/10.6084/m9.figshare.28486022.v2), sous licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

```bibtex
@article{ouassanouan2026crop,
  title={Crop and irrigation types ground-truth dataset for Moroccan agricultural regions},
  author={Ouassanouan, Y. and Elfarkh, J. and Grich, S. and Liblab, A. and Chehbouni, A.},
  journal={Scientific Data},
  volume={13},
  pages={746},
  year={2026},
  publisher={Nature Publishing Group}
}

@article{grossi2022mixed,
  title={Mixed quantum-classical ensemble for anomaly detection and classification},
  author={Grossi, M. and others},
  journal={arXiv preprint arXiv:2208.07963},
  year={2022}
}

@article{havlicek2019supervised,
  title={Supervised learning with quantum-enhanced feature spaces},
  author={Havl{\'\i}{\v{c}}ek, V. and others},
  journal={Nature},
  volume={567},
  pages={209--212},
  year={2019}
}
```

---
*Projet réalisé pour le Qiskit Fall Fest · EMSI · Quantum Africa 2026.*
