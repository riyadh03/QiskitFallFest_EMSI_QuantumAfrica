# Guide des Sprints & Architecture Expérimentale
**Qiskit Fall Fest · Moroccan Crop Classification Challenge**
*Équipe : AI/Data (Coordinateur), Industrial Engineering (Quantum), Cybersecurity (Reliability)*

---

## 1. Les Notions Clés Expliquées Simplement

### Qu'est-ce qu'une "Seed" (Seed 0, 1, 2) ?
- Les réseaux de neurones (quantiques et classiques) initialisent leurs poids de manière aléatoire.
- Selon le point de départ aléatoire, le résultat peut varier légèrement.
- Une **seed** fixe le générateur aléatoire pour garantir que tout le monde obtient exactement les mêmes valeurs.
- **Règle des juges :** Évaluer chaque modèle final sur **trois seeds (0, 1, 2)** et reporter la moyenne ± écart-type (ex. Macro-F1 = $0.862 \pm 0.008$) pour prouver la stabilité de l'approche.

### Qu'est-ce que $L$ (Data Re-uploading) ?
- **Modèle de départ ($L=1$) :** Encode les 4 features satellite une seule fois, applique 8 angles d'entraînement et mesure. Circuit peu profond, expressivité limitée.
- **Data Re-uploading ($L=2, 3$) :** On ré-injecte les données entre chaque bloc de rotations entraînables :
  $$\text{Input}(x) \longrightarrow \text{Rotation Layer } 1 \longrightarrow \text{Input}(x) \longrightarrow \text{Rotation Layer } 2 \dots$$
  - $L=1$ : 8 paramètres, circuit de profondeur 6, 3 portes CX (~6 min).
  - $L=2$ : 16 paramètres, circuit de profondeur 11, 6 portes CX (~12 min).
  - $L=3$ : 24 paramètres, circuit de profondeur 16, 9 portes CX (~20 min).

### Qu'est-ce que $h$ (`hidden_units`) dans le Contrôle Classique ?
- **Règle d'équité imposée par le jury :** Le modèle classique de contrôle doit avoir un nombre de paramètres comparable à celui du QNN (**écart $\le 20\%$**).
- Notre réseau classique (MLP) a pour formule : $\mathbf{8h + 3}$ paramètres.
  - $h=1$ : 11 paramètres (contrôle de base).
  - $h=2$ : 19 paramètres $\longleftrightarrow$ compare avec $L=2$ QNN (16 angles, écart de 18.7%).
  - $h=3$ : 27 paramètres $\longleftrightarrow$ compare avec $L=3$ QNN (24 angles, écart de 12.5%).

---

## 2. Architecture & Collaboration : La Règle d'Or (Zéro Conflit Git)

### ⚠️ Pourquoi NE PAS commenter / décommenter dans le Notebook ?
Si chacun commente ou décommente des cellules dans le notebook `.ipynb`, Git va générer des conflits de fusion massifs, et les sorties graphiques (outputs) des autres seront effacées à chaque commit.

###  La Solution : Le Pattern "Saved Artifacts"
1. **Les calculs écrivent dans des dossiers séparés :**
   - Laptop 1 (L=1) génère : `outputs/screening/L1/runs.json`
   - Laptop 2 (L=2) génère : `outputs/screening/L2/runs.json`
   - Laptop 3 (L=3) génère : `outputs/screening/L3/runs.json`
2. **Git ne fusionne que des fichiers distincts :** Zéro conflit !
3. **Le Notebook final ne recalcule pas tout :** Il charge simplement tous les fichiers `runs.json` du dossier `outputs/` en 5 secondes pour afficher tous les graphiques et tableaux comparatifs !

---

## 3. Les Commandes Prêtes à l'Emploi

Grâce aux mises à jour effectuées dans `challenge/models.py` et `challenge/coverage.py`, chacun peut lancer son expérience en une seule commande :

### Phase A : Screening (Seed 0 uniquement sur les 914 parcelles)
- **Laptop 1 ($L=1$, $h=1$) :**
  ```powershell
  python -m challenge.coverage train --size 914 --layers 1 --hidden-units 1 --seeds 0 --results outputs/screening/L1
  ```
- **Laptop 2 ($L=2$, $h=2$) :**
  ```powershell
  python -m challenge.coverage train --size 914 --layers 2 --hidden-units 2 --seeds 0 --results outputs/screening/L2
  ```
- **Laptop 3 ($L=3$, $h=3$) :**
  ```powershell
  python -m challenge.coverage train --size 914 --layers 3 --hidden-units 3 --seeds 0 --results outputs/screening/L3
  ```

### Phase B : Run Final (Après gel à l'Heure 14 sur le modèle gagnant, ex: L=2 sur 1 seul PC)
```powershell
python -m challenge.coverage train --size 914 --layers 2 --hidden-units 2 --seeds 0 1 2 --results outputs/final_L2
```

### Phase C : Comparaison Statistique (Spatial-Block Bootstrap)
```powershell
python -m challenge.coverage compare --baseline results --candidate outputs/final_L2
```

---

## 4. Planning des 24 Heures

| Sprint | Heures | AI/Data (Toi) | Industrial Eng (Quantique) | Cybersecurity (Fiabilité) |
| :--- | :--- | :--- | :--- | :--- |
| **0 : Setup** | 0–2 | Code mis à jour (`models.py`, `coverage.py`). Vérifier `coverage check`. | Cloner, setup `.venv`, tester 1 run rapide. | Créer/vérifier compte IBM Quantum, tokens et accès aux backends. ✅ *Fait : compte `ibm_quantum_platform` vérifié (ibm_fez, ibm_kingston, ibm_marrakesh), `python -m challenge.ibm_check`.* |
| **1 : Build & Test** | 2–8 | Valider l'agrégation des résultats. Vérifier convergence des pertes. | Vérifier propriétés quantiques ($L=1, 2, 3$) : profondeur, CX, temps/évaluation. | Écrire les tests de perturbation (bruit gaussien, feature drop). Tester 1 inférence IBM. ✅ *Fait : `challenge/robustness.py` (QNN vs MLP) + `challenge/qk_reliability.py` ; 1ʳᵉ inférence IBM `dc01gtinj2vs73dda5l0`.* |
| **2 : Screen** | 8–14 | Coordonner le screening Seed 0 sur les 3 laptops ($L=1, 2, 3$). | Mesurer les temps d'exécution et ressources de circuit. | Exécuter le bootstrap spatial (`coverage compare`). ✅ *Fait : bootstrap spatial apparié (18 blocs, 2000 tirages) dans `challenge/qk_run.py`.* |
| **GEL** | **14h** | **Choix définitif du meilleur $L$ ($L=2$ ou $3$) et du contrôle. Plus aucune modification de code.** | | |
| **3 : Final Runs** | 14–19 | Exécuter seeds 0, 1, 2 sur le modèle gelé (sur 1 seule machine). | Générer les graphiques de trade-off (F1 vs profondeur, F1 vs temps). | Lancer le job IBM Quantum vérifié (10–20 échantillons) pour le **bonus +5**. ✅ *Fait : 10 échantillons sur ibm_kingston (jobs `dc01ihanj2vs73dda82g`, `dc027sls2omc7388l0q0`), 381 s.* |
| **4 : Package** | 19–23 | Assembler le notebook final propre qui lit les `outputs/` et affiche tout. | Rédiger les **3 à 5 slides** de présentation. | Tester la reproductibilité sur un clone propre (vérifier tous les checksums). ✅ *Fait : clone propre + `.venv` neuf → checksums OK et `outputs/qk` reproduit à l’identique (après correctif `.gitattributes`).* |
| **Buffer** | 23–24 | Relecture finale, vérification des fichiers de soumission et des slides, soumission. | | |


---

## 5. Tests de Perturbation & Robustesse (Mission Cybersecurity)

### Pourquoi ce test donne un avantage compétitif décisif ?
En télédétection satellitaire, les données réelles sont souvent imparfaites (couverture nuageuse, bruit de capteur optique).
Ce test évalue la **résilience en inférence** : *Face à des données dégradées, quel modèle s'effondre le plus vite ?*

### Le Protocole Expérimental (100% équitable)
1. Les modèles (QNN et Contrôle Classique MLP) sont **déjà entraînés sur les données propres** (on ne ré-entraîne pas).
2. On applique **exactement la même dégradation** au jeu de validation pour les deux modèles :
   - **Baseline propre :** validation standard.
   - **Bruit de capteur faible :** ajout de bruit gaussien $\mathcal{N}(0, \sigma=0.05)$.
   - **Bruit de capteur fort :** ajout de bruit gaussien $\mathcal{N}(0, \sigma=0.10)$.
   - **Feature Drop (Nuage/Perte de canal) :** mise à zéro d'une composante spectrale.
3. On mesure la chute de Macro-F1 :
   $$\Delta \text{F1} = \text{F1}_{\text{clean}} - \text{F1}_{\text{noisy}}$$
   Si le QNN résiste mieux au bruit que le MLP (dégradation plus faible), vous avez découvert et prouvé une propriété scientifique majeure pour vos slides !

### Script de test rapide à exécuter :
```python
import numpy as np
from challenge.metrics import score

def run_robustness_test(qnn_model, classical_model, X_val, y_val, labels):
    results = {}
    rng = np.random.default_rng(2026)
    
    scenarios = {
        'clean': X_val,
        'noise_0.05': X_val + rng.normal(0, 0.05, X_val.shape),
        'noise_0.10': X_val + rng.normal(0, 0.10, X_val.shape),
        'drop_feature_0': np.column_stack([np.zeros(len(X_val)), X_val[:, 1:]])
    }
    
    for name, X_test in scenarios.items():
        pred_q = qnn_model.predict(X_test)
        pred_c = classical_model.predict(X_test)
        results[name] = {
            'qnn_macro_f1': score(y_val, pred_q, labels)['macro_f1'],
            'classical_macro_f1': score(y_val, pred_c, labels)['macro_f1']
        }
    return results
```

### ✅ Résultats (exécuté : seeds 0, 1, 2 · L = h = 1, 2, 3 · 5 tirages de bruit partagés par tous les modèles)
Commande : `python -m challenge.robustness --layers 1 2 3` (ou un processus par L/seed avec `--seeds`, puis `--report`). Sorties : `outputs/robustness/robustness.json`, `fig_robustness.png`.
Feature drop = une bande satellite brute remplacée par sa moyenne d'entraînement (nuage / canal perdu), plus réaliste que la mise à zéro d'une composante PCA. ΔF1 = F1 propre − F1 dégradé (plus haut = moins robuste), moyenne ± écart-type sur les seeds.

| Modèle | F1 propre | ΔF1 σ=0.05 | ΔF1 σ=0.10 | ΔF1 σ=0.20 | ΔF1 moyen par bande (pire) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `reference` | 0.880 ± 0.000 | +0.001 ± 0.000 | +0.003 ± 0.000 | +0.011 ± 0.000 | 0.078 (NDVI) |
| `qnn_L1` | 0.856 ± 0.006 | +0.006 ± 0.006 | -0.000 ± 0.005 | +0.006 ± 0.004 | 0.026 (NDVI) |
| `mlp_h1` | 0.581 ± 0.106 | -0.001 ± 0.004 | +0.008 ± 0.011 | +0.019 ± 0.010 | 0.026 (B12) |
| `qnn_L2` | 0.819 ± 0.007 | +0.001 ± 0.002 | +0.009 ± 0.004 | +0.030 ± 0.009 | 0.033 (NDVI) |
| `mlp_h2` | 0.880 ± 0.021 | +0.005 ± 0.006 | +0.007 ± 0.009 | +0.018 ± 0.013 | 0.022 (B12) |
| `qnn_L3` | 0.819 ± 0.065 | -0.001 ± 0.009 | +0.009 ± 0.017 | +0.043 ± 0.016 | 0.028 (B12) |
| `mlp_h3` | 0.855 ± 0.009 | +0.003 ± 0.003 | +0.001 ± 0.004 | +0.013 ± 0.002 | 0.012 (B12) |

**Conclusion honnête :**
- **σ = 0.05 et 0.10 (le protocole) :** toutes les chutes sont ≤ 0.01, du même ordre que la variation entre seeds. Aucun modèle n'est démontré plus robuste.
- **σ = 0.20 :** les QNN profonds chutent **plus** que leurs MLP appariés (L=2 : 0.030 vs 0.018 ; L=3 : 0.043 vs 0.013). Seul le QNN L=1 reste très stable (0.006), mais son contrôle h=1 est un modèle faible (F1 0.581 ± 0.106), donc la comparaison L=1 n'est pas équitable en capacité.
- **Bandes manquantes :** B12 et NDVI sont les bandes critiques pour tous les modèles 4-PCA. La référence (13 bandes brutes) est la plus sensible en moyenne (perte de NDVI, B12 ou B6).
- **Message pour les slides :** « pas de robustesse quantique supplémentaire mesurée ; plus de profondeur rend le QNN plus sensible au bruit fort ». Il ne faut pas annoncer d'avantage quantique.

---

## 6. IBM Quantum : accès et bonus +5 (Mission Cybersecurity)

### Configurer son accès (sans exposer le token)
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-hardware.txt
.\.venv\Scripts\python.exe -m challenge.ibm_check --save   # demande la clé API en saisie masquée
.\.venv\Scripts\python.exe -m challenge.ibm_check          # backends, files d'attente, quota restant
```
- La clé API se crée sur **https://quantum.cloud.ibm.com**. Le canal `ibm_quantum` est retiré : il faut utiliser `ibm_quantum_platform`.
- Ne jamais coller le token dans une commande (`token='...'`), car il reste dans l'historique du terminal. Il est stocké dans `~/.qiskit/qiskit-ibm.json`, hors du dépôt (`.qiskit/`, `.env` et `*.key` sont dans `.gitignore`).
- Avoir accès aux backends **ne suffit pas** pour le bonus : le jury demande un job vérifié (backend, job ID et comparaison avec un simulateur idéal).

### Ce qui a été fait (inférence uniquement, aucun entraînement sur le matériel)
| | Valeur |
| :--- | :--- |
| Backend | `ibm_kingston` (Heron, 156 qubits) |
| Jobs | `dc01ihanj2vs73dda82g` (3 champs, 1000 shots, 573 circuits), `dc027sls2omc7388l0q0` (7 champs, 500 shots, 1337 circuits) |
| Temps machine | 381 s sur 600 s/mois (plan Open) |
| Bonnes classes sur 10 champs | exact 10/10 · simulateur idéal (mêmes shots) 7/10 · matériel corrigé 5/10 |

Chaque champ correspond à 190 circuits compute–uncompute (un par vecteur de support), plus 1 circuit de calibration U(x)†U(x) protégé par une barrière, sinon le transpileur le supprime. La valeur K corrigée de ibm_kingston colle au simulateur idéal à shots finis, mais le SVM exige une précision de noyau bien plus grande : même en idéal, il faut 10⁵ shots pour atteindre environ 0.85 de macro-F1.
Commandes : `python -m challenge.qk_hardware submit|analyze|shots|summary`. Tous les rapports sont dans `outputs/qk/hardware/`.

