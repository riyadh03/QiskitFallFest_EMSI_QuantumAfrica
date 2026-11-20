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
| **0 : Setup** | 0–2 | Code mis à jour (`models.py`, `coverage.py`). Vérifier `coverage check`. | Cloner, setup `.venv`, tester 1 run rapide. | Créer/vérifier compte IBM Quantum, tokens et accès aux backends. |
| **1 : Build & Test** | 2–8 | Valider l'agrégation des résultats. Vérifier convergence des pertes. | Vérifier propriétés quantiques ($L=1, 2, 3$) : profondeur, CX, temps/évaluation. | Écrire les tests de perturbation (bruit gaussien, feature drop). Tester 1 inférence IBM. |
| **2 : Screen** | 8–14 | Coordonner le screening Seed 0 sur les 3 laptops ($L=1, 2, 3$). | Mesurer les temps d'exécution et ressources de circuit. | Exécuter le bootstrap spatial (`coverage compare`). |
| **GEL** | **14h** | **Choix définitif du meilleur $L$ ($L=2$ ou $3$) et du contrôle. Plus aucune modification de code.** | | |
| **3 : Final Runs** | 14–19 | Exécuter seeds 0, 1, 2 sur le modèle gelé (sur 1 seule machine). | Générer les graphiques de trade-off (F1 vs profondeur, F1 vs temps). | Lancer le job IBM Quantum vérifié (10–20 échantillons) pour le **bonus +5**. |
| **4 : Package** | 19–23 | Assembler le notebook final propre qui lit les `outputs/` et affiche tout. | Rédiger les **3 à 5 slides** de présentation. | Tester la reproductibilité sur un clone propre (vérifier tous les checksums). |
| **Buffer** | 23–24 | Relecture finale, vérification des fichiers de soumission et des slides, soumission. | | |
