# Projet CCF Forward Looking – IFRS9  
*Université Paris 1 - Master MOSEF 2025*

## Objectif
Ce projet vise à modéliser et projeter un **Credit Conversion Factor (CCF)** sensible à la conjoncture économique (approche *Forward Looking*) pour estimer plus finement l’**EAD IFRS9**, à l’instar de la PD et LGD.

## Méthodologie

### 1. Analyse exploratoire
- Étude de la **stationnarité** des séries CCF par segment (test ADF).
- Exclusion du **segment 6** (CCF constant).
- Traitement des données non stationnaires (différence, log, etc.).
- Corrélation avec les **variables macroéconomiques** (PIB, inflation, chômage, immobilier).

### 2. Modélisation économétrique
- Régression linéaire pondérée (WLS) par segment.
- Variables explicatives enrichies (ratios, interactions, indicateurs de crise).
- Analyse des résidus et VIF.
- Test de modèles alternatifs (Huber, PLS, Random Forest à titre comparatif).

### 3. Projection à 3 ans
- Scénarios macroéconomiques : **Optimiste**, **Central**, **Pessimiste**.
- Projections trimestrielles sur 2024–2026 avec pondération des scénarios.
- Comparaison avec une approche non linéaire (Gradient Boosting).

## Structure du projet

```bash
📁 data/                 # Données sources et séries macroéconomiques
📄 model.py             # Modèles économétriques segmentés
📄 processing.py        # Préparation, transformations des données
📄 prediction.py        # Génération des prévisions CCF
📄 main.ipynb / .py     # Pipeline principal (à exécuter)
📄 explo.ipynb          # Analyse exploratoire
📄 final_To.ipynb       # Synthèse et résolution finale
```

