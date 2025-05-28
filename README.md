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
📁 data/                    # Données sources et séries macroéconomiques
📄 model.py                 # Modèles économétriques segmentés
📄 processing.py            # Préparation, transformations des données
📄 prediction.py            # Génération des prévisions CCF
📄 main.ipynb               # Pipeline principal (à exécuter)
📄 NotebookComplet.ipynb    # Exploration, tests et résultats
📄 requirements.txt         # Dépendances Python nécessaires

```
## 📁 Données confidentielles

⚠️ Pour des raisons de confidentialité, les données utilisées dans ce projet ne sont **pas incluses** dans le dépôt GitHub.

Vous devez donc **créer manuellement** l’arborescence de dossiers attendue, notamment un dossier `data/` pour stocker les fichiers sources, ainsi qu’un sous-dossier `macrovariables/` dédié aux séries économiques utilisées pour la projection.

``` bash
data/
├── Données_CCF_PAR_SEGMENT.csv
├── Données_CCF_SERIE_GLOBALE_VF.csv
└── macrovariables/
└── historique_macro_variables_projet_CCF_FowardLooking_IFRS9.xlsx

```

Placez les fichiers de données nécessaires dans ces dossiers selon la structure prévue par le code. L’ensemble du pipeline repose sur cette architecture pour charger et traiter les données correctement.

Merci de respecter cette organisation pour garantir le bon fonctionnement des scripts.



Afin d'installer les différentes dépendances vous pouvez configurer un environnement virtuel puis exécuter cette commande :
```bash
pip install -r requirements.txt
```

## 👥 Auteurs
- [Emma Eberle](https://github.com/emmaebrl)
- [Morgan Jowitt](https://github.com/morganjowitt)
- [Gaétan Dumas](https://github.com/gaetan250)
- [Pierre Liberge](https://github.com/ton1rvr)

