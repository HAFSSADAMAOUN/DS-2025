# Compte Rendu : Description de la Base de Données "Wine Quality"

Ce document présente une analyse descriptive du jeu de données utilisé dans le cadre du projet de Machine Learning (fichier `ML.ipynb`). Les données proviennent du célèbre dépôt **UCI Machine Learning Repository**.

***

## 1. Identification et Origine

*   **Nom du dataset :** Wine Quality (Variante "White" / Vin Blanc).
*   **Source :** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).
*   **Origine géographique :** Échantillons de vin "Vinho Verde" du nord du Portugal.
*   **Type de tâche :** Classification (le notebook transforme le problème initialement de régression/classification multiclasse en classification binaire).

## 2. Structure Générale des Données

L'analyse du code (notamment les cellules utilisant `pandas`) révèle les caractéristiques dimensionnelles suivantes :

*   **Nombre d'instances (échantillons) :** 4 898 observations.
*   **Nombre de variables (colonnes) :** 12 colonnes au total.
    *   **Variables explicatives (Features) :** 11 variables physico-chimiques.
    *   **Variable cible (Target) :** 1 variable sensorielle (`quality`).
*   **Type de données :**
    *   Toutes les variables explicatives sont numériques (flottants `float64`).
    *   La variable cible est un entier (`int64`).
*   **Qualité des données :** Le résumé `df.info()` indique qu'il n'y a **aucune valeur manquante** (*non-null count* = 4898 pour toutes les colonnes).

## 3. Description des Variables

### A. Variables Explicatives (Input)
Ces variables sont basées sur des tests physico-chimiques objectifs :

1.  **fixed acidity** (Acidité fixe) : Acides majoritaires présents dans le vin (tartrique).
2.  **volatile acidity** (Acidité volatile) : Quantité d'acide acétique (un taux trop élevé donne un goût de vinaigre).
3.  **citric acid** (Acide citrique) : Utilisé pour ajouter de la fraîcheur et de la saveur.
4.  **residual sugar** (Sucre résiduel) : Quantité de sucre restant après la fin de la fermentation.
5.  **chlorides** (Chlorures) : Quantité de sel dans le vin.
6.  **free sulfur dioxide** (Dioxyde de soufre libre) : Forme d'équilibre du SO2, protège contre l'oxydation.
7.  **total sulfur dioxide** (Dioxyde de soufre total) : Quantité de formes libres et liées de SO2.
8.  **density** (Densité) : Proche de celle de l'eau, dépend du taux d'alcool et de sucre.
9.  **pH** : Décrit l'acidité ou la basicité du vin (échelle de 0 à 14).
10. **sulphates** (Sulfates) : Additif contribuant aux niveaux de dioxyde de soufre.
11. **alcohol** (Alcool) : Le pourcentage d'alcool par volume.

### B. Variable Cible (Output)

*   **Nom :** `quality`
*   **Nature initiale :** Score sensoriel (donnée catégorielle ordinale) attribué par des experts, compris entre 0 et 10.
*   **Distribution initiale :** Les données sont déséquilibrées. La majorité des vins sont notés 5, 6 ou 7.
    *   Note 6 : 2198 vins
    *   Note 5 : 1457 vins
    *   Note 7 : 880 vins
    *   (Les notes extrêmes 3, 4, 8, 9 sont rares).

## 4. Prétraitement et Transformation dans le Notebook

Le notebook effectue plusieurs opérations critiques modifiant la structure ou l'interprétation de la base de données :

### 1. Binarisation de la Cible
Le problème est simplifié en une classification binaire (Cellule 11) :
*   **Classe 0 (Mauvais vin) :** Si `quality` $\le$ 5.
*   **Classe 1 (Bon vin) :** Si `quality` > 5.

### 2. Analyse Exploratoire
*   **Boxplots :** Générés pour visualiser la distribution des variables explicatives et identifier la présence d'**outliers** (valeurs aberrantes), qui semblent nombreux compte tenu des écarts d'échelle.
*   **Matrice de corrélation :** Utilisée pour analyser les relations linéaires entre les variables physico-chimiques.

### 3. Partitionnement (Splitting)
Les données sont divisées de manière stratifiée (pour conserver les proportions des classes) :
*   Un jeu de **Test** (1/3 des données).
*   Un jeu d'**Apprentissage** (`Xa`, `Ya`).
*   Un jeu de **Validation** (`Xv`, `Yv`).

### 4. Normalisation
L'algorithme utilisé étant le **K-Nearest Neighbors (KNN)**, une étape de standardisation (`StandardScaler`) est appliquée pour mettre toutes les variables à la même échelle (moyenne = 0, écart-type = 1), évitant ainsi que les variables à grandes valeurs (comme le dioxyde de soufre total) ne dominent le calcul de distance.
