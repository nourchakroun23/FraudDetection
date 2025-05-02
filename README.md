# Analyse de Données et Modélisation - Rapport de Projet

## Description

Ce projet a pour objectif d'étudier différents jeux de données, d’en extraire des caractéristiques pertinentes et d’entraîner plusieurs modèles de classification pour prédire les classes associées. Les étapes comprennent :

- L'analyse exploratoire des données.
- Le prétraitement.
- La sélection de caractéristiques.
- L'entraînement de plusieurs modèles.
- L’évaluation des performances.

## Jeux de Données

Trois jeux de données ont été analysés :

1. **Titanic** :
   - Objectif : prédire la survie des passagers.
   - Variables clés : classe, sexe, âge, tarif, etc.
   - Traitement : imputation des valeurs manquantes, encodage des variables catégorielles.

2. **Iris** :
   - Objectif : classification des espèces d’iris (setosa, versicolor, virginica).
   - Données bien équilibrées, sans valeurs manquantes.
   - Traitement minimal requis.

3. **Wine Quality** :
   - Objectif : prédire la qualité du vin (classification binaire : bon vs mauvais).
   - Traitement : binarisation de la cible, standardisation des variables numériques.

## Modèles Entraînés

Les modèles suivants ont été entraînés et évalués sur chacun des jeux de données :

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Naive Bayes**

## Évaluation des Performances

Chaque modèle a été évalué à l'aide de métriques telles que :

- **Accuracy**
- **Recall**
- **Precision**
- **F1-score**
- **Matrice de confusion**

Les meilleures performances ont été observées avec :

- **Random Forest** sur Titanic et Wine Quality.
- **SVM** sur Iris.

## Conclusion

Les résultats montrent que la qualité du prétraitement a un impact significatif sur les performances. Des modèles plus complexes comme les forêts aléatoires ont mieux capté la structure des données que les modèles linéaires dans la majorité des cas.

## Structure du Répertoire

