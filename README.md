# SNCF Data Challenge  

Ce projet participe au **SNCF Data Challenge** organisé par l'ENS Paris, dont l'objectif est d'améliorer la précision des prévisions de temps d'attente des trains Transilien.  

## 📌 Objectif  
Prédire l'écart entre le temps d'attente théorique et réel d'un train à une station donnée, en utilisant des données historiques.  

## 📂 Données  
Les données sont constituées de :  
- **x_train.csv** : données d'entraînement  
- **y_train.csv** : valeurs cibles  
- **x_test.csv** : données de test  
- **y_sample.csv** : exemple de soumission  

Chaque ligne correspond à un arrêt de train et contient des informations sur les horaires théoriques/réels, la station et des historiques de passage.  

## 🛠️ Techniques utilisées  

- Utilisation de la librairie AutoML autogluon
- FasAI NeuralNetwork classifier

## 📎 Liens utiles  
- [Challenge sur ENS Data](https://challengedata.ens.fr/participants/challenges/166/)  
- [Dépôt GitHub](https://github.com/arthurdrk/SNCF-Data-Challenge)  

-
Ce README donne un aperçu rapide et efficace du projet ! 😊
