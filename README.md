# SNCF Data Challenge  
<img width="458" alt="Suivi" src="https://github.com/user-attachments/assets/c93a9274-0820-4bd5-8648-6aaf07323c43" />

Ce projet est ma participation au sujet SNCF Transilien du **Challenge Data 2025** organisé par l'ENS Paris, l'institut Louis Bachelier et le Collège de France. L'objectif est d'améliorer la précision des prévisions de temps d'attente des trains SNCF Transilien.  

## 📌 Objectif  
Prédire l'écart entre le temps d'attente théorique et réel d'un train à une station donnée, en utilisant des données historiques.  

## 📂 Données  
Les données sont constituées de :  
- **x_train_final.csv** : données d'entraînement  
- **y_train_final_j5KGWWK.csv** : valeurs cibles  
- **x_test_final.csv** : données de test  

Chaque ligne correspond à un arrêt de train et contient des informations sur les horaires théoriques/réels, la station et des historiques de passage.  

## 🛠️ Techniques utilisées  

- Utilisation de la librairie AutoML autogluon
- FastAI NeuralNetwork classifier

## 📎 Liens utiles  
- [Site des Challenges ENS](https://challengedata.ens.fr/participants/challenges/166/)  
