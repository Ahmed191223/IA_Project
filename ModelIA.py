# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 15:06:21 2025

@author: computer house 41
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
os.chdir(r"C:\Users\computer house 41\Desktop\tek-up\Projet IA")

# Chargement du fichier CSV contenant les résultats de matchs
dataset = pd.read_csv("results.csv")

# On calcule la différence de buts pour chaque match
dataset['diff'] = dataset['home_score'] - dataset['away_score']

# Si l’équipe à domicile a gagné → 1, sinon 0
dataset['gagné'] = (dataset['home_score'] > dataset['away_score']).astype(int)

# Sélection des colonnes utiles
X = dataset[['home_score', 'away_score', 'diff']]
y = dataset['gagné']

# On divise les données pour entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entraînement avec un modèle de forêt aléatoire (Random Forest)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Sauvegarde du modèle pour utilisation dans l’interface web
joblib.dump(clf, 'modele_foot.pkl')
print(" Modèle entraîné et sauvegardé dans model_foot.pkl")  
