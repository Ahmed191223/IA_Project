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

# 1. Charger les données
df = pd.read_csv("results.csv")

# 2. Nettoyer : filtrer les lignes valides
df = df.dropna()
df = df[df['home_score'].notnull() & df['away_score'].notnull()]

# 3. Feature engineering
df['goal_diff'] = df['home_score'] - df['away_score']
df['is_win'] = (df['home_score'] > df['away_score']).astype(int)

# 4. Sélection des colonnes
X = df[['home_score', 'away_score', 'goal_diff']]
y = df['is_win']

# 5. Split données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Entraînement modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Évaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Sauvegarde du modèle
joblib.dump(model, 'model_foot.pkl')
print("✅ Modèle entraîné et sauvegardé dans model_foot.pkl")
