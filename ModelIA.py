# -*- coding: utf-8 -*-
"""
Script de prédiction des résultats de matchs de football
Auteur : Yess
Date : 05/07/2025
Ce script entraîne plusieurs modèles :
- pour prédire le résultat d'un match
- pour estimer la possession de balle
- pour estimer le score final de chaque équipe
"""

# === Imports ===
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# === Configuration ===
DATA_DIR = r"C:\Users\computer house 41\Desktop\tek-up\Projet IA"
CSV_FILENAME = "results.csv"
MODEL_FILE_RESULT = "modele_foot.pkl"
MODEL_FILE_POSSESSION = "modele_possession.pkl"
MODEL_FILE_SCORE_HOME = "modele_score_home.pkl"
MODEL_FILE_SCORE_AWAY = "modele_score_away.pkl"

# === Chargement et préparation des données ===
os.chdir(DATA_DIR)
df = pd.read_csv(CSV_FILENAME)

# Suppression des lignes avec des valeurs manquantes
df.dropna(subset=['home_score', 'away_score'], inplace=True)

# Création de colonnes utiles
df['diff'] = df['home_score'] - df['away_score']
df['gagné'] = (df['home_score'] > df['away_score']).astype(int)

# Simulation de la possession de balle (car absente du dataset)
np.random.seed(42)
df['possession_home'] = 50 + (df['diff'] * 2.5) + np.random.normal(0, 2, len(df))
df['possession_home'] = df['possession_home'].clip(lower=30, upper=70)

# === Features communes ===
X_features = df[['home_score', 'away_score', 'diff']]

# === Entraînement du modèle de résultat (classification) ===
y_result = df['gagné']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_features, y_result, test_size=0.2)
model_result = RandomForestClassifier(n_estimators=100)
model_result.fit(X_train_r, y_train_r)
joblib.dump(model_result, MODEL_FILE_RESULT)
print(f"[✓] Modèle de résultat sauvegardé → {MODEL_FILE_RESULT}")

# === Entraînement du modèle de possession (régression) ===
y_possession = df['possession_home']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_features, y_possession, test_size=0.2)
model_possession = RandomForestRegressor(n_estimators=100)
model_possession.fit(X_train_p, y_train_p)
joblib.dump(model_possession, MODEL_FILE_POSSESSION)
print(f"[✓] Modèle de possession sauvegardé → {MODEL_FILE_POSSESSION}")

# === Entraînement des modèles de score (régression) ===
model_score_home = RandomForestRegressor(n_estimators=100)
model_score_home.fit(X_features, df['home_score'])
joblib.dump(model_score_home, MODEL_FILE_SCORE_HOME)

model_score_away = RandomForestRegressor(n_estimators=100)
model_score_away.fit(X_features, df['away_score'])
joblib.dump(model_score_away, MODEL_FILE_SCORE_AWAY)

print(f"[✓] Modèles de score sauvegardés → {MODEL_FILE_SCORE_HOME}, {MODEL_FILE_SCORE_AWAY}")

# === Fonction d'extraction des features entre 2 équipes ===
def extraire_caracteristiques(team1, team2, df_matches):
    t1 = team1.strip().lower()
    t2 = team2.strip().lower()

    df = df_matches.copy()
    df['home_team'] = df['home_team'].str.lower()
    df['away_team'] = df['away_team'].str.lower()

    matchs = df[((df['home_team'] == t1) & (df['away_team'] == t2)) |
                ((df['home_team'] == t2) & (df['away_team'] == t1))]

    if matchs.empty:
        print(f"[!] Aucun historique entre {team1.title()} et {team2.title()}.")
        return None, 0, ""

    n = len(matchs)

    t1_home = matchs[matchs['home_team'] == t1]
    t1_away = matchs[matchs['away_team'] == t1]
    t2_home = matchs[matchs['home_team'] == t2]
    t2_away = matchs[matchs['away_team'] == t2]

    t1_avg = (t1_home['home_score'].mean() + t1_away['away_score'].mean()) / 2
    t2_avg = (t2_home['home_score'].mean() + t2_away['away_score'].mean()) / 2
    diff = t1_avg - t2_avg

    t1_total = t1_home['home_score'].sum() + t1_away['away_score'].sum()
    t2_total = t2_home['home_score'].sum() + t2_away['away_score'].sum()

    description = f"{team1.title()} a marqué {t1_total} buts contre {t2_total} pour {team2.title()} sur {n} matchs."
    return [[t1_avg, t2_avg, diff]], n, description

# === Fonction principale de prédiction ===
def predire_match(team1, team2):
    print(f"\n=== Prédiction entre {team1} et {team2} ===")
    features, n_matchs, resume = extraire_caracteristiques(team1, team2, df)
    if not features:
        return

    print(f"[📘] {n_matchs} confrontation(s) historique(s) entre {team1.title()} et {team2.title()}")
    print(f"[📝] {resume}")

    # Chargement des modèles
    clf = joblib.load(MODEL_FILE_RESULT)
    reg_poss = joblib.load(MODEL_FILE_POSSESSION)
    reg_score_home = joblib.load(MODEL_FILE_SCORE_HOME)
    reg_score_away = joblib.load(MODEL_FILE_SCORE_AWAY)

    # Prédiction du résultat
    result = clf.predict(features)[0]

    # Prédiction de la possession
    p1 = reg_poss.predict(features)[0]
    features_inv = [[features[0][1], features[0][0], -features[0][2]]]
    p2 = reg_poss.predict(features_inv)[0]
    total = p1 + p2
    p1_norm = (p1 / total) * 100
    p2_norm = (p2 / total) * 100

    # Prédiction des scores
    score1 = round(reg_score_home.predict(features)[0])
    score2 = round(reg_score_away.predict(features)[0])

    # Résultat
    if result == 1:
        print(f"[🎯] {team1.title()} a plus de chances de remporter le match.")
    else:
        print(f"[🎯] {team2.title()} semble favori ou un match nul est probable.")

    # Affichage
    print("[📊] Possession estimée :")
    print(f"   - {team1.title()} : {p1_norm:.2f}%")
    print(f"   - {team2.title()} : {p2_norm:.2f}%")

    print("[⚽️] Score final estimé :")
    print(f"   - {team1.title()} : {score1} but(s)")
    print(f"   - {team2.title()} : {score2} but(s)")

# === Exemple d’utilisation ===
if __name__ == "__main__":
    predire_match("Tunisia", "France")
    predire_match("Brazil", "Argentina")
