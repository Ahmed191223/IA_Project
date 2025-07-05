 

# Importation des bibliothèques nécessaires
import pandas as pd                      # pour manipuler les données tabulaires
import numpy as np                       # pour les calculs numériques et statistiques
import os                                # pour gérer les chemins de fichiers
from sklearn.model_selection import train_test_split          # pour diviser les données en train/test
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # modèles ML
import joblib                            # pour sauvegarder et recharger les modèles

# === Chemins et fichiers de configuration ===
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = "results.csv"               # Fichier CSV avec les résultats des matchs
MODEL_FILE_RESULT = "modele_foot.pkl"      # Fichier pour sauvegarder le modèle de résultat
MODEL_FILE_POSSESSION = "modele_possession.pkl"  # Fichier pour sauvegarder le modèle de possession

# === Chargement des données ===
os.chdir(DATA_DIR)                          # On se place dans le dossier de données
df = pd.read_csv(CSV_FILENAME)              # Lecture du fichier CSV dans un DataFrame

# === Nettoyage des données ===
df.dropna(subset=['home_score', 'away_score'], inplace=True)  # Suppression des lignes sans scores

# === Création de nouvelles colonnes (features) ===
df['diff'] = df['home_score'] - df['away_score']              # Différence de buts entre domicile et extérieur
df['gagné'] = (df['home_score'] > df['away_score']).astype(int)  # 1 si victoire à domicile, 0 sinon

# === Simulation de la possession de balle ===
np.random.seed(42)                         # Pour des résultats reproductibles

# Possession simulée en fonction de l’écart de score avec un peu de bruit
df['possession_home'] = 50 + (df['diff'] * 2.5) + np.random.normal(0, 2, len(df))
df['possession_home'] = df['possession_home'].clip(lower=30, upper=70)  # On limite entre 30% et 70%

# === Entraînement du modèle de prédiction du résultat ===
X_res = df[['home_score', 'away_score', 'diff']]              # Features d’entrée pour le modèle
y_res = df['gagné']                                            # Label (victoire ou non)

# Séparation en jeu d’entraînement et test
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.2)

# Création et entraînement du modèle de classification
model_result = RandomForestClassifier(n_estimators=100)       # 100 arbres dans la forêt
model_result.fit(X_train_res, y_train_res)                    # Entraînement du modèle

# Sauvegarde du modèle de résultat
joblib.dump(model_result, MODEL_FILE_RESULT)
print(f"[✓] Modèle 'résultat' entraîné et sauvegardé → {MODEL_FILE_RESULT}")

# === Entraînement du modèle de prédiction de possession ===
X_poss = X_res                          # Même features que pour le modèle de résultat
y_poss = df['possession_home']         # Cible : possession simulée

# Division train/test pour le modèle de possession
X_train_poss, X_test_poss, y_train_poss, y_test_poss = train_test_split(X_poss, y_poss, test_size=0.2)

# Entraînement du modèle de régression
model_possession = RandomForestRegressor(n_estimators=100)
model_possession.fit(X_train_poss, y_train_poss)

# Sauvegarde du modèle de possession
joblib.dump(model_possession, MODEL_FILE_POSSESSION)
print(f"[✓] Modèle 'possession' entraîné et sauvegardé → {MODEL_FILE_POSSESSION}")

# === Fonction pour extraire les statistiques historiques entre deux équipes ===
def extraire_caracteristiques(team1, team2, df_matches):
    """
    Fonction qui extrait les statistiques utiles pour prédire un match
    :param team1: nom de la première équipe
    :param team2: nom de la deuxième équipe
    :param df_matches: DataFrame des matchs historiques
    :return: features pour prédiction, nb de matchs, description du bilan
    """
    # Mise en minuscules pour éviter les erreurs de casse
    t1 = team1.strip().lower()
    t2 = team2.strip().lower()

    # Copie des données + standardisation des noms d’équipes
    df = df_matches.copy()
    df['home_team'] = df['home_team'].str.lower()
    df['away_team'] = df['away_team'].str.lower()

    # Filtrage des matchs entre les deux équipes (dans les deux sens)
    matchs = df[((df['home_team'] == t1) & (df['away_team'] == t2)) |
                ((df['home_team'] == t2) & (df['away_team'] == t1))]

    # Si aucun match trouvé, on arrête ici
    if matchs.empty:
        print(f"[!] Aucun historique entre {team1.title()} et {team2.title()}.")
        return None, 0, ""

    n = len(matchs)  # Nombre de matchs trouvés

    # Calcul des moyennes de score pour chaque équipe
    t1_home = matchs[matchs['home_team'] == t1]
    t1_away = matchs[matchs['away_team'] == t1]
    t2_home = matchs[matchs['home_team'] == t2]
    t2_away = matchs[matchs['away_team'] == t2]

    t1_avg = (t1_home['home_score'].mean() + t1_away['away_score'].mean()) / 2
    t2_avg = (t2_home['home_score'].mean() + t2_away['away_score'].mean()) / 2
    diff = t1_avg - t2_avg  # Différence de score moyenne

    # Total des buts marqués par chaque équipe
    t1_buts = t1_home['home_score'].sum() + t1_away['away_score'].sum()
    t2_buts = t2_home['home_score'].sum() + t2_away['away_score'].sum()

    # Création d'une description textuelle
    description = f"{team1.title()} a marqué {t1_buts} buts contre {t2_buts} pour {team2.title()} sur {n} matchs."

    return [[t1_avg, t2_avg, diff]], n, description

def predire_match(team1, team2):
    """
    Fonction principale pour prédire le résultat et la possession d’un match
    entre deux équipes données. Retourne un dictionnaire avec les infos.
    """
    # Récupération des features + historique
    features, n_matchs, resume = extraire_caracteristiques(team1, team2, df)
    if not features:
        return {"error": "Aucune donnée disponible pour cette confrontation."}

    # Chargement des modèles
    clf = joblib.load(MODEL_FILE_RESULT)
    reg = joblib.load(MODEL_FILE_POSSESSION)

    # Prédiction du résultat (0 = perd/nul, 1 = gagne)
    gagnant = clf.predict(features)[0]

    # Prédiction de la possession pour team1
    p1 = reg.predict(features)[0]

    # Inverser les features pour team2
    features_inv = [[features[0][1], features[0][0], -features[0][2]]]
    p2 = reg.predict(features_inv)[0]

    # Normaliser la possession
    total = p1 + p2
    p1_norm = (p1 / total) * 100
    p2_norm = (p2 / total) * 100

    # Résumé gagnant
    if gagnant == 1:
        gagnant_txt = f"{team1.title()} a plus de chances de remporter le match."
    else:
        gagnant_txt = f"{team2.title()} semble favori ou un match nul est probable."

    # Retour sous forme de dictionnaire
    return {
        "resume": resume,
        "confrontations": n_matchs,
        "gagnant": gagnant_txt,
        "possession": {
            team1: round(p1_norm, 2),
            team2: round(p2_norm, 2)
        }
    }

# === Lancement du script principal ===
if __name__ == "__main__":
    # Exemple : prédiction pour un match entre Tunisia et France
    predire_match("Tunisia", "France")
    # Tu peux décommenter ci-dessous pour tester d’autres confrontations :
    # predire_match("Argentina", "Germany")
