import streamlit as st
import os
import json
import ModelIA

# Charger mapping ISO ↔ Pays
with open("flags/countries.json", "r", encoding="utf-8") as f:
    iso_to_name = json.load(f)

name_to_iso = {v: k for k, v in iso_to_name.items()}
country_names = sorted(name_to_iso.keys())

st.set_page_config(page_title="Prédiction Match IA", layout="centered")
st.title("⚽ Sélectionnez deux nations pour la prédiction")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("🇦🇺 Équipe 1", country_names, key="team1")
    iso1 = name_to_iso[team1].upper()
    svg1_path = f"flags/png100px/{iso1}.svg"

    if os.path.exists(svg1_path):
        with open(svg1_path, "r", encoding="utf-8") as f:
            svg1 = f.read()
        st.markdown(f"<div style='text-align:center'>{svg1}</div>", unsafe_allow_html=True)
    else:
        st.error(f"Drapeau non trouvé pour {team1}")

with col2:
    team2 = st.selectbox("🇧🇷 Équipe 2", country_names, key="team2")
    iso2 = name_to_iso[team2].upper()
    svg2_path = f"flags/png100px/{iso2}.svg"

    if os.path.exists(svg2_path):
        with open(svg2_path, "r", encoding="utf-8") as f:
            svg2 = f.read()
        st.markdown(f"<div style='text-align:center'>{svg2}</div>", unsafe_allow_html=True)
    else:
        st.error(f"Drapeau non trouvé pour {team2}")

# Vérifier si les deux pays sont différents
if team1 == team2:
    st.warning("Veuillez choisir deux pays différents.")
# Vérifier si les deux pays sont différents
if team1 == team2:
    st.warning("Veuillez choisir deux pays différents.")
else:
    resultat = ModelIA.predire_match(team1, team2)

    if "error" in resultat:
       st.error(resultat["error"])
    else:
        st.success(resultat["gagnant"])
        st.markdown(f"📘 Confrontations historiques : {resultat['confrontations']}")
        st.markdown(f"📝 Résumé : {resultat['resume']}")
        st.markdown("📊 **Possession estimée :**")
        st.markdown(f"- {team1} : {resultat['possession'][team1]}%")
        st.markdown(f"- {team2} : {resultat['possession'][team2]}%")
