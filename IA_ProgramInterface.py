import streamlit as st
import os
import json
import base64
import ModelIA

# === Config Streamlit ===
st.set_page_config(page_title="Prédiction Match IA", layout="centered")

# === Fonction d'encodage image locale en base64 ===
def local_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

# === CSS pour style global, image de fond, drapeaux en cercle, carte centrée ===
st.markdown(f"""
    <style>
    body {{
        background-image: url('{local_bg_image("assets/images/wr.webp")}');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}

    .stApp {{
        background-color: rgba(255, 255, 255, 0.6);
    }}

    .main-card {{
        background-color: white;
        margin: 3rem auto;
        padding: 2rem 2.5rem;
        border-radius: 20px;
        box-shadow: 0px 8px 30px rgba(0,0,0,0.1);
        max-width: 950px;
    }}

    .flag-square {{
        width: 90px;
        height: 90px;
        object-fit: contain;
        display: block;
        margin: 0 auto;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}

    .flag-label {{
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin-top: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(255, 255, 255, 0.4);  /* Ajuste ici la transparence */
        z-index: -1;
    }
    </style>
    <div class="overlay"></div>
""", unsafe_allow_html=True)


# === Chargement des données des pays ===
with open("flags/countries.json", "r", encoding="utf-8") as f:
    iso_to_name = json.load(f)
name_to_iso = {v: k for k, v in iso_to_name.items()}
country_names = sorted(name_to_iso.keys())

# === Wrapper de contenu ===
st.markdown("<h1 style='text-align:center; color:#1a73e8;'>⚽ Prédiction de match international</h1>", unsafe_allow_html=True)

# === Sélection des équipes ===
col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("🌍 Équipe 1", country_names, key="team1")
    iso1 = name_to_iso[team1].upper()
    svg1_path = f"flags/png100px/{iso1}.svg"
with col2:
    team2 = st.selectbox("🌍 Équipe 2", country_names, key="team2")
    iso2 = name_to_iso[team2].upper()
    svg2_path = f"flags/png100px/{iso2}.svg"

# === Affichage des drapeaux ronds ===
st.markdown("<hr>", unsafe_allow_html=True)
flag_col1, flag_col2 = st.columns(2)

def render_flag_circle(path, name):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            svg = f.read()
        return f"<div class='flag-circle'>{svg}</div><div class='flag-label'>{name}</div>"
    else:
        return f"<div class='flag-label' style='color:red;'>Drapeau manquant<br>{name}</div>"

with flag_col1:
    st.markdown(render_flag_circle(svg1_path, team1), unsafe_allow_html=True)
with flag_col2:
    st.markdown(render_flag_circle(svg2_path, team2), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# === Résultat prédiction ===
if team1 == team2:
    st.warning("🚫 Veuillez choisir deux pays différents.")
else:
    resultat = ModelIA.predire_match(team1, team2)

    if resultat["error"]:
        st.error(resultat["error"])
    else:
        # Résultat
        st.markdown(f"""
        <div style="background-color:#e8f5e9; padding: 20px; border-left: 5px solid #43a047; border-radius: 10px;">
            <h4 style="color:#2e7d32;">🎯 Résultat prédit</h4>
            <p style="font-size:18px;">{resultat['prediction']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Possession
        st.markdown("### 📊 Possession estimée")
        p1, p2 = st.columns(2)
        teams = list(resultat["possession"].keys())
        with p1:
            st.markdown(f"""
            <div style="background-color:#e3f2fd; padding:15px; border-radius:10px; text-align:center;">
                <b>{teams[0]}</b><br><span style="font-size:20px;">{resultat['possession'][teams[0]]:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        with p2:
            st.markdown(f"""
            <div style="background-color:#f1f8e9; padding:15px; border-radius:10px; text-align:center;">
                <b>{teams[1]}</b><br><span style="font-size:20px;">{resultat['possession'][teams[1]]:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Score
        st.markdown("### 🧮 Score estimé")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"<div style='text-align:center; font-size:22px'><b>{team1}</b><br>⚽ {resultat['scores'][team1]}</div>", unsafe_allow_html=True)
        with s2:
            st.markdown(f"<div style='text-align:center; font-size:22px'><b>{team2}</b><br>⚽ {resultat['scores'][team2]}</div>", unsafe_allow_html=True)

        # Historique
        st.markdown("### 📝 Historique des confrontations")
        st.markdown(f"""
        <div style="padding:15px; background-color:#fffde7; border-radius:10px; border:1px solid #f0f0f0;">
            <b>Nombre :</b> {resultat['confrontations']}<br>
            <b>Résumé :</b> {resultat['resume']}
        </div>
        """, unsafe_allow_html=True)

# === Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:gray;'>© 2025 - Projet IA Football - Streamlit</div>", unsafe_allow_html=True)
 