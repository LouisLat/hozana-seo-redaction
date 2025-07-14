import streamlit as st
import os

# Configuration de la page
st.set_page_config(page_title="Accueil Hozana Tools", layout="wide")

# Authentification simple
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Accès restreint")
    st.markdown("Veuillez vous identifier pour accéder aux outils Hozana.")

    with st.form("login_form"):
        email = st.text_input("Adresse email")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

        if submitted:
            auth = st.secrets["auth"]
            if email == auth["louis_email"] and password == auth["louis_password"]:
                st.session_state.authenticated = True
                st.success("Connexion réussie. Chargement...")
                st.rerun()
            else:
                st.error("Email ou mot de passe incorrect.")
    st.stop()
# Style harmonisé
st.markdown("""
    <style>
        html, body {
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
        }
        .title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }

        }
        .stButton>button {
            width: 100%;
            padding: 0.75rem;
            font-weight: 600;
            font-size: 1rem;
            color: white;
            background-color: #f00020;
            border-radius: 0.5rem;
            border: none;
        }
        .stButton>button:hover {
            color: white;
            background-color: #f00020;
        }
    </style>
""", unsafe_allow_html=True)

# Titre
st.markdown('<div class="title">Outils SEO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bienvenue sur la plateforme interne des outils Hozana. Choisissez un module à ouvrir :</div>', unsafe_allow_html=True)

# Dictionnaire des modules
modules = {
    "Traduction multilingue d’articles": "redaction_article",
    "Publication d’articles traduits": "2_Publication_articles",
    # Ajouter d'autres outils ici
}

# Interface
for label, page_script in modules.items():
    with st.container():
        if st.button(f"{label}", key=label):
            st.switch_page(f"pages/{page_script}.py")
        st.markdown('</div>', unsafe_allow_html=True)
