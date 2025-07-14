import streamlit as st
import os

# Configuration
st.set_page_config(page_title="Accueil Hozana Tools", layout="wide")

# Authentification
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
            secrets = st.secrets["auth"]
            if email == secrets["louis_email"] and password == secrets["louis_password"]:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.success("Connexion réussie. Redirection...")
                st.rerun()
            else:
                st.error("Email ou mot de passe incorrect.")
    st.stop()

# Interface (après connexion)
st.markdown("""
    <style>
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
        .stButton>button {
            width: 100%;
            padding: 0.75rem;
            font-weight: 600;
            font-size: 1rem;
