import streamlit as st

st.set_page_config(page_title="Accueil Hozana Tools", layout="wide")

if "selected_page" in st.session_state:
    page = st.session_state.pop("selected_page")
    st.switch_page(f"pages/{page}.py")

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

# Interface
modules = {
    "Rédaction d’article SEO multilingue": "redaction_article",
    "Traduction multilingue d’articles": "1_Traduction_articles",
    "Publication d’articles traduits": "2_Publication_articles"
}

for label, script in modules.items():
    if st.button(label, key=label):
        st.session_state.selected_page = script
        st.experimental_rerun()

