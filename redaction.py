import streamlit as st

st.set_page_config(page_title="Assistant Hozana", page_icon="✨", layout="centered")

st.title("✨ Assistant Hozana")
st.markdown("Bienvenue dans l'assistant Hozana. Choisissez un outil dans le menu à gauche.")
st.markdown("---")

# 🔍 Liste des pages acceptées par Streamlit (pour debug)
# st.sidebar.write("Pages disponibles :", list(st._runtime.scriptrunner._main_script_pages.keys()))

st.subheader("🧠 Outils disponibles")

# Dictionnaire des pages : label → nom exact défini par set_page_config dans chaque page
tools = {
    "🧾 Rédaction d'article SEO multilingue": "Rédaction d'article SEO multilingue",
    "📝 Traduction multilingue d'articles": "Traduction multilingue d'articles",
    "🚀 Publication automatique d'articles": "Publication automatique d'articles",
    "📸 Insertion automatique d’images réalistes": "Insertion automatique d’images réalistes",
    "🔗 Liens internes & suggestions de communautés": "Liens internes & suggestions de communautés",
}

# Affichage des boutons et redirection avec switch_page()
for label, page_title in tools.items():
    if st.button(label):
        st.switch_page(page_title)

st.markdown("---")
st.info("Pour toute question ou bug, contactez l’équipe technique.")
