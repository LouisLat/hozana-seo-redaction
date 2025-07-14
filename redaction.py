import streamlit as st

st.set_page_config(page_title="Assistant Hozana", page_icon="✨", layout="centered")

st.title("✨ Assistant Hozana")
st.markdown("Bienvenue dans l'assistant Hozana. Choisissez un outil dans le menu à gauche.")
st.markdown("---")

st.subheader("🧠 Outils disponibles")

tools = {
    "🧾 Rédaction d'article SEO multilingue": "pages/redaction_article.py",
    "📝 Traduction multilingue d'articles": "pages/traduction_multilingue.py",
    "🚀 Publication automatique d'articles": "pages/publication_automatique.py",
    "📸 Insertion automatique d’images réalistes": "pages/insertion_images.py",
    "🔗 Liens internes & suggestions de communautés": "pages/liens_communautes.py",
}

for label, page_path in tools.items():
    if st.button(label):
        st.switch_page(page_path)

st.markdown("---")
st.info("Pour toute question ou bug, contactez l’équipe technique.")
