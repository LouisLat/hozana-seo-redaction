import streamlit as st

st.set_page_config(page_title="Assistant Hozana", page_icon="✨", layout="centered")

st.title("✨ Assistant Hozana")

st.markdown("Bienvenue dans l'assistant Hozana. Choisissez un outil dans le menu à gauche.")

st.markdown("---")

st.subheader("🧠 Outils disponibles")

tools = {
    "📝 Traduction multilingue d'articles": "redaction_multilingue",
    "📈 Assistant SEO multilingue": "assistant_seo",
    "🚀 Publication automatique d'articles": "publication_automatique",
    "📸 Insertion automatique d’images réalistes": "insertion_images",
    "🔗 Liens internes & suggestions de communautés": "liens_communautes",
}

for label, script_name in tools.items():
    st.markdown(f"- [{label}](/{script_name})")

st.markdown("---")
st.info("Pour toute question ou bug, contactez l’équipe technique.")
