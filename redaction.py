import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import time
import tempfile
from fpdf import FPDF
import statistics
import trafilatura
import re
import unicodedata
from google.oauth2 import service_account
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
import yaml
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.ads.googleads.v22.enums.types.keyword_plan_network import KeywordPlanNetwork

import os
from io import StringIO
from typing import List

import pkg_resources
st.write("Version Google Ads :", pkg_resources.get_distribution("google-ads").version)


SERP_API_KEY = st.secrets["serp_api_key"]
MAGISTERIUM_API_KEY = st.secrets["magisterium_api_key"]
DEEPL_API_KEY = st.secrets["deepl_api_key"]  # si tu utilises DeepL ailleurs
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Configuration Streamlit
st.set_page_config(page_title="Assistant SEO Multilingue", layout="wide")

# 🎨 CSS pour encadrés
st.markdown("""
<style>
.block {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px 15px;
    background-color: #f9f9f9;
    margin-bottom: 15px;
    font-size: 15px;
}
a {
    color: #4B8BBE;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Assistant de rédaction SEO multilingue")

keyword = st.text_input("Mot-clé principal (en français)")

# ✅ Options de l’analyse à activer ou non
st.markdown("### ⚙️ Options de l’analyse")
run_length_analysis = st.checkbox("Analyser la longueur optimale", value=True)
run_keyword_variants = st.checkbox("Rechercher les variantes de mots-clés", value=True)
run_google_ads_data = st.checkbox("Afficher les volumes Google Ads", value=True)
run_community_suggestions = st.checkbox("Suggérer des communautés à promouvoir", value=True)
run_link_suggestions = st.checkbox("Suggérer des liens internes avec ancrage", value=True)

# (Optionnel) Résumé visuel
st.markdown("#### 📝 Parties à analyser :")
if run_length_analysis:
    st.markdown("- 📏 Longueur des concurrents")
if run_keyword_variants:
    st.markdown("- 🧠 Variantes de mots-clés")
if run_google_ads_data:
    st.markdown("- 📊 Volumes Google Ads")
if run_community_suggestions:
    st.markdown("- 🙏 Suggestions de communautés")
if run_link_suggestions:
    st.markdown("- 🔗 Liens internes recommandés")


top_n = st.slider("Nombre de résultats Google FR à analyser", 5, 20, 10)

total_tokens_used = 0
def estimate_cost(tokens_used):
    return round(tokens_used / 1000 * 0.01, 4)

def interroger_magisterium_contenu(section_title, mot_cle):
    url = "https://www.magisterium.com/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MAGISTERIUM_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Tu es un assistant d'écriture catholique fidèle au Magistère. À partir du thème suivant : « {section_title} » (dans le contexte de l'article sur « {mot_cle} »), propose **5 à 7 bullet points** bien rédigés en français qui serviront de contenu pour cette section.

Chaque point doit :
- S’appuyer sur l’enseignement officiel de l’Église catholique,
- Inclure des références explicites si possible (ex : catéchisme, conciles, encycliques…),
- Être clair, structuré, et utilisable directement pour la rédaction.

Réponds uniquement par la liste des bullet points.
    """

    data = {
        "model": "magisterium-1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
        "return_related_questions": False
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        return [f"❌ Erreur Magisterium {response.status_code}"], "", []

    content = response.json()["choices"][0]["message"]["content"]
    citations = response.json()["choices"][0]["message"].get("citations", [])
    lines = [line.strip("•- ").strip() for line in content.splitlines() if line.strip()]

    return lines, content, citations

def get_serp_data(query, lang='fr', country='fr', top_n=10):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "hl": lang,
        "gl": country,
        "num": top_n
    }
    try:
        response = requests.get(url, params=params)
        return response.json().get("organic_results", [])
    except Exception:
        return []

def extract_relevant_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n").strip()
        return text[:8000]
    except Exception:
        return ""

def generate_plan(keyword, top_n, median_words):
    serp_results = get_serp_data(keyword, lang='fr', country='fr', top_n=top_n)
    urls = [r.get("link") for r in serp_results if r.get("link")]
    pages_data = [{"url": url, "content": extract_relevant_text(url)[:1000]} for url in urls]

    prompt = f"""
Tu es un expert SEO spécialisé dans les contenus qui se positionnent en première position sur Google.

À partir des extraits suivants issus des meilleurs résultats de recherche, construis un **plan ultra structuré** pour un article optimisé sur « {keyword} ».

Voici les contraintes :
- La longueur cible de l’article est d’environ **{median_words} mots**
- Propose un nombre raisonnable de H2 et H3 pour cette longueur
- Utilise un plan clair structuré en **H2 > H3**
- Commence chaque ligne strictement par `H2:` ou `H3:` (pas de tirets, pas de numérotation).
- Couvre tous les angles importants pour surpasser les concurrents
- Commence par une introduction et termine par une conclusion mais qui ne contiennent jamais de H3
- Utilise des titres riches en mots-clés SEO longue traîne
- Réponds uniquement par le plan (pas d’explication)

Voici les extraits à analyser :

{json.dumps(pages_data, ensure_ascii=False, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    global total_tokens_used
    total_tokens_used += response.usage.total_tokens
    return response.choices[0].message.content


def extract_sections(plan_text):
    lines = plan_text.strip().splitlines()
    sections = []
    current_h2 = None

    for line in lines:
        match = re.match(r"(H2|H3):\s*(.+)", line)
        if match:
            level, title = match.groups()
            title = title.strip()

            if level == "H2":
                current_h2 = title
                sections.append(("H2", title))
            elif level == "H3" and current_h2:
                sections.append(("H3", title, current_h2))
    return sections

def parse_and_display_plan(plan_text):
    lines = plan_text.strip().splitlines()
    html = ""

    for line in lines:
        match = re.match(r"(H2|H3)[\s:.-]+(.+)", line.strip())
        if match:
            level, content = match.groups()
            content = content.strip()
            if level == "H2":
                html += f"<h2 style='font-size: 20px; margin-top: 1em;'>{content}</h2>\n"
            elif level == "H3":
                html += f"<h3 style='font-size: 16px; margin-left: 1em;'>{content}</h3>\n"

    st.markdown(html, unsafe_allow_html=True)


def generate_pdf(keyword, plan_text, section_bullets):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Fiche SEO : {keyword}", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, plan_text)

    for section, bullets in section_bullets.items():
        pdf.set_font("Arial", "B", 14)
        pdf.ln(8)
        pdf.multi_cell(0, 10, f"📌 {section}")

        pdf.set_font("Arial", "", 11)
        for b in bullets:
            pdf.ln(2)
            pdf.multi_cell(0, 7, b.strip())

    tmp_path = tempfile.mktemp(".pdf")
    pdf.output(tmp_path)
    return tmp_path

def count_words_in_page(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return 0
        extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not extracted or len(extracted) < 100:
            return 0
        word_count = len(extracted.split())
        return word_count
    except Exception:
        return 0

def clean_keyword_variant(text):
    text = re.sub(r"^\d+\s*", "", text)  # Supprime les numéros en début de ligne
    text = re.sub(r"[\"“”']", "", text)  # Supprime les guillemets
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")  # Normalise accents
    return text.strip().lower()

def get_keyword_variants(keyword):
    prompt = f"""
Tu es un expert SEO.

À partir de la requête suivante d'un internaute : « {keyword} »

1. Identifie le **mot-clé principal** que Google associerait à cette recherche (ex : “confession” pour “qu'est-ce que la confession ?”).

2. Propose ensuite 5 à 8 **variantes SEO couramment recherchées** pour ce mot-clé principal, incluant :
- ses formulations les plus naturelles ou raccourcies (ex : “confession”, “sacrement de réconciliation”),
- ses synonymes ou formes longues,
- les formulations proches ou complémentaires que l'on pourrait cibler dans un article SEO.

Réponds uniquement par une liste de mots-clés, sans numérotation, sans phrases complètes.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = response.choices[0].message.content
    global total_tokens_used
    total_tokens_used += response.usage.total_tokens

    # Nettoyage
    variants = [clean_keyword_variant(line) for line in text.strip().splitlines() if line.strip()]
    return list(set(variants))

def get_google_ads_metrics(keywords: List[str]) -> pd.DataFrame:
    def get_google_ads_client():
        if os.path.exists("google-ads.yaml"):
            return GoogleAdsClient.load_from_storage("google-ads.yaml")

        yaml_config = st.secrets["google_ads"]
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            yaml.dump(dict(yaml_config), tmp)
            tmp.flush()
            return GoogleAdsClient.load_from_storage(tmp.name)

    client = get_google_ads_client()
    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")

    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = st.secrets["google_ads"]["login_customer_id"]

    # ✅ Compatibilité avec toutes versions : on utilise get_type
    KeywordPlanNetworkEnum = client.get_type("KeywordPlanNetworkEnum")
    request.keyword_plan_network = KeywordPlanNetworkEnum.KeywordPlanNetwork.GOOGLE_SEARCH

    request.keyword_seed.keywords.extend(keywords)

    try:
        response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        data = []
        for idea in response:
            keyword = idea.text
            volume = idea.keyword_idea_metrics.avg_monthly_searches
            data.append({"Mot-clé": keyword, "Volume mensuel": volume})
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erreur Google Ads : {e}")
        return pd.DataFrame(columns=["Mot-clé", "Volume mensuel"])


def estimate_optimal_word_count(keyword, top_n=10):
    serp_results = get_serp_data(keyword, lang='fr', country='fr', top_n=top_n)
    urls = [r.get("link") for r in serp_results if r.get("link")]

    word_counts = []
    for url in urls:
        wc = count_words_in_page(url)
        if wc > 100:
            word_counts.append((url, wc))  # conserve aussi l’URL

    if not word_counts:
        return 0, 0, None, []

    lengths_only = [w for _, w in word_counts]
    avg = int(sum(lengths_only) / len(lengths_only))
    median = int(statistics.median(lengths_only))

    first_url, first_word_count = word_counts[0]

    return avg, median, (first_url, first_word_count), word_counts


if keyword:

    if run_length_analysis:
        with st.spinner("📊 Analyse de la longueur optimale..."):
            avg_words, median_words, first_result_info, raw_counts = estimate_optimal_word_count(keyword, top_n)

        if avg_words:
            st.markdown("### ✍️ Longueur idéale estimée")
            st.markdown(f"- Moyenne des articles concurrents : **{avg_words} mots**")
            st.markdown(f"- Médiane des articles concurrents : **{median_words} mots**")
            if first_result_info:
                url, wc = first_result_info
                st.markdown(f"- Article en **1re position** : **{wc} mots** ([voir]({url}))")
        else:
            st.warning("⚠️ Impossible d'estimer la longueur optimale : contenu insuffisant ou bloqué.")
    else:
        median_words = 1500  # Valeur par défaut si non calculée
    
    if run_keyword_variants:
        with st.spinner("🔁 Recherche de formulations alternatives..."):
            keyword_variants = get_keyword_variants(keyword)
    else:
        keyword_variants = []


    if run_google_ads_data and keyword_variants:
        with st.spinner("📊 Récupération des volumes de recherche Google Ads..."):
            keyword_data = get_google_ads_metrics(keyword_variants)

        import pandas as pd
        df_keywords = pd.DataFrame(keyword_data)
        df_keywords = df_keywords[["Mot-clé", "Volume mensuel"]]
        df_keywords = df_keywords[df_keywords["Volume mensuel"] != 0]
        df_keywords = df_keywords[df_keywords["Volume mensuel"] != "Erreur"]
        df_keywords = df_keywords.sort_values(by="Volume mensuel", ascending=False)

        st.markdown("### 📈 Volumes de recherche des formulations")
        st.dataframe(df_keywords, use_container_width=True)

   
    if run_community_suggestions:
        # 🙏 Suggestions de communautés
        st.markdown("---")
        st.markdown("### 🙏 Suggestions de communautés à mettre en avant")

        with st.spinner("🔍 Recherche des meilleures communautés..."):

            import numpy as np
            import json
            import os
            import pandas as pd
            from sklearn.metrics.pairwise import cosine_similarity
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            SHEET_ID = "1HWgw3qhjGxaFE1gDFwymFHcPodt88hzXYvk1YPxLxWw"  # ⬅️ Remplace par l’ID de ton Google Sheet
            SHEET_RANGE = "Question 7352!A2:D"
            OPENAI_EMBED_MODEL = "text-embedding-3-small"
            TOP_SEMANTIC = 20
            TOP_FINAL = 10

            def get_embedding(text):
                if "embedding_cache" not in st.session_state:
                    st.session_state.embedding_cache = {}

                if text in st.session_state.embedding_cache:
                    return st.session_state.embedding_cache[text]

                response = client.embeddings.create(
                    model=OPENAI_EMBED_MODEL,
                    input=[text]
                )
                embedding = response.data[0].embedding
                st.session_state.embedding_cache[text] = embedding
                return embedding

            def evaluer_accroche_chatgpt(titre):
                try:
                    prompt = f"""
        Tu es un expert en marketing chrétien.

        Note ce titre sur 5 pour sa capacité à faire cliquer un internaute catholique sur le site Hozana.

        Critères :
        - ✔️ Titre court (3–8 mots)
        - ✔️ Clarté immédiate
        - ✔️ Appel à l'action (implicite ou explicite)
        - ✔️ Impact émotionnel
        - ❌ Les titres vagues, trop longs ou peu engageants doivent recevoir une note basse.

        Règles :
        - 5 = exceptionnel
        - 4 = très bon
        - 3 = correct
        - 2 ou moins = faible

        Titre : « {titre} »

        Réponds uniquement par un chiffre entre 1 et 5, sans commentaire.
        """
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    contenu = response.choices[0].message.content.strip()
                    return min(max(float(contenu.replace(",", ".")), 0), 5)
                except Exception:
                    return 2.5

            def suggest_ctas(article_title, lang="fr"):
                credentials_dict = st.secrets["gcp_service_account"]
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_dict,
                    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
                )
                service = build("sheets", "v4", credentials=credentials)
                result = service.spreadsheets().values().get(
                    spreadsheetId=SHEET_ID, range=SHEET_RANGE
                ).execute()
                values = result.get("values", [])
                df = pd.DataFrame(values, columns=["Lang", "Community ID", "Name"])
                df = df[df["Lang"] == lang].copy()

                df["embedding"] = df["Name"].apply(get_embedding)
                df["embedding_vec"] = df["embedding"].apply(np.array)
                article_embedding = get_embedding(article_title)

                similarities = cosine_similarity([article_embedding], np.vstack(df["embedding_vec"].values))[0]
                df["similarity"] = similarities

                top_df = df.sort_values("similarity", ascending=False).head(TOP_SEMANTIC).copy()
                top_df["accroche_score"] = top_df["Name"].apply(evaluer_accroche_chatgpt)
                top_df["accroche_score_norm"] = top_df["accroche_score"] / 5

                top_df["score_total"] = 0.8 * top_df["similarity"] + 0.2 * top_df["accroche_score_norm"]
                final_df = top_df.sort_values("score_total", ascending=False).head(TOP_FINAL)

                if final_df["score_total"].max() < 0.6:
                    default_communities = {
                        "fr": ("7714", "NEUVAINE IRRÉSISTIBLE de Padre Pio pour les causes difficiles"),
                        "en": ("7667", "A Prayer A Day"),
                        "es": ("8601", "Novena a la Virgen de Guadalupe"),
                        "pt": ("10222", "Novena de São Padre Pio"),
                    }
                    if lang in default_communities:
                        default_id, default_name = default_communities[lang]
                        default_row = pd.DataFrame([{
                            "Community ID": default_id,
                            "Name": default_name,
                            "similarity": 0,
                            "accroche_score": 5,
                            "score_total": 1.0
                        }])
                        final_df = pd.concat([default_row, final_df], ignore_index=True)

                return final_df[["Community ID", "Name", "similarity", "accroche_score", "score_total"]].head(TOP_FINAL)

            df_ctas = suggest_ctas(keyword, lang="fr")

            st.markdown("#### 🔗 Communautés recommandées")
            for _, row in df_ctas.iterrows():
                st.markdown(f"- [{row['Name']}](https://hozana.org/communaute/{row['Community ID']})")

            st.dataframe(df_ctas, use_container_width=True)




    
    with st.spinner("⏳ Génération du plan SEO..."):
        plan_text = generate_plan(keyword, top_n, median_words)
        sections = extract_sections(plan_text)

    st.success("✅ Plan généré.")
    st.markdown("### 📄 Plan proposé")
    parse_and_display_plan(plan_text)


    st.markdown("---")
    
    section_bullets = {}
    
    for item in sections:
        if item[0] == "H2":
            level, title = item
            parent = None
        else:
            level, title, parent = item

        show_bullets = False

        if title.lower().startswith("introduction"):
            show_bullets = True
        elif level == "H3" and parent.lower() != "introduction":
            show_bullets = True

        if show_bullets:
            st.markdown(f"""
    <div style='border: 1px solid #ccc; border-radius: 8px; padding: 10px 15px; background-color: #f9f9f9; margin-bottom: 15px;'>
    <b>📌 {title}</b><br>
    </div>
    """, unsafe_allow_html=True)

            with st.spinner("📖 Enrichissement doctrinal avec Magisterium..."):
                bullets, texte_brut, citations = interroger_magisterium_contenu(title, keyword)

                if not bullets:
                    st.warning("❗ Aucun contenu généré pour cette section.")
                else:
                    for b in bullets:
                        propre = re.sub(r"^[-•◦\s]+", "", b.strip())
                        st.markdown(f"- {propre}")
                    section_bullets[title] = bullets

                    if citations:
                        st.markdown("#### 📂 Sources brutes citées")
                        for citation in citations:
                            st.markdown(f"- {citation.strip()}")

        else:
            st.markdown(f"## 📌 {title}")


    st.markdown("---")
    st.success(f"💰 Coût estimé OpenAI : **{estimate_cost(total_tokens_used)} $** ({total_tokens_used} tokens)")

    # 🔗 Recommandations de liens internes avec phrases d'ancrage (GPT-3.5)
    st.markdown("### 🔗 Liens recommandés à insérer dans l’article")

    with st.spinner("🧠 Analyse sémantique et recommandations..."):

        import os
        import json

        # 🔹 Chargement ou initialisation du cache d'embeddings
        EMBED_CACHE_FILE = "cache_embeddings.json"
        if os.path.exists(EMBED_CACHE_FILE):
            with open(EMBED_CACHE_FILE, "r") as f:
                st.session_state.embedding_cache = json.load(f)
        else:
            st.session_state.embedding_cache = {}

        def get_embedding(text):
            if text in st.session_state.embedding_cache:
                return st.session_state.embedding_cache[text]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=[text]
            )
            embedding = response.data[0].embedding
            st.session_state.embedding_cache[text] = embedding
            with open(EMBED_CACHE_FILE, "w") as f:
                json.dump(st.session_state.embedding_cache, f)
            return embedding

        # 🔹 Chargement des articles existants (Question 7364)
        SHEET_ID = "1HWgw3qhjGxaFE1gDFwymFHcPodt88hzXYvk1YPxLxWw"
        SHEET_RANGE = "Question 7364!A2:B"  # A = URL, B = Title

        credentials_dict = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        service = build("sheets", "v4", credentials=credentials)
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID, range=SHEET_RANGE
        ).execute()
        values = result.get("values", [])
        df_articles = pd.DataFrame(values, columns=["URL", "Title"]).dropna()

        df_articles["embedding"] = df_articles["Title"].apply(get_embedding)
        df_articles["embedding_vec"] = df_articles["embedding"].apply(np.array)

        # 🔹 H3 uniquement pour l’ancrage
        ancrages = [titre for level, *rest in sections if level == "H3" for titre in rest]

        suggestions = []

        for titre in ancrages:
            emb_titre = get_embedding(titre)
            sims = cosine_similarity([emb_titre], np.vstack(df_articles["embedding_vec"].values))[0]
            df_articles["similarity"] = sims
            best_row = df_articles.sort_values("similarity", ascending=False).iloc[0]

            suggestions.append({
                "Titre du lien": best_row["Title"],
                "URL": best_row["URL"],
                "À insérer après": titre,
                "Score de similarité": round(best_row["similarity"], 3)
            })

        # 🔹 Sélection des 15 meilleures suggestions
        top_suggestions = sorted(suggestions, key=lambda x: x["Score de similarité"], reverse=True)[:15]

        # 🔹 Génération des phrases d’ancrage (GPT-3.5-turbo)
        for s in top_suggestions:
            prompt = f"""
    Tu es un rédacteur web catholique.

    Insère naturellement le lien suivant dans une phrase fluide, courte (max 25 mots), destinée à un article sur « {keyword} ».

    Titre du lien : {s['Titre du lien']}
    URL : {s['URL']}
    Contexte d’insertion : après la section « {s['À insérer après']} »

    Utilise une **ancre naturelle**, pas un copier-coller du titre. Réponds uniquement par une phrase en markdown.
    """
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )
                phrase = response.choices[0].message.content.strip()
            except Exception:
                phrase = "—"

            s["Phrase avec ancre"] = phrase

        df_top = pd.DataFrame(top_suggestions)
        st.markdown("Voici les **15 liens internes** les plus pertinents à intégrer dans l’article, avec leur phrase d’ancrage :")
        st.dataframe(df_top[["Titre du lien", "URL", "À insérer après", "Phrase avec ancre"]], use_container_width=True)


    if st.button("📥 Télécharger la fiche SEO en PDF"):
        path = generate_pdf(keyword, plan_text, section_bullets)
        with open(path, "rb") as f:
            st.download_button(
                label="📄 Télécharger le PDF",
                data=f,
                file_name=f"fiche-seo-{keyword.replace(' ', '-')}.pdf",
                mime="application/pdf"
            )
