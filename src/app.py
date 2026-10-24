import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.credit_scorer import CreditScorer
from src.document_extractor import DocumentExtractor
from src.regulatory_rag import RegulatoryRAG, profile_question

st.set_page_config(page_title="Copilote Risque Credit", layout="wide")
st.title("🏦 Copilote de Risque de Credit & Conformite")

THRESHOLD = 0.06


@st.cache_resource
def load_models():
    scorer = CreditScorer(
        "models/natixis_bankruptcy_xgb_model.pkl", "models/colonnes_requises.pkl"
    ).load()
    rag = RegulatoryRAG("data/chroma_v2").load()
    extractor = DocumentExtractor()
    return scorer, rag, extractor


try:
    scorer, rag, extractor = load_models()
except Exception as e:
    st.error(f"Erreur de chargement des modeles : {e}")
    st.stop()


def plot_shap(explanation_dict):
    features = explanation_dict["feature_names"]
    shap_vals = explanation_dict["shap_values"]

    sorted_idx = np.argsort(np.abs(shap_vals))
    sorted_features = [features[i] for i in sorted_idx]
    sorted_vals = [shap_vals[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["red" if val > 0 else "blue" for val in sorted_vals]
    ax.barh(sorted_features, sorted_vals, color=colors)
    ax.set_xlabel("Impact sur le modele (Valeur SHAP)")
    ax.set_title("Explication des variables")
    st.pyplot(fig)


def render_factors(explanation):
    for key, label in (
        ("risk_factors", "Facteurs aggravants"),
        ("protective_factors", "Facteurs protecteurs"),
    ):
        st.markdown(f"**{label}**")
        for factor in explanation[key]:
            st.write(
                f"• {factor['feature'].strip()} : {factor['value']:.4g} "
                f"(SHAP {factor['shap']:+.4f})"
            )
        if not explanation[key]:
            st.caption("Aucun facteur identifié dans ce sens.")
    st.caption(
        "Contributions au score du modèle, pas des effets causaux ni des points de pourcentage."
    )


def render_audit(prob, explanation):
    question = profile_question(prob, explanation)
    try:
        docs = rag.retrieve(question)
        st.write_stream(rag.stream_answer(question, docs))
        with st.expander("Sources consultées — vérifier les affirmations dans les extraits"):
            for source in rag.sources(docs):
                st.markdown(
                    f"**[{source['id']}] {source['document']} — page PDF "
                    f"{source['page_pdf']} — {source['section']}**"
                )
                st.text(source["excerpt"])
            st.caption("La page PDF peut différer de la pagination imprimée du rapport.")
    except Exception as exc:
        st.error(f"Avis documentaire indisponible : {exc}")


tab1, tab2 = st.tabs(["📄 Nouveau Dossier (Upload)", "📊 Portefeuille Existant"])

with tab1:
    st.markdown("### Analyse d'un nouveau Bilan Financier")
    uploaded_file = st.file_uploader(
        "Chargez le bilan financier du client (TXT ou rapport texte brut)", type=["txt", "md"]
    )
    text_input = st.text_area("OU copiez-collez le texte du client directement ici :", height=150)

    text_content = ""
    if uploaded_file is not None:
        text_content = uploaded_file.read().decode("utf-8")
    elif text_input:
        text_content = text_input

    if text_content:
        compatible_units = st.checkbox(
            "Les indicateurs sont exprimés dans la même représentation que le dataset d'entraînement.",
            help="Le notebook utilise les valeurs du CSV Kaggle directement, sans scaler. "
            "La conversion d'un bilan brut vers ces valeurs n'est pas disponible. "
            "Si cette compatibilité n'est pas vérifiée, ne lancez pas le scoring.",
        )
        if st.button("Lancer l'Audit Complet", disabled=not compatible_units):
            with st.spinner("🤖 Etape 1: Extraction des donnees comptables (LLM)..."):
                try:
                    features_dict = extractor.extract(text_content)
                    st.success("Extraction reussie !")
                    st.json(features_dict)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            with st.spinner("🧮 Etape 2: Calcul du Score XGBoost et SHAP..."):
                df_client = pd.DataFrame([features_dict])
                prob = float(scorer.predict(df_client))
                explanation = scorer.explain(df_client)

                if prob > THRESHOLD:
                    st.error(f"🚨 RISQUE DE DEFAUT : {prob:.2%} (Seuil: {THRESHOLD:.2%})")
                else:
                    st.success(f"✅ DOSSIER SAIN : {prob:.2%} (Seuil: {THRESHOLD:.2%})")

                render_factors(explanation)
                plot_shap(explanation)

            with st.spinner("⚖️ Etape 3: Audit de conformite (Recherche Hybrid + Streaming)..."):
                st.markdown("### Avis de Conformite Reglementaire")
                render_audit(prob, explanation)

with tab2:
    st.markdown("### Vision Globale du Portefeuille")
    try:
        df = pd.read_csv("data/clients_test_db.csv")
        if "Client_ID" in df.columns:
            df = df.set_index("Client_ID")
        else:
            df.index = range(1001, 1001 + len(df))
            df.index.name = "Client_ID"

        all_probs = [float(scorer.predict(df.loc[[idx]])) for idx in df.index]
        df_display = df.copy()
        df_display["Risque de Defaut"] = all_probs
        df_display["Statut"] = ["🚨 ALERTE" if p > THRESHOLD else "✅ SAIN" for p in all_probs]

        st.write(
            f"Nombre de clients en alerte : {len([p for p in all_probs if p > THRESHOLD])} / {len(df)}"
        )
        st.dataframe(
            df_display.style.map(
                lambda x: (
                    "background-color: #ffcccc" if x == "🚨 ALERTE" else "background-color: #ccffcc"
                ),
                subset=["Statut"],
            ),
            use_container_width=True,
        )

        st.divider()
        st.markdown("### 🔎 Lancer un Audit de Conformite (RAG)")
        client_id = st.selectbox("Choisir un client a auditer en detail :", df.index)

        if st.button("Generer le rapport d'Audit"):
            client_data = df.loc[[client_id]]
            prob = float(scorer.predict(client_data))
            explanation = scorer.explain(client_data)

            if prob > THRESHOLD:
                st.error(f"🚨 RISQUE DE DEFAUT : {prob:.2%} (Seuil: {THRESHOLD:.2%})")
            else:
                st.success(f"✅ DOSSIER SAIN : {prob:.2%} (Seuil: {THRESHOLD:.2%})")

            render_factors(explanation)
            plot_shap(explanation)

            st.markdown("### Avis Reglementaire")
            render_audit(prob, explanation)
    except ValueError as exc:
        st.error(str(exc))
    except FileNotFoundError:
        st.warning("Veuillez regenerer le fichier clients_test_db.csv avec vos nouvelles colonnes.")
