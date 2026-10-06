import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from credit_scorer import CreditScorer
from regulatory_rag import RegulatoryRAG
from document_extractor import DocumentExtractor

st.set_page_config(page_title="Copilote Risque Credit", layout="wide")
st.title("🏦 Copilote de Risque de Credit & Conformite")

THRESHOLD = 0.06

@st.cache_resource
def load_models():
    scorer = CreditScorer("models/natixis_bankruptcy_xgb_model.pkl", "models/colonnes_requises.pkl").load()
    bm25_path = "data/chroma_db/bm25_docs.json"
    bm25_docs = None
    if os.path.exists(bm25_path):
        with open(bm25_path, "r", encoding="utf-8") as f:
            bm25_docs = json.load(f)
            
    rag = RegulatoryRAG("data/chroma_db").load(docs_for_bm25=bm25_docs)
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
    colors = ['red' if val > 0 else 'blue' for val in sorted_vals]
    ax.barh(sorted_features, sorted_vals, color=colors)
    ax.set_xlabel("Impact sur le modele (Valeur SHAP)")
    ax.set_title("Explication des variables")
    st.pyplot(fig)

tab1, tab2 = st.tabs(["📄 Nouveau Dossier (Upload)", "📊 Portefeuille Existant"])

with tab1:
    st.markdown("### Analyse d'un nouveau Bilan Financier")
    uploaded_file = st.file_uploader("Chargez le bilan financier du client (TXT ou rapport texte brut)", type=["txt", "md"])
    text_input = st.text_area("OU copiez-collez le texte du client directement ici :", height=150)
    
    text_content = ""
    if uploaded_file is not None:
        text_content = uploaded_file.read().decode("utf-8")
    elif text_input:
        text_content = text_input
        
    if text_content:
        if st.button("Lancer l'Audit Complet"):
            with st.spinner("🤖 Etape 1: Extraction des donnees comptables (LLM)..."):
                try:
                    features_dict = extractor.extract(text_content)
                    st.success("Extraction reussie !")
                    st.json(features_dict)
                except Exception as e:
                    st.error("Erreur d'extraction.")
                    st.stop()
            
            with st.spinner("🧮 Etape 2: Calcul du Score XGBoost et SHAP..."):
                df_client = pd.DataFrame([features_dict])
                prob = float(scorer.predict(df_client))
                explanation = scorer.explain(df_client)
                top_features = explanation["top_3_features"]
                top_vals = explanation["top_3_values"]
                
                if prob > THRESHOLD:
                    st.error(f"🚨 RISQUE DE DEFAUT : {prob:.2%} (Seuil: {THRESHOLD:.2%})")
                else:
                    st.success(f"✅ DOSSIER SAIN : {prob:.2%} (Seuil: {THRESHOLD:.2%})")
                    
                st.info(f"🔍 Top 3 Facteurs de risque :\n1. **{top_features[0]}** ({top_vals[0]:.4f})\n2. **{top_features[1]}** ({top_vals[1]:.4f})\n3. **{top_features[2]}** ({top_vals[2]:.4f})")
                plot_shap(explanation)
            
            with st.spinner("⚖️ Etape 3: Audit de conformite (Recherche Hybrid + Streaming)..."):
                st.markdown("### Avis de Conformite Reglementaire")
                st.write_stream(rag.stream_query(prob, top_features, top_vals))

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
        
        st.write(f"Nombre de clients en alerte : {len([p for p in all_probs if p > THRESHOLD])} / {len(df)}")
        st.dataframe(df_display.style.map(lambda x: "background-color: #ffcccc" if x == "🚨 ALERTE" else "background-color: #ccffcc", subset=["Statut"]), use_container_width=True)
        
        st.divider()
        st.markdown("### 🔎 Lancer un Audit de Conformite (RAG)")
        client_id = st.selectbox("Choisir un client a auditer en detail :", df.index)
        
        if st.button("Generer le rapport d'Audit"):
            client_data = df.loc[[client_id]]
            prob = float(scorer.predict(client_data))
            explanation = scorer.explain(client_data)
            top_features = explanation["top_3_features"]
            top_vals = explanation["top_3_values"]
            
            if prob > THRESHOLD:
                st.error(f"🚨 RISQUE DE DEFAUT : {prob:.2%} (Seuil: {THRESHOLD:.2%})")
            else:
                st.success(f"✅ DOSSIER SAIN : {prob:.2%} (Seuil: {THRESHOLD:.2%})")
                
            st.info(f"🔍 Top 3 Facteurs de risque :\n1. **{top_features[0]}** ({top_vals[0]:.4f})\n2. **{top_features[1]}** ({top_vals[1]:.4f})\n3. **{top_features[2]}** ({top_vals[2]:.4f})")
            plot_shap(explanation)
            
            st.markdown("### Avis Reglementaire")
            st.write_stream(rag.stream_query(prob, top_features, top_vals))
    except FileNotFoundError:
        st.warning("Veuillez regenerer le fichier clients_test_db.csv avec vos nouvelles colonnes.")