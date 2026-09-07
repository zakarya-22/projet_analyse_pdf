'''Frontend et streamlit app'''
import streamlit as st
import pandas as pd
import time
from main_decision import load_system, calculate_risk, run_rag

st.set_page_config(page_title="Copilote Risque Crédit", layout="wide")
st.title("🏦 Tableau de Bord - Analyse du Risque")

# Cache pour charger les modèles une seule fois au démarrage
@st.cache_resource
def init_backend():
    return load_system()

@st.cache_data
def load_data():
    return pd.read_csv('data/clients_test_db.csv')

# Initialisation
try:
    model_xgb, colonnes_requises, llm, vectorstore, cross_encoder = init_backend()
    df_test = load_data()
except Exception as e:
    st.error(f"Erreur de chargement du système : {e}")
    st.stop()

st.sidebar.header("📁 Base de données")
liste_clients = df_test['Client_ID'].tolist()
client_id_choisi = st.sidebar.selectbox("Sélectionnez le client (ID) :", liste_clients)

client_cible = df_test[df_test['Client_ID'] == client_id_choisi].copy()
st.subheader("Données financières du client")
st.dataframe(client_cible, use_container_width=True)

st.write("---")

if st.button("🚀 Lancer l'Audit de Risque", type="primary"):
    # Appel de la fonction de calcul du fichier main_decision.py
    prob_defaut = calculate_risk(client_cible, model_xgb, colonnes_requises)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Score XGBoost (Probabilité de Défaut)", value=f"{prob_defaut * 100:.2f}%")
        
    if prob_defaut >= 0.50:
        with col2:
            st.error("⚠️ ALERTE CRITIQUE : Seuil dépassé (>= 50%)")
            
        st.subheader("📑 Rapport Réglementaire IA")
        question_ia = "Quels sont les critères stricts pour classer une entreprise en défaut de paiement et l'assigner à la strate 3 ?"
        
        with st.spinner("Analyse du rapport BNP Paribas en cours..."):
            debut = time.time()
            # Appel de la fonction RAG du fichier main_decision.py
            rapport_legal = run_rag(question_ia, llm, vectorstore, cross_encoder)
            fin = time.time()
            
        st.info(rapport_legal)
        st.caption(f"⏱️ Rapport généré en {fin - debut:.2f} secondes")
        
    else:
        with col2:
            st.success("✅ Dossier sain (< 50%). Aucune alerte levée.")