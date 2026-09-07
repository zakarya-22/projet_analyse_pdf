'''Backend et pipeline rag finale'''
import joblib
import pandas as pd
import time
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def load_system():
    """Charge les modèles ML et NLP en mémoire."""
    model_xgb = joblib.load('models/natixis_bankruptcy_xgb_model.pkl')
    colonnes_requises = joblib.load('models/colonnes_requises.pkl')
    
    llm = Ollama(model="mistral")
    embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-base")
    vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    
    return model_xgb, colonnes_requises, llm, vectorstore, cross_encoder

def calculate_risk(client_data, model_xgb, colonnes_requises):
    """Formate les données et calcule la probabilité de défaut."""
    df_client = client_data.reindex(columns=colonnes_requises, fill_value=0)
    prob_defaut = model_xgb.predict_proba(df_client)[0][1]
    return prob_defaut

def run_rag(question, llm, vectorstore, cross_encoder):
    """Exécute la recherche documentaire et la génération de réponse."""
    prompt_template = """
    Tu es un analyste expert du risque de crédit chez BNP Paribas.
    Utilise UNIQUEMENT le contexte suivant pour répondre à la question.
    Si la réponse n'est pas dans le contexte, dis exactement "Le document ne contient pas cette information".

    Contexte :
    {context}

    Question : {input}

    Réponse de l'expert :"""
    PROMPT = PromptTemplate.from_template(prompt_template)
    
    documents_candidats = vectorstore.similarity_search(question, k=15)
    paires = [[question, doc.page_content] for doc in documents_candidats]
    scores = cross_encoder.predict(paires)
    docs_tries = sorted(list(zip(documents_candidats, scores)), key=lambda x: x[1], reverse=True)
    meilleurs_docs = [doc for doc, score in docs_tries[:2]]
    
    texte_contexte = "\n\n".join([doc.page_content for doc in meilleurs_docs])
    prompt_final = PROMPT.format(context=texte_contexte, input=question)
    
    return llm.invoke(prompt_final)

# Permet de tester le script dans le terminal sans lancer Streamlit
if __name__ == '__main__':
    print("Test du backend en cours...")
    model, cols, llm_ia, vec_db, cross_enc = load_system()
    df_test = pd.read_csv('data/clients_test_db.csv')
    
    # Sélection précise de l'entreprise en danger (ID 1079) au lieu d'un échantillon aléatoire
    client_cible = df_test[df_test['Client_ID'] == 1079]
    
    if not client_cible.empty:
        print(f"👤 Évaluation du client n° {client_cible['Client_ID'].values[0]}")
        risque = calculate_risk(client_cible, model, cols)
        print(f"Probabilité de défaut : {risque * 100:.2f}%")
        
        if risque >= 0.50:
            print("⚠️ Risque élevé (>= 50%). Lancement RAG...")
            reponse = run_rag("Quels sont les critères stricts pour classer une entreprise en défaut de paiement et l'assigner à la strate 3 ?", llm_ia, vec_db, cross_enc)
            print("\n✅ RAPPORT FINAL :")
            print(reponse)
        else:
            print("✅ Dossier sain (< 50%).")
    else:
        print("Erreur : Client introuvable.")