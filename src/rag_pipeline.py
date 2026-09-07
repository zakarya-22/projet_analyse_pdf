'''Banc test RAG'''
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder  # L'outil mathématique natif
import time

# ==========================================
# 1. INITIALISATION DES OUTILS
# ==========================================
print("🤖 Chargement du modèle Mistral...")
llm = Ollama(model="mistral")

print("📚 Connexion à la base documentaire ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-base")
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

print("🧠 Chargement du Cross-Encoder natif (BAAI/bge-reranker-base)...")
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

# ==========================================
# 2. LE PROMPT
# ==========================================
prompt_template = """
Tu es un analyste expert du risque de crédit chez BNP Paribas.
Utilise UNIQUEMENT le contexte suivant (qui contient du texte narratif et des tableaux financiers au format Markdown) pour répondre à la question.
Analyse aussi bien les paragraphes textuels que les données chiffrées des tableaux si elles répondent à la question.
Si la réponse n'est ni dans le texte ni dans les tableaux fournis, dis exactement "Le document ne contient pas cette information".

Contexte :
{context}

Question : {input}

Réponse de l'expert :"""

PROMPT = PromptTemplate.from_template(prompt_template)

# ==========================================
# 3. BOUCLE INTERACTIVE DU TERMINAL
# ==========================================
print("\n" + "="*50)
print("✅ Pipeline RAG Hybride Prêt. Tapez 'q' pour quitter.")
print("="*50 + "\n")

while True:
    question = input("🔍 Posez votre question : ")
    
    if question.lower() == 'q':
        print("Arrêt du pipeline.")
        break
        
    start_time = time.time()

    # --- ÉTAPE A : Le filet large (ChromaDB) ---
    print("⏳ 1/3 - Recherche vectorielle des 15 meilleurs extraits...")
    documents_candidats = vectorstore.similarity_search(question, k=15)

    # --- ÉTAPE B : Le Re-ranking (Manuel et stable) ---
    print("⏳ 2/3 - Re-ranking chirurgical par le Cross-Encoder...")
    # On prépare les données pour le modèle : [[Question, Texte1], [Question, Texte2]...]
    paires_a_noter = [[question, doc.page_content] for doc in documents_candidats]
    
    # Le modèle donne un score de pertinence à chaque paire
    scores = cross_encoder.predict(paires_a_noter)
    
    # On associe chaque document à son score et on trie du meilleur au pire
    docs_avec_scores = list(zip(documents_candidats, scores))
    docs_tries = sorted(docs_avec_scores, key=lambda x: x[1], reverse=True)
    
    # On ne garde que le contenu des 2 meilleurs documents
    meilleurs_docs = [doc for doc, score in docs_tries[:2]]

    # --- ÉTAPE C : Génération ---
    print(f"⏳ 3/3 - Rédaction par Mistral (basée sur les 2 meilleurs extraits)...\n")
    texte_contexte = "\n\n".join([doc.page_content for doc in meilleurs_docs])
    prompt_final = PROMPT.format(context=texte_contexte, input=question)

    reponse = llm.invoke(prompt_final)

    temps_ecoule = time.time() - start_time

    print("✅ RAPPORT FINAL :\n")
    print(reponse)
    print(f"\n⏱️ Temps d'exécution total : {temps_ecoule:.2f} secondes\n")
    print("-" * 50 + "\n")