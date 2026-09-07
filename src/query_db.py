'''Banc test chroma db'''

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Chemins d'accès
dossier_chroma = "data/chroma_db"

print("⏳ Chargement du cerveau mathématique (Camembert)...")
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-base")

print("📂 Connexion à la base ChromaDB existante...")
# On pointe vers le dossier où la base a été sauvegardée
vectorstore = Chroma(persist_directory=dossier_chroma, embedding_function=embeddings)

# --- LA QUESTION ---
question = "Quels sont les critères stricts pour classer une entreprise en défaut de paiement ou créance douteuse ?"
print(f"\n🔍 Recherche sémantique en cours pour : '{question}'\n")

# --- LA RECHERCHE ---
# k=3 signifie qu'on veut extraire les 3 morceaux de texte les plus pertinents
resultats = vectorstore.similarity_search(question, k=3)

# --- L'AFFICHAGE BRUT (Sans IA générative) ---
print("📑 VOICI LES EXTRAITS BRUTS DU PDF :\n")
for i, doc in enumerate(resultats):
    print(f"--- EXTRAIT N°{i+1} ---")
    print(doc.page_content)
    print("-" * 60 + "\n")