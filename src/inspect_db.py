'''Banc test embedding et pipeline'''
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Connexion à la base
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-base")
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

# 2. On interroge la base sur la définition du défaut
question = "Comment la BNP définit-elle le risque de crédit ? ?"
documents_trouves = vectorstore.similarity_search(question, k=3)

# 3. On affiche ce que ChromaDB a trouvé mot pour mot
print(f"--- RÉSULTATS DE LA RECHERCHE POUR : '{question}' ---\n")
for i, doc in enumerate(documents_trouves):
    print(f"📌 [Extrait n°{i+1}] (Provenance approximative : Page {doc.metadata.get('page', 'Inconnue')})")
    print(doc.page_content) # On affiche les 400 premiers caractères de l'extrait
    print("-" * 50)