from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

fichier_markdown = "data/bnp_risque_credit.md"
dossier_chroma = "data/chroma_db"

print("📖 Chargement et découpage sémantique du Markdown...")
with open(fichier_markdown, "r", encoding="utf-8") as f:
    markdown_text = f.read()

# Étape A : On découpe d'abord par structure Markdown (pages et tableaux)
headers_to_split_on = [
    ("##", "Page"),
    ("###", "Section_Tableau")
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_text)

# Étape B : On affine avec un découpage par taille pour les gros blocs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(md_header_splits)
print(f"✂️ Texte découpé en {len(chunks)} chunks sémantiques propres.")

# Étape C : Vectorisation (identique à ton code)
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-base")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=dossier_chroma
)
print(f"✅ Base ChromaDB mise à jour dans : {dossier_chroma}")