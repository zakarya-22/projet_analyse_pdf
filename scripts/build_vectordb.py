"""
Construction de la base vectorielle ChromaDB depuis le fichier Markdown.
Usage : uv run python scripts/build_vectordb.py
"""
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

ROOT            = Path(__file__).parent.parent
MD_PATH         = ROOT / "data" / "bnp_risque_credit.md"
DB_PATH         = ROOT / "data" / "chroma_db"
EMBEDDING_MODEL = "dangvantuan/sentence-camembert-base"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200


def build(md_path: Path, db_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "Page"), ("###", "Section")]
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks   = splitter.split_documents(md_splitter.split_text(text))
    print(f"{len(chunks)} chunks generes")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(db_path))
    print(f"ChromaDB mis a jour : {db_path}")


if __name__ == "__main__":
    build(MD_PATH, DB_PATH)
