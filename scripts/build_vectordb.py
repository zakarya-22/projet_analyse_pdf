import hashlib
import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent.parent
MD_PATH = ROOT / "data" / "bnp_risque_credit.md"
DB_PATH = ROOT / "data" / "chroma_v2"
BM25_PATH = ROOT / "data" / "chroma_v2" / "bm25_docs.json"
EMBEDDING_MODEL = "dangvantuan/sentence-camembert-base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def build(md_path: Path, db_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "Page"), ("###", "Section")]
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [
        c for c in splitter.split_documents(md_splitter.split_text(text)) if "Page" in c.metadata
    ]
    for chunk in chunks:
        chunk.metadata["source"] = md_path.name
        chunk.metadata["chunk_id"] = hashlib.sha256(
            (str(chunk.metadata) + chunk.page_content).encode()
        ).hexdigest()[:20]
    print(f"{len(chunks)} chunks generes")

    # 1. Sauvegarde pour Vector Search (Chroma)
    import torch

    torch.set_num_threads(2)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, encode_kwargs={"batch_size": 4}, show_progress=True
    )
    db_path.mkdir(parents=True, exist_ok=True)
    store = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
    # Upsert stable IDs; remove obsolete chunks after successful embedding.
    existing = set(store.get()["ids"])
    ids = [c.metadata["chunk_id"] for c in chunks]
    store.add_documents(chunks, ids=ids)
    obsolete = existing - set(ids)
    if obsolete:
        store.delete(ids=list(obsolete))

    # 2. Sauvegarde pour BM25 (Hybrid Search)
    bm25_texts = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    (db_path / "bm25_docs.json").write_text(
        json.dumps(bm25_texts, ensure_ascii=False), encoding="utf-8"
    )

    print(f"ChromaDB et Dictionnaire BM25 mis a jour : {db_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    build(MD_PATH, args.db)
