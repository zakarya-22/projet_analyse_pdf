from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
import torch
import warnings
warnings.filterwarnings("ignore")

_PROMPT = PromptTemplate.from_template(
    "Tu es un strict copilote de conformite bancaire. "
    "N'utilise AUCUNE de tes connaissances generales.\n"
    "Reponds UNIQUEMENT a partir de la reglementation fournie ci-dessous.\n"
    "Si l'information est absente, dis 'Absence de directive dans le document BNP.'\n"
    "Sois extremement concis. Fais 3 bullet points maximum. Pas d'introduction.\n\n"
    "REGLEMENTATION BNP :\n{context}\n\n"
    "PROFIL CLIENT : {input}\n\n"
    "AVIS ET MESURES PRUDENTIELLES :"
)

class RegulatoryRAG:
    def __init__(
        self,
        db_path: str | Path,
        embedding_model: str = "dangvantuan/sentence-camembert-base",
        llm_model: str = "mistral",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.db_path = str(Path(db_path))
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.reranker_model = reranker_model
        self.vectorstore = None
        self.bm25 = None
        self.llm = None
        self._reranker = None

    def load(self, docs_for_bm25=None) -> "RegulatoryRAG":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': device}
        )
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
        )
        # BM25 : Si on a les chunks, on initialise le BM25, sinon on fait sans
        if docs_for_bm25:
            self.bm25 = BM25Retriever.from_texts(docs_for_bm25)
            self.bm25.k = 3

        self.llm = Ollama(model=self.llm_model)
        self._reranker = CrossEncoder(self.reranker_model, device=device)
        return self

    def _rerank(self, query: str, candidates: list, top_k: int = 2) -> list:
        if not candidates: return []
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def stream_query(self, prob: float, top_features: list, top_vals: list):
        question = f"Le client a un risque de defaut de {prob:.2%}. Les 3 facteurs critiques sont '{top_features[0]}' ({top_vals[0]:.2f}), '{top_features[1]}' ({top_vals[1]:.2f}) et '{top_features[2]}' ({top_vals[2]:.2f}). Quelles sont les regles applicables ?"
        
        # Hybrid Search si BM25 est present, sinon Vectoriel pur
        if self.bm25:
            retriever = EnsembleRetriever(
                retrievers=[self.bm25, self.vectorstore.as_retriever(search_kwargs={"k": 4})],
                weights=[0.3, 0.7]
            )
            candidates = retriever.invoke(question)
        else:
            candidates = self.vectorstore.similarity_search(question, k=5)
            
        top_docs = self._rerank(question, candidates)
        context = "\n\n".join(doc.page_content for doc in top_docs)
        prompt = _PROMPT.format(context=context, input=question)
        
        # On utilise stream() pour l'UX temps reel
        for chunk in self.llm.stream(prompt):
            yield chunk


