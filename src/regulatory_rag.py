from pathlib import Path
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import torch


_PROMPT = PromptTemplate.from_template(
    "Tu es un analyste expert du risque de credit.\n"
    "Reponds UNIQUEMENT en te basant sur le contexte fourni.\n"
    "Si la reponse est absente du contexte, dis exactement : "
    "\"Le document ne contient pas cette information.\"\n\n"
    "Contexte :\n{context}\n\n"
    "Question : {input}\n\n"
    "Analyse :"
)

_QUERY_TEMPLATES: dict[str, str] = {
    "dti": (
        "Le taux d'endettement du client est de {value:.1%}, ce qui depasse les seuils prudentiels. "
        "Quelles sont les obligations reglementaires concernant l'evaluation du surendettement "
        "et les regles de provisionnement associees ?"
    ),
    "delinq_2yrs": (
        "Le client presente {value:.0f} incident(s) de paiement sur 24 mois. "
        "Quels sont les criteres de classification en defaut "
        "et les mesures de suivi exigees lors d'un incident de paiement ?"
    ),
    "pub_rec": (
        "Le client a {value:.0f} incident(s) public(s) enregistre(s). "
        "Quelles sont les regles de provisionnement et les criteres de defaut applicables ?"
    ),
    "annual_inc": (
        "La capacite de remboursement du client est limitee (revenus annuels : {value:.0f}). "
        "Quelles sont les obligations reglementaires concernant l'evaluation "
        "de la capacite de remboursement avant l'octroi d'un credit ?"
    ),
    "_default": (
        "La probabilite de defaut du client est de {prob:.1%}. "
        "La variable `{feature}` (valeur : {value:.4f}) est le principal facteur de risque. "
        "Quelles sont les regles de classification en defaut et les mesures prudentielles applicables "
        "a ce profil de risque ?"
    ),
}


def build_rag_query(prob: float, top_feature: str, top_value: float) -> str:
    template = _QUERY_TEMPLATES.get(top_feature, _QUERY_TEMPLATES["_default"])
    return template.format(prob=prob, feature=top_feature, value=top_value)


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
        self.llm = None
        self._reranker: CrossEncoder | None = None

    def load(self) -> "RegulatoryRAG":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"u26a1 Initialisation du RAG sur : {device.upper()}")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model, 
            model_kwargs={'device': device}
        )
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
        )
        self.llm = Ollama(model=self.llm_model)
        self._reranker = CrossEncoder(self.reranker_model, device=device)
        return self

    def _rerank(self, query: str, candidates: list, top_k: int = 2) -> list:
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def query(self, question: str, k_candidates: int = 5) -> str:
        candidates = self.vectorstore.similarity_search(question, k=k_candidates)
        top_docs = self._rerank(question, candidates)
        context = "\n\n".join(doc.page_content for doc in top_docs)
        prompt = _PROMPT.format(context=context, input=question)
        return self.llm.invoke(prompt)




