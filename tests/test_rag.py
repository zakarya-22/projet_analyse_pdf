from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from scripts.evaluate_rag import abstained, load_questions, summarize
from src.regulatory_rag import ABSTENTION, RegulatoryRAG, fuse_rankings, profile_question


def doc(name, page):
    return Document(page_content=name, metadata={"chunk_id": name, "Page": str(page)})


def fixture_rag():
    a, b, c = doc("A", 426), doc("B", 427), doc("C", 428)
    rag = RegulatoryRAG("unused")
    rag.vectorstore = SimpleNamespace(similarity_search=lambda q, k: [a, b])
    rag.bm25 = SimpleNamespace(invoke=lambda q: [c, b])
    rag._reranker = SimpleNamespace(predict=lambda pairs: [1 if p[1] != "C" else 9 for p in pairs])
    rag.llm = SimpleNamespace(stream=lambda prompt: iter(["Avis ", "[S1]"]))
    return rag


def test_fusion_deduplicates_and_preserves_metadata():
    a, b = doc("A", 426), doc("B", 427)
    result = fuse_rankings([[b, b, a], [a, b]])
    assert len(result) == 2
    assert result[0].metadata["Page"] == "426"


def test_modes_and_reranker_order():
    rag = fixture_rag()
    assert [d.page_content for d in rag.retrieve("q", "vector")] == ["A", "B"]
    assert rag.retrieve("q", "hybrid")[0].page_content == "B"
    assert rag.retrieve("q", "hybrid_rerank")[0].page_content == "C"
    with pytest.raises(ValueError):
        rag.retrieve("q", "unknown")
    rag.bm25 = None
    with pytest.raises(ValueError, match="BM25"):
        rag.retrieve("q", "hybrid")


def test_citations_correspond_to_returned_passages():
    rag = fixture_rag()
    docs = rag.retrieve("q")
    sources = rag.sources(docs)
    prompt = rag.prompt("q", docs)
    assert sources[0]["page_pdf"] == "428"
    assert sources[0]["excerpt"] == "C"
    assert "[S1] Page PDF 428" in prompt
    assert rag.answer("q", docs) == "Avis [S1]"
    assert rag.answer("q", []) == ABSTENTION


def test_profile_carries_shap_direction_even_when_only_one_factor():
    explanation = {
        "risk_factors": [],
        "protective_factors": [{"feature": " Cash/Total Assets", "value": 0.2, "shap": -1.2}],
    }
    question = profile_question(0.06, explanation)
    assert "Facteurs aggravants : aucun identifié" in question
    assert "Facteurs protecteurs : Cash/Total Assets=0.2" in question


def test_dataset_is_traceable_and_contains_unanswerable_cases():
    rows = load_questions("evaluation/questions.jsonl")
    assert len(rows) == 40
    assert sum(r["answerable"] for r in rows) == 32
    assert all(r["reference_contexts"] for r in rows if r["answerable"])


def test_summary_does_not_hide_missing_scores():
    summary = summarize(
        [
            {"mode": "vector", "faithfulness": 0.5},
            {"mode": "vector", "faithfulness": None, "error": "timeout"},
        ]
    )["vector"]
    assert summary["faithfulness"] == {"mean": 0.5, "n_valid": 1}
    assert summary["generation_errors"] == 1
    assert abstained(ABSTENTION)
    assert not abstained("Je recommande un crédit. " + ABSTENTION)
