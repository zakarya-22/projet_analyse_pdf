"""Reproducible, resumable benchmark. No cloud API or model training."""

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

from src.regulatory_rag import ABSTENTION, MODES, RegulatoryRAG

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_questions(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Identifiants de questions dupliqués.")
    corpus = (ROOT / "data/bnp_risque_credit.md").read_text(encoding="utf-8")
    parts = re.split(r"^## Page (\d+)\s*$", corpus, flags=re.M)
    pages = {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}
    for row in rows:
        if row["answerable"] and (
            not row["reference_pages"]
            or not any(row["evidence_anchor"] in pages.get(p, "") for p in row["reference_pages"])
        ):
            raise ValueError(f"Référence non vérifiable : {row['id']}")
    return rows


def abstained(response):
    return response.strip().strip('"').rstrip(".").casefold() == ABSTENTION.rstrip(".").casefold()


def summarize(rows):
    summaries = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        if not selected:
            continue
        summary = {"n": len(selected), "generation_errors": sum("error" in r for r in selected)}
        for metric in (
            "page_hit_at_2",
            "faithfulness",
            "context_precision",
            "context_recall",
            "abstention_correct",
            "false_abstention",
            "citation_id_validity",
            "retrieval_seconds",
            "generation_seconds",
        ):
            vals = [
                r[metric]
                for r in selected
                if isinstance(r.get(metric), (int, float)) and math.isfinite(r[metric])
            ]
            summary[metric] = {
                "mean": statistics.mean(vals) if vals else None,
                "n_valid": len(vals),
            }
        summaries[mode] = summary
    return summaries


async def score_row(row, judge, timeout, compact=False):
    # Legacy API intentionally pinned to Ragas 0.4.3 for local ChatOllama compatibility.
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

    sample = SingleTurnSample(
        user_input=row["question"],
        response=row["response"],
        reference=row["reference"],
        retrieved_contexts=row["retrieved_contexts"],
    )
    metrics = [
        ("context_precision", LLMContextPrecisionWithReference(llm=judge)),
        ("context_recall", LLMContextRecall(llm=judge)),
    ]
    # Unanswerable cases are assessed separately; abstentions are not faithful answers.
    if not abstained(row["response"]):
        metrics.append(("faithfulness", Faithfulness(llm=judge)))
    for name, metric in metrics:
        if compact:
            for prompt in metric.get_prompts().values():
                prompt.examples = []
                prompt.instruction += " Keep each explanation concise (one sentence)."
        try:
            score = await asyncio.wait_for(metric.single_turn_ascore(sample), timeout)
            if not math.isfinite(score):
                raise ValueError("Le juge n'a pas produit de score fini.")
            row[name] = score
            print(f"{row['id']} {row['mode']} {name}={score:.3f}", flush=True)
        except Exception as exc:
            row[name] = None
            print(f"{row['id']} {row['mode']} {name}: {type(exc).__name__}", flush=True)
            row.setdefault("metric_errors", {})[name] = f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/questions.jsonl")
    parser.add_argument("--db", type=Path, default=ROOT / "data/chroma_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/rag")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--judge-model", default="mistral")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--compact-judge", action="store_true")
    args = parser.parse_args()
    questions = load_questions(args.dataset)
    if args.ids:
        questions = [r for r in questions if r["id"] in args.ids]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("Aucune question sélectionnée.")
    args.output.mkdir(parents=True, exist_ok=True)
    rag = RegulatoryRAG(args.db, llm_model=args.model, base_url=args.base_url).load()
    stored = rag.vectorstore.get(include=["documents", "metadatas"])
    index_hash = hashlib.sha256(
        json.dumps(
            sorted(zip(stored["ids"], stored["documents"], stored["metadatas"], strict=True)),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    model_tags = None
    if not args.retrieval_only:
        with urlopen(args.base_url + "/api/tags", timeout=10) as response:
            model_tags = json.load(response)["models"]
    from huggingface_hub import try_to_load_from_cache

    revisions = {}
    for model_name in (rag.embedding_model, rag.reranker_model):
        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str):
            parts = Path(cached).parts
            revisions[model_name] = (
                parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
            )
        else:
            revisions[model_name] = None
    config = dict(
        dataset_sha256=sha(args.dataset),
        corpus_sha256=sha(ROOT / "data/bnp_risque_credit.md"),
        code_sha256=sha(__file__),
        rag_sha256=sha(ROOT / "src/regulatory_rag.py"),
        lock_sha256=sha(ROOT / "uv.lock"),
        index_sha256=index_hash,
        question_ids=[q["id"] for q in questions],
        model=args.model,
        judge=args.judge_model,
        compact_judge=args.compact_judge,
        base_url=args.base_url,
        ollama_models=model_tags,
        retrieval_only=args.retrieval_only,
        skip_ragas=args.skip_ragas,
        huggingface_revisions=revisions,
        embedding=rag.embedding_model,
        reranker=rag.reranker_model,
        top_k=2,
        seed=42,
        temperature=0,
        modes=MODES,
    )
    manifest_path = args.output / "manifest.json"
    # JSON roundtrip normalizes tuples before comparing persisted configuration.
    config = json.loads(json.dumps(config))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Configuration modifiée : choisissez un nouveau dossier --output.")
    else:
        manifest_path.write_text(
            json.dumps(
                dict(
                    config=config,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    platform=platform.platform(),
                    python=platform.python_version(),
                    packages={
                        p: importlib.metadata.version(p)
                        for p in ("ragas", "langchain-ollama", "sentence-transformers", "chromadb")
                    },
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    result_path = args.output / "results.jsonl"
    rows = (
        [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
        if result_path.exists()
        else []
    )
    completed = {(r["id"], r["mode"]) for r in rows}
    # Retrieve all candidates first, then release neural retrieval models before Ollama.
    rag.retrieve("procédures de surveillance du risque de crédit")
    prepared = []
    for question in questions:
        for mode in MODES:
            if (question["id"], mode) in completed:
                continue
            row = {**question, "mode": mode}
            started = time.perf_counter()
            docs = rag.retrieve(question["question"], mode=mode)
            row["retrieval_seconds"] = time.perf_counter() - started
            row["sources"] = rag.sources(docs)
            row["retrieved_contexts"] = [d.page_content for d in docs]
            if question["answerable"]:
                row["page_hit_at_2"] = float(
                    any(
                        str(d.metadata.get("Page", "")).removeprefix("Page ")
                        in question["reference_pages"]
                        for d in docs
                    )
                )
            prepared.append((row, docs))
    (args.output / "retrieval.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row, _ in prepared) + "\n",
        encoding="utf-8",
    )
    rag.vectorstore = rag.bm25 = rag._reranker = None
    gc.collect()
    print(
        f"Recherche terminée : {len(prepared)} paires ; modèles de recherche libérés.", flush=True
    )
    judge = None
    if not args.retrieval_only and not args.skip_ragas:
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig

        judge = LangchainLLMWrapper(
            ChatOllama(
                model=args.judge_model,
                base_url=args.base_url,
                temperature=0,
                seed=42,
                num_ctx=8192,
                num_predict=768 if args.compact_judge else 2048,
                client_kwargs={"timeout": args.timeout},
            ),
            run_config=RunConfig(timeout=args.timeout, max_retries=1, max_workers=1),
        )
    for row, docs in prepared:
        if not args.retrieval_only:
            started = time.perf_counter()
            try:
                row["response"] = rag.answer(row["question"], docs)
                row["generation_seconds"] = time.perf_counter() - started
                did_abstain = abstained(row["response"])
                key = "false_abstention" if row["answerable"] else "abstention_correct"
                row[key] = float(did_abstain)
                citations = re.findall(r"\[S(\d+)\]", row["response"])
                if not did_abstain:
                    row["citation_id_validity"] = float(
                        bool(citations) and all(1 <= int(c) <= len(docs) for c in citations)
                    )
                if judge and row["answerable"]:
                    asyncio.run(score_row(row, judge, args.timeout, args.compact_judge))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        with result_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (args.output / "summary.json").write_text(
            json.dumps(summarize(rows), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"{len(rows)}/{len(questions) * len(MODES)} {row['id']} {row['mode']}", flush=True)
        if "error" in row:
            raise SystemExit(
                "Génération échouée : résultats conservés, corriger le service avant de poursuivre."
            )
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    if any("error" in r or r.get("metric_errors") for r in rows):
        raise SystemExit("Évaluation partiellement échouée : consultez les erreurs et n_valid.")


if __name__ == "__main__":
    main()
