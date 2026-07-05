from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rag_mvp.pipeline import RagPipeline

REFUSAL_PATTERNS = (
    "kh\u00f4ng \u0111\u1ee7 th\u00f4ng tin",
    "kh\u00f4ng t\u00ecm th\u1ea5y",
    "kh\u00f4ng c\u00f3 th\u00f4ng tin",
    "khong du thong tin",
    "khong tim thay",
    "khong co thong tin",
    "i don't know",
    "not enough information",
    "answer was not found",
    "not found in the provided context",
    "not found in the context",
    "provided context does not contain",
    "context does not contain",
    "does not provide",
    "cannot provide",
    "cannot determine",
    "does not explicitly state",
    "not explicitly state",
)


@dataclass
class EvaluationRow:
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    latency_ms: float
    expected_behavior: str = "answer"
    required_citation: str = "yes"


def extract_citations(answer: str) -> set[int]:
    return {int(match) for match in re.findall(r"\[S(\d+)\]", answer)}


def is_refusal(answer: str) -> bool:
    normalized = answer.lower()
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def is_not_supported(answer: str) -> bool:
    normalized = answer.lower()
    return (
        is_refusal(answer)
        or "not found" in normalized
        or "not supported" in normalized
        or "does not support" in normalized
        or "does not show" in normalized
        or "does not suggest" in normalized
        or "does not conclusively state" in normalized
        or "cannot be concluded" in normalized
        or "do not explicitly confirm" in normalized
        or "not explicitly state" in normalized
        or "not identified" in normalized
    )


def false_refusal(answer: str, ground_truth: str, expected_behavior: str = "answer") -> int:
    refused = is_refusal(answer)
    answer_expected = expected_behavior in {"answer", "claim_verification"}
    answerable = bool(ground_truth.strip())
    return int(refused and answerable and answer_expected)


def refusal(answer: str) -> int:
    return int(is_refusal(answer))


def refusal_precision(answer: str, expected_behavior: str) -> float | None:
    if not is_refusal(answer):
        return None
    return float(expected_behavior in {"refuse", "state_not_supported"})


def correct_refusal_behavior(answer: str, expected_behavior: str) -> float | None:
    if expected_behavior == "refuse":
        return float(is_refusal(answer))
    if expected_behavior == "state_not_supported":
        return float(is_not_supported(answer))
    return None


def expected_behavior_accuracy(answer: str, expected_behavior: str) -> float:
    if expected_behavior == "refuse":
        return float(is_refusal(answer) or "not found" in answer.lower())
    if expected_behavior == "state_not_supported":
        return float(is_not_supported(answer))
    return 1.0


def citation_accuracy(answer: str, contexts: list[str]) -> float:
    citations = extract_citations(answer)
    if not citations:
        return 0.0
    valid = {idx for idx in citations if 1 <= idx <= len(contexts)}
    return len(valid) / len(citations)


def citation_strict_accuracy(answer: str, contexts: list[str], ground_truth: str) -> float:
    citations = extract_citations(answer)
    if not citations:
        return 0.0
    normalized_contexts = [ctx.lower() for ctx in contexts]
    key_terms = _keywords(ground_truth)
    if not key_terms:
        return citation_accuracy(answer, contexts)
    supported = 0
    for citation in citations:
        if 1 <= citation <= len(normalized_contexts):
            context = normalized_contexts[citation - 1]
            if any(term in context for term in key_terms):
                supported += 1
    return supported / len(citations)


def unsupported_claim_accuracy(answer: str, contexts: list[str]) -> float:
    claims = [sentence.strip() for sentence in re.split(r"[.!?\n]+", answer) if len(sentence.strip()) > 20]
    if not claims:
        return 1.0
    joined_context = " ".join(contexts).lower()
    supported = 0
    for claim in claims:
        terms = _keywords(claim)
        if not terms or any(term in joined_context for term in terms):
            supported += 1
    return supported / len(claims)


def evidence_hit_rate(contexts: list[str], ground_truth: str) -> float:
    key_terms = _keywords(ground_truth)
    if not key_terms:
        return 0.0
    normalized_contexts = [ctx.lower() for ctx in contexts]
    return float(any(any(term in context for term in key_terms) for context in normalized_contexts))


def evidence_mrr(contexts: list[str], ground_truth: str) -> float:
    key_terms = _keywords(ground_truth)
    if not key_terms:
        return 0.0
    for rank, context in enumerate((ctx.lower() for ctx in contexts), start=1):
        if any(term in context for term in key_terms):
            return 1 / rank
    return 0.0


def evaluate_custom(row: EvaluationRow) -> dict[str, float]:
    citations = extract_citations(row.answer)
    citation_required = str(row.required_citation).strip().lower() != "no"
    citation_score = citation_accuracy(row.answer, row.contexts)
    strict_citation_score = citation_strict_accuracy(row.answer, row.contexts, row.ground_truth)
    if not citation_required and not citations:
        citation_score = 1.0
        strict_citation_score = 1.0
    unsupported_score = unsupported_claim_accuracy(row.answer, row.contexts)

    return {
        "false_refusal": false_refusal(row.answer, row.ground_truth, row.expected_behavior),
        "refusal": refusal(row.answer),
        "refusal_precision": refusal_precision(row.answer, row.expected_behavior),
        "correct_refusal_behavior": correct_refusal_behavior(row.answer, row.expected_behavior),
        "expected_behavior_accuracy": expected_behavior_accuracy(row.answer, row.expected_behavior),
        "evidence_hit_rate": evidence_hit_rate(row.contexts, row.ground_truth),
        "mrr": evidence_mrr(row.contexts, row.ground_truth),
        "citation_accuracy": citation_score,
        "citation_strict_accuracy": strict_citation_score,
        "unsupported_claim_accuracy": unsupported_score,
        "hallucination_rate": 1 - unsupported_score,
        "latency_ms": row.latency_ms,
    }


def run_local_evaluation(
    pipeline: RagPipeline,
    dataset_path: Path,
    output_path: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, Any]] = []
    records = dataset.to_dict(orient="records")
    total = len(records)
    for index, item in enumerate(records, start=1):
        question = str(item["question"])
        if progress_callback:
            progress_callback(index, total, question)
        started = time.perf_counter()
        result = pipeline.answer(question)
        latency_ms = result.get("latency_ms") or round((time.perf_counter() - started) * 1000, 2)
        contexts = [ctx["text"] for ctx in result["contexts"]]
        eval_row = EvaluationRow(
            question=question,
            ground_truth=str(item.get("ground_truth", "")),
            answer=result["answer"],
            contexts=contexts,
            latency_ms=float(latency_ms),
            expected_behavior=str(item.get("expected_behavior", "answer")),
            required_citation=str(item.get("required_citation", "yes")),
        )
        metadata = {key: value for key, value in item.items() if key not in {"question", "ground_truth"}}
        rows.append(
            {
                **metadata,
                "question": eval_row.question,
                "ground_truth": eval_row.ground_truth,
                "answer": eval_row.answer,
                "contexts": json.dumps(contexts, ensure_ascii=False),
                **evaluate_custom(eval_row),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame


def run_ragas_core_evaluation(
    local_results_path: Path,
    output_path: Path,
    *,
    ollama_base_url: str,
    ollama_model: str,
    embedding_model: str,
    temperature: float,
    num_ctx: int | None = None,
    raise_exceptions: bool = False,
    timeout: int = 600,
    max_workers: int = 1,
    max_retries: int = 1,
    batch_size: int = 1,
    answer_relevancy_strictness: int = 1,
) -> pd.DataFrame:
    """Run RAGAS core metrics from previously generated local evaluation rows."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    llm, embeddings = _build_ragas_runtime(
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        embedding_model=embedding_model,
        temperature=temperature,
        num_ctx=num_ctx,
    )

    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy

    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=answer_relevancy_strictness),
        ContextPrecision(),
        ContextRecall(),
    ]
    run_config = RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_wait=10,
        max_workers=max_workers,
    )

    frame = pd.read_csv(local_results_path)
    contexts = [json.loads(value) if isinstance(value, str) and value else [] for value in frame["contexts"].tolist()]
    dataset = Dataset.from_dict(
        {
            "user_input": frame["question"].astype(str).tolist(),
            "response": frame["answer"].astype(str).tolist(),
            "retrieved_contexts": contexts,
            "reference": frame["ground_truth"].astype(str).tolist(),
        }
    )
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=True,
        batch_size=batch_size,
        raise_exceptions=raise_exceptions,
    )
    result_frame = result.to_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_path, index=False, encoding="utf-8")
    return result_frame


def _build_ragas_runtime(
    *,
    ollama_base_url: str,
    ollama_model: str,
    embedding_model: str,
    temperature: float,
    num_ctx: int | None,
) -> tuple[Any, Any]:
    """Create local RAGAS judge and embedding objects so RAGAS never falls back to OpenAI."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS local judging requires an Ollama LangChain integration. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS answer relevancy requires a local embedding integration. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

    llm_kwargs: dict[str, Any] = {
        "base_url": ollama_base_url,
        "model": ollama_model,
        "temperature": temperature,
        "format": "json",
        "timeout": 240,
    }
    if num_ctx:
        llm_kwargs["num_ctx"] = num_ctx
    llm = ChatOllama(**llm_kwargs)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    return llm, embeddings


def ragas_metrics_available() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[\w\u00c0-\u1ef9]+", text.lower())
    stopwords = {
        "v\u00e0",
        "l\u00e0",
        "c\u1ee7a",
        "c\u00f3",
        "cho",
        "trong",
        "the",
        "a",
        "an",
        "of",
        "to",
        "is",
    }
    return [word for word in words if len(word) > 3 and word not in stopwords][:12]
