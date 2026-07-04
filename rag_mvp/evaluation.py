from __future__ import annotations

import json
import re
import time
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


def evaluate_custom(row: EvaluationRow) -> dict[str, float]:
    citations = extract_citations(row.answer)
    citation_required = str(row.required_citation).strip().lower() != "no"
    citation_score = citation_accuracy(row.answer, row.contexts)
    strict_citation_score = citation_strict_accuracy(row.answer, row.contexts, row.ground_truth)
    if not citation_required and not citations:
        citation_score = 1.0
        strict_citation_score = 1.0

    return {
        "false_refusal": false_refusal(row.answer, row.ground_truth, row.expected_behavior),
        "expected_behavior_accuracy": expected_behavior_accuracy(row.answer, row.expected_behavior),
        "citation_accuracy": citation_score,
        "citation_strict_accuracy": strict_citation_score,
        "unsupported_claim_accuracy": unsupported_claim_accuracy(row.answer, row.contexts),
        "latency_ms": row.latency_ms,
    }


def run_local_evaluation(pipeline: RagPipeline, dataset_path: Path, output_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, Any]] = []
    for item in dataset.to_dict(orient="records"):
        started = time.perf_counter()
        result = pipeline.answer(str(item["question"]))
        latency_ms = result.get("latency_ms") or round((time.perf_counter() - started) * 1000, 2)
        contexts = [ctx["text"] for ctx in result["contexts"]]
        eval_row = EvaluationRow(
            question=str(item["question"]),
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


def run_ragas_core_evaluation(local_results_path: Path, output_path: Path) -> pd.DataFrame:
    """Run RAGAS core metrics from previously generated local evaluation rows."""
    from datasets import Dataset
    from ragas import evaluate

    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    except ImportError:
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]

    frame = pd.read_csv(local_results_path)
    contexts = [json.loads(value) if isinstance(value, str) and value else [] for value in frame["contexts"].tolist()]
    dataset = Dataset.from_dict(
        {
            "question": frame["question"].astype(str).tolist(),
            "answer": frame["answer"].astype(str).tolist(),
            "contexts": contexts,
            "ground_truth": frame["ground_truth"].astype(str).tolist(),
            "reference": frame["ground_truth"].astype(str).tolist(),
        }
    )
    result = evaluate(dataset, metrics=metrics)
    result_frame = result.to_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_path, index=False, encoding="utf-8")
    return result_frame


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
