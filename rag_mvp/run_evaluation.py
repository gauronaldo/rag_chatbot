from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from rag_mvp.config import get_config
from rag_mvp.evaluation import (
    correct_refusal_behavior,
    evidence_hit_rate,
    evidence_mrr,
    false_refusal,
    is_refusal,
    ragas_metrics_available,
    refusal_precision,
    run_local_evaluation,
    run_ragas_core_evaluation,
)
from rag_mvp.pipeline import RagPipeline

CORE_RAGAS_METRICS = {
    "Faithfulness": ("faithfulness",),
    "Context Recall": ("context_recall",),
    "Context Precision": ("context_precision",),
    "Answer Relevancy": ("answer_relevancy", "answer_relevance", "response_relevancy"),
}

CORE_CUSTOM_METRICS = {
    "Latency": "latency_ms",
    "Refusal Rate": "refusal",
    "Hallucination Rate": "hallucination_rate",
}

STAGE_METRICS = {
    "Retrieval": {
        "Context Recall": ("ragas", ("context_recall",)),
        "Context Precision": ("ragas", ("context_precision",)),
        "Evidence Hit Rate": ("custom", ("evidence_hit_rate",)),
        "MRR": ("custom", ("mrr",)),
    },
    "Generation": {
        "Faithfulness": ("ragas", ("faithfulness",)),
        "Answer Relevancy": ("ragas", ("answer_relevancy", "answer_relevance", "response_relevancy")),
        "Hallucination Rate": ("custom", ("hallucination_rate",)),
    },
    "Citation": {
        "Citation Accuracy": ("custom", ("citation_accuracy",)),
        "Citation Strict Accuracy": ("custom", ("citation_strict_accuracy",)),
    },
    "Refusal": {
        "Refusal Rate": ("custom", ("refusal",)),
        "False Refusal Rate": ("custom", ("false_refusal",)),
        "Refusal Precision": ("custom", ("refusal_precision",)),
        "Correct Refusal Behavior": ("custom", ("correct_refusal_behavior",)),
    },
    "Logging": {
        "Latency": ("custom", ("latency_ms",)),
    },
}


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description="Run RAG MVP evaluation outside the Streamlit UI.")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--w18347-all",
        action="store_true",
        help="Run dev, holdout, and stress evaluation sets for w18347.",
    )
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        help="Run multiple evaluation CSV files in one command.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.eval_dataset_path,
        help="Single CSV with question,ground_truth. Ignored when --datasets or --w18347-all is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for custom metric results. Defaults to reports/<dataset_stem>_results.csv.",
    )
    parser.add_argument(
        "--ragas-output",
        type=Path,
        default=None,
        help="Path for RAGAS core metric results. Defaults to reports/<dataset_stem>_ragas_results.csv.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Path for Markdown metric summary. Defaults to reports/<dataset_stem>_summary.md.",
    )
    parser.add_argument(
        "--combined-summary-output",
        type=Path,
        default=config.reports_dir / "evaluation_summary.md",
        help="Path for a combined Markdown summary when running multiple datasets.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS core metrics. Intended only for quick local debugging.",
    )
    parser.add_argument(
        "--ragas-only",
        action="store_true",
        help="Run only RAGAS scoring from an existing custom results CSV passed with --output.",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Deprecated: RAGAS runs by default. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--ragas-model",
        default=None,
        help="Ollama model used as the RAGAS judge. Defaults to OLLAMA_MODEL.",
    )
    parser.add_argument(
        "--ragas-temperature",
        type=float,
        default=0.0,
        help="Temperature for the RAGAS judge. Defaults to 0 for deterministic scoring.",
    )
    parser.add_argument(
        "--ragas-num-ctx",
        type=int,
        default=8192,
        help="Ollama context window for the RAGAS judge.",
    )
    parser.add_argument(
        "--ragas-raise-exceptions",
        action="store_true",
        help="Fail fast on RAGAS metric errors instead of writing NaN.",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=600,
        help="Per-operation RAGAS timeout in seconds.",
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=1,
        help="Maximum concurrent RAGAS jobs. Keep this low for local Ollama judges.",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=1,
        help="RAGAS batch size. Keep this at 1 for local Ollama judges.",
    )
    parser.add_argument(
        "--ragas-max-retries",
        type=int,
        default=1,
        help="Maximum RAGAS retry attempts per operation.",
    )
    parser.add_argument(
        "--ragas-answer-strictness",
        type=int,
        default=1,
        help="Number of generated questions used by answer relevancy.",
    )
    return parser.parse_args()


def resolve_datasets(args: argparse.Namespace) -> list[Path]:
    if args.w18347_all:
        return [
            Path("evaluation/w18347_dev_eval.csv"),
            Path("evaluation/w18347_holdout_eval.csv"),
            Path("evaluation/w18347_stress_eval.csv"),
        ]
    if args.datasets:
        return args.datasets
    return [args.dataset]


def resolve_output_paths(args: argparse.Namespace, dataset_path: Path, multiple: bool) -> tuple[Path, Path, Path]:
    config = get_config()
    if multiple and (args.output or args.ragas_output or args.summary_output):
        raise ValueError("--output, --ragas-output, and --summary-output can only be used with a single dataset.")
    dataset_stem = dataset_path.stem
    output = args.output or config.reports_dir / f"{dataset_stem}_results.csv"
    ragas_output = args.ragas_output or config.reports_dir / f"{dataset_stem}_ragas_results.csv"
    summary_output = args.summary_output or config.reports_dir / f"{dataset_stem}_summary.md"
    return output, ragas_output, summary_output


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    dataset_paths = resolve_datasets(args)
    multiple = len(dataset_paths) > 1
    pipeline = None if args.ragas_only else RagPipeline(get_config())
    summaries = []

    for dataset_path in dataset_paths:
        output_path, ragas_output_path, summary_output_path = resolve_output_paths(args, dataset_path, multiple)
        summary = run_one_dataset(
            pipeline=pipeline,
            dataset_path=dataset_path,
            output_path=output_path,
            ragas_output_path=ragas_output_path,
            summary_output_path=summary_output_path,
            run_ragas=not args.skip_ragas,
            ragas_only=args.ragas_only,
            ragas_model=args.ragas_model,
            ragas_temperature=args.ragas_temperature,
            ragas_num_ctx=args.ragas_num_ctx,
            ragas_raise_exceptions=args.ragas_raise_exceptions,
            ragas_timeout=args.ragas_timeout,
            ragas_max_workers=args.ragas_max_workers,
            ragas_batch_size=args.ragas_batch_size,
            ragas_max_retries=args.ragas_max_retries,
            ragas_answer_strictness=args.ragas_answer_strictness,
        )
        summaries.append(summary)

    if multiple:
        write_combined_summary(summaries, args.combined_summary_output)
        print(f"Saved combined metric summary to {args.combined_summary_output}")


def run_one_dataset(
    pipeline: RagPipeline | None,
    dataset_path: Path,
    output_path: Path,
    ragas_output_path: Path,
    summary_output_path: Path,
    run_ragas: bool,
    ragas_only: bool = False,
    ragas_model: str | None = None,
    ragas_temperature: float = 0.0,
    ragas_num_ctx: int | None = 8192,
    ragas_raise_exceptions: bool = False,
    ragas_timeout: int = 600,
    ragas_max_workers: int = 1,
    ragas_batch_size: int = 1,
    ragas_max_retries: int = 1,
    ragas_answer_strictness: int = 1,
) -> dict[str, Path]:
    print(f"\n=== Evaluating {dataset_path} ===", flush=True)
    print(f"Custom metrics output: {output_path}", flush=True)

    def report_progress(index: int, total: int, question: str) -> None:
        print(f"[custom {index}/{total}] {question}", flush=True)

    if ragas_only:
        if not output_path.exists():
            raise FileNotFoundError(
                f"--ragas-only requires an existing custom results CSV at {output_path}. "
                "Run without --ragas-only first."
            )
        print(f"[custom] Skipped; reusing existing results from {output_path}", flush=True)
    else:
        if pipeline is None:
            raise RuntimeError("Pipeline is required unless --ragas-only is used.")
        frame = run_local_evaluation(pipeline, dataset_path, output_path, progress_callback=report_progress)
        print(f"Saved custom evaluation results to {output_path}")
        display_columns = [
            "question",
            "false_refusal",
            "refusal",
            "refusal_precision",
            "correct_refusal_behavior",
            "expected_behavior_accuracy",
            "evidence_hit_rate",
            "mrr",
            "citation_accuracy",
            "citation_strict_accuracy",
            "unsupported_claim_accuracy",
            "hallucination_rate",
            "latency_ms",
        ]
        print(frame[[column for column in display_columns if column in frame.columns]].to_string(index=False))

    ragas_results_path = None
    if run_ragas:
        if not ragas_metrics_available():
            raise RuntimeError("RAGAS is required for core metrics but is not installed in this environment.")
        config = get_config()
        judge_model = ragas_model or config.ollama_model
        print(f"[ragas] Running core metrics for {dataset_path}", flush=True)
        print(
            "[ragas] Judge: "
            f"Ollama model={judge_model}, base_url={config.ollama_base_url}, "
            f"temperature={ragas_temperature}, num_ctx={ragas_num_ctx}, timeout={ragas_timeout}s, "
            f"max_workers={ragas_max_workers}, batch_size={ragas_batch_size}",
            flush=True,
        )
        print(f"[ragas] Embeddings: {config.embedding_model}", flush=True)
        print(f"RAGAS metrics output: {ragas_output_path}", flush=True)
        ragas_frame = run_ragas_core_evaluation(
            output_path,
            ragas_output_path,
            ollama_base_url=config.ollama_base_url,
            ollama_model=judge_model,
            embedding_model=config.embedding_model,
            temperature=ragas_temperature,
            num_ctx=ragas_num_ctx,
            raise_exceptions=ragas_raise_exceptions,
            timeout=ragas_timeout,
            max_workers=ragas_max_workers,
            max_retries=ragas_max_retries,
            batch_size=ragas_batch_size,
            answer_relevancy_strictness=ragas_answer_strictness,
        )
        ragas_results_path = ragas_output_path
        print(f"Saved RAGAS results to {ragas_output_path}")
        print(ragas_frame.to_string(index=False))

    write_markdown_summary(dataset_path, output_path, ragas_results_path, summary_output_path)
    print(f"Saved metric summary to {summary_output_path}")
    return {
        "dataset": dataset_path,
        "custom_results": output_path,
        "ragas_results": ragas_results_path,
        "summary": summary_output_path,
    }


def write_markdown_summary(
    dataset_path: Path,
    custom_results_path: Path,
    ragas_results_path: Path | None,
    summary_output_path: Path,
) -> None:
    custom_frame = pd.read_csv(custom_results_path)
    custom_frame = _ensure_derived_custom_metrics(custom_frame)
    ragas_frame = pd.read_csv(ragas_results_path) if ragas_results_path and ragas_results_path.exists() else None
    lines = [
        "# RAG Evaluation Summary",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Custom results: `{custom_results_path}`",
        f"- RAGAS results: `{ragas_results_path}`" if ragas_results_path else "- RAGAS results: not run",
        f"- Questions: {len(custom_frame)}",
        "",
        "## Core Metrics",
        "",
        "| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for metric_name, column_names in CORE_RAGAS_METRICS.items():
        stats = _metric_stats(ragas_frame, column_names)
        lines.append(_metric_row(metric_name, "RAGAS", stats))
    for metric_name, column_name in CORE_CUSTOM_METRICS.items():
        stats = _metric_stats(custom_frame, (column_name,))
        lines.append(_metric_row(metric_name, "Custom", stats))

    lines.extend(
        [
            "",
            "## Stage Metrics",
            "",
        ]
    )
    for stage_name, metrics in STAGE_METRICS.items():
        lines.extend(
            [
                f"### {stage_name}",
                "",
                "| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for metric_name, (source, column_names) in metrics.items():
            frame = ragas_frame if source == "ragas" else custom_frame
            stats = _metric_stats(frame, column_names)
            lines.append(_metric_row(metric_name, source.upper(), stats))
        lines.append("")

    lines.extend(_segment_breakdown(custom_frame, ragas_frame))

    lines.extend(
        [
            "",
            "## Additional Custom Metrics",
            "",
            "| Metric | Average | Strict Average | Valid / Total | NaN Count |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for column_name in (
        "expected_behavior_accuracy",
        "evidence_hit_rate",
        "mrr",
        "citation_accuracy",
        "citation_strict_accuracy",
        "unsupported_claim_accuracy",
    ):
        stats = _metric_stats(custom_frame, (column_name,))
        lines.append(_metric_row_without_source(column_name, stats))

    lines.extend(
        [
            "",
            "## Metric Sources",
            "",
            "- Faithfulness, Context Recall, Context Precision, and Answer Relevancy come from RAGAS.",
            "- Evidence Hit Rate, MRR, Latency, Refusal Rate, and Hallucination Rate are custom metrics from "
            "the local evaluation runner.",
            "- Hallucination Rate is computed as `1 - unsupported_claim_accuracy`.",
            "- Strict Average treats missing or NaN metric values as 0, so evaluator failures are visible.",
            "- Answerable and refusal/out-of-scope rows are reported separately because RAGAS factual QA metrics "
            "do not directly measure correct refusal behavior.",
            "",
        ]
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text("\n".join(lines), encoding="utf-8")


def write_combined_summary(summaries: list[dict[str, Path]], output_path: Path) -> None:
    lines = [
        "# Combined RAG Evaluation Summary",
        "",
        "| Dataset | Summary | Custom Results | RAGAS Results |",
        "|---|---|---|---|",
    ]
    for summary in summaries:
        dataset = summary["dataset"]
        single_summary = summary["summary"]
        custom_results = summary["custom_results"]
        ragas_results = summary["ragas_results"]
        lines.append(
            f"| `{dataset}` | `{single_summary}` | `{custom_results}` | "
            f"`{ragas_results}` |"
        )

    lines.extend(
        [
            "",
            "## Core Metrics",
            "",
            "Each split summary includes these core metrics:",
            "",
            "- Faithfulness: RAGAS",
            "- Context Recall: RAGAS",
            "- Context Precision: RAGAS",
            "- Answer Relevancy: RAGAS",
            "- Evidence Hit Rate: custom",
            "- MRR: custom",
            "- Latency: custom",
            "- Refusal Rate: custom",
            "- False Refusal Rate: custom",
            "- Refusal Precision: custom",
            "- Correct Refusal Behavior: custom",
            "- Hallucination Rate: custom",
            "- Each split summary reports NaN counts and strict averages where NaN is counted as 0.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _mean_from_first_available_column(frame: pd.DataFrame | None, column_names: tuple[str, ...]) -> float | None:
    if frame is None:
        return None
    normalized_columns = {column.lower(): column for column in frame.columns}
    for column_name in column_names:
        actual_column = normalized_columns.get(column_name.lower())
        if actual_column:
            return float(pd.to_numeric(frame[actual_column], errors="coerce").mean())
    return None


def _ensure_derived_custom_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if "answer" in frame.columns:
        frame["refusal"] = frame["answer"].astype(str).map(lambda answer: int(is_refusal(answer)))
    if {"answer", "ground_truth", "expected_behavior"}.issubset(frame.columns):
        frame["false_refusal"] = frame.apply(
            lambda row: false_refusal(
                str(row["answer"]),
                str(row["ground_truth"]),
                str(row["expected_behavior"]),
            ),
            axis=1,
        )
        frame["refusal_precision"] = frame.apply(
            lambda row: refusal_precision(str(row["answer"]), str(row["expected_behavior"])),
            axis=1,
        )
        frame["correct_refusal_behavior"] = frame.apply(
            lambda row: correct_refusal_behavior(str(row["answer"]), str(row["expected_behavior"])),
            axis=1,
        )
    if "hallucination_rate" not in frame.columns and "unsupported_claim_accuracy" in frame.columns:
        frame["hallucination_rate"] = 1 - pd.to_numeric(
            frame["unsupported_claim_accuracy"],
            errors="coerce",
        )
    if {"contexts", "ground_truth"}.issubset(frame.columns):
        contexts = frame["contexts"].map(_parse_contexts)
        if "evidence_hit_rate" not in frame.columns:
            frame["evidence_hit_rate"] = [
                evidence_hit_rate(row_contexts, str(ground_truth))
                for row_contexts, ground_truth in zip(contexts, frame["ground_truth"], strict=False)
            ]
        if "mrr" not in frame.columns:
            frame["mrr"] = [
                evidence_mrr(row_contexts, str(ground_truth))
                for row_contexts, ground_truth in zip(contexts, frame["ground_truth"], strict=False)
            ]
    return frame


def _parse_contexts(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _metric_stats(frame: pd.DataFrame | None, column_names: tuple[str, ...]) -> dict[str, float | int | None]:
    if frame is None:
        return {"mean": None, "strict_mean": None, "valid": 0, "total": 0, "nan": 0}
    normalized_columns = {column.lower(): column for column in frame.columns}
    for column_name in column_names:
        actual_column = normalized_columns.get(column_name.lower())
        if actual_column:
            values = pd.to_numeric(frame[actual_column], errors="coerce")
            total = len(values)
            valid = int(values.notna().sum())
            nan_count = int(values.isna().sum())
            mean = float(values.mean()) if valid else None
            strict_mean = float(values.fillna(0).mean()) if total else None
            return {
                "mean": mean,
                "strict_mean": strict_mean,
                "valid": valid,
                "total": total,
                "nan": nan_count,
            }
    return {"mean": None, "strict_mean": None, "valid": 0, "total": len(frame), "nan": len(frame)}


def _metric_row(metric_name: str, source: str, stats: dict[str, float | int | None]) -> str:
    return (
        f"| {metric_name} | {source} | {_format_metric(stats['mean'])} | "
        f"{_format_metric(stats['strict_mean'])} | {stats['valid']} / {stats['total']} | {stats['nan']} |"
    )


def _metric_row_without_source(metric_name: str, stats: dict[str, float | int | None]) -> str:
    return (
        f"| {metric_name} | {_format_metric(stats['mean'])} | {_format_metric(stats['strict_mean'])} | "
        f"{stats['valid']} / {stats['total']} | {stats['nan']} |"
    )


def _segment_breakdown(custom_frame: pd.DataFrame, ragas_frame: pd.DataFrame | None) -> list[str]:
    if "expected_behavior" not in custom_frame.columns:
        return []
    ragas_metrics = {
        "Faithfulness": ("faithfulness",),
        "Context Recall": ("context_recall",),
        "Context Precision": ("context_precision",),
        "Answer Relevancy": ("answer_relevancy", "answer_relevance", "response_relevancy"),
    }
    segments = {
        "Answerable Rows": custom_frame["expected_behavior"].isin(["answer", "claim_verification"]),
        "Refusal / Not-Supported Rows": custom_frame["expected_behavior"].isin(["refuse", "state_not_supported"]),
    }
    lines = [
        "",
        "## Segment Breakdown",
        "",
    ]
    for segment_name, mask in segments.items():
        segment_custom = custom_frame.loc[mask].reset_index(drop=True)
        segment_ragas = ragas_frame.loc[mask].reset_index(drop=True) if ragas_frame is not None else None
        lines.extend(
            [
                f"### {segment_name}",
                "",
                f"- Rows: {len(segment_custom)}",
                "",
                "| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for metric_name, column_names in ragas_metrics.items():
            lines.append(_metric_row(metric_name, "RAGAS", _metric_stats(segment_ragas, column_names)))
        for metric_name, column_names in {
            "Refusal Rate": ("refusal",),
            "False Refusal Rate": ("false_refusal",),
            "Correct Refusal Behavior": ("correct_refusal_behavior",),
            "Hallucination Rate": ("hallucination_rate",),
            "Citation Strict Accuracy": ("citation_strict_accuracy",),
        }.items():
            lines.append(_metric_row(metric_name, "CUSTOM", _metric_stats(segment_custom, column_names)))
        lines.append("")
    return lines


def _format_metric(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "not run"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
