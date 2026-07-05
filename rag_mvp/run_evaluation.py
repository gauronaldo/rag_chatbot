from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rag_mvp.config import get_config
from rag_mvp.evaluation import (
    is_refusal,
    ragas_metrics_available,
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
        "--ragas",
        action="store_true",
        help="Deprecated: RAGAS runs by default. Kept for backward compatibility.",
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
    pipeline = RagPipeline(get_config())
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
        )
        summaries.append(summary)

    if multiple:
        write_combined_summary(summaries, args.combined_summary_output)
        print(f"Saved combined metric summary to {args.combined_summary_output}")


def run_one_dataset(
    pipeline: RagPipeline,
    dataset_path: Path,
    output_path: Path,
    ragas_output_path: Path,
    summary_output_path: Path,
    run_ragas: bool,
) -> dict[str, Path]:
    print(f"\n=== Evaluating {dataset_path} ===", flush=True)
    print(f"Custom metrics output: {output_path}", flush=True)

    def report_progress(index: int, total: int, question: str) -> None:
        print(f"[custom {index}/{total}] {question}", flush=True)

    frame = run_local_evaluation(pipeline, dataset_path, output_path, progress_callback=report_progress)
    print(f"Saved custom evaluation results to {output_path}")
    display_columns = [
        "question",
        "false_refusal",
        "refusal",
        "expected_behavior_accuracy",
        "citation_accuracy",
        "citation_strict_accuracy",
        "unsupported_claim_accuracy",
        "hallucination_rate",
        "latency_ms",
    ]
    print(frame[display_columns].to_string(index=False))

    ragas_results_path = None
    if run_ragas:
        if not ragas_metrics_available():
            raise RuntimeError("RAGAS is required for core metrics but is not installed in this environment.")
        print(f"[ragas] Running core metrics for {dataset_path}", flush=True)
        print(f"RAGAS metrics output: {ragas_output_path}", flush=True)
        ragas_frame = run_ragas_core_evaluation(output_path, ragas_output_path)
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
    if "refusal" not in custom_frame.columns and "answer" in custom_frame.columns:
        custom_frame["refusal"] = custom_frame["answer"].astype(str).map(lambda answer: int(is_refusal(answer)))
    if "hallucination_rate" not in custom_frame.columns and "unsupported_claim_accuracy" in custom_frame.columns:
        custom_frame["hallucination_rate"] = 1 - pd.to_numeric(
            custom_frame["unsupported_claim_accuracy"],
            errors="coerce",
        )
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
        "| Metric | Source | Average |",
        "|---|---|---:|",
    ]

    for metric_name, column_names in CORE_RAGAS_METRICS.items():
        value = _mean_from_first_available_column(ragas_frame, column_names)
        lines.append(f"| {metric_name} | RAGAS | {_format_metric(value)} |")
    for metric_name, column_name in CORE_CUSTOM_METRICS.items():
        value = _mean_from_first_available_column(custom_frame, (column_name,))
        lines.append(f"| {metric_name} | Custom | {_format_metric(value)} |")

    lines.extend(
        [
            "",
            "## Additional Custom Metrics",
            "",
            "| Metric | Average |",
            "|---|---:|",
        ]
    )
    for column_name in (
        "false_refusal",
        "expected_behavior_accuracy",
        "citation_accuracy",
        "citation_strict_accuracy",
        "unsupported_claim_accuracy",
    ):
        value = _mean_from_first_available_column(custom_frame, (column_name,))
        lines.append(f"| {column_name} | {_format_metric(value)} |")

    lines.extend(
        [
            "",
            "## Metric Sources",
            "",
            "- Faithfulness, Context Recall, Context Precision, and Answer Relevancy come from RAGAS.",
            "- Latency, Refusal Rate, and Hallucination Rate are custom metrics from the local evaluation runner.",
            "- Hallucination Rate is computed as `1 - unsupported_claim_accuracy`.",
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
            "- Latency: custom",
            "- Refusal Rate: custom",
            "- Hallucination Rate: custom",
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


def _format_metric(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "not run"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
