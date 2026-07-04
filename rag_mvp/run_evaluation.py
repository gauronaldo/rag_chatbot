from __future__ import annotations

import argparse
from pathlib import Path

from rag_mvp.config import get_config
from rag_mvp.evaluation import ragas_metrics_available, run_local_evaluation, run_ragas_core_evaluation
from rag_mvp.pipeline import RagPipeline


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description="Run RAG MVP evaluation outside the Streamlit UI.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.eval_dataset_path,
        help="CSV with question,ground_truth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.reports_dir / "rag_mvp_eval_results.csv",
        help="Path for custom metric results.",
    )
    parser.add_argument(
        "--ragas-output",
        type=Path,
        default=config.reports_dir / "ragas_core_results.csv",
        help="Path for optional RAGAS metric results.",
    )
    parser.add_argument("--ragas", action="store_true", help="Run optional RAGAS core metrics after local evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = RagPipeline(get_config())
    frame = run_local_evaluation(pipeline, args.dataset, args.output)
    print(f"Saved custom evaluation results to {args.output}")
    print(frame.to_string(index=False))

    if args.ragas:
        if not ragas_metrics_available():
            raise RuntimeError("RAGAS is not installed in this environment.")
        ragas_frame = run_ragas_core_evaluation(args.output, args.ragas_output)
        print(f"Saved RAGAS results to {args.ragas_output}")
        print(ragas_frame.to_string(index=False))


if __name__ == "__main__":
    main()
