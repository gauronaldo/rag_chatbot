# RAG Evaluation Summary

- Dataset: `evaluation\w18347_dev_eval.csv`
- Custom results: `reports\w18347_dev_results.csv`
- RAGAS results: not run
- Questions: 30

## Core Metrics

| Metric | Source | Average |
|---|---|---:|
| Faithfulness | RAGAS | not run |
| Context Recall | RAGAS | not run |
| Context Precision | RAGAS | not run |
| Answer Relevancy | RAGAS | not run |
| Latency | Custom | 5227.5670 |
| Refusal Rate | Custom | 0.0000 |
| Hallucination Rate | Custom | 0.0083 |

## Additional Custom Metrics

| Metric | Average |
|---|---:|
| false_refusal | 0.0000 |
| expected_behavior_accuracy | 1.0000 |
| citation_accuracy | 1.0000 |
| citation_strict_accuracy | 0.9750 |
| unsupported_claim_accuracy | 0.9917 |

## Metric Sources

- Faithfulness, Context Recall, Context Precision, and Answer Relevancy come from RAGAS when `--ragas` is used.
- Latency, Refusal Rate, and Hallucination Rate are custom metrics from the local evaluation runner.
- Hallucination Rate is computed as `1 - unsupported_claim_accuracy`.
