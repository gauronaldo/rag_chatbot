# RAG Evaluation Summary

- Dataset: `evaluation\w18347_dev_eval.csv`
- Custom results: `reports\w18347_dev_eval_results.csv`
- RAGAS results: `reports\w18347_dev_eval_ragas_results.csv`
- Questions: 30

## Core Metrics

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8644 | 0.8644 | 30 / 30 | 0 |
| Context Recall | RAGAS | 0.8833 | 0.8833 | 30 / 30 | 0 |
| Context Precision | RAGAS | 0.6630 | 0.6630 | 30 / 30 | 0 |
| Answer Relevancy | RAGAS | 0.7391 | 0.7391 | 30 / 30 | 0 |
| Latency | Custom | 7430.3670 | 7430.3670 | 30 / 30 | 0 |
| Refusal Rate | Custom | 0.0667 | 0.0667 | 30 / 30 | 0 |
| Hallucination Rate | Custom | 0.0114 | 0.0114 | 30 / 30 | 0 |

## Stage Metrics

### Retrieval

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Context Recall | RAGAS | 0.8833 | 0.8833 | 30 / 30 | 0 |
| Context Precision | RAGAS | 0.6630 | 0.6630 | 30 / 30 | 0 |
| Evidence Hit Rate | CUSTOM | 1.0000 | 1.0000 | 30 / 30 | 0 |
| MRR | CUSTOM | 0.9833 | 0.9833 | 30 / 30 | 0 |

### Generation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8644 | 0.8644 | 30 / 30 | 0 |
| Answer Relevancy | RAGAS | 0.7391 | 0.7391 | 30 / 30 | 0 |
| Hallucination Rate | CUSTOM | 0.0114 | 0.0114 | 30 / 30 | 0 |

### Citation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Citation Accuracy | CUSTOM | 1.0000 | 1.0000 | 30 / 30 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.9917 | 0.9917 | 30 / 30 | 0 |

### Refusal

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Refusal Rate | CUSTOM | 0.0667 | 0.0667 | 30 / 30 | 0 |
| False Refusal Rate | CUSTOM | 0.0667 | 0.0667 | 30 / 30 | 0 |
| Refusal Precision | CUSTOM | 0.0000 | 0.0000 | 2 / 30 | 28 |
| Correct Refusal Behavior | CUSTOM | not run | 0.0000 | 0 / 30 | 30 |

### Logging

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Latency | CUSTOM | 7430.3670 | 7430.3670 | 30 / 30 | 0 |


## Segment Breakdown

### Answerable Rows

- Rows: 30

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8644 | 0.8644 | 30 / 30 | 0 |
| Context Recall | RAGAS | 0.8833 | 0.8833 | 30 / 30 | 0 |
| Context Precision | RAGAS | 0.6630 | 0.6630 | 30 / 30 | 0 |
| Answer Relevancy | RAGAS | 0.7391 | 0.7391 | 30 / 30 | 0 |
| Refusal Rate | CUSTOM | 0.0667 | 0.0667 | 30 / 30 | 0 |
| False Refusal Rate | CUSTOM | 0.0667 | 0.0667 | 30 / 30 | 0 |
| Correct Refusal Behavior | CUSTOM | not run | 0.0000 | 0 / 30 | 30 |
| Hallucination Rate | CUSTOM | 0.0114 | 0.0114 | 30 / 30 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.9917 | 0.9917 | 30 / 30 | 0 |

### Refusal / Not-Supported Rows

- Rows: 0

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | not run | not run | 0 / 0 | 0 |
| Context Recall | RAGAS | not run | not run | 0 / 0 | 0 |
| Context Precision | RAGAS | not run | not run | 0 / 0 | 0 |
| Answer Relevancy | RAGAS | not run | not run | 0 / 0 | 0 |
| Refusal Rate | CUSTOM | not run | not run | 0 / 0 | 0 |
| False Refusal Rate | CUSTOM | not run | not run | 0 / 0 | 0 |
| Correct Refusal Behavior | CUSTOM | not run | not run | 0 / 0 | 0 |
| Hallucination Rate | CUSTOM | not run | not run | 0 / 0 | 0 |
| Citation Strict Accuracy | CUSTOM | not run | not run | 0 / 0 | 0 |


## Additional Custom Metrics

| Metric | Average | Strict Average | Valid / Total | NaN Count |
|---|---:|---:|---:|---:|
| expected_behavior_accuracy | 1.0000 | 1.0000 | 30 / 30 | 0 |
| evidence_hit_rate | 1.0000 | 1.0000 | 30 / 30 | 0 |
| mrr | 0.9833 | 0.9833 | 30 / 30 | 0 |
| citation_accuracy | 1.0000 | 1.0000 | 30 / 30 | 0 |
| citation_strict_accuracy | 0.9917 | 0.9917 | 30 / 30 | 0 |
| unsupported_claim_accuracy | 0.9886 | 0.9886 | 30 / 30 | 0 |

## Metric Sources

- Faithfulness, Context Recall, Context Precision, and Answer Relevancy come from RAGAS.
- Evidence Hit Rate, MRR, Latency, Refusal Rate, and Hallucination Rate are custom metrics from the local evaluation runner.
- Hallucination Rate is computed as `1 - unsupported_claim_accuracy`.
- Strict Average treats missing or NaN metric values as 0, so evaluator failures are visible.
- Answerable and refusal/out-of-scope rows are reported separately because RAGAS factual QA metrics do not directly measure correct refusal behavior.
