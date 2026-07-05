# RAG Evaluation Summary

- Dataset: `evaluation\w18347_stress_eval.csv`
- Custom results: `reports\w18347_stress_eval_results.csv`
- RAGAS results: `reports\w18347_stress_eval_ragas_results.csv`
- Questions: 20

## Core Metrics

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8000 | 0.8000 | 20 / 20 | 0 |
| Context Recall | RAGAS | 0.4583 | 0.4583 | 20 / 20 | 0 |
| Context Precision | RAGAS | 0.3167 | 0.3167 | 20 / 20 | 0 |
| Answer Relevancy | RAGAS | 0.2620 | 0.2620 | 20 / 20 | 0 |
| Latency | Custom | 4381.7500 | 4381.7500 | 20 / 20 | 0 |
| Refusal Rate | Custom | 0.7000 | 0.7000 | 20 / 20 | 0 |
| Hallucination Rate | Custom | 0.0817 | 0.0817 | 20 / 20 | 0 |

## Stage Metrics

### Retrieval

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Context Recall | RAGAS | 0.4583 | 0.4583 | 20 / 20 | 0 |
| Context Precision | RAGAS | 0.3167 | 0.3167 | 20 / 20 | 0 |
| Evidence Hit Rate | CUSTOM | 0.9000 | 0.9000 | 20 / 20 | 0 |
| MRR | CUSTOM | 0.8292 | 0.8292 | 20 / 20 | 0 |

### Generation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8000 | 0.8000 | 20 / 20 | 0 |
| Answer Relevancy | RAGAS | 0.2620 | 0.2620 | 20 / 20 | 0 |
| Hallucination Rate | CUSTOM | 0.0817 | 0.0817 | 20 / 20 | 0 |

### Citation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Citation Accuracy | CUSTOM | 0.9500 | 0.9500 | 20 / 20 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.6833 | 0.6833 | 20 / 20 | 0 |

### Refusal

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Refusal Rate | CUSTOM | 0.7000 | 0.7000 | 20 / 20 | 0 |
| False Refusal Rate | CUSTOM | 0.1500 | 0.1500 | 20 / 20 | 0 |
| Refusal Precision | CUSTOM | 0.7857 | 0.5500 | 14 / 20 | 6 |
| Correct Refusal Behavior | CUSTOM | 1.0000 | 0.6000 | 12 / 20 | 8 |

### Logging

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Latency | CUSTOM | 4381.7500 | 4381.7500 | 20 / 20 | 0 |


## Segment Breakdown

### Answerable Rows

- Rows: 8

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.7396 | 0.7396 | 8 / 8 | 0 |
| Context Recall | RAGAS | 0.1250 | 0.1250 | 8 / 8 | 0 |
| Context Precision | RAGAS | 0.3437 | 0.3437 | 8 / 8 | 0 |
| Answer Relevancy | RAGAS | 0.3772 | 0.3772 | 8 / 8 | 0 |
| Refusal Rate | CUSTOM | 0.3750 | 0.3750 | 8 / 8 | 0 |
| False Refusal Rate | CUSTOM | 0.3750 | 0.3750 | 8 / 8 | 0 |
| Correct Refusal Behavior | CUSTOM | not run | 0.0000 | 0 / 8 | 8 |
| Hallucination Rate | CUSTOM | 0.0375 | 0.0375 | 8 / 8 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.8333 | 0.8333 | 8 / 8 | 0 |

### Refusal / Not-Supported Rows

- Rows: 12

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8403 | 0.8403 | 12 / 12 | 0 |
| Context Recall | RAGAS | 0.6806 | 0.6806 | 12 / 12 | 0 |
| Context Precision | RAGAS | 0.2986 | 0.2986 | 12 / 12 | 0 |
| Answer Relevancy | RAGAS | 0.1852 | 0.1852 | 12 / 12 | 0 |
| Refusal Rate | CUSTOM | 0.9167 | 0.9167 | 12 / 12 | 0 |
| False Refusal Rate | CUSTOM | 0.0000 | 0.0000 | 12 / 12 | 0 |
| Correct Refusal Behavior | CUSTOM | 1.0000 | 1.0000 | 12 / 12 | 0 |
| Hallucination Rate | CUSTOM | 0.1111 | 0.1111 | 12 / 12 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.5833 | 0.5833 | 12 / 12 | 0 |


## Additional Custom Metrics

| Metric | Average | Strict Average | Valid / Total | NaN Count |
|---|---:|---:|---:|---:|
| expected_behavior_accuracy | 1.0000 | 1.0000 | 20 / 20 | 0 |
| evidence_hit_rate | 0.9000 | 0.9000 | 20 / 20 | 0 |
| mrr | 0.8292 | 0.8292 | 20 / 20 | 0 |
| citation_accuracy | 0.9500 | 0.9500 | 20 / 20 | 0 |
| citation_strict_accuracy | 0.6833 | 0.6833 | 20 / 20 | 0 |
| unsupported_claim_accuracy | 0.9183 | 0.9183 | 20 / 20 | 0 |

## Metric Sources

- Faithfulness, Context Recall, Context Precision, and Answer Relevancy come from RAGAS.
- Evidence Hit Rate, MRR, Latency, Refusal Rate, and Hallucination Rate are custom metrics from the local evaluation runner.
- Hallucination Rate is computed as `1 - unsupported_claim_accuracy`.
- Strict Average treats missing or NaN metric values as 0, so evaluator failures are visible.
- Answerable and refusal/out-of-scope rows are reported separately because RAGAS factual QA metrics do not directly measure correct refusal behavior.
