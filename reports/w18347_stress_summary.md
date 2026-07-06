# RAG Evaluation Summary

- Dataset: `evaluation\w18347_stress_eval.csv`
- Results: `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_results.csv`
- Questions: 50

## Core Metrics

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8080 | 0.8080 | 50 / 50 | 0 |
| Context Recall | RAGAS | 0.6033 | 0.6033 | 50 / 50 | 0 |
| Context Precision | RAGAS | 0.4789 | 0.4789 | 50 / 50 | 0 |
| Answer Relevancy | RAGAS | 0.2862 | 0.2862 | 50 / 50 | 0 |
| Latency | Custom | 6571.1470 | 6571.1470 | 50 / 50 | 0 |
| Refusal Rate | Custom | 0.4200 | 0.4200 | 50 / 50 | 0 |
| Hallucination Rate | Custom | 0.1420 | 0.1420 | 50 / 50 | 0 |

## Stage Metrics

### Retrieval

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Context Recall | RAGAS | 0.6033 | 0.6033 | 50 / 50 | 0 |
| Context Precision | RAGAS | 0.4789 | 0.4789 | 50 / 50 | 0 |
| Evidence Hit Rate | CUSTOM | 0.9600 | 0.9600 | 50 / 50 | 0 |
| MRR | CUSTOM | 0.8650 | 0.8650 | 50 / 50 | 0 |

### Generation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8080 | 0.8080 | 50 / 50 | 0 |
| Answer Relevancy | RAGAS | 0.2862 | 0.2862 | 50 / 50 | 0 |
| Hallucination Rate | CUSTOM | 0.1420 | 0.1420 | 50 / 50 | 0 |

### Citation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Citation Accuracy | CUSTOM | 0.9800 | 0.9800 | 50 / 50 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.8517 | 0.8517 | 50 / 50 | 0 |

### Refusal

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Refusal Rate | CUSTOM | 0.4200 | 0.4200 | 50 / 50 | 0 |
| False Refusal Rate | CUSTOM | 0.1000 | 0.1000 | 50 / 50 | 0 |
| Refusal Precision | CUSTOM | 0.7619 | 0.3200 | 21 / 50 | 29 |
| Correct Refusal Behavior | CUSTOM | 0.8750 | 0.4200 | 24 / 50 | 26 |

### Logging

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Latency | CUSTOM | 6571.1470 | 6571.1470 | 50 / 50 | 0 |


## Segment Breakdown

### Answerable Rows

- Rows: 26

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.7808 | 0.7808 | 26 / 26 | 0 |
| Context Recall | RAGAS | 0.5641 | 0.5641 | 26 / 26 | 0 |
| Context Precision | RAGAS | 0.6335 | 0.6335 | 26 / 26 | 0 |
| Answer Relevancy | RAGAS | 0.4596 | 0.4596 | 26 / 26 | 0 |
| Refusal Rate | CUSTOM | 0.1923 | 0.1923 | 26 / 26 | 0 |
| False Refusal Rate | CUSTOM | 0.1923 | 0.1923 | 26 / 26 | 0 |
| Correct Refusal Behavior | CUSTOM | not run | 0.0000 | 0 / 26 | 26 |
| Hallucination Rate | CUSTOM | 0.0641 | 0.0641 | 26 / 26 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.9487 | 0.9487 | 26 / 26 | 0 |

### Refusal / Not-Supported Rows

- Rows: 24

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8375 | 0.8375 | 24 / 24 | 0 |
| Context Recall | RAGAS | 0.6458 | 0.6458 | 24 / 24 | 0 |
| Context Precision | RAGAS | 0.3113 | 0.3113 | 24 / 24 | 0 |
| Answer Relevancy | RAGAS | 0.0983 | 0.0983 | 24 / 24 | 0 |
| Refusal Rate | CUSTOM | 0.6667 | 0.6667 | 24 / 24 | 0 |
| False Refusal Rate | CUSTOM | 0.0000 | 0.0000 | 24 / 24 | 0 |
| Correct Refusal Behavior | CUSTOM | 0.8750 | 0.8750 | 24 / 24 | 0 |
| Hallucination Rate | CUSTOM | 0.2264 | 0.2264 | 24 / 24 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.7465 | 0.7465 | 24 / 24 | 0 |


## Additional Custom Metrics

| Metric | Average | Strict Average | Valid / Total | NaN Count |
|---|---:|---:|---:|---:|
| expected_behavior_accuracy | 0.9400 | 0.9400 | 50 / 50 | 0 |
| evidence_hit_rate | 0.9600 | 0.9600 | 50 / 50 | 0 |
| mrr | 0.8650 | 0.8650 | 50 / 50 | 0 |
| citation_accuracy | 0.9800 | 0.9800 | 50 / 50 | 0 |
| citation_strict_accuracy | 0.8517 | 0.8517 | 50 / 50 | 0 |
| unsupported_claim_accuracy | 0.8580 | 0.8580 | 50 / 50 | 0 |

## Metric Sources

- Faithfulness, Context Recall, Context Precision, and Answer Relevancy come from RAGAS.
- Evidence Hit Rate, MRR, Latency, Refusal Rate, and Hallucination Rate are custom metrics from the local evaluation runner.
- Hallucination Rate is computed as `1 - unsupported_claim_accuracy`.
- Strict Average treats missing or NaN metric values as 0, so evaluator failures are visible.
- Answerable and refusal/out-of-scope rows are reported separately because RAGAS factual QA metrics do not directly measure correct refusal behavior.
