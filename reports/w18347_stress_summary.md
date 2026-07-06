# RAG Evaluation Summary

- Dataset: `evaluation\w18347_stress_eval.csv`
- Results: `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_results.csv`
- Questions: 50

## Core Metrics

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.6807 | 0.6807 | 50 / 50 | 0 |
| Context Recall | RAGAS | 0.2067 | 0.2067 | 50 / 50 | 0 |
| Context Precision | RAGAS | 0.0533 | 0.0533 | 50 / 50 | 0 |
| Answer Relevancy | RAGAS | 0.1508 | 0.1508 | 50 / 50 | 0 |
| Latency | Custom | 5171.2226 | 5171.2226 | 50 / 50 | 0 |
| Refusal Rate | Custom | 0.6800 | 0.6800 | 50 / 50 | 0 |
| Hallucination Rate | Custom | 0.2104 | 0.2104 | 50 / 50 | 0 |

## Stage Metrics

### Retrieval

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Context Recall | RAGAS | 0.2067 | 0.2067 | 50 / 50 | 0 |
| Context Precision | RAGAS | 0.0533 | 0.0533 | 50 / 50 | 0 |
| Evidence Hit Rate | CUSTOM | 0.8800 | 0.8800 | 50 / 50 | 0 |
| MRR | CUSTOM | 0.7400 | 0.7400 | 50 / 50 | 0 |

### Generation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.6807 | 0.6807 | 50 / 50 | 0 |
| Answer Relevancy | RAGAS | 0.1508 | 0.1508 | 50 / 50 | 0 |
| Hallucination Rate | CUSTOM | 0.2104 | 0.2104 | 50 / 50 | 0 |

### Citation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Citation Accuracy | CUSTOM | 0.8760 | 0.8760 | 50 / 50 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.5567 | 0.5567 | 50 / 50 | 0 |

### Refusal

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Refusal Rate | CUSTOM | 0.6800 | 0.6800 | 50 / 50 | 0 |
| False Refusal Rate | CUSTOM | 0.3200 | 0.3200 | 50 / 50 | 0 |
| Refusal Precision | CUSTOM | 0.5294 | 0.3600 | 34 / 50 | 16 |
| Correct Refusal Behavior | CUSTOM | 0.8333 | 0.4000 | 24 / 50 | 26 |

### Logging

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Latency | CUSTOM | 5171.2226 | 5171.2226 | 50 / 50 | 0 |


## Segment Breakdown

### Answerable Rows

- Rows: 26

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.6705 | 0.6705 | 26 / 26 | 0 |
| Context Recall | RAGAS | 0.0385 | 0.0385 | 26 / 26 | 0 |
| Context Precision | RAGAS | 0.0321 | 0.0321 | 26 / 26 | 0 |
| Answer Relevancy | RAGAS | 0.1874 | 0.1874 | 26 / 26 | 0 |
| Refusal Rate | CUSTOM | 0.6154 | 0.6154 | 26 / 26 | 0 |
| False Refusal Rate | CUSTOM | 0.6154 | 0.6154 | 26 / 26 | 0 |
| Correct Refusal Behavior | CUSTOM | not run | 0.0000 | 0 / 26 | 26 |
| Hallucination Rate | CUSTOM | 0.1538 | 0.1538 | 26 / 26 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.5769 | 0.5769 | 26 / 26 | 0 |

### Refusal / Not-Supported Rows

- Rows: 24

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.6917 | 0.6917 | 24 / 24 | 0 |
| Context Recall | RAGAS | 0.3889 | 0.3889 | 24 / 24 | 0 |
| Context Precision | RAGAS | 0.0764 | 0.0764 | 24 / 24 | 0 |
| Answer Relevancy | RAGAS | 0.1112 | 0.1112 | 24 / 24 | 0 |
| Refusal Rate | CUSTOM | 0.7500 | 0.7500 | 24 / 24 | 0 |
| False Refusal Rate | CUSTOM | 0.0000 | 0.0000 | 24 / 24 | 0 |
| Correct Refusal Behavior | CUSTOM | 0.8333 | 0.8333 | 24 / 24 | 0 |
| Hallucination Rate | CUSTOM | 0.2717 | 0.2717 | 24 / 24 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.5347 | 0.5347 | 24 / 24 | 0 |


## Additional Custom Metrics

| Metric | Average | Strict Average | Valid / Total | NaN Count |
|---|---:|---:|---:|---:|
| expected_behavior_accuracy | 0.9200 | 0.9200 | 50 / 50 | 0 |
| evidence_hit_rate | 0.8800 | 0.8800 | 50 / 50 | 0 |
| mrr | 0.7400 | 0.7400 | 50 / 50 | 0 |
| citation_accuracy | 0.8760 | 0.8760 | 50 / 50 | 0 |
| citation_strict_accuracy | 0.5567 | 0.5567 | 50 / 50 | 0 |
| unsupported_claim_accuracy | 0.7896 | 0.7896 | 50 / 50 | 0 |

## Metric Sources

- Faithfulness, Context Recall, Context Precision, and Answer Relevancy come from RAGAS.
- Evidence Hit Rate, MRR, Latency, Refusal Rate, and Hallucination Rate are custom metrics from the local evaluation runner.
- Hallucination Rate is computed as `1 - unsupported_claim_accuracy`.
- Strict Average treats missing or NaN metric values as 0, so evaluator failures are visible.
- Answerable and refusal/out-of-scope rows are reported separately because RAGAS factual QA metrics do not directly measure correct refusal behavior.
