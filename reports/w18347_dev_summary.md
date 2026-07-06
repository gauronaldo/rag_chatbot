# RAG Evaluation Summary

- Dataset: `evaluation\w18347_dev_eval.csv`
- Results: `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_dev_results.csv`
- Questions: 50

## Core Metrics

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8682 | 0.8682 | 50 / 50 | 0 |
| Context Recall | RAGAS | 0.6017 | 0.6017 | 50 / 50 | 0 |
| Context Precision | RAGAS | 0.6094 | 0.6094 | 50 / 50 | 0 |
| Answer Relevancy | RAGAS | 0.7439 | 0.7439 | 50 / 50 | 0 |
| Latency | Custom | 5564.0196 | 5564.0196 | 50 / 50 | 0 |
| Refusal Rate | Custom | 0.0800 | 0.0800 | 50 / 50 | 0 |
| Hallucination Rate | Custom | 0.0233 | 0.0233 | 50 / 50 | 0 |

## Stage Metrics

### Retrieval

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Context Recall | RAGAS | 0.6017 | 0.6017 | 50 / 50 | 0 |
| Context Precision | RAGAS | 0.6094 | 0.6094 | 50 / 50 | 0 |
| Evidence Hit Rate | CUSTOM | 0.9800 | 0.9800 | 50 / 50 | 0 |
| MRR | CUSTOM | 0.9667 | 0.9667 | 50 / 50 | 0 |

### Generation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8682 | 0.8682 | 50 / 50 | 0 |
| Answer Relevancy | RAGAS | 0.7439 | 0.7439 | 50 / 50 | 0 |
| Hallucination Rate | CUSTOM | 0.0233 | 0.0233 | 50 / 50 | 0 |

### Citation

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Citation Accuracy | CUSTOM | 0.9200 | 0.9200 | 50 / 50 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.9000 | 0.9000 | 50 / 50 | 0 |

### Refusal

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Refusal Rate | CUSTOM | 0.0800 | 0.0800 | 50 / 50 | 0 |
| False Refusal Rate | CUSTOM | 0.0800 | 0.0800 | 50 / 50 | 0 |
| Refusal Precision | CUSTOM | 0.0000 | 0.0000 | 4 / 50 | 46 |
| Correct Refusal Behavior | CUSTOM | not run | 0.0000 | 0 / 50 | 50 |

### Logging

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Latency | CUSTOM | 5564.0196 | 5564.0196 | 50 / 50 | 0 |


## Segment Breakdown

### Answerable Rows

- Rows: 50

| Metric | Source | Average | Strict Average | Valid / Total | NaN Count |
|---|---|---:|---:|---:|---:|
| Faithfulness | RAGAS | 0.8682 | 0.8682 | 50 / 50 | 0 |
| Context Recall | RAGAS | 0.6017 | 0.6017 | 50 / 50 | 0 |
| Context Precision | RAGAS | 0.6094 | 0.6094 | 50 / 50 | 0 |
| Answer Relevancy | RAGAS | 0.7439 | 0.7439 | 50 / 50 | 0 |
| Refusal Rate | CUSTOM | 0.0800 | 0.0800 | 50 / 50 | 0 |
| False Refusal Rate | CUSTOM | 0.0800 | 0.0800 | 50 / 50 | 0 |
| Correct Refusal Behavior | CUSTOM | not run | 0.0000 | 0 / 50 | 50 |
| Hallucination Rate | CUSTOM | 0.0233 | 0.0233 | 50 / 50 | 0 |
| Citation Strict Accuracy | CUSTOM | 0.9000 | 0.9000 | 50 / 50 | 0 |

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
| expected_behavior_accuracy | 1.0000 | 1.0000 | 50 / 50 | 0 |
| evidence_hit_rate | 0.9800 | 0.9800 | 50 / 50 | 0 |
| mrr | 0.9667 | 0.9667 | 50 / 50 | 0 |
| citation_accuracy | 0.9200 | 0.9200 | 50 / 50 | 0 |
| citation_strict_accuracy | 0.9000 | 0.9000 | 50 / 50 | 0 |
| unsupported_claim_accuracy | 0.9767 | 0.9767 | 50 / 50 | 0 |

## Metric Sources

- Faithfulness, Context Recall, Context Precision, and Answer Relevancy come from RAGAS.
- Evidence Hit Rate, MRR, Latency, Refusal Rate, and Hallucination Rate are custom metrics from the local evaluation runner.
- Hallucination Rate is computed as `1 - unsupported_claim_accuracy`.
- Strict Average treats missing or NaN metric values as 0, so evaluator failures are visible.
- Answerable and refusal/out-of-scope rows are reported separately because RAGAS factual QA metrics do not directly measure correct refusal behavior.
