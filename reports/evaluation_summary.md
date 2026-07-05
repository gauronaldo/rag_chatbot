# Combined RAG Evaluation Summary

| Dataset | Summary | Custom Results | RAGAS Results |
|---|---|---|---|
| `evaluation\w18347_dev_eval.csv` | `reports\w18347_dev_eval_summary.md` | `reports\w18347_dev_eval_results.csv` | `reports\w18347_dev_eval_ragas_results.csv` |
| `evaluation\w18347_holdout_eval.csv` | `reports\w18347_holdout_eval_summary.md` | `reports\w18347_holdout_eval_results.csv` | `reports\w18347_holdout_eval_ragas_results.csv` |
| `evaluation\w18347_stress_eval.csv` | `reports\w18347_stress_eval_summary.md` | `reports\w18347_stress_eval_results.csv` | `reports\w18347_stress_eval_ragas_results.csv` |

## Aggregate Metrics

| Split | Questions | Faithfulness | Context Recall | Context Precision | Answer Relevancy | Hallucination Rate | False Refusal Rate | Citation Strict Accuracy | Latency ms | RAGAS NaNs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 30 | 0.8644 | 0.8833 | 0.6630 | 0.7391 | 0.0114 | 0.0667 | 0.9917 | 7430.3670 | 0 |
| holdout | 30 | 0.8050 | 0.7167 | 0.6296 | 0.8030 | 0.0000 | 0.0667 | 0.9333 | 4979.4697 | 0 |
| stress | 20 | 0.8000 | 0.4583 | 0.3167 | 0.2620 | 0.0817 | 0.1500 | 0.6833 | 4381.7500 | 0 |
| **Overall** | 80 | 0.8260 | 0.7146 | 0.5639 | 0.6438 | 0.0247 | 0.0875 | 0.8927 | 5749.1262 | 0 |

## Core Metrics

Each split summary includes these core metrics:

- Faithfulness: RAGAS
- Context Recall: RAGAS
- Context Precision: RAGAS
- Answer Relevancy: RAGAS
- Evidence Hit Rate: custom
- MRR: custom
- Latency: custom
- Refusal Rate: custom
- False Refusal Rate: custom
- Refusal Precision: custom
- Correct Refusal Behavior: custom
- Hallucination Rate: custom
- Each split summary reports NaN counts and strict averages where NaN is counted as 0.
