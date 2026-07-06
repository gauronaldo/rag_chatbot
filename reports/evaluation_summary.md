# Combined RAG Evaluation Summary

| Dataset | Questions | Results | Summary | Faithfulness | Context Recall | Context Precision | Answer Relevancy | Latency | Refusal Rate | Hallucination Rate |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `evaluation\w18347_dev_eval.csv` | 50 | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_dev_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_dev_summary.md` | 0.8682 | 0.6017 | 0.6094 | 0.7439 | 5564.0196 | 0.0800 | 0.0233 |
| `evaluation\w18347_holdout_eval.csv` | 50 | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_holdout_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_holdout_summary.md` | 0.8177 | 0.6450 | 0.5789 | 0.7498 | 5217.4806 | 0.1000 | 0.0339 |
| `evaluation\w18347_stress_eval.csv` | 50 | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_summary.md` | 0.8080 | 0.6033 | 0.4789 | 0.2862 | 6571.1470 | 0.4200 | 0.1420 |

## Output Contract

- Each dataset writes one merged CSV result file and one Markdown summary.
- RAGAS metrics are merged into the same CSV as custom metrics.
- This combined summary aggregates the core metric averages across all datasets run in the command.
