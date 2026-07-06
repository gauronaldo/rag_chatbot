# Combined RAG Evaluation Summary

| Dataset | Questions | Results | Summary | Faithfulness | Context Recall | Context Precision | Answer Relevancy | Latency | Refusal Rate | Hallucination Rate |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `evaluation\w18347_dev_eval.csv` | 50 | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_dev_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_dev_summary.md` | 0.5766 | 0.0300 | 0.0050 | 0.2687 | 4165.9864 | 0.6400 | 0.2470 |
| `evaluation\w18347_holdout_eval.csv` | 50 | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_holdout_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_holdout_summary.md` | 0.5377 | 0.0067 | 0.0000 | 0.1740 | 5077.6424 | 0.7400 | 0.1358 |
| `evaluation\w18347_stress_eval.csv` | 50 | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_summary.md` | 0.6807 | 0.2067 | 0.0533 | 0.1508 | 5171.2226 | 0.6800 | 0.2104 |

## Output Contract

- Each dataset writes one merged CSV result file and one Markdown summary.
- RAGAS metrics are merged into the same CSV as custom metrics.
- This combined summary aggregates the core metric averages across all datasets run in the command.
