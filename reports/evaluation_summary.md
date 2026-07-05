# Combined RAG Evaluation Summary

| Dataset | Summary | Custom Results | RAGAS Results |
|---|---|---|---|
| `evaluation\w18347_dev_eval.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_dev_eval_summary.md` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_dev_eval_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_dev_eval_ragas_results.csv` |
| `evaluation\w18347_holdout_eval.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_holdout_eval_summary.md` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_holdout_eval_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_holdout_eval_ragas_results.csv` |
| `evaluation\w18347_stress_eval.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_eval_summary.md` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_eval_results.csv` | `C:\Users\ADMIN\Documents\rag_chatbot_new\reports\w18347_stress_eval_ragas_results.csv` |

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
