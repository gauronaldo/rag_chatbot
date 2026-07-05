# English RAG MVP - Architecture and Evaluation Guide

## 1. Muc Tieu Project

Project nay la mot Streamlit RAG MVP chay local, tap trung vao hoi dap tai lieu tieng Anh.

Stack chinh:

- UI: Streamlit
- LLM: Ollama, mac dinh `qwen2.5:7b`
- Embedding: `BAAI/bge-m3`
- Vector database: Chroma persistent store
- Registry: JSON registry
- Evaluation: custom metrics + RAGAS core metrics

Project hien tai khong dung FastAPI/React backend/frontend cu, khong dung SQLite/MySQL.

## 2. Cau Truc Thu Muc

```text
rag_mvp/
  config.py          Runtime config
  documents.py       Load file, decode, preprocess, chunk
  registry.py        JSON registry cho indexed documents
  vector_store.py    Chroma + sentence-transformers embeddings
  ollama_client.py   Client goi Ollama generate/stream
  pipeline.py        Orchestration ingest/retrieve/answer
  evaluation.py      Custom evaluation metrics + RAGAS runner
  run_evaluation.py  CLI chay evaluation

evaluation/
  w18347_dev_eval.csv
  w18347_holdout_eval.csv
  w18347_stress_eval.csv
  sample_eval_set.csv

reports/
  *_results.csv
  *_ragas_results.csv
  *_summary.md

streamlit_app.py     Streamlit app entrypoint
requirements.txt     Python dependencies
pyproject.toml        Ruff + pytest config
```

## 3. Runtime Config

Config nam trong `rag_mvp/config.py`.

Gia tri mac dinh quan trong:

- `OLLAMA_BASE_URL`: `http://localhost:11434`
- `OLLAMA_MODEL`: `qwen2.5:7b`
- `EMBEDDING_MODEL`: `BAAI/bge-m3`
- `CHROMA_COLLECTION`: `vietnamese_rag_mvp`
- `TOP_K`: `4`
- `CHUNK_SIZE`: `1000`
- `CHUNK_OVERLAP`: `120`

Luu y: collection name van la `vietnamese_rag_mvp` de khong pha store cu. Logic app hien tai da la English-only.

## 4. Ingestion Flow

Khi user upload file trong Streamlit:

1. `streamlit_app.py` nhan file upload.
2. `RagPipeline.ingest_file()` duoc goi.
3. `documents.load_file()` doc noi dung `.md`, `.txt`, hoac `.pdf`.
4. Text duoc normalize bang `preprocess_text()`:
   - Unicode NFC
   - chuan hoa newline
   - bo control characters
   - rut gon khoang trang
5. `stable_id(filename)` tao `document_id`.
6. `stable_id(text)` tao `version_hash`.
7. Registry kiem tra document da index chua:
   - cung filename + cung content: bo qua
   - cung filename + content moi: xoa stale chunks cu roi index lai
8. `split_into_chunks()` tach document thanh chunks.
9. `VectorStore.add_chunks()` embed chunks bang `BAAI/bge-m3`.
10. Chroma luu embeddings, text, metadata.
11. JSON registry luu `document_id`, `filename`, `version_hash`, `chunk_ids`.

## 5. Retrieval And Answer Flow

Khi user dat cau hoi:

1. `streamlit_app.py` nhan question tu `st.chat_input`.
2. `RagPipeline.stream_answer(question)` duoc goi.
3. `VectorStore.search()` embed query.
4. Chroma tra ve top-k chunks theo cosine similarity.
5. `RagPipeline.build_prompt()` tao prompt:
   - chi tra loi bang tieng Anh
   - chi dung provided context
   - neu khong co thong tin thi noi answer not found in context
   - citation format `[S1]`, `[S2]`, ...
6. `OllamaClient.stream()` goi `/api/generate` cua Ollama voi stream mode.
7. Streamlit hien token streaming, latency, va retrieved contexts.

## 6. English-Only Behavior

Prompt hien tai trong `rag_mvp/pipeline.py` yeu cau:

```text
You are an English-only RAG assistant.
Answer only in English, even if the user's question uses another language.
```

Vi vay ban test app o phien ban English-only. Branch `multilingual` giu lai ban cu de doi chieu.

## 7. Registry And Vector Store

Registry:

```text
vector_store/rag_mvp_registry.json
```

Chroma store:

```text
vector_store/rag_mvp/
```

Hai thu muc/file nay la runtime state local. Chung khong nen push mac dinh vi co the lon, phu thuoc may, va de rebuild bang ingestion.

`reports/` thi duoc track de push evaluation artifacts.

## 8. Evaluation Dataset

Project hien co 3 split cho document `w18347.pdf`:

- `evaluation/w18347_dev_eval.csv`: 30 cau
- `evaluation/w18347_holdout_eval.csv`: 30 cau
- `evaluation/w18347_stress_eval.csv`: 20 cau

Tong: 80 cau.

Schema:

```text
id
evaluation_split
document_profile
question_type
difficulty
expected_behavior
question
ground_truth
required_citation
notes
```

Y nghia split:

- Dev: debug/tune prompt, top-k, citation behavior.
- Holdout: final reporting, khong nen tune theo tung loi.
- Stress: out-of-scope, unsupported claims, private/current info, format/language checks.

## 9. Evaluation Command

Neu da activate venv:

```powershell
python -m rag_mvp.run_evaluation --w18347-all
```

Chay mot split:

```powershell
python -m rag_mvp.run_evaluation --dataset evaluation\w18347_holdout_eval.csv
```

Chay nhieu dataset tuy chon:

```powershell
python -m rag_mvp.run_evaluation --datasets evaluation\w18347_dev_eval.csv evaluation\w18347_holdout_eval.csv evaluation\w18347_stress_eval.csv
```

RAGAS core metrics chay mac dinh. `--skip-ragas` chi dung de debug nhanh, khong dung cho final report.

## 10. Evaluation Outputs

Voi dataset `evaluation/w18347_holdout_eval.csv`, output mac dinh:

```text
reports/w18347_holdout_eval_results.csv
reports/w18347_holdout_eval_ragas_results.csv
reports/w18347_holdout_eval_summary.md
```

Khi chay `--w18347-all`, co them:

```text
reports/evaluation_summary.md
```

## 11. Progress Logging

CLI evaluation in progress ra terminal:

```text
=== Evaluating evaluation/w18347_holdout_eval.csv ===
Custom metrics output: reports/w18347_holdout_eval_results.csv
[custom 1/30] What NBER working paper number is assigned to the document?
[custom 2/30] What JEL codes are listed for the paper?
...
[ragas] Running core metrics for evaluation/w18347_holdout_eval.csv
RAGAS metrics output: reports/w18347_holdout_eval_ragas_results.csv
```

Muc dich la de biet script dang chay, khong phai bi treo.

## 12. Metrics

### 12.1 RAGAS Core Metrics

RAGAS chay cac metric core:

- Faithfulness
- Context Recall
- Context Precision
- Answer Relevancy

Day la nhom metric core, khong phai optional trong final evaluation.

### 12.2 Custom Core Metrics

Custom metrics bo sung:

- Latency
- Refusal Rate
- Hallucination Rate

`Latency` la thoi gian retrieve + generate cho tung cau.

`Refusal Rate` la ty le answer bi detect la refusal.

`Hallucination Rate` la proxy custom:

```text
1 - unsupported_claim_accuracy
```

### 12.3 Additional Custom Metrics

File summary cung hien:

- `false_refusal`
- `expected_behavior_accuracy`
- `citation_accuracy`
- `citation_strict_accuracy`
- `unsupported_claim_accuracy`

`false_refusal`: model tu choi sai khi question dang ra co the tra loi.

`expected_behavior_accuracy`: model co dung behavior mong doi theo CSV khong:

- `answer`
- `refuse`
- `state_not_supported`
- `claim_verification`

`citation_accuracy`: citation `[S1]`, `[S2]` co nam trong retrieved context index hop le khong.

`citation_strict_accuracy`: citation co context chua keyword lien quan den `ground_truth` khong.

`unsupported_claim_accuracy`: cac claim trong answer co duoc support boi retrieved contexts khong.

## 13. RAGAS Data Flow

Custom evaluation tao CSV truoc, gom:

- question
- ground_truth
- answer
- contexts JSON
- custom metrics

Sau do RAGAS doc custom results CSV va tao dataset gom:

```text
question
answer
contexts
ground_truth
reference
```

RAGAS output duoc luu rieng vao:

```text
reports/<dataset_stem>_ragas_results.csv
```

Markdown summary doc ca custom CSV va RAGAS CSV de tong hop metric.

## 14. Development Checks

Lint:

```powershell
ruff check README.md streamlit_app.py rag_mvp tests evaluation
```

Tests:

```powershell
pytest -q tests/test_rag_mvp.py
```

Neu venv da activate, dung `python`, `pytest`, `ruff` truc tiep. Neu chua activate, dung duong dan day du trong `.venv\Scripts`.

## 15. Common Pitfalls

1. Chua index document ma chay evaluation:
   - retrieval khong co context
   - metrics khong co y nghia

2. Dataset khong khop document:
   - model co the tra loi sai hoac refuse dung
   - metrics khong phan anh dung chat luong RAG

3. Ground truth la placeholder:
   - RAGAS va strict citation khong dang tin

4. RAGAS thieu evaluator setup:
   - command co the fail o buoc RAGAS
   - can cai dat/cau hinh RAGAS theo environment dang dung

5. `.venv\Scripts\python.exe` bi tro toi Python path cu:
   - recreate venv la cach sach nhat
   - sau khi activate venv, command ngan la `python -m rag_mvp.run_evaluation --w18347-all`

