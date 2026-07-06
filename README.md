# English RAG Assistant

Local-first Retrieval-Augmented Generation MVP for English document Q&A.

This project is designed as a portfolio-ready RAG application: it runs locally, indexes PDFs into a persistent vector
store, answers with source citations, and includes an end-to-end evaluation pipeline with RAGAS plus custom metrics.

## Highlights

- Streamlit app for upload, ingestion progress, document-scoped chat, and retrieved-source inspection.
- Local generation through Ollama, defaulting to `qwen2.5:7b`.
- Local embeddings with `BAAI/bge-m3`.
- Chroma persistent vector store.
- JSON registry for incremental indexing and precise chunk deletion.
- PDF-to-Markdown ingestion with PyMuPDF4LLM.
- Section, table, and figure-caption metadata for retrieval debugging.
- English-only answering with citations such as `[S1]`.
- Evaluation pipeline with RAGAS core metrics and custom RAG metrics.
- Saved evaluation reports for dev, holdout, stress, and combined reporting.

## Current Scope

The current branch focuses on one English-only RAG MVP. It does not use the older backend/frontend architecture, SQLite,
or MySQL. Multilingual behavior was intentionally removed from this branch so the retrieval and evaluation pipeline can
be tested cleanly in English.

The showcase document is:

```text
w18347.pdf
```

## Architecture

```text
app.py
  Streamlit UI
  - upload files
  - show ingestion progress
  - select one indexed document or all documents as the retrieval scope
  - chat with the indexed corpus
  - inspect retrieved contexts and scores

rag_mvp/
  config.py          Runtime configuration and default paths
  documents.py       PDF/Markdown/Text loading, PyMuPDF4LLM conversion, structured block parsing, chunking
  registry.py        JSON document registry
  vector_store.py    Chroma + BAAI/bge-m3 retrieval with metadata-aware boosting
  ollama_client.py   Ollama generation client
  pipeline.py        Ingestion, retrieval, prompting, answer orchestration
  evaluation.py      Custom metrics and RAGAS integration helpers
  run_evaluation.py  Evaluation CLI
  debug_retrieval.py Retrieval/chunking diagnostic CLI

evaluation/
  w18347_dev_eval.csv
  w18347_holdout_eval.csv
  w18347_stress_eval.csv

processed_docs/
  <document_id>_<filename>.structured.md
  <document_id>_<filename>.blocks.json

reports/
  w18347_dev_results.csv
  w18347_dev_summary.md
  w18347_holdout_results.csv
  w18347_holdout_summary.md
  w18347_stress_results.csv
  w18347_stress_summary.md
  evaluation_summary.md
```

## Setup

Install Python 3.12 and Ollama first.

Pull the default local model:

```powershell
ollama pull qwen2.5:7b
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

Run the app:

```powershell
streamlit run app.py
```

Open the Streamlit URL printed in the terminal, usually `http://localhost:8501`.

## Configuration

Runtime settings are configured with environment variables:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:7b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_TEMPERATURE = "0.2"
$env:EMBEDDING_MODEL = "BAAI/bge-m3"
$env:CHROMA_COLLECTION = "structured_rag_mvp"
$env:TOP_K = "4"
$env:CHUNK_SIZE = "1000"
$env:CHUNK_OVERLAP = "120"
```

Changing `EMBEDDING_MODEL` requires rebuilding the vector store because embeddings from different models are not
compatible.

## Ingestion Flow

1. The user uploads `.pdf`, `.md`, or `.txt` files in `app.py`.
2. `RagPipeline.ingest_file()` computes a document ID and source version hash.
3. The registry skips ingestion if the same file content is already indexed.
4. PDFs are converted to Markdown with PyMuPDF4LLM.
5. The loader normalizes text and parses structured blocks.
6. Structured Markdown and block metadata are saved under `processed_docs/` as local debug artifacts.
7. The chunker splits by section and separates table / figure-caption chunks where possible.
8. Each chunk receives metadata such as filename, page, section, content type, table ID, and figure ID.
9. Chunks are embedded with `BAAI/bge-m3`.
10. Chunks and metadata are stored in Chroma.
11. The JSON registry stores document IDs, version hashes, and Chroma chunk IDs.

The saved `processed_docs/` files are intentionally useful during local debugging, but they are ignored by git because
they are generated from source documents and can be rebuilt by re-ingesting.

## Retrieval And Answering Flow

1. The user asks a question in the Streamlit chat.
2. The question is embedded with the same embedding model.
3. Chroma retrieves candidate chunks.
4. If a document is selected in the sidebar, retrieval is filtered to that document ID.
5. Metadata-aware boosting promotes candidates that match explicit section, table, figure, numeric, or content-type
signals in the question.
6. The final contexts are inserted into a grounded prompt.
7. Ollama/Qwen answers using only the provided contexts.
8. The answer is returned in English with citations.
9. The UI shows latency and the retrieved contexts, including similarity score and metadata boost.

The prompt is intentionally conservative: if the retrieved context is insufficient, the model should say that the answer
was not found in the context.

## Evaluation

The evaluation pipeline runs outside the UI. It requires the target document to be indexed first.

Run all three evaluation sets:

```powershell
python -m rag_mvp.run_evaluation --w18347-all
```

Run one split:

```powershell
python -m rag_mvp.run_evaluation --dataset evaluation\w18347_holdout_eval.csv
```

Run several explicit datasets:

```powershell
python -m rag_mvp.run_evaluation --datasets evaluation\w18347_dev_eval.csv evaluation\w18347_holdout_eval.csv evaluation\w18347_stress_eval.csv
```

If the virtual environment is not activated, use the venv Python explicitly:

```powershell
.\.venv\Scripts\python.exe -m rag_mvp.run_evaluation --w18347-all
```

RAGAS runs by default. Use `--skip-ragas` only for quick debugging because Faithfulness, Context Recall, Context
Precision, and Answer Relevancy are core metrics for the final report.

Useful RAGAS stability options for local Ollama judges:

```powershell
python -m rag_mvp.run_evaluation --dataset evaluation\w18347_dev_eval.csv --ragas-raise-exceptions --ragas-num-ctx 8192 --ragas-max-workers 1 --ragas-batch-size 1
```

## Evaluation Outputs

Each dataset writes exactly one merged result CSV and one Markdown summary:

```text
reports/<dataset_stem_without_eval>_results.csv
reports/<dataset_stem_without_eval>_summary.md
```

For example:

```text
evaluation/w18347_holdout_eval.csv
reports/w18347_holdout_results.csv
reports/w18347_holdout_summary.md
```

When multiple datasets are run, the CLI also writes:

```text
reports/evaluation_summary.md
```

The result CSV contains row-level answers, retrieved contexts, custom metrics, and RAGAS metrics in one file. The
Markdown summaries report averages, strict averages, valid counts, and NaN counts.

## Metrics

RAGAS core metrics:

- Faithfulness
- Context Recall
- Context Precision
- Answer Relevancy

Custom metrics:

- Latency
- Refusal Rate
- False Refusal Rate
- Refusal Precision
- Correct Refusal Behavior
- Evidence Hit Rate
- MRR
- Citation Accuracy
- Citation Strict Accuracy
- Unsupported Claim Accuracy
- Hallucination Rate

Hallucination Rate is computed as:

```text
1 - unsupported_claim_accuracy
```

Answerable rows and refusal / not-supported rows are reported separately because RAGAS factual QA metrics are not the
best primary signal for correct refusal behavior.

## Latest Evaluation Snapshot

The latest full run contains 50 questions per split.

| Dataset | Faithfulness | Context Recall | Context Precision | Answer Relevancy | Hallucination Rate | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Dev | 0.8682 | 0.6017 | 0.6094 | 0.7439 | 0.0233 | 5564 ms |
| Holdout | 0.8177 | 0.6450 | 0.5789 | 0.7498 | 0.0339 | 5217 ms |
| Stress | 0.8080 | 0.6033 | 0.4789 | 0.2862 | 0.1420 | 6571 ms |

Interpretation:

- Dev and holdout show solid grounded-answer behavior for a local RAG MVP.
- Hallucination is low on dev and holdout.
- Stress is intentionally harder and mixes answerable, refusal, unsupported, and claim-verification rows.
- The main current weakness is false refusal on detailed table, definition, and numeric questions.
- Table/numeric retrieval and reranking are natural next improvements.

## Engineering Tradeoffs And Known Limitations

This MVP intentionally favors a transparent local architecture over a heavier production stack.

- **Local-first runtime:** Ollama, Chroma, JSON registry, and local embeddings keep the project easy to run without
  hosted services. The tradeoff is slower latency than managed inference and less robust evaluator reliability than a
  stronger hosted judge.
- **Conservative prompting:** The assistant is instructed to answer only from retrieved context. This keeps hallucination
  low, but can produce false refusals when retrieval misses a specific table, definition, or numeric fact.
- **Lightweight PDF parsing:** PyMuPDF4LLM is fast and simple for text-first papers. It does not interpret chart pixels
  or image-only figure content; figure support is caption-aware rather than vision-based.
- **Simple retrieval stack:** Dense retrieval plus metadata-aware boosting is explainable and portfolio-friendly. The
  current weakness is detailed table/numeric retrieval; a production version would add table row normalization, adjacent
  chunk expansion, and a reranker.
- **Evaluation scope:** The benchmark reports are for `w18347.pdf`. Multi-document chat is supported in the app, but the
  formal metrics should be read as single-document benchmark evidence.

## Retrieval Debugging

Use the diagnostic CLI when the answer is wrong but the information appears in `processed_docs/`:

```powershell
python -m rag_mvp.debug_retrieval "what is the abstract?" --document w18347 --skip-vector
```

The output helps isolate whether the problem is:

- PDF-to-Markdown conversion
- section/table/figure metadata parsing
- chunk reconstruction
- Chroma metadata lookup
- vector retrieval and metadata boosting

If parsing or chunks are wrong, reset and re-ingest after parser changes. If metadata matches are correct but final
retrieval is wrong, tune retrieval, boost logic, top-k, or add a reranker.

## Runtime Artifacts

Ignored local runtime state:

```text
vector_store/rag_mvp/
vector_store/rag_mvp_registry.json
processed_docs/
.env
.venv/
```

Tracked portfolio artifacts:

```text
reports/
evaluation/
```

This means another user can inspect the evaluation datasets and reports, while rebuilding local processed documents and
the vector store on their machine by ingesting `w18347.pdf`.
