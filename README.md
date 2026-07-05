# English RAG Assistant

Local-first Retrieval-Augmented Generation MVP for English document Q&A.

This project is designed as a portfolio-ready RAG application: small enough to run locally, but complete enough to show
the full pipeline from ingestion to retrieval, generation, citation, and evaluation.

## Features

- Streamlit app for upload, chat, and source inspection.
- Local LLM through Ollama, defaulting to `qwen2.5:7b`.
- Retrieval embeddings with `BAAI/bge-m3`.
- Chroma persistent vector store.
- JSON document registry for incremental indexing and precise chunk deletion.
- English-only prompting with source citations like `[S1]`.
- Evaluation layer with RAGAS core metrics plus custom MVP metrics.

## Architecture

The MVP has one local Streamlit entrypoint and a small Python package:

1. Users upload `.md`, `.txt`, or `.pdf` files in Streamlit.
2. The loader converts documents into structured Markdown internally.
3. PDF ingestion detects pages, headings/sections, and tables where possible.
4. The chunker splits by section and keeps tables as table-aware chunks.
5. `BAAI/bge-m3` embeds chunks and stores them in Chroma with page, section, and content-type metadata.
6. The JSON registry stores document IDs, version hashes, and chunk IDs.
7. User questions retrieve candidate chunks from Chroma and apply metadata-aware boosting for table/section queries.
8. The prompt instructs Ollama/Qwen to answer in English with citations.
9. The app displays the answer, latency, retrieved contexts, and retrieval scores.

## Setup

Install Python 3.12 and Ollama first.

Pull the local Qwen model:

```powershell
ollama pull qwen2.5:7b
```

Create a virtual environment and install dependencies:

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
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

## Configuration

The app can be configured with environment variables:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:7b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:EMBEDDING_MODEL = "BAAI/bge-m3"
$env:TOP_K = "4"
$env:CHUNK_SIZE = "1000"
$env:CHUNK_OVERLAP = "120"
```

Changing the embedding model requires rebuilding the vector store because embeddings from different models are not
compatible.

## How It Works

1. Upload `.md`, `.txt`, or `.pdf` files in the sidebar.
2. The loader decodes text, normalizes Unicode to NFC, and converts content into structured Markdown internally.
3. PDF ingestion preserves page numbers and detects headings/sections. With `pdfplumber`, detected tables are converted
   to Markdown tables.
4. The chunker splits text by section and keeps table chunks separate, attaching metadata such as `page`, `section`,
   `content_type`, and `table_id`.
5. `BAAI/bge-m3` embeds chunks and stores them in Chroma.
6. The JSON registry stores document IDs, version hashes, and chunk IDs.
7. At question time, the app retrieves more candidates than the final `TOP_K`, then boosts table/section candidates
   when the query mentions table IDs, section IDs, or numeric/table-like terms.
8. Ollama runs Qwen locally and returns an English-only answer with citations.
9. The UI displays the answer, latency, retrieved contexts, base similarity, and metadata boost.

## Evaluation

The MVP includes custom evaluation metrics in `rag_mvp/evaluation.py`:

- False refusal rate
- Refusal precision
- Correct refusal behavior
- Evidence hit rate
- MRR
- Citation accuracy
- Citation strict accuracy
- Unsupported claim accuracy
- Latency

The RAGAS core runner targets:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

RAGAS uses the local Ollama judge configured by `OLLAMA_MODEL`, defaulting to `qwen2.5:7b`, and local
`BAAI/bge-m3` embeddings. It does not require `OPENAI_API_KEY`.

A sample evaluation file is available at `evaluation/sample_eval_set.csv`. Results are saved to
`reports/rag_mvp_eval_results.csv`.

Run evaluation after indexing documents. RAGAS core metrics run by default. If your virtual environment is already
activated, use `python`; otherwise call the venv Python explicitly.

```powershell
python -m rag_mvp.run_evaluation --dataset evaluation\w18347_holdout_eval.csv
```

Run all three w18347 splits in one command:

```powershell
python -m rag_mvp.run_evaluation --w18347-all
```

Test only the RAGAS evaluator on the dev split, reusing an existing custom results CSV:

```powershell
python -m rag_mvp.run_evaluation --dataset evaluation\w18347_dev_eval.csv --output reports\w18347_dev_eval_results.csv --ragas-only --ragas-raise-exceptions --ragas-num-ctx 8192
```

For local Ollama judges, RAGAS defaults to `--ragas-max-workers 1`, `--ragas-batch-size 1`,
`--ragas-timeout 600`, and `--ragas-answer-strictness 1` to avoid overloading the model during scoring.

Run multiple custom datasets in one command:

```powershell
python -m rag_mvp.run_evaluation --datasets evaluation\w18347_dev_eval.csv evaluation\w18347_holdout_eval.csv evaluation\w18347_stress_eval.csv
```

By default this writes:

```text
reports/<dataset_stem>_results.csv
reports/<dataset_stem>_ragas_results.csv
reports/<dataset_stem>_summary.md
```

When running multiple datasets, the script also writes `reports/evaluation_summary.md`.

Each Markdown summary reports metrics by RAG stage, valid counts, NaN counts, and a strict average where NaN is counted
as 0. Answerable rows are separated from refusal/not-supported rows because RAGAS factual QA metrics do not directly
measure correct refusal behavior.

For quick debugging without RAGAS, use `--skip-ragas`; final evaluation should not skip it because Faithfulness,
Context Recall, Context Precision, and Answer Relevancy are core metrics.

## JSON Registry

The MVP uses `vector_store/rag_mvp_registry.json` to track indexed documents, version hashes, and Chroma chunk IDs.
This keeps setup simple and avoids database services for a local portfolio demo.

## Project Layout

```text
rag_mvp/
  config.py          Runtime config
  documents.py       Loading, structured Markdown conversion, section/table chunking
  registry.py        JSON document registry
  vector_store.py    Chroma + BAAI/bge-m3 retrieval with metadata-aware boosting
  ollama_client.py   Local Ollama generation
  pipeline.py        Ingest, retrieve, answer orchestration
  evaluation.py      RAG evaluation helpers
streamlit_app.py     Streamlit MVP UI
evaluation/          Sample evaluation dataset
tests/               Unit tests
requirements.txt     Python dependencies for venv + pip setup
```

The MVP entrypoint is `streamlit_app.py`.
