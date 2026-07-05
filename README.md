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
2. The loader decodes and normalizes text.
3. The chunker splits documents and preserves source metadata.
4. `BAAI/bge-m3` embeds chunks and stores them in Chroma.
5. The JSON registry stores document IDs, version hashes, and chunk IDs.
6. User questions retrieve top-k chunks from Chroma.
7. The prompt instructs Ollama/Qwen to answer in English with citations.
8. The app displays the answer, latency, and retrieved contexts.

## Setup

Install Python 3.12 and Ollama first.

Pull the local Qwen model:

```powershell
ollama pull qwen2.5:7b
```

Create a virtual environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

Run the app:

```powershell
.\.venv\Scripts\streamlit run streamlit_app.py
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
2. The loader decodes text and normalizes Unicode to NFC.
3. The chunker splits documents while preserving source metadata.
4. `BAAI/bge-m3` embeds chunks and stores them in Chroma.
5. The JSON registry stores document IDs, version hashes, and chunk IDs.
6. At question time, the app retrieves top-k chunks and builds an English-only RAG prompt.
7. Ollama runs Qwen locally and returns an answer with citations.
8. The UI displays the answer, latency, and retrieved contexts.

## Evaluation

The MVP includes custom evaluation metrics in `rag_mvp/evaluation.py`:

- False refusal rate
- Citation accuracy
- Citation strict accuracy
- Unsupported claim accuracy
- Latency

The RAGAS core runner targets:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

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

Or, if `make` is available:

```powershell
make eval-w18347
```

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

For quick debugging without RAGAS, use `--skip-ragas`; final evaluation should not skip it because Faithfulness,
Context Recall, Context Precision, and Answer Relevancy are core metrics.

## JSON Registry

The MVP uses `vector_store/rag_mvp_registry.json` to track indexed documents, version hashes, and Chroma chunk IDs.
This keeps setup simple and avoids database services for a local portfolio demo.

## Project Layout

```text
rag_mvp/
  config.py          Runtime config
  documents.py       Loading, decoding, preprocessing, chunking
  registry.py        JSON document registry
  vector_store.py    Chroma + BAAI/bge-m3 retrieval
  ollama_client.py   Local Ollama generation
  pipeline.py        Ingest, retrieve, answer orchestration
  evaluation.py      RAG evaluation helpers
streamlit_app.py     Streamlit MVP UI
evaluation/          Sample evaluation dataset
tests/               Unit tests
requirements.txt     Python dependencies for venv + pip setup
```

The MVP entrypoint is `streamlit_app.py`.
