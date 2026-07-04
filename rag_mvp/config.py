from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    app_title: str = "English RAG Assistant"
    docs_dir: Path = ROOT_PATH / "docs"
    vector_store_dir: Path = ROOT_PATH / "vector_store" / "rag_mvp"
    registry_path: Path = ROOT_PATH / "vector_store" / "rag_mvp_registry.json"
    reports_dir: Path = ROOT_PATH / "reports"
    eval_dataset_path: Path = ROOT_PATH / "evaluation" / "sample_eval_set.csv"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "vietnamese_rag_mvp")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))


def get_config() -> AppConfig:
    return AppConfig()
