from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

from rag_mvp.config import AppConfig
from rag_mvp.documents import Document, load_file, source_version_hash, split_into_chunks, stable_id
from rag_mvp.ollama_client import OllamaClient
from rag_mvp.registry import DocumentRecord, JsonDocumentRegistry
from rag_mvp.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are an English-only RAG assistant.\n"
    "Answer only in English, even if the user's question uses another language.\n"
    "Use only the provided context. If the context is insufficient, say that the answer was not found in the context, "
    "in English.\n"
    "Keep the answer concise and structured.\n"
    "When using information from documents, cite sources as [S1], [S2], ...\n"
)


class RagPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.registry = JsonDocumentRegistry(config.registry_path)
        self.vector_store = VectorStore(config.vector_store_dir, config.collection_name, config.embedding_model)
        self.llm = OllamaClient(config.ollama_base_url, config.ollama_model, config.temperature)

    def ingest_file(
        self,
        filename: str,
        content: bytes,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> DocumentRecord:
        self._progress(progress_callback, 0.05, f"Loading {filename}")
        doc_id = stable_id(filename)
        content_version_hash = source_version_hash(filename, content)
        existing = self.registry.get(doc_id)
        if existing and existing.version_hash == content_version_hash:
            self._progress(progress_callback, 1.0, f"{filename} is already indexed")
            return existing

        document = load_file(filename, content)
        self._progress(progress_callback, 0.25, f"Loaded {filename}")

        existing = self.registry.get(document.metadata["document_id"])
        if existing and existing.version_hash == document.metadata["version_hash"]:
            self._progress(progress_callback, 1.0, f"{filename} is already indexed")
            return existing
        if existing:
            self._progress(progress_callback, 0.35, f"Removing stale chunks for {filename}")
            self.vector_store.delete_document(existing.document_id, existing.chunk_ids)
            self._delete_processed_artifacts(existing.document_id)

        self._progress(progress_callback, 0.4, f"Saving structured artifacts for {filename}")
        self._save_processed_artifacts(document)

        self._progress(progress_callback, 0.45, f"Chunking {filename}")
        chunks = split_into_chunks(document, self.config.chunk_size, self.config.chunk_overlap)

        self._progress(progress_callback, 0.65, f"Embedding {len(chunks)} chunks")
        chunk_ids = self.vector_store.add_chunks(chunks)
        record = DocumentRecord(
            document_id=document.metadata["document_id"],
            filename=document.metadata["filename"],
            source=document.metadata["source"],
            size=document.metadata["size"],
            version_hash=document.metadata["version_hash"],
            chunk_ids=chunk_ids,
        )
        self._progress(progress_callback, 0.9, f"Saving registry record for {filename}")
        self.registry.upsert(record)
        self._progress(progress_callback, 1.0, f"Indexed {filename}")
        return record

    def delete_document(self, document_id: str) -> None:
        record = self.registry.get(document_id)
        if not record:
            return
        self.vector_store.delete_document(record.document_id, record.chunk_ids)
        self.registry.remove(document_id)
        self._delete_processed_artifacts(record.document_id)

    def reset(self) -> None:
        self.vector_store.reset()
        self.registry.clear()
        self._clear_processed_artifacts()

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        return self.vector_store.search(question, top_k or self.config.top_k)

    def build_prompt(self, question: str, contexts: list[dict]) -> str:
        context_blocks = []
        for index, ctx in enumerate(contexts, start=1):
            metadata = ctx["metadata"]
            filename = metadata.get("filename", "unknown")
            page = metadata.get("page")
            section = metadata.get("section")
            content_type = metadata.get("content_type")
            table_id = metadata.get("table_id")
            figure_id = metadata.get("figure_id")
            source_parts = [filename]
            if page:
                source_parts.append(f"page {page}")
            if section:
                source_parts.append(f"section: {section}")
            if content_type:
                source_parts.append(f"type: {content_type}")
            if table_id:
                source_parts.append(f"table: {table_id}")
            if figure_id:
                source_parts.append(f"figure: {figure_id}")
            context_blocks.append(f"[S{index}] {'; '.join(source_parts)}\n{ctx['text']}")
        context_text = "\n\n".join(context_blocks)
        return (
            f"{SYSTEM_PROMPT}\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer in English with citations:\n"
        )

    def answer(self, question: str) -> dict:
        started = time.perf_counter()
        contexts = self.retrieve(question)
        prompt = self.build_prompt(question, contexts)
        answer = self.llm.generate(prompt)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"answer": answer, "contexts": contexts, "latency_ms": latency_ms}

    def stream_answer(self, question: str) -> tuple[list[dict], Iterator[str], float]:
        started = time.perf_counter()
        contexts = self.retrieve(question)
        prompt = self.build_prompt(question, contexts)
        return contexts, self.llm.stream(prompt), started

    @staticmethod
    def _progress(
        progress_callback: Callable[[float, str], None] | None,
        progress: float,
        message: str,
    ) -> None:
        if progress_callback:
            progress_callback(progress, message)

    def _artifact_prefix(self, document: Document) -> str:
        filename = Path(document.metadata["filename"]).stem
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "document"
        return f"{document.metadata['document_id']}_{safe_name}"

    def _save_processed_artifacts(self, document: Document) -> None:
        self.config.processed_docs_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._artifact_prefix(document)
        markdown_path = self.config.processed_docs_dir / f"{prefix}.structured.md"
        blocks_path = self.config.processed_docs_dir / f"{prefix}.blocks.json"
        blocks = document.metadata.get("structured_blocks") or []

        markdown_path.write_text(document.text, encoding="utf-8")
        blocks_payload = {
            "document_id": document.metadata["document_id"],
            "filename": document.metadata["filename"],
            "version_hash": document.metadata["version_hash"],
            "blocks": [asdict(block) for block in blocks],
        }
        blocks_path.write_text(json.dumps(blocks_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _delete_processed_artifacts(self, document_id: str) -> None:
        if not self.config.processed_docs_dir.exists():
            return
        for path in self.config.processed_docs_dir.glob(f"{document_id}_*"):
            if path.is_file():
                path.unlink()

    def _clear_processed_artifacts(self) -> None:
        if not self.config.processed_docs_dir.exists():
            return
        for path in self.config.processed_docs_dir.glob("*"):
            if path.is_file():
                path.unlink()
