from __future__ import annotations

import time
from collections.abc import Callable, Iterator

from rag_mvp.config import AppConfig
from rag_mvp.documents import load_file, split_into_chunks
from rag_mvp.ollama_client import OllamaClient
from rag_mvp.registry import DocumentRecord, JsonDocumentRegistry
from rag_mvp.vector_store import VectorStore


SYSTEM_PROMPT = (
    "You are a multilingual RAG assistant.\n"
    "Answer in the same language as the user's question.\n"
    "If the question is in English, answer in English. If the question is in Vietnamese, answer in Vietnamese.\n"
    "Do not answer in Chinese unless the user's question is explicitly in Chinese.\n"
    "Use only the provided context. If the context is insufficient, say that the answer was not found in the context, "
    "using the same language as the question.\n"
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
        document = load_file(filename, content)
        self._progress(progress_callback, 0.25, f"Loaded {filename}")

        existing = self.registry.get(document.metadata["document_id"])
        if existing and existing.version_hash == document.metadata["version_hash"]:
            self._progress(progress_callback, 1.0, f"{filename} is already indexed")
            return existing
        if existing:
            self._progress(progress_callback, 0.35, f"Removing stale chunks for {filename}")
            self.vector_store.delete_document(existing.document_id, existing.chunk_ids)

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

    def reset(self) -> None:
        self.vector_store.reset()
        self.registry.clear()

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        return self.vector_store.search(question, top_k or self.config.top_k)

    def build_prompt(self, question: str, contexts: list[dict]) -> str:
        context_blocks = []
        for index, ctx in enumerate(contexts, start=1):
            filename = ctx["metadata"].get("filename", "unknown")
            context_blocks.append(f"[S{index}] {filename}\n{ctx['text']}")
        context_text = "\n\n".join(context_blocks)
        return (
            f"{SYSTEM_PROMPT}\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer in the same language as the question, with citations:\n"
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
