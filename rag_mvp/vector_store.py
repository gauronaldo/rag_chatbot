from __future__ import annotations

from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from rag_mvp.documents import Chunk


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedding_model: str):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model, device=device, trust_remote_code=True)

    def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=embeddings,
        )
        return ids

    def delete_document(self, document_id: str, chunk_ids: list[str] | None = None) -> None:
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
        else:
            self.collection.delete(where={"document_id": document_id})

    def reset(self) -> None:
        items = self.collection.get()
        ids = items.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def search(self, query: str, top_k: int) -> list[dict]:
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedder.encode(query, normalize_embeddings=True, show_progress_bar=False).tolist()
        result = self.collection.query(query_embeddings=[query_embedding], n_results=min(top_k, self.collection.count()))
        rows: list[dict] = []
        for doc, metadata, distance, chunk_id in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        ):
            rows.append(
                {
                    "id": chunk_id,
                    "text": doc,
                    "metadata": metadata or {},
                    "score": round(1 - float(distance), 4),
                }
            )
        return rows
