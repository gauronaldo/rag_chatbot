from __future__ import annotations

import re
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

    @staticmethod
    def _query_features(query: str) -> dict:
        lowered = query.lower()
        table_match = re.search(r"\btable\s+([a-z]?\d+(?:\.\d+)?)\b", lowered)
        figure_match = re.search(r"\bfig(?:ure)?\.?\s+([a-z]?\d+(?:\.\d+)?)\b", lowered)
        section_match = re.search(r"\bsection\s+(\d+(?:\.\d+)*)\b", lowered)
        section_names = {
            "abstract": "Abstract",
            "introduction": "Introduction",
            "conclusion": "Concluding Remarks",
            "concluding remarks": "Concluding Remarks",
            "references": "References",
            "appendix": "Appendix",
            "data": "Data",
            "results": "Results",
            "methodology": "Methodology",
            "discussion": "Discussion",
        }
        requested_section = next((title for term, title in section_names.items() if term in lowered), None)
        numeric_terms = {
            "percent",
            "percentage",
            "rate",
            "mean",
            "median",
            "coefficient",
            "estimate",
            "ratio",
            "value",
            "number",
            "table",
            "increase",
            "decrease",
        }
        return {
            "table_id": table_match.group(1) if table_match else None,
            "figure_id": figure_match.group(1) if figure_match else None,
            "section_number": section_match.group(1) if section_match else None,
            "section_name": requested_section,
            "wants_table": bool(table_match) or any(term in lowered for term in numeric_terms),
            "wants_figure": bool(figure_match) or "figure" in lowered or "fig." in lowered,
        }

    @staticmethod
    def _metadata_boost(query_features: dict, metadata: dict) -> float:
        boost = 0.0
        content_type = str(metadata.get("content_type", "")).lower()
        section = str(metadata.get("section", "")).lower()
        table_id = str(metadata.get("table_id", "")).lower()
        figure_id = str(metadata.get("figure_id", "")).lower()

        if query_features["wants_table"] and content_type == "table":
            boost += 0.08
        if query_features["table_id"] and query_features["table_id"] == table_id:
            boost += 0.18
        if query_features["wants_figure"] and content_type == "figure_caption":
            boost += 0.08
        if query_features["figure_id"] and query_features["figure_id"] == figure_id:
            boost += 0.18
        if query_features["section_number"] and section.startswith(query_features["section_number"]):
            boost += 0.12
        if query_features["section_name"] and section == query_features["section_name"].lower():
            boost += 0.2
        return boost

    def _metadata_candidates(self, query_features: dict, top_k: int) -> list[dict]:
        where = None
        if query_features["section_name"]:
            where = {"section": query_features["section_name"]}
        elif query_features["table_id"]:
            where = {"table_id": query_features["table_id"]}
        elif query_features["figure_id"]:
            where = {"figure_id": query_features["figure_id"]}
        if where is None:
            return []

        result = self.collection.get(where=where, limit=top_k)
        rows: list[dict] = []
        for doc, metadata, chunk_id in zip(
            result.get("documents", []),
            result.get("metadatas", []),
            result.get("ids", []),
        ):
            metadata = metadata or {}
            boost = self._metadata_boost(query_features, metadata)
            rows.append(
                {
                    "id": chunk_id,
                    "text": doc,
                    "metadata": metadata,
                    "score": round(1.0 + boost, 4),
                    "base_score": 1.0,
                    "metadata_boost": round(boost, 4),
                    "match_type": "metadata",
                }
            )
        return rows

    def search(self, query: str, top_k: int) -> list[dict]:
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedder.encode(query, normalize_embeddings=True, show_progress_bar=False).tolist()
        candidate_count = min(max(top_k * 10, 50), self.collection.count())
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
        )
        rows: list[dict] = []
        query_features = self._query_features(query)
        rows.extend(self._metadata_candidates(query_features, top_k))
        for doc, metadata, distance, chunk_id in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        ):
            metadata = metadata or {}
            base_score = 1 - float(distance)
            boost = self._metadata_boost(query_features, metadata)
            rows.append(
                {
                    "id": chunk_id,
                    "text": doc,
                    "metadata": metadata,
                    "score": round(base_score + boost, 4),
                    "base_score": round(base_score, 4),
                    "metadata_boost": round(boost, 4),
                    "match_type": "vector",
                }
            )
        deduped: dict[str, dict] = {}
        for row in sorted(rows, key=lambda row: row["score"], reverse=True):
            deduped.setdefault(row["id"], row)
        return list(deduped.values())[:top_k]
