from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    source: str
    size: int
    version_hash: str
    chunk_ids: list[str]


class JsonDocumentRegistry:
    """Small JSON registry for MVP incremental indexing."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[DocumentRecord]:
        return list(self._load().values())

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._load().get(document_id)

    def upsert(self, record: DocumentRecord) -> None:
        data = self._load()
        data[record.document_id] = record
        self._save(data)

    def remove(self, document_id: str) -> None:
        data = self._load()
        data.pop(document_id, None)
        self._save(data)

    def clear(self) -> None:
        self._save({})

    def _load(self) -> dict[str, DocumentRecord]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {doc_id: DocumentRecord(**record) for doc_id, record in raw.items()}

    def _save(self, data: dict[str, DocumentRecord]) -> None:
        payload = {doc_id: asdict(record) for doc_id, record in data.items()}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
