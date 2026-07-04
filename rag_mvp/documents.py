from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def stable_id(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def preprocess_text(text: str) -> str:
    """Normalize text for Vietnamese RAG without stripping accents."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1258"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_pdf(file_obj: BinaryIO) -> str:
    reader = PdfReader(file_obj)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_file(filename: str, content: bytes) -> Document:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    if suffix == ".pdf":
        from io import BytesIO

        raw_text = read_pdf(BytesIO(content))
    else:
        raw_text = decode_text(content)

    text = preprocess_text(raw_text)
    doc_id = stable_id(filename)
    version_hash = stable_id(text)
    return Document(
        text=text,
        metadata={
            "document_id": doc_id,
            "filename": Path(filename).name,
            "source": Path(filename).name,
            "extension": suffix,
            "size": len(content),
            "version_hash": version_hash,
        },
    )


def split_into_chunks(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paragraphs = re.split(r"\n\s*\n", document.text)
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = paragraph

        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap :]

    if current:
        chunks.append(current)

    result: list[Chunk] = []
    for index, text in enumerate(chunks):
        chunk_id = stable_id(f"{document.metadata['document_id']}:{index}:{text}")
        metadata = {
            **document.metadata,
            "chunk_index": index,
            "chunk_id": chunk_id,
        }
        result.append(Chunk(text=text, metadata=metadata))
    return result
