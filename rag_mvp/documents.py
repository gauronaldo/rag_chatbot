from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO
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


@dataclass
class StructuredBlock:
    text: str
    page: int | None
    section: str
    content_type: str = "text"
    table_id: str | None = None


def stable_id(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def preprocess_text(text: str) -> str:
    """Normalize document text before chunking."""
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


def _normalize_section_title(line: str) -> str | None:
    line = line.strip()
    markdown_match = re.match(r"^(#{1,6})\s+(.+)$", line)
    if markdown_match:
        return markdown_match.group(2).strip()

    numbered_match = re.match(r"^(\d+(?:\.\d+)*\.?)\s+([A-Z][^\n]{2,120})$", line)
    if numbered_match and len(line.split()) <= 14:
        return line.rstrip(".")

    keywords = {
        "abstract",
        "introduction",
        "conclusion",
        "references",
        "appendix",
        "data",
        "results",
        "methodology",
        "model",
        "discussion",
    }
    lowered = line.lower().strip(":")
    if lowered in keywords:
        return line.strip(":")
    return None


def _markdown_table_from_rows(rows: list[list[object | None]]) -> str:
    clean_rows = [[str(cell or "").strip().replace("\n", " ") for cell in row] for row in rows if row]
    clean_rows = [row for row in clean_rows if any(cell for cell in row)]
    if not clean_rows:
        return ""

    max_cols = max(len(row) for row in clean_rows)
    padded = [row + [""] * (max_cols - len(row)) for row in clean_rows]
    header = padded[0]
    body = padded[1:]
    separator = ["---"] * max_cols
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _looks_like_markdown_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    return "|" in lines[0] and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[1])


def _extract_table_number(text: str, fallback_index: int | None = None) -> str | None:
    match = re.search(r"\btable\s+([a-z]?\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    if fallback_index is not None:
        return str(fallback_index)
    return None


def _blocks_from_markdown(text: str, default_page: int | None = None) -> list[StructuredBlock]:
    blocks: list[StructuredBlock] = []
    section = "Document"
    paragraph: list[str] = []
    table_lines: list[str] = []
    table_index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        content = "\n".join(paragraph).strip()
        if content:
            blocks.append(StructuredBlock(content, default_page, section))
        paragraph = []

    def flush_table() -> None:
        nonlocal table_lines, table_index
        content = "\n".join(table_lines).strip()
        if content:
            table_index += 1
            table_id = _extract_table_number(content, table_index)
            blocks.append(StructuredBlock(content, default_page, section, "table", table_id))
        table_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush_table()
            flush_paragraph()
            continue

        heading = _normalize_section_title(line)
        if heading:
            flush_table()
            flush_paragraph()
            section = heading
            continue

        if "|" in line and (table_lines or line.count("|") >= 2):
            flush_paragraph()
            table_lines.append(line)
            continue

        flush_table()
        paragraph.append(line)

    flush_table()
    flush_paragraph()
    return blocks


def _blocks_to_markdown(blocks: list[StructuredBlock]) -> str:
    lines: list[str] = []
    current_section = ""
    for block in blocks:
        if block.section != current_section:
            current_section = block.section
            lines.append(f"## {current_section}")
        if block.page is not None:
            lines.append(f"<!-- page: {block.page} -->")
        if block.content_type == "table":
            table_label = f"Table {block.table_id}" if block.table_id else "Table"
            lines.append(f"**{table_label}**")
        lines.append(block.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def _read_pdf_structured(content: bytes) -> tuple[str, list[StructuredBlock]]:
    try:
        import pdfplumber
    except ImportError:
        text = read_pdf(BytesIO(content))
        blocks = _blocks_from_markdown(text)
        return _blocks_to_markdown(blocks), blocks

    blocks: list[StructuredBlock] = []
    current_section = "Document"
    table_index = 0

    with pdfplumber.open(BytesIO(content)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = preprocess_text(page.extract_text() or "")
            page_blocks = _blocks_from_markdown(page_text, default_page=page_index)
            for block in page_blocks:
                if block.section != "Document":
                    current_section = block.section
                block.section = current_section
                blocks.append(block)

            for table in page.extract_tables() or []:
                table_markdown = _markdown_table_from_rows(table)
                if not table_markdown:
                    continue
                table_index += 1
                table_id = _extract_table_number(table_markdown, table_index)
                blocks.append(
                    StructuredBlock(
                        text=table_markdown,
                        page=page_index,
                        section=current_section,
                        content_type="table",
                        table_id=table_id,
                    )
                )

    markdown = _blocks_to_markdown(blocks)
    return markdown, blocks


def _block_metadata(block: StructuredBlock) -> dict:
    return {
        "page": block.page,
        "section": block.section,
        "content_type": block.content_type,
        "table_id": block.table_id,
    }


def _prepared_block_text(block: StructuredBlock) -> str:
    prefix_parts = [f"Section: {block.section}"]
    if block.page is not None:
        prefix_parts.append(f"Page: {block.page}")
    if block.content_type == "table":
        table_label = f"Table {block.table_id}" if block.table_id else "Table"
        prefix_parts.append(f"Content type: {table_label}")
    return "\n".join(prefix_parts) + "\n\n" + block.text.strip()


def _split_long_text(text: str, metadata: dict, chunk_size: int, chunk_overlap: int) -> list[tuple[str, dict]]:
    chunks: list[tuple[str, dict]] = []
    text = text.strip()
    while len(text) > chunk_size:
        chunks.append((text[:chunk_size], metadata))
        text = text[chunk_size - chunk_overlap :]
    if text:
        chunks.append((text, metadata))
    return chunks


def load_file(filename: str, content: bytes) -> Document:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    structured_blocks: list[StructuredBlock]
    if suffix == ".pdf":
        text, structured_blocks = _read_pdf_structured(content)
    else:
        raw_text = decode_text(content)
        text = preprocess_text(raw_text)
        structured_blocks = _blocks_from_markdown(text)

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
            "structured_blocks": structured_blocks,
        },
    )


def split_into_chunks(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    blocks = document.metadata.get("structured_blocks") or _blocks_from_markdown(document.text)
    chunks: list[tuple[str, dict]] = []
    current_text = ""
    current_metadata: dict | None = None

    def flush_current() -> None:
        nonlocal current_text, current_metadata
        if current_text and current_metadata:
            chunks.extend(_split_long_text(current_text, current_metadata, chunk_size, chunk_overlap))
        current_text = ""
        current_metadata = None

    for block in blocks:
        if not block.text.strip():
            continue

        block_metadata = _block_metadata(block)
        prepared_text = _prepared_block_text(block)

        if block.content_type == "table":
            flush_current()
            chunks.extend(_split_long_text(prepared_text, block_metadata, chunk_size, chunk_overlap))
            continue

        candidate = f"{current_text}\n\n{prepared_text}".strip() if current_text else prepared_text
        same_section = current_metadata and current_metadata.get("section") == block.section
        if not current_text or (same_section and len(candidate) <= chunk_size):
            current_text = candidate
            current_metadata = block_metadata
        else:
            flush_current()
            current_text = prepared_text
            current_metadata = block_metadata

    flush_current()

    result: list[Chunk] = []
    base_metadata = {key: value for key, value in document.metadata.items() if key != "structured_blocks"}
    for index, (text, chunk_metadata) in enumerate(chunks):
        chunk_id = stable_id(f"{document.metadata['document_id']}:{index}:{text}")
        metadata = {
            **base_metadata,
            **{key: value for key, value in chunk_metadata.items() if value is not None},
            "chunk_index": index,
            "chunk_id": chunk_id,
        }
        result.append(Chunk(text=text, metadata=metadata))
    return result
