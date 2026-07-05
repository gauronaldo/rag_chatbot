from __future__ import annotations

import hashlib
import importlib.util
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
PARSER_VERSION = "docling-v1"


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


def source_version_hash(filename: str, content: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(PARSER_VERSION.encode("utf-8"))
    digest.update(Path(filename).suffix.lower().encode("utf-8"))
    digest.update(content)
    return digest.hexdigest()


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
    if numbered_match and len(line.split()) <= 14 and "@" not in line and "," not in line:
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


def _extract_figure_number(text: str) -> str | None:
    match = re.search(r"\bfig(?:ure)?\.?\s+([a-z]?\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _figure_caption_from_line(line: str) -> str | None:
    if re.match(r"^\s*(fig(?:ure)?\.?\s+[a-z]?\d+(?:\.\d+)?[\.:)]?)\s+", line, flags=re.IGNORECASE):
        return line.strip()
    return None


def _section_from_docling_heading(line: str) -> str | None:
    heading = _normalize_section_title(line)
    if heading:
        return heading
    return None


def _append_text_blocks(
    blocks: list[StructuredBlock],
    content: str,
    page: int | None,
    section: str,
) -> str:
    if section != "Abstract":
        blocks.append(StructuredBlock(content, page, section))
        return section

    if re.match(r"^\d+\s+[A-Z][^\n]+(?:Email:|@)", content):
        blocks.append(StructuredBlock(content, page, "Front Matter", "footnote"))
        return section

    acknowledgement_markers = (
        "i thank ",
        "this paper benefited from",
        "i acknowledge financial support",
    )
    if content.lower().startswith(acknowledgement_markers):
        blocks.append(StructuredBlock(content, page, "Front Matter", "acknowledgements"))
        return section

    keyword_match = re.match(r"(?is)^(Keywords?\b.*?\bJEL Codes?[^\n]*)(?:\n(.+))?$", content)
    if keyword_match:
        blocks.append(StructuredBlock(keyword_match.group(1).strip(), page, "Abstract", "keywords"))
        introduction_part = (keyword_match.group(2) or "").strip()
        if introduction_part:
            blocks.append(StructuredBlock(introduction_part, page, "Introduction"))
        return "Introduction"

    footnote_match = re.search(r"\n\d+\s+[A-Z][^\n]+(?:Email:|@)", content)
    if footnote_match:
        abstract_part = content[: footnote_match.start()].strip()
        footnote_part = content[footnote_match.start() :].strip()
        if abstract_part:
            blocks.append(StructuredBlock(abstract_part, page, "Abstract"))
        if footnote_part:
            blocks.append(StructuredBlock(footnote_part, page, "Front Matter", "footnote"))
        return section

    jel_match = re.search(r"(?is)(.*?\bJEL Codes?[^\n]*)(\n.+)", content)
    if jel_match:
        abstract_part = jel_match.group(1).strip()
        introduction_part = jel_match.group(2).strip()
        if abstract_part:
            blocks.append(StructuredBlock(abstract_part, page, "Abstract"))
        if introduction_part:
            blocks.append(StructuredBlock(introduction_part, page, "Introduction"))
        return "Introduction"

    blocks.append(StructuredBlock(content, page, section))
    return section


def _blocks_from_markdown(text: str, default_page: int | None = None) -> list[StructuredBlock]:
    blocks: list[StructuredBlock] = []
    section = "Document"
    paragraph: list[str] = []
    table_lines: list[str] = []
    table_index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph, section
        content = "\n".join(paragraph).strip()
        if content:
            section = _append_text_blocks(blocks, content, default_page, section)
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

        figure_caption = _figure_caption_from_line(line)
        if figure_caption:
            flush_table()
            flush_paragraph()
            blocks.append(
                StructuredBlock(
                    figure_caption,
                    default_page,
                    section,
                    "figure_caption",
                    _extract_figure_number(figure_caption),
                )
            )
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
        if block.content_type == "figure_caption":
            figure_label = f"Figure {block.table_id}" if block.table_id else "Figure"
            lines.append(f"**{figure_label} caption**")
        lines.append(block.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def _convert_pdf_with_docling(content: bytes) -> str:
    if importlib.util.find_spec("docling") is None:
        raise RuntimeError(
            "PDF ingestion on the docling-test branch requires Docling. "
            "Install it with `pip install -r requirements.txt`."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "uploaded.pdf"
        pdf_path.write_bytes(content)
        result = _docling_converter().convert(pdf_path)
        document = result.document
        if hasattr(document, "export_to_markdown"):
            return document.export_to_markdown()
        if hasattr(document, "export_to_text"):
            return document.export_to_text()
        return str(document)


@lru_cache(maxsize=1)
def _docling_converter():
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def _read_pdf_structured(content: bytes) -> tuple[str, list[StructuredBlock]]:
    try:
        markdown = _convert_pdf_with_docling(content)
    except RuntimeError:
        raise
    except Exception:
        raw_text = read_pdf(BytesIO(content))
        markdown = preprocess_text(raw_text)

    text = preprocess_text(markdown)
    blocks = _blocks_from_markdown(text)
    return _blocks_to_markdown(blocks), blocks


def _block_metadata(block: StructuredBlock) -> dict:
    metadata = {
        "page": block.page,
        "section": block.section,
        "content_type": block.content_type,
    }
    if block.content_type == "table":
        metadata["table_id"] = block.table_id
    if block.content_type == "figure_caption":
        metadata["figure_id"] = block.table_id
    return metadata


def _prepared_block_text(block: StructuredBlock) -> str:
    prefix_parts = [f"Section: {block.section}"]
    if block.page is not None:
        prefix_parts.append(f"Page: {block.page}")
    if block.content_type == "table":
        table_label = f"Table {block.table_id}" if block.table_id else "Table"
        prefix_parts.append(f"Content type: {table_label}")
    if block.content_type == "figure_caption":
        figure_label = f"Figure {block.table_id}" if block.table_id else "Figure"
        prefix_parts.append(f"Content type: {figure_label} caption")
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
    version_hash = source_version_hash(filename, content)
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
