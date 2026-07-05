from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import chromadb

from rag_mvp.config import get_config
from rag_mvp.documents import Document, StructuredBlock, split_into_chunks
from rag_mvp.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug RAG parsing, chunking, and retrieval for one query.")
    parser.add_argument("query", help="Question or search query to diagnose.")
    parser.add_argument("--document", help="Optional filename or document_id substring to filter processed docs.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks/results to display.")
    parser.add_argument("--skip-vector", action="store_true", help="Skip vector retrieval and only inspect metadata.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory containing *.structured.md and *.blocks.json artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    processed_dir = args.processed_dir or config.processed_docs_dir
    query_features = VectorStore._query_features(args.query)

    print(f"Query: {args.query}")
    print(f"Query features: {json.dumps(query_features, ensure_ascii=False)}")
    print()

    artifacts = _find_artifacts(processed_dir, args.document)
    if not artifacts:
        print(f"No processed artifacts found in {processed_dir}. Ingest a document first.")
        return

    for blocks_path, markdown_path in artifacts:
        print(f"Processed artifact: {blocks_path.name}")
        payload = json.loads(blocks_path.read_text(encoding="utf-8"))
        blocks = [_block_from_dict(item) for item in payload.get("blocks", [])]
        _print_block_summary(blocks)
        _print_matching_blocks(blocks, query_features, args.query, args.top_k)
        _print_matching_chunks(
            payload,
            blocks,
            query_features,
            args.query,
            config.chunk_size,
            config.chunk_overlap,
            args.top_k,
        )
        if markdown_path.exists():
            _print_markdown_hits(markdown_path, args.query, args.top_k)
        print()

    _print_index_metadata_matches(config.vector_store_dir, config.collection_name, query_features, args.top_k)
    if not args.skip_vector:
        _print_vector_results(args.query, args.top_k)


def _find_artifacts(processed_dir: Path, document_filter: str | None) -> list[tuple[Path, Path]]:
    if not processed_dir.exists():
        return []

    artifacts: list[tuple[Path, Path]] = []
    for blocks_path in sorted(processed_dir.glob("*.blocks.json")):
        if document_filter and document_filter.lower() not in blocks_path.name.lower():
            continue
        markdown_path = blocks_path.with_name(blocks_path.name.replace(".blocks.json", ".structured.md"))
        artifacts.append((blocks_path, markdown_path))
    return artifacts


def _block_from_dict(item: dict) -> StructuredBlock:
    return StructuredBlock(
        text=item.get("text", ""),
        page=item.get("page"),
        section=item.get("section") or "Document",
        content_type=item.get("content_type") or "text",
        table_id=item.get("table_id") or item.get("figure_id"),
    )


def _print_block_summary(blocks: list[StructuredBlock]) -> None:
    section_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for block in blocks:
        section_counts[block.section] = section_counts.get(block.section, 0) + 1
        type_counts[block.content_type] = type_counts.get(block.content_type, 0) + 1

    print(f"Blocks: {len(blocks)}")
    print(f"Content types: {_format_counts(type_counts)}")
    print(f"Top sections: {_format_counts(section_counts, limit=8)}")


def _print_matching_blocks(
    blocks: list[StructuredBlock],
    query_features: dict,
    query: str,
    top_k: int,
) -> None:
    matches = []
    query_terms = _query_terms(query)
    for index, block in enumerate(blocks):
        score = _offline_match_score(block, query_features, query_terms)
        if score > 0:
            matches.append((score, index, block))

    print("\n[1] Parsed block matches")
    if not matches:
        print("No parsed blocks matched the query intent/keywords.")
        return
    for score, index, block in sorted(matches, key=lambda item: item[0], reverse=True)[:top_k]:
        print(_format_block(f"block={index} offline_score={score}", block))


def _print_matching_chunks(
    payload: dict,
    blocks: list[StructuredBlock],
    query_features: dict,
    query: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> None:
    document = Document(
        text="",
        metadata={
            "document_id": payload.get("document_id", "debug-doc"),
            "filename": payload.get("filename", "debug-doc"),
            "source": payload.get("filename", "debug-doc"),
            "extension": ".debug",
            "size": 0,
            "version_hash": payload.get("version_hash", "debug"),
            "structured_blocks": blocks,
        },
    )
    chunks = split_into_chunks(document, chunk_size, chunk_overlap)
    query_terms = _query_terms(query)
    matches = []
    for index, chunk in enumerate(chunks):
        pseudo_block = StructuredBlock(
            text=chunk.text,
            page=chunk.metadata.get("page"),
            section=chunk.metadata.get("section", "Document"),
            content_type=chunk.metadata.get("content_type", "text"),
            table_id=chunk.metadata.get("table_id") or chunk.metadata.get("figure_id"),
        )
        score = _offline_match_score(pseudo_block, query_features, query_terms)
        if score > 0:
            matches.append((score, index, chunk))

    print("\n[2] Reconstructed chunk matches")
    if not matches:
        print("No reconstructed chunks matched the query intent/keywords.")
        return
    for score, index, chunk in sorted(matches, key=lambda item: item[0], reverse=True)[:top_k]:
        print(_format_chunk(f"chunk={index} offline_score={score}", chunk.text, chunk.metadata))


def _print_markdown_hits(markdown_path: Path, query: str, top_k: int) -> None:
    terms = _query_terms(query)
    if not terms:
        return

    hits = []
    lines = markdown_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_no, line in enumerate(lines, start=1):
        lowered = line.lower()
        if any(term in lowered for term in terms):
            hits.append((line_no, line.strip()))

    print("\n[3] Structured Markdown keyword hits")
    if not hits:
        print("No keyword hits in structured Markdown.")
        return
    for line_no, line in hits[:top_k]:
        print(f"{markdown_path.name}:{line_no}: {line[:220]}")


def _print_index_metadata_matches(
    persist_dir: Path,
    collection_name: str,
    query_features: dict,
    top_k: int,
) -> None:
    print("\n[4] Chroma metadata matches")
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=None)
    where = _metadata_where(query_features)
    if where is None:
        print("No metadata lookup key detected for this query.")
        return

    result = collection.get(where=where, limit=top_k)
    documents = result.get("documents", [])
    if not documents:
        print(f"No indexed chunks matched metadata filter {where}. Re-ingest may be needed.")
        return

    for doc, metadata, chunk_id in zip(documents, result.get("metadatas", []), result.get("ids", [])):
        metadata = metadata or {}
        print(_format_chunk(f"id={chunk_id} where={where}", doc, metadata))


def _print_vector_results(query: str, top_k: int) -> None:
    print("\n[5] Final vector + metadata retrieval")
    pipeline_config = get_config()
    store = VectorStore(
        pipeline_config.vector_store_dir,
        pipeline_config.collection_name,
        pipeline_config.embedding_model,
    )
    rows = store.search(query, top_k)
    if not rows:
        print("No retrieval results.")
        return
    for index, row in enumerate(rows, start=1):
        header = (
            f"rank={index} score={row.get('score')} base={row.get('base_score')} "
            f"boost={row.get('metadata_boost')} match={row.get('match_type')}"
        )
        print(_format_chunk(header, row["text"], row["metadata"]))


def _metadata_where(query_features: dict) -> dict | None:
    if query_features.get("section_name"):
        return {"section": query_features["section_name"]}
    if query_features.get("table_id"):
        return {"table_id": query_features["table_id"]}
    if query_features.get("figure_id"):
        return {"figure_id": query_features["figure_id"]}
    return None


def _offline_match_score(block: StructuredBlock, query_features: dict, query_terms: list[str]) -> int:
    score = 0
    section = block.section.lower()
    content_type = block.content_type.lower()
    text = block.text.lower()
    element_id = str(block.table_id or "").lower()

    if query_features.get("section_name") and section == query_features["section_name"].lower():
        score += 100
    if query_features.get("table_id") and content_type == "table" and element_id == query_features["table_id"]:
        score += 100
    if (
        query_features.get("figure_id")
        and content_type == "figure_caption"
        and element_id == query_features["figure_id"]
    ):
        score += 100
    score += sum(1 for term in query_terms if term in text or term in section)
    return score


def _query_terms(query: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "which",
    }
    return [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2 and term not in stopwords]


def _format_counts(counts: dict[str, int], limit: int | None = None) -> str:
    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if limit is not None:
        items = items[:limit]
    return ", ".join(f"{name}={count}" for name, count in items)


def _format_block(header: str, block: StructuredBlock) -> str:
    metadata = {
        "page": block.page,
        "section": block.section,
        "content_type": block.content_type,
        "element_id": block.table_id,
    }
    return _format_chunk(header, block.text, metadata)


def _format_chunk(header: str, text: str, metadata: dict) -> str:
    clean_text = re.sub(r"\s+", " ", text).strip()
    metadata_bits = [
        f"page={metadata.get('page')}",
        f"section={metadata.get('section')}",
        f"type={metadata.get('content_type')}",
    ]
    if metadata.get("table_id"):
        metadata_bits.append(f"table={metadata.get('table_id')}")
    if metadata.get("figure_id"):
        metadata_bits.append(f"figure={metadata.get('figure_id')}")
    if metadata.get("element_id"):
        metadata_bits.append(f"element={metadata.get('element_id')}")
    return f"- {header} | {'; '.join(metadata_bits)}\n  {clean_text[:500]}"


if __name__ == "__main__":
    main()
