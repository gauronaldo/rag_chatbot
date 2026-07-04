from pathlib import Path

from rag_mvp.documents import load_file, preprocess_text, split_into_chunks
from rag_mvp.evaluation import citation_accuracy, false_refusal
from rag_mvp.registry import DocumentRecord, JsonDocumentRegistry


def test_preprocess_preserves_vietnamese_accents():
    text = preprocess_text("Ti\u1ebfng Vi\u1ec7t\r\nc\u00f3 d\u1ea5u")
    assert "Ti\u1ebfng Vi\u1ec7t" in text
    assert "\r" not in text


def test_load_and_split_vietnamese_markdown():
    content = "# Xin ch\u00e0o\n\n\u0110\u00e2y l\u00e0 t\u00e0i li\u1ec7u ti\u1ebfng Vi\u1ec7t."
    doc = load_file("demo.md", content.encode("utf-8"))
    chunks = split_into_chunks(doc, chunk_size=40, chunk_overlap=5)
    assert doc.metadata["filename"] == "demo.md"
    assert chunks
    assert chunks[0].metadata["document_id"] == doc.metadata["document_id"]


def test_json_registry_round_trip(tmp_path: Path):
    registry = JsonDocumentRegistry(tmp_path / "registry.json")
    record = DocumentRecord("doc-1", "demo.md", "demo.md", 10, "hash", ["c1", "c2"])
    registry.upsert(record)
    assert registry.get("doc-1") == record
    registry.remove("doc-1")
    assert registry.get("doc-1") is None


def test_custom_evaluation_metrics():
    refusal = "Kh\u00f4ng \u0111\u1ee7 th\u00f4ng tin trong ng\u1eef c\u1ea3nh."
    ground_truth = "C\u00f3 c\u00e2u tr\u1ea3 l\u1eddi"
    assert false_refusal(refusal, ground_truth) == 1
    assert citation_accuracy("C\u00e2u tr\u1ea3 l\u1eddi [S1] [S3]", ["ctx1", "ctx2"]) == 0.5
