from pathlib import Path

from rag_mvp.documents import load_file, preprocess_text, split_into_chunks
from rag_mvp.evaluation import citation_accuracy, false_refusal
from rag_mvp.registry import DocumentRecord, JsonDocumentRegistry


def test_preprocess_normalizes_line_endings():
    text = preprocess_text("English text\r\nwith line endings")
    assert "English text" in text
    assert "\r" not in text


def test_load_and_split_english_markdown():
    content = "# Hello\n\nThis is an English document."
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
    refusal = "Not enough information in the provided context."
    ground_truth = "There is an answer"
    assert false_refusal(refusal, ground_truth) == 1
    assert citation_accuracy("The answer is here [S1] [S3]", ["ctx1", "ctx2"]) == 0.5
