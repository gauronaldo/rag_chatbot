from pathlib import Path

from rag_mvp.documents import load_file, preprocess_text, source_version_hash, split_into_chunks
from rag_mvp.evaluation import citation_accuracy, false_refusal
from rag_mvp.registry import DocumentRecord, JsonDocumentRegistry
from rag_mvp.vector_store import VectorStore


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


def test_source_version_hash_tracks_file_content():
    first = source_version_hash("demo.pdf", b"same content")
    second = source_version_hash("demo.pdf", b"same content")
    changed = source_version_hash("demo.pdf", b"different content")
    assert first == second
    assert first != changed


def test_split_preserves_section_and_table_metadata():
    content = """# Results

This section summarizes the findings.

| Table 3 | Estimate |
| --- | --- |
| A | 0.42 |
"""
    doc = load_file("demo.md", content.encode("utf-8"))
    chunks = split_into_chunks(doc, chunk_size=300, chunk_overlap=20)
    table_chunks = [chunk for chunk in chunks if chunk.metadata.get("content_type") == "table"]
    assert table_chunks
    assert table_chunks[0].metadata["section"] == "Results"
    assert table_chunks[0].metadata["table_id"] == "3"


def test_split_preserves_figure_caption_metadata():
    content = """# Results

Figure 2. Employment effects over time.
"""
    doc = load_file("demo.md", content.encode("utf-8"))
    chunks = split_into_chunks(doc, chunk_size=300, chunk_overlap=20)
    figure_chunks = [chunk for chunk in chunks if chunk.metadata.get("content_type") == "figure_caption"]
    assert figure_chunks
    assert figure_chunks[0].metadata["section"] == "Results"
    assert figure_chunks[0].metadata["figure_id"] == "2"


def test_abstract_stops_before_keyword_following_intro_text():
    content = """# Abstract

The paper studies minimum wages.
Keywords Minimum wage.
JEL Codes J31
The minimum wage is a common policy in developed and developing countries.
"""
    doc = load_file("demo.md", content.encode("utf-8"))
    chunks = split_into_chunks(doc, chunk_size=300, chunk_overlap=20)
    abstract_chunks = [chunk for chunk in chunks if chunk.metadata.get("section") == "Abstract"]
    introduction_chunks = [chunk for chunk in chunks if chunk.metadata.get("section") == "Introduction"]
    assert abstract_chunks
    assert introduction_chunks
    assert "common policy" not in abstract_chunks[0].text
    assert "common policy" in introduction_chunks[0].text


def test_abstract_excludes_front_matter_and_acknowledgements():
    content = """# Abstract

The paper studies minimum wages.

1 Banco de Mexico. Email: author@example.com.

I thank several colleagues for helpful comments.

Keywords Minimum wage. JEL Codes J31

The minimum wage is a common policy in developed and developing countries.
"""
    doc = load_file("demo.md", content.encode("utf-8"))
    chunks = split_into_chunks(doc, chunk_size=300, chunk_overlap=20)
    abstract_chunks = [chunk for chunk in chunks if chunk.metadata.get("section") == "Abstract"]
    front_matter_chunks = [chunk for chunk in chunks if chunk.metadata.get("section") == "Front Matter"]
    introduction_chunks = [chunk for chunk in chunks if chunk.metadata.get("section") == "Introduction"]
    assert abstract_chunks
    assert front_matter_chunks
    assert introduction_chunks
    assert "I thank" not in abstract_chunks[0].text
    assert "Email:" not in abstract_chunks[0].text


def test_metadata_boost_prioritizes_matching_table():
    features = VectorStore._query_features("What does Table 3 report?")
    exact_table = {"content_type": "table", "table_id": "3", "section": "Results"}
    plain_text = {"content_type": "text", "section": "Results"}
    assert VectorStore._metadata_boost(features, exact_table) > VectorStore._metadata_boost(features, plain_text)


def test_metadata_boost_prioritizes_matching_figure():
    features = VectorStore._query_features("What does Figure 2 show?")
    exact_figure = {"content_type": "figure_caption", "figure_id": "2", "section": "Results"}
    plain_text = {"content_type": "text", "section": "Results"}
    assert VectorStore._metadata_boost(features, exact_figure) > VectorStore._metadata_boost(features, plain_text)


def test_metadata_boost_prioritizes_named_section():
    features = VectorStore._query_features("What is the abstract?")
    abstract_text = {"content_type": "text", "section": "Abstract"}
    introduction_text = {"content_type": "text", "section": "Introduction"}
    assert features["section_name"] == "Abstract"
    assert VectorStore._metadata_boost(features, abstract_text) > VectorStore._metadata_boost(
        features,
        introduction_text,
    )


def test_query_maps_conclusion_to_concluding_remarks():
    features = VectorStore._query_features("What is the conclusion?")
    conclusion_text = {"content_type": "text", "section": "Concluding Remarks"}
    method_text = {"content_type": "text", "section": "A Differences-in-Differences Strategy"}
    assert features["section_name"] == "Concluding Remarks"
    assert VectorStore._metadata_boost(features, conclusion_text) > VectorStore._metadata_boost(features, method_text)


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
