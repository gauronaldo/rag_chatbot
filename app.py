from __future__ import annotations

import time
from html import escape

import streamlit as st

from rag_mvp.config import get_config
from rag_mvp.pipeline import RagPipeline

ALL_DOCUMENTS = "__all_documents__"

st.set_page_config(page_title="English RAG MVP", layout="wide")

st.markdown(
    """
    <style>
      .block-container {
        max-width: 1180px;
        padding-top: 3.25rem;
        padding-bottom: 6rem;
      }
      [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.85rem;
      }
      div[data-testid="stChatMessage"] {
        padding: 0.25rem 0;
      }
      .scope-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border: 1px solid rgba(250, 250, 250, 0.18);
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.82rem;
        color: rgba(250, 250, 250, 0.82);
        background: rgba(250, 250, 250, 0.04);
      }
      .doc-row {
        border: 1px solid rgba(250, 250, 250, 0.12);
        border-radius: 0.5rem;
        padding: 0.55rem 0.65rem;
        margin-bottom: 0.45rem;
        background: rgba(250, 250, 250, 0.03);
      }
      .doc-row small {
        color: rgba(250, 250, 250, 0.58);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model and Chroma...")
def get_pipeline() -> RagPipeline:
    return RagPipeline(get_config())


def selected_document_ids(scope: str) -> list[str] | None:
    if scope == ALL_DOCUMENTS:
        return None
    return [scope]


def format_scope(scope: str, record_by_id: dict[str, object]) -> str:
    if scope == ALL_DOCUMENTS:
        return "All indexed documents"
    record = record_by_id.get(scope)
    return record.filename if record else "All indexed documents"


def safe_label(value: object) -> str:
    return escape(str(value or ""))


def render_contexts(contexts: list[dict]) -> None:
    for index, context in enumerate(contexts, start=1):
        metadata = context["metadata"]
        label_parts = [metadata.get("filename", "unknown")]
        if metadata.get("page"):
            label_parts.append(f"page {metadata['page']}")
        if metadata.get("section"):
            label_parts.append(f"section: {metadata['section']}")
        if metadata.get("content_type"):
            label_parts.append(f"type: {metadata['content_type']}")
        if metadata.get("table_id"):
            label_parts.append(f"table: {metadata['table_id']}")
        if metadata.get("figure_id"):
            label_parts.append(f"figure: {metadata['figure_id']}")
        st.markdown(f"**[S{index}] {'; '.join(label_parts)}**")
        st.caption(
            f"Score: {context['score']} "
            f"(similarity {context.get('base_score', context['score'])}, "
            f"metadata boost {context.get('metadata_boost', 0.0)}, "
            f"match {context.get('match_type', 'vector')})",
        )
        st.write(context["text"][:1200])


config = get_config()
pipeline = get_pipeline()
records = sorted(pipeline.registry.all(), key=lambda record: record.filename.lower())
record_by_id = {record.document_id: record for record in records}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_scope" not in st.session_state:
    st.session_state.document_scope = ALL_DOCUMENTS
if st.session_state.document_scope not in {ALL_DOCUMENTS, *record_by_id.keys()}:
    st.session_state.document_scope = ALL_DOCUMENTS

with st.sidebar:
    st.subheader("Local runtime")
    runtime_cols = st.columns(2)
    runtime_cols[0].metric("Top K", config.top_k)
    runtime_cols[1].metric("Docs", len(records))
    st.caption(f"LLM: `{config.ollama_model}`")
    st.caption(f"Embedding: `{config.embedding_model}`")
    if pipeline.llm.healthcheck():
        st.success("Ollama is reachable")
    else:
        st.error("Ollama is not reachable. Run `ollama serve` and pull the Qwen model.")

    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload English documents",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
    )
    ingesting = st.session_state.get("ingesting_documents", False)
    ingest_clicked = st.button(
        "Ingest documents",
        type="primary",
        disabled=not uploaded_files or ingesting,
        use_container_width=True,
    )
    if uploaded_files and ingest_clicked:
        st.session_state.ingesting_documents = True
        progress_panel = st.container(border=True)
        overall_progress = progress_panel.progress(0.0, text="Preparing ingestion...")
        status_text = progress_panel.empty()
        file_status = progress_panel.empty()
        total_files = len(uploaded_files)

        try:
            for file_index, uploaded in enumerate(uploaded_files):
                file_number = file_index + 1
                file_status.write(f"File {file_number}/{total_files}: `{uploaded.name}`")
                file_progress = progress_panel.progress(0.0, text=f"Starting {uploaded.name}")

                def update_file_progress(progress: float, message: str) -> None:
                    bounded_progress = min(max(progress, 0.0), 1.0)
                    file_progress.progress(bounded_progress, text=message)
                    completed_files = file_index / total_files
                    overall = completed_files + bounded_progress / total_files
                    overall_progress.progress(
                        min(overall, 1.0),
                        text=f"Overall ingestion: {round(overall * 100)}%",
                    )
                    status_text.info(message)

                with st.spinner(f"Ingesting {uploaded.name}..."):
                    record = pipeline.ingest_file(uploaded.name, uploaded.getvalue(), update_file_progress)
                file_progress.progress(1.0, text=f"Indexed {record.filename}")
                progress_panel.success(f"Indexed {record.filename} ({len(record.chunk_ids)} chunks)")

            overall_progress.progress(1.0, text="Ingestion completed.")
            status_text.success("Ingestion completed.")
            st.cache_resource.clear()
            st.session_state.ingesting_documents = False
            st.rerun()
        except Exception:
            st.session_state.ingesting_documents = False
            raise

    if records:
        scope_options = [ALL_DOCUMENTS, *[record.document_id for record in records]]
        selected_scope = st.radio(
            "Search scope",
            options=scope_options,
            format_func=lambda scope: format_scope(scope, record_by_id),
            key="document_scope",
            help="Limit retrieval to one document, or search across all indexed documents.",
        )
        st.caption(f"Active: {format_scope(selected_scope, record_by_id)}")

        with st.expander("Indexed documents", expanded=True):
            for record in records:
                st.markdown(
                    f"""
                    <div class="doc-row">
                      <strong>{safe_label(record.filename)}</strong><br />
                      <small>{len(record.chunk_ids)} chunks</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Delete", key=f"delete-{record.document_id}", use_container_width=True):
                    pipeline.delete_document(record.document_id)
                    if st.session_state.document_scope == record.document_id:
                        st.session_state.document_scope = ALL_DOCUMENTS
                    st.cache_resource.clear()
                    st.rerun()
    else:
        st.info("No documents indexed yet.")

    control_cols = st.columns(2)
    if control_cols[0].button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if control_cols[1].button("Reset index", use_container_width=True):
        pipeline.reset()
        st.session_state.messages = []
        st.session_state.document_scope = ALL_DOCUMENTS
        st.cache_resource.clear()
        st.rerun()

st.title("English RAG Assistant")
st.caption("Local English RAG MVP using Streamlit, Chroma, BAAI/bge-m3, and Ollama Qwen.")
active_scope = format_scope(st.session_state.document_scope, record_by_id)
st.markdown(f'<span class="scope-pill">Search scope: {safe_label(active_scope)}</span>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about the selected document scope in English...", disabled=not records)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    scoped_document_ids = selected_document_ids(st.session_state.document_scope)
    contexts, stream, started = pipeline.stream_answer(question, document_ids=scoped_document_ids)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        for token in stream:
            answer += token
            placeholder.markdown(answer)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        st.caption(f"Latency: {latency_ms} ms")
        if not contexts:
            st.warning("No contexts were retrieved for the selected document scope.")
        with st.expander("Retrieved contexts"):
            render_contexts(contexts)

    st.session_state.messages.append({"role": "assistant", "content": answer})
