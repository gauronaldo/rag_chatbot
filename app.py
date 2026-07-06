from __future__ import annotations

import time

import streamlit as st

from rag_mvp.config import get_config
from rag_mvp.pipeline import RagPipeline

st.set_page_config(page_title="English RAG MVP", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and Chroma...")
def get_pipeline() -> RagPipeline:
    return RagPipeline(get_config())


config = get_config()
pipeline = get_pipeline()

st.title("English RAG Assistant")
st.caption("Local English RAG MVP using Streamlit, Chroma, BAAI/bge-m3, and Ollama Qwen.")

with st.sidebar:
    st.subheader("Local runtime")
    st.write(f"LLM: `{config.ollama_model}`")
    st.write(f"Embedding: `{config.embedding_model}`")
    st.write(f"Top K: `{config.top_k}`")
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

    records = pipeline.registry.all()
    if records:
        for record in records:
            cols = st.columns([4, 1])
            cols[0].write(f"{record.filename} - {len(record.chunk_ids)} chunks")
            if cols[1].button("Delete", key=f"delete-{record.document_id}"):
                pipeline.delete_document(record.document_id)
                st.cache_resource.clear()
                st.rerun()
    else:
        st.info("No documents indexed yet.")

    if st.button("Reset index"):
        pipeline.reset()
        st.cache_resource.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about your documents in English...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    contexts, stream, started = pipeline.stream_answer(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        for token in stream:
            answer += token
            placeholder.markdown(answer)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        st.caption(f"Latency: {latency_ms} ms")
        with st.expander("Retrieved contexts"):
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
                    f"match {context.get('match_type', 'vector')})"
                )
                st.write(context["text"][:1200])

    st.session_state.messages.append({"role": "assistant", "content": answer})
