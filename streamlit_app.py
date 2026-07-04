from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_mvp.config import get_config
from rag_mvp.evaluation import ragas_metrics_available, run_local_evaluation, run_ragas_core_evaluation
from rag_mvp.pipeline import RagPipeline


st.set_page_config(page_title="Multilingual RAG MVP", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and Chroma...")
def get_pipeline() -> RagPipeline:
    return RagPipeline(get_config())


config = get_config()
pipeline = get_pipeline()

st.title("Multilingual RAG Assistant")
st.caption("Local multilingual RAG MVP using Streamlit, Chroma, BAAI/bge-m3, and Ollama Qwen.")

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
        "Upload Vietnamese or English documents",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Ingest documents", type="primary"):
        overall_progress = st.progress(0.0)
        status_text = st.empty()
        total_files = len(uploaded_files)

        for file_index, uploaded in enumerate(uploaded_files):
            file_progress = st.progress(0.0)

            def update_file_progress(progress: float, message: str) -> None:
                file_progress.progress(progress)
                completed_files = file_index / total_files
                overall_progress.progress(completed_files + progress / total_files)
                status_text.write(message)

            record = pipeline.ingest_file(uploaded.name, uploaded.getvalue(), update_file_progress)
            st.success(f"Indexed {record.filename} ({len(record.chunk_ids)} chunks)")

        overall_progress.progress(1.0)
        status_text.write("Ingestion completed.")
        st.cache_resource.clear()
        st.rerun()

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

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about your documents...")
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
                    st.markdown(f"**[S{index}] {context['metadata'].get('filename', 'unknown')}**")
                    st.caption(f"Similarity: {context['score']}")
                    st.write(context["text"][:1200])

        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab_eval:
    st.subheader("RAG evaluation")
    st.write(
        "Custom MVP metrics run locally: false refusal rate, citation accuracy, strict citation accuracy, "
        "unsupported claim accuracy, and latency. Optional RAGAS metrics target faithfulness, answer relevancy, "
        "context precision, and context recall."
    )
    st.write(f"RAGAS installed: `{ragas_metrics_available()}`")
    dataset_path = st.text_input("Evaluation CSV", value=str(config.eval_dataset_path))
    output_path = config.reports_dir / "rag_mvp_eval_results.csv"
    ragas_output_path = config.reports_dir / "ragas_core_results.csv"
    if st.button("Run custom evaluation"):
        frame = run_local_evaluation(pipeline, Path(dataset_path), output_path)
        st.success(f"Saved results to {output_path}")
        st.dataframe(frame)
    elif output_path.exists():
        st.dataframe(pd.read_csv(output_path))

    if st.button("Run RAGAS core metrics"):
        if not output_path.exists():
            st.warning("Run custom evaluation first to generate answers and contexts.")
        else:
            try:
                frame = run_ragas_core_evaluation(output_path, ragas_output_path)
                st.success(f"Saved RAGAS results to {ragas_output_path}")
                st.dataframe(frame)
            except Exception as exc:
                st.error(f"RAGAS evaluation could not run in this environment: {exc}")
