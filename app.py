from pathlib import Path
import importlib
import streamlit as st

# we import logger directly (doesn't need reload)
from rag_pipeline.logger import get_logger


def build_fresh_pipeline():
    """
    Force-reload the pipeline modules so Streamlit does NOT reuse the old, cached
    Pydantic VectorStore class (the one without add_chunks).

    Steps:
    1. Reload rag_pipeline.storage.vector_store (new class with add_chunks).
    2. Reload rag_pipeline.pipeline1 so it picks up that new VectorStore.
    3. Construct a brand new RAGPipeline from that reloaded module.
    """
    import rag_pipeline.storage.vector_store as vs_mod
    importlib.reload(vs_mod)

    import rag_pipeline.pipeline1 as pipe_mod
    importlib.reload(pipe_mod)

    return pipe_mod.RAGPipeline()


def init_state():
    """
    Always create a brand new RAGPipeline() on each Streamlit rerun.

    This guarantees:
    - self.vstore is the NEW VectorStore (with add_chunks)
    - no stale Pydantic model objects hanging around
    - thread-safe SQLite connections
    """
    st.session_state.rag = build_fresh_pipeline()
    st.session_state.log = get_logger()
    if "last_build_complete" not in st.session_state:
        st.session_state.last_build_complete = False


def save_uploaded_files(uploaded_files, data_dir: Path):
    """
    Save uploaded .pdf/.docx/.txt files to data_dir.
    Return list[Path] of the saved files.
    """
    saved_paths = []
    data_dir.mkdir(parents=True, exist_ok=True)

    for up in uploaded_files:
        filename = up.name
        suffix = Path(filename).suffix.lower()
        if suffix not in [".pdf", ".docx", ".txt"]:
            # skip unsupported file types silently
            continue

        dest_path = data_dir / filename
        with open(dest_path, "wb") as f:
            f.write(up.getbuffer())

        saved_paths.append(dest_path)

    return saved_paths


def tail_file(path: Path, max_lines: int = 200) -> str:
    """
    Read last max_lines from a log file.
    """
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-max_lines:]
    return "".join(lines)


def render_chunks(hits):
    """
    Show top retrieved chunks (max 12).
    Each one:
    - score (RRF / reranker normalized score, 0..1)
    - source doc
    - char span
    - ~first 300 chars
    """
    st.subheader("Top Retrieved Chunks (debug)")
    if not hits:
        st.info("No context chunks found.")
        return

    for idx, hit in enumerate(hits[:12], start=1):
        disp_score = "-" if hit.score is None else f"{hit.score:.3f}"
        source_str = hit.chunk.doc_id
        span_str = f"{hit.chunk.start_char}-{hit.chunk.end_char}"

        preview_text = (hit.chunk.text or "")[:300].replace("\n", " ")
        preview_str = preview_text + "..."

        with st.expander(f"{idx}. {source_str} | score {disp_score} | span {span_str}"):
            st.write(preview_str)


def main():
    st.set_page_config(page_title="RAG QA Dashboard", layout="wide")

    # make sure we always have a fresh pipeline + logger
    init_state()

    rag = st.session_state.rag
    log = st.session_state.log

    # layout: left column (upload + logs), right column (ask)
    col_left, col_right = st.columns([1, 2])

    # -------------------------------------------------
    # LEFT SIDE: upload + incremental indexing + logs
    # -------------------------------------------------
    with col_left:
        st.header("Ingest & Logs")
        st.write("Upload .pdf / .docx / .txt. Only the NEW uploads will be indexed and appended.")

        uploaded_files = st.file_uploader(
            "Upload new files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            data_dir = Path(rag.settings.data_dir)
            saved_paths = save_uploaded_files(uploaded_files, data_dir)

            if saved_paths:
                log.info("Uploaded %d new files: %s", len(saved_paths), [p.name for p in saved_paths])

                # INCREMENTAL INDEX UPDATE:
                # this calls VectorStore.add_chunks() under the hood
                rag.build_index(new_paths=saved_paths)

                st.session_state.last_build_complete = True
                st.success(f"Indexed {len(saved_paths)} new file(s).")
            else:
                st.warning("No supported files to index (only .pdf, .docx, .txt).")

        if st.session_state.last_build_complete:
            st.caption("✅ Latest index update complete (incremental append).")

        # --- logs UI
        st.subheader("Logs")

        app_log_path = Path("logs/app.log")
        err_log_path = Path("logs/error.log")

        tabs = st.tabs(["app.log", "error.log"])

        with tabs[0]:
            app_tail = tail_file(app_log_path, max_lines=200)
            if app_tail.strip():
                st.text_area("app.log (last 200 lines)", app_tail, height=300)
            else:
                st.info("No app.log yet.")

        with tabs[1]:
            err_tail = tail_file(err_log_path, max_lines=200)
            if err_tail.strip():
                st.text_area("error.log (last 200 lines)", err_tail, height=300)
            else:
                st.info("No error.log yet.")

    # -------------------------------------------------
    # RIGHT SIDE: question/answer panel
    # -------------------------------------------------
    with col_right:
        st.header("Ask your data")

        question = st.text_input(
            "Your question",
            placeholder="e.g. who is geethika",
        )

        use_llm = st.checkbox(
            "Use Gemini 2.5 Flash summarizer (≤50 words, bullet points)",
            value=True,
        )

        ask_btn = st.button("Ask")

        if ask_btn:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                log.info("UI question: %r (use_llm=%s)", question, use_llm)

                # run retrieval + RRF rerank + answer
                pkg = rag.ask(question.strip(), use_llm=use_llm)

                # answer section
                st.subheader("Answer")
                st.success(pkg.answer)

                # confidence metric (0..1 from top fused hit) -> show as %
                st.metric("Confidence", f"{pkg.confidence * 100:.1f}%")

                # debug chunks (with their fused/normalized scores)
                render_chunks(pkg.context_chunks)

                log.info(
                    "UI answer len=%d chars conf=%.3f",
                    len(pkg.answer),
                    pkg.confidence,
                )


if __name__ == "__main__":
    main()
