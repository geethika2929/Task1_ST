from pathlib import Path
from typing import List, Tuple, Iterable

from rag_pipeline.config import get_settings, Settings
from rag_pipeline.ingestion.pdf_loader import DocumentIngestor
from rag_pipeline.ingestion.chunker import chunk_corpus
from rag_pipeline.embeddings.ebedding import Embedder
from rag_pipeline.storage.vector_store import VectorStore
from rag_pipeline.models import QueryRequest, AnswerPackage, RetrievalHit
from rag_pipeline.pipeline.generator import generate_answer
from rag_pipeline.re_rankers.re_rankers import build_reranker
from rag_pipeline.logger import get_logger

log = get_logger()


def _detect_targets(question: str) -> Tuple[bool, bool]:
    """
    Look for hints that the user is asking specifically about RLEG / DINOv3.
    We use that to build boosted queries in retrieval.
    """
    q = question.lower()
    want_rleg = "rleg" in q
    want_dino = "dinov3" in q or "dino v3" in q or "dino v 3" in q
    return want_rleg, want_dino


class RAGPipeline:
    """
    High-level orchestrator:
    - builds / updates the vector DB
    - retrieves chunks
    - reranks/fuses them (RRF/MMR/etc.)
    - generates final answer text
    - returns confidence score
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        log.debug(
            "Initializing RAGPipeline with settings: %s",
            self.settings.model_dump(mode="python"),
        )

        # embedder for both indexing + retrieval
        self.embedder = Embedder(model_name=self.settings.embedding_model_name)
        log.debug(
            "Embedder initialized with model: %s",
            self.settings.embedding_model_name,
        )

        # vector store (FAISS + sqlite)
        self.vstore = VectorStore(
            sqlite_path=Path(self.settings.sqlite_path),
            faiss_index_path=Path(self.settings.faiss_index_path),
            id_map_path=Path(self.settings.id_map_path),
        )
        log.debug(
            "VectorStore initialized. sqlite=%s faiss=%s idmap=%s",
            self.settings.sqlite_path,
            self.settings.faiss_index_path,
            self.settings.id_map_path,
        )

        # reranker / rank fusion (RRF, MMR, cross-encoder, gemini...)
        self.reranker = build_reranker(self.settings, self.embedder)

    # ---------------------------------------------------------------------------------
    # INDEXING
    # ---------------------------------------------------------------------------------

    def _full_rebuild_from_dir(self, data_dir: Path):
        """
        Helper: scan ENTIRE data_dir, chunk everything, wipe+rebuild FAISS+sqlite.

        This is our fallback path if VectorStore doesn't have incremental add_chunks().
        """
        ingestor = DocumentIngestor()

        log.info("Full rebuild fallback: scanning %s", data_dir)
        docs = ingestor.load_all_documents(data_dir)
        log.info("Loaded %d docs for full rebuild fallback", len(docs))

        chunks = chunk_corpus(
            docs,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        log.info(
            "Full rebuild fallback produced %d chunks (size=%d overlap=%d)",
            len(chunks),
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

        # blow away and recreate index
        self.vstore.build_from_chunks(chunks, self.embedder)
        log.info("Full rebuild fallback complete.")

    def build_index(
        self,
        rebuild: bool = False,
        new_paths: Iterable[Path] | None = None,
    ):
        """
        Two modes:

        1. rebuild=True:
           - ignore new_paths
           - wipe and rebuild the entire FAISS/sqlite index from ALL docs in data_dir

        2. new_paths != None:
           - ingest / chunk ONLY those new files
           - if VectorStore has .add_chunks(): append incrementally
           - else: fall back to full rebuild of ENTIRE data_dir
             (slower, but guaranteed to work even with the old Pydantic VectorStore)

        If neither rebuild nor new_paths is given, we do nothing.
        """
        data_dir = Path(self.settings.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # mode 1: force full rebuild
        if rebuild:
            log.info("Full rebuild requested. Re-indexing everything in %s", data_dir)
            self._full_rebuild_from_dir(data_dir)
            return

        # mode 2: incremental new uploads
        if new_paths:
            new_paths = [Path(p) for p in new_paths]
            log.info(
                "build_index() incremental mode with %d new file(s): %s",
                len(new_paths),
                [p.name for p in new_paths],
            )

            ingestor = DocumentIngestor()
            docs = ingestor.load_documents_from_paths(new_paths)
            log.info("Loaded %d docs from new_paths for incremental add", len(docs))

            if not docs:
                log.warning("No supported docs found in upload. Skipping index update.")
                return

            chunks = chunk_corpus(
                docs,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            log.info(
                "New docs chunked into %d chunks (size=%d overlap=%d)",
                len(chunks),
                self.settings.chunk_size,
                self.settings.chunk_overlap,
            )

            # here's the important part:
            # if the VectorStore has the nice incremental API (add_chunks), use it.
            # if not, fallback to a full rebuild of EVERYTHING so we don't explode with AttributeError.
            if hasattr(self.vstore, "add_chunks"):
                log.info("VectorStore supports add_chunks(); performing incremental append.")
                self.vstore.add_chunks(chunks, self.embedder)
                log.info("Incremental append complete.")
            else:
                log.warning(
                    "VectorStore has NO add_chunks() (likely old Pydantic version). "
                    "Falling back to full rebuild of entire corpus."
                )
                self._full_rebuild_from_dir(data_dir)

            return

        # nothing asked
        log.info("build_index called with no rebuild and no new_paths; skipping.")

    # ---------------------------------------------------------------------------------
    # QA FLOW
    # ---------------------------------------------------------------------------------

    def _compute_confidence(self, ranked: List[RetrievalHit]) -> float:
        """
        Confidence is just the normalized score of the top-ranked hit.
        Our rerankers (RRF/MMR/etc.) set hit.score to something in [0,1].
        If nothing or missing: 0.0
        """
        if not ranked:
            return 0.0
        top = ranked[0]
        s = top.score if top.score is not None else 0.0
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        return float(s)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        use_llm: bool = False,
    ) -> AnswerPackage:
        """
        1. Retrieve multiple candidate lists (generic, boosted queries)
        2. Fuse / rerank them (RRF, etc.)
        3. Generate final short answer
        4. Return chunks + confidence for UI
        """
        want_rleg, want_dino = _detect_targets(question)

        k_final = top_k or self.settings.default_top_k          # how many final chunks we keep
        k_init = max(k_final, self.settings.reranker_candidates)  # how many to recall from FAISS per query

        log.info(
            "ask() question=%r k_init=%d k_final=%d use_llm=%s",
            question,
            k_init,
            k_final,
            use_llm,
        )
        log.debug("topic flags: want_rleg=%s want_dino=%s", want_rleg, want_dino)

        # 1. multi-query retrieval
        lists_for_rerank: List[List[RetrievalHit]] = []

        generic_req = QueryRequest(question=question, top_k=k_init)
        generic_hits = self.vstore.search(generic_req, self.embedder)
        lists_for_rerank.append(generic_hits)
        log.debug("generic retrieval -> %d hits", len(generic_hits))

        if want_rleg:
            req_rleg = QueryRequest(question="what is RLEG", top_k=k_init)
            rleg_hits = self.vstore.search(req_rleg, self.embedder)
            lists_for_rerank.append(rleg_hits)
            log.debug("RLEG retrieval -> %d hits", len(rleg_hits))

        if want_dino:
            req_dino = QueryRequest(question="what is DINOv3", top_k=k_init)
            dino_hits = self.vstore.search(req_dino, self.embedder)
            lists_for_rerank.append(dino_hits)
            log.debug("DINOv3 retrieval -> %d hits", len(dino_hits))

        # 2. rerank / fuse (RRF by default per settings.reranker_type="rrf")
        if self.reranker is not None:
            ranked = self.reranker.rerank_multi(
                question=question,
                list_of_lists=lists_for_rerank,
                top_k=k_final,
            )
        else:
            # fallback: just concat first list(s), dedup in order
            seen_ids = set()
            ranked_temp: List[RetrievalHit] = []
            for hit_list in lists_for_rerank:
                for h in hit_list:
                    cid = h.chunk.chunk_id
                    if cid not in seen_ids:
                        ranked_temp.append(h)
                        seen_ids.add(cid)
            ranked = ranked_temp[:k_final]

        # 3. generate final natural-language answer
        ans_text = generate_answer(
            question=question,
            hits=ranked,
            use_llm=use_llm,
        )
        log.info("answer generated, %d chars", len(ans_text))

        # 4. compute confidence (0..1) from top-ranked fused hit
        conf = self._compute_confidence(ranked)
        log.info("answer confidence=%.3f", conf)

        return AnswerPackage(
            question=question,
            answer=ans_text,
            context_chunks=ranked,
            confidence=conf,
        )

    def close(self):
        log.debug("RAGPipeline.close() called.")
        self.vstore.close()
