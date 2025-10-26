from pathlib import Path
from typing import List, Tuple, Set

from rag_pipeline.config import get_settings, Settings
from rag_pipeline.ingestion.pdf_loader import PDFIngestor
from rag_pipeline.ingestion.chunker import chunk_corpus
from rag_pipeline.embeddings.ebedding import Embedder
from rag_pipeline.storage.vector_store import VectorStore
from rag_pipeline.models import QueryRequest, AnswerPackage, RetrievalHit
from rag_pipeline.pipeline.generator import generate_answer


def _detect_targets(question: str) -> Tuple[bool, bool]:
    """
    Decide if the user explicitly asked about RLEG and/or DINOv3.
    IMPORTANT CHANGE:
    - If the question does NOT mention either, we DO NOT force them.
      (So 'what is gamescon' will not force RLEG/DINOv3.)
    """
    q = question.lower()
    want_rleg = "rleg" in q
    want_dino = "dinov3" in q or "dino v3" in q or "dino v 3" in q
    return want_rleg, want_dino


def _dedup_hits(hits: List[RetrievalHit]) -> List[RetrievalHit]:
    """
    Deduplicate chunks by their sqlite row id (chunk_id).
    """
    seen: Set[int] = set()
    out: List[RetrievalHit] = []
    for h in hits:
        cid = h.chunk.chunk_id
        if cid not in seen:
            out.append(h)
            seen.add(cid)
    return out


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        self.embedder = Embedder(model_name=self.settings.embedding_model_name)

        self.vstore = VectorStore(
            sqlite_path=Path(self.settings.sqlite_path),
            faiss_index_path=Path(self.settings.faiss_index_path),
            id_map_path=Path(self.settings.id_map_path),
        )

    def build_index(self, rebuild: bool = False):
        """
        If rebuild=True:
        - Ingest all PDFs in data/
        - Chunk them
        - Embed them
        - Persist FAISS index + SQLite metadata
        """
        data_dir = Path(self.settings.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if rebuild:
            ingestor = PDFIngestor()
            docs = ingestor.load_all_pdfs(data_dir)

            chunks = chunk_corpus(
                docs,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )

            self.vstore.build_from_chunks(chunks, self.embedder)

    def ask(self, question: str, top_k: int | None = None, use_llm: bool = False) -> AnswerPackage:
        """
        NEW LOGIC:
        1. Always retrieve for the ACTUAL user question (this covers new PDFs like 'gamescon').
        2. If the question specifically mentions RLEG and/or DINOv3, ALSO retrieve those
           individually with focused queries ('what is RLEG', 'what is DINOv3').
        3. Merge + dedupe.
        4. Hand all hits to the generator, which will decide how to summarize.
        """

        want_rleg, want_dino = _detect_targets(question)
        k = top_k or self.settings.default_top_k

        all_hits: List[RetrievalHit] = []

        # 1. retrieve using the exact user question (gamescon, anomaly transformer, etc.)
        generic_req = QueryRequest(
            question=question,
            top_k=k,
        )
        generic_hits = self.vstore.search(generic_req, self.embedder)
        all_hits.extend(generic_hits)

        # 2. targeted RLEG retrieval only if explicitly asked
        if want_rleg:
            req_rleg = QueryRequest(
                question="what is RLEG",
                top_k=k,
            )
            rleg_hits = self.vstore.search(req_rleg, self.embedder)
            all_hits.extend(rleg_hits)

        # 3. targeted DINOv3 retrieval only if explicitly asked
        if want_dino:
            req_dino = QueryRequest(
                question="what is DINOv3",
                top_k=k,
            )
            dino_hits = self.vstore.search(req_dino, self.embedder)
            all_hits.extend(dino_hits)

        # 4. dedupe chunks
        all_hits = _dedup_hits(all_hits)

        # 5. let the generator build the final concise answer
        ans_text = generate_answer(
            question=question,
            hits=all_hits,
            use_llm=use_llm,
        )

        return AnswerPackage(
            question=question,
            answer=ans_text,
            context_chunks=all_hits,
        )

    def close(self):
        self.vstore.close()
