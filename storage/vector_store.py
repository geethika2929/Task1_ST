from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import sqlite3
import numpy as np
import faiss

from rag_pipeline.models import Chunk, RetrievalHit, QueryRequest
from rag_pipeline.embeddings.ebedding import Embedder
from rag_pipeline.logger import get_logger

log = get_logger()


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    Row-wise L2 normalize for cosine similarity via inner product.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / norms


class VectorStore:
    """
    Persistent store for RAG:
    - SQLite: chunks + metadata
    - FAISS:  dense embeddings
    - id_map.npy: FAISS row index -> SQLite row id

    Thread-safe for Streamlit:
    - no long-lived sqlite conn
    - reloadable FAISS/index
    - supports incremental add if add_chunks() is present
    """

    def __init__(
        self,
        sqlite_path: Path,
        faiss_index_path: Path,
        id_map_path: Path,
    ):
        self.sqlite_path = Path(sqlite_path)
        self.faiss_index_path = Path(faiss_index_path)
        self.id_map_path = Path(id_map_path)

        self.index: faiss.Index | None = None
        self.id_map: np.ndarray | None = None

    # -----------------------------------------
    # internals
    # -----------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            str(self.sqlite_path),
            check_same_thread=False,
        )

    def _ensure_sqlite_table(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                source TEXT,
                start_char INTEGER,
                end_char INTEGER,
                text TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _ensure_index_loaded(self):
        if self.index is not None and self.id_map is not None:
            return
        if not self.faiss_index_path.exists() or not self.id_map_path.exists():
            self.index = None
            self.id_map = None
            return

        log.debug("Loading FAISS index from %s", self.faiss_index_path)
        self.index = faiss.read_index(str(self.faiss_index_path))

        log.debug("Loading id_map from %s", self.id_map_path)
        self.id_map = np.load(self.id_map_path).astype("int64")

    # -----------------------------------------
    # FULL REBUILD
    # -----------------------------------------

    def build_from_chunks(self, chunks: List[Chunk], embedder: Embedder):
        log.info("VectorStore.build_from_chunks: full rebuild with %d chunks", len(chunks))

        conn = self._connect()
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS chunks")
        cur.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                source TEXT,
                start_char INTEGER,
                end_char INTEGER,
                text TEXT
            )
            """
        )

        row_ids: List[int] = []
        texts: List[str] = []

        for ch in chunks:
            cur.execute(
                """
                INSERT INTO chunks (doc_id, source, start_char, end_char, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ch.doc_id,
                    ch.source,
                    ch.start_char,
                    ch.end_char,
                    ch.text,
                ),
            )
            rid = cur.lastrowid
            row_ids.append(int(rid))
            texts.append(ch.text)

        conn.commit()
        conn.close()

        # embed everything -> normalize -> build FAISS
        embeds = embedder.embed_texts(texts)
        embeds = np.asarray(embeds, dtype="float32")
        embeds = _l2_normalize(embeds)

        dim = embeds.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeds)

        faiss.write_index(index, str(self.faiss_index_path))
        np.save(self.id_map_path, np.array(row_ids, dtype="int64"))

        self.index = index
        self.id_map = np.array(row_ids, dtype="int64")

        log.info("Full rebuild complete. Total vectors: %d", len(row_ids))

    # -----------------------------------------
    # INCREMENTAL APPEND
    # -----------------------------------------

    def add_chunks(self, chunks: List[Chunk], embedder: Embedder):
        if not chunks:
            log.info("add_chunks called with 0 chunks, skipping.")
            return

        log.info("VectorStore.add_chunks: appending %d chunks", len(chunks))

        self._ensure_sqlite_table()

        conn = self._connect()
        cur = conn.cursor()

        new_row_ids: List[int] = []
        new_texts: List[str] = []

        for ch in chunks:
            cur.execute(
                """
                INSERT INTO chunks (doc_id, source, start_char, end_char, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ch.doc_id,
                    ch.source,
                    ch.start_char,
                    ch.end_char,
                    ch.text,
                ),
            )
            rid = cur.lastrowid
            new_row_ids.append(int(rid))
            new_texts.append(ch.text)

        conn.commit()
        conn.close()

        log.debug("Inserted %d new chunks into SQLite", len(new_row_ids))

        # embed and normalize new chunks
        new_embeds = embedder.embed_texts(new_texts)
        new_embeds = np.asarray(new_embeds, dtype="float32")
        new_embeds = _l2_normalize(new_embeds)

        self._ensure_index_loaded()

        if self.index is None or self.id_map is None:
            # first time
            dim = new_embeds.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(new_embeds)
            self.id_map = np.array(new_row_ids, dtype="int64")
        else:
            self.index.add(new_embeds)
            self.id_map = np.concatenate(
                [self.id_map, np.array(new_row_ids, dtype="int64")],
                axis=0,
            )

        faiss.write_index(self.index, str(self.faiss_index_path))
        np.save(self.id_map_path, self.id_map.astype("int64"))

        log.info(
            "Appended %d new vectors. Total vectors now: %d",
            len(new_row_ids),
            self.id_map.shape[0],
        )

    # -----------------------------------------
    # SEARCH
    # -----------------------------------------

    def search(
        self,
        req: QueryRequest,
        embedder: Embedder,
    ) -> List[RetrievalHit]:
        self._ensure_index_loaded()
        self._ensure_sqlite_table()

        if self.index is None or self.id_map is None or self.index.ntotal == 0:
            log.debug("search(): index empty, returning [].")
            return []

        # embed + normalize query
        q_vec = embedder.embed_text(req.question)
        q_vec = np.asarray(q_vec, dtype="float32").reshape(1, -1)
        q_vec = _l2_normalize(q_vec)

        D, I = self.index.search(q_vec, req.top_k)  # cosine sims because of normalization
        faiss_rows = I[0].tolist()
        scores = D[0].tolist()

        sqlite_row_ids: List[int] = []
        for row_idx in faiss_rows:
            if row_idx == -1:
                continue
            sqlite_row_ids.append(int(self.id_map[row_idx]))

        rows_by_id: Dict[int, Tuple] = {}
        if sqlite_row_ids:
            conn = self._connect()
            cur = conn.cursor()

            placeholders = ",".join("?" * len(sqlite_row_ids))
            cur.execute(
                f"""
                SELECT id, doc_id, source, start_char, end_char, text
                FROM chunks
                WHERE id IN ({placeholders})
                """,
                sqlite_row_ids,
            )
            db_rows = cur.fetchall()
            conn.close()

            for row in db_rows:
                rid, doc_id, source, start_c, end_c, text = row
                rows_by_id[int(rid)] = (
                    int(rid),
                    doc_id,
                    source,
                    int(start_c),
                    int(end_c),
                    text,
                )

        hits: List[RetrievalHit] = []
        for faiss_row, sim in zip(faiss_rows, scores):
            if faiss_row == -1:
                continue
            sqlite_id = int(self.id_map[faiss_row])
            if sqlite_id not in rows_by_id:
                continue

            rid, doc_id, source, start_c, end_c, text = rows_by_id[sqlite_id]

            ch = Chunk(
                chunk_id=rid,
                doc_id=doc_id,
                source=source,
                start_char=start_c,
                end_char=end_c,
                text=text,
            )

            hit = RetrievalHit(
                score=float(sim),  # raw cosine from FAISS
                chunk=ch,
            )
            hits.append(hit)

        return hits

    # -----------------------------------------
    # CLOSE
    # -----------------------------------------

    def close(self):
        log.debug("VectorStore.close() no-op (no persistent connection).")
