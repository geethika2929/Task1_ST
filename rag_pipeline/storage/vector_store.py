from __future__ import annotations

import os
import sqlite3
import numpy as np
import faiss
from typing import List
from pathlib import Path
from pydantic import BaseModel

from rag_pipeline.models import Chunk, RetrievalHit, QueryRequest
from rag_pipeline.embeddings.ebedding import Embedder

class VectorStore(BaseModel):
    sqlite_path: Path
    faiss_index_path: Path
    id_map_path: Path

    _index: faiss.IndexFlatIP | None = None
    _id_map: np.ndarray | None = None
    _conn: sqlite3.Connection | None = None

    class Config:
        arbitrary_types_allowed = True

    # --- internal helpers ---

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(self.sqlite_path.parent, exist_ok=True)
            self._conn = sqlite3.connect(self.sqlite_path)
        return self._conn

    def _ensure_index_loaded(self):
        if self._index is None:
            if not self.faiss_index_path.exists():
                raise RuntimeError("FAISS index file not found, did you build the store?")
            self._index = faiss.read_index(str(self.faiss_index_path))

        if self._id_map is None:
            if not self.id_map_path.exists():
                raise RuntimeError("ID map not found, did you build the store?")
            self._id_map = np.load(self.id_map_path)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- build / persist ---

    def build_from_chunks(
        self,
        chunks: List[Chunk],
        embedder: Embedder,
    ):
        """
        Full rebuild:
        - recreate SQLite table
        - insert chunks
        - embed chunks
        - build FAISS index
        """
        conn = self._ensure_conn()
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS chunks;")
        cur.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                source TEXT,
                start_char INTEGER,
                end_char INTEGER,
                text TEXT
            );
            """
        )
        conn.commit()

        texts_to_embed: List[str] = []
        row_ids: List[int] = []

        for ch in chunks:
            cur.execute(
                """
                INSERT INTO chunks (doc_id, source, start_char, end_char, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ch.doc_id, ch.source, ch.start_char, ch.end_char, ch.text),
            )
            row_id = cur.lastrowid
            row_ids.append(row_id)
            texts_to_embed.append(ch.text)

        conn.commit()

        # embed all chunks
        embeddings = embedder.embed_texts(texts_to_embed)  # (N, dim)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        os.makedirs(self.faiss_index_path.parent, exist_ok=True)
        faiss.write_index(index, str(self.faiss_index_path))

        id_map = np.array(row_ids, dtype=np.int64)
        np.save(self.id_map_path, id_map)

        self._index = index
        self._id_map = id_map

    # --- retrieval helpers ---

    def _fetch_chunks_by_row_ids(self, row_ids: List[int]) -> List[Chunk]:
        conn = self._ensure_conn()
        cur = conn.cursor()

        placeholders = ",".join(["?"] * len(row_ids))
        cur.execute(
            f"""
            SELECT id, doc_id, source, start_char, end_char, text
            FROM chunks
            WHERE id IN ({placeholders})
            """,
            row_ids,
        )
        rows = cur.fetchall()

        by_id = {}
        for row in rows:
            cid, doc_id, source, start_char, end_char, text = row
            by_id[cid] = Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                source=source,
                start_char=start_char,
                end_char=end_char,
                text=text,
            )

        ordered = [by_id[rid] for rid in row_ids if rid in by_id]
        return ordered

    def search(
        self,
        query: QueryRequest,
        embedder: Embedder
    ) -> List[RetrievalHit]:
        """
        Embed question, run FAISS top_k, map back to chunks.
        """
        self._ensure_index_loaded()
        assert self._index is not None
        assert self._id_map is not None

        q_vec = embedder.embed_text(query.question).reshape(1, -1)

        D, I = self._index.search(q_vec.astype("float32"), query.top_k)

        hits: List[RetrievalHit] = []
        for score, faiss_row in zip(D[0], I[0]):
            if faiss_row == -1:
                continue
            sqlite_row_id = int(self._id_map[faiss_row])

            chunk_list = self._fetch_chunks_by_row_ids([sqlite_row_id])
            if not chunk_list:
                continue

            hits.append(
                RetrievalHit(
                    chunk=chunk_list[0],
                    score=float(score),
                )
            )
        return hits
