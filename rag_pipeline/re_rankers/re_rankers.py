from __future__ import annotations
import os
import re
from typing import List, Optional, Dict, Iterable
import numpy as np

from rag_pipeline.models import RetrievalHit
from rag_pipeline.logger import get_logger

log = get_logger()


# ------------------------- utils -------------------------

def _dedup_hits_preserve_first(hits: Iterable[RetrievalHit]) -> List[RetrievalHit]:
    """Remove duplicate chunks by chunk_id, keep first occurrence."""
    seen = set()
    out: List[RetrievalHit] = []
    for h in hits:
        cid = h.chunk.chunk_id
        if cid not in seen:
            out.append(h)
            seen.add(cid)
    return out


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    Row-wise L2 normalize.
    If mat is (n, d), each row becomes length 1.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / norms


# ------------------------- base class -------------------------

class BaseReranker:
    """
    All rerankers must implement at least rerank().
    We provide a default rerank_multi() that:
    - Concats all lists
    - Dedups
    - Calls rerank()
    """

    def rerank(self, question: str, hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        raise NotImplementedError

    def rerank_multi(
        self,
        question: str,
        list_of_lists: List[List[RetrievalHit]],
        top_k: int,
    ) -> List[RetrievalHit]:
        concat = []
        for L in list_of_lists:
            concat.extend(L)
        concat = _dedup_hits_preserve_first(concat)
        return self.rerank(question, concat, top_k)


# ------------------------- MMR -------------------------

class MMRReranker(BaseReranker):
    """
    Maximal Marginal Relevance:
    - encourages both high similarity to the query
      and diversity among selected chunks
    """

    def __init__(self, embedder, lambda_mult: float = 0.6):
        self.embedder = embedder
        self.lambda_mult = float(lambda_mult)

    def rerank(self, question: str, hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        if not hits:
            return []

        texts = [h.chunk.text for h in hits]

        # embed docs (no normalize arg; we normalize manually)
        doc_embs = self.embedder.embed_texts(texts)
        doc_embs = np.asarray(doc_embs, dtype="float32")
        doc_embs = _l2_normalize(doc_embs)  # (n,d)

        # embed query
        q_vec = self.embedder.embed_text(question)
        q_vec = np.asarray(q_vec, dtype="float32").reshape(1, -1)
        q_vec = _l2_normalize(q_vec)[0]  # (d,)

        # cosine relevance (dot because we normalized)
        rel = np.dot(doc_embs, q_vec)  # shape (n,)

        selected: List[int] = []
        pool = list(range(len(hits)))
        k = min(top_k, len(hits))

        while len(selected) < k and pool:
            best_i = None
            best_score = -1e9
            for i in pool:
                if not selected:
                    div_penalty = 0.0
                else:
                    sel_vecs = doc_embs[selected]  # (m,d)
                    # max sim to already selected (to penalize redundancy)
                    div_penalty = float(np.max(np.dot(doc_embs[i], sel_vecs.T)))

                mmr_score = self.lambda_mult * float(rel[i]) - (1.0 - self.lambda_mult) * div_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_i = i

            selected.append(best_i)
            pool.remove(best_i)

        # assign normalized relevance (0..1-ish) to .score for UI/confidence
        # rel[i] is cosine sim in [-1,1]; map to [0,1]
        final_hits: List[RetrievalHit] = []
        for i in selected:
            score_cos = float(rel[i])
            norm_score = max(0.0, min(1.0, (score_cos + 1.0) / 2.0))
            hits[i].score = norm_score
            final_hits.append(hits[i])

        return final_hits


# ------------------------- Cross-Encoder -------------------------

class CrossEncoderReranker(BaseReranker):
    """
    Uses a cross-encoder model that directly scores (question, passage).
    Falls back to MMR if that dependency/model isn't available.
    """

    def __init__(self, model_name: str, mmr_fallback: MMRReranker):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            log.warning("Cross-encoder unavailable (%s). Falling back to MMR.", e)
            self.model = None
            self.fallback = mmr_fallback
            return

        self.model = CrossEncoder(model_name)
        self.fallback = None

    def rerank(self, question: str, hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        if not hits:
            return []
        if self.model is None:
            return self.fallback.rerank(question, hits, top_k)

        pairs = [(question, h.chunk.text) for h in hits]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # scores are arbitrary scale but higher is better.
        # We'll min-max normalize to [0,1] for UI.
        arr = np.asarray(scores, dtype="float32")
        if arr.size > 0:
            lo = float(arr.min())
            hi = float(arr.max())
            rng = max(hi - lo, 1e-9)
            norm = (arr - lo) / rng
        else:
            norm = arr

        order = np.argsort(-arr).tolist()
        out: List[RetrievalHit] = []
        for rank_idx, hit_i in enumerate(order[:top_k]):
            hits[hit_i].score = float(norm[hit_i])  # [0,1]
            out.append(hits[hit_i])
        return out


# ------------------------- Gemini -------------------------

class GeminiReranker(BaseReranker):
    """
    Calls Gemini to score each (q, passage) in [0,1].
    Falls back to MMR if the API/lib/key aren't there.
    """

    def __init__(self, model_name: str, mmr_fallback: MMRReranker):
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            log.warning("google-generativeai not available (%s). Falling back to MMR.", e)
            self.model = None
            self.fallback = mmr_fallback
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            log.warning("GOOGLE_API_KEY not set. Falling back to MMR.")
            self.model = None
            self.fallback = mmr_fallback
            return

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = genai.GenerativeModel(model_name)
        self.fallback = None
        self._score_re = re.compile(r"([01]?\.\d+|0|1)\b")

    def _score_pair(self, question: str, passage: str) -> float:
        prompt = (
            "You are a passage reranker.\n"
            "Given a QUESTION and a PASSAGE, output a single relevance score in [0,1]. "
            "Higher = more relevant. Output ONLY the number.\n\n"
            f"QUESTION: {question}\n"
            f"PASSAGE: {passage[:2000]}\n"
            "Score:"
        )
        try:
            resp = self.model.generate_content(prompt)
            txt = (resp.text or "").strip()
            m = self._score_re.search(txt)
            if not m:
                return 0.0
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except Exception as e:
            log.warning("Gemini scoring failed: %s", e)
            return 0.0

    def rerank(self, question: str, hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        if not hits:
            return []
        if self.model is None:
            return self.fallback.rerank(question, hits, top_k)

        scores = [self._score_pair(question, h.chunk.text) for h in hits]
        arr = np.asarray(scores, dtype="float32")

        order = np.argsort(-arr).tolist()
        out: List[RetrievalHit] = []
        for idx in order[:top_k]:
            hits[idx].score = float(arr[idx])  # already 0..1 from Gemini
            out.append(hits[idx])
        return out


# ------------------------- RRF -------------------------

class RRFReranker(BaseReranker):
    """
    Reciprocal Rank Fusion:
    Merge multiple ranked lists using 1 / (k_base + rank),
    sum contributions, then normalize so best doc = 1.0.
    """

    def __init__(self, k_base: int = 60):
        self.k_base = int(k_base)

    def rerank(self, question: str, hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        if not hits:
            return []

        fused: Dict[int, float] = {}
        keep: Dict[int, RetrievalHit] = {}

        for rank, h in enumerate(hits, start=1):
            cid = h.chunk.chunk_id
            contrib = 1.0 / (self.k_base + rank)
            fused[cid] = fused.get(cid, 0.0) + contrib
            if cid not in keep:
                keep[cid] = h

        return self._finalize(fused, keep, top_k)

    def rerank_multi(
        self,
        question: str,
        list_of_lists: List[List[RetrievalHit]],
        top_k: int,
    ) -> List[RetrievalHit]:
        fused: Dict[int, float] = {}
        keep: Dict[int, RetrievalHit] = {}

        for hit_list in list_of_lists:
            for rank, h in enumerate(hit_list, start=1):
                cid = h.chunk.chunk_id
                if cid is None:
                    continue
                contrib = 1.0 / (self.k_base + rank)
                fused[cid] = fused.get(cid, 0.0) + contrib
                if cid not in keep:
                    keep[cid] = h

        return self._finalize(fused, keep, top_k)

    def _finalize(
        self,
        fused: Dict[int, float],
        keep: Dict[int, RetrievalHit],
        top_k: int,
    ) -> List[RetrievalHit]:
        if not fused:
            return []

        sorted_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)
        max_score = fused[sorted_ids[0]]
        norm_denom = max(max_score, 1e-9)

        out: List[RetrievalHit] = []
        for cid in sorted_ids[:top_k]:
            hit = keep[cid]
            hit.score = float(fused[cid] / norm_denom)  # [0,1], best is 1.0
            out.append(hit)

        return out


# ------------------------- factory -------------------------

def build_reranker(settings, embedder) -> BaseReranker | None:
    t = (settings.reranker_type or "none").lower()

    if t == "none":
        log.info("Reranker disabled.")
        return None

    if t == "rrf":
        log.info("Using RRF reranker (k_base=%d).", settings.rrf_k_base)
        return RRFReranker(k_base=settings.rrf_k_base)

    if t == "mmr":
        log.info("Using MMR reranker (lambda=%.2f).", settings.mmr_lambda)
        return MMRReranker(embedder, settings.mmr_lambda)

    if t == "cross_encoder":
        mmr_fb = MMRReranker(embedder, settings.mmr_lambda)
        log.info("Using Cross-Encoder reranker: %s (MMR fallback).", settings.cross_encoder_model)
        return CrossEncoderReranker(settings.cross_encoder_model, mmr_fb)

    if t == "gemini":
        mmr_fb = MMRReranker(embedder, settings.mmr_lambda)
        log.info("Using Gemini reranker: %s (MMR fallback).", settings.gemini_reranker_model)
        return GeminiReranker(settings.gemini_reranker_model, mmr_fb)

    log.warning("Unknown reranker_type=%r. Disabling reranker.", settings.reranker_type)
    return None
