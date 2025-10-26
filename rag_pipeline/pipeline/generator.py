from typing import List
from google import genai
from rag_pipeline.models import RetrievalHit


def _build_context_block(hits: List[RetrievalHit]) -> str:
    """
    Build a single big context string from all retrieved chunks,
    no grouping, no headers. We just feed this to Gemini.
    """
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(
            f"[Chunk {i} | {h.chunk.doc_id} {h.chunk.start_char}-{h.chunk.end_char}]\n"
            f"{h.chunk.text}\n"
        )
    return "\n\n".join(parts)


def _call_gemini_unified(question: str, hits: List[RetrievalHit]) -> str:
    """
    Ask Gemini 2.5 Flash to answer the user's question using ALL context,
    and produce ONE concise bullet-list answer.

    No per-PDF headers.
    No 'Not in context'.
    Just the best factual summary from whatever is available.
    Hard cap ~50 words total.
    """
    client = genai.Client()  # uses GEMINI_API_KEY from env

    context_block = _build_context_block(hits)

    prompt = (
        "You are part of a RAG pipeline.\n"
        "You will be given:\n"
        "- A user question that may mention multiple topics (e.g. RLEG and Gamescon).\n"
        "- Retrieved context chunks from multiple PDFs.\n\n"
        "Your job:\n"
        "1. Use ONLY the provided context.\n"
        "2. Answer the whole question in a SINGLE bullet-list summary.\n"
        "3. Do NOT add section headers like 'RLEG:' or 'DINOv3:'.\n"
        "4. Do NOT write 'Not in context'. If some topic isn't covered, just skip it.\n"
        "5. Keep TOTAL output under 50 words.\n"
        "6. Bullets must start with '- '.\n\n"
        f"User question:\n{question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Now answer as described."
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return (resp.text or "").strip()


def _local_fallback_unified(hits: List[RetrievalHit]) -> str:
    """
    Offline fallback if Gemini isn't available.
    We'll just smash together top chunks and compress.
    Still: no headers, no 'Not in context'.
    """
    merged = " ".join(h.chunk.text.replace("\n", " ") for h in hits)
    merged = merged[:400].strip()  # rough cutoff

    # naive bullet splat under ~50 words
    # we'll take first ~50 words from merged
    words = merged.split()
    short_words = words[:50]
    short_text = " ".join(short_words)

    return f"- {short_text}..."


def generate_answer(question: str, hits: List[RetrievalHit], use_llm: bool) -> str:
    """
    Unified mode:
    - We don't branch by RLEG / DINOv3 / Gamescon anymore.
    - We always summarize everything together into one concise bullet list.
    - We allow multiple bullets, but it's all one block of bullets, no headers.
    - We skip anything we can't support from context instead of saying 'Not in context'.
    """

    if not hits:
        # absolutely nothing retrieved
        return "- (no relevant info found in documents)"

    if use_llm:
        try:
            return _call_gemini_unified(question, hits)
        except Exception:
            return _local_fallback_unified(hits)

    # offline mode
    return _local_fallback_unified(hits)
