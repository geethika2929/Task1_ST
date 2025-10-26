from typing import List
from rag_pipeline.models import Chunk, Document


def chunk_document(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    starting_chunk_id: int = 0,
) -> List[Chunk]:
    """
    Sliding window chunking on raw text to produce semantic windows.
    Overlap keeps context stable.
    """
    chunks: List[Chunk] = []
    text = doc.text
    n = len(text)

    i = 0
    cid = starting_chunk_id
    while i < n:
        start = i
        end = min(i + chunk_size, n)
        chunk_text = text[start:end]

        # basic safety: don't store totally empty chunks
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    source=doc.source_path,
                    start_char=start,
                    end_char=end,
                    text=chunk_text.strip(),
                )
            )

        cid += 1
        i += max(1, chunk_size - chunk_overlap)

        if i <= start:
            # fallback guard if chunk_size <= overlap
            break

    return chunks


def chunk_corpus(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """
    Chunk all documents into one big chunk list with unique chunk_ids.
    """
    all_chunks: List[Chunk] = []
    next_id = 0
    for doc in docs:
        cset = chunk_document(
            doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            starting_chunk_id=next_id,
        )
        all_chunks.extend(cset)
        next_id += len(cset)

    return all_chunks
