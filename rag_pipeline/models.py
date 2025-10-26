from pydantic import BaseModel
from typing import List

class Document(BaseModel):
    doc_id: str           # e.g. "RLEG.pdf"
    source_path: str      # full path
    text: str             # full extracted text

class Chunk(BaseModel):
    chunk_id: int
    doc_id: str
    source: str
    start_char: int
    end_char: int
    text: str

class RetrievalHit(BaseModel):
    chunk: Chunk
    score: float  # FAISS similarity score (higher = closer)

class QueryRequest(BaseModel):
    question: str
    top_k: int

class AnswerPackage(BaseModel):
    question: str
    answer: str
    context_chunks: List[RetrievalHit]
