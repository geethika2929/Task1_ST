from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_id: str
    source_path: str
    text: str


class Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: Optional[int] = None
    doc_id: str
    source: str
    start_char: int
    end_char: int
    text: str


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    score: Optional[float] = None  # we'll overwrite this with reranker / RRF score
    chunk: Chunk


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    top_k: int


class AnswerPackage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str
    context_chunks: List[RetrievalHit]
    confidence: float  # 0.0 - 1.0
