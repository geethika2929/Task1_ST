from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from pydantic import BaseModel


class Embedder(BaseModel):
    model_name: str
    _model: SentenceTransformer | None = None  # lazy init, not in schema

    class Config:
        arbitrary_types_allowed = True  # to allow SentenceTransformer

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        # normalize_embeddings=True -> cosine sim == dot == inner product
        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
