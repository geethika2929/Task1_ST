from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    data_dir: Path = Path("data")

    sqlite_path: Path = Path("storage/chunks.db")
    faiss_index_path: Path = Path("storage/index.faiss")
    id_map_path: Path = Path("storage/chunk_ids.npy")

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 200

    # bumped from 4 -> 12 to widen cosine similarity search results
    default_top_k: int = 12

def get_settings() -> Settings:
    return Settings()
