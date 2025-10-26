from pydantic import BaseModel, ConfigDict

class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # storage
    data_dir: str = "data"
    sqlite_path: str = "store/chunks.sqlite3"
    faiss_index_path: str = "store/index.faiss"
    id_map_path: str = "store/chunk_ids.npy"

    # retrieval / chunking
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 300

    # retrieval fan-out and display limits
    default_top_k: int = 12          # how many chunks we actually pass to answer + show in UI
    reranker_candidates: int = 50    # how many we pull from FAISS per query before fusion/rerank

    # reranking config
    reranker_type: str = "mmr"       # one of: "rrf", "mmr", "cross_encoder", "gemini", "none"
    mmr_lambda: float = 0.6          # used only if reranker_type == "mmr"

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional
    gemini_reranker_model: str = "gemini-2.5-flash"                    # optional (GOOGLE_API_KEY required)

    # RRF config
    rrf_k_base: int = 60             # standard RRF constant (controls how fast scores decay)

def get_settings() -> Settings:
    return Settings()
