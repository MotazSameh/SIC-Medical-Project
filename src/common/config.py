"""
API configuration
"""
from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    """API settings"""
    
    # API metadata
    API_TITLE: str = "Evidence-Bound Drug RAG API"
    API_VERSION: str = "0.1.0"
    API_DESCRIPTION: str = "Retrieval API for evidence-bound drug information (Phase 0 PoC)"
    
    # Paths
    DATA_DIR: Path = Path("data/processed")
    VECTOR_INDEX_PATH: Path = DATA_DIR / "vector_index.pkl"
    BM25_INDEX_PATH: Path = DATA_DIR / "bm25_index.pkl"
    CHUNKS_PATH: Path = DATA_DIR / "chunks_with_embeddings.pkl"
    
    # Retrieval defaults
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 50
    DEFAULT_RETRIEVER: str = "vector"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_prefix = "RAG_"
        case_sensitive = True


settings = Settings()
