from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment / runtime
    app_env: str = "dev"  # e.g. "dev", "prod", "test"

    # Paths
    data_dir: str = "./data"
    chroma_db_dir: str = "./data/chromadb"
    docs_dir: str = "./docs"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # External services
    openai_api_key: str | None = None
    llama_cloud_api_key: str | None = None
    groq_api_key: str | None = None   # NEW

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
