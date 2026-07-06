"""
Central configuration. Every tunable is a field on `Settings` and is
overridable via an environment variable of the same name (case-insensitive) or
a `.env` file — e.g. `TOP_K=3`, `EMBEDDING_BATCH_SIZE=8`, `LLAMACPP_URL=...`.

Import the singleton: `from config import settings`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,   # TOP_K / top_k both map to the field
        extra="ignore",         # ignore unrelated env vars (PYTHONUNBUFFERED, …)
    )

    # Paths
    data_dir: str = "data"
    vector_store_dir: str = "vector_store"
    index_dir: str = "indexdir"

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_batch_size: int = 32   # lower on a memory-tight Jetson

    # Chunking / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k: int = 10
    rrf_k: int = 60   # Reciprocal Rank Fusion constant (Cormack et al. 2009)

    # LLM generation (llama.cpp native /completion endpoint)
    llamacpp_url: str = "http://host.docker.internal:8080/completion"
    llamacpp_n_predict: int = 256
    llamacpp_temperature: float = 0.2
    llamacpp_connect_timeout: int = 10
    llamacpp_read_timeout: int = 300     # generous for slow Jetson inference
    llamacpp_max_context_chars: int = 2000
    llamacpp_top_n_chunks: int = 5

    # API
    api_token: str = ""   # when set, /download requires the X-API-Token header

    # Downloader (mavat / ArcGIS)
    mavat_request_timeout: int = 60
    mavat_insecure_ssl: bool = False   # escape hatch; keeps TLS verification on

    @property
    def raw_dir(self) -> str:
        return f"{self.data_dir}/raw"

    @property
    def processed_dir(self) -> str:
        return f"{self.data_dir}/processed"


settings = Settings()
