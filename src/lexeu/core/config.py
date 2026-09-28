"""Typed application settings, loaded from environment variables and `.env`.

Nested sections map to env vars with a double underscore, e.g. `POSTGRES__HOST`.
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5432
    user: str = "lexeu"
    password: SecretStr = SecretStr("lexeu")
    db: str = "lexeu"

    @property
    def dsn(self) -> str:
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"


class QdrantSettings(BaseModel):
    url: str = "http://127.0.0.1:6333"
    api_key: SecretStr | None = None


class RedisSettings(BaseModel):
    url: str = "redis://127.0.0.1:6379/0"


class ObjectStoreSettings(BaseModel):
    """S3-compatible object store (RustFS locally, AWS S3 / any S3 API in the cloud)."""

    endpoint: str = "http://127.0.0.1:9000"
    access_key: str = "lexeu"
    secret_key: SecretStr = SecretStr("lexeu-secret")
    bucket_raw: str = "lexeu-raw"


class IngestionSettings(BaseModel):
    # Official EU Publications Office API: content negotiation by CELEX number.
    cellar_url: str = "https://publications.europa.eu/resource/celex"
    user_agent: str = "rag-LexEU/0.1 (+https://github.com/zakariae92/rag-LexEU)"
    timeout_s: float = 60.0
    max_retries: int = 3
    max_concurrency: int = 3  # be polite to a public service
    chunk_max_chars: int = 1800


class EmbeddingSettings(BaseModel):
    url: str = "http://127.0.0.1:8081"  # text-embeddings-inference (compose profile `ml`)
    model_id: str = "BAAI/bge-m3"
    revision: str = (
        "5617a9f61b028005a4858fdac845db406aefb181"  # pinned on both batch and online paths
    )
    batch_size: int = 8  # TEI's ONNX CPU backend caps batches at 8
    cache_path: str = "data/embeddings/bge-m3.sqlite"


class EvalSettings(BaseModel):
    golden_path: str = "eval/golden/golden_v1.yaml"
    thresholds_path: str = "eval/thresholds.yaml"
    reports_dir: str = "eval/reports"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "rag-lexeu-retrieval"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = False
    probe_timeout_s: float = 2.0

    postgres: PostgresSettings = PostgresSettings()
    qdrant: QdrantSettings = QdrantSettings()
    redis: RedisSettings = RedisSettings()
    object_store: ObjectStoreSettings = ObjectStoreSettings()
    ingestion: IngestionSettings = IngestionSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    eval: EvalSettings = EvalSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
