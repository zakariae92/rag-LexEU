"""Typed application settings, loaded from environment variables and `.env`.

Nested sections map to env vars with a double underscore, e.g. `POSTGRES__HOST`.
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from lexeu.retrieval.config import RetrievalConfig


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


# The configuration the API serves and the CI gate evaluates, chosen by the M3 ablation (ADR 0006).
PRODUCTION_RETRIEVAL = RetrievalConfig(
    name="dense-recitals-demote",
    description="BGE-M3 dense, recital scores x0.9 so articles win close calls",
    recitals="demote",
    recital_penalty=0.9,
)


class GenerationSettings(BaseModel):
    """LLM calls go through LiteLLM: switching provider is one string (`gemini/`, `ollama/`)."""

    model: str = "gemini/gemini-3.1-flash-lite"  # M4: cheapest tested model, p95 < 2 s
    # A stronger model than the generator grades the answers. Flash, not Pro: the Pro free tier
    # allows about one judged run per day (ADR 0007).
    judge_model: str = "gemini/gemini-3.8-flash"
    judge_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "low"
    temperature: float | None = None  # None = provider default (Gemini 3 advises against < 1)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "minimal"
    max_output_tokens: int = 1024
    k: int = 8  # provisions given to the model as numbered sources
    expand_chars: int = 6000  # each source = its whole provision, up to this size (0 = off)
    timeout_s: float = 30.0
    max_retries: int = 3
    concurrency: int = 4  # parallel questions during evaluation (provider rate limits)
    cache_path: str = "data/llm/responses.sqlite"  # evaluation only: identical calls are free
    # Load tests only: replace the model with a simulated one of this latency. Never in production.
    mock_latency_ms: float | None = None


class ServingSettings(BaseModel):
    # Answers computed concurrently before new ones wait, then get a 503 (admission control).
    # Sized from the load test: the CPU embedding server sustains about 6 answers/s (ADR 0009).
    max_inflight_answers: int = 8
    queue_timeout_s: float = 1.0


class AuthSettings(BaseModel):
    required: bool = True  # secure by default; set AUTH__REQUIRED=false for local exploration
    default_rate_limit_per_min: int = 30  # for new API keys
    anonymous_rate_limit_per_min: int = 10  # per client IP, when auth is not required


class CacheSettings(BaseModel):
    """Exact-match answer cache (ADR 0008: a similarity-based cache served wrong answers)."""

    enabled: bool = True
    ttl_s: int = 7 * 24 * 3600  # the law changes slowly; a new index changes the namespace anyway


class TracingSettings(BaseModel):
    enabled: bool = True
    service_name: str = "rag-lexeu-api"
    otlp_endpoint: str | None = None  # e.g. Jaeger: http://127.0.0.1:4318/v1/traces (profile `obs`)
    sample_ratio: float = 1.0  # keep every trace at this volume; lower it under real traffic
    capture_content: bool = True  # questions, prompts and answers in spans (see tracing.py)
    langfuse_host: str = "https://cloud.langfuse.com"  # EU region
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None


class EvalSettings(BaseModel):
    golden_path: str = "eval/golden/golden_v1.yaml"
    experiments_path: str = "eval/experiments.yaml"
    thresholds_path: str = "eval/thresholds.yaml"
    reports_dir: str = "eval/reports"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "rag-lexeu-retrieval"
    mlflow_experiment_answers: str = "rag-lexeu-answers"


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
    gemini_api_key: SecretStr | None = None  # GEMINI_API_KEY

    postgres: PostgresSettings = PostgresSettings()
    qdrant: QdrantSettings = QdrantSettings()
    redis: RedisSettings = RedisSettings()
    object_store: ObjectStoreSettings = ObjectStoreSettings()
    ingestion: IngestionSettings = IngestionSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalConfig = PRODUCTION_RETRIEVAL
    generation: GenerationSettings = GenerationSettings()
    auth: AuthSettings = AuthSettings()
    serving: ServingSettings = ServingSettings()
    cache: CacheSettings = CacheSettings()
    tracing: TracingSettings = TracingSettings()
    eval: EvalSettings = EvalSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
