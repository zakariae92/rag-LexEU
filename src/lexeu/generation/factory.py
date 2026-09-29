"""Build the LLM clients from settings (API, CLI and evaluation share this)."""

from pathlib import Path

import structlog

from lexeu.core.config import Settings
from lexeu.generation.llm import CachedLlm, LiteLlmClient, LlmClient, MockLlm, ResponseCache

log = structlog.get_logger(__name__)


class MissingApiKeyError(RuntimeError):
    pass


# The generator and the judge can come from different providers, in one process: each call gets
# the key of its own model's provider.
PROVIDER_KEYS = {"gemini/": "gemini_api_key", "mistral/": "mistral_api_key"}


def api_key_for(settings: Settings, model: str) -> str | None:
    """The API key for this model's provider; None for providers configured another way."""
    for prefix, field in PROVIDER_KEYS.items():
        if model.startswith(prefix):
            secret = getattr(settings, field)
            if secret is None or not secret.get_secret_value():  # unset, or `KEY=` in a .env
                raise MissingApiKeyError(f"{field.upper()} is not set (see .env.example)")
            return str(secret.get_secret_value())
    return None  # e.g. ollama/: local, or read by LiteLLM from its own environment variables


def make_llm(
    settings: Settings,
    model: str | None = None,
    cache: bool = False,
    reasoning_effort: str | None = None,
) -> tuple[LlmClient, ResponseCache | None]:
    cfg = settings.generation
    if cfg.mock_latency_ms is not None:
        log.warning("llm_mocked_for_load_test", latency_ms=cfg.mock_latency_ms)
        return MockLlm(cfg.mock_latency_ms), None
    model = model or cfg.model
    client = LiteLlmClient(
        model,
        api_key=api_key_for(settings, model),
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
        reasoning_effort=reasoning_effort or cfg.reasoning_effort,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        requests_per_minute=cfg.requests_per_minute,
    )
    if not cache:
        return client, None
    store = ResponseCache(Path(cfg.cache_path))
    return CachedLlm(client, store), store
