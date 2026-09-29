"""Build the LLM clients from settings (API, CLI and evaluation share this)."""

from pathlib import Path

from lexeu.core.config import Settings
from lexeu.generation.llm import CachedLlm, LiteLlmClient, LlmClient, ResponseCache


class MissingApiKeyError(RuntimeError):
    pass


def make_llm(
    settings: Settings,
    model: str | None = None,
    cache: bool = False,
    reasoning_effort: str | None = None,
) -> tuple[LlmClient, ResponseCache | None]:
    cfg = settings.generation
    model = model or cfg.model
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    if model.startswith("gemini/") and not key:
        raise MissingApiKeyError("GEMINI_API_KEY is not set (see .env.example)")
    client = LiteLlmClient(
        model,
        api_key=key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
        reasoning_effort=reasoning_effort or cfg.reasoning_effort,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )
    if not cache:
        return client, None
    store = ResponseCache(Path(cfg.cache_path))
    return CachedLlm(client, store), store
