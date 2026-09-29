"""LLM calls through LiteLLM, with structured output, cost accounting and an evaluation cache.

LiteLLM gives one interface over Gemini, OpenAI, Ollama, vLLM...: switching provider is a
settings change, and every call reports tokens and cost from the same price table. The table is
the one shipped with the pinned LiteLLM version (no network fetch at import): costs are
reproducible and reviewed like any dependency bump.
"""

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool = False


class LlmClient(Protocol):
    @property
    def model_id(self) -> str: ...

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion: ...


class LiteLlmClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "timeout": timeout_s,
            "num_retries": max_retries,  # exponential backoff on 429 / 5xx
        }
        if temperature is not None:
            self._params["temperature"] = temperature
        if reasoning_effort:
            self._params["reasoning_effort"] = reasoning_effort
        self._api_key = api_key
        # Importing LiteLLM takes seconds: do it at startup, not inside the first request, where
        # it would block the event loop (and every concurrent request) while it loads.
        import litellm

        self._litellm = litellm

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def params(self) -> dict[str, Any]:
        return {k: v for k, v in self._params.items() if k not in ("timeout", "num_retries")}

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        litellm = self._litellm
        start = time.perf_counter()
        resp = await litellm.acompletion(
            model=self._model,
            messages=messages,
            response_format=schema,
            api_key=self._api_key,
            **self._params,
        )
        latency = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:  # model missing from the price table: report 0 rather than fail
            cost = 0.0
        return Completion(
            content=resp.choices[0].message.content or "",
            model=self._model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=cost,
            latency_ms=round(latency, 1),
        )


class ResponseCache:
    """SQLite cache of completions, keyed by everything that determines the output.

    Evaluation only: re-running the golden set after a change that does not touch the prompt or
    the retrieved sources costs nothing, and CI runs are reproducible.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")

    @staticmethod
    def key(model: str, params: dict[str, Any], messages: Messages, schema: type[BaseModel]) -> str:
        payload = json.dumps(
            [model, params, messages, schema.model_json_schema()],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Completion | None:
        row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return Completion(**json.loads(row[0])) if row else None

    def put(self, key: str, completion: Completion) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(asdict(completion)))
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class CachedLlm:
    """A cache hit returns the original tokens, cost and latency (flagged `cached`), so reports
    describe the system, not the cache."""

    def __init__(self, inner: LiteLlmClient, cache: ResponseCache) -> None:
        self._inner = inner
        self._cache = cache
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        key = ResponseCache.key(self._inner.model_id, self._inner.params, messages, schema)
        if (hit := self._cache.get(key)) is not None:
            self.hits += 1
            return replace(hit, cached=True)
        self.misses += 1
        completion = await self._inner.complete(messages, schema)
        self._cache.put(key, completion)
        return completion
