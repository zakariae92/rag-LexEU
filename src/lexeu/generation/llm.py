"""LLM calls through LiteLLM, with structured output, cost accounting and an evaluation cache.

LiteLLM gives one interface over Gemini, OpenAI, Ollama, vLLM...: switching provider is a
settings change, and every call reports tokens and cost from the same price table. The table is
the one shipped with the pinned LiteLLM version (no network fetch at import): costs are
reproducible and reviewed like any dependency bump.
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import AsyncIterator
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
    first_token_ms: float | None = None  # streaming only: when the first text arrived


class Completer(Protocol):
    @property
    def model_id(self) -> str: ...

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion: ...


class LlmClient(Completer, Protocol):
    def stream(
        self, messages: Messages, schema: type[BaseModel]
    ) -> AsyncIterator[str | Completion]:
        """Text chunks as they arrive, then one final Completion (usage, cost, timings)."""
        ...


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
        return self._completion(resp, (time.perf_counter() - start) * 1000)

    async def stream(
        self, messages: Messages, schema: type[BaseModel]
    ) -> AsyncIterator[str | Completion]:
        litellm = self._litellm
        start = time.perf_counter()
        first: float | None = None
        resp = await litellm.acompletion(
            model=self._model,
            messages=messages,
            response_format=schema,
            api_key=self._api_key,
            stream=True,
            stream_options={"include_usage": True},  # usage arrives with the last chunk
            **self._params,
        )
        chunks = []
        async for chunk in resp:
            chunks.append(chunk)
            text = chunk.choices[0].delta.content if chunk.choices else None
            if text:
                first = first or (time.perf_counter() - start) * 1000
                yield text
        full = litellm.stream_chunk_builder(chunks)
        completion = self._completion(full, (time.perf_counter() - start) * 1000)
        yield replace(completion, first_token_ms=round(first, 1) if first else None)

    def _completion(self, resp: Any, latency_ms: float) -> Completion:
        usage = getattr(resp, "usage", None)
        try:
            cost = float(self._litellm.completion_cost(completion_response=resp))
        except Exception:  # model missing from the price table: report 0 rather than fail
            cost = 0.0
        return Completion(
            content=resp.choices[0].message.content or "",
            model=self._model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=cost,
            latency_ms=round(latency_ms, 1),
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

    async def stream(
        self, messages: Messages, schema: type[BaseModel]
    ) -> AsyncIterator[str | Completion]:
        completion = await self.complete(messages, schema)  # evaluation path: no real streaming
        yield completion.content
        yield completion


class MockLlm:
    """A stand-in model for load tests: fixed latency, a valid cited answer, no provider call.

    Load tests measure our own stack (embedding, search, databases); a real provider would add
    its own variance and its rate limits (the free tier allows a few requests per minute).
    """

    def __init__(self, latency_ms: float, first_token_ms: float | None = None) -> None:
        self._latency = latency_ms / 1000
        self._first = (first_token_ms if first_token_ms is not None else latency_ms * 0.6) / 1000
        self._content = json.dumps(
            {"kind": "answer", "answer": "Simulated answer for a load test [1]. It cites [2]."}
        )

    @property
    def model_id(self) -> str:
        return "mock/llm"

    def _completion(self) -> Completion:
        return Completion(self._content, self.model_id, 2400, 40, 0.0, self._latency * 1000)

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        await asyncio.sleep(self._latency)
        return self._completion()

    async def stream(
        self, messages: Messages, schema: type[BaseModel]
    ) -> AsyncIterator[str | Completion]:
        await asyncio.sleep(self._first)
        yield self._content
        await asyncio.sleep(self._latency - self._first)
        yield replace(self._completion(), first_token_ms=self._first * 1000)
