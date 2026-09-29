"""Answer cache: exact match on the normalised question, in Redis.

Not a semantic cache, by measurement (`lexeu eval cache`, ADR 0008): in legal questions, the
closest embeddings belong to questions with different answers ("maximum fines for essential
entities" vs "... important entities" at 0.98 cosine), so any similarity threshold that catches
paraphrases also serves wrong answers. An exact match on the normalised text is safe and still
absorbs real repeats (frequent questions, client retries, page reloads).

The key namespace contains everything that shapes an answer (index, model, prompt, retrieval
config): changing any of them invalidates the cache without a flush.
"""

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import AsyncIterator
from dataclasses import asdict

import structlog
from redis.asyncio import Redis

from lexeu.generation.answer import Answer, Answerer, Citation, Sources, StreamEvent
from lexeu.generation.conversation import Turn
from lexeu.observability import spans
from lexeu.retrieval.search import Hit
from lexeu.retrieval.sparse import detect_lang

log = structlog.get_logger(__name__)

CACHEABLE_REFUSALS = {"no_sources", "out_of_scope", "not_in_sources"}  # never model glitches
# Typographic quotes and guillemets as plain quotes: "l'IA" typed either way is one question.
_QUOTES = str.maketrans(
    {chr(c): "'" for c in (0x2018, 0x2019)} | {chr(c): '"' for c in (0x201C, 0x201D, 0xAB, 0xBB)}
)


def normalise(question: str) -> str:
    """Case, Unicode forms, quotes, spacing and trailing punctuation do not change a question."""
    text = unicodedata.normalize("NFKC", question).translate(_QUOTES).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[\s?!.;:]+$", "", text)


def namespace(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class AnswerCache:
    def __init__(self, redis: Redis, namespace: str, ttl_s: int = 7 * 24 * 3600) -> None:
        self._redis = redis
        self._ns = namespace
        self._ttl = ttl_s

    def key(self, question: str, lang: str) -> str:
        digest = hashlib.sha256(normalise(question).encode()).hexdigest()
        return f"answers:{self._ns}:{lang}:{digest}"

    async def get(self, question: str, lang: str) -> Answer | None:
        raw = await self._redis.get(self.key(question, lang))
        return _load(raw, question) if raw else None

    async def put(self, answer: Answer) -> bool:
        if answer.refused and answer.refusal_reason not in CACHEABLE_REFUSALS:
            return False
        await self._redis.set(self.key(answer.question, answer.lang), _dump(answer), ex=self._ttl)
        return True


def _dump(a: Answer) -> str:
    return json.dumps(
        {
            "lang": a.lang,
            "text": a.text,
            "refused": a.refused,
            "refusal_reason": a.refusal_reason,
            "citations": [asdict(c) for c in a.citations],
            # Source labels only: enough to show them again, without megabytes of legal text.
            "sources": [
                {"provision_key": h.provision_key, "citation": h.citation, "lang": h.lang}
                for h in a.sources
            ],
            "model": a.model,
            "prompt_version": a.prompt_version,
            "conversation": a.conversation,
        },
        ensure_ascii=False,
    )


def _load(raw: bytes | str, question: str) -> Answer:
    d = json.loads(raw)
    return Answer(
        question=question,
        lang=d["lang"],
        text=d["text"],
        refused=d["refused"],
        refusal_reason=d["refusal_reason"],
        citations=[Citation(**c) for c in d["citations"]],
        sources=[
            Hit("", s["provision_key"], s["lang"], s["citation"], 0.0, "") for s in d["sources"]
        ],
        model=d["model"],
        prompt_version=d["prompt_version"],
        conversation=d.get("conversation", False),
        cached=True,  # tokens and cost stay 0: a cache hit calls no model
    )


class CachingAnswerer:
    """The Answerer behind the answer cache. A cache outage only costs a model call."""

    def __init__(self, inner: Answerer, cache: AnswerCache) -> None:
        self._inner = inner
        self._cache = cache

    async def answer(
        self, question: str, lang: str | None = None, history: list[Turn] | None = None
    ) -> Answer:
        if history:  # a follow-up means something else in every conversation: never cached
            return await self._inner.answer(question, lang=lang, history=history)
        start = time.perf_counter()
        lang = lang or detect_lang(question)
        if (hit := await self._get(question, lang, start)) is not None:
            return hit
        answer = await self._inner.answer(question, lang=lang)
        await self._put(answer)
        return answer

    async def stream(
        self, question: str, lang: str | None = None, history: list[Turn] | None = None
    ) -> AsyncIterator[StreamEvent]:
        if history:
            async for event in self._inner.stream(question, lang=lang, history=history):
                yield event
            return
        start = time.perf_counter()
        lang = lang or detect_lang(question)
        if (hit := await self._get(question, lang, start)) is not None:
            yield Sources(hit.sources)
            yield hit
            return
        async for event in self._inner.stream(question, lang=lang):
            if isinstance(event, Answer):
                await self._put(event)
            yield event

    async def _get(self, question: str, lang: str, start: float) -> Answer | None:
        with spans.tracer.start_as_current_span("answer.cache_lookup") as span:
            try:
                hit = await self._cache.get(question, lang)
            except Exception as exc:
                log.warning("answer_cache_unavailable", error=repr(exc))
                span.record_exception(exc)
                return None
            span.set_attribute("lexeu.answer.cache_hit", hit is not None)
            if hit is not None:
                elapsed = round((time.perf_counter() - start) * 1000, 1)
                hit.timings_ms = {"cache": elapsed, "total": elapsed}
                spans.start_answer(span, question, lang)
                spans.finish_answer(span, hit)
            return hit

    async def _put(self, answer: Answer) -> None:
        try:
            await self._cache.put(answer)
        except Exception as exc:
            log.warning("answer_cache_unavailable", error=repr(exc))
