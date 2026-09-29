import json
from typing import Any

import pytest

from lexeu.generation.answer import Answer, Answerer, Delta, Sources
from lexeu.infra.answer_cache import AnswerCache, CachingAnswerer, namespace, normalise
from tests.unit.test_generation import SOURCES, FakeLlm, FakeRetriever
from tests.unit.test_streaming import StreamingLlm

RSQ = chr(0x2019)  # typographic apostrophe, as typed on phones and in Word


class DictRedis:
    def __init__(self, broken: bool = False) -> None:
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int] = {}
        self.broken = broken

    async def get(self, key: str) -> str | None:
        if self.broken:
            raise ConnectionError("redis down")
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int) -> None:
        if self.broken:
            raise ConnectionError("redis down")
        self.data[key], self.ttl[key] = value, ex


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("What is personal data?", "  what is PERSONAL data ??"),
        (f"Qu{RSQ}est-ce que l{RSQ}IA ?", "qu'est-ce que l'IA"),
        ("Deadline\tfor  notification?", "deadline for notification"),
    ],
)
def test_normalisation_ignores_form_not_content(a: str, b: str) -> None:
    assert normalise(a) == normalise(b)


def test_normalisation_keeps_what_changes_the_answer() -> None:
    assert normalise("fines for essential entities") != normalise("fines for important entities")


def test_namespace_changes_with_anything_that_shapes_answers() -> None:
    base = namespace("fp1", "gemini/a", "2")
    assert base == namespace("fp1", "gemini/a", "2")
    assert base != namespace("fp2", "gemini/a", "2")  # rebuilt index
    assert base != namespace("fp1", "gemini/a", "3")  # new prompt


def _caching(llm: Any, redis: DictRedis) -> CachingAnswerer:
    inner = Answerer(FakeRetriever(SOURCES), llm, k=8)  # type: ignore[arg-type]
    return CachingAnswerer(inner, AnswerCache(redis, "ns", ttl_s=60))  # type: ignore[arg-type]


async def test_second_identical_question_is_served_from_cache() -> None:
    llm = FakeLlm({"kind": "answer", "answer": "Within 72 hours [1]."})
    redis = DictRedis()
    cached = _caching(llm, redis)

    first = await cached.answer("Breach deadline?", lang="en")
    second = await cached.answer("breach deadline", lang="en")

    assert len(llm.calls) == 1
    assert not first.cached and second.cached
    assert second.text == first.text and second.citations == first.citations
    assert second.cost_usd == 0.0 and set(second.timings_ms) == {"cache", "total"}
    assert list(redis.ttl.values()) == [60]


async def test_languages_do_not_share_entries() -> None:
    llm = FakeLlm({"kind": "answer", "answer": "Within 72 hours [1]."})
    cached = _caching(llm, DictRedis())
    await cached.answer("GDPR", lang="en")
    await cached.answer("GDPR", lang="fr")
    assert len(llm.calls) == 2


async def test_model_glitches_are_not_cached() -> None:
    llm = FakeLlm({"kind": "answer", "answer": "Uncited claim [9]."})  # -> ungrounded refusal
    redis = DictRedis()
    answer = await _caching(llm, redis).answer("Anything?", lang="en")
    assert answer.refusal_reason == "ungrounded"
    assert redis.data == {}


async def test_a_cache_outage_falls_back_to_the_model() -> None:
    llm = FakeLlm({"kind": "answer", "answer": "Within 72 hours [1]."})
    answer = await _caching(llm, DictRedis(broken=True)).answer("Breach deadline?", lang="en")
    assert not answer.refused and len(llm.calls) == 1


async def test_streaming_hit_replays_sources_and_answer() -> None:
    llm = StreamingLlm({"kind": "answer", "answer": "Within 72 hours [1]."})
    redis = DictRedis()
    cached = _caching(llm, redis)
    first = [e async for e in cached.stream("Breach deadline?", lang="en")]
    second = [e async for e in cached.stream("Breach deadline?", lang="en")]

    assert any(isinstance(e, Delta) for e in first)
    assert [type(e) for e in second] == [Sources, Answer]
    stored = json.loads(next(iter(redis.data.values())))
    assert "text of" not in json.dumps(stored["sources"])  # labels only, not the legal text
