import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from lexeu.generation.answer import REFUSALS, Answerer
from lexeu.generation.grounding import check_grounding
from lexeu.generation.llm import CachedLlm, Completion, LiteLlmClient, Messages, ResponseCache
from lexeu.generation.prompt import build_messages
from lexeu.retrieval.search import Hit

# ----------------------------------------------------------------------------- grounding


def test_valid_citations_are_kept_in_order() -> None:
    check = check_grounding("The deadline is 72 hours [2]. The DPO must be informed [1, 2].", 3)
    assert check.cited == [2, 1]
    assert check.invalid == []
    assert check.grounded
    assert check.uncited_sentences == 0


def test_invalid_citations_are_removed_and_reported() -> None:
    check = check_grounding("Fines reach 4% of turnover [7]. Controllers must notify [1].", 2)
    assert check.invalid == [7]
    assert "[7]" not in check.text
    assert check.cited == [1]
    assert check.uncited_sentences == 1  # the fine sentence lost its only (invalid) citation


def test_answer_without_citations_is_not_grounded() -> None:
    check = check_grounding("The GDPR applies to everyone processing personal data.", 5)
    assert not check.grounded
    assert check.uncited_ratio == 1.0


def test_citation_after_the_full_stop_belongs_to_the_sentence() -> None:
    answer = "The controller notifies within 72 hours. [1]\nThen it documents the breach. [2]"
    check = check_grounding(answer, 2)
    assert check.uncited_sentences == 0


# ----------------------------------------------------------------------------- prompt


def _hit(key: str, citation: str, kind: str = "article") -> Hit:
    return Hit(f"{key}:en", key, "en", citation, 0.8, f"text of {key}", kind, "Article header")


SOURCES = [
    _hit("32016R0679:art_33:p1", "Art. 33(1) GDPR"),
    _hit("32016R0679:art_34:p1", "Art. 34(1) GDPR"),
]


def test_prompt_numbers_sources_and_sets_the_answer_language() -> None:
    system, user = build_messages("Quel est le délai ?", SOURCES, "fr")
    assert "Always write in French" in system["content"]
    assert "[1] Art. 33(1) GDPR" in user["content"]
    assert "[2] Art. 34(1) GDPR" in user["content"]
    assert user["content"].endswith("Question: Quel est le délai ?")


# ----------------------------------------------------------------------------- answerer


class FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    async def search(self, query: str, k: int = 10, lang: str | None = None) -> list[Hit]:
        return self.hits[:k]


class FakeLlm:
    model_id = "fake/llm"

    def __init__(self, payload: dict[str, object] | str) -> None:
        self.content = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls: list[Messages] = []

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        self.calls.append(messages)
        return Completion(self.content, self.model_id, 1000, 50, 0.001, 900.0)


def _answerer(llm: FakeLlm, hits: list[Hit] = SOURCES) -> Answerer:
    return Answerer(FakeRetriever(hits), llm, k=8)  # type: ignore[arg-type]


async def test_answer_maps_citations_back_to_provisions() -> None:
    llm = FakeLlm({"kind": "answer", "answer": "Within 72 hours [1]; data subjects too [2]."})
    a = await _answerer(llm).answer("How fast must a breach be notified?", lang="en")
    assert not a.refused
    assert [c.provision_key for c in a.citations] == [
        "32016R0679:art_33:p1",
        "32016R0679:art_34:p1",
    ]
    assert a.cost_usd == 0.001
    assert set(a.timings_ms) == {"retrieval", "generation", "total"}


async def test_model_refusal_becomes_a_localised_refusal() -> None:
    llm = FakeLlm({"kind": "not_in_sources", "answer": ""})
    a = await _answerer(llm).answer("Quelle est la TVA en Belgique ?", lang="fr")
    assert a.refused and a.refusal_reason == "not_in_sources"
    assert a.text == REFUSALS["fr"]
    assert a.citations == []


async def test_uncited_answer_is_refused_as_ungrounded() -> None:
    llm = FakeLlm({"kind": "answer", "answer": "Yes, always, see Article 99 [9]."})
    a = await _answerer(llm).answer("Is it always allowed?", lang="en")
    assert a.refused and a.refusal_reason == "ungrounded"
    assert a.invalid_citations == [9]


async def test_no_sources_skips_the_llm() -> None:
    llm = FakeLlm({"kind": "answer", "answer": "x [1]"})
    a = await _answerer(llm, hits=[]).answer("Anything?", lang="en")
    assert a.refused and a.refusal_reason == "no_sources"
    assert llm.calls == []


async def test_malformed_json_is_a_refusal_not_a_crash() -> None:
    a = await _answerer(FakeLlm("not json")).answer("Anything at all?", lang="en")
    assert a.refused and a.refusal_reason == "invalid_output"


# ----------------------------------------------------------------------------- cache


class CountingClient(LiteLlmClient):
    def __init__(self) -> None:
        super().__init__("fake/model")
        self.calls = 0

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        self.calls += 1
        return Completion('{"ok": true}', "fake/model", 10, 2, 0.5, 1234.0)


class Out(BaseModel):
    ok: bool


async def test_cache_returns_the_original_cost_and_latency(tmp_path: Path) -> None:
    inner = CountingClient()
    cache = ResponseCache(tmp_path / "llm.sqlite")
    llm = CachedLlm(inner, cache)
    msgs = [{"role": "user", "content": "hi"}]

    first = await llm.complete(msgs, Out)
    second = await llm.complete(msgs, Out)
    other = await llm.complete([{"role": "user", "content": "hello"}], Out)
    cache.close()

    assert inner.calls == 2  # the repeated call was served from the cache
    assert not first.cached and second.cached and not other.cached
    assert second.cost_usd == 0.5 and second.latency_ms == 1234.0
    assert (llm.hits, llm.misses) == (1, 2)


def test_cache_key_changes_with_generation_parameters() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    base = ResponseCache.key("m", {"reasoning_effort": "low"}, msgs, Out)
    assert base != ResponseCache.key("m", {"reasoning_effort": "minimal"}, msgs, Out)
    assert base != ResponseCache.key("other", {"reasoning_effort": "low"}, msgs, Out)


@pytest.mark.parametrize("effort", [None, "low"])
def test_reasoning_effort_is_only_sent_when_set(effort: str | None) -> None:
    client = LiteLlmClient("gemini/x", reasoning_effort=effort)
    assert ("reasoning_effort" in client.params) == (effort is not None)


async def test_load_test_mock_is_explicit_and_answers_with_citations() -> None:
    from lexeu.core.config import Settings
    from lexeu.generation.factory import make_llm

    settings = Settings(_env_file=None)
    settings.generation.mock_latency_ms = 5
    llm, _ = make_llm(settings)
    assert llm.model_id == "mock/llm"  # visible as such in metrics and the answer log

    answer = await Answerer(FakeRetriever(SOURCES), llm, k=8).answer("Anything?", lang="en")  # type: ignore[arg-type]
    assert not answer.refused and [c.n for c in answer.citations] == [1, 2]
