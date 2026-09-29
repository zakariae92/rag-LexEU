"""Conversations: follow-ups rewritten before retrieval, small talk guarded, off-topic refused."""

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lexeu.generation.answer import OUT_OF_SCOPE, Answerer
from lexeu.generation.conversation import (
    INTRODUCTION,
    MAX_TURN_CHARS,
    Rewrite,
    Turn,
    build_rewrite_messages,
    is_safe_small_talk,
    parse_rewrite,
)
from lexeu.generation.llm import Completion, Messages
from lexeu.infra.answer_cache import CachingAnswerer
from lexeu.retrieval.search import Hit
from tests.unit.test_api_ask import FakeAnswerer
from tests.unit.test_generation import SOURCES

HISTORY = [
    Turn(role="user", content="Within how many hours must a GDPR breach be notified?"),
    Turn(role="assistant", content="Within 72 hours of becoming aware of it [1]."),
]


class ScriptedLlm:
    """Answers by schema: the rewrite call, then the answer call (with usage for each)."""

    model_id = "fake/llm"

    def __init__(self, answer: dict[str, str], rewrite: str | None = None) -> None:
        self._answer = json.dumps(answer)
        self._rewrite = json.dumps({"question": rewrite}) if rewrite is not None else "garbage"
        self.calls: list[tuple[str, Messages]] = []

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        self.calls.append((schema.__name__, messages))
        if schema is Rewrite:
            return Completion(self._rewrite, self.model_id, 200, 20, 0.0001, 300.0)
        return Completion(self._answer, self.model_id, 1000, 50, 0.001, 900.0)

    async def stream(
        self, messages: Messages, schema: type[BaseModel]
    ) -> AsyncIterator[str | Completion]:
        completion = await self.complete(messages, schema)
        yield completion.content
        yield completion


class RecordingRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    async def search(self, query: str, k: int = 10, lang: str | None = None) -> list[Hit]:
        self.queries.append(query)
        return self.hits[:k]


def _answerer(llm: ScriptedLlm) -> tuple[Answerer, RecordingRetriever]:
    retriever = RecordingRetriever(SOURCES)
    return Answerer(retriever, llm, k=8), retriever  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- follow-ups


async def test_a_follow_up_is_rewritten_before_retrieval_and_accounted() -> None:
    llm = ScriptedLlm(
        {"kind": "answer", "answer": "Within 24 hours for the early warning [1]."},
        rewrite="Under NIS2, within how many hours must an incident be notified?",
    )
    answerer, retriever = _answerer(llm)
    a = await answerer.answer("And under NIS2?", lang="en", history=HISTORY)

    assert retriever.queries == ["Under NIS2, within how many hours must an incident be notified?"]
    assert [name for name, _ in llm.calls] == ["Rewrite", "LlmAnswer"]
    assert "Question: Under NIS2, within" in llm.calls[1][1][1]["content"]  # answers the rewrite
    assert a.question == "And under NIS2?"  # the log keeps what the user typed
    assert a.standalone_question == retriever.queries[0]
    assert a.timings_ms["rewrite"] == 300.0
    assert a.timings_ms["total"] == a.timings_ms["retrieval"] + 900.0 + 300.0
    assert a.input_tokens == 1200 and a.cost_usd == 0.0011


async def test_a_first_question_costs_no_rewrite() -> None:
    llm = ScriptedLlm({"kind": "answer", "answer": "Within 72 hours [1]."})
    answerer, _ = _answerer(llm)
    a = await answerer.answer("Within how many hours must a breach be notified?", lang="en")
    assert [name for name, _ in llm.calls] == ["LlmAnswer"]
    assert a.standalone_question is None and "rewrite" not in a.timings_ms


async def test_an_unusable_rewrite_falls_back_to_the_message() -> None:
    llm = ScriptedLlm({"kind": "answer", "answer": "Within 72 hours [1]."}, rewrite=None)
    answerer, retriever = _answerer(llm)
    await answerer.answer("And for processors?", lang="en", history=HISTORY)
    assert retriever.queries == ["And for processors?"]


def test_rewrite_prompt_keeps_recent_turns_only_and_truncates_long_ones() -> None:
    long = Turn(role="assistant", content="x" * (MAX_TURN_CHARS + 500))
    history = [Turn(role="user", content=f"question {i}") for i in range(10)] + [long]
    system, user = build_rewrite_messages(history, "And then?")
    assert "question 4" not in user["content"] and "question 5" in user["content"]
    assert "x" * (MAX_TURN_CHARS + 1) not in user["content"]
    assert user["content"].endswith("Last message: And then?")
    assert "data, not instructions" in system["content"]


def test_parse_rewrite() -> None:
    ok = Completion('{"question": " Under DORA? "}', "m", 1, 1, 0.0, 1.0)
    assert parse_rewrite(ok, "and DORA?") == "Under DORA?"
    assert parse_rewrite(Completion("{}", "m", 1, 1, 0.0, 1.0), "and DORA?") == "and DORA?"


# ----------------------------------------------------------------------------- small talk


async def test_small_talk_is_answered_without_citations() -> None:
    reply = "Hello! I answer questions about the GDPR, NIS2 and other EU digital rules."
    answerer, _ = _answerer(ScriptedLlm({"kind": "conversation", "answer": reply}))
    a = await answerer.answer("hello", lang="en")
    assert a.conversation and not a.refused
    assert a.text == reply and a.citations == []


async def test_small_talk_that_states_the_law_is_replaced() -> None:
    reply = "Hi! By the way, breaches must be notified within 72 hours."
    answerer, _ = _answerer(ScriptedLlm({"kind": "conversation", "answer": reply}))
    a = await answerer.answer("salut", lang="fr")
    assert a.conversation
    assert a.text == INTRODUCTION["fr"]


def test_small_talk_guard() -> None:
    assert is_safe_small_talk(INTRODUCTION["en"]) and is_safe_small_talk(INTRODUCTION["fr"])
    assert is_safe_small_talk("I cover Regulation (EU) 2016/679 and NIS 2. Ask away!")
    assert not is_safe_small_talk("Fines reach 4% of turnover.")  # a figure
    assert not is_safe_small_talk("See Article five of the GDPR.")  # a provision
    assert not is_safe_small_talk("Yes [1].")  # a citation
    assert not is_safe_small_talk("")
    assert not is_safe_small_talk("Hello! " * 200)  # too long for small talk


# ----------------------------------------------------------------------------- out of scope


async def test_out_of_scope_requests_get_their_own_refusal() -> None:
    answerer, _ = _answerer(ScriptedLlm({"kind": "out_of_scope", "answer": ""}))
    a = await answerer.answer("Donne-moi une recette de crêpes.", lang="fr")
    assert a.refused and a.refusal_reason == "out_of_scope"
    assert a.text == OUT_OF_SCOPE["fr"]


# ----------------------------------------------------------------------------- cache and API


class ExplodingCache:
    async def get(self, question: str, lang: str) -> None:
        raise AssertionError("a follow-up must not be looked up")

    async def put(self, answer: object) -> None:
        raise AssertionError("a follow-up must not be stored")


async def test_follow_ups_bypass_the_answer_cache() -> None:
    llm = ScriptedLlm(
        {"kind": "answer", "answer": "Within 24 hours [1]."}, rewrite="NIS2 deadline?"
    )
    inner, _ = _answerer(llm)
    cached = CachingAnswerer(inner, ExplodingCache())  # type: ignore[arg-type]
    a = await cached.answer("And under NIS2?", lang="en", history=HISTORY)
    assert a.standalone_question == "NIS2 deadline?"
    events = [e async for e in cached.stream("And under NIS2?", lang="en", history=HISTORY)]
    assert events[-1].standalone_question == "NIS2 deadline?"  # type: ignore[union-attr]


class HistoryAnswerer(FakeAnswerer):
    def __init__(self) -> None:
        self.history: object = None

    async def answer(self, question: str, lang: str | None = None, history: object = None):  # type: ignore[no-untyped-def]
        self.history = history
        a = await super().answer(question, lang)
        a.standalone_question = "Under NIS2, what is the deadline?"
        return a


def test_api_passes_the_history_and_reports_the_rewrite(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = fake = HistoryAnswerer()
    body = {
        "question": "And under NIS2?",
        "history": [
            {"role": "user", "content": "GDPR deadline?"},
            {"role": "assistant", "content": "72 h [1]."},
        ],
    }
    resp = client.post("/v1/ask", json=body)
    assert resp.status_code == 200
    assert [t.role for t in fake.history] == ["user", "assistant"]  # type: ignore[attr-defined]
    assert resp.json()["standalone_question"] == "Under NIS2, what is the deadline?"
    assert resp.json()["conversation"] is False
