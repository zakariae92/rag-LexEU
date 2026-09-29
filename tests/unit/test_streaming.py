import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lexeu.generation.answer import REFUSALS, Answer, Answerer, Delta, Sources
from lexeu.generation.llm import Completion, Messages
from lexeu.generation.streaming import JsonFieldStream
from tests.unit.test_api_ask import FakeAnswerer
from tests.unit.test_generation import SOURCES, FakeRetriever

# ----------------------------------------------------------------------------- JSON field stream

PAYLOAD = json.dumps(
    {"answerable": True, "answer": 'Within "72 hours" [1].\nSee\tArt. 33 — délai [2].'},
    ensure_ascii=True,  # \u escapes for the accented characters and the dash
)


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_field_is_decoded_whatever_the_chunk_boundaries(size: int) -> None:
    stream = JsonFieldStream("answer")
    out = "".join(stream.feed(PAYLOAD[i : i + size]) for i in range(0, len(PAYLOAD), size))
    assert out == json.loads(PAYLOAD)["answer"]


def test_a_longer_key_with_the_same_prefix_is_not_the_field() -> None:
    stream = JsonFieldStream("answer")
    assert stream.feed('{"answerable": true, ') == ""
    assert stream.feed('"answer": "ok"}') == "ok"


def test_nothing_is_emitted_for_an_empty_answer() -> None:
    assert JsonFieldStream("answer").feed('{"answerable": false, "answer": ""}') == ""


# ----------------------------------------------------------------------------- answerer


class StreamingLlm:
    model_id = "fake/stream"

    def __init__(self, payload: dict[str, object]) -> None:
        self.content = json.dumps(payload)

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        raise AssertionError("the streaming path must not call complete()")

    async def stream(
        self, messages: Messages, schema: type[BaseModel]
    ) -> AsyncIterator[str | Completion]:
        for i in range(0, len(self.content), 5):
            yield self.content[i : i + 5]
        yield Completion(self.content, self.model_id, 900, 40, 0.001, 1200.0, first_token_ms=300.0)


async def _events(payload: dict[str, object]) -> list[object]:
    answerer = Answerer(FakeRetriever(SOURCES), StreamingLlm(payload), k=8)  # type: ignore[arg-type]
    return [e async for e in answerer.stream("Breach deadline?", lang="en")]


async def test_stream_sends_sources_then_text_then_the_checked_answer() -> None:
    events = await _events({"answerable": True, "answer": "Within 72 hours [1]."})
    assert isinstance(events[0], Sources) and len(events[0].hits) == 2
    deltas = [e.text for e in events if isinstance(e, Delta)]
    assert len(deltas) > 1 and "".join(deltas) == "Within 72 hours [1]."
    final = events[-1]
    assert isinstance(final, Answer) and not final.refused
    assert final.timings_ms["first_token"] >= 300.0  # retrieval + first token


async def test_a_streamed_draft_can_end_as_a_refusal() -> None:
    events = await _events({"answerable": True, "answer": "Always allowed [9]."})
    assert "".join(e.text for e in events if isinstance(e, Delta)) == "Always allowed [9]."
    final = events[-1]
    assert isinstance(final, Answer) and final.refused and final.refusal_reason == "ungrounded"
    assert final.text == REFUSALS["en"]


# ----------------------------------------------------------------------------- SSE route


class StreamingAnswerer(FakeAnswerer):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def stream(self, question: str, lang: str | None = None) -> AsyncIterator[object]:
        yield Sources([])
        yield Delta("Within 72 ")
        if self.fail:
            raise TimeoutError("provider timed out")
        yield Delta("hours [1].")
        yield await self.answer(question, lang)


def _parse(body: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in body.strip().split("\n\n"):
        event, data = block.split("\n")
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_sse_events_in_order(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = StreamingAnswerer()
    resp = client.post("/v1/ask/stream", json={"question": "Breach notification deadline?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "x-ratelimit-remaining" in resp.headers

    events = _parse(resp.text)
    assert [e for e, _ in events] == ["sources", "delta", "delta", "done"]
    answer_id = events[0][1]["answer_id"]
    assert events[-1][1]["answer_id"] == answer_id
    assert events[-1][1]["citations"][0]["citation"] == "Art. 33(1) GDPR"
    assert answer_id in app.state.answer_log.answers  # logged once the stream is complete


def test_provider_failure_mid_stream_is_an_error_event(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = StreamingAnswerer(fail=True)
    events = _parse(client.post("/v1/ask/stream", json={"question": "Deadline?"}).text)
    assert [e for e, _ in events] == ["sources", "delta", "error"]
    assert app.state.answer_log.answers == {}  # nothing half-answered is logged
