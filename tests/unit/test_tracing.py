"""Spans of the answer pipeline: names, nesting, GenAI attributes, content capture."""

from collections.abc import AsyncGenerator, Iterator
from typing import Any, cast

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lexeu.generation.answer import Answerer, Delta
from lexeu.observability.spans import set_capture_content
from lexeu.observability.tracing import langfuse_otlp
from tests.unit.test_generation import SOURCES, FakeLlm, FakeRetriever
from tests.unit.test_streaming import StreamingLlm

_exporter = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def _provider() -> None:
    # The global provider can be set once per process: install ours for this module's spans.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_exporter))
    trace.set_tracer_provider(provider)


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    _exporter.clear()
    set_capture_content(True)
    yield _exporter
    set_capture_content(True)


def _by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {s.name: s for s in exporter.get_finished_spans()}


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


ANSWER = {"kind": "answer", "answer": "Within 72 hours [1]."}


async def test_answer_trace_has_retrieval_and_generation_children(
    spans: InMemorySpanExporter,
) -> None:
    answerer = Answerer(FakeRetriever(SOURCES), FakeLlm(ANSWER), k=8)  # type: ignore[arg-type]
    await answerer.answer("How fast must a breach be notified?", lang="en")

    by_name = _by_name(spans)
    root, retrieval, gen = by_name["answer"], by_name["retrieval"], by_name["llm.generate"]
    assert retrieval.parent is not None and retrieval.parent.span_id == root.context.span_id
    assert gen.parent is not None and gen.parent.span_id == root.context.span_id
    assert _attrs(gen)["gen_ai.request.model"] == "fake/llm"
    assert _attrs(gen)["gen_ai.usage.cost"] == 0.001
    assert _attrs(gen)["langfuse.observation.type"] == "generation"
    assert _attrs(root)["langfuse.trace.input"] == "How fast must a breach be notified?"
    assert _attrs(root)["lexeu.answer.citations"] == ("32016R0679:art_33:p1",)
    assert _attrs(retrieval)["lexeu.retrieval.hits"] == 2


async def test_capture_content_off_keeps_user_text_out(spans: InMemorySpanExporter) -> None:
    set_capture_content(False)
    answerer = Answerer(FakeRetriever(SOURCES), FakeLlm(ANSWER), k=8)  # type: ignore[arg-type]
    await answerer.answer("My name is Jane Doe, may my employer read my email?", lang="en")

    for span in spans.get_finished_spans():
        values = " ".join(str(v) for v in (span.attributes or {}).values())
        assert "Jane Doe" not in values and "72 hours" not in values
    assert _attrs(_by_name(spans)["llm.generate"])["gen_ai.usage.input_tokens"] == 1000


async def test_streaming_spans_do_not_leak_into_the_consumer(
    spans: InMemorySpanExporter,
) -> None:
    answerer = Answerer(FakeRetriever(SOURCES), StreamingLlm(ANSWER), k=8)  # type: ignore[arg-type]
    current_during_deltas = []
    async for event in answerer.stream("Breach deadline?", lang="en"):
        if isinstance(event, Delta):
            current_during_deltas.append(trace.get_current_span().get_span_context().span_id)

    by_name = _by_name(spans)
    assert {"answer", "retrieval", "llm.generate"} <= set(by_name)
    assert by_name["llm.generate"].parent.span_id == by_name["answer"].context.span_id  # type: ignore[union-attr]
    assert _attrs(by_name["llm.generate"])["lexeu.llm.first_token_ms"] == 300.0
    # The consumer never runs inside the generator's spans.
    assert set(current_during_deltas) == {0}


async def test_a_disconnected_stream_still_ends_its_span(spans: InMemorySpanExporter) -> None:
    answerer = Answerer(FakeRetriever(SOURCES), StreamingLlm(ANSWER), k=8)  # type: ignore[arg-type]
    events = cast(AsyncGenerator[object], answerer.stream("Breach deadline?", lang="en"))
    await anext(events)  # sources, then the client goes away
    await events.aclose()
    root = _by_name(spans)["answer"]
    assert root.end_time is not None
    assert not root.status.is_ok


def test_langfuse_uses_basic_auth_on_its_otel_endpoint() -> None:
    endpoint, headers = langfuse_otlp("https://cloud.langfuse.com/", "pk-lf-1", "sk-lf-2")
    assert endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert headers == {"Authorization": "Basic cGstbGYtMTpzay1sZi0y"}  # base64("pk-lf-1:sk-lf-2")
