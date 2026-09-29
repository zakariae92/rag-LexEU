"""Span attributes for the answer pipeline, readable by any OTLP backend and by Langfuse.

Standard GenAI semantic conventions (`gen_ai.*`) carry model, tokens and cost; Langfuse's own
`langfuse.*` attributes carry the input and output it displays. Without a configured tracer
provider, every call here is a no-op.
"""

import json
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.trace import Span

if TYPE_CHECKING:
    from lexeu.generation.answer import Answer
    from lexeu.generation.llm import Completion, Messages
    from lexeu.retrieval.search import Hit

tracer = trace.get_tracer("lexeu")

_settings = {"capture_content": True}


def set_capture_content(enabled: bool) -> None:
    _settings["capture_content"] = enabled


def _content(span: Span, key: str, value: Any) -> None:
    if _settings["capture_content"]:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        span.set_attribute(key, text)


def start_answer(span: Span, question: str, lang: str | None) -> None:
    span.set_attribute("langfuse.trace.name", "answer")
    if lang:
        span.set_attribute("lexeu.lang", lang)
    _content(span, "langfuse.trace.input", question)
    _content(span, "langfuse.observation.input", question)


def retrieval(span: Span, hits: "list[Hit]", k: int, expanded: bool) -> None:
    span.set_attribute("lexeu.retrieval.k", k)
    span.set_attribute("lexeu.retrieval.hits", len(hits))
    span.set_attribute("lexeu.retrieval.expanded", expanded)
    span.set_attribute("lexeu.retrieval.provisions", [h.provision_key for h in hits])
    if hits:
        span.set_attribute("lexeu.retrieval.top_score", hits[0].score)


def generation(span: Span, model: str, messages: "Messages", completion: "Completion") -> None:
    span.set_attribute("langfuse.observation.type", "generation")
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.input_tokens", completion.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", completion.output_tokens)
    span.set_attribute("gen_ai.usage.cost", completion.cost_usd)
    if completion.first_token_ms is not None:
        span.set_attribute("lexeu.llm.first_token_ms", completion.first_token_ms)
    _content(span, "langfuse.observation.input", messages)
    _content(span, "langfuse.observation.output", completion.content)


def finish_answer(span: Span, answer: "Answer") -> None:
    span.set_attribute("lexeu.answer.refused", answer.refused)
    if answer.refusal_reason:
        span.set_attribute("lexeu.answer.refusal_reason", answer.refusal_reason)
    span.set_attribute("lexeu.answer.citations", [c.provision_key for c in answer.citations])
    span.set_attribute("lexeu.answer.invalid_citations", len(answer.invalid_citations))
    span.set_attribute("lexeu.answer.cost_usd", answer.cost_usd)
    span.set_attribute("lexeu.answer.cache_hit", answer.cached)
    _content(span, "langfuse.trace.output", answer.text)
    _content(span, "langfuse.observation.output", answer.text)
