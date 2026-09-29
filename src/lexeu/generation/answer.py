"""Question -> retrieval -> grounded, cited answer (or an explicit refusal).

    question -> language -> top-k provisions -> prompt -> LLM (JSON) -> grounding check
             -> answer with citations mapped back to provisions | refusal

Refusal has three causes, all reported: nothing retrieved, the model says the sources do not
answer, or the answer cites no valid source (treated as ungrounded, never shown).
"""

import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import structlog
from opentelemetry import context as otel_context
from opentelemetry import trace
from pydantic import ValidationError

from lexeu.generation.grounding import check_grounding
from lexeu.generation.llm import Completion, LlmClient
from lexeu.generation.prompt import PROMPT_VERSION, LlmAnswer, build_messages
from lexeu.generation.streaming import JsonFieldStream
from lexeu.observability import spans
from lexeu.retrieval.search import Hit, Retriever
from lexeu.retrieval.sparse import detect_lang

log = structlog.get_logger(__name__)

RefusalReason = Literal["no_sources", "not_in_sources", "ungrounded", "invalid_output"]

REFUSALS = {
    "en": "I could not find the answer in the EU texts I cover (GDPR, AI Act, DORA, NIS2, DSA, "
    "Data Act). Try rephrasing, or check the official text on EUR-Lex.",
    "fr": "Je n'ai pas trouvé la réponse dans les textes européens que je couvre (RGPD, AI Act, "
    "DORA, NIS2, DSA, Data Act). Reformulez la question ou consultez le texte officiel sur "
    "EUR-Lex.",
}


@dataclass(frozen=True)
class Citation:
    n: int  # the [n] marker in the answer
    provision_key: str
    citation: str  # human-readable label, e.g. "Art. 33(1) GDPR"
    lang: str
    chunk_id: str


@dataclass(frozen=True)
class Sources:
    hits: list[Hit]


@dataclass(frozen=True)
class Delta:
    text: str


@dataclass
class Answer:
    question: str
    lang: str
    text: str
    refused: bool
    refusal_reason: RefusalReason | None
    citations: list[Citation]
    sources: list[Hit]
    model: str
    prompt_version: str = PROMPT_VERSION
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    uncited_ratio: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sources"] = [
            {"n": i, "provision_key": h.provision_key, "citation": h.citation, "score": h.score}
            for i, h in enumerate(self.sources, start=1)
        ]
        return d


StreamEvent = Sources | Delta | Answer


class Answerer:
    def __init__(
        self, retriever: Retriever, llm: LlmClient, k: int = 8, expand_chars: int = 0
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k
        self._expand_chars = expand_chars  # 0 = give the model the retrieved parts only

    async def answer(self, question: str, lang: str | None = None) -> Answer:
        with spans.tracer.start_as_current_span("answer") as root:
            spans.start_answer(root, question, lang)
            lang, hits, t_retrieval = await self._retrieve(question, lang)
            if not hits:
                answer = self._refuse(
                    question, lang, "no_sources", hits, None, {"retrieval": t_retrieval}
                )
            else:
                messages = build_messages(question, hits, lang)
                with spans.tracer.start_as_current_span("llm.generate") as gen:
                    completion = await self._llm.complete(messages, LlmAnswer)
                    spans.generation(gen, self._llm.model_id, messages, completion)
                answer = self._finish(question, lang, hits, completion, t_retrieval)
            spans.finish_answer(root, answer)
            return answer

    async def stream(self, question: str, lang: str | None = None) -> AsyncIterator[StreamEvent]:
        """`Sources` right after retrieval, `Delta`s while the model writes, then the `Answer`.

        The final Answer is authoritative: the grounding check runs on the complete text, so a
        streamed draft can still end as a refusal (clients replace the draft with it).
        """
        # Spans are passed explicitly, never made "current" across a `yield`: the consumer runs
        # between yields and would otherwise inherit (and corrupt) this generator's context.
        root = spans.tracer.start_span("answer")
        root_ctx = trace.set_span_in_context(root)
        spans.start_answer(root, question, lang)
        try:
            token = otel_context.attach(root_ctx)
            try:
                lang, hits, t_retrieval = await self._retrieve(question, lang)
            finally:
                otel_context.detach(token)
            yield Sources(hits)
            if not hits:
                answer = self._refuse(
                    question, lang, "no_sources", hits, None, {"retrieval": t_retrieval}
                )
                spans.finish_answer(root, answer)
                yield answer
                return
            field = JsonFieldStream("answer")
            completion: Completion | None = None
            messages = build_messages(question, hits, lang)
            gen = spans.tracer.start_span("llm.generate", context=root_ctx)
            try:
                async for part in self._llm.stream(messages, LlmAnswer):
                    if isinstance(part, Completion):
                        completion = part
                    elif text := field.feed(part):
                        yield Delta(text)
                if completion is None:  # a client ending its stream without usage breaks it
                    raise RuntimeError("LLM stream ended without a final completion")
                spans.generation(gen, self._llm.model_id, messages, completion)
            finally:
                gen.end()
            answer = self._finish(question, lang, hits, completion, t_retrieval)
            spans.finish_answer(root, answer)
            yield answer
        except BaseException as exc:  # includes a client disconnect (GeneratorExit)
            root.record_exception(exc)
            root.set_status(trace.Status(trace.StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            root.end()

    async def _retrieve(self, question: str, lang: str | None) -> tuple[str, list[Hit], float]:
        start = time.perf_counter()
        with spans.tracer.start_as_current_span("retrieval") as span:
            lang = lang or detect_lang(question)
            hits = await self._retriever.search(question, k=self._k)
            if self._expand_chars:
                hits = await self._retriever.expand(hits, max_chars=self._expand_chars)
            spans.retrieval(span, hits, self._k, bool(self._expand_chars))
        return lang, hits, round((time.perf_counter() - start) * 1000, 1)

    def _finish(
        self, question: str, lang: str, hits: list[Hit], completion: Completion, t_retrieval: float
    ) -> Answer:
        timings = {
            "retrieval": t_retrieval,
            "generation": completion.latency_ms,
            "total": round(t_retrieval + completion.latency_ms, 1),
        }
        if completion.first_token_ms is not None:  # time to first token, as the user sees it
            timings["first_token"] = round(t_retrieval + completion.first_token_ms, 1)

        try:
            out = LlmAnswer.model_validate_json(completion.content)
        except ValidationError:
            log.warning("llm_invalid_output", model=completion.model, content=completion.content)
            return self._refuse(question, lang, "invalid_output", hits, completion, timings)
        if not out.answerable or not out.answer.strip():
            return self._refuse(question, lang, "not_in_sources", hits, completion, timings)

        check = check_grounding(out.answer, len(hits))
        if not check.grounded:
            log.warning("llm_ungrounded_answer", model=completion.model, answer=out.answer)
            answer = self._refuse(question, lang, "ungrounded", hits, completion, timings)
            answer.invalid_citations = check.invalid
            return answer

        citations = [
            Citation(n, hits[n - 1].provision_key, hits[n - 1].citation, hits[n - 1].lang,
                     hits[n - 1].chunk_id)
            for n in check.cited
        ]  # fmt: skip
        return Answer(
            question=question,
            lang=lang,
            text=check.text,
            refused=False,
            refusal_reason=None,
            citations=citations,
            sources=hits,
            invalid_citations=check.invalid,
            uncited_ratio=round(check.uncited_ratio, 3),
            timings_ms=timings,
            **_usage(completion),
        )

    def _refuse(
        self,
        question: str,
        lang: str,
        reason: RefusalReason,
        hits: list[Hit],
        completion: Completion | None,
        timings: dict[str, float],
    ) -> Answer:
        return Answer(
            question=question,
            lang=lang,
            text=REFUSALS.get(lang, REFUSALS["en"]),
            refused=True,
            refusal_reason=reason,
            citations=[],
            sources=hits,
            timings_ms=timings,
            **(_usage(completion) if completion else {"model": self._llm.model_id}),
        )


def _usage(c: Completion) -> dict[str, Any]:
    return {
        "model": c.model,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "cost_usd": c.cost_usd,
        "cached": c.cached,
    }
