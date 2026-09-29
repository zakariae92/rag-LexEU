"""Message -> retrieval -> grounded, cited answer, small talk, or an explicit refusal.

    message (+ recent turns) -> language -> standalone question (follow-ups only)
            -> top-k provisions -> prompt -> LLM (JSON: kind + answer) -> by kind:
               answer         -> grounding check -> citations mapped back to provisions
               conversation   -> small-talk guard -> reply without citations
               out_of_scope, not_in_sources -> refusal

Refusal causes, all reported: nothing retrieved, a request outside the covered regulations, the
model says the sources do not answer, or the answer cites no valid source (ungrounded, never
shown).
"""

import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import structlog
from opentelemetry import context as otel_context
from opentelemetry import trace
from pydantic import ValidationError

from lexeu.generation.conversation import (
    INTRODUCTION,
    Rewrite,
    Turn,
    build_rewrite_messages,
    is_safe_small_talk,
    parse_rewrite,
)
from lexeu.generation.grounding import check_grounding
from lexeu.generation.llm import Completion, LlmClient
from lexeu.generation.prompt import PROMPT_VERSION, LlmAnswer, build_messages
from lexeu.generation.streaming import JsonFieldStream
from lexeu.observability import spans
from lexeu.retrieval.search import Hit, Retriever
from lexeu.retrieval.sparse import detect_lang

log = structlog.get_logger(__name__)

RefusalReason = Literal[
    "no_sources", "out_of_scope", "not_in_sources", "ungrounded", "invalid_output"
]

REFUSALS = {
    "en": "I could not find the answer in the EU texts I cover (GDPR, AI Act, DORA, NIS2, DSA, "
    "Data Act). Try rephrasing, or check the official text on EUR-Lex.",
    "fr": "Je n'ai pas trouvé la réponse dans les textes européens que je couvre (RGPD, AI Act, "
    "DORA, NIS2, DSA, Data Act). Reformulez la question ou consultez le texte officiel sur "
    "EUR-Lex.",
}

OUT_OF_SCOPE = {
    "en": "I can't answer that question: I only cover EU digital regulation (GDPR, AI Act, DORA, "
    "NIS2, DSA, Data Act). Ask me about these texts.",
    "fr": "Je ne peux pas répondre à cette question : je couvre uniquement la réglementation "
    "numérique de l'UE (RGPD, AI Act, DORA, NIS 2, DSA, Data Act). Posez-moi une question sur "
    "ces textes.",
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
    conversation: bool = False  # small talk: no citations, guarded instead of grounded
    standalone_question: str | None = None  # a follow-up as rewritten for retrieval

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

    async def answer(
        self, question: str, lang: str | None = None, history: list[Turn] | None = None
    ) -> Answer:
        with spans.tracer.start_as_current_span("answer") as root:
            spans.start_answer(root, question, lang)
            lang = lang or detect_lang(question)  # the user's language, before any rewrite
            query, rewrite = await self._standalone(question, history)
            hits, t_retrieval = await self._retrieve(query)
            if not hits:
                answer = self._refuse(
                    question, lang, "no_sources", hits, None, {"retrieval": t_retrieval}
                )
            else:
                messages = build_messages(query, hits, lang)
                with spans.tracer.start_as_current_span("llm.generate") as gen:
                    completion = await self._llm.complete(messages, LlmAnswer)
                    spans.generation(gen, self._llm.model_id, messages, completion)
                answer = self._finish(question, lang, hits, completion, t_retrieval)
            _with_rewrite(answer, query, rewrite)
            spans.finish_answer(root, answer)
            return answer

    async def stream(
        self, question: str, lang: str | None = None, history: list[Turn] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """`Sources` right after retrieval, `Delta`s while the model writes, then the `Answer`.

        The final Answer is authoritative: the grounding check (or the small-talk guard) runs on
        the complete text, so a streamed draft can still end as a refusal or be replaced
        (clients replace the draft with it).
        """
        # Spans are passed explicitly, never made "current" across a `yield`: the consumer runs
        # between yields and would otherwise inherit (and corrupt) this generator's context.
        root = spans.tracer.start_span("answer")
        root_ctx = trace.set_span_in_context(root)
        spans.start_answer(root, question, lang)
        try:
            lang = lang or detect_lang(question)
            token = otel_context.attach(root_ctx)
            try:
                query, rewrite = await self._standalone(question, history)
                hits, t_retrieval = await self._retrieve(query)
            finally:
                otel_context.detach(token)
            yield Sources(hits)
            if not hits:
                answer = self._refuse(
                    question, lang, "no_sources", hits, None, {"retrieval": t_retrieval}
                )
                _with_rewrite(answer, query, rewrite)
                spans.finish_answer(root, answer)
                yield answer
                return
            field = JsonFieldStream("answer")
            completion: Completion | None = None
            messages = build_messages(query, hits, lang)
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
            _with_rewrite(answer, query, rewrite)
            spans.finish_answer(root, answer)
            yield answer
        except BaseException as exc:  # includes a client disconnect (GeneratorExit)
            root.record_exception(exc)
            root.set_status(trace.Status(trace.StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            root.end()

    async def _standalone(
        self, question: str, history: list[Turn] | None
    ) -> tuple[str, Completion | None]:
        """The question to search: a follow-up is rewritten from the conversation first."""
        if not history:
            return question, None
        messages = build_rewrite_messages(history, question)
        with spans.tracer.start_as_current_span("llm.rewrite") as span:
            completion = await self._llm.complete(messages, Rewrite)
            spans.generation(span, self._llm.model_id, messages, completion)
        return parse_rewrite(completion, question), completion

    async def _retrieve(self, query: str) -> tuple[list[Hit], float]:
        start = time.perf_counter()
        with spans.tracer.start_as_current_span("retrieval") as span:
            hits = await self._retriever.search(query, k=self._k)
            if self._expand_chars:
                hits = await self._retriever.expand(hits, max_chars=self._expand_chars)
            spans.retrieval(span, hits, self._k, bool(self._expand_chars))
        return hits, round((time.perf_counter() - start) * 1000, 1)

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
        if out.kind == "conversation":
            return self._small_talk(question, lang, hits, out.answer, completion, timings)
        if out.kind == "out_of_scope":
            return self._refuse(question, lang, "out_of_scope", hits, completion, timings)
        if out.kind == "not_in_sources" or not out.answer.strip():
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

    def _small_talk(
        self,
        question: str,
        lang: str,
        hits: list[Hit],
        reply: str,
        completion: Completion,
        timings: dict[str, float],
    ) -> Answer:
        """A generated reply, unless it states something that would need a source."""
        text = reply.strip()
        if not is_safe_small_talk(text):
            log.warning("small_talk_replaced", model=completion.model, reply=text[:300])
            text = INTRODUCTION.get(lang, INTRODUCTION["en"])
        return Answer(
            question=question,
            lang=lang,
            text=text,
            refused=False,
            refusal_reason=None,
            citations=[],
            sources=hits,
            timings_ms=timings,
            conversation=True,
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
        messages = OUT_OF_SCOPE if reason == "out_of_scope" else REFUSALS
        return Answer(
            question=question,
            lang=lang,
            text=messages.get(lang, messages["en"]),
            refused=True,
            refusal_reason=reason,
            citations=[],
            sources=hits,
            timings_ms=timings,
            **(_usage(completion) if completion else {"model": self._llm.model_id}),
        )


def _with_rewrite(answer: Answer, query: str, rewrite: Completion | None) -> None:
    """Account for the follow-up rewrite: its question, latency, tokens and cost."""
    if rewrite is None:
        return
    answer.standalone_question = query
    t = answer.timings_ms
    t["rewrite"] = rewrite.latency_ms
    for stage in ("total", "first_token"):
        if stage in t:
            t[stage] = round(t[stage] + rewrite.latency_ms, 1)
    if "total" not in t:  # refused before generation: retrieval was the only other stage
        t["total"] = round(t.get("retrieval", 0.0) + rewrite.latency_ms, 1)
    answer.input_tokens += rewrite.input_tokens
    answer.output_tokens += rewrite.output_tokens
    answer.cost_usd += rewrite.cost_usd
    answer.cached = answer.cached and rewrite.cached


def _usage(c: Completion) -> dict[str, Any]:
    return {
        "model": c.model,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "cost_usd": c.cost_usd,
        "cached": c.cached,
    }
