"""Question -> retrieval -> grounded, cited answer (or an explicit refusal).

    question -> language -> top-k provisions -> prompt -> LLM (JSON) -> grounding check
             -> answer with citations mapped back to provisions | refusal

Refusal has three causes, all reported: nothing retrieved, the model says the sources do not
answer, or the answer cites no valid source (treated as ungrounded, never shown).
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import structlog
from pydantic import ValidationError

from lexeu.generation.grounding import check_grounding
from lexeu.generation.llm import Completion, LlmClient
from lexeu.generation.prompt import PROMPT_VERSION, LlmAnswer, build_messages
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


class Answerer:
    def __init__(
        self, retriever: Retriever, llm: LlmClient, k: int = 8, expand_chars: int = 0
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k
        self._expand_chars = expand_chars  # 0 = give the model the retrieved parts only

    async def answer(self, question: str, lang: str | None = None) -> Answer:
        start = time.perf_counter()
        lang = lang or detect_lang(question)
        hits = await self._retriever.search(question, k=self._k)
        if self._expand_chars:
            hits = await self._retriever.expand(hits, max_chars=self._expand_chars)
        t_retrieval = (time.perf_counter() - start) * 1000

        if not hits:
            return self._refuse(
                question, lang, "no_sources", hits, None, {"retrieval": t_retrieval}
            )

        completion = await self._llm.complete(build_messages(question, hits, lang), LlmAnswer)
        timings = {
            "retrieval": round(t_retrieval, 1),
            "generation": completion.latency_ms,
            "total": round(t_retrieval + completion.latency_ms, 1),
        }

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
