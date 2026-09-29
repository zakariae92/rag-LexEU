"""`POST /v1/ask`: a grounded answer with citations, or an explicit refusal."""

from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from lexeu.generation.answer import Answer, Answerer

router = APIRouter(prefix="/v1", tags=["answers"])
log = structlog.get_logger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    lang: Literal["en", "fr"] | None = Field(
        default=None, description="Answer language; detected from the question when omitted."
    )


class CitationOut(BaseModel):
    n: int
    citation: str
    provision_key: str
    lang: str


class Usage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class AskResponse(BaseModel):
    answer: str
    lang: str
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    usage: Usage
    timings_ms: dict[str, float]
    prompt_version: str

    @classmethod
    def from_answer(cls, a: Answer) -> "AskResponse":
        return cls(
            answer=a.text,
            lang=a.lang,
            refused=a.refused,
            refusal_reason=a.refusal_reason,
            citations=[
                CitationOut(n=c.n, citation=c.citation, provision_key=c.provision_key, lang=c.lang)
                for c in a.citations
            ],
            usage=Usage(
                model=a.model,
                input_tokens=a.input_tokens,
                output_tokens=a.output_tokens,
                cost_usd=a.cost_usd,
            ),
            timings_ms=a.timings_ms,
            prompt_version=a.prompt_version,
        )


@router.post("/ask")
async def ask(body: AskRequest, request: Request) -> AskResponse:
    answerer: Answerer | None = getattr(request.app.state, "answerer", None)
    if answerer is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "answer generation is not configured"
        )
    try:
        answer = await answerer.answer(body.question, lang=body.lang)
    except Exception as exc:  # provider timeout, quota, outage: retryable, not a server bug
        log.error("answer_failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "answer generation is temporarily unavailable, please retry",
            headers={"Retry-After": "30"},
        ) from exc
    log.info(
        "answer",
        lang=answer.lang,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        citations=len(answer.citations),
        model=answer.model,
        cost_usd=answer.cost_usd,
        **{f"{k}_ms": v for k, v in answer.timings_ms.items()},
    )
    return AskResponse.from_answer(answer)
