"""`POST /v1/ask`: a grounded answer with citations, or an explicit refusal. `POST /v1/feedback`."""

import uuid
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from lexeu.api.deps import Caller, caller
from lexeu.generation.answer import Answer, Answerer
from lexeu.infra.answer_log import AnswerLog
from lexeu.infra.api_keys import ApiKey

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
    answer_id: str  # send it back with /v1/feedback
    answer: str
    lang: str
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    usage: Usage
    timings_ms: dict[str, float]
    prompt_version: str

    @classmethod
    def from_answer(cls, answer_id: str, a: Answer) -> "AskResponse":
        return cls(
            answer_id=answer_id,
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
async def ask(
    body: AskRequest,
    request: Request,
    background: BackgroundTasks,
    who: Annotated[Caller, Depends(caller)],
) -> AskResponse:
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
    answer_id = uuid.uuid4().hex
    # Logged after the response is sent: the client never waits for the database.
    background.add_task(_record, request.app.state.answer_log, answer_id, answer, request, who.key)
    log.info(
        "answer",
        answer_id=answer_id,
        lang=answer.lang,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        citations=len(answer.citations),
        model=answer.model,
        cost_usd=answer.cost_usd,
        **{f"{k}_ms": v for k, v in answer.timings_ms.items()},
    )
    return AskResponse.from_answer(answer_id, answer)


async def _record(
    answer_log: AnswerLog, answer_id: str, answer: Answer, request: Request, key: ApiKey | None
) -> None:
    try:
        await answer_log.record(
            answer_id,
            answer,
            retrieval=request.app.state.settings.retrieval.name,
            api_key_id=key.id if key else None,
        )
    except Exception as exc:  # losing a log line must never break answering
        log.error("answer_log_failed", answer_id=answer_id, error=repr(exc))


class FeedbackRequest(BaseModel):
    answer_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    rating: Literal[1, -1] = Field(description="1 = helpful, -1 = not helpful")
    comment: str | None = Field(default=None, max_length=2000)


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(
    body: FeedbackRequest, request: Request, _: Annotated[Caller, Depends(caller)]
) -> Response:
    found = await request.app.state.answer_log.feedback(body.answer_id, body.rating, body.comment)
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown answer_id")
    log.info("feedback", answer_id=body.answer_id, rating=body.rating)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
