"""Answers: `POST /v1/ask` (JSON), `POST /v1/ask/stream` (server-sent events), `POST /v1/feedback`.

Both answer routes return a grounded answer with citations, or an explicit refusal.
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lexeu.api.deps import Caller, caller
from lexeu.generation.answer import Answer, Answerer, Delta, Sources
from lexeu.infra.answer_cache import CachingAnswerer
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
    answerer = _answerer(request)
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
    _log_answer(answer_id, answer)
    return AskResponse.from_answer(answer_id, answer)


@router.post(
    "/ask/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-sent events"}},
)
async def ask_stream(
    body: AskRequest, request: Request, who: Annotated[Caller, Depends(caller)]
) -> StreamingResponse:
    """Events: `sources` (after retrieval), `delta` (text as it is written), then `done` with
    the same body as `/v1/ask`. `done` is authoritative: if the final grounding check fails, it
    carries a refusal and the client replaces the streamed draft. `error` if the provider fails.
    """
    answerer = _answerer(request)
    answer_id = uuid.uuid4().hex

    async def events() -> AsyncIterator[str]:
        final: Answer | None = None
        try:
            async for event in answerer.stream(body.question, lang=body.lang):
                if isinstance(event, Sources):
                    sources = [
                        {"n": i, "citation": h.citation, "provision_key": h.provision_key}
                        for i, h in enumerate(event.hits, start=1)
                    ]
                    yield _sse("sources", {"answer_id": answer_id, "sources": sources})
                elif isinstance(event, Delta):
                    yield _sse("delta", {"text": event.text})
                else:
                    final = event
                    yield _sse("done", AskResponse.from_answer(answer_id, event).model_dump())
        except Exception as exc:  # headers are sent already: report the failure in the stream
            log.error("answer_failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
            yield _sse("error", {"detail": "answer generation is temporarily unavailable"})
            return
        if final is not None:
            _log_answer(answer_id, final)
            await _record(request.app.state.answer_log, answer_id, final, request, who.key)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # proxies (nginx) must not buffer the stream
            **request.state.rate_limit_headers,
        },
    )


def _answerer(request: Request) -> Answerer | CachingAnswerer:
    answerer: Answerer | CachingAnswerer | None = getattr(request.app.state, "answerer", None)
    if answerer is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "answer generation is not configured"
        )
    return answerer


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _log_answer(answer_id: str, answer: Answer) -> None:
    log.info(
        "answer",
        answer_id=answer_id,
        lang=answer.lang,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        citations=len(answer.citations),
        model=answer.model,
        cost_usd=answer.cost_usd,
        cache_hit=answer.cached,
        **{f"{k}_ms": v for k, v in answer.timings_ms.items()},
    )


async def _record(
    answer_log: AnswerLog, answer_id: str, answer: Answer, request: Request, key: ApiKey | None
) -> None:
    try:
        await answer_log.record(
            answer_id,
            answer,
            retrieval=request.app.state.settings.retrieval.name,
            api_key_id=key.id if key else None,
            cache_hit=answer.cached,
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
