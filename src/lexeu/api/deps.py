"""Request dependencies: who is calling (API key), and may they call now (rate limit)."""

from dataclasses import dataclass

import structlog
from fastapi import HTTPException, Request, Response, status

from lexeu.core.config import Settings
from lexeu.infra.api_keys import ApiKey
from lexeu.observability.metrics import RATE_LIMITED

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Caller:
    key: ApiKey | None  # None: anonymous (only when auth is not required)
    subject: str  # what the rate limit counts: "key:<id>" or "ip:<address>"
    limit_per_min: int


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


async def caller(request: Request, response: Response) -> Caller:
    settings: Settings = request.app.state.settings
    token = _bearer(request)
    key = await request.app.state.api_keys.authenticate(token) if token else None

    if key is not None:
        who = Caller(key, f"key:{key.id}", key.rate_limit_per_min)
    elif settings.auth.required or token:  # a wrong key is an error even when auth is optional
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
        ip = request.client.host if request.client else "unknown"
        who = Caller(None, f"ip:{ip}", settings.auth.anonymous_rate_limit_per_min)

    decision = await request.app.state.rate_limiter.hit(who.subject, who.limit_per_min)
    headers = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
    }
    if not decision.allowed:
        log.warning("rate_limited", subject=who.subject, limit=decision.limit)
        RATE_LIMITED.inc()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit exceeded",
            headers={**headers, "Retry-After": str(decision.retry_after_s)},
        )
    response.headers.update(headers)
    request.state.rate_limit_headers = headers  # for streaming responses, built by hand
    structlog.contextvars.bind_contextvars(caller=key.name if key else who.subject)
    return who
