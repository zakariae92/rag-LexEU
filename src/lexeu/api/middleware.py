"""Request context: a request id on every log line and response, plus one access log."""

import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lexeu.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS

REQUEST_ID_HEADER = "x-request-id"

log = structlog.get_logger("lexeu.access")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        incoming = headers.get(REQUEST_ID_HEADER.encode())
        request_id = incoming.decode() if incoming else uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status = 500
        start = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode())
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            duration = time.perf_counter() - start
            # Route template ("/v1/ask"), never the raw path: bounded label cardinality.
            route = getattr(scope.get("route"), "path", "unmatched")
            HTTP_REQUESTS.labels(method=scope["method"], route=route, status=str(status)).inc()
            HTTP_LATENCY.labels(route=route).observe(duration)
            log.info(
                "request",
                method=scope["method"],
                path=scope["path"],
                status=status,
                duration_ms=round(duration * 1000, 1),
            )
            structlog.contextvars.unbind_contextvars("request_id", "caller")
