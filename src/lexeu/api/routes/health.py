"""Liveness and readiness endpoints (Kubernetes / load-balancer style).

- `/health`: the process is up. Never touches dependencies.
- `/ready`:  every backing service answers within the timeout; 503 otherwise.
- `/metrics`: Prometheus exposition format. Unauthenticated like the probes: in a deployment it is
  reachable from the monitoring network only, never through the public ingress.
"""

import asyncio
import time
from typing import Literal

import structlog
from fastapi import APIRouter, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from lexeu import __version__
from lexeu.infra.resources import Probe

router = APIRouter(tags=["health"])
log = structlog.get_logger(__name__)


class DependencyStatus(BaseModel):
    status: Literal["up", "down"]
    latency_ms: float
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dependencies: dict[str, DependencyStatus]


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


async def _run_probe(name: str, probe: Probe, timeout_s: float) -> DependencyStatus:
    start = time.perf_counter()
    try:
        await asyncio.wait_for(probe(), timeout=timeout_s)
    except Exception as exc:
        log.warning("dependency_down", dependency=name, error=repr(exc))
        return DependencyStatus(
            status="down",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            error=type(exc).__name__,  # don't leak hosts/credentials from messages
        )
    return DependencyStatus(status="up", latency_ms=round((time.perf_counter() - start) * 1000, 1))


@router.get("/ready", responses={503: {"model": ReadyResponse}})
async def ready(request: Request, response: Response) -> ReadyResponse:
    probes: dict[str, Probe] = request.app.state.probes
    timeout_s: float = request.app.state.settings.probe_timeout_s

    results = await asyncio.gather(
        *(_run_probe(name, probe, timeout_s) for name, probe in probes.items())
    )
    deps = dict(zip(probes, results, strict=True))
    all_up = all(d.status == "up" for d in deps.values())
    if not all_up:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(status="ready" if all_up else "not_ready", dependencies=deps)


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
