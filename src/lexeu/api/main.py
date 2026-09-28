"""FastAPI application factory.

Run with: uvicorn lexeu.api.main:create_app --factory
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from lexeu import __version__
from lexeu.api.middleware import RequestContextMiddleware
from lexeu.api.routes import health
from lexeu.core.config import Settings, get_settings
from lexeu.core.logging import configure_logging
from lexeu.infra.resources import Resources

log = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = Resources.create(settings)
        app.state.resources = resources
        app.state.probes = resources.probes()
        log.info("startup", env=settings.env, version=__version__)
        try:
            yield
        finally:
            await resources.close()
            log.info("shutdown")

    app = FastAPI(
        title="rag-LexEU",
        version=__version__,
        description="Bilingual RAG assistant over EU regulations.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    return app
