"""FastAPI application factory.

Run with: uvicorn lexeu.api.main:create_app --factory
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from lexeu import __version__
from lexeu.api.middleware import RequestContextMiddleware
from lexeu.api.routes import ask, health
from lexeu.core.config import Settings, get_settings
from lexeu.core.logging import configure_logging
from lexeu.core.tls import use_system_trust
from lexeu.generation.answer import Answerer
from lexeu.generation.factory import MissingApiKeyError, make_llm
from lexeu.infra.answer_log import AnswerLog
from lexeu.infra.api_keys import ApiKeyStore
from lexeu.infra.rate_limit import RateLimiter
from lexeu.infra.resources import Resources
from lexeu.retrieval.embeddings import TeiEmbedder
from lexeu.retrieval.search import Retriever
from lexeu.retrieval.sparse import detect_lang

log = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    use_system_trust()  # outgoing HTTPS (LLM provider) trusts the OS store, as the CLI does

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = Resources.create(settings)
        app.state.resources = resources
        app.state.probes = resources.probes()
        embedder = TeiEmbedder(settings.embeddings.url, settings.embeddings.model_id)
        app.state.answerer = _answerer(settings, resources, embedder)
        app.state.api_keys = ApiKeyStore(resources.db)
        app.state.answer_log = AnswerLog(resources.db)
        app.state.rate_limiter = RateLimiter(resources.redis)
        log.info("startup", env=settings.env, version=__version__)
        try:
            yield
        finally:
            await embedder.aclose()
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
    app.include_router(ask.router)
    return app


def _answerer(settings: Settings, resources: Resources, embedder: TeiEmbedder) -> Answerer | None:
    """Without an LLM key the API still serves health checks; /v1/ask answers 503."""
    try:
        llm, _ = make_llm(settings)
    except MissingApiKeyError as exc:
        log.warning("generation_disabled", reason=str(exc))
        return None
    retriever = Retriever(resources.qdrant, embedder, settings.retrieval)
    detect_lang("warm-up")  # loads the language models now, not on the first user's request
    return Answerer(
        retriever, llm, k=settings.generation.k, expand_chars=settings.generation.expand_chars
    )
