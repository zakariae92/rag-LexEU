"""FastAPI application factory.

Run with: uvicorn lexeu.api.main:create_app --factory
"""

import asyncio
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
from lexeu.generation.prompt import PROMPT_VERSION
from lexeu.infra.answer_cache import AnswerCache, CachingAnswerer, namespace
from lexeu.infra.answer_log import AnswerLog
from lexeu.infra.api_keys import ApiKeyStore
from lexeu.infra.db import SqlIndexRegistry
from lexeu.infra.rate_limit import RateLimiter
from lexeu.infra.resources import Resources
from lexeu.observability.tracing import setup_tracing
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
        app.state.answerer = await _answerer(settings, resources, embedder)
        app.state.api_keys = ApiKeyStore(resources.db)
        app.state.answer_log = AnswerLog(resources.db)
        app.state.rate_limiter = RateLimiter(resources.redis)
        log.info("startup", env=settings.env, version=__version__)
        try:
            yield
        finally:
            await embedder.aclose()
            await resources.close()
            if app.state.tracer_provider is not None:
                app.state.tracer_provider.shutdown()  # flush the last spans
            log.info("shutdown")

    app = FastAPI(
        title="rag-LexEU",
        version=__version__,
        description="Bilingual RAG assistant over EU regulations.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.tracer_provider = setup_tracing(app, settings)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(ask.router)
    return app


async def _answerer(
    settings: Settings, resources: Resources, embedder: TeiEmbedder
) -> Answerer | CachingAnswerer | None:
    """Without an LLM key the API still serves health checks; /v1/ask answers 503."""
    try:
        llm, _ = make_llm(settings)
    except MissingApiKeyError as exc:
        log.warning("generation_disabled", reason=str(exc))
        return None
    retriever = Retriever(resources.qdrant, embedder, settings.retrieval)
    detect_lang("warm-up")  # loads the language models now, not on the first user's request
    gen = settings.generation
    answerer = Answerer(retriever, llm, k=gen.k, expand_chars=gen.expand_chars)
    if not settings.cache.enabled:
        return answerer
    try:  # the cache namespace pins the live index: a rebuilt index never serves old answers
        index = await asyncio.wait_for(SqlIndexRegistry(resources.db).active("chunks"), 5)
    except Exception as exc:
        log.warning("answer_cache_disabled", reason=repr(exc))
        return answerer
    ns = namespace(
        index.fingerprint if index else None, gen.model, PROMPT_VERSION, settings.retrieval,
        gen.k, gen.expand_chars, gen.reasoning_effort,
    )  # fmt: skip
    log.info("answer_cache_enabled", namespace=ns)
    return CachingAnswerer(answerer, AnswerCache(resources.redis, ns, settings.cache.ttl_s))
