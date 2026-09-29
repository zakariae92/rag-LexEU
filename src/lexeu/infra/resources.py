"""Clients for backing services, created once per process and shared.

Clients are lazy: nothing connects until first use, so the API starts even if
a dependency is down and `/ready` reports which one.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from lexeu.core.config import Settings

Probe = Callable[[], Awaitable[None]]
"""A readiness check: returns if healthy, raises otherwise."""


def make_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.postgres.dsn, pool_pre_ping=True)


def make_qdrant(settings: Settings) -> AsyncQdrantClient:
    api_key = settings.qdrant.api_key  # Qdrant Cloud; none for the local container
    return AsyncQdrantClient(
        url=settings.qdrant.url, api_key=api_key.get_secret_value() if api_key else None
    )


@dataclass
class Resources:
    settings: Settings
    db: AsyncEngine
    qdrant: AsyncQdrantClient
    redis: Redis
    http: httpx.AsyncClient

    @classmethod
    def create(cls, settings: Settings) -> "Resources":
        return cls(
            settings=settings,
            db=make_engine(settings),
            qdrant=make_qdrant(settings),
            redis=Redis.from_url(settings.redis.url),
            http=httpx.AsyncClient(timeout=settings.probe_timeout_s),
        )

    def probes(self) -> dict[str, Probe]:
        return {
            "postgres": self._check_postgres,
            "qdrant": self._check_qdrant,
            "redis": self._check_redis,
            "embeddings": self._check_embeddings,
        }

    async def _check_postgres(self) -> None:
        async with self.db.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def _check_qdrant(self) -> None:
        await self.qdrant.get_collections()

    async def _check_redis(self) -> None:
        await self.redis.ping()

    async def _check_embeddings(self) -> None:
        resp = await self.http.get(f"{self.settings.embeddings.url.rstrip('/')}/health")
        resp.raise_for_status()

    async def close(self) -> None:
        await self.db.dispose()
        await self.qdrant.close()
        await self.redis.aclose()
        await self.http.aclose()
