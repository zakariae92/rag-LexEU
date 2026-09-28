"""Integration fixtures: real Postgres + object store from the compose stack.

Tests run against a dedicated `lexeu_test` database and `lexeu-test` bucket, so they never
touch development data. The schema is built with the real Alembic migrations.
"""

from collections.abc import AsyncIterator, Iterator

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from lexeu.core.config import Settings, get_settings
from lexeu.infra.db import SqlRegistry
from lexeu.infra.storage import S3RawStore

TEST_DB = "lexeu_test"
TEST_BUCKET = "lexeu-test"


@pytest.fixture(scope="session")
def it_settings() -> Iterator[Settings]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("POSTGRES__DB", TEST_DB)
        mp.setenv("OBJECT_STORE__BUCKET_RAW", TEST_BUCKET)
        get_settings.cache_clear()  # alembic's env.py reads settings too
        yield get_settings()
    get_settings.cache_clear()


async def _create_database(settings: Settings) -> None:
    pg = settings.postgres
    conn = await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password.get_secret_value(),
        database="postgres",
    )
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB):
            await conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def alembic_cfg() -> Config:
    return Config("alembic.ini")


@pytest.fixture(scope="session")
def migrated(it_settings: Settings, alembic_cfg: Config) -> Settings:
    import asyncio

    asyncio.run(_create_database(it_settings))
    command.upgrade(alembic_cfg, "head")
    return it_settings


@pytest.fixture
async def engine(migrated: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated.postgres.dsn)
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE documents, chunks, ingestion_runs RESTART IDENTITY CASCADE")
        )
    yield engine
    await engine.dispose()


@pytest.fixture
def registry(engine: AsyncEngine) -> SqlRegistry:
    return SqlRegistry(engine)


@pytest.fixture
async def store(migrated: Settings) -> S3RawStore:
    s = S3RawStore(migrated.object_store)
    await s.ensure_bucket()
    return s
