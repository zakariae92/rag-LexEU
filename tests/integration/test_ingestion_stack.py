"""End-to-end ingestion against real Postgres and S3 (network to Cellar is faked)."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine

from lexeu.core.config import Settings
from lexeu.infra.db import SqlRegistry
from lexeu.infra.storage import S3RawStore
from lexeu.ingestion.pipeline import sync
from tests.fakes import CORPUS, XHTML, FakeFetcher

pytestmark = pytest.mark.integration


async def test_migrations_create_the_schema(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
    assert {"documents", "chunks", "ingestion_runs", "alembic_version"} <= tables


def test_migrations_are_reversible(migrated: Settings, alembic_cfg: Config) -> None:
    command.downgrade(alembic_cfg, "base")
    command.upgrade(alembic_cfg, "head")


async def test_full_sync_lifecycle(registry: SqlRegistry, store: S3RawStore) -> None:
    fetcher = FakeFetcher()

    # 1. first sync: everything is created, raw bytes are archived
    report = await sync(CORPUS, fetcher, store, registry, max_chars=1800)
    assert report.counts == {"created": 2}

    chunks = await registry.get_chunks("32016R0679", "en")
    assert [c.citation for c in chunks] == [
        "Recital 1 GDPR",
        "Recital 2 GDPR",
        "Art. 4 GDPR",
        "Art. 6(1) GDPR",
        "Art. 6(2) GDPR",
        "Annex I GDPR",
    ]
    assert [c.ordinal for c in chunks] == list(range(6))

    (doc,) = [d for d in await registry.list_documents() if d.lang == "en"]
    key = f"raw/32016R0679/en/{doc.content_sha256}.xhtml"
    assert await store.get(key) == XHTML

    # 2. nothing changed: no-op
    assert (await sync(CORPUS, fetcher, store, registry, max_chars=1800)).counts == {"unchanged": 2}

    # 3. new chunking config: chunks are replaced atomically, old ones are gone
    report = await sync(CORPUS, fetcher, store, registry, max_chars=180)
    assert report.counts == {"updated": 2}
    art4 = await registry.get_chunks("32016R0679", "en", "art_4")
    assert [c.part for c in art4] == [1, 2, 3]
    art6 = await registry.get_chunks("32016R0679", "en", "art_6")
    assert [(c.paragraph, c.part) for c in art6] == [(1, 1), (1, 2), (2, 1)]
    assert len(await registry.get_chunks("32016R0679", "en")) == 9  # 6 - 2 + 3 + 2

    # 4. act dropped from the corpus: document and chunks are deleted (FK cascade)
    report = await sync(
        CORPUS.model_copy(update={"acts": []}), fetcher, store, registry, max_chars=180
    )
    assert report.counts == {"deleted": 2}
    assert await registry.get_chunks("32016R0679", "en") == []


async def test_ingestion_runs_are_audited(registry: SqlRegistry) -> None:
    run_id = await registry.start_run()
    await registry.finish_run(run_id, "succeeded", {"counts": {"created": 2}})
    run_id_2 = await registry.start_run()
    assert run_id_2 == run_id + 1
