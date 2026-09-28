from dataclasses import dataclass, field
from typing import Any

import pytest

from lexeu.ingestion.corpus import Corpus
from lexeu.ingestion.pipeline import SyncReport, sync
from tests.fakes import CORPUS, XHTML, FakeFetcher, FakeRegistry, FakeStore


@dataclass
class Harness:
    fetcher: FakeFetcher = field(default_factory=FakeFetcher)
    store: FakeStore = field(default_factory=FakeStore)
    registry: FakeRegistry = field(default_factory=FakeRegistry)

    async def sync(self, corpus: Corpus = CORPUS, **kw: Any) -> SyncReport:
        kw.setdefault("max_chars", 1800)
        return await sync(corpus, self.fetcher, self.store, self.registry, **kw)


@pytest.fixture
def h() -> Harness:
    return Harness()


async def test_first_sync_creates_every_document(h: Harness) -> None:
    report = await h.sync()

    assert report.counts == {"created": 2}
    assert report.ok
    assert set(h.registry.docs) == {("32016R0679", "en"), ("32016R0679", "fr")}
    assert len(h.store.objects) == 2
    meta, chunks = h.registry.docs[("32016R0679", "en")]
    assert meta.raw_object_key == f"raw/32016R0679/en/{meta.content_sha256}.xhtml"
    assert meta.pipeline_version == "parser=2;max_chars=1800"
    assert len(chunks) == 6


async def test_second_sync_is_a_no_op(h: Harness) -> None:
    await h.sync()
    report = await h.sync()
    assert report.counts == {"unchanged": 2}
    assert {r.n_chunks for r in report.results} == {6}


async def test_changed_content_is_reprocessed_and_archived(h: Harness) -> None:
    await h.sync()
    h.fetcher.content = XHTML.replace(b"Member States may", b"Member States shall")

    report = await h.sync()

    assert report.counts == {"updated": 2}
    assert len(h.store.objects) == 4  # old versions are kept (content-addressed)
    _, chunks = h.registry.docs[("32016R0679", "en")]
    assert any("Member States shall" in c.text for c in chunks)


async def test_pipeline_change_triggers_reprocessing(h: Harness) -> None:
    await h.sync()
    report = await h.sync(max_chars=180)
    assert report.counts == {"updated": 2}


async def test_force_reprocesses_unchanged_documents(h: Harness) -> None:
    await h.sync()
    assert (await h.sync(force=True)).counts == {"updated": 2}


async def test_acts_removed_from_corpus_are_deleted(h: Harness) -> None:
    await h.sync()
    report = await h.sync(corpus=CORPUS.model_copy(update={"languages": ["en"]}))
    assert report.counts == {"unchanged": 1, "deleted": 1}
    assert set(h.registry.docs) == {("32016R0679", "en")}


async def test_partial_sync_never_deletes(h: Harness) -> None:
    await h.sync()
    await h.sync(corpus=CORPUS.model_copy(update={"languages": ["en"]}), only={"32016R0679"})
    assert len(h.registry.docs) == 2


async def test_one_failure_does_not_stop_the_others(h: Harness) -> None:
    h.fetcher.fail = {("32016R0679", "fr")}

    report = await h.sync()

    assert report.counts == {"created": 1, "failed": 1}
    assert not report.ok
    failed = next(r for r in report.results if r.action == "failed")
    assert failed.error == "FetchError: boom"
    assert set(h.registry.docs) == {("32016R0679", "en")}
