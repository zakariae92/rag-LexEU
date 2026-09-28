"""Incremental sync: corpus definition -> raw store -> parsed chunks -> registry.

For every (act, language):
    fetch -> sha256 -> unchanged? skip
                    -> otherwise: archive raw bytes, parse, chunk, replace chunks atomically
Acts removed from the corpus are deleted from the registry (full syncs only).

A document is re-processed when its bytes change OR when the pipeline itself changes
(parser version, chunking parameters): `pipeline_version` captures both.
"""

import asyncio
import hashlib
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

import structlog

from lexeu.infra.db import DocumentMeta, DocumentRecord
from lexeu.ingestion.chunker import chunk_act
from lexeu.ingestion.corpus import ActSpec, Corpus, Lang
from lexeu.ingestion.fetcher import FetchedDoc
from lexeu.ingestion.models import Chunk
from lexeu.ingestion.parser import PARSER_VERSION, parse_act

log = structlog.get_logger(__name__)

Action = Literal["created", "updated", "unchanged", "deleted", "failed"]


class Fetcher(Protocol):
    async def fetch(self, celex: str, lang: Lang) -> FetchedDoc: ...


class RawStore(Protocol):
    async def put(self, key: str, data: bytes, content_type: str) -> None: ...
    async def exists(self, key: str) -> bool: ...


class Registry(Protocol):
    async def get_document(self, celex: str, lang: str) -> DocumentRecord | None: ...
    async def list_documents(self) -> list[DocumentRecord]: ...
    async def save_document(self, meta: DocumentMeta, chunks: list[Chunk]) -> None: ...
    async def delete_document(self, celex: str, lang: str) -> None: ...


@dataclass(frozen=True)
class DocResult:
    celex: str
    lang: str
    action: Action
    n_chunks: int = 0
    duration_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class SyncReport:
    results: list[DocResult]

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(r.action for r in self.results))

    @property
    def ok(self) -> bool:
        return all(r.action != "failed" for r in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {"counts": self.counts, "results": [asdict(r) for r in self.results]}


def pipeline_version(max_chars: int) -> str:
    return f"parser={PARSER_VERSION};max_chars={max_chars}"


async def sync(
    corpus: Corpus,
    fetcher: Fetcher,
    store: RawStore,
    registry: Registry,
    *,
    max_chars: int,
    max_concurrency: int = 3,
    only: set[str] | None = None,
    force: bool = False,
) -> SyncReport:
    """Bring the registry in line with the corpus. `only` restricts to some CELEX ids."""
    version = pipeline_version(max_chars)
    targets = [
        (act, lang)
        for act in corpus.acts
        if only is None or act.celex in only
        for lang in corpus.languages
    ]
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(act: ActSpec, lang: Lang) -> DocResult:
        async with semaphore:
            return await _sync_document(
                act, lang, fetcher, store, registry, version, max_chars, force
            )

    results = list(await asyncio.gather(*(one(act, lang) for act, lang in targets)))

    if only is None:  # full sync: drop what is no longer in the corpus
        wanted = {(act.celex, lang) for act, lang in targets}
        for doc in await registry.list_documents():
            if (doc.celex, doc.lang) not in wanted:
                await registry.delete_document(doc.celex, doc.lang)
                log.info("document_deleted", celex=doc.celex, lang=doc.lang)
                results.append(DocResult(doc.celex, doc.lang, "deleted"))

    return SyncReport(results)


async def _sync_document(
    act: ActSpec,
    lang: Lang,
    fetcher: Fetcher,
    store: RawStore,
    registry: Registry,
    version: str,
    max_chars: int,
    force: bool,
) -> DocResult:
    start = time.perf_counter()

    def ms() -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    bound = log.bind(celex=act.celex, lang=lang)

    try:
        fetched = await fetcher.fetch(act.celex, lang)
        sha = hashlib.sha256(fetched.content).hexdigest()
        existing = await registry.get_document(act.celex, lang)

        if (
            existing
            and not force
            and existing.content_sha256 == sha
            and existing.pipeline_version == version
        ):
            bound.info("document_unchanged")
            return DocResult(act.celex, lang, "unchanged", existing.n_chunks, ms())

        key = f"raw/{act.celex}/{lang}/{sha}.xhtml"
        if not await store.exists(key):
            await store.put(key, fetched.content, "application/xhtml+xml")

        parsed = parse_act(fetched.content, act.celex, lang)
        chunks = chunk_act(parsed, act, max_chars)
        meta = DocumentMeta(
            celex=act.celex,
            lang=lang,
            short_name=act.short_name[lang],
            title=parsed.title,
            source_url=fetched.source_url,
            manifestation_url=fetched.manifestation_url,
            etag=fetched.etag,
            content_sha256=sha,
            raw_object_key=key,
            pipeline_version=version,
        )
        await registry.save_document(meta, chunks)

        action: Action = "updated" if existing else "created"
        bound.info(f"document_{action}", chunks=len(chunks), duration_ms=ms())
        return DocResult(act.celex, lang, action, len(chunks), ms())

    except Exception as exc:
        bound.exception("document_failed")
        return DocResult(act.celex, lang, "failed", 0, ms(), f"{type(exc).__name__}: {exc}")
