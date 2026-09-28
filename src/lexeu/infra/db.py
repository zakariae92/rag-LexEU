"""Postgres schema (SQLAlchemy 2 typed ORM) and the document registry.

- `documents`: one row per (act, language) with its lineage: content hash, raw object key,
  parser version and chunking config. This is what makes incremental sync possible.
- `chunks`: the processed zone, the input of indexing and evaluation.
- `ingestion_runs`: an audit trail of every sync (when, what changed, errors).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from lexeu.ingestion.models import Chunk


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("celex", "lang"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    celex: Mapped[str] = mapped_column(String(16))
    lang: Mapped[str] = mapped_column(String(2))
    short_name: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    manifestation_url: Mapped[str] = mapped_column(Text)
    etag: Mapped[str | None] = mapped_column(Text)
    content_sha256: Mapped[str] = mapped_column(String(64))
    raw_object_key: Mapped[str] = mapped_column(Text)
    pipeline_version: Mapped[str] = mapped_column(String(64))
    n_chunks: Mapped[int] = mapped_column(Integer)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    provision_key: Mapped[str] = mapped_column(Text, index=True)
    celex: Mapped[str] = mapped_column(String(16))
    lang: Mapped[str] = mapped_column(String(2))
    kind: Mapped[str] = mapped_column(String(16))
    eli_id: Mapped[str] = mapped_column(String(32))
    paragraph: Mapped[int | None] = mapped_column(Integer)
    part: Mapped[int] = mapped_column(Integer)
    ordinal: Mapped[int] = mapped_column(Integer)  # reading order within the document
    citation: Mapped[str] = mapped_column(Text)
    header: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)


class IngestionRunRow(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16))  # running | succeeded | failed
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class SearchIndexRow(Base):
    """Every vector index ever built: which model, which chunks, and which one is live."""

    __tablename__ = "search_indexes"

    id: Mapped[int] = mapped_column(primary_key=True)
    collection: Mapped[str] = mapped_column(String(64), unique=True)
    alias: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(Text)
    dim: Mapped[int] = mapped_column(Integer)
    fingerprint: Mapped[str] = mapped_column(String(64))
    n_points: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


@dataclass(frozen=True)
class DocumentRecord:
    """What the pipeline needs to know about an already-ingested document."""

    celex: str
    lang: str
    content_sha256: str
    pipeline_version: str
    n_chunks: int


@dataclass(frozen=True)
class DocumentMeta:
    celex: str
    lang: str
    short_name: str
    title: str
    source_url: str
    manifestation_url: str
    etag: str | None
    content_sha256: str
    raw_object_key: str
    pipeline_version: str


class SqlRegistry:
    def __init__(self, engine: AsyncEngine) -> None:
        self._session = async_sessionmaker(engine, expire_on_commit=False)

    async def get_document(self, celex: str, lang: str) -> DocumentRecord | None:
        async with self._session() as s:
            row = await s.scalar(
                select(DocumentRow).where(DocumentRow.celex == celex, DocumentRow.lang == lang)
            )
            return _record(row) if row else None

    async def list_documents(self) -> list[DocumentRecord]:
        async with self._session() as s:
            rows = await s.scalars(
                select(DocumentRow).order_by(DocumentRow.celex, DocumentRow.lang)
            )
            return [_record(r) for r in rows]

    async def save_document(self, meta: DocumentMeta, chunks: list[Chunk]) -> None:
        """Upsert the document and replace all its chunks atomically."""
        async with self._session.begin() as s:
            row = await s.scalar(
                select(DocumentRow).where(
                    DocumentRow.celex == meta.celex, DocumentRow.lang == meta.lang
                )
            )
            if row is None:
                row = DocumentRow(celex=meta.celex, lang=meta.lang)
                s.add(row)
            for field, value in vars(meta).items():
                setattr(row, field, value)
            row.n_chunks = len(chunks)
            row.ingested_at = datetime.now(UTC)
            await s.flush()  # assigns row.id for new documents

            await s.execute(delete(ChunkRow).where(ChunkRow.document_id == row.id))
            s.add_all(
                ChunkRow(
                    chunk_id=c.chunk_id,
                    document_id=row.id,
                    provision_key=c.provision_key,
                    celex=c.celex,
                    lang=c.lang,
                    kind=c.kind,
                    eli_id=c.eli_id,
                    paragraph=c.paragraph,
                    part=c.part,
                    ordinal=i,
                    citation=c.citation,
                    header=c.header,
                    text=c.text,
                )
                for i, c in enumerate(chunks)
            )

    async def delete_document(self, celex: str, lang: str) -> None:
        async with self._session.begin() as s:
            await s.execute(
                delete(DocumentRow).where(DocumentRow.celex == celex, DocumentRow.lang == lang)
            )

    async def start_run(self) -> int:
        async with self._session.begin() as s:
            run = IngestionRunRow(status="running")
            s.add(run)
            await s.flush()
            return run.id

    async def finish_run(self, run_id: int, status: str, stats: dict[str, Any]) -> None:
        async with self._session.begin() as s:
            run = await s.get_one(IngestionRunRow, run_id)
            run.status = status
            run.stats = stats
            run.finished_at = datetime.now(UTC)

    async def get_chunks(self, celex: str, lang: str, eli_id: str | None = None) -> list[ChunkRow]:
        async with self._session() as s:
            query = select(ChunkRow).where(ChunkRow.celex == celex, ChunkRow.lang == lang)
            if eli_id:
                query = query.where(ChunkRow.eli_id == eli_id)
            return list(await s.scalars(query.order_by(ChunkRow.ordinal)))


def _record(row: DocumentRow) -> DocumentRecord:
    return DocumentRecord(
        celex=row.celex,
        lang=row.lang,
        content_sha256=row.content_sha256,
        pipeline_version=row.pipeline_version,
        n_chunks=row.n_chunks,
    )


@dataclass(frozen=True)
class IndexRecord:
    collection: str
    alias: str
    model_id: str
    dim: int
    fingerprint: str
    n_points: int


class SqlIndexRegistry:
    """Lineage of vector indexes: which collection is live behind an alias, with which model."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._session = async_sessionmaker(engine, expire_on_commit=False)

    async def all_chunks(self) -> list[ChunkRow]:
        async with self._session() as s:
            query = select(ChunkRow).order_by(ChunkRow.celex, ChunkRow.lang, ChunkRow.ordinal)
            return list(await s.scalars(query))

    async def activate(self, record: IndexRecord) -> None:
        """Record a newly built index and mark it as the live one for its alias."""
        async with self._session.begin() as s:
            await s.execute(
                update(SearchIndexRow)
                .where(SearchIndexRow.alias == record.alias)
                .values(active=False)
            )
            row = await s.scalar(
                select(SearchIndexRow).where(SearchIndexRow.collection == record.collection)
            )
            if row is None:
                row = SearchIndexRow(**vars(record))
                s.add(row)
            row.active = True

    async def active(self, alias: str) -> IndexRecord | None:
        async with self._session() as s:
            row = await s.scalar(
                select(SearchIndexRow).where(
                    SearchIndexRow.alias == alias, SearchIndexRow.active.is_(True)
                )
            )
            if row is None:
                return None
            return IndexRecord(
                row.collection, row.alias, row.model_id, row.dim, row.fingerprint, row.n_points
            )
