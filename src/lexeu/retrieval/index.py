"""Build the vector index in Qdrant with blue/green deployment behind an alias.

    chunks (Postgres) -> embeddings (cached) -> new collection `chunks_<fingerprint>`
                      -> atomic alias switch `chunks` -> previous collection kept for rollback

The fingerprint covers the embedding model and every chunk's id and text: rebuilding an
unchanged corpus with the same model is a no-op, and any change produces a new collection
that only goes live once fully written.
"""

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

import structlog
from qdrant_client import AsyncQdrantClient, models

from lexeu.infra.db import ChunkRow, IndexRecord
from lexeu.retrieval.embeddings import Embedder

log = structlog.get_logger(__name__)

KEYWORD_FIELDS = ("lang", "celex", "kind", "provision_key", "eli_id")
_POINT_NS = uuid.UUID("5b2f6d1e-3c1a-4e57-9d0b-0b6a6f6c9e11")


class IndexRegistry(Protocol):
    async def all_chunks(self) -> list[ChunkRow]: ...
    async def activate(self, record: IndexRecord) -> None: ...
    async def active(self, alias: str) -> IndexRecord | None: ...


class IndexModelMismatchError(RuntimeError):
    """Queries would be embedded with a different model than the documents."""


@dataclass(frozen=True)
class IndexReport:
    action: Literal["built", "unchanged", "reactivated"]
    collection: str
    n_points: int
    embed_hits: int
    embed_misses: int
    duration_s: float


def fingerprint(model_id: str, chunks: list[ChunkRow]) -> str:
    h = hashlib.sha256(model_id.encode())
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(f"\x00{c.chunk_id}\x00{c.header}\x00{c.text}".encode())
    return h.hexdigest()[:12]


def embedding_text(chunk: ChunkRow) -> str:
    return f"{chunk.header}\n\n{chunk.text}"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, chunk_id))


async def build_index(
    registry: IndexRegistry,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
    alias: str = "chunks",
    batch_size: int = 256,
) -> IndexReport:
    start = time.perf_counter()
    chunks = await registry.all_chunks()
    if not chunks:
        raise RuntimeError("No chunks in the registry: run `lexeu ingest` first")

    fp = fingerprint(embedder.model_id, chunks)
    collection = f"{alias}_{fp}"
    hits = misses = 0

    active = await registry.active(alias)
    if active and active.collection == collection and await qdrant.collection_exists(collection):
        log.info("index_unchanged", collection=collection)
        return IndexReport("unchanged", collection, active.n_points, 0, 0, _since(start))

    if await qdrant.collection_exists(collection):  # built earlier, e.g. rolled back from
        count = (await qdrant.count(collection, exact=True)).count
        dim = int(_vector_size(await qdrant.get_collection(collection)))
        action: Literal["built", "reactivated"] = "reactivated"
    else:
        vectors = await embedder.embed([embedding_text(c) for c in chunks])
        hits, misses = getattr(embedder, "hits", 0), getattr(embedder, "misses", 0)
        dim = int(vectors.shape[1])
        await qdrant.create_collection(
            collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field in KEYWORD_FIELDS:
            await qdrant.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD)
        for i in range(0, len(chunks), batch_size):
            await qdrant.upsert(
                collection,
                points=[
                    models.PointStruct(
                        id=point_id(c.chunk_id), vector=v.tolist(), payload=_payload(c)
                    )
                    for c, v in zip(
                        chunks[i : i + batch_size], vectors[i : i + batch_size], strict=True
                    )
                ],
                wait=True,
            )
        count = len(chunks)
        action = "built"

    await _switch_alias(qdrant, alias, collection)
    await registry.activate(IndexRecord(collection, alias, embedder.model_id, dim, fp, count))
    previous = {active.collection} if active else set()
    await _prune(qdrant, alias, keep={collection, *previous})  # previous = instant rollback
    log.info("index_activated", collection=collection, action=action, points=count)
    return IndexReport(action, collection, count, hits, misses, _since(start))


async def ensure_same_model(
    registry: IndexRegistry, model_id: str, alias: str = "chunks"
) -> IndexRecord:
    """Fail fast if the live index was built with another embedding model than the query side."""
    active = await registry.active(alias)
    if active is None:
        raise IndexModelMismatchError(
            f"No active index behind alias '{alias}': run `lexeu index build`"
        )
    if active.model_id != model_id:
        raise IndexModelMismatchError(
            f"Index '{active.collection}' was built with {active.model_id!r} "
            f"but queries would use {model_id!r}"
        )
    return active


def _payload(c: ChunkRow) -> dict[str, object]:
    return {
        "chunk_id": c.chunk_id,
        "provision_key": c.provision_key,
        "celex": c.celex,
        "lang": c.lang,
        "kind": c.kind,
        "eli_id": c.eli_id,
        "paragraph": c.paragraph,
        "part": c.part,
        "citation": c.citation,
        "header": c.header,
        "text": c.text,
    }


async def _switch_alias(qdrant: AsyncQdrantClient, alias: str, collection: str) -> None:
    ops: list[models.CreateAliasOperation | models.DeleteAliasOperation] = []
    existing = await qdrant.get_aliases()
    if any(a.alias_name == alias for a in existing.aliases):
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    await qdrant.update_collection_aliases(change_aliases_operations=ops)  # atomic


async def _prune(qdrant: AsyncQdrantClient, alias: str, keep: set[str]) -> None:
    """Delete older generations of this alias, keeping the live and the previous collection."""
    for c in (await qdrant.get_collections()).collections:
        if c.name.startswith(f"{alias}_") and c.name not in keep:
            await qdrant.delete_collection(c.name)
            log.info("index_pruned", collection=c.name)


def _vector_size(info: models.CollectionInfo) -> int:
    vectors = info.config.params.vectors
    assert isinstance(vectors, models.VectorParams)  # noqa: S101 - single unnamed vector by construction
    return vectors.size


def _since(start: float) -> float:
    return round(time.perf_counter() - start, 2)
