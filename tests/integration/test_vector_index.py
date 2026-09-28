"""Blue/green vector index on real Qdrant + Postgres, with a deterministic fake embedder."""

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from lexeu.core.config import Settings
from lexeu.infra.db import SqlIndexRegistry, SqlRegistry
from lexeu.infra.storage import S3RawStore
from lexeu.ingestion.pipeline import sync
from lexeu.retrieval.embeddings import Vectors
from lexeu.retrieval.index import IndexModelMismatchError, build_index, ensure_same_model
from lexeu.retrieval.search import DenseRetriever
from tests.fakes import CORPUS, FakeFetcher

pytestmark = pytest.mark.integration
ALIAS = "it_chunks"


class BagOfWordsEmbedder:
    """Deterministic 64-d hashed bag of words: similar wording -> similar vectors."""

    def __init__(self, model_id: str = "test/bow-64") -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts: list[str]) -> Vectors:
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0  # noqa: S324
        norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return (out / norms).astype(np.float32)


@pytest.fixture
async def qdrant(migrated: Settings) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=migrated.qdrant.url)
    for c in (await client.get_collections()).collections:
        if c.name.startswith(f"{ALIAS}_"):
            await client.delete_collection(c.name)
    yield client
    await client.close()


async def _collections(qdrant: AsyncQdrantClient) -> set[str]:
    return {
        c.name
        for c in (await qdrant.get_collections()).collections
        if c.name.startswith(f"{ALIAS}_")
    }


async def test_blue_green_lifecycle(
    registry: SqlRegistry,
    store: S3RawStore,
    engine: AsyncEngine,
    qdrant: AsyncQdrantClient,
) -> None:
    fetcher = FakeFetcher()
    await sync(CORPUS, fetcher, store, registry, max_chars=1800)
    index_registry = SqlIndexRegistry(engine)
    embedder = BagOfWordsEmbedder()

    # 1. first build: a new collection goes live behind the alias
    first = await build_index(index_registry, qdrant, embedder, alias=ALIAS)
    assert first.action == "built"
    assert first.n_points == 12  # 6 chunks x 2 languages
    assert (await index_registry.active(ALIAS)).collection == first.collection  # type: ignore[union-attr]

    # 2. same corpus, same model: nothing to do
    assert (await build_index(index_registry, qdrant, embedder, alias=ALIAS)).action == "unchanged"

    # 3. the corpus changes: a new generation is built, the previous one is kept for rollback
    fetcher.content = fetcher.content.replace(b"Member States may", b"Member States shall")
    await sync(CORPUS, fetcher, store, registry, max_chars=1800)
    second = await build_index(index_registry, qdrant, embedder, alias=ALIAS)
    assert second.action == "built" and second.collection != first.collection
    assert await _collections(qdrant) == {first.collection, second.collection}

    # 4. rollback: content goes back, the old collection is re-activated without re-embedding
    fetcher.content = fetcher.content.replace(b"Member States shall", b"Member States may")
    await sync(CORPUS, fetcher, store, registry, max_chars=1800)
    rollback = await build_index(index_registry, qdrant, embedder, alias=ALIAS)
    assert (rollback.action, rollback.collection) == ("reactivated", first.collection)


async def test_retrieval_returns_distinct_provisions(
    registry: SqlRegistry,
    store: S3RawStore,
    engine: AsyncEngine,
    qdrant: AsyncQdrantClient,
) -> None:
    await sync(CORPUS, FakeFetcher(), store, registry, max_chars=1800)
    embedder = BagOfWordsEmbedder()
    await build_index(SqlIndexRegistry(engine), qdrant, embedder, alias=ALIAS)

    hits = await DenseRetriever(qdrant, embedder, alias=ALIAS).search(
        "processing shall be lawful only if the data subject has given consent", k=3
    )

    assert hits[0].provision_key == "32016R0679:art_6:p1"
    keys = [h.provision_key for h in hits]
    assert len(keys) == len(set(keys))  # EN and FR copies are collapsed into one slot


async def test_query_model_must_match_index_model(
    registry: SqlRegistry,
    store: S3RawStore,
    engine: AsyncEngine,
    qdrant: AsyncQdrantClient,
) -> None:
    await sync(CORPUS, FakeFetcher(), store, registry, max_chars=1800)
    index_registry = SqlIndexRegistry(engine)
    await build_index(index_registry, qdrant, BagOfWordsEmbedder("model-a"), alias=ALIAS)

    assert (await ensure_same_model(index_registry, "model-a", alias=ALIAS)).model_id == "model-a"
    with pytest.raises(IndexModelMismatchError, match="built with 'model-a'"):
        await ensure_same_model(index_registry, "model-b", alias=ALIAS)
