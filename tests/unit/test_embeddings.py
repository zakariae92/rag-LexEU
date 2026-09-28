import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from lexeu.retrieval.embeddings import CachedEmbedder, EmbeddingCache, TeiEmbedder, Vectors


class CountingEmbedder:
    def __init__(self, model_id: str = "test/model") -> None:
        self._model_id = model_id
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts: list[str]) -> Vectors:
        self.calls.append(texts)
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


async def test_only_cache_misses_reach_the_model(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, EmbeddingCache(tmp_path / "e.sqlite"))

    first = await cached.embed(["a", "bb", "a"])
    second = await cached.embed(["bb", "ccc"])

    assert inner.calls == [["a", "bb"], ["ccc"]]  # duplicates and hits are never recomputed
    assert first.tolist() == [[1, 1], [2, 1], [1, 1]]
    assert second.tolist() == [[2, 1], [3, 1]]
    assert (cached.hits, cached.misses) == (1, 3)  # 1 served from cache, 3 unique texts computed


async def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "e.sqlite"
    await CachedEmbedder(CountingEmbedder(), EmbeddingCache(path)).embed(["hello"])

    inner = CountingEmbedder()
    await CachedEmbedder(inner, EmbeddingCache(path)).embed(["hello"])
    assert inner.calls == []


async def test_cache_is_keyed_by_model(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    await CachedEmbedder(CountingEmbedder("model-a"), cache).embed(["hello"])

    other = CountingEmbedder("model-b")
    await CachedEmbedder(other, cache).embed(["hello"])
    assert other.calls == [["hello"]]  # same text, other model: must be recomputed


async def test_tei_client_batches_and_normalises() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json=[[0.6, 0.8]] * len(body["inputs"]))

    tei = TeiEmbedder(
        "http://tei",
        "BAAI/bge-m3",
        batch_size=2,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    vectors = await tei.embed(["a", "b", "c"])

    assert [b["inputs"] for b in bodies] == [["a", "b"], ["c"]]
    assert all(b["normalize"] is True for b in bodies)
    assert vectors.shape == (3, 2)
    assert vectors.dtype == np.float32


def test_min_cosine_detects_the_worst_row() -> None:
    from lexeu.retrieval.embeddings import min_cosine

    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert min_cosine(a, a * 3) == pytest.approx(1.0)  # scale-invariant
    b = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    assert min_cosine(a, b) == pytest.approx(2**-0.5)
