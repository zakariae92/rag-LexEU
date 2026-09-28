"""Text embeddings: a TEI client, and a persistent cache so nothing is embedded twice.

The cache is keyed by sha256(model_id + text): the same text embedded with another model is
a miss, which is what guarantees queries and documents live in the same vector space.
It is a single SQLite file per model, so a batch job on a GPU machine can produce it and the
indexer on any other machine can reuse it without recomputing anything.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import numpy.typing as npt
import structlog

log = structlog.get_logger(__name__)

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    async def embed(self, texts: list[str]) -> Vectors: ...


class TeiEmbedder:
    """Client for a HuggingFace text-embeddings-inference server (returns L2-normalised vectors)."""

    def __init__(
        self,
        url: str,
        model_id: str,
        batch_size: int = 8,
        timeout_s: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._model_id = model_id
        self._batch_size = batch_size
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts: list[str]) -> Vectors:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            resp = await self._client.post(
                f"{self._url}/embed",
                json={
                    "inputs": texts[i : i + self._batch_size],
                    "normalize": True,
                    "truncate": True,
                },
            )
            resp.raise_for_status()
            out.extend(resp.json())
        return np.asarray(out, dtype=np.float32)

    async def served_model(self) -> str:
        """The model TEI actually loaded: checked against `model_id` before indexing."""
        resp = await self._client.get(f"{self._url}/info")
        resp.raise_for_status()
        return str(resp.json()["model_id"])

    async def aclose(self) -> None:
        await self._client.aclose()


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")

    @staticmethod
    def key(model_id: str, text: str) -> str:
        return hashlib.sha256(f"{model_id}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, Vectors]:
        found: dict[str, Vectors] = {}
        for i in range(0, len(keys), 500):  # stay under SQLite's variable limit
            batch = keys[i : i + 500]
            marks = ",".join("?" * len(batch))
            rows = self._db.execute(f"SELECT key, vec FROM emb WHERE key IN ({marks})", batch)  # noqa: S608
            for key, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: dict[str, Vectors]) -> None:
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
                [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
            )

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM emb").fetchone()[0])

    def close(self) -> None:
        self._db.close()


class CachedEmbedder:
    """Wraps an embedder: cache hits are free, only misses reach the model."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache, chunk_size: int = 64) -> None:
        self._inner = inner
        self._cache = cache
        self._chunk_size = chunk_size  # persist progress regularly on long runs
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    async def embed(self, texts: list[str]) -> Vectors:
        keys = [EmbeddingCache.key(self.model_id, t) for t in texts]
        found = self._cache.get_many(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(k for k in keys if k not in found))
        text_of = dict(zip(keys, texts, strict=True))
        self.hits += len(keys) - sum(1 for k in keys if k not in found)
        self.misses += len(missing)

        for i in range(0, len(missing), self._chunk_size):
            batch = missing[i : i + self._chunk_size]
            vectors = await self._inner.embed([text_of[k] for k in batch])
            new = dict(zip(batch, vectors, strict=True))
            self._cache.put_many(new)
            found.update(new)
            if len(missing) > self._chunk_size:
                log.info(
                    "embedding_progress", done=min(i + len(batch), len(missing)), total=len(missing)
                )

        return np.stack([found[k] for k in keys]) if keys else np.empty((0, 0), np.float32)


def min_cosine(a: Vectors, b: Vectors) -> float:
    """Lowest row-wise cosine similarity between two embeddings of the same texts."""
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float(np.min(np.sum(a_n * b_n, axis=1)))
