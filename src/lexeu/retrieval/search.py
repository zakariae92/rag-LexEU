"""Dense retrieval over the live index (the M2 baseline; hybrid + rerank come in M3)."""

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from lexeu.retrieval.embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    provision_key: str
    lang: str
    citation: str
    score: float
    text: str


class DenseRetriever:
    def __init__(
        self, qdrant: AsyncQdrantClient, embedder: Embedder, alias: str = "chunks"
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._alias = alias

    async def search(self, query: str, k: int = 10, lang: str | None = None) -> list[Hit]:
        """Top-k *provisions*: EN and FR versions (or split parts) of the same provision
        would otherwise fill several slots with the same answer."""
        vector = (await self._embedder.embed([query]))[0]
        flt = (
            models.Filter(
                must=[models.FieldCondition(key="lang", match=models.MatchValue(value=lang))]
            )
            if lang
            else None
        )
        result = await self._qdrant.query_points(
            self._alias, query=vector.tolist(), query_filter=flt, limit=k * 4, with_payload=True
        )
        hits: list[Hit] = []
        seen: set[str] = set()
        for p in result.points:
            payload = p.payload or {}
            key = str(payload["provision_key"])
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                Hit(
                    chunk_id=str(payload["chunk_id"]),
                    provision_key=key,
                    lang=str(payload["lang"]),
                    citation=str(payload["citation"]),
                    score=float(p.score),
                    text=str(payload["text"]),
                )
            )
            if len(hits) == k:
                break
        return hits
