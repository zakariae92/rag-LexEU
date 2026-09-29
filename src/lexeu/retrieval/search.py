"""Configurable retrieval over the live index.

question -> [language detection] -> candidates (dense | dense+BM25 fused in Qdrant)
         -> recital policy -> one slot per provision -> [cross-encoder rerank] -> top k
"""

from dataclasses import dataclass, replace
from typing import Protocol

from qdrant_client import AsyncQdrantClient, models

from lexeu.retrieval.config import RetrievalConfig
from lexeu.retrieval.embeddings import Embedder
from lexeu.retrieval.index import DENSE, SPARSE
from lexeu.retrieval.sparse import bm25_query, detect_lang


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    provision_key: str
    lang: str
    citation: str
    score: float
    text: str
    kind: str = "article"
    header: str = ""


class Reranker(Protocol):
    @property
    def model_id(self) -> str: ...

    async def score(self, query: str, passages: list[str]) -> list[float]: ...


class Retriever:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
        alias: str = "chunks",
    ) -> None:
        self.config = config or RetrievalConfig()
        if self.config.rerank and reranker is None:
            raise ValueError(f"config '{self.config.name}' needs a reranker ({self.config.rerank})")
        self._qdrant = qdrant
        self._embedder = embedder
        self._reranker = reranker
        self._alias = alias

    async def search(self, query: str, k: int = 10, lang: str | None = None) -> list[Hit]:
        """Top-k *provisions*: EN and FR versions (or split parts) of one provision share a slot.

        `lang` forces a language filter; otherwise the config decides.
        """
        cfg = self.config
        needs_lang = cfg.mode != "dense" or cfg.lang == "query"
        query_lang = detect_lang(query) if needs_lang else None
        flt = self._filter(lang or (query_lang if cfg.lang == "query" else None))

        if cfg.mode == "sparse":
            result = await self._qdrant.query_points(
                self._alias, query=bm25_query(query, query_lang or "en"), using=SPARSE,
                query_filter=flt, limit=cfg.candidates, with_payload=True,
            )  # fmt: skip
        elif cfg.mode == "dense":
            vector = (await self._embedder.embed([query]))[0].tolist()
            result = await self._qdrant.query_points(
                self._alias, query=vector, using=DENSE, query_filter=flt,
                limit=cfg.candidates, with_payload=True,
            )  # fmt: skip
        else:
            vector = (await self._embedder.embed([query]))[0].tolist()
            fusion: models.Query
            if cfg.fusion == "dbsf":
                fusion = models.FusionQuery(fusion=models.Fusion.DBSF)
            else:  # weighted RRF: dense and BM25 rankings do not have to count equally
                fusion = models.RrfQuery(rrf=models.Rrf(weights=[cfg.dense_weight, 1.0]))
            result = await self._qdrant.query_points(
                self._alias,
                prefetch=[
                    models.Prefetch(query=vector, using=DENSE, filter=flt, limit=cfg.candidates),
                    models.Prefetch(
                        query=bm25_query(query, query_lang or "en"),
                        using=SPARSE,
                        filter=flt,
                        limit=cfg.candidates,
                    ),
                ],
                query=fusion,
                limit=cfg.candidates,
                with_payload=True,
            )

        hits = [_to_hit(p) for p in result.points]
        if cfg.recitals == "demote":
            hits = [
                replace(h, score=h.score * cfg.recital_penalty) if h.kind == "recital" else h
                for h in hits
            ]
        hits = _one_per_provision(sorted(hits, key=lambda h: h.score, reverse=True))

        if self._reranker is not None and cfg.rerank:
            head = hits[: cfg.rerank_top]
            scores = await self._reranker.score(query, [f"{h.header}\n\n{h.text}" for h in head])
            reranked = [replace(h, score=s) for h, s in zip(head, scores, strict=True)]
            if cfg.recitals == "demote":  # the policy still applies to cross-encoder scores
                reranked = [
                    replace(h, score=h.score * cfg.recital_penalty) if h.kind == "recital" else h
                    for h in reranked
                ]
            hits = sorted(reranked, key=lambda h: h.score, reverse=True)
        return hits[:k]

    def _filter(self, lang: str | None) -> models.Filter | None:
        must: list[models.Condition] = []
        must_not: list[models.Condition] = []
        if lang:
            must.append(models.FieldCondition(key="lang", match=models.MatchValue(value=lang)))
        if self.config.recitals == "exclude":
            must_not.append(
                models.FieldCondition(key="kind", match=models.MatchValue(value="recital"))
            )
        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)


# Backwards-compatible name for the plain dense configuration.
DenseRetriever = Retriever


def _to_hit(p: models.ScoredPoint) -> Hit:
    payload = p.payload or {}
    return Hit(
        chunk_id=str(payload["chunk_id"]),
        provision_key=str(payload["provision_key"]),
        lang=str(payload["lang"]),
        citation=str(payload["citation"]),
        score=float(p.score),
        text=str(payload["text"]),
        kind=str(payload.get("kind", "article")),
        header=str(payload.get("header", "")),
    )


def _one_per_provision(hits: list[Hit]) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        if h.provision_key not in seen:
            seen.add(h.provision_key)
            out.append(h)
    return out
