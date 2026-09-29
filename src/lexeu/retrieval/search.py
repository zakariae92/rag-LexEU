"""Configurable retrieval over the live index.

question -> [language detection] -> candidates (dense | dense+BM25 fused in Qdrant)
         -> recital policy -> one slot per provision -> [cross-encoder rerank] -> top k
         -> [expand: every part of each provision, for generation]
"""

from dataclasses import dataclass, replace
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient, models

from lexeu.observability.spans import tracer
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
            result = await self._query_points(
                self._alias, query=bm25_query(query, query_lang or "en"), using=SPARSE,
                query_filter=flt, limit=cfg.candidates, with_payload=True,
            )  # fmt: skip
        elif cfg.mode == "dense":
            vector = await self._embed(query)
            result = await self._query_points(
                self._alias, query=vector, using=DENSE, query_filter=flt,
                limit=cfg.candidates, with_payload=True,
            )  # fmt: skip
        else:
            vector = await self._embed(query)
            fusion: models.Query
            if cfg.fusion == "dbsf":
                fusion = models.FusionQuery(fusion=models.Fusion.DBSF)
            else:  # weighted RRF: dense and BM25 rankings do not have to count equally
                fusion = models.RrfQuery(rrf=models.Rrf(weights=[cfg.dense_weight, 1.0]))
            result = await self._query_points(
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

    async def _embed(self, query: str) -> list[float]:
        with tracer.start_as_current_span("embed.query") as span:
            span.set_attribute("gen_ai.request.model", self._embedder.model_id)
            vector: list[float] = (await self._embedder.embed([query]))[0].tolist()
            return vector

    async def _query_points(self, *args: Any, **kwargs: Any) -> models.QueryResponse:
        with tracer.start_as_current_span("qdrant.query_points") as span:
            span.set_attribute("db.system.name", "qdrant")
            span.set_attribute("lexeu.retrieval.candidates", int(kwargs.get("limit", 0)))
            return await self._qdrant.query_points(*args, **kwargs)

    async def expand(self, hits: list[Hit], max_chars: int = 6000) -> list[Hit]:
        """Small-to-big: search on chunks, but give the LLM the whole provision.

        Long provisions are split into parts (e.g. points (a)-(f) and (g)-(i) of a list); the
        search keeps one part per provision, which can hand the model half a list. Each hit is
        replaced by its provision's parts (same language), in order, as many as fit in
        `max_chars` around the part that was found. One Qdrant query for all hits.
        """
        if not hits:
            return hits
        keys = sorted({h.provision_key for h in hits})
        with tracer.start_as_current_span("qdrant.scroll"):
            points, _ = await self._qdrant.scroll(
                self._alias,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="provision_key", match=models.MatchAny(any=keys))
                    ]
                ),
                limit=2000,
                with_payload=True,
                with_vectors=False,
            )
        parts: dict[tuple[str, str], list[dict[str, object]]] = {}
        for pt in points:
            payload = pt.payload or {}
            parts.setdefault((str(payload["provision_key"]), str(payload["lang"])), []).append(
                payload
            )
        return [_merge(h, parts.get((h.provision_key, h.lang), []), max_chars) for h in hits]

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


def _merge(hit: Hit, siblings: list[dict[str, object]], max_chars: int) -> Hit:
    ordered = sorted(siblings, key=lambda p: int(str(p.get("part", 0))))
    if len(ordered) < 2:
        return hit
    ids = [str(p["chunk_id"]) for p in ordered]
    i = ids.index(hit.chunk_id) if hit.chunk_id in ids else 0
    lo, hi, size = i, i, len(str(ordered[i]["text"]))
    while True:  # grow a window around the part that was found, nearest parts first
        grown = False
        for j in (hi + 1, lo - 1):
            if 0 <= j < len(ordered) and size + len(str(ordered[j]["text"])) <= max_chars:
                size += len(str(ordered[j]["text"]))
                lo, hi = min(lo, j), max(hi, j)
                grown = True
        if not grown:
            break
    window = ordered[lo : hi + 1]
    texts = [str(p["text"]) for p in window]
    lead = texts[0].split("\n", 1)[0]  # parts repeat the paragraph's lead-in line
    body = texts[0] + "".join(
        "\n" + (t.split("\n", 1)[1] if t.startswith(lead + "\n") else t) for t in texts[1:]
    )
    return replace(hit, text=body, citation=_common_citation([str(p["citation"]) for p in window]))


def _common_citation(labels: list[str]) -> str:
    """ "Art. 30(2) DORA, points (a)-(f)" + "..., points (g)-(i)" -> "Art. 30(2) DORA"."""
    if len(set(labels)) == 1:
        return labels[0]
    prefix = labels[0]
    for label in labels[1:]:
        while not label.startswith(prefix):
            prefix = prefix[:-1]
    # Cut at the last comma: "Art. 30(2) DORA, points (" -> "Art. 30(2) DORA" (EN and FR labels
    # put the point range after a comma).
    head = prefix.rsplit(",", 1)[0] if "," in prefix else prefix
    return head.strip() or labels[0]


def _one_per_provision(hits: list[Hit]) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        if h.provision_key not in seen:
            seen.add(h.provision_key)
            out.append(h)
    return out
