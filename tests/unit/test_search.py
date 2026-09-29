from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qdrant_client import models

from lexeu.retrieval.config import RetrievalConfig
from lexeu.retrieval.embeddings import Vectors
from lexeu.retrieval.search import Retriever


def _point(key: str, score: float, kind: str = "article", lang: str = "en") -> models.ScoredPoint:
    return models.ScoredPoint(
        id=abs(hash(key + lang)) % 10**9,
        version=0,
        score=score,
        payload={
            "chunk_id": f"{key}:{lang}", "provision_key": key, "lang": lang, "kind": kind,
            "citation": key, "text": f"text of {key}", "header": "",
        },
    )  # fmt: skip


class FakeQdrant:
    def __init__(self, points: list[models.ScoredPoint]) -> None:
        self.points = points
        self.calls: list[dict[str, Any]] = []

    async def query_points(self, collection: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points)  # the retriever only reads `.points`


class FakeEmbedder:
    model_id = "fake"

    async def embed(self, texts: list[str]) -> Vectors:
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeReranker:
    model_id = "fake-ce"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    async def score(self, query: str, passages: list[str]) -> list[float]:
        return [self.scores[p.split("text of ")[1]] for p in passages]


POINTS = [
    _point("gdpr:rct_85:u1", 0.80, kind="recital"),
    _point("gdpr:art_33:p1", 0.78),
    _point("gdpr:art_33:p1", 0.77, lang="fr"),  # FR copy of the same provision
    _point("gdpr:art_34:p1", 0.70),
]


def _retriever(cfg: RetrievalConfig, reranker: FakeReranker | None = None) -> Retriever:
    return Retriever(FakeQdrant(POINTS), FakeEmbedder(), cfg, reranker=reranker)  # type: ignore[arg-type]


async def test_keep_ranks_by_similarity_and_collapses_languages() -> None:
    hits = await _retriever(RetrievalConfig()).search("breach notification deadline")
    assert [h.provision_key for h in hits] == ["gdpr:rct_85:u1", "gdpr:art_33:p1", "gdpr:art_34:p1"]


async def test_demote_lets_the_article_win_a_close_call() -> None:
    cfg = RetrievalConfig(recitals="demote", recital_penalty=0.9)
    hits = await _retriever(cfg).search("breach notification deadline")
    assert hits[0].provision_key == "gdpr:art_33:p1"
    assert hits[1].score == pytest.approx(0.72)  # 0.80 x 0.9


async def test_exclude_and_language_become_qdrant_filters() -> None:
    qdrant = FakeQdrant(POINTS)
    cfg = RetrievalConfig(recitals="exclude", lang="query")
    await Retriever(qdrant, FakeEmbedder(), cfg).search("Quel est le délai de notification ?")  # type: ignore[arg-type]

    flt = qdrant.calls[0]["query_filter"]
    assert flt.must[0].match.value == "fr"  # detected from the question
    assert flt.must_not[0].match.value == "recital"


async def test_rerank_reorders_and_keeps_the_recital_policy() -> None:
    cfg = RetrievalConfig(recitals="demote", recital_penalty=0.5, rerank="fake-ce", rerank_top=3)
    reranker = FakeReranker(
        {"gdpr:rct_85:u1": 0.99, "gdpr:art_33:p1": 0.90, "gdpr:art_34:p1": 0.20}
    )
    hits = await _retriever(cfg, reranker).search("breach notification deadline")
    assert [h.provision_key for h in hits] == ["gdpr:art_33:p1", "gdpr:rct_85:u1", "gdpr:art_34:p1"]
    assert hits[0].score == pytest.approx(0.90)


def test_rerank_config_requires_a_reranker() -> None:
    with pytest.raises(ValueError, match="needs a reranker"):
        _retriever(RetrievalConfig(rerank="some/model"))
