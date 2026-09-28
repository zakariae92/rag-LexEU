from pathlib import Path

import pytest

from lexeu.eval.golden import GoldenItem
from lexeu.eval.retrieval import check_gate, evaluate_retrieval, to_markdown
from lexeu.retrieval.search import Hit

ITEMS = [
    GoldenItem(
        id="gdpr-001", lang="en", category="lookup", question="Breach deadline under GDPR?",
        expected=["32016R0679:art_33:p1"], reference="72 hours",
    ),
    GoldenItem(
        id="gdpr-002", lang="fr", category="definition", question="Définition des données ?",
        expected=["32016R0679:art_4"], reference="art. 4",
    ),
    GoldenItem(
        id="none-001", lang="en", category="unanswerable", question="Minimum wage in Germany?",
        answerable=False, reference="n/a",
    ),
]  # fmt: skip

RANKINGS = {
    "Breach deadline under GDPR?": (["32016R0679:art_33:p1", "32016R0679:art_34:p1"], 0.82),
    "Définition des données ?": (["32016R0679:art_5:p1", "32016R0679:art_4:u1"], 0.74),
    "Minimum wage in Germany?": (["32022R2554:art_2:p1"], 0.41),
}


class FakeRetriever:
    async def search(self, query: str, k: int = 10, lang: str | None = None) -> list[Hit]:
        keys, top = RANKINGS[query]
        return [
            Hit(f"{key}:1", key, "en", key.split(":", 1)[1], top - i * 0.1, "")
            for i, key in enumerate(keys[:k])
        ]


@pytest.fixture
async def report():  # type: ignore[no-untyped-def]
    return await evaluate_retrieval(ITEMS, FakeRetriever(), k=10)


async def test_overall_metrics_exclude_unanswerable(report) -> None:  # type: ignore[no-untyped-def]
    assert report.overall["n"] == 2
    assert report.overall["hit@1"] == 0.5
    assert report.overall["hit@3"] == 1.0
    assert report.overall["mrr@10"] == pytest.approx(0.75)


async def test_slices_and_score_separation(report) -> None:  # type: ignore[no-untyped-def]
    assert report.by_lang["fr"]["hit@1"] == 0.0
    assert report.by_category["lookup"]["hit@1"] == 1.0
    assert report.scores["top_score_answerable_median"] == pytest.approx(0.78)
    assert report.scores["top_score_unanswerable_median"] == pytest.approx(0.41)


async def test_markdown_lists_slices_and_misses(report) -> None:  # type: ignore[no-untyped-def]
    md = to_markdown(report)
    assert "| **overall** | 2 |" in md
    assert "category: definition" in md
    assert "Misses" not in md  # both answerable items are found within k=10


async def test_gate(report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    thresholds = tmp_path / "t.yaml"
    thresholds.write_text("overall:\n  hit@3: 0.9\nby_category:\n  definition:\n    hit@1: 0.5\n")
    assert check_gate(report, thresholds) == ["definition hit@1 = 0.0 < 0.5"]
