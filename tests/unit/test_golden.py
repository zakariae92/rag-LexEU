from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from lexeu.eval.golden import (
    GoldenItem,
    Review,
    load_golden,
    load_reviews,
    save_reviews,
    status_of,
)

GOLDEN = Path(__file__).parents[2] / "eval" / "golden" / "golden_v1.yaml"


def _item(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "gdpr-999",
        "lang": "en",
        "category": "lookup",
        "question": "What is the maximum GDPR fine?",
        "expected": ["32016R0679:art_83:p5"],
        "reference": "EUR 20 million or 4 %.",
    }
    return base | overrides


def test_the_committed_golden_set_is_valid() -> None:
    gs = load_golden(GOLDEN)
    assert len(gs.items) >= 150
    assert {i.lang for i in gs.items} == {"en", "fr"}
    assert all(i.expected for i in gs.items if i.answerable)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"expected": ["GDPR art 83"]}, "malformed provision key"),
        ({"expected": []}, "need at least one expected provision"),
        ({"answerable": False}, "must not list expected provisions"),
        ({"category": "unanswerable", "expected": []}, "requires answerable: false"),
        ({"id": "GDPR_1"}, "String should match pattern"),
    ],
)
def test_item_validation(overrides: dict[str, object], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        GoldenItem.model_validate(_item(**overrides))


def test_article_level_keys_are_allowed() -> None:
    assert GoldenItem.model_validate(_item(expected=["32016R0679:art_4"])).expected


def test_reviews_round_trip(tmp_path: Path) -> None:
    golden = tmp_path / "golden_v1.yaml"
    item = GoldenItem.model_validate(_item())
    assert status_of(item, load_reviews(golden)) == "draft"

    save_reviews(golden, {item.id: Review(status="verified", reviewer="z", date=date(2026, 9, 29))})

    reviews = load_reviews(golden)
    assert status_of(item, reviews) == "verified"
    assert (tmp_path / "reviews.yaml").read_text(encoding="utf-8").startswith("# Human review")
