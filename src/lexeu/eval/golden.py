"""The golden set: questions with the provisions that answer them, and their human review status.

Questions live in `golden_vN.yaml`; review decisions live next to it in `reviews.yaml`, so that
reviewing never rewrites the question file (and its comments).
"""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from lexeu.generation.conversation import Turn
from lexeu.ingestion.corpus import Lang

Category = Literal[
    "lookup",
    "definition",
    "obligation",
    "temporal",
    "cross_regulation",
    "unanswerable",
    "false_premise",
    "follow_up",  # asked after earlier turns (`history`): only meaningful in context
    "conversation",  # small talk: expects a reply without citations, not a refusal
    "out_of_scope",  # not about the covered regulations: expects the out-of-scope refusal
]
NO_LEGAL_ANSWER = {"unanswerable", "conversation", "out_of_scope"}
ReviewStatus = Literal["draft", "verified", "rejected"]

_KEY_PATTERN = r"^3\d{4}[RLD]\d{4}:(art|rct|anx)_\w+(:(p\d+|u\d+))?$"


class GoldenItem(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9]+-\d{3}$")
    lang: Lang
    category: Category
    question: str = Field(min_length=2)  # "Hi!" is a valid small-talk item
    expected: list[str] = Field(default_factory=list)
    answerable: bool = True
    reference: str
    notes: str | None = None
    history: list[Turn] = Field(default_factory=list)  # earlier turns, for follow-ups

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        for key in self.expected:
            if not _matches_pattern(key):
                raise ValueError(f"{self.id}: malformed provision key {key!r}")
        if self.category in NO_LEGAL_ANSWER and self.answerable:
            raise ValueError(f"{self.id}: category {self.category!r} requires answerable: false")
        if (self.category == "follow_up") != bool(self.history):
            raise ValueError(f"{self.id}: follow_up items, and only they, carry a history")
        if self.category != "conversation" and len(self.question) < 10:
            raise ValueError(f"{self.id}: question too short")
        if self.answerable and not self.expected:
            raise ValueError(f"{self.id}: answerable items need at least one expected provision")
        if not self.answerable and self.expected:
            raise ValueError(f"{self.id}: unanswerable items must not list expected provisions")
        return self

    @property
    def has_retrieval_target(self) -> bool:
        return bool(self.expected)


class GoldenSet(BaseModel):
    version: int
    items: list[GoldenItem]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [i.id for i in self.items]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate ids: {sorted(dupes)}")
        return self


class Review(BaseModel):
    status: ReviewStatus
    reviewer: str
    date: date
    comment: str | None = None


def load_golden(path: Path) -> GoldenSet:
    return GoldenSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def reviews_path(golden_path: Path) -> Path:
    return golden_path.with_name("reviews.yaml")


def load_reviews(golden_path: Path) -> dict[str, Review]:
    path = reviews_path(golden_path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k: Review.model_validate(v) for k, v in raw.items()}


def save_reviews(golden_path: Path, reviews: dict[str, Review]) -> None:
    data = {k: v.model_dump(mode="json", exclude_none=True) for k, v in sorted(reviews.items())}
    reviews_path(golden_path).write_text(
        "# Human review decisions for the golden set (written by `lexeu eval review`).\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def status_of(item: GoldenItem, reviews: dict[str, Review]) -> ReviewStatus:
    review = reviews.get(item.id)
    return review.status if review else "draft"


def _matches_pattern(key: str) -> bool:
    import re

    return re.match(_KEY_PATTERN, key) is not None
