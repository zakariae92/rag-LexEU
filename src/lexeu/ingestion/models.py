"""Domain objects produced by the parser and the chunker."""

from dataclasses import dataclass, field
from typing import Literal

from lexeu.ingestion.corpus import Lang

ProvisionKind = Literal["recital", "article", "annex"]


@dataclass(frozen=True)
class Block:
    """A top-level piece of text inside a provision (a sentence, or one lettered point).

    `label` is the point label when the block is a list item, e.g. "(a)", "(1)", "2.".
    """

    text: str
    label: str | None = None


@dataclass(frozen=True)
class Unit:
    """The smallest citable part of a provision: a numbered paragraph, or the whole body."""

    blocks: list[Block]
    paragraph: int | None = None  # 1 for "006.001"; None when the article is not numbered


@dataclass(frozen=True)
class Provision:
    kind: ProvisionKind
    eli_id: str  # "art_6", "rct_12", "anx_III": identical across languages
    number: str  # "6", "12", "III"
    title: str | None  # "Lawfulness of processing"
    heading_path: list[str]  # ["CHAPTER II: Principles"]
    units: list[Unit]


@dataclass(frozen=True)
class ParsedAct:
    celex: str
    lang: Lang
    title: str
    provisions: list[Provision] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "32016R0679:en:art_6:p1:1"; stable, used for upserts and eval labels
    provision_key: str  # "32016R0679:art_6:p1"; language-neutral, aligns EN and FR
    celex: str
    lang: Lang
    kind: ProvisionKind
    eli_id: str
    paragraph: int | None
    part: int  # 1-based index when a long unit is split
    citation: str  # "Art. 6(1) GDPR"
    header: str  # breadcrumb prepended to the text before embedding
    text: str

    @property
    def embedding_text(self) -> str:
        return f"{self.header}\n\n{self.text}"
