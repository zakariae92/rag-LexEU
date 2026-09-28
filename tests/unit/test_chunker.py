from pathlib import Path

import pytest

from lexeu.ingestion.chunker import chunk_act
from lexeu.ingestion.corpus import ActSpec
from lexeu.ingestion.models import Chunk, ParsedAct
from lexeu.ingestion.parser import parse_act

FIXTURE = Path(__file__).parents[1] / "fixtures" / "oj_act_en.xhtml"
GDPR = ActSpec(
    celex="32016R0679",
    official_name="Regulation (EU) 2016/679",
    short_name={"en": "GDPR", "fr": "RGPD"},
)


@pytest.fixture(scope="module")
def act_en() -> ParsedAct:
    return parse_act(FIXTURE.read_bytes(), "32016R0679", "en")


@pytest.fixture(scope="module")
def act_fr() -> ParsedAct:
    return parse_act(FIXTURE.read_bytes(), "32016R0679", "fr")


def _by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def test_one_chunk_per_citable_unit(act_en: ParsedAct) -> None:
    assert [c.chunk_id for c in chunk_act(act_en, GDPR)] == [
        "32016R0679:en:rct_1:u1:1",
        "32016R0679:en:rct_2:u1:1",
        "32016R0679:en:art_4:u1:1",
        "32016R0679:en:art_6:p1:1",
        "32016R0679:en:art_6:p2:1",
        "32016R0679:en:anx_I:u1:1",
    ]


def test_citations_and_headers_en(act_en: ParsedAct) -> None:
    chunks = _by_id(chunk_act(act_en, GDPR))
    c = chunks["32016R0679:en:art_6:p1:1"]
    assert c.citation == "Art. 6(1) GDPR"
    assert c.header == (
        "GDPR (Regulation (EU) 2016/679) > CHAPTER II: Principles > Section 1: Lawfulness"
        " > Article 6: Lawfulness of processing > Paragraph 1"
    )
    assert c.embedding_text.startswith(c.header + "\n\n1. Processing shall be lawful")
    assert chunks["32016R0679:en:rct_2:u1:1"].citation == "Recital 2 GDPR"
    assert chunks["32016R0679:en:anx_I:u1:1"].citation == "Annex I GDPR"
    assert chunks["32016R0679:en:art_4:u1:1"].citation == "Art. 4 GDPR"


def test_citations_and_headers_fr(act_fr: ParsedAct) -> None:
    c = _by_id(chunk_act(act_fr, GDPR))["32016R0679:fr:art_6:p1:1"]
    assert c.citation == "Art. 6, par. 1, RGPD"
    assert c.header.startswith("RGPD (Règlement (UE) 2016/679) >")
    assert c.header.endswith("Article 6 : Lawfulness of processing > Paragraphe 1")


def test_provision_key_is_language_neutral(act_en: ParsedAct, act_fr: ParsedAct) -> None:
    en = {c.provision_key for c in chunk_act(act_en, GDPR)}
    fr = {c.provision_key for c in chunk_act(act_fr, GDPR)}
    assert en == fr
    assert "32016R0679:art_6:p1" in en


def test_long_units_split_between_points_repeating_lead_in(act_en: ParsedAct) -> None:
    parts = [c for c in chunk_act(act_en, GDPR, max_chars=180) if c.eli_id == "art_4"]

    assert [c.part for c in parts] == [1, 2, 3]
    for c in parts:
        assert c.text.startswith("For the purposes of this Regulation:\n(")
    assert [c.citation for c in parts] == [
        "Art. 4 GDPR, point (1)",
        "Art. 4 GDPR, point (2)",
        "Art. 4 GDPR, point (3)",
    ]
    assert len({c.provision_key for c in parts}) == 1  # parts share their provision


def test_split_citation_uses_point_ranges(act_en: ParsedAct) -> None:
    parts = [c for c in chunk_act(act_en, GDPR, max_chars=260) if c.eli_id == "art_4"]
    assert [c.citation for c in parts] == [
        "Art. 4 GDPR, points (1)\u2013(2)",  # en dash, as in legal citations
        "Art. 4 GDPR, point (3)",
    ]


def test_chunk_ids_are_unique_and_stable(act_en: ParsedAct) -> None:
    first = [c.chunk_id for c in chunk_act(act_en, GDPR, max_chars=180)]
    second = [c.chunk_id for c in chunk_act(act_en, GDPR, max_chars=180)]
    assert first == second
    assert len(first) == len(set(first))
