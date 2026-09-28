from pathlib import Path

import pytest

from lexeu.ingestion.models import ParsedAct, Provision
from lexeu.ingestion.parser import ParseError, parse_act

FIXTURE = Path(__file__).parents[1] / "fixtures" / "oj_act_en.xhtml"


@pytest.fixture(scope="module")
def act() -> ParsedAct:
    return parse_act(FIXTURE.read_bytes(), "32016R0679", "en")


def _get(act: ParsedAct, eli_id: str) -> Provision:
    return next(p for p in act.provisions if p.eli_id == eli_id)


def test_finds_every_provision_kind(act: ParsedAct) -> None:
    assert [(p.kind, p.eli_id) for p in act.provisions] == [
        ("recital", "rct_1"),
        ("recital", "rct_2"),
        ("article", "art_4"),
        ("article", "art_6"),
        ("annex", "anx_I"),
    ]
    assert act.title.startswith("REGULATION (EU) 2016/679")
    assert "of 27 April 2016" in act.title


def test_recital_text_without_its_number(act: ParsedAct) -> None:
    recital = _get(act, "rct_1")
    assert recital.number == "1"
    assert recital.units[0].blocks[0].text.startswith("The protection of natural persons")


def test_article_heading_path_includes_chapter_and_section(act: ParsedAct) -> None:
    art6 = _get(act, "art_6")
    assert art6.title == "Lawfulness of processing"
    assert art6.heading_path == ["CHAPTER II: Principles", "Section 1: Lawfulness"]
    assert _get(act, "art_4").heading_path == ["CHAPTER I: General provisions"]


def test_numbered_paragraphs_become_units(act: ParsedAct) -> None:
    units = _get(act, "art_6").units
    assert [u.paragraph for u in units] == [1, 2]
    assert units[1].blocks[0].text == "2. Member States may maintain more specific provisions."


def test_points_are_labelled_and_nested_points_indented(act: ParsedAct) -> None:
    blocks = _get(act, "art_6").units[0].blocks
    assert [b.label for b in blocks] == [None, "(a)", "(b)"]
    assert (
        blocks[2].text == "(b) processing is necessary for:\n  (i) the performance of a contract;"
    )


def test_unnumbered_article_is_one_unit_with_lead_in(act: ParsedAct) -> None:
    (unit,) = _get(act, "art_4").units
    assert unit.paragraph is None
    assert unit.blocks[0].text == "For the purposes of this Regulation:"
    assert [b.label for b in unit.blocks[1:]] == ["(1)", "(2)", "(3)"]


def test_footnote_markers_and_bodies_are_removed(act: ParsedAct) -> None:
    definition = _get(act, "art_4").units[0].blocks[3].text
    assert definition.endswith("as defined in Directive 95/46/EC;")
    all_text = " ".join(b.text for p in act.provisions for u in p.units for b in u.blocks)
    assert "of the European Parliament and of the Council." not in all_text  # footnote body


def test_annex_title_and_data_table(act: ParsedAct) -> None:
    annex = _get(act, "anx_I")
    assert (annex.number, annex.title) == ("I", "SECTORS OF HIGH CRITICALITY")
    texts = [b.text for b in annex.units[0].blocks]
    assert texts == [
        "Sector | Subsector | Type of entity",
        "1. Energy | (a) Electricity | Distribution system operators",
    ]


def test_french_point_labels() -> None:
    xhtml = FIXTURE.read_bytes().replace(b'<p class="oj-normal">(a)</p>', b"<p>a)</p>")
    art6 = next(p for p in parse_act(xhtml, "32016R0679", "fr").provisions if p.eli_id == "art_6")
    assert art6.units[0].blocks[1].label == "a)"


def test_rejects_documents_that_are_not_oj_acts() -> None:
    with pytest.raises(ParseError, match="no title"):
        parse_act(b"<html xmlns='http://www.w3.org/1999/xhtml'><body/></html>", "X", "en")


def test_does_not_resolve_external_entities() -> None:
    xxe = (
        b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        b"<html xmlns='http://www.w3.org/1999/xhtml'><div id='tit_1'>&x;</div></html>"
    )
    with pytest.raises(ParseError):  # parses safely, finds no articles
        parse_act(xxe, "X", "en")


def test_exponents_and_ordinals_in_superscript() -> None:
    xhtml = FIXTURE.read_bytes().replace(
        b"Member States may maintain more specific provisions.",
        b'Models above 10<span class="oj-super">25</span> FLOPs, see Regulation (EC) '
        b'N<span class="oj-super">o</span> 45/2001.',
    )
    art6 = next(p for p in parse_act(xhtml, "32016R0679", "en").provisions if p.eli_id == "art_6")
    assert art6.units[1].blocks[0].text == (
        "2. Models above 10^25 FLOPs, see Regulation (EC) No 45/2001."
    )
