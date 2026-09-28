"""Parse Official Journal XHTML (Cellar "oj-convex" format) into a legal structure.

The format carries ELI identifiers that are identical in every language version:
    rct_12            recital 12
    cpt_II[.sct_1]    chapter / section containers (with their titles)
    art_6             article 6, with title in `art_6.tit_1`
    006.001           numbered paragraph 1 of article 6
    anx_III           annex III
Lettered points ("(a)", "(1)", "2.") are two-cell tables: label | content, possibly nested.
"""

import re

from lxml import etree

from lexeu.ingestion.corpus import Lang
from lexeu.ingestion.models import Block, ParsedAct, Provision, Unit

PARSER_VERSION = "2"  # bump when output changes: forces a re-parse of unchanged documents

_RE_ARTICLE = re.compile(r"^art_\w+$")
_RE_RECITAL = re.compile(r"^rct_(\d+)$")
_RE_ANNEX = re.compile(r"^anx_\w+$")
_RE_CONTAINER = re.compile(r"^(?:prt|tis|cpt|sct)_[^.]+(?:\.(?:tis|cpt|sct)_[^.]+)*$")
_RE_PARAGRAPH = re.compile(r"^\d{3}\.(\d{3})$")
# Point labels: "(a)" "(iv)" "(12)" (EN), "a)" "iv)" (FR), "1." "12." and dash items.
_RE_LABEL = re.compile(r"^(?:\(\w{1,6}\)|\w{1,6}\)|\w{1,4}\.|[\u2014\u2013-])$")
_RE_SPACES = re.compile(r"\s+")
_RE_EMPTY_NOTE = re.compile(r" ?\(\s*\)")  # brackets left around a removed footnote marker

_XML_PARSER = etree.XMLParser(
    resolve_entities=False, no_network=True, load_dtd=False, huge_tree=True, remove_comments=True
)


class ParseError(ValueError):
    pass


def parse_act(xhtml: bytes, celex: str, lang: Lang) -> ParsedAct:
    root = etree.fromstring(xhtml, parser=_XML_PARSER)
    by_id = {eli_id: el for el in root.iter() if (eli_id := el.get("id"))}

    title_el = by_id.get("tit_1")
    if title_el is None:
        raise ParseError(f"{celex}/{lang}: no title (tit_1), not an OJ XHTML document?")

    provisions: list[Provision] = []
    for eli_id, el in by_id.items():
        if m := _RE_RECITAL.match(eli_id):
            provisions.append(_recital(el, eli_id, m.group(1)))
        elif _RE_ARTICLE.match(eli_id) and "eli-subdivision" in _classes(el):
            provisions.append(_article(el, eli_id, by_id))
        elif _RE_ANNEX.match(eli_id):
            provisions.append(_annex(el, eli_id))

    if not any(p.kind == "article" for p in provisions):
        raise ParseError(f"{celex}/{lang}: no articles found")
    return ParsedAct(celex=celex, lang=lang, title=_text(title_el), provisions=provisions)


# --------------------------------------------------------------------------- provisions


def _recital(el: etree._Element, eli_id: str, number: str) -> Provision:
    # <table><tr><td>(12)</td><td>text…</td></tr></table>
    cells = _find(el, ".//*[local-name()='td']")
    content = cells[1] if len(cells) >= 2 else el
    blocks = _blocks(list(content))
    return Provision("recital", eli_id, number, None, [], [Unit(blocks)])


def _article(el: etree._Element, eli_id: str, by_id: dict[str, etree._Element]) -> Provision:
    title_el = by_id.get(f"{eli_id}.tit_1")
    units: list[Unit] = []
    loose: list[etree._Element] = []  # content not wrapped in a numbered paragraph

    def flush_loose() -> None:
        if loose and (blocks := _blocks(loose)):
            units.append(Unit(blocks))
        loose.clear()

    for child in el:
        if "oj-ti-art" in _classes(child) or child is title_el:
            continue
        if m := _RE_PARAGRAPH.match(child.get("id", "")):
            flush_loose()
            units.append(Unit(_blocks(list(child)), paragraph=int(m.group(1))))
        else:
            loose.append(child)
    flush_loose()

    return Provision(
        kind="article",
        eli_id=eli_id,
        number=eli_id.removeprefix("art_"),
        title=_text(title_el) if title_el is not None else None,
        heading_path=_heading_path(el),
        units=units,
    )


def _annex(el: etree._Element, eli_id: str) -> Provision:
    children = list(el)
    # First oj-doc-ti is "ANNEX III", the following ones are its title.
    titles = [c for c in children if "oj-doc-ti" in _classes(c)]
    body = [c for c in children if c not in titles]
    title = " ".join(_text(t) for t in titles[1:]) or None
    return Provision("annex", eli_id, eli_id.removeprefix("anx_"), title, [], [Unit(_blocks(body))])


def _heading_path(el: etree._Element) -> list[str]:
    """["CHAPTER II: Principles", "Section 1: …"] from the enclosing containers."""
    path: list[str] = []
    for anc in el.iterancestors():
        if not _RE_CONTAINER.match(anc.get("id", "")):
            continue
        label = next((_text(c) for c in anc if "oj-ti-section-1" in _classes(c)), "")
        title = next((_text(c) for c in anc if "eli-title" in _classes(c)), "")
        heading = ": ".join(p for p in (label, title) if p)
        if heading:
            path.append(heading)
    return list(reversed(path))


# --------------------------------------------------------------------------- text rendering


def _blocks(nodes: list[etree._Element]) -> list[Block]:
    """Top-level blocks: each sentence-paragraph or lettered point becomes one Block."""
    out: list[Block] = []
    for node in nodes:
        tag = _local(node)
        if tag == "p":
            if "oj-note" in _classes(node):  # footnote body
                continue
            if text := _text(node):
                out.append(Block(text))
        elif tag == "table":
            for row in _rows(node):
                if item := _labelled(row):
                    label, content = item
                    lines = _lines(list(content), depth=1)
                    text = (
                        "\n".join([f"{label} {lines[0].strip()}", *lines[1:]]) if lines else label
                    )
                    out.append(Block(text, label=label))
                elif text := " | ".join(_text(c) for c in row if _text(c)):
                    out.append(Block(text))  # a real data table (e.g. NIS2 annex sectors)
        elif tag == "div":
            out.extend(_blocks(list(node)))
    return out


def _lines(nodes: list[etree._Element], depth: int) -> list[str]:
    """Nested content as indented lines, so sub-points keep their hierarchy."""
    indent = "  " * depth
    out: list[str] = []
    for node in nodes:
        tag = _local(node)
        if tag == "p" and (text := _text(node)):
            out.append(indent + text)
        elif tag == "table":
            for row in _rows(node):
                if item := _labelled(row):
                    label, content = item
                    sub = _lines(list(content), depth + 1)
                    first = sub[0].strip() if sub else ""
                    out.append(f"{indent}{label} {first}".rstrip())
                    out.extend(sub[1:])
                elif text := " | ".join(_text(c) for c in row if _text(c)):
                    out.append(indent + text)
        elif tag == "div":
            out.extend(_lines(list(node), depth))
    return out


def _rows(table: etree._Element) -> list[list[etree._Element]]:
    rows = _find(table, "./*[local-name()='tbody']/*[local-name()='tr'] | ./*[local-name()='tr']")
    return [[c for c in row if _local(c) == "td"] for row in rows]


def _labelled(cells: list[etree._Element]) -> tuple[str, etree._Element] | None:
    if len(cells) == 2 and _RE_LABEL.match(label := _text(cells[0])):
        return label, cells[1]
    return None


def _text(el: etree._Element) -> str:
    """Visible text, without footnote markers, whitespace normalised.

    `oj-super` marks footnote markers (dropped), ordinals ("No", "1er": kept inline) and
    exponents: "10<sup>25</sup>" must read "10^25", not "1025".
    """
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        classes = _classes(node)
        if "oj-note-tag" not in classes:
            if node.text:
                is_exponent = (
                    "oj-super" in classes
                    and node.text.strip().isdigit()
                    and "".join(parts)[-1:].isdigit()
                )
                parts.append(f"^{node.text.strip()}" if is_exponent else node.text)
            for child in node:
                walk(child)
        if node is not el and node.tail:
            parts.append(node.tail)

    walk(el)
    text = _RE_SPACES.sub(" ", "".join(parts).replace("\xa0", " "))
    return _RE_EMPTY_NOTE.sub("", text).strip()  # "Council (1)" -> "Council ()" -> "Council"


def _find(el: etree._Element, xpath: str) -> list[etree._Element]:
    result = el.xpath(xpath)
    return [n for n in result if isinstance(n, etree._Element)] if isinstance(result, list) else []


def _classes(el: etree._Element) -> list[str]:
    return str(el.get("class", "")).split()


def _local(el: etree._Element) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""
