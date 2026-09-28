"""Turn a parsed act into retrieval chunks that follow the legal structure.

- One chunk per citable unit: a numbered paragraph, a recital, an unnumbered article body.
- Units longer than `max_chars` are split between points (never inside one), and every
  part repeats the unit's lead-in sentence so it stays self-explanatory.
- Every chunk gets a breadcrumb header (act > chapter > article > paragraph) and a
  human citation ("Art. 6(1) GDPR"), and a language-neutral key aligning EN and FR.
"""

from lexeu.ingestion.corpus import ActSpec, Lang
from lexeu.ingestion.models import Block, Chunk, ParsedAct, Provision, Unit

DEFAULT_MAX_CHARS = 1800  # ~400 tokens

_WORDS: dict[Lang, dict[str, str]] = {
    "en": {"article": "Article", "recital": "Recital", "annex": "Annex",
           "paragraph": "Paragraph", "point": "point", "points": "points", "to": "\u2013"},
    "fr": {"article": "Article", "recital": "Considérant", "annex": "Annexe",
           "paragraph": "Paragraphe", "point": "point", "points": "points", "to": " à "},
}  # fmt: skip


def chunk_act(act: ParsedAct, spec: ActSpec, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for provision in act.provisions:
        for index, unit in enumerate(provision.units, start=1):
            for part, blocks in enumerate(_split(unit.blocks, max_chars), start=1):
                chunks.append(_make_chunk(act, spec, provision, unit, index, part, blocks))
    return chunks


def _split(blocks: list[Block], max_chars: int) -> list[list[Block]]:
    if _size(blocks) <= max_chars:
        return [blocks]

    # Lead-in: the unlabelled sentences before the first point ("… the following applies:").
    lead: list[Block] = []
    for block in blocks:
        if block.label is not None:
            break
        lead.append(block)
    items = blocks[len(lead) :]
    if not items:  # long prose without points: pack sentences, no lead to repeat
        lead, items = [], blocks

    parts: list[list[Block]] = []
    current: list[Block] = []
    for item in items:
        if current and _size([*lead, *current, item]) > max_chars:
            parts.append([*lead, *current])
            current = []
        current.append(item)  # a single oversized point stays whole
    parts.append([*lead, *current])
    return parts


def _make_chunk(
    act: ParsedAct,
    spec: ActSpec,
    provision: Provision,
    unit: Unit,
    unit_index: int,
    part: int,
    blocks: list[Block],
) -> Chunk:
    lang = act.lang
    short = spec.short_name[lang]

    unit_key = f"p{unit.paragraph}" if unit.paragraph is not None else f"u{unit_index}"
    provision_key = f"{act.celex}:{provision.eli_id}:{unit_key}"

    labels = [b.label for b in blocks if b.label]
    split = len(labels) != sum(1 for b in unit.blocks if b.label)  # only part of the points

    return Chunk(
        chunk_id=f"{act.celex}:{lang}:{provision.eli_id}:{unit_key}:{part}",
        provision_key=provision_key,
        celex=act.celex,
        lang=lang,
        kind=provision.kind,
        eli_id=provision.eli_id,
        paragraph=unit.paragraph,
        part=part,
        citation=_citation(provision, unit, short, labels if split else [], lang),
        header=_header(spec, provision, unit, lang),
        text="\n".join(b.text for b in blocks),
    )


def _citation(provision: Provision, unit: Unit, short: str, points: list[str], lang: Lang) -> str:
    w = _WORDS[lang]
    n = provision.number
    if provision.kind == "recital":
        base = f"{w['recital']} {n} {short}"
    elif provision.kind == "annex":
        base = f"{w['annex']} {n} {short}"
    elif unit.paragraph is None:
        base = f"Art. {n} {short}"
    elif lang == "fr":
        base = f"Art. {n}, par. {unit.paragraph}, {short}"
    else:
        base = f"Art. {n}({unit.paragraph}) {short}"

    if not points:
        return base
    if len(points) == 1:
        return f"{base}, {w['point']} {points[0]}"
    return f"{base}, {w['points']} {points[0]}{w['to']}{points[-1]}"


def _header(spec: ActSpec, provision: Provision, unit: Unit, lang: Lang) -> str:
    w = _WORDS[lang]
    sep = " : " if lang == "fr" else ": "
    name = spec.official_name
    if lang == "fr":
        name = name.replace("Regulation (EU)", "Règlement (UE)").replace(
            "Directive (EU)", "Directive (UE)"
        )

    label = f"{w[provision.kind]} {provision.number}"
    if provision.title:
        label += f"{sep}{provision.title}"
    parts = [f"{spec.short_name[lang]} ({name})", *provision.heading_path, label]
    if unit.paragraph is not None:
        parts.append(f"{w['paragraph']} {unit.paragraph}")
    return " > ".join(parts)


def _size(blocks: list[Block]) -> int:
    return sum(len(b.text) + 1 for b in blocks)
