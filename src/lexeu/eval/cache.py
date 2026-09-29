"""Choose the semantic-cache similarity threshold from data.

For each candidate threshold: the share of paraphrases that would hit the cache (the benefit),
and the number of near-misses that would hit it (answers served for the wrong question). Every
pair of distinct golden-set questions in the same language is an extra negative. The recommended
threshold is the lowest one with zero false hits, plus a safety margin.
"""

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel

from lexeu.eval.golden import GoldenItem
from lexeu.retrieval.embeddings import Embedder
from lexeu.retrieval.sparse import detect_lang

THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(20)]  # 0.80 .. 0.99
MARGIN = 0.02


class Variant(BaseModel):
    same: bool
    text: str


class PairGroup(BaseModel):
    base: str
    variants: list[Variant]


class CachePairs(BaseModel):
    version: int
    pairs: list[PairGroup]


@dataclass(frozen=True)
class ScoredPair:
    a: str
    b: str
    same: bool
    similarity: float
    source: str  # "pairs" (hand-written) or "golden" (two distinct golden questions)


@dataclass(frozen=True)
class CacheReport:
    n_paraphrases: int
    n_near_misses: int
    n_golden_negatives: int
    by_threshold: list[dict[str, float]]
    max_negative: ScoredPair
    min_paraphrase: ScoredPair
    recommended: float | None  # None: no threshold separates them (do not cache)
    hardest_negatives: list[ScoredPair]
    missed_paraphrases: list[ScoredPair]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pairs(path: Path) -> CachePairs:
    return CachePairs.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


async def evaluate_cache(
    pairs: CachePairs, golden: list[GoldenItem], embedder: Embedder
) -> CacheReport:
    texts = sorted(
        {g.base for g in pairs.pairs}
        | {v.text for g in pairs.pairs for v in g.variants}
        | {i.question for i in golden}
    )
    vectors = await embedder.embed(texts)
    index = {t: vectors[i] for i, t in enumerate(texts)}

    def sim(a: str, b: str) -> float:
        return float(np.dot(index[a], index[b]))  # vectors are L2-normalised

    scored = [
        ScoredPair(g.base, v.text, v.same, sim(g.base, v.text), "pairs")
        for g in pairs.pairs
        for v in g.variants
    ]
    # The cache only matches within one answer language: compare same-language questions.
    by_lang: dict[str, list[str]] = {}
    for item in golden:
        by_lang.setdefault(item.lang, []).append(item.question)
    for questions in by_lang.values():
        scored += [
            ScoredPair(a, b, False, sim(a, b), "golden")
            for a, b in itertools.combinations(questions, 2)
        ]

    positives = [p for p in scored if p.same]
    negatives = [p for p in scored if not p.same]
    rows = []
    for t in THRESHOLDS:
        rows.append(
            {
                "threshold": t,
                "paraphrase_hit_rate": round(
                    sum(p.similarity >= t for p in positives) / len(positives), 4
                ),
                "near_miss_hits": sum(p.similarity >= t for p in negatives if p.source == "pairs"),
                "golden_pair_hits": sum(
                    p.similarity >= t for p in negatives if p.source == "golden"
                ),
            }
        )
    max_negative = max(negatives, key=lambda p: p.similarity)
    min_paraphrase = min(positives, key=lambda p: p.similarity)
    safe = math.ceil((max_negative.similarity + MARGIN) * 100) / 100
    recommended = safe if safe < 1.0 else None
    return CacheReport(
        n_paraphrases=len(positives),
        n_near_misses=sum(1 for p in negatives if p.source == "pairs"),
        n_golden_negatives=sum(1 for p in negatives if p.source == "golden"),
        by_threshold=rows,
        max_negative=max_negative,
        min_paraphrase=min_paraphrase,
        recommended=recommended,
        hardest_negatives=sorted(negatives, key=lambda p: -p.similarity)[:10],
        missed_paraphrases=sorted(
            (p for p in positives if recommended is None or p.similarity < recommended),
            key=lambda p: p.similarity,
        ),
    )


def to_markdown(report: CacheReport) -> str:
    r = report
    lines = [
        "## Semantic cache threshold",
        "",
        f"{r.n_paraphrases} paraphrases (should hit), {r.n_near_misses} hand-written near-misses "
        f"and {r.n_golden_negatives} pairs of distinct golden questions (must never hit).",
        "",
        "| Threshold | Paraphrases served from cache | Near-miss hits | Golden-pair hits |",
        "|---|---|---|---|",
    ]
    for row in r.by_threshold:
        lines.append(
            f"| {row['threshold']:.2f} | {row['paraphrase_hit_rate']:.1%} | "
            f"{int(row['near_miss_hits'])} | {int(row['golden_pair_hits'])} |"
        )
    rec = f"**{r.recommended:.2f}**" if r.recommended else "**none: do not cache**"
    lines += [
        "",
        f"Most similar negative: {r.max_negative.similarity:.3f} "
        f'("{r.max_negative.a}" vs "{r.max_negative.b}").',
        f"Least similar paraphrase: {r.min_paraphrase.similarity:.3f} "
        f'("{r.min_paraphrase.a}" vs "{r.min_paraphrase.b}").',
        f"Recommended threshold (max negative + {MARGIN}): {rec}.",
        "",
        "### Hardest negatives",
        "",
    ]
    lines += [f"- {p.similarity:.3f} [{p.source}] {p.a} / {p.b}" for p in r.hardest_negatives]
    lines += ["", f"### Paraphrases that would miss ({len(r.missed_paraphrases)})", ""]
    lines += [f"- {p.similarity:.3f} {p.a} / {p.b}" for p in r.missed_paraphrases]
    return "\n".join(lines) + "\n"


def check_lang_consistency(pairs: CachePairs) -> list[str]:
    """Variants must be in the base question's language (the cache never crosses languages)."""
    return [
        f"{g.base!r} / {v.text!r}"
        for g in pairs.pairs
        for v in g.variants
        if detect_lang(v.text) != detect_lang(g.base)
    ]
