"""Retrieval metrics over provision keys (binary relevance).

`expected` keys may be paragraph-level ("…:art_33:p1") or article-level ("…:art_4"); a retrieved
key is relevant to an expected key if it is equal to it or nested under it.
"""

import math


def matches(retrieved: str, expected: str) -> bool:
    return retrieved == expected or retrieved.startswith(f"{expected}:")


def _relevant(retrieved: str, expected: list[str]) -> bool:
    return any(matches(retrieved, e) for e in expected)


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """1 if any relevant provision is in the top k."""
    return float(any(_relevant(r, expected) for r in retrieved[:k]))


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Share of expected provisions found in the top k (matters for multi-provision questions)."""
    if not expected:
        return 0.0
    found = sum(1 for e in expected if any(matches(r, e) for r in retrieved[:k]))
    return found / len(expected)


def mrr_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Reciprocal rank of the first relevant provision (0 if none in the top k)."""
    for rank, r in enumerate(retrieved[:k], start=1):
        if _relevant(r, expected):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Each expected provision can be credited once, at the rank where it is first found."""
    if not expected:
        return 0.0
    credited: set[str] = set()
    dcg = 0.0
    for rank, r in enumerate(retrieved[:k]):
        for e in expected:
            if e not in credited and matches(r, e):
                credited.add(e)
                dcg += 1.0 / math.log2(rank + 2)
                break
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(expected), k)))
    return dcg / ideal
