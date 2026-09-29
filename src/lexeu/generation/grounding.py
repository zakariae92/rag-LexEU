"""Deterministic grounding checks on a generated answer, run on every request.

They cannot tell whether a cited source really supports a sentence (the LLM judge measures that
offline), but they catch the failures that must never reach a user: citations to sources that
were not given, and answers that cite nothing at all.
"""

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?!\[)|\n+")  # "... 72 hours. [1]" is one sentence
_MIN_SENTENCE_CHARS = 25  # shorter fragments are headings or list labels, not claims


@dataclass(frozen=True)
class GroundingCheck:
    text: str  # the answer with invalid markers removed
    cited: list[int]  # valid source numbers, in order of first citation
    invalid: list[int]  # numbers that match no source
    sentences: int
    uncited_sentences: int

    @property
    def grounded(self) -> bool:
        return bool(self.cited)

    @property
    def uncited_ratio(self) -> float:
        return self.uncited_sentences / self.sentences if self.sentences else 0.0


def _numbers(marker: str) -> list[int]:
    return [int(x) for x in marker.split(",")]


def check_grounding(answer: str, n_sources: int) -> GroundingCheck:
    cited: list[int] = []
    invalid: list[int] = []
    for m in _CITATION.finditer(answer):
        for n in _numbers(m.group(1)):
            bucket = cited if 1 <= n <= n_sources else invalid
            if n not in bucket:
                bucket.append(n)

    def keep_valid(m: re.Match[str]) -> str:
        valid = [n for n in _numbers(m.group(1)) if 1 <= n <= n_sources]
        return "".join(f"[{n}]" for n in valid)

    text = _CITATION.sub(keep_valid, answer).strip()
    claims = [s for s in _SENTENCE_END.split(text) if len(s.strip()) >= _MIN_SENTENCE_CHARS]
    uncited = sum(1 for s in claims if not _CITATION.search(s))
    return GroundingCheck(text, cited, invalid, len(claims), uncited)
