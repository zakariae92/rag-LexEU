"""Lexical side of hybrid search: BM25 with language-specific stemming, and language detection.

Documents are stemmed with the stemmer of their own language; a query with the stemmer of the
language it is written in. Term frequencies are computed here, IDF by Qdrant (`Modifier.IDF`),
so adding documents never requires recomputing existing vectors.
"""

from functools import cache
from typing import TYPE_CHECKING

from qdrant_client import models

from lexeu.ingestion.corpus import Lang

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding
    from lingua import LanguageDetector

BM25_MODEL = "Qdrant/bm25"
BM25_AVG_LEN = 150.0  # average chunk length in words (header + text), used for length normalisation
_STEMMER: dict[Lang, str] = {"en": "english", "fr": "french"}


@cache
def _bm25(lang: Lang) -> "SparseTextEmbedding":
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(BM25_MODEL, language=_STEMMER[lang], avg_len=BM25_AVG_LEN)


@cache
def _detector() -> "LanguageDetector":
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.FRENCH).build()


def detect_lang(text: str) -> Lang:
    from lingua import Language

    return "fr" if _detector().detect_language_of(text) == Language.FRENCH else "en"


def bm25_documents(texts: list[str], lang: Lang) -> list[models.SparseVector]:
    return [
        models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in _bm25(lang).embed(texts)
    ]


def bm25_query(text: str, lang: Lang) -> models.SparseVector:
    e = next(iter(_bm25(lang).query_embed([text])))
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
