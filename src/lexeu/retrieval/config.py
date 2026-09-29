"""A retrieval configuration is one experiment; ablations compare them on the golden set."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    name: str = "dense"
    description: str = ""
    mode: Literal["dense", "sparse", "hybrid"] = "dense"  # sparse = BM25 only (diagnostic)
    fusion: Literal["rrf", "dbsf"] = "rrf"  # how dense and BM25 rankings are merged (hybrid only)
    dense_weight: float = Field(default=1.0, gt=0)  # RRF weight of the dense ranking (BM25 = 1)
    candidates: int = Field(default=60, ge=10)  # points fetched before policies and dedup
    recitals: Literal["keep", "demote", "exclude"] = "keep"
    recital_penalty: float = Field(default=0.9, gt=0, le=1)  # score multiplier when demoting
    lang: Literal["any", "query"] = "any"  # "query": only chunks in the question's language
    rerank: str | None = None  # cross-encoder model id, applied to the top `rerank_top`
    rerank_top: int = Field(default=30, ge=1)


def load_experiments(path: Path) -> list[RetrievalConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [RetrievalConfig.model_validate(e) for e in raw["experiments"]]
