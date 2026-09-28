"""The corpus definition: which acts, in which languages."""

import tomllib
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel, Field

Lang = Literal["en", "fr"]


class ActSpec(BaseModel):
    celex: str = Field(pattern=r"^3\d{4}[RLD]\d{4}$")
    official_name: str
    short_name: dict[Lang, str]


class Corpus(BaseModel):
    languages: list[Lang]
    acts: list[ActSpec]

    def get(self, celex: str) -> ActSpec:
        for act in self.acts:
            if act.celex == celex:
                return act
        raise KeyError(f"{celex} is not in the corpus")


def load_corpus() -> Corpus:
    raw = files("lexeu.ingestion").joinpath("corpus.toml").read_text(encoding="utf-8")
    return Corpus.model_validate(tomllib.loads(raw))
