"""rag-LexEU: bilingual RAG assistant over EU regulations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rag-lexeu")
except PackageNotFoundError:  # source mounted without install, e.g. inside a Modal container
    __version__ = "0.0.0+unknown"
