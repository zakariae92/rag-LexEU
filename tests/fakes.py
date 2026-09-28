"""In-memory stand-ins for the pipeline's ports (fetcher, raw store, registry)."""

from pathlib import Path

from lexeu.infra.db import DocumentMeta, DocumentRecord
from lexeu.ingestion.corpus import ActSpec, Corpus, Lang
from lexeu.ingestion.fetcher import FetchedDoc, FetchError
from lexeu.ingestion.models import Chunk

XHTML = (Path(__file__).parent / "fixtures" / "oj_act_en.xhtml").read_bytes()
GDPR = ActSpec(
    celex="32016R0679",
    official_name="Regulation (EU) 2016/679",
    short_name={"en": "GDPR", "fr": "RGPD"},
)
CORPUS = Corpus(languages=["en", "fr"], acts=[GDPR])


class FakeFetcher:
    def __init__(self) -> None:
        self.content = XHTML
        self.fail: set[tuple[str, str]] = set()

    async def fetch(self, celex: str, lang: Lang) -> FetchedDoc:
        if (celex, lang) in self.fail:
            raise FetchError("boom")
        return FetchedDoc(self.content, f"https://x/{celex}", f"https://x/{celex}/v1", None)


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data

    async def exists(self, key: str) -> bool:
        return key in self.objects


class FakeRegistry:
    def __init__(self) -> None:
        self.docs: dict[tuple[str, str], tuple[DocumentMeta, list[Chunk]]] = {}

    async def get_document(self, celex: str, lang: str) -> DocumentRecord | None:
        if (celex, lang) not in self.docs:
            return None
        meta, chunks = self.docs[(celex, lang)]
        return DocumentRecord(celex, lang, meta.content_sha256, meta.pipeline_version, len(chunks))

    async def list_documents(self) -> list[DocumentRecord]:
        return [r for k in self.docs if (r := await self.get_document(*k))]

    async def save_document(self, meta: DocumentMeta, chunks: list[Chunk]) -> None:
        self.docs[(meta.celex, meta.lang)] = (meta, chunks)

    async def delete_document(self, celex: str, lang: str) -> None:
        del self.docs[(celex, lang)]
