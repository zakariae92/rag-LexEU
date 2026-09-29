"""In-memory stand-ins for the pipeline's ports (fetcher, raw store, registry) and for the
serving side (API keys, rate limiter, answer log)."""

from pathlib import Path

from lexeu.infra.api_keys import ApiKey
from lexeu.infra.db import DocumentMeta, DocumentRecord
from lexeu.infra.rate_limit import RateDecision
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


# ----------------------------------------------------------------------------- serving side

TEST_KEY = "lx_test-key"


class FakeKeys:
    """Accepts exactly TEST_KEY."""

    async def authenticate(self, secret: str) -> ApiKey | None:
        return ApiKey(1, "tests", "lx_test", 100) if secret == TEST_KEY else None


class FakeLimiter:
    """Allows `budget` requests per subject, then refuses."""

    def __init__(self, budget: int = 1000) -> None:
        self.budget = budget
        self.seen: dict[str, int] = {}

    async def hit(self, subject: str, limit: int) -> RateDecision:
        self.seen[subject] = self.seen.get(subject, 0) + 1
        cap = min(self.budget, limit)
        allowed = self.seen[subject] <= cap
        return RateDecision(allowed, limit, max(0, cap - self.seen[subject]), 0 if allowed else 42)


class MemoryAnswerLog:
    def __init__(self) -> None:
        self.answers: dict[str, object] = {}
        self.feedbacks: dict[str, tuple[int, str | None]] = {}

    async def record(self, answer_id: str, answer: object, **_: object) -> None:
        self.answers[answer_id] = answer

    async def feedback(self, answer_id: str, rating: int, comment: str | None) -> bool:
        if answer_id not in self.answers:
            return False
        self.feedbacks[answer_id] = (rating, comment)
        return True
