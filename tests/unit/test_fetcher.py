import ssl

import httpx
import pytest

from lexeu.core.config import IngestionSettings
from lexeu.ingestion.fetcher import CellarFetcher, FetchError

SETTINGS = IngestionSettings(cellar_url="https://cellar.test/celex", max_retries=3)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("lexeu.ingestion.fetcher.asyncio.sleep", instant)


def _fetcher(handler: httpx.MockTransport) -> CellarFetcher:
    return CellarFetcher(SETTINGS, httpx.AsyncClient(transport=handler, follow_redirects=True))


async def test_negotiates_language_and_follows_redirect() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/celex/32016R0679":
            return httpx.Response(303, headers={"location": "https://cellar.test/cellar/v9/DOC_1"})
        return httpx.Response(200, content=b"<xhtml/>", headers={"etag": '"v9"'})

    doc = await _fetcher(httpx.MockTransport(handler)).fetch("32016R0679", "fr")

    assert seen[0].headers["accept"] == "application/xhtml+xml"
    assert seen[0].headers["accept-language"] == "fra"
    assert doc.content == b"<xhtml/>"
    assert doc.source_url == "https://cellar.test/celex/32016R0679"
    assert doc.manifestation_url == "https://cellar.test/cellar/v9/DOC_1"
    assert doc.etag == '"v9"'


async def test_retries_transient_errors() -> None:
    responses = iter([httpx.Response(503), httpx.Response(200, content=b"ok")])
    doc = await _fetcher(httpx.MockTransport(lambda _: next(responses))).fetch("X", "en")
    assert doc.content == b"ok"


async def test_gives_up_after_max_retries() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502)

    with pytest.raises(FetchError, match="giving up after 3 attempts"):
        await _fetcher(httpx.MockTransport(handler)).fetch("X", "en")
    assert calls == 3


async def test_client_errors_fail_fast() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    with pytest.raises(FetchError, match="HTTP 404"):
        await _fetcher(httpx.MockTransport(handler)).fetch("32099R9999", "en")
    assert calls == 1


async def test_tls_verification_errors_are_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("cert", request=request) from ssl.SSLCertVerificationError()

    with pytest.raises(FetchError, match="TLS verification failed"):
        await _fetcher(httpx.MockTransport(handler)).fetch("X", "en")
    assert calls == 1
