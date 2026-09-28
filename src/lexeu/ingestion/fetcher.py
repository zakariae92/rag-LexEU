"""Fetch legal acts from the EU Publications Office (Cellar) REST API.

`GET /resource/celex/{CELEX}` with `Accept: application/xhtml+xml` and
`Accept-Language: eng|fra` redirects (303) to the XHTML manifestation of the act.
We use the official API rather than scraping eur-lex.europa.eu, which sits behind a bot
challenge and whose HTML is not meant for machines.
"""

import asyncio
import ssl
from dataclasses import dataclass

import httpx
import structlog
import truststore

from lexeu.core.config import IngestionSettings
from lexeu.ingestion.corpus import Lang

log = structlog.get_logger(__name__)

_CELLAR_LANG: dict[Lang, str] = {"en": "eng", "fr": "fra"}
_RETRYABLE = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class FetchedDoc:
    content: bytes
    source_url: str  # the stable CELEX URL we asked for
    manifestation_url: str  # where Cellar actually served it from (versioned)
    etag: str | None


class FetchError(RuntimeError):
    pass


class CellarFetcher:
    def __init__(self, settings: IngestionSettings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            # OS certificate store instead of certifi: works behind corporate proxies and
            # antivirus HTTPS inspection, and uses the system CA bundle in containers.
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            timeout=settings.timeout_s,
            follow_redirects=True,
            headers={"User-Agent": settings.user_agent},
        )

    async def fetch(self, celex: str, lang: Lang) -> FetchedDoc:
        url = f"{self._settings.cellar_url}/{celex}"
        headers = {"Accept": "application/xhtml+xml", "Accept-Language": _CELLAR_LANG[lang]}

        for attempt in range(1, self._settings.max_retries + 1):
            try:
                resp = await self._client.get(url, headers=headers)
            except httpx.TransportError as exc:
                if _is_tls_verification_error(exc):  # permanent: retrying won't help
                    raise FetchError(f"{celex}/{lang}: TLS verification failed: {exc}") from exc
                error: str = repr(exc)
            else:
                if resp.status_code == 200:
                    return FetchedDoc(
                        content=resp.content,
                        source_url=url,
                        manifestation_url=str(resp.url),
                        etag=resp.headers.get("etag"),
                    )
                if resp.status_code not in _RETRYABLE:
                    raise FetchError(f"{celex}/{lang}: HTTP {resp.status_code} from {url}")
                error = f"HTTP {resp.status_code}"

            if attempt < self._settings.max_retries:
                delay = 2**attempt
                log.warning("fetch_retry", celex=celex, lang=lang, error=error, retry_in_s=delay)
                await asyncio.sleep(delay)

        raise FetchError(f"{celex}/{lang}: giving up after {self._settings.max_retries} attempts")

    async def aclose(self) -> None:
        await self._client.aclose()


def _is_tls_verification_error(exc: BaseException) -> bool:
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, ssl.SSLCertVerificationError):
            return True
        cause = cause.__cause__ or cause.__context__
    return False
