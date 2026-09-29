from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexeu.generation.answer import Answer, Citation


class FakeAnswerer:
    async def answer(self, question: str, lang: str | None = None) -> Answer:
        return Answer(
            question=question,
            lang=lang or "en",
            text="Within 72 hours [1].",
            refused=False,
            refusal_reason=None,
            citations=[Citation(1, "32016R0679:art_33:p1", "Art. 33(1) GDPR", "en", "c1")],
            sources=[],
            model="fake/llm",
            input_tokens=900,
            output_tokens=20,
            cost_usd=0.0008,
            timings_ms={"retrieval": 25.0, "generation": 900.0, "total": 925.0},
        )


def test_ask_is_unavailable_without_an_llm_key(client: TestClient) -> None:
    # The test settings carry no GEMINI_API_KEY: the API starts, generation is disabled.
    resp = client.post("/v1/ask", json={"question": "What is personal data?"})
    assert resp.status_code == 503


def test_ask_returns_answer_citations_and_usage(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FakeAnswerer()
    resp = client.post("/v1/ask", json={"question": "Breach notification deadline?", "lang": "en"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Within 72 hours [1]."
    assert body["citations"] == [
        {
            "n": 1,
            "citation": "Art. 33(1) GDPR",
            "provision_key": "32016R0679:art_33:p1",
            "lang": "en",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679#art_33",
        }
    ]
    assert body["usage"]["cost_usd"] == 0.0008
    assert body["timings_ms"]["total"] == 925.0


def test_ask_validates_the_request(client: TestClient) -> None:
    assert client.post("/v1/ask", json={"question": "x"}).status_code == 422
    assert (
        client.post("/v1/ask", json={"question": "Valid question", "lang": "de"}).status_code == 422
    )


class FailingAnswerer:
    async def answer(self, question: str, lang: str | None = None) -> Answer:
        raise TimeoutError("provider timed out")


def test_provider_failure_is_a_retryable_503(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FailingAnswerer()
    resp = client.post("/v1/ask", json={"question": "Breach notification deadline?"})
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"
