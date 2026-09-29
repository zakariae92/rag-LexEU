"""Authentication, rate limiting, answer ids and feedback on the public API."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexeu.api.main import create_app
from lexeu.core.config import Settings
from tests.fakes import FakeLimiter, MemoryAnswerLog
from tests.unit.test_api_ask import FakeAnswerer

QUESTION = {"question": "Breach notification deadline?"}


def test_missing_or_wrong_key_is_401(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FakeAnswerer()
    no_key = client.post("/v1/ask", json=QUESTION, headers={"Authorization": ""})
    wrong = client.post("/v1/ask", json=QUESTION, headers={"Authorization": "Bearer lx_nope"})
    assert no_key.status_code == wrong.status_code == 401
    assert no_key.headers["www-authenticate"] == "Bearer"


def test_health_needs_no_key(client: TestClient) -> None:
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200


def test_anonymous_access_when_auth_is_optional(settings: Settings) -> None:
    settings.auth.required = False
    app = create_app(settings)
    limiter = FakeLimiter()
    with TestClient(app) as c:
        app.state.rate_limiter = limiter
        app.state.answer_log = MemoryAnswerLog()
        app.state.answerer = FakeAnswerer()
        assert c.post("/v1/ask", json=QUESTION).status_code == 200
    assert list(limiter.seen) == ["ip:testclient"]  # anonymous callers are limited per IP


def test_rate_limit_returns_429_with_retry_after(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FakeAnswerer()
    app.state.rate_limiter = FakeLimiter(budget=2)
    ok = [client.post("/v1/ask", json=QUESTION) for _ in range(2)]
    limited = client.post("/v1/ask", json=QUESTION)

    assert [r.status_code for r in ok] == [200, 200]
    assert ok[0].headers["x-ratelimit-remaining"] == "1"
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "42"


def test_answers_get_an_id_and_accept_feedback(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FakeAnswerer()
    answer_id = client.post("/v1/ask", json=QUESTION).json()["answer_id"]
    assert answer_id in app.state.answer_log.answers  # logged in the background

    resp = client.post(
        "/v1/feedback", json={"answer_id": answer_id, "rating": -1, "comment": "wrong article"}
    )
    assert resp.status_code == 204
    assert app.state.answer_log.feedbacks[answer_id] == (-1, "wrong article")


def test_feedback_validation(client: TestClient) -> None:
    unknown = {"answer_id": "0" * 32, "rating": 1}
    assert client.post("/v1/feedback", json=unknown).status_code == 404
    assert client.post("/v1/feedback", json={**unknown, "rating": 5}).status_code == 422
    assert client.post("/v1/feedback", json={"answer_id": "x", "rating": 1}).status_code == 422
