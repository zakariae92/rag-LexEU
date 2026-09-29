"""Prometheus metrics exposed on /metrics and fed by the answer routes."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from tests.fakes import FakeLimiter
from tests.unit.test_api_ask import FailingAnswerer, FakeAnswerer


def _value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_is_public_and_in_prometheus_format(client: TestClient) -> None:
    resp = client.get("/metrics", headers={"Authorization": ""})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "lexeu_http_requests_total" in resp.text


def test_answers_feed_outcome_latency_tokens_and_cost(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FakeAnswerer()
    labels = {"outcome": "answered", "reason": "none", "cache": "miss", "lang": "en"}
    before = _value("lexeu_answers_total", **labels)
    cost_before = _value("lexeu_llm_cost_usd_total", model="fake/llm")
    total_before = _value("lexeu_answer_stage_seconds_count", stage="total", cache="miss")

    client.post("/v1/ask", json={"question": "Breach notification deadline?", "lang": "en"})

    assert _value("lexeu_answers_total", **labels) == before + 1
    assert _value("lexeu_llm_cost_usd_total", model="fake/llm") == cost_before + 0.0008
    assert _value("lexeu_answer_stage_seconds_count", stage="total", cache="miss") == (
        total_before + 1
    )


def test_http_metrics_use_the_route_template(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FakeAnswerer()
    before = _value("lexeu_http_requests_total", method="POST", route="/v1/ask", status="200")
    client.post("/v1/ask", json={"question": "Breach notification deadline?"})
    after = _value("lexeu_http_requests_total", method="POST", route="/v1/ask", status="200")
    assert after == before + 1


def test_failures_and_rate_limits_are_counted(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FailingAnswerer()
    errors = _value("lexeu_llm_errors_total", error="TimeoutError")
    client.post("/v1/ask", json={"question": "Breach notification deadline?"})
    assert _value("lexeu_llm_errors_total", error="TimeoutError") == errors + 1

    app.state.rate_limiter = FakeLimiter(budget=0)
    limited = _value("lexeu_rate_limited_total")
    assert client.post("/v1/ask", json={"question": "Any question?"}).status_code == 429
    assert _value("lexeu_rate_limited_total") == limited + 1
