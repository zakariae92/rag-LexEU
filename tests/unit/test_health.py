import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexeu import __version__


async def _ok() -> None:
    return None


async def _boom() -> None:
    raise ConnectionError("connection refused to 10.0.0.1 with password=hunter2")


async def _slow() -> None:
    await asyncio.sleep(5)


def test_health_is_ok_without_dependencies(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}


def test_ready_when_all_dependencies_up(app: FastAPI, client: TestClient) -> None:
    app.state.probes = {"postgres": _ok, "qdrant": _ok}

    resp = client.get("/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert {d["status"] for d in body["dependencies"].values()} == {"up"}


def test_not_ready_when_a_dependency_fails(app: FastAPI, client: TestClient) -> None:
    app.state.probes = {"postgres": _ok, "qdrant": _boom}

    resp = client.get("/ready")

    assert resp.status_code == 503
    deps = resp.json()["dependencies"]
    assert deps["postgres"]["status"] == "up"
    assert deps["qdrant"] == {
        "status": "down",
        "latency_ms": deps["qdrant"]["latency_ms"],
        "error": "ConnectionError",
    }
    assert "hunter2" not in resp.text  # error details never leak


def test_slow_dependency_times_out(app: FastAPI, client: TestClient) -> None:
    app.state.probes = {"redis": _slow}

    resp = client.get("/ready")

    assert resp.status_code == 503
    assert resp.json()["dependencies"]["redis"]["error"] == "TimeoutError"


def test_request_id_is_generated_and_propagated(client: TestClient) -> None:
    generated = client.get("/health").headers["x-request-id"]
    assert len(generated) == 32

    propagated = client.get("/health", headers={"x-request-id": "abc-123"})
    assert propagated.headers["x-request-id"] == "abc-123"
