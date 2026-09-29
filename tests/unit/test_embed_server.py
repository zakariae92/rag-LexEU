"""The embedding server's HTTP contract, which TeiEmbedder relies on (same as TEI's)."""

import numpy as np
import pytest
from starlette.testclient import TestClient

from lexeu.embed_server import InputTooLong, Limits, Vectors, create_app


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], truncate: bool) -> Vectors:
        self.calls.append(texts)
        if not truncate and any(len(t) > 10 for t in texts):
            raise InputTooLong("too long")
        return np.array([[3.0, 4.0]] * len(texts), dtype=np.float32)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def client(encoder: FakeEncoder) -> TestClient:
    return TestClient(create_app(encoder, Limits(max_batch=2)))


def test_health(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_embeds_a_batch_normalised_by_default(client: TestClient) -> None:
    resp = client.post("/embed", json={"inputs": ["a", "b"]})
    assert resp.status_code == 200
    assert resp.json() == [[pytest.approx(0.6), pytest.approx(0.8)]] * 2


def test_a_single_string_is_a_batch_of_one(client: TestClient, encoder: FakeEncoder) -> None:
    assert client.post("/embed", json={"inputs": "a", "normalize": False}).json() == [[3.0, 4.0]]
    assert encoder.calls == [["a"]]


def test_rejects_bad_inputs_and_oversized_batches(client: TestClient) -> None:
    assert client.post("/embed", json={"inputs": []}).status_code == 422
    assert client.post("/embed", json={"inputs": [1]}).status_code == 422
    assert client.post("/embed", content=b"not json").status_code == 422
    resp = client.post("/embed", json={"inputs": ["a", "b", "c"]})
    assert resp.status_code == 413 and "error" in resp.json()


def test_long_input_is_refused_only_without_truncation(client: TestClient) -> None:
    long = "x" * 20
    assert client.post("/embed", json={"inputs": [long]}).status_code == 200
    assert client.post("/embed", json={"inputs": [long], "truncate": False}).status_code == 413
