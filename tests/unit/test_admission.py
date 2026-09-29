import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexeu.api.admission import Admission, Overloaded
from tests.unit.test_api_ask import FailingAnswerer, FakeAnswerer
from tests.unit.test_streaming import StreamingAnswerer

QUESTION = {"question": "Breach notification deadline?"}


async def test_excess_requests_are_shed_after_the_queue_timeout() -> None:
    admission = Admission(max_inflight=2, queue_timeout_s=0.05)
    await admission.acquire()
    await admission.acquire()
    with pytest.raises(Overloaded):
        await admission.acquire()
    admission.release()
    await asyncio.wait_for(admission.acquire(), 1)  # a freed slot is reusable


async def test_a_waiting_request_gets_the_next_free_slot() -> None:
    admission = Admission(max_inflight=1, queue_timeout_s=1.0)
    await admission.acquire()
    waiter = asyncio.create_task(admission.acquire())
    await asyncio.sleep(0.01)
    admission.release()
    await asyncio.wait_for(waiter, 1)  # queued, not refused


def test_full_capacity_is_a_fast_retryable_503(app: FastAPI, client: TestClient) -> None:
    app.state.answerer = FakeAnswerer()
    app.state.admission = admission = Admission(max_inflight=1, queue_timeout_s=0.01)
    assert client.portal is not None
    client.portal.call(admission.acquire)  # another request holds the only slot

    resp = client.post("/v1/ask", json=QUESTION)
    assert resp.status_code == 503 and resp.headers["retry-after"] == "2"
    stream = client.post("/v1/ask/stream", json=QUESTION)
    assert stream.status_code == 503  # refused before the event stream starts

    admission.release()
    assert client.post("/v1/ask", json=QUESTION).status_code == 200


def test_slots_are_released_after_success_failure_and_streams(
    app: FastAPI, client: TestClient
) -> None:
    app.state.admission = admission = Admission(max_inflight=1, queue_timeout_s=0.01)
    app.state.answerer = FakeAnswerer()
    client.post("/v1/ask", json=QUESTION)
    app.state.answerer = FailingAnswerer()
    assert client.post("/v1/ask", json=QUESTION).status_code == 503
    app.state.answerer = StreamingAnswerer()
    client.post("/v1/ask/stream", json=QUESTION)
    app.state.answerer = StreamingAnswerer(fail=True)
    client.post("/v1/ask/stream", json=QUESTION)

    app.state.answerer = FakeAnswerer()
    assert client.post("/v1/ask", json=QUESTION).status_code == 200  # the single slot is free
    assert admission._slots._value == 1
