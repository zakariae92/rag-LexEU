"""Admission control: a bound on answers in flight, and a fast 503 when it is reached.

The load test (ADR 0009) showed the CPU embedding server saturating at about 6 answers/s: past
that point every request queues behind it, and p95 grew from 1.6 s to 6.4 s for everyone. Shedding
the excess keeps admitted requests within the latency objective and tells the rest to retry
(503 + Retry-After) after a short wait, instead of letting all of them time out slowly.
"""

import asyncio

from lexeu.observability.metrics import INFLIGHT_ANSWERS, SHED_REQUESTS


class Overloaded(Exception):
    pass


class Admission:
    def __init__(self, max_inflight: int, queue_timeout_s: float) -> None:
        self._slots = asyncio.Semaphore(max_inflight)
        self._timeout = queue_timeout_s

    async def acquire(self) -> None:
        """Wait up to the queue timeout for a slot; raise Overloaded otherwise."""
        try:
            await asyncio.wait_for(self._slots.acquire(), self._timeout)
        except TimeoutError:
            SHED_REQUESTS.inc()
            raise Overloaded from None
        INFLIGHT_ANSWERS.inc()

    def release(self) -> None:
        INFLIGHT_ANSWERS.dec()
        self._slots.release()
