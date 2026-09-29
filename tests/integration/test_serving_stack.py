"""API keys, answer log and rate limiting against the real Postgres and Redis."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from lexeu.core.config import Settings
from lexeu.generation.answer import Answer, Citation
from lexeu.infra.answer_log import AnswerLog
from lexeu.infra.api_keys import ApiKeyStore
from lexeu.infra.rate_limit import RateLimiter

pytestmark = pytest.mark.integration


@pytest.fixture
async def serving_engine(engine: AsyncEngine) -> AsyncEngine:
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE api_keys, answers, feedback RESTART IDENTITY CASCADE"))
    return engine


@pytest.fixture
async def redis(migrated: Settings) -> AsyncIterator[Redis]:
    client = Redis.from_url(migrated.redis.url)
    async for key in client.scan_iter("test-ratelimit:*"):
        await client.delete(key)
    yield client
    await client.aclose()


async def test_key_lifecycle(serving_engine: AsyncEngine) -> None:
    store = ApiKeyStore(serving_engine)
    secret, key = await store.create("alice", 30)

    assert secret.startswith("lx_") and len(secret) > 40
    async with serving_engine.connect() as conn:
        stored: str = (await conn.execute(text("SELECT key_hash FROM api_keys"))).scalar_one()
    assert secret not in stored  # only the hash is stored

    found = await store.authenticate(secret)
    assert found is not None and found.id == key.id
    assert await store.authenticate(secret + "x") is None
    assert await store.revoke("alice")
    assert await store.authenticate(secret) is None
    assert not await store.revoke("alice")  # already revoked


async def test_answer_log_and_feedback(serving_engine: AsyncEngine) -> None:
    log = AnswerLog(serving_engine)
    answer = Answer(
        question="Breach deadline?", lang="en", text="72 hours [1].", refused=False,
        refusal_reason=None, sources=[], model="fake/llm", cost_usd=0.0008,
        citations=[Citation(1, "32016R0679:art_33:p1", "Art. 33(1) GDPR", "en", "c1")],
        timings_ms={"total": 900.0},
    )  # fmt: skip
    await log.record("a" * 32, answer, retrieval="dense-recitals-demote")

    assert await log.feedback("a" * 32, 1, None)
    assert await log.feedback("a" * 32, -1, "changed my mind")  # latest opinion wins
    assert not await log.feedback("b" * 32, 1, None)
    async with serving_engine.connect() as conn:
        rows = (await conn.execute(text("SELECT rating, comment FROM feedback"))).all()
    assert [tuple(r) for r in rows] == [(-1, "changed my mind")]


async def test_sliding_window_limits_and_recovers(redis: Redis) -> None:
    limiter = RateLimiter(redis, prefix="test-ratelimit")
    t0 = 60 * 16_667 + 20.0  # 20 s into a window
    decisions = [await limiter.hit("k1", 3, now=t0) for _ in range(4)]
    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert decisions[0].remaining == 2
    assert decisions[3].retry_after_s == 40  # nothing frees up before the window rolls over

    # Next window, 30 s in: half of the previous 3 still count (1.5), so one more fits.
    later = t0 + 70
    assert [(await limiter.hit("k1", 3, now=later)).allowed for _ in range(2)] == [True, False]
    assert (await limiter.hit("k2", 3, now=t0)).allowed  # counters are per subject


async def test_concurrent_requests_cannot_exceed_the_limit(redis: Redis) -> None:
    limiter = RateLimiter(redis, prefix="test-ratelimit")
    results = await asyncio.gather(*(limiter.hit("burst", 10, now=2_000_010.0) for _ in range(50)))
    assert sum(d.allowed for d in results) == 10  # the Lua script makes check+increment atomic
