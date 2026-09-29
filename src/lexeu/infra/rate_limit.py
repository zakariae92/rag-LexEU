"""Per-key rate limiting in Redis: a sliding-window counter.

A fixed one-minute window lets a client send twice its limit around the minute boundary. The
sliding-window counter weights the previous window by how much of it still overlaps the last 60 s:
two counters per key, O(1) memory, accurate to a few percent. The check-and-increment runs as one
Lua script (atomic in Redis, so concurrent requests cannot both slip through), and the state lives
in Redis, so the limit holds across API replicas.
"""

import math
import time
from dataclasses import dataclass

from redis.asyncio import Redis

WINDOW_S = 60

# KEYS: current window, previous window. ARGV: limit, fraction of the current window elapsed,
# window length. Returns {allowed (0/1), estimated count before this request}.
_SLIDING_WINDOW = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local previous = tonumber(redis.call('GET', KEYS[2]) or '0')
local estimated = previous * (1 - tonumber(ARGV[2])) + current
if estimated + 1 > tonumber(ARGV[1]) then
  return {0, tostring(estimated), previous, current}
end
redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]) * 2)
return {1, tostring(estimated), previous, current}
"""


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_s: int  # 0 when allowed; a hint, clients get a fresh one if still limited


class RateLimiter:
    def __init__(self, redis: Redis, prefix: str = "ratelimit") -> None:
        self._prefix = prefix
        self._script = redis.register_script(_SLIDING_WINDOW)

    async def hit(self, subject: str, limit: int, now: float | None = None) -> RateDecision:
        now = time.time() if now is None else now
        window = int(now // WINDOW_S)
        elapsed = (now % WINDOW_S) / WINDOW_S
        keys = [f"{self._prefix}:{subject}:{window}", f"{self._prefix}:{subject}:{window - 1}"]
        allowed, estimated_s, previous, current = await self._script(
            keys=keys, args=[limit, elapsed, WINDOW_S]
        )
        estimated = float(estimated_s)
        if allowed:
            return RateDecision(True, limit, max(0, math.floor(limit - estimated - 1)), 0)

        if int(current) + 1 > limit or not int(previous):
            wait = (1 - elapsed) * WINDOW_S  # nothing frees up before the window rolls over
        else:  # the previous window's weight decays linearly: solve for one free slot
            wait = (estimated + 1 - limit) / int(previous) * WINDOW_S
        return RateDecision(False, limit, 0, max(1, math.ceil(round(wait, 6))))  # 40.000001 -> 40
