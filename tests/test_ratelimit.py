from __future__ import annotations

import asyncio

from augur.ratelimit import RateLimiter


class FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


def make(rate: float, burst: int = 1, conc: int = 1) -> tuple[RateLimiter, FakeTime]:
    t = FakeTime()
    return RateLimiter(rate, burst, conc, monotonic=t.monotonic, sleep=t.sleep), t


async def test_rate_limits_acquisitions() -> None:
    limiter, t = make(rate=2.0, burst=3)
    start = t.now
    n = 11
    for _ in range(n):
        async with limiter.acquire():
            pass
    assert t.now - start >= (n - 3) / 2.0 - 1e-9


async def test_concurrency_cap() -> None:
    limiter, _ = make(rate=1000.0, burst=100, conc=2)
    current = 0
    peak = 0

    async def worker() -> None:
        nonlocal current, peak
        async with limiter.acquire():
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            current -= 1

    await asyncio.gather(*(worker() for _ in range(8)))
    assert peak == 2


async def test_pause_delays_next_acquisition() -> None:
    limiter, t = make(rate=100.0, burst=5)
    start = t.now
    limiter.pause(7.0)
    async with limiter.acquire():
        pass
    assert t.now - start >= 7.0
