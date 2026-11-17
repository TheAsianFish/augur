"""An async token bucket with a concurrency cap.

Every provider and API call goes through one of these. ``acquire`` waits for
both a token (rate) and a slot (concurrency); the slot is released on exit.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager


class RateLimiter:
    def __init__(
        self,
        rate_per_second: float,
        burst: int = 1,
        max_concurrent: int = 1,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate_per_second <= 0 or burst < 1 or max_concurrent < 1:
            raise ValueError("rate must be positive and burst and concurrency at least 1")
        self._rate = rate_per_second
        self._burst = float(burst)
        self._tokens = float(burst)
        self._monotonic = monotonic
        self._sleep = sleep
        self._last = monotonic()
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max_concurrent)
        self._paused_until = 0.0

    def pause(self, seconds: float) -> None:
        """Hold every caller for ``seconds`` (a server's ``Retry-After``)."""
        self._paused_until = max(self._paused_until, self._monotonic() + seconds)

    async def _take_token(self) -> None:
        async with self._lock:
            while True:
                now = self._monotonic()
                if now < self._paused_until:
                    await self._sleep(self._paused_until - now)
                    continue
                self._tokens = min(self._burst, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await self._sleep((1.0 - self._tokens) / self._rate)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        async with self._slots:
            await self._take_token()
            yield
