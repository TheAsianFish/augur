"""Time is injected, never read directly.

Every datetime in Augur is timezone-aware UTC. Code that needs "now" takes a
:class:`Clock`; backtests pin it to a question's open date.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """The current instant, timezone-aware UTC."""
        ...


class SystemClock:
    """The wall clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """A clock pinned to one instant (backtests and tests)."""

    def __init__(self, instant: datetime) -> None:
        self._instant = ensure_utc(instant)

    def now(self) -> datetime:
        return self._instant

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._instant = self._instant + timedelta(seconds=seconds)


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` in UTC; refuse naive datetimes instead of guessing."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"naive datetime {value!r}: every datetime must carry a timezone")
    return value.astimezone(UTC)


def to_iso(value: datetime) -> str:
    """Fixed-width UTC ISO 8601, so that string order equals time order in SQLite."""
    return ensure_utc(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(text: str) -> datetime:
    """Parse an ISO 8601 timestamp that carries an offset or a trailing ``Z``."""
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return ensure_utc(value)
