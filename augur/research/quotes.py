"""A market price as of an instant: the structured half of a market source.

A :class:`MarketQuote` is what the price-only and blend arms of the replay
evaluation read, and what :mod:`augur.research.markets` turns into a
:class:`~augur.research.source.Source` for the brief. Its ``price_time`` is
the timestamp of the trade, history point or aggregate entry that set the
price, never the moment it was fetched, and it can never be later than
``as_of``: the model refuses to exist otherwise, so a client bug that kept a
later point fails loudly instead of leaking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from augur.clock import ensure_utc

Platform = Literal["polymarket", "manifold", "kalshi", "metaculus"]
Match = Literal["exact", "search"]
"""``exact``: the market *is* the question (a ForecastBench market question,
or a Metaculus question's own aggregate). ``search``: a similar market found
by searching, which may ask something different."""


class MarketQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    platform: Platform
    market_id: str
    title: str
    url: str
    outcome: str
    """The outcome priced; ``Yes`` for a binary market."""
    probability: float
    price_time: datetime
    as_of: datetime
    match: Match
    volume: float | None = None
    closes_at: datetime | None = None

    @field_validator("price_time", "as_of", "closes_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @model_validator(mode="after")
    def _dated_at_or_before_as_of(self) -> Self:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"price {self.probability} is outside [0, 1]")
        if self.price_time > self.as_of:
            raise ValueError(
                f"{self.platform} price for {self.market_id} is dated "
                f"{self.price_time.isoformat()}, after the instant it was asked for "
                f"({self.as_of.isoformat()})"
            )
        return self
