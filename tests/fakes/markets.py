"""A fake of the public market APIs the ``markets`` searcher reads.

One handler answers Manifold, Polymarket (gamma and CLOB) and Kalshi. Each
platform holds one market whose price history has points before *and after*
the instant asked for, and with ``ignore_bounds`` (the default) the fake
returns every point whatever the request's time bounds say: a server that
misbehaves exactly the way a leak would need.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

AS_OF = datetime(2026, 6, 11, tzinfo=UTC)
BEFORE = AS_OF - timedelta(hours=2)
AFTER = AS_OF + timedelta(hours=1)

MANIFOLD_ID = "mf1"
POLY_CONDITION = "0xabc"
POLY_YES_TOKEN = "111"
KALSHI_TICKER = "KXTEST-26"
KALSHI_SERIES = "KXTEST"


def _ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def _json(body: Any) -> httpx.Response:
    """A JSON reply that may carry NaN, as a non-compliant server's can (httpx refuses it)."""
    return httpx.Response(
        200, content=json.dumps(body), headers={"content-type": "application/json"}
    )


class FakeMarkets:
    def __init__(
        self, *, ignore_bounds: bool = True, only_late: bool = False, nan_point: bool = False
    ) -> None:
        self.ignore_bounds = ignore_bounds
        self.only_late = only_late
        self.nan_point = nan_point
        self.requests: list[httpx.Request] = []

    # Each platform's history: (time, price). The later point is the answer.
    def _points(self) -> list[tuple[datetime, float]]:
        early = [(BEFORE - timedelta(days=1), 0.30), (BEFORE, 0.40)]
        if self.nan_point:  # the latest point before the cutoff has no number
            early.append((BEFORE + timedelta(hours=1), math.nan))
        late = [(AFTER, 0.97)]
        return late if self.only_late else early + late

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host, path, params = request.url.host, request.url.path, request.url.params
        if host == "api.manifold.markets":
            return self._manifold(path, params)
        if host == "gamma-api.polymarket.com":
            return self._gamma(path, params)
        if host == "clob.polymarket.com":
            return self._clob(params)
        if host == "api.elections.kalshi.com":
            return self._kalshi(path, params)
        return httpx.Response(404)

    def _manifold(self, path: str, params: httpx.QueryParams) -> httpx.Response:
        if path.endswith("/bets"):
            assert params["contractId"] == MANIFOLD_ID
            cutoff = int(params["beforeTime"])
            bets = [
                {"createdTime": _ms(t), "probAfter": p, "isCancelled": False}
                for t, p in self._points()
                if self.ignore_bounds or _ms(t) < cutoff
            ]
            return _json(sorted(bets, key=lambda b: -b["createdTime"]))
        if path.endswith("/search-markets"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": MANIFOLD_ID,
                        "question": "Will the test event happen in 2026?",
                        "url": "https://manifold.markets/u/test-event",
                        "createdTime": _ms(AS_OF - timedelta(days=30)),
                    },
                    {
                        "id": "closed-earlier",
                        "question": "Will the test event happen in 2026?",
                        "createdTime": _ms(AS_OF - timedelta(days=900)),
                        "closeTime": _ms(AS_OF - timedelta(days=500)),
                    },
                    {
                        "id": "born-later",
                        "question": "Will the test event happen in 2026?",
                        "createdTime": _ms(AFTER),
                    },
                ],
            )
        return httpx.Response(404)

    def _poly_market(self) -> dict[str, Any]:
        return {
            "conditionId": POLY_CONDITION,
            "question": "Will the test event happen in 2026?",
            "slug": "test-event",
            "clobTokenIds": json.dumps([POLY_YES_TOKEN, "222"]),
            "outcomes": json.dumps(["Yes", "No"]),
            "volumeNum": 1234.0,
            "endDate": "2026-12-31T00:00:00Z",
            "startDate": "2026-01-01T00:00:00Z",
        }

    def _gamma(self, path: str, params: httpx.QueryParams) -> httpx.Response:
        if path == "/markets":
            found = params["condition_ids"] == POLY_CONDITION and params["closed"] == "true"
            return httpx.Response(200, json=[self._poly_market()] if found else [])
        if path == "/public-search":
            return httpx.Response(200, json={"events": [{"markets": [self._poly_market()]}]})
        return httpx.Response(404)

    def _clob(self, params: httpx.QueryParams) -> httpx.Response:
        assert params["market"] == POLY_YES_TOKEN
        end = int(params["endTs"])
        history = [
            {"t": int(t.timestamp()), "p": p}
            for t, p in self._points()
            if self.ignore_bounds or t.timestamp() <= end
        ]
        return _json({"history": history})

    def _kalshi(self, path: str, params: httpx.QueryParams) -> httpx.Response:
        market = {
            "ticker": KALSHI_TICKER,
            "event_ticker": "KXTEST-EV",
            "title": "Will the test event happen in 2026?",
            "open_time": "2026-01-01T00:00:00Z",
            "close_time": "2026-12-31T00:00:00Z",
            "yes_sub_title": "",
        }
        if path.endswith(f"/markets/{KALSHI_TICKER}/candlesticks"):
            end = int(params["end_ts"])
            candles = [
                {"end_period_ts": int(t.timestamp()), "price": {"close_dollars": f"{p:.4f}"}}
                for t, p in self._points()
                if self.ignore_bounds or t.timestamp() <= end
            ]
            return httpx.Response(200, json={"candlesticks": candles})
        if path.endswith(f"/markets/{KALSHI_TICKER}"):
            return httpx.Response(200, json={"market": market})
        if path.endswith("/events/KXTEST-EV"):
            return httpx.Response(200, json={"event": {"series_ticker": KALSHI_SERIES}})
        if path.endswith("/events"):
            event = {
                "title": "Test event in 2026",
                "series_ticker": KALSHI_SERIES,
                "markets": [market],
            }
            return httpx.Response(200, json={"events": [event], "cursor": ""})
        return httpx.Response(404)


class FakeMetaculus:
    """``get_post`` with an aggregate history that has an entry after ``AS_OF``."""

    def __init__(self, *, hidden: bool = False) -> None:
        self.hidden = hidden

    async def get_post(self, post_id: int) -> dict[str, Any]:
        history = [
            {
                "start_time": BEFORE.timestamp() - 86400,
                "end_time": BEFORE.timestamp(),
                "centers": [0.2],
            },
            {"start_time": BEFORE.timestamp(), "end_time": AFTER.timestamp(), "centers": [0.35]},
            {"start_time": AFTER.timestamp(), "end_time": None, "centers": [0.99]},
        ]
        return {
            "id": post_id,
            "title": "Will the test event happen in 2026?",
            "question": {
                "id": post_id + 1,
                "type": "binary",
                "aggregations": {"recency_weighted": {"history": [] if self.hidden else history}},
            },
        }
