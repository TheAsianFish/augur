"""M12's statistical arm in a shadow tick: the ``stat`` and ``blend`` reference rows,
the dated ``series`` source behind them, and what happens when the arm has nothing."""

from __future__ import annotations

import asyncio
import io
import json
import math
import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from augur import cli
from augur.clock import FixedClock
from augur.config import Settings
from augur.distributions import Scaling, mix_cdfs, quantile, validate_cdf
from augur.leakage import scan_ledger
from augur.ledger import Ledger
from augur.llm.endpoints import PRESETS, Route
from augur.llm.router import Router
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter
from augur.replay import BLEND_WEIGHT
from augur.research.series import SeriesFetcher
from augur.runner import ForecastRun, RunReport
from augur.statarm import BaseRates
from augur.statlive import LiveStatArm

from .conftest import T0
from .fakes.llm import ScriptedProvider
from .fakes.metaculus import FAKE_TOKEN, NUMERIC_DETAIL, FakeMetaculus, binary_post, load_fixture
from .fakes.search import FixedSearcher
from .helpers import free_catalog

ROUTE = Route(endpoint=PRESETS["openrouter"], model="nvidia/nemotron-3-ultra-550b-a55b:free")
NY = -5 * 3600
NVDA_TITLE = "What will Nvidia's (NVDA) closing share price be on Friday, November 27, 2026?"
DRONES_TITLE = "How many drones will Russia launch at Ukraine overnight on 26–27 November 2026?"
ABSURD = 99999.0
LAST_ADMITTED = date(2026, 11, 16)
"""T0 is 17 November 18:00 UTC: the close of the 16th (dated the 17th 00:00) is the
last one admitted; the 17th's is dated the 18th 00:00, after the cutoff."""
RATES = BaseRates.fit([("metaculus", 1)] * 6 + [("metaculus", 0)] * 24, fitted_on="test")
"""Metaculus: 6 of 30, so (6 + 1) / (30 + 2)."""

Handler = Callable[[httpx.Request], httpx.Response]


# -- the fakes ---------------------------------------------------------------------


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _weekdays(end: date, count: int) -> list[date]:
    days: list[date] = []
    d = end
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return days[::-1]


def _closes() -> list[tuple[date, float]]:
    """About two years of weekday closes near 180 ending on the 16th, then two absurd
    closes on days whose closes are dated after the cutoff."""
    days = _weekdays(LAST_ADMITTED, 500)
    points = [(d, round(180.0 * math.exp(0.08 * math.sin(i / 7.0)), 4)) for i, d in enumerate(days)]
    return [*points, (date(2026, 11, 17), ABSURD), (date(2026, 11, 18), ABSURD)]


def _stamp(day: date) -> int:
    """A daily bar's timestamp: 09:30 New York time on ``day``."""
    return int(datetime(day.year, day.month, day.day, 9, 30, tzinfo=UTC).timestamp()) - NY


def _chart_body(points: list[tuple[date, float]]) -> dict[str, Any]:
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "symbol": "NVDA",
                        "gmtoffset": NY,
                        "longName": "NVIDIA Corporation",
                    },
                    "timestamp": [_stamp(d) for d, _ in points],
                    "indicators": {"quote": [{"close": [c for _, c in points]}]},
                }
            ],
            "error": None,
        }
    }


class FakeYahoo:
    """The search and chart endpoints for NVDA; every request recorded."""

    def __init__(self, *, status: int = 200, raise_error: bool = False) -> None:
        self.status = status
        self.raise_error = raise_error
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_error:
            raise httpx.ConnectError("yahoo unreachable", request=request)
        if self.status != 200:
            return httpx.Response(self.status, headers={"Retry-After": "1"})
        if request.url.path == "/v1/finance/search":
            quotes = [{"symbol": "NVDA.TO", "quoteType": "EQUITY"}]
            quotes.append({"symbol": "NVDA", "quoteType": "EQUITY"})
            return httpx.Response(200, json={"quotes": quotes, "news": []})
        if request.url.path == "/v8/finance/chart/NVDA":
            return httpx.Response(200, json=_chart_body(_closes()))
        return httpx.Response(404, json={"chart": {"result": None, "error": "not found"}})


def numeric_post(
    post_id: int, question_id: int, title: str, range_min: float, range_max: float
) -> dict[str, Any]:
    """The real numeric fixture, retitled, rescaled and closing the day after T0's week."""
    post = load_fixture(NUMERIC_DETAIL)
    post["id"] = post_id
    post["title"] = title
    q = post["question"]
    q["id"] = question_id
    q["post_id"] = post_id
    q["title"] = title
    q["short_title"] = title
    q["unit"] = ""
    q["open_time"] = "2026-11-10T00:00:00Z"
    q["scheduled_close_time"] = "2026-11-27T12:00:00Z"
    q["scheduled_resolve_time"] = "2026-11-28T00:00:00Z"
    post["scheduled_close_time"] = q["scheduled_close_time"]
    post["scheduled_resolve_time"] = q["scheduled_resolve_time"]
    step = (range_max - range_min) / 200
    q["scaling"] = {
        "continuous_range": [range_min + step * i for i in range(201)],
        "inbound_outcome_count": 200,
        "nominal_min": range_min,
        "nominal_max": range_max,
        "open_lower_bound": True,
        "open_upper_bound": True,
        "range_min": range_min,
        "range_max": range_max,
        "zero_point": None,
    }
    q["open_lower_bound"] = True
    q["open_upper_bound"] = True
    return post


def _client(http: httpx.AsyncClient) -> MetaculusClient:
    return MetaculusClient(
        "https://www.metaculus.com/api",
        SecretStr(FAKE_TOKEN),
        http,
        RateLimiter(1000, burst=1000, max_concurrent=4),
    )


async def _tick(
    tmp_path: Path,
    server: FakeMetaculus,
    yahoo: FakeYahoo | None,
    *,
    stat: bool = True,
    timeout_s: float = 60.0,
) -> tuple[RunReport, Path, str]:
    """One shadow tick with the statistical arm on a fresh ledger."""
    clock = FixedClock(T0)
    path = tmp_path / "run.sqlite"
    out = io.StringIO()
    provider = ScriptedProvider()
    t = FakeTime()
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    handler: Handler = yahoo.handle if yahoo is not None else FakeYahoo().handle
    async with (
        server.client() as http,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as yahoo_http,
    ):
        arm = (
            LiveStatArm(
                rates=RATES,
                fetcher=SeriesFetcher(yahoo_http, clock, limiter=limiter),
                timeout_s=timeout_s,
            )
            if stat
            else None
        )
        with Ledger.open(path, clock) as ledger:
            run = ForecastRun(
                settings=Settings.model_validate(
                    {"provider": "replay", "model": "qwen2.5:7b", "ledger_path": path}
                ),
                clock=clock,
                ledger=ledger,
                router=Router.single(ROUTE, provider, free_catalog(ROUTE), clock),
                searchers=[FixedSearcher()],
                client=_client(http),
                out=out,
                probe_providers={"openrouter": provider},
                stat_arm=arm,
            )
            report = await run.run(
                "minibench", submit=False, samples=None, max_questions=10, shadow=frozenset()
            )
    return report, path, out.getvalue()


# -- ledger reads ------------------------------------------------------------------


def _rows(path: Path, sql: str, *args: Any) -> list[sqlite3.Row]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def _references(path: Path, question_id: int) -> dict[str, sqlite3.Row]:
    rows = _rows(
        path,
        "SELECT * FROM reference_forecasts "
        "WHERE question_id = ? AND forecaster IN ('stat', 'blend')",
        question_id,
    )
    by: dict[str, sqlite3.Row] = {}
    for row in rows:
        assert row["forecaster"] not in by, f"two {row['forecaster']} rows"
        by[row["forecaster"]] = row
    return by


def _aggregate(path: Path, question_id: int) -> sqlite3.Row:
    (row,) = _rows(
        path,
        "SELECT * FROM forecasts WHERE question_id = ? AND kind = 'aggregate'",
        question_id,
    )
    return row


def _series_uses(path: Path, question_id: int) -> list[sqlite3.Row]:
    return _rows(
        path,
        "SELECT s.*, u.admitted FROM source_uses u JOIN sources s USING (source_id) "
        "WHERE u.question_id = ? AND s.searcher = 'series'",
        question_id,
    )


def _outcome(path: Path, question_id: int) -> sqlite3.Row:
    (row,) = _rows(path, "SELECT * FROM question_outcomes WHERE question_id = ?", question_id)
    return row


def _values(row: sqlite3.Row) -> dict[str, Any]:
    data = json.loads(row["values_json"])
    assert isinstance(data, dict)
    return data


SCALING = Scaling(range_min=50.0, range_max=400.0, open_lower_bound=True, open_upper_bound=True)


# -- the tests ---------------------------------------------------------------------


async def test_a_series_question_gets_a_dated_source_a_stat_row_and_its_blend(
    tmp_path: Path,
) -> None:
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(numeric_post(600, 601, NVDA_TITLE, 50.0, 400.0), "minibench")
    yahoo = FakeYahoo()
    report, path, printed = await _tick(tmp_path, server, yahoo)
    assert report.count("forecast") == 1
    assert server.forecasts == []

    # One series source, admitted for the question, with no close past the cutoff.
    (source,) = _series_uses(path, 601)
    assert source["admitted"] == 1
    assert "99999" not in source["excerpt"]
    table = source["excerpt"].splitlines()[1:]
    assert len(table) == 500
    assert table[-1].split()[0] == LAST_ADMITTED.isoformat()
    assert f"{LAST_ADMITTED.isoformat()} " in source["excerpt"]
    assert source["published_at"].startswith("2026-11-17T00:00:00")
    assert [r.url.path for r in yahoo.requests] == ["/v1/finance/search", "/v8/finance/chart/NVDA"]

    refs = _references(path, 601)
    assert set(refs) == {"stat", "blend"}
    stat_cdf = _values(refs["stat"])["cdf"]
    assert validate_cdf(stat_cdf, SCALING) == []
    last_close = _closes()[-3][1]
    median = quantile(stat_cdf, SCALING, 0.5)
    assert abs(median - last_close) / last_close < 0.15, (median, last_close)
    assert refs["stat"]["detail"].startswith("series: NVIDIA Corporation (NVDA)")

    augur_cdf = _values(_aggregate(path, 601))["cdf"]
    expected = mix_cdfs([augur_cdf, stat_cdf], [BLEND_WEIGHT, 1.0 - BLEND_WEIGHT])
    blend_cdf = _values(refs["blend"])["cdf"]
    assert abs(BLEND_WEIGHT - 1 / 3) < 1e-12
    assert len(blend_cdf) == len(expected)
    assert max(abs(a - b) for a, b in zip(blend_cdf, expected, strict=True)) < 1e-9
    assert validate_cdf(blend_cdf, SCALING) == []
    assert refs["blend"]["detail"] == "one-third Augur, two-thirds series"
    assert refs["blend"]["as_of"].startswith("2026-11-17T18:00:00")
    assert "stat arm (series)" in printed
    assert scan_ledger(path) == []


async def test_a_binary_question_without_a_price_gets_the_base_rate_and_augur_unchanged(
    tmp_path: Path,
) -> None:
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(binary_post(610, 611, title="Will it rain?"), "minibench")
    yahoo = FakeYahoo()
    report, path, _ = await _tick(tmp_path, server, yahoo)
    assert report.count("forecast") == 1
    refs = _references(path, 611)
    assert refs["stat"]["probability"] == (6 + 1) / (30 + 2)
    assert "metaculus base rate, 6 of 30" in refs["stat"]["detail"]
    aggregate = _aggregate(path, 611)["probability"]
    assert refs["blend"]["probability"] == aggregate
    assert "unchanged" in refs["blend"]["detail"]
    assert "base-rate" in refs["blend"]["detail"]
    assert yahoo.requests == []
    assert _series_uses(path, 611) == []
    assert scan_ledger(path) == []


async def test_a_numeric_question_the_series_rule_misses_gets_only_an_unchanged_blend(
    tmp_path: Path,
) -> None:
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(numeric_post(620, 621, DRONES_TITLE, 0.0, 1000.0), "minibench")
    yahoo = FakeYahoo()
    report, path, _ = await _tick(tmp_path, server, yahoo)
    assert report.count("forecast") == 1
    refs = _references(path, 621)
    assert set(refs) == {"blend"}
    assert _values(refs["blend"])["cdf"] == _values(_aggregate(path, 621))["cdf"]
    assert "series rule does not reach" in refs["blend"]["detail"]
    assert _series_uses(path, 621) == []
    assert yahoo.requests == []


async def test_a_failing_series_lookup_degrades_to_augur_unchanged(tmp_path: Path) -> None:
    for name, yahoo in (("503", FakeYahoo(status=503)), ("raise", FakeYahoo(raise_error=True))):
        server = FakeMetaculus(clock=FixedClock(T0))
        server.add_post(numeric_post(630, 631, NVDA_TITLE, 50.0, 400.0), "minibench")
        workdir = tmp_path / name
        workdir.mkdir()
        report, path, _ = await _tick(workdir, server, yahoo)
        assert report.count("forecast") == 1, name
        outcome = _outcome(path, 631)
        assert outcome["status"] == "forecast"
        assert outcome["submitted"] == 0
        refs = _references(path, 631)
        assert set(refs) == {"blend"}, name
        assert _values(refs["blend"])["cdf"] == _values(_aggregate(path, 631))["cdf"]
        assert "unchanged" in refs["blend"]["detail"]
        assert "series lookup failed" in refs["blend"]["detail"]
        assert len(yahoo.requests) == 3, "the search is tried three times, then the arm abstains"
        assert _series_uses(path, 631) == []
        assert server.forecasts == []
        assert server.comments == []


async def test_without_the_arm_no_stat_or_blend_row_is_written(tmp_path: Path) -> None:
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(numeric_post(640, 641, NVDA_TITLE, 50.0, 400.0), "minibench")
    server.add_post(binary_post(650, 651, title="Will it rain?"), "minibench")
    yahoo = FakeYahoo()
    report, path, printed = await _tick(tmp_path, server, yahoo, stat=False)
    assert report.count("forecast") == 2
    assert _rows(path, "SELECT * FROM reference_forecasts") == []
    assert _rows(path, "SELECT * FROM sources WHERE searcher = 'series'") == []
    assert yahoo.requests == []
    assert "stat arm" not in printed


async def test_a_slow_series_lookup_is_cut_off_and_the_blend_still_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hang(self: SeriesFetcher, match: object) -> str | None:
        await asyncio.sleep(30)
        return None

    monkeypatch.setattr(SeriesFetcher, "resolve", hang)
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(numeric_post(640, 641, NVDA_TITLE, 50.0, 400.0), "minibench")
    report, path, out = await _tick(tmp_path, server, FakeYahoo(), timeout_s=0.05)
    assert report.count("forecast") == 1
    refs = _references(path, 641)
    assert set(refs) == {"blend"}
    assert "longer than 0.05 s" in refs["blend"]["detail"]
    assert _values(refs["blend"])["cdf"] == _values(_aggregate(path, 641))["cdf"]
    assert "stat arm (abstains)" in out


def test_the_arm_never_runs_beside_submit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METACULUS_TOKEN", FAKE_TOKEN)
    code = cli.main(["forecast", "--submit", "--stat-arm", "--ledger", str(tmp_path / "l.sqlite")])
    assert code != 0
    assert "--stat-arm runs only on unposted runs" in capsys.readouterr().err
    assert not (tmp_path / "l.sqlite").exists()
