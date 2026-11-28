"""M18's DBnomics source: a fake v22 server sending observations past each freeze instant."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from augur import fbseason
from augur.clock import FixedClock
from augur.errors import LeakageError, SearchError
from augur.fbdataset import DQuestion
from augur.leakage import Admission, RunMode, admit_source, scan_ledger
from augur.ledger import Ledger
from augur.ratelimit import RateLimiter
from augur.research.dbnomics import DBnomicsFetcher, DSeries, code_from_id, to_source

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
FREEZE = datetime(2026, 9, 3, 0, 0, tzinfo=UTC)
ID = "meteofrance_TEMPERATURE_celsius.07005.D"
CODE = "meteofrance/TEMPERATURE/celsius.07005.D"
Handler = Callable[[httpx.Request], httpx.Response]


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def fetcher(handler: Handler) -> tuple[DBnomicsFetcher, httpx.AsyncClient]:
    t = FakeTime()
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DBnomicsFetcher(http, FixedClock(NOW), limiter=limiter), http


def body(points: list[tuple[str, Any]], name: str = "Station 07005") -> dict[str, Any]:
    return {
        "series": {
            "docs": [
                {
                    "series_name": name,
                    "period": [p for p, _ in points],
                    "value": [v for _, v in points],
                }
            ]
        }
    }


def serve(doc: dict[str, Any], seen: list[httpx.Request] | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=doc)

    return handler


def _question(qid: str = ID) -> DQuestion:
    return DQuestion(
        question_set="2026-09-13",
        source="dbnomics",
        id=qid,
        freeze=FREEZE,
        freeze_value=15.0,
        url="https://db.nomics.world/meteofrance/TEMPERATURE",
        resolutions=((date(2026, 9, 20), 1),),
    )


def _server_points() -> list[tuple[str, Any]]:
    pts: list[tuple[str, Any]] = [
        (f"2025-09-{d:02d}", 10.0 + d / 10) for d in range(10, 17)
    ]  # the due-date window of 2025
    pts += [(f"2025-09-{d:02d}", 12.0 + d / 10) for d in range(17, 24)]  # r's window
    pts += [
        ("2026-09-02", 777.0),  # dated 3 September 00:00: admitted, but no arm reads this year
        ("2026-09-03", 555.0),  # dated 4 September: after the freeze
        ("2026-09-20", 9999.0),  # the resolution day itself
    ]
    return pts


async def test_an_observation_dated_after_the_freeze_instant_is_refused(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    f, http = fetcher(serve(body(_server_points()), seen))
    path = tmp_path / "dbnomics.sqlite"
    q = _question()
    async with http:
        with Ledger.open(path, FixedClock(NOW)) as ledger:
            report = await fbseason.record_observations(
                todo=[(q, [date(2026, 9, 20)])],
                fetcher=f,
                ledger=ledger,
                code_version="test",
                out=lambda _: None,
            )
    assert report.recorded == 1
    assert report.stations == 1
    assert report.skipped == {}
    assert report.refused == 2  # 3 September and 20 September
    (request,) = seen
    assert request.url.path == f"/v22/series/{CODE}"
    assert request.url.params["observations"] == "1"

    with sqlite3.connect(path) as conn:
        excerpt, published = conn.execute("SELECT excerpt, published_at FROM sources").fetchone()
    for sentinel in ("9999", "555", "777"):
        assert sentinel not in excerpt
    assert "2026-09-03" not in excerpt.split("\n", 1)[1]
    assert "2026-09-20" not in excerpt.split("\n", 1)[1]
    assert published.startswith("2025-09-24T00:00:00")  # the last kept day, 23 September

    got = fbseason.ledger_observations(path)
    (obs,) = got.values()
    assert obs
    assert max(obs) <= date(2026, 9, 2)
    assert max(obs) == date(2025, 9, 23)
    assert obs[date(2025, 9, 13)] == pytest.approx(11.3)
    assert scan_ledger(path) == []

    assert fbseason.recorded_ids(path) == {q.question_id}


async def test_a_rerun_can_see_what_the_ledger_holds(tmp_path: Path) -> None:
    f, http = fetcher(serve(body(_server_points())))
    path = tmp_path / "dbnomics.sqlite"
    q = _question()
    assert fbseason.recorded_ids(path) == set()
    async with http:
        with Ledger.open(path, FixedClock(NOW)) as ledger:
            await fbseason.record_observations(
                todo=[(q, [date(2026, 9, 20)])],
                fetcher=f,
                ledger=ledger,
                code_version="test",
                out=lambda _: None,
            )
    held = fbseason.recorded_ids(path)
    assert held == {q.question_id}
    todo = [t for t in [(q, [date(2026, 9, 20)])] if t[0].question_id not in held]
    assert todo == []


async def test_a_failed_lookup_is_skipped_not_recorded(tmp_path: Path) -> None:
    f, http = fetcher(lambda r: httpx.Response(404))
    path = tmp_path / "dbnomics.sqlite"
    async with http:
        with Ledger.open(path, FixedClock(NOW)) as ledger:
            report = await fbseason.record_observations(
                todo=[(_question(), [date(2026, 9, 20)])],
                fetcher=f,
                ledger=ledger,
                code_version="test",
                out=lambda _: None,
            )
    assert report.recorded == 0
    assert report.stations == 0
    assert sum(report.skipped.values()) == 1


async def test_fetch_keeps_admitted_days_and_drops_na_and_later_days() -> None:
    pts: list[tuple[str, Any]] = [
        ("2026-09-01", 20.5),
        ("2026-09-02", "NA"),
        ("2026-09-03", 21.0),
        ("2026-09-04", 22.0),
        ("2026-09", 1.0),  # not a daily period
    ]
    f, http = fetcher(serve(body(pts)))
    async with http:
        s = await f.fetch(CODE, datetime(2026, 9, 4, 0, 0, tzinfo=UTC))
    assert [p.day for p in s.points] == [date(2026, 9, 1), date(2026, 9, 3)]
    assert s.points[0].dated_at == datetime(2026, 9, 2, tzinfo=UTC)
    assert s.refused == 1  # 4 September, dated the 5th
    assert s.name == "Station 07005"
    assert s.url == f"https://api.db.nomics.world/v22/series/{CODE}?observations=1"
    assert s.retrieved_at == NOW


async def test_retry_after_429_then_success() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=body([("2026-09-01", 20.0)]))

    f, http = fetcher(handler)
    async with http:
        s = await f.fetch(CODE, FREEZE)
    assert calls == 2
    assert len(s.points) == 1


async def test_persistent_429_raises_search_error() -> None:
    f, http = fetcher(lambda r: httpx.Response(429))
    async with http:
        with pytest.raises(SearchError, match="failed after 3 attempts"):
            await f.fetch(CODE, FREEZE)


async def test_non_200_raises_search_error() -> None:
    f, http = fetcher(lambda r: httpx.Response(404))
    async with http:
        with pytest.raises(SearchError, match="HTTP 404"):
            await f.fetch(CODE, FREEZE)


async def test_a_body_without_one_series_raises() -> None:
    f, http = fetcher(serve({"series": {"docs": []}}))
    async with http:
        with pytest.raises(SearchError, match="no single series"):
            await f.fetch(CODE, FREEZE)


def _series() -> DSeries:
    return DSeries(
        code=CODE,
        name="Station",
        url="https://example.test/s",
        points=tuple(_point(date(2026, 9, 1) + timedelta(days=k), float(k)) for k in range(4)),
        refused=1,
        as_of=datetime(2026, 9, 6, tzinfo=UTC),
        retrieved_at=NOW,
    )


def _point(day: date, value: float) -> Any:
    from augur.research.dbnomics import Point

    return Point(day, value, datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC))


def test_until_moves_later_points_to_refused() -> None:
    s = _series()
    earlier = s.until(datetime(2026, 9, 3, tzinfo=UTC))  # keeps days 1 and 2 September
    assert [p.day for p in earlier.points] == [date(2026, 9, 1), date(2026, 9, 2)]
    assert earlier.refused == 1 + 2
    assert earlier.as_of == datetime(2026, 9, 3, tzinfo=UTC)
    assert s.until(s.as_of).points == s.points


def test_until_refuses_a_later_as_of() -> None:
    with pytest.raises(SearchError, match="not later"):
        _series().until(datetime(2026, 9, 7, tzinfo=UTC))


def test_code_from_id() -> None:
    assert code_from_id(ID) == CODE
    for bad in ("meteofrance", "meteofrance_TEMPERATURE", "a__b", "_b_c"):
        with pytest.raises(SearchError, match="not a DBnomics"):
            code_from_id(bad)


def test_to_source_refuses_when_no_wanted_day_is_admitted() -> None:
    with pytest.raises(SearchError, match="no observation"):
        to_source(_series(), "q", [date(2020, 1, 1)])


def _tampered_run(path: Path, excerpt_suffix: str) -> None:
    q = _question()
    s = _series().until(FREEZE)
    source = to_source(s, q.series, [date(2026, 9, 1), date(2026, 9, 2)])
    tampered = source.model_copy(update={"excerpt": source.excerpt + excerpt_suffix})
    with Ledger.open(path, FixedClock(NOW)) as ledger:
        run_id = ledger.start_run(
            mode=RunMode.BACKTEST,
            tournament=fbseason.TOURNAMENT,
            provider="dbnomics-v22",
            model="none",
            settings={},
            code_version="test",
        )
        lq = fbseason.ledger_question(q)
        ledger.record_question(run_id, lq, q.freeze, {"id": q.id})
        ledger.record_source(tampered)
        ledger.record_admission(
            run_id, lq.question_id, admit_source(tampered, q.freeze, RunMode.BACKTEST)
        )


def test_reading_back_refuses_a_line_dated_after_the_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite"
    _tampered_run(path, "\n2026-09-05 777")
    with pytest.raises(fbseason.SeasonError, match="after its cutoff"):
        fbseason.ledger_observations(path)


def test_reading_back_accepts_the_untampered_source(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite"
    _tampered_run(path, "")
    (obs,) = fbseason.ledger_observations(path).values()
    assert obs == {date(2026, 9, 1): 0.0, date(2026, 9, 2): 1.0}


def test_the_ledger_refuses_a_forged_admission_of_a_late_source(tmp_path: Path) -> None:
    q = _question()
    late = _series()  # as_of 6 September, last point dated 5 September: after the freeze
    source = to_source(late, q.series, [date(2026, 9, 4)])
    decision = admit_source(source, q.freeze, RunMode.BACKTEST)
    assert not decision.admitted
    with Ledger.open(tmp_path / "l.sqlite", FixedClock(NOW)) as ledger:
        run_id = ledger.start_run(
            mode=RunMode.BACKTEST,
            tournament=fbseason.TOURNAMENT,
            provider="dbnomics-v22",
            model="none",
            settings={},
            code_version="test",
        )
        lq = fbseason.ledger_question(q)
        ledger.record_question(run_id, lq, q.freeze, {"id": q.id})
        ledger.record_source(source)
        forged = Admission(source_id=decision.source_id, admitted=True, reason="forged")
        with pytest.raises(LeakageError):
            ledger.record_admission(run_id, lq.question_id, forged)
