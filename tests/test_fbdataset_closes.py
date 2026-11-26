"""M15's closes ledger: a fake chart server sending closes past each question's freeze instant."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from augur import fbdataset
from augur.clock import FixedClock
from augur.errors import LeakageError
from augur.leakage import Admission, RunMode, admit_source, scan_ledger
from augur.ledger import Ledger
from augur.research.series import Observation, Series, to_source

from .test_series import NOW, chart_body, fetcher, serve

FREEZE = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
"""The 2026-08-30 set's freeze instant: the close of 19 August is the last admissible one."""


def _question(ticker: str = "NVDA") -> fbdataset.DQuestion:
    return fbdataset.DQuestion(
        question_set="2026-08-30",
        source="yfinance",
        id=ticker,
        freeze=FREEZE,
        freeze_value=180.0,
        url=f"https://finance.yahoo.com/quote/{ticker}",
        resolutions=((date(2026, 9, 6), 1), (date(2026, 9, 29), 0)),
    )


def _points() -> list[tuple[date, float | None]]:
    days = [date(2026, 8, 10) + timedelta(days=k) for k in range(10)]  # 10 to 19 August
    points: list[tuple[date, float | None]] = [(d, 100.0 + k) for k, d in enumerate(days)]
    points.append((date(2026, 8, 20), 5000.0))  # dated 21 August 00:00 UTC: after the freeze
    points.append((date(2026, 9, 29), 9999.0))  # the resolution day itself
    return points


async def test_a_close_dated_after_the_freeze_instant_is_refused(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    f, http = fetcher(serve(chart_body(_points()), seen))
    path = tmp_path / "closes.sqlite"
    async with http:
        with Ledger.open(path, FixedClock(NOW)) as ledger:
            report = await fbdataset.record_closes(
                questions=[_question()], fetcher=f, ledger=ledger, code_version="test", out=print
            )
    assert report.recorded == 1
    assert report.refused == 2
    (request,) = seen
    assert "range" not in request.url.params  # an explicit period, not two years from today
    assert int(request.url.params["period1"]) == int(
        (FREEZE - timedelta(days=fbdataset.FETCH_DAYS)).timestamp()
    )
    assert int(request.url.params["period2"]) == int((FREEZE + timedelta(days=1)).timestamp())

    closes = fbdataset.ledger_closes(path)
    (got,) = closes.values()
    assert got[-1] == (date(2026, 8, 19), 109.0)
    assert all(v < 1000 for _, v in got)
    with sqlite3.connect(path) as conn:
        excerpt, published = conn.execute("SELECT excerpt, published_at FROM sources").fetchone()
    assert "5000" not in excerpt
    assert "9999" not in excerpt
    assert published == "2026-08-20T00:00:00.000000Z"
    assert scan_ledger(path) == []


async def test_a_rerun_skips_what_the_ledger_holds(tmp_path: Path) -> None:
    f, http = fetcher(serve(chart_body(_points())))
    path = tmp_path / "closes.sqlite"
    q = _question()
    async with http:
        with Ledger.open(path, FixedClock(NOW)) as ledger:
            await fbdataset.record_closes(
                questions=[q], fetcher=f, ledger=ledger, code_version="test", out=print
            )
    assert fbdataset.recorded_ids(path) == {q.question_id}


def _late_series(last: date) -> Series:
    obs = tuple(
        Observation(
            day=last - timedelta(days=k),
            close=100.0 + k,
            dated_at=datetime.combine(last - timedelta(days=k - 1), datetime.min.time(), UTC),
        )
        for k in range(3, -1, -1)
    )
    return Series(
        symbol="NVDA",
        name="NVIDIA Corporation",
        currency="USD",
        url="https://query1.finance.yahoo.com/v8/finance/chart/NVDA",
        observations=obs,
        refused=0,
        as_of=NOW,
        retrieved_at=NOW,
    )


def _run_with_question(ledger: Ledger, q: fbdataset.DQuestion) -> str:
    run_id = ledger.start_run(
        mode=RunMode.BACKTEST,
        tournament=fbdataset.TOURNAMENT,
        provider="yahoo-chart",
        model="none",
        settings={},
        code_version="test",
    )
    ledger.record_question(run_id, q.to_question(), q.freeze, {"id": q.id})
    return run_id


def test_the_gate_and_the_ledger_refuse_a_source_dated_after_the_freeze(tmp_path: Path) -> None:
    q = _question()
    source = to_source(_late_series(date(2026, 8, 20)), q.id, keep=10)  # dated 21 August
    decision = admit_source(source, q.freeze, RunMode.BACKTEST)
    assert not decision.admitted
    assert "after the cutoff" in decision.reason
    with Ledger.open(tmp_path / "l.sqlite", FixedClock(NOW)) as ledger:
        run_id = _run_with_question(ledger, q)
        ledger.record_source(source)
        forged = Admission(source_id=decision.source_id, admitted=True, reason="forged")
        with pytest.raises(LeakageError):
            ledger.record_admission(run_id, q.question_id, forged)


def test_reading_back_refuses_a_close_after_the_cutoff_even_in_an_admitted_source(
    tmp_path: Path,
) -> None:
    # A source dated at the freeze whose table still carries a later line: the
    # reader checks every close, not only the source's date.
    q = _question()
    series = _late_series(date(2026, 8, 19))
    source = to_source(series, q.id, keep=10)
    tampered = source.model_copy(update={"excerpt": source.excerpt + "\n2026-08-25 777"})
    path = tmp_path / "l.sqlite"
    with Ledger.open(path, FixedClock(NOW)) as ledger:
        run_id = _run_with_question(ledger, q)
        ledger.record_source(tampered)
        ledger.record_admission(
            run_id, q.question_id, admit_source(tampered, q.freeze, RunMode.BACKTEST)
        )
    with pytest.raises(fbdataset.DatasetError, match="after its cutoff"):
        fbdataset.ledger_closes(path)
