"""``run_forecastbench`` with a scripted model: what the ledger holds afterwards."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from augur.backtest import PinnedClock, done_questions
from augur.forecastbench import load_set
from augur.leakage import scan_ledger
from augur.llm.endpoints import PRESETS, Route

from .fakes.forecastbench import (
    QUESTION_SET,
    RELEASED,
    raw_question,
    raw_resolution,
    run_backtest,
    small_set,
    write_pair,
)
from .helpers import catalog_of


def _rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(path)) as conn:
        return [tuple(r) for r in conn.execute(sql)]


async def test_a_backtest_is_one_run_with_one_aggregate_per_question(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    result = await run_backtest(db, questions)
    assert (result.forecast, result.resumed, result.skipped, result.failed) == (3, 0, 0, 0)
    [(run_id, mode, tournament, model)] = _rows(
        db, "SELECT run_id, mode, tournament, model FROM runs"
    )
    assert result.run_id == run_id
    assert (mode, tournament, model) == ("backtest", f"forecastbench:{QUESTION_SET}", "qwen2.5:7b")
    aggregates = _rows(
        db, "SELECT question_id, probability FROM forecasts WHERE kind = 'aggregate' ORDER BY rowid"
    )
    assert [qid for qid, _ in aggregates] == [q.question_id for q in questions]
    assert [p for _, p in aggregates] == [0.3, 0.3, 0.3]
    members = _rows(db, "SELECT COUNT(*) FROM forecasts WHERE kind = 'member'")
    assert members == [(9,)]  # 3 draws on each of 3 questions
    assert _rows(db, "SELECT COUNT(*) FROM llm_calls WHERE model_release_date IS NOT NULL") == [
        (9,)
    ]


async def test_the_ledger_records_outcomes_cutoffs_and_no_outcome_payload(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    await run_backtest(db, questions)
    outcomes = _rows(
        db,
        "SELECT question_id, status, reason, submitted, question_type FROM question_outcomes "
        "ORDER BY rowid",
    )
    assert outcomes == [
        (q.question_id, "forecast", "forecast with no retrieval", 0, "binary") for q in questions
    ]
    cutoffs = _rows(db, "SELECT cutoff, open_time FROM questions")
    assert cutoffs == [("2025-03-01T00:00:00.000000Z", "2025-03-01T00:00:00.000000Z")] * 3
    payloads = _rows(db, "SELECT payload_json FROM questions")
    assert all('"outcome"' not in p for (p,) in payloads)  # the answer is never stored with it
    assert all('"freeze_value"' in p for (p,) in payloads)  # the reference price is
    assert scan_ledger(db) == []


async def test_the_prompt_carries_no_market_price(tmp_path: Path) -> None:
    qs, rs = write_pair(tmp_path, [raw_question("a", value="0.7123")], [raw_resolution("a", 1.0)])
    questions = load_set(qs, rs, released=RELEASED).kept
    db = tmp_path / "bt.sqlite"
    await run_backtest(db, questions)
    requests = [r for (r,) in _rows(db, "SELECT request_json FROM llm_calls")]
    assert requests
    for text in requests:
        assert "0.7123" not in text
        assert "No research was done" in text
        assert "Will event a happen?" in text


async def test_a_second_run_forecasts_nothing_new(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    await run_backtest(db, questions)
    before = _rows(db, "SELECT COUNT(*) FROM forecasts")
    again = await run_backtest(db, questions)
    assert (again.forecast, again.resumed, again.skipped, again.failed) == (0, 3, 0, 0)
    assert _rows(db, "SELECT COUNT(*) FROM forecasts") == before
    assert _rows(db, "SELECT COUNT(*) FROM runs") == [(2,)]
    assert _rows(db, "SELECT COUNT(*) FROM llm_calls") == [(9,)]


async def test_a_resume_forecasts_only_the_missing_questions(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    await run_backtest(db, questions[:2])
    result = await run_backtest(db, questions)
    assert (result.forecast, result.resumed) == (1, 2)
    assert done_questions_of(db) == {q.question_id for q in questions}


def done_questions_of(db: Path) -> set[int]:
    from augur.clock import SystemClock
    from augur.ledger import Ledger

    with Ledger.open(db, SystemClock()) as ledger:
        return done_questions(ledger, f"forecastbench:{QUESTION_SET}")


async def test_done_questions_is_per_tournament(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    await run_backtest(db, questions, tournament="forecastbench:other.json")
    result = await run_backtest(db, questions)
    assert (result.forecast, result.resumed) == (3, 0)


async def test_a_model_released_after_the_freeze_is_skipped_with_a_reason(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)  # frozen 2025-03-01
    route = Route(endpoint=PRESETS["openrouter"], model="acme/new-model")
    catalog = catalog_of(("acme/new-model", 0.0, 0.0), listed=datetime(2025, 6, 1, tzinfo=UTC))
    db = tmp_path / "bt.sqlite"
    result = await run_backtest(db, questions, route=route, catalog=catalog)
    assert (result.forecast, result.skipped, result.failed) == (0, 3, 0)
    outcomes = _rows(db, "SELECT status, reason FROM question_outcomes")
    assert len(outcomes) == 3
    assert all(s == "skipped" and "released 2025-06-01" in r for s, r in outcomes)
    assert _rows(db, "SELECT COUNT(*) FROM forecasts") == [(0,)]
    assert _rows(db, "SELECT COUNT(*) FROM llm_calls") == [(0,)]
    assert scan_ledger(db) == []


async def test_a_model_with_an_unknown_release_is_skipped(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    route = Route(endpoint=PRESETS["openrouter"], model="acme/unlisted")
    db = tmp_path / "bt.sqlite"
    result = await run_backtest(db, questions, route=route, catalog=catalog_of())
    assert result.skipped == 3
    [(reason,)] = _rows(db, "SELECT DISTINCT reason FROM question_outcomes")
    assert "release date unknown" in str(reason)


async def test_a_model_released_the_day_the_question_opened_is_skipped(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    route = Route(endpoint=PRESETS["openrouter"], model="acme/same-day")
    catalog = catalog_of(("acme/same-day", 0.0, 0.0), listed=datetime(2025, 3, 1, 9, tzinfo=UTC))
    result = await run_backtest(tmp_path / "bt.sqlite", questions, route=route, catalog=catalog)
    assert result.skipped == 3


async def test_a_question_whose_draws_all_fail_is_failed_not_fatal(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    result = await run_backtest(db, questions[:2], percents=(None,))
    assert (result.forecast, result.failed) == (0, 2)
    statuses = _rows(db, "SELECT status, reason FROM question_outcomes")
    assert [s for s, _ in statuses] == ["failed", "failed"]
    assert all("draws parsed" in r for _, r in statuses)
    assert _rows(db, "SELECT COUNT(*) FROM forecasts WHERE kind = 'aggregate'") == [(0,)]


async def test_a_failed_question_is_tried_again_on_resume(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    await run_backtest(db, questions[:1], percents=(None,))
    result = await run_backtest(db, questions[:1])
    assert (result.forecast, result.resumed) == (1, 0)


def test_pinned_clock_moves_and_stays_utc() -> None:
    clock = PinnedClock(datetime(2025, 3, 1, tzinfo=UTC))
    clock.set(datetime(2025, 4, 1, tzinfo=UTC))
    assert clock.now() == datetime(2025, 4, 1, tzinfo=UTC)


async def test_the_run_row_records_the_real_start_not_a_freeze_date(tmp_path: Path) -> None:
    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    await run_backtest(db, questions)
    [(started,)] = _rows(db, "SELECT started_at FROM runs")
    # The harness's clock starts at 2026-01-01; every freeze date is earlier or later.
    assert started == "2026-01-01T00:00:00.000000Z"
    assert all(started != q.freeze_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for q in questions)
