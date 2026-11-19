from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from augur.clock import FixedClock
from augur.errors import LedgerError
from augur.leakage import RunMode, admit_source
from augur.ledger import APPEND_ONLY_TABLES, Ledger

from .conftest import T0
from .helpers import binary_question, source


def _run(ledger: Ledger) -> str:
    return ledger.start_run(
        mode=RunMode.DRY_RUN,
        tournament="minibench",
        provider="replay",
        model="qwen2.5:7b",
        settings={"forecasters": 5},
        code_version="test",
    )


def test_migrations_apply_once_and_reopen(tmp_path: Path, clock: FixedClock) -> None:
    path = tmp_path / "l.sqlite"
    with Ledger.open(path, clock) as first:
        assert first.schema_version() == 5
    with Ledger.open(path, clock) as again:
        assert again.schema_version() == 5


@pytest.mark.parametrize("table", APPEND_ONLY_TABLES)
def test_every_table_refuses_update_and_delete(ledger: Ledger, table: str) -> None:
    run_id = _run(ledger)
    q, post = binary_question(T0 - timedelta(hours=1))
    ledger.record_question(run_id, q, T0, post)
    s = source(T0 - timedelta(days=1), retrieved_at=T0)
    ledger.record_source(s)
    ledger.record_admission(run_id, q.question_id, admit_source(s, T0, RunMode.DRY_RUN))
    call_id = ledger.record_call(
        run_id=run_id,
        question_id=q.question_id,
        role="forecaster",
        member=0,
        provider="replay",
        model="qwen2.5:7b",
        model_release_date=None,
        request_hash="h",
        request={"m": 1},
        response_text="Probability: 40%",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1,
        error=None,
        started_at=T0,
        source_ids=[s.source_id],
    )
    ledger.record_forecast(
        run_id=run_id,
        question_id=q.question_id,
        kind="member",
        member=0,
        method="draw",
        probability=0.4,
        reasoning="r",
        call_id=call_id,
    )
    ledger.record_submission(
        run_id=run_id,
        question_id=q.question_id,
        post_id=q.post_id,
        kind="forecast",
        payload=[{"probability_yes": 0.4}],
        status="dry_run",
        http_status=None,
        response_excerpt=None,
    )
    ledger.record_switch(
        run_id=run_id,
        question_id=q.question_id,
        call_role="forecaster",
        member=0,
        from_route="openrouter:a",
        to_route="openrouter:b",
        reason="rate_limited",
        outcome="switched",
        detail="429",
    )
    ledger.record_outcome(
        run_id=run_id,
        question_id=q.question_id,
        post_id=q.post_id,
        question_type=q.type,
        status="forecast",
        reason="",
        submitted=False,
        closes_at=q.scheduled_close_time,
        minutes_left=90.0,
        cost_usd=0.0,
        seconds=12.5,
    )
    ledger.record_probe(
        {
            "run_id": run_id,
            "session_id": "s",
            "route": "openrouter:a",
            "endpoint": "openrouter",
            "endpoint_host": "openrouter.ai",
            "model": "a",
            "roles": "forecast",
            "outcome": "answered",
            "price_basis": "test",
            "release_basis": "unknown",
            "in_tier": 0,
            "limit_headers_json": "{}",
            "detail": "ok",
        }
    )
    ledger.record_limit_test(
        {
            "session_id": "s",
            "route": "openrouter:a",
            "endpoint_host": "openrouter.ai",
            "requests_cap": 3,
            "requests_sent": 3,
            "requests_ok": 3,
            "input_tokens": 3,
            "output_tokens": 3,
            "seconds": 1.0,
            "limit_headers_json": "{}",
            "started_at": "2026-11-17T18:00:00.000000Z",
        }
    )
    ledger.record_resolution(
        question_id=q.question_id,
        post_id=q.post_id,
        question_type="binary",
        status="resolved",
        value="yes",
        resolved_at=T0,
        source="manual",
        evidence="test",
    )
    ledger.record_reference(
        question_id=q.question_id,
        post_id=q.post_id,
        question_type="binary",
        forecaster="template",
        detail="test",
        values={"probability": 0.3},
        as_of=T0,
        source="test",
    )
    ledger.record_metaculus_score(
        question_id=q.question_id,
        post_id=q.post_id,
        forecaster="augur",
        score_type="baseline_score",
        score=1.0,
    )
    conn = ledger.connection
    assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] >= 1
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute(f"DELETE FROM {table}")
    first_col = conn.execute(f"PRAGMA table_info({table})").fetchone()[1]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute(f"UPDATE {table} SET {first_col} = {first_col}")


def test_same_source_twice_is_stored_once(ledger: Ledger) -> None:
    s = source(T0 - timedelta(days=1), retrieved_at=T0)
    ledger.record_source(s)
    ledger.record_source(s)
    assert ledger.count("sources") == 1


def test_probability_must_be_inside_zero_one(ledger: Ledger) -> None:
    run_id = _run(ledger)
    q, post = binary_question(T0 - timedelta(hours=1))
    ledger.record_question(run_id, q, T0, post)
    with pytest.raises(LedgerError, match="CHECK"):
        ledger.record_forecast(
            run_id=run_id,
            question_id=q.question_id,
            kind="aggregate",
            member=None,
            method="median",
            probability=1.0,
            reasoning="",
            call_id=None,
        )


def test_unknown_table_count_is_refused(ledger: Ledger) -> None:
    with pytest.raises(LedgerError, match="unknown table"):
        ledger.count("sqlite_master")


def test_a_repair_posts_the_forecasting_runs_comment_not_a_later_dry_runs(
    ledger: Ledger, clock: FixedClock
) -> None:
    q, post = binary_question(T0 - timedelta(hours=1))
    live = ledger.start_run(
        mode=RunMode.LIVE,
        tournament="minibench",
        provider="replay",
        model="m",
        settings={},
        code_version="test",
    )
    ledger.record_question(live, q, T0, post)

    def sub(run_id: str, kind: str, status: str, text: str = "") -> None:
        payload = {"text": text} if kind == "comment" else {"probability_yes": 0.4}
        ledger.record_submission(
            run_id=run_id,
            question_id=q.question_id,
            post_id=q.post_id,
            kind=kind,
            payload=payload,
            status=status,
            http_status=None,
            response_excerpt=None,
        )

    sub(live, "forecast", "submitted")
    sub(live, "comment", "failed", "**Forecast: 40%** live")
    clock.advance(300)
    dry = _run(ledger)
    ledger.record_question(dry, q, clock.now(), post)
    sub(dry, "forecast", "dry_run")
    sub(dry, "comment", "dry_run", "**Forecast: 80%** dry")
    assert ledger.unposted_comment(q.question_id) == "**Forecast: 40%** live"

    sub(dry, "comment", "submitted", "**Forecast: 40%** live")
    assert ledger.unposted_comment(q.question_id) is None


def test_a_schema_1_ledger_is_migrated_in_place(tmp_path: Path, clock: FixedClock) -> None:
    """The committed M1 ledgers open under the current schema with their rows intact."""
    import shutil

    from augur.leakage import scan_ledger

    original = Path(__file__).resolve().parent.parent / "data/ledgers/m1-dryrun-samples.sqlite"
    copy = tmp_path / "m1.sqlite"
    shutil.copy(original, copy)
    with Ledger.open(copy, clock) as led:
        assert led.schema_version() == 5
        assert led.count("llm_calls") == 7
        assert led.count("model_switches") == 0
    assert scan_ledger(copy) == []


def test_migration_4_keeps_every_forecast_of_an_older_ledger(tmp_path: Path) -> None:
    import shutil

    committed = (
        Path(__file__).resolve().parent.parent / "data/ledgers/m1-live-bot-testing-area.sqlite"
    )
    copy = tmp_path / "old.sqlite"
    shutil.copy(committed, copy)
    with closing(sqlite3.connect(f"file:{committed}?mode=ro", uri=True)) as old:
        before = old.execute(
            "SELECT forecast_id, probability, reasoning FROM forecasts ORDER BY forecast_id"
        ).fetchall()
    assert before
    with Ledger.open(copy, FixedClock(T0)) as led:
        assert led.schema_version() == 5
        conn = led.connection
        after = conn.execute(
            "SELECT forecast_id, probability, reasoning FROM forecasts ORDER BY forecast_id"
        ).fetchall()
        assert after == before
        types = {r[0] for r in conn.execute("SELECT question_type FROM forecasts")}
        assert types == {"binary"}
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM forecasts")


def test_a_resolution_needs_a_value_exactly_when_resolved(ledger: Ledger) -> None:
    common = {
        "question_id": 1,
        "post_id": 1,
        "question_type": "binary",
        "resolved_at": T0,
        "source": "manual",
        "evidence": "test",
    }
    with pytest.raises(LedgerError):
        ledger.record_resolution(status="resolved", value=None, **common)  # type: ignore[arg-type]
    with pytest.raises(LedgerError):
        ledger.record_resolution(status="annulled", value="yes", **common)  # type: ignore[arg-type]
    ledger.record_resolution(status="resolved", value="no", **common)  # type: ignore[arg-type]
    ledger.record_resolution(status="annulled", value=None, **common)  # type: ignore[arg-type]
    # A correction is a new row, and the latest one counts.
    assert ledger.latest_resolution(1) == ("annulled", None)
    assert ledger.latest_resolution(2) is None
