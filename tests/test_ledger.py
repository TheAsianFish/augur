from __future__ import annotations

import sqlite3
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
        assert first.schema_version() == 1
    with Ledger.open(path, clock) as again:
        assert again.schema_version() == 1


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
