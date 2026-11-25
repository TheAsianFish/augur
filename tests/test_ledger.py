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
from .helpers import binary_question, market_source, source


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
        assert first.schema_version() == 8
    with Ledger.open(path, clock) as again:
        assert again.schema_version() == 8


@pytest.mark.parametrize("table", APPEND_ONLY_TABLES)
def test_every_table_refuses_update_and_delete(ledger: Ledger, table: str) -> None:
    run_id = _run(ledger)
    q, post = binary_question(T0 - timedelta(hours=1))
    ledger.record_question(run_id, q, T0, post)
    s = source(T0 - timedelta(days=1), retrieved_at=T0)
    ledger.record_source(s)
    ledger.record_admission(run_id, q.question_id, admit_source(s, T0, RunMode.DRY_RUN))
    ledger.record_source(market_source(T0 - timedelta(hours=2), as_of=T0))
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
    ledger.record_judgment(
        run_id=run_id,
        question_id=q.question_id,
        source_id=s.source_id,
        call_id=call_id,
        judge_model="qwen2.5:7b",
        score=0,
        rationale="no post-cutoff information",
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
        assert led.schema_version() == 8
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
        assert led.schema_version() == 8
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


def test_a_market_source_stores_its_price_beside_it(ledger: Ledger) -> None:
    s = market_source(T0 - timedelta(hours=2), as_of=T0, probability=0.27)
    ledger.record_source(s)
    ledger.record_source(s)  # the same source twice is still one row
    rows = ledger.connection.execute(
        "SELECT source_id, platform, probability, price_time, match FROM market_quotes"
    ).fetchall()
    assert rows == [(s.source_id, "manifold", 0.27, "2026-11-17T16:00:00.000000Z", "exact")]


def test_a_price_dated_after_its_instant_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="after the instant it was asked for"):
        market_source(T0 + timedelta(seconds=1), as_of=T0)


def test_a_market_source_dated_unlike_its_price_is_refused(ledger: Ledger) -> None:
    s = market_source(T0 - timedelta(hours=2), as_of=T0)
    assert s.quote is not None
    wrong = s.model_copy(update={"published_at": T0 - timedelta(hours=1)})
    with pytest.raises(LedgerError, match="must be its price's"):
        ledger.record_source(wrong)
    # Refused whole: no market-dated source is left behind without its price.
    assert ledger.count("sources") == 0
    assert ledger.count("market_quotes") == 0


def test_a_failed_price_write_leaves_no_source_behind(ledger: Ledger) -> None:
    s = market_source(T0 - timedelta(hours=2), as_of=T0)
    ledger.connection.execute(
        "CREATE TRIGGER fail_quote BEFORE INSERT ON market_quotes "
        "BEGIN SELECT RAISE(ABORT, 'injected'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        ledger.record_source(s)
    assert ledger.count("sources") == 0


def test_the_rebuilt_sources_table_still_refuses_a_late_price(ledger: Ledger) -> None:
    run_id = _run(ledger)
    q, post = binary_question(T0 - timedelta(hours=1))
    cutoff = T0 - timedelta(hours=3)
    ledger.record_question(run_id, q, cutoff, post)
    late = market_source(T0 - timedelta(hours=2), as_of=T0)  # after the cutoff
    ledger.record_source(late)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.connection.execute(
            "INSERT INTO source_uses (run_id, question_id, source_id, admitted, reason, "
            "recorded_at) VALUES (?, ?, ?, 1, 'forced', ?)",
            (run_id, q.question_id, late.source_id, "2026-11-17T12:00:00.000000Z"),
        )


def test_the_scan_flags_a_price_whose_date_was_edited(tmp_path: Path) -> None:
    from augur.leakage import scan_ledger

    path = tmp_path / "l.sqlite"
    with Ledger.open(path, FixedClock(T0)) as led:
        led.record_source(market_source(T0 - timedelta(hours=2), as_of=T0))
    assert scan_ledger(path) == []
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("DROP TRIGGER sources_no_update")
        raw.execute("UPDATE sources SET published_at = '2026-11-17T11:00:00.000000Z'")
        raw.commit()
    kinds = {v.kind for v in scan_ledger(path)}
    assert "late_price" in kinds


def test_the_scan_flags_a_market_source_without_its_price(tmp_path: Path) -> None:
    from augur.leakage import scan_ledger

    path = tmp_path / "l.sqlite"
    with Ledger.open(path, FixedClock(T0)) as led:
        bare = market_source(T0 - timedelta(hours=2), as_of=T0).model_copy(update={"quote": None})
        led.record_source(bare)
    violations = scan_ledger(path)
    assert [v.kind for v in violations] == ["late_price"]
    assert "has no price row" in violations[0].detail


def test_migration_8_keeps_every_reference_and_admits_stat_and_blend(tmp_path: Path) -> None:
    """M12: a schema-7 ledger's template rows survive, and stat and blend rows are accepted."""
    import shutil

    committed = Path(__file__).resolve().parent.parent / "data/ledgers/m4-shadow-template.sqlite"
    copy = tmp_path / "m4.sqlite"
    shutil.copy(committed, copy)
    with closing(sqlite3.connect(f"file:{committed}?mode=ro", uri=True)) as old:
        before = old.execute("SELECT * FROM reference_forecasts ORDER BY reference_id").fetchall()
    assert before
    with Ledger.open(copy, FixedClock(T0)) as led:
        assert led.schema_version() == 8
        conn = led.connection
        after = conn.execute("SELECT * FROM reference_forecasts ORDER BY reference_id").fetchall()
        assert after == before
        qid, post_id, qtype = before[0][1], before[0][2], before[0][3]
        for who in ("stat", "blend"):
            led.record_reference(
                question_id=qid,
                post_id=post_id,
                question_type=qtype,
                forecaster=who,
                detail="test",
                values={"probability": 0.4},
                as_of=T0,
                source="test",
            )
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM reference_forecasts")
