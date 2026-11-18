"""Leakage is a build failure.

The gate must reject a source dated after its question's cutoff, the ledger
must refuse to store an admitted one, and :func:`scan_ledger` must find one in
any ledger that holds it anyway. The last test scans every committed ledger in
the repository; any violation there fails the suite.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from augur.clock import FixedClock
from augur.errors import LeakageError, LedgerError
from augur.leakage import Admission, RunMode, admit_model, admit_source, scan_ledger
from augur.ledger import Ledger
from augur.research.source import DateEvidence

from .conftest import T0
from .helpers import binary_question, source

REPO = Path(__file__).resolve().parent.parent
COMMITTED_LEDGERS = sorted((REPO / "data" / "ledgers").glob("**/*.sqlite"))

CUTOFF = T0
BEFORE = T0 - timedelta(days=2)
AFTER = T0 + timedelta(seconds=1)


# -- the gate -------------------------------------------------------------


@pytest.mark.parametrize("mode", list(RunMode))
def test_a_source_dated_after_the_cutoff_is_rejected(mode: RunMode) -> None:
    decision = admit_source(source(AFTER, retrieved_at=AFTER), CUTOFF, mode)
    assert not decision.admitted
    assert "after the cutoff" in decision.reason


def test_a_source_dated_exactly_at_the_cutoff_is_admitted() -> None:
    assert admit_source(source(CUTOFF, retrieved_at=CUTOFF), CUTOFF, RunMode.BACKTEST).admitted


def test_undated_and_unproven_sources_are_refused_in_backtests_only() -> None:
    undated = source(None, retrieved_at=T0)
    unproven = source(BEFORE, retrieved_at=T0, evidence=DateEvidence.NONE)
    for s in (undated, unproven):
        assert not admit_source(s, CUTOFF, RunMode.BACKTEST).admitted
        assert admit_source(s, CUTOFF, RunMode.LIVE).admitted


def test_models_must_predate_the_question_in_backtests() -> None:
    opened = T0
    assert admit_model(date(2024, 9, 19), opened, RunMode.BACKTEST).admitted
    assert not admit_model(opened.date(), opened, RunMode.BACKTEST).admitted
    assert not admit_model(None, opened, RunMode.BACKTEST).admitted
    assert admit_model(None, opened, RunMode.LIVE).admitted


# -- the store refuses -----------------------------------------------------


def _ledger_with_question(tmp: Path, mode: RunMode) -> tuple[Ledger, str, int]:
    ledger = Ledger.open(tmp / f"{mode.value}.sqlite", FixedClock(T0))
    run_id = ledger.start_run(
        mode=mode,
        tournament="t",
        provider="replay",
        model="qwen2.5:7b",
        settings={},
        code_version="test",
    )
    # A backtest's cutoff is its question's open time; live, it is now.
    opened = T0 if mode is RunMode.BACKTEST else T0 - timedelta(days=30)
    q, post = binary_question(opened)
    ledger.record_question(run_id, q, CUTOFF, post)
    return ledger, run_id, q.question_id


def test_the_ledger_refuses_an_admitted_late_source(tmp_path: Path) -> None:
    ledger, run_id, qid = _ledger_with_question(tmp_path, RunMode.LIVE)
    late = source(AFTER, retrieved_at=AFTER)
    ledger.record_source(late)
    forged = Admission(late.source_id, True, "a buggy gate said yes")
    with pytest.raises(LeakageError, match="after the question cutoff"):
        ledger.record_admission(run_id, qid, forged)
    # Recording the honest rejection is fine.
    ledger.record_admission(run_id, qid, admit_source(late, CUTOFF, RunMode.LIVE))
    assert scan_ledger(tmp_path / "live.sqlite") == []
    ledger.close()


def test_the_ledger_refuses_an_admitted_undated_source_in_a_backtest(tmp_path: Path) -> None:
    ledger, run_id, qid = _ledger_with_question(tmp_path, RunMode.BACKTEST)
    undated = source(None, retrieved_at=T0)
    ledger.record_source(undated)
    with pytest.raises(LeakageError, match="undated"):
        ledger.record_admission(run_id, qid, Admission(undated.source_id, True, "forged"))
    ledger.close()


def test_a_model_call_cannot_cite_an_unadmitted_source(tmp_path: Path) -> None:
    ledger, run_id, qid = _ledger_with_question(tmp_path, RunMode.LIVE)
    late = source(AFTER, retrieved_at=AFTER)
    ledger.record_source(late)
    ledger.record_admission(run_id, qid, admit_source(late, CUTOFF, RunMode.LIVE))
    with pytest.raises(LedgerError, match="did not admit"):
        ledger.record_call(
            run_id=run_id,
            question_id=qid,
            role="forecaster",
            member=0,
            provider="replay",
            model="qwen2.5:7b",
            model_release_date=date(2024, 9, 19),
            request_hash="h",
            request={},
            response_text="",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
            latency_ms=0,
            error=None,
            started_at=T0,
            source_ids=[late.source_id],
        )
    assert ledger.count("llm_calls") == 0, "the failed call must not be half-written"
    ledger.close()


# -- the scan finds what slipped past --------------------------------------


def _plant(path: Path, mode: RunMode, kind: str) -> None:
    """Write a violation directly, as a buggy or foreign writer would."""
    ledger, run_id, qid = _ledger_with_question(path.parent, mode)
    ledger.close()
    conn = sqlite3.connect(path, isolation_level=None)
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    if kind in ("late_source", "undated_source", "unadmitted_use"):
        published = {"late_source": "2099-01-01T00:00:00.000000Z", "undated_source": None}.get(
            kind, "2020-01-01T00:00:00.000000Z"
        )
        conn.execute(
            "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "src_planted",
                "wikipedia",
                "q",
                "u",
                "t",
                "x",
                "h",
                published,
                "2026-01-01T00:00:00.000000Z",
                "2026-01-01T00:00:00.000000Z",
                "none" if kind == "undated_source" else "revision_timestamp",
                "planted",
                "2026-01-01T00:00:00.000000Z",
            ),
        )
        admitted = 0 if kind == "unadmitted_use" else 1
        conn.execute(
            "INSERT INTO source_uses (run_id, question_id, source_id, admitted, reason, "
            "recorded_at) VALUES (?, ?, 'src_planted', ?, 'planted', 'x')",
            (run_id, qid, admitted),
        )
    if kind in ("unadmitted_use", "model_too_new"):
        release = "2099-01-01" if kind == "model_too_new" else "2020-01-01"
        conn.execute(
            "INSERT INTO llm_calls (call_id, run_id, question_id, role, provider, model, "
            "model_release_date, request_hash, request_json, latency_ms, started_at) "
            "VALUES ('call_planted', ?, ?, 'forecaster', 'replay', 'm', ?, 'h', '{}', 0, 'x')",
            (run_id, qid, release),
        )
    if kind == "unadmitted_use":
        conn.execute("INSERT INTO call_sources VALUES ('call_planted', 'src_planted')")
    conn.close()


@pytest.mark.parametrize(
    ("kind", "mode"),
    [
        ("late_source", RunMode.LIVE),
        ("late_source", RunMode.BACKTEST),
        ("undated_source", RunMode.BACKTEST),
        ("unadmitted_use", RunMode.LIVE),
        ("model_too_new", RunMode.BACKTEST),
    ],
)
def test_a_ledger_holding_a_violation_fails_the_scan(
    tmp_path: Path, kind: str, mode: RunMode
) -> None:
    path = tmp_path / f"{mode.value}.sqlite"
    _plant(path, mode, kind)
    violations = scan_ledger(path)
    # _plant drops the triggers, as a foreign writer might; that is reported too.
    assert [v.kind for v in violations if v.kind != "missing_guard"] == [kind]
    assert {v.kind for v in violations} >= {"missing_guard"}


def test_a_ledger_without_its_guards_fails_the_scan(tmp_path: Path) -> None:
    path = tmp_path / "live.sqlite"
    ledger, _, _ = _ledger_with_question(tmp_path, RunMode.LIVE)
    ledger.close()
    assert scan_ledger(path) == []
    conn = sqlite3.connect(path)
    conn.execute("DROP TRIGGER call_sources_admitted_only")
    conn.close()
    assert [v.kind for v in scan_ledger(path)] == ["missing_guard"]


def test_insert_or_replace_cannot_rewrite_a_row(tmp_path: Path) -> None:
    ledger, run_id, _ = _ledger_with_question(tmp_path, RunMode.BACKTEST)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.connection.execute(
            "INSERT OR REPLACE INTO runs SELECT run_id, started_at, 'live', tournament, "
            "provider, model, settings_json, code_version FROM runs WHERE run_id = ?",
            (run_id,),
        )
    ledger.close()


@pytest.mark.parametrize(
    ("mode", "cutoff_shift", "open_shift"),
    [
        (RunMode.LIVE, timedelta(hours=1), timedelta(days=-30)),
        (RunMode.BACKTEST, timedelta(0), timedelta(days=-1)),
    ],
)
def test_a_cutoff_the_scan_cannot_trust_is_a_violation(
    tmp_path: Path, mode: RunMode, cutoff_shift: timedelta, open_shift: timedelta
) -> None:
    """A future cutoff (live) or one after the open time (backtest) would admit leaks."""
    ledger = Ledger.open(tmp_path / "l.sqlite", FixedClock(T0))
    run_id = ledger.start_run(
        mode=mode, tournament="t", provider="p", model="m", settings={}, code_version="t"
    )
    q, post = binary_question(T0 + open_shift)
    ledger.record_question(run_id, q, T0 + cutoff_shift, post)
    ledger.close()
    assert [v.kind for v in scan_ledger(tmp_path / "l.sqlite")] == ["bad_cutoff"]


@pytest.mark.parametrize("published", ["'2024-10-01 18:00:00'", "1730000000"])
def test_a_timestamp_not_in_the_ledgers_own_form_fails_the_scan(
    tmp_path: Path, published: str
) -> None:
    """Times compare as text, so a late time in another form slips past the triggers."""
    ledger, run_id, qid = _ledger_with_question(tmp_path, RunMode.LIVE)
    ok = source(BEFORE, retrieved_at=BEFORE)
    ledger.record_source(ok)
    conn = ledger.connection
    columns = [row[1] for row in conn.execute("PRAGMA table_info(sources)")]
    picked = ", ".join(
        "'foreign'" if c == "source_id" else published if c == "published_at" else c
        for c in columns
    )
    conn.execute(f"INSERT INTO sources SELECT {picked} FROM sources")
    # A late source in another format: the guard's text comparison lets it in.
    ledger.record_admission(run_id, qid, Admission("foreign", True, "a foreign writer"))
    ledger.close()
    kinds = [v.kind for v in scan_ledger(tmp_path / "live.sqlite")]
    assert kinds == ["bad_timestamp"]


def test_undated_sources_in_live_ledgers_are_not_violations(tmp_path: Path) -> None:
    path = tmp_path / "live.sqlite"
    _plant(path, RunMode.LIVE, "undated_source")
    assert [v.kind for v in scan_ledger(path)] == ["missing_guard"] * 3


@pytest.mark.parametrize("path", COMMITTED_LEDGERS, ids=lambda p: str(p.relative_to(REPO)))
def test_committed_ledgers_hold_no_leakage(path: Path) -> None:
    violations = scan_ledger(path)
    assert violations == [], "\n".join(f"{v.kind}: {v.detail}" for v in violations)
