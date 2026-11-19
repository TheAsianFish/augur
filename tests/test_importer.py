"""``augur ledger import``: rebuild one ledger from JSONL exports."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from augur.clock import FixedClock
from augur.errors import LedgerError
from augur.export import export_jsonl
from augur.importer import export_files, import_exports
from augur.leakage import RunMode, admit_source, scan_ledger
from augur.ledger import APPEND_ONLY_TABLES, Ledger

from .conftest import T0
from .fakes.forecastbench import run_backtest, small_set
from .helpers import binary_question, source


def _populate(
    path: Path, clock: FixedClock, *, post_id: int = 100, mode: RunMode = RunMode.LIVE
) -> str:
    """One run touching most tables: question, source, use, call, forecasts, submission..."""
    with Ledger.open(path, clock) as ledger:
        run_id = ledger.start_run(
            mode=mode,
            tournament="bot-testing-area",
            provider="replay",
            model="qwen2.5:7b",
            settings={"forecasters": 2, "note": "naïve ✓"},
            code_version="test",
        )
        q, post = binary_question(T0 - timedelta(hours=1), post_id=post_id, question_id=post_id + 1)
        ledger.record_question(run_id, q, T0, post)
        s = source(
            T0 - timedelta(days=1),
            retrieved_at=T0,
            text=f"text for {post_id}",
            url=f"https://en.wikipedia.org/w/index.php?oldid={post_id}",
        )
        ledger.record_source(s)
        ledger.record_admission(run_id, q.question_id, admit_source(s, T0, mode))
        call_id = ledger.record_call(
            run_id=run_id,
            question_id=q.question_id,
            role="forecaster",
            member=0,
            provider="replay",
            model="qwen2.5:7b",
            model_release_date=date(2024, 9, 19),
            request_hash="h",
            request={"messages": []},
            response_text="Probability: 30%",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
            latency_ms=3,
            error=None,
            started_at=T0,
            source_ids=[s.source_id],
        )
        for member, p in ((0, 0.3), (1, 0.4)):
            ledger.record_forecast(
                run_id=run_id,
                question_id=q.question_id,
                kind="member",
                member=member,
                method="draw",
                probability=p,
                reasoning="r",
                call_id=call_id if member == 0 else None,
            )
        ledger.record_forecast(
            run_id=run_id,
            question_id=q.question_id,
            kind="aggregate",
            member=None,
            method="median",
            probability=0.35,
            reasoning="median of 2",
            call_id=None,
        )
        ledger.record_submission(
            run_id=run_id,
            question_id=q.question_id,
            post_id=post_id,
            kind="forecast",
            payload={"probability_yes": 0.35},
            status="submitted",
            http_status=201,
            response_excerpt=None,
        )
        ledger.record_outcome(
            run_id=run_id,
            question_id=q.question_id,
            post_id=post_id,
            question_type="binary",
            status="forecast",
            reason="ok",
            submitted=True,
            closes_at=T0 + timedelta(hours=2),
            minutes_left=120.0,
            cost_usd=0.0,
            seconds=1.5,
        )
        ledger.record_switch(
            run_id=run_id,
            question_id=q.question_id,
            call_role="forecaster",
            member=0,
            from_route="a:b",
            to_route="c:d",
            reason="rate_limited",
            outcome="switched",
            detail="x",
        )
        ledger.record_resolution(
            question_id=q.question_id,
            post_id=post_id,
            question_type="binary",
            status="resolved",
            value="no",
            resolved_at=T0,
            source="manual",
            evidence="the page",
        )
        ledger.record_reference(
            question_id=q.question_id,
            post_id=post_id,
            question_type="binary",
            forecaster="template",
            detail="d",
            values={"probability": 0.5},
            as_of=T0,
            source="s",
        )
        ledger.record_metaculus_score(
            question_id=q.question_id,
            post_id=post_id,
            forecaster="augur",
            score_type="baseline_score",
            score=12.5,
        )
    return run_id


def _counts(path: Path) -> dict[str, int]:
    with closing(sqlite3.connect(path)) as conn:
        names = [
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {n: int(conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]) for n in names}


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _without(lines: list[str], table: str) -> list[str]:
    return [line for line in lines if json.loads(line)["table"] != table]


@pytest.fixture
def populated(tmp_path: Path, clock: FixedClock) -> tuple[Path, Path, str]:
    db = tmp_path / "a.sqlite"
    run_id = _populate(db, clock)
    export = tmp_path / "exports" / "2026-11-17T180000Z.jsonl"
    export_jsonl(db, export)
    return db, export, run_id


def test_the_fixture_populates_every_table_it_means_to(populated: tuple[Path, Path, str]) -> None:
    db, _, _ = populated
    counts = _counts(db)
    for table in (
        "runs",
        "questions",
        "sources",
        "source_uses",
        "llm_calls",
        "call_sources",
        "forecasts",
        "submissions",
        "model_switches",
        "question_outcomes",
        "resolutions",
        "reference_forecasts",
        "metaculus_scores",
    ):
        assert counts[table] >= 1, table


def test_import_reproduces_every_row_count(
    populated: tuple[Path, Path, str], tmp_path: Path, clock: FixedClock
) -> None:
    db, export, _ = populated
    out = tmp_path / "imported.sqlite"
    report = import_exports([export], out, clock)
    original, imported = _counts(db), _counts(out)
    assert imported == original
    assert report.files == 1
    assert report.duplicates == {}
    assert report.inserted.keys() == {
        t for t, n in original.items() if n and t != "schema_migrations"
    }
    assert report.total == sum(n for t, n in original.items() if t != "schema_migrations")
    assert scan_ledger(out) == []


def test_a_reexport_of_the_import_is_byte_identical_for_one_export(
    populated: tuple[Path, Path, str], tmp_path: Path, clock: FixedClock
) -> None:
    _, export, _ = populated
    out = tmp_path / "imported.sqlite"
    import_exports([export], out, clock)
    again = tmp_path / "again.jsonl"
    export_jsonl(out, again)
    # With the same clock, even schema_migrations matches; with one export source_uses ids
    # are assigned 1..n in the original order, so they match too.
    assert again.read_bytes() == export.read_bytes()


def test_source_use_ids_are_reassigned_but_content_is_kept(
    populated: tuple[Path, Path, str], tmp_path: Path, clock: FixedClock
) -> None:
    _, export, _ = populated
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    # The second ledger already holds an unrelated source use, so its ids start higher.
    _populate(second, clock, post_id=500)
    import_exports([export], first, clock)
    import_exports([export], second, clock)
    with closing(sqlite3.connect(second)) as conn:
        ids = [r[0] for r in conn.execute("SELECT use_id FROM source_uses ORDER BY use_id")]
        assert ids == [1, 2]
    columns = "run_id, question_id, source_id, admitted, reason, recorded_at"
    with closing(sqlite3.connect(first)) as a, closing(sqlite3.connect(second)) as b:
        rows_a = a.execute(f"SELECT {columns} FROM source_uses").fetchall()
        rows_b = b.execute(f"SELECT {columns} FROM source_uses ORDER BY use_id").fetchall()
    assert rows_a[0] in rows_b


def test_importing_the_same_export_twice_adds_nothing(
    populated: tuple[Path, Path, str], tmp_path: Path, clock: FixedClock
) -> None:
    _, export, _ = populated
    out = tmp_path / "imported.sqlite"
    first = import_exports([export], out, clock)
    before = _counts(out)
    second = import_exports([export], out, clock)
    assert second.total == 0
    assert second.inserted == {}
    assert _counts(out) == before
    assert second.duplicates == {t: n for t, n in first.inserted.items()}
    # Passing the file twice in one call is the same.
    third = import_exports([export, export], out, clock)
    assert third.total == 0
    assert _counts(out) == before


def test_two_ledgers_merge_into_one(tmp_path: Path, clock: FixedClock) -> None:
    exports = []
    run_ids = []
    for i, post_id in enumerate((100, 200)):
        db = tmp_path / f"l{i}.sqlite"
        run_ids.append(_populate(db, clock, post_id=post_id))
        exports.append(tmp_path / "ex" / f"2026-11-1{i}T000000Z.jsonl")
        export_jsonl(db, exports[-1])
    out = tmp_path / "merged.sqlite"
    report = import_exports(exports, out, clock)
    assert report.files == 2
    with closing(sqlite3.connect(out)) as conn:
        assert {r[0] for r in conn.execute("SELECT run_id FROM runs")} == set(run_ids)
        assert {r[0] for r in conn.execute("SELECT post_id FROM questions")} == {100, 200}
        assert conn.execute("SELECT COUNT(*) FROM source_uses").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 6
        assert conn.execute("SELECT COUNT(*) FROM call_sources").fetchone()[0] == 2
    assert scan_ledger(out) == []


def test_a_backtest_ledger_round_trips(tmp_path: Path, clock: FixedClock) -> None:
    import asyncio

    *_, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    asyncio.run(run_backtest(db, questions))
    export = tmp_path / "bt.jsonl"
    export_jsonl(db, export)
    out = tmp_path / "bt-imported.sqlite"
    import_exports([export], out, FixedClock(datetime(2026, 1, 1, tzinfo=UTC)))
    assert _counts(out) == _counts(db)
    assert scan_ledger(out) == []


# -- refusals ------------------------------------------------------------------


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_late_admitted_source_is_refused_and_nothing_is_committed(
    populated: tuple[Path, Path, str], tmp_path: Path, clock: FixedClock
) -> None:
    _, export, _ = populated
    forged = []
    for line in _lines(export):
        obj = json.loads(line)
        if obj["table"] == "sources":
            obj["row"]["published_at"] = "2099-01-01T00:00:00.000000Z"  # after the cutoff
        forged.append(json.dumps(obj, sort_keys=True))
    bad = _write(tmp_path / "forged.jsonl", forged)
    out = tmp_path / "imported.sqlite"
    with pytest.raises(LedgerError, match="leakage"):
        import_exports([bad], out, clock)
    counts = _counts(out)
    assert {t: n for t, n in counts.items() if n and t != "schema_migrations"} == {}


def test_a_refusal_rolls_back_rows_from_earlier_files_too(
    populated: tuple[Path, Path, str], tmp_path: Path, clock: FixedClock
) -> None:
    _, export, _ = populated
    other = tmp_path / "other.sqlite"
    _populate(other, clock, post_id=300)
    good = tmp_path / "ex" / "a-good.jsonl"
    export_jsonl(other, good)
    forged = [
        json.dumps(
            {**obj, "row": {**obj["row"], "published_at": "2099-01-01T00:00:00.000000Z"}}
            if obj["table"] == "sources"
            else obj,
            sort_keys=True,
        )
        for obj in map(json.loads, _lines(export))
    ]
    bad = _write(tmp_path / "ex" / "b-bad.jsonl", forged)
    out = tmp_path / "imported.sqlite"
    with pytest.raises(LedgerError, match="leakage"):
        import_exports([tmp_path / "ex"], out, clock)
    assert bad.exists()
    assert sum(n for t, n in _counts(out).items() if t != "schema_migrations") == 0


def test_an_unknown_table_is_refused(tmp_path: Path, clock: FixedClock) -> None:
    bad = _write(tmp_path / "x.jsonl", [json.dumps({"table": "widgets", "row": {"a": 1}})])
    with pytest.raises(LedgerError, match="'widgets' is not in this ledger's schema"):
        import_exports([bad], tmp_path / "o.sqlite", clock)


def test_an_unknown_column_is_refused(
    populated: tuple[Path, Path, str], tmp_path: Path, clock: FixedClock
) -> None:
    _, export, _ = populated
    lines = _lines(export)
    first_run = next(i for i, line in enumerate(lines) if json.loads(line)["table"] == "runs")
    obj = json.loads(lines[first_run])
    obj["row"]["from_the_future"] = 1
    lines[first_run] = json.dumps(obj, sort_keys=True)
    bad = _write(tmp_path / "x.jsonl", lines)
    out = tmp_path / "o.sqlite"
    with pytest.raises(LedgerError, match="columns this schema lacks \\(from_the_future\\)"):
        import_exports([bad], out, clock)
    assert _counts(out)["runs"] == 0


def test_a_row_violating_a_constraint_is_a_ledger_error(tmp_path: Path, clock: FixedClock) -> None:
    row = {
        "resolution_id": "res_x",
        "question_id": 1,
        "post_id": 1,
        "question_type": "binary",
        "status": "resolved",
        "value": None,  # a resolved row needs a value
        "resolved_at": None,
        "source": "manual",
        "evidence": "e",
        "recorded_at": "2026-11-17T18:00:00.000000Z",
    }
    bad = _write(tmp_path / "x.jsonl", [json.dumps({"table": "resolutions", "row": row})])
    with pytest.raises(LedgerError, match="import refused a resolutions row"):
        import_exports([bad], tmp_path / "o.sqlite", clock)


def test_schema_migrations_rows_in_an_export_are_ignored(
    populated: tuple[Path, Path, str], tmp_path: Path, clock: FixedClock
) -> None:
    _, export, _ = populated
    stripped = _write(tmp_path / "s.jsonl", _without(_lines(export), "schema_migrations"))
    out = tmp_path / "o.sqlite"
    import_exports([stripped], out, clock)
    with closing(sqlite3.connect(out)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert n == _counts(populated[0])["schema_migrations"]


def test_malformed_json_is_a_value_error_not_a_silent_skip(
    tmp_path: Path, clock: FixedClock
) -> None:
    bad = _write(tmp_path / "x.jsonl", ["not json"])
    with pytest.raises(ValueError, match="not JSON"):
        import_exports([bad], tmp_path / "o.sqlite", clock)


# -- finding the files ---------------------------------------------------------


def test_a_directory_is_searched_recursively_in_name_order(tmp_path: Path) -> None:
    root = tmp_path / "ledger-branch"
    for rel in ("b/2026-02.jsonl", "a/2026-03.jsonl", "2026-01.jsonl", "c/d/2026-04.jsonl"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (root / "notes.txt").write_text("x", encoding="utf-8")
    found = export_files([root])
    assert [p.name for p in found] == [
        "2026-01.jsonl",
        "2026-02.jsonl",
        "2026-03.jsonl",
        "2026-04.jsonl",
    ]


def test_files_and_directories_mix_and_keep_argument_order(tmp_path: Path) -> None:
    lone = tmp_path / "z-last.jsonl"
    lone.write_text("", encoding="utf-8")
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.jsonl").write_text("", encoding="utf-8")
    assert [p.name for p in export_files([lone, d])] == ["z-last.jsonl", "a.jsonl"]


def test_a_missing_path_is_a_ledger_error(tmp_path: Path) -> None:
    with pytest.raises(LedgerError, match="no export at"):
        export_files([tmp_path / "nope.jsonl"])


def test_every_ledger_table_is_known_to_the_exporter(populated: tuple[Path, Path, str]) -> None:
    db, _, _ = populated
    assert set(_counts(db)) <= set(APPEND_ONLY_TABLES)
