from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from augur.clock import FixedClock
from augur.export import export_jsonl, main, read_jsonl
from augur.leakage import RunMode, admit_source
from augur.ledger import Ledger

from .conftest import T0
from .helpers import binary_question, source

COMMITTED = sorted((Path(__file__).resolve().parent.parent / "data" / "ledgers").glob("*.sqlite"))
FAKE_TOKEN = "tok_9f8e7d6c5b4a39281706f5e4d3c2b1a0"


def _fill(path: Path, clock: FixedClock) -> None:
    with Ledger.open(path, clock) as ledger:
        run_id = ledger.start_run(
            mode=RunMode.DRY_RUN,
            tournament="minibench",
            provider="replay",
            model="qwen2.5:7b",
            settings={"forecasters": 2, "note": "naïve ✓"},
            code_version="test",
        )
        q, post = binary_question(T0 - timedelta(hours=1))
        ledger.record_question(run_id, q, T0, post)
        s = source(T0 - timedelta(days=1), retrieved_at=T0)
        ledger.record_source(s)
        ledger.record_admission(run_id, q.question_id, admit_source(s, T0, RunMode.DRY_RUN))
        for member in (0, 1):
            ledger.record_forecast(
                run_id=run_id,
                question_id=q.question_id,
                kind="member",
                member=member,
                method="draw",
                probability=0.3 + member / 10,
                reasoning="r",
                call_id=None,
            )


def test_export_is_byte_stable_and_counts_rows(
    tmp_path: Path, clock: FixedClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("METACULUS_TOKEN", "OPENROUTER_API_KEY", "ASKNEWS_CLIENT_SECRET"):
        monkeypatch.setenv(name, FAKE_TOKEN)
    db = tmp_path / "l.sqlite"
    _fill(db, clock)
    first, second = tmp_path / "a.jsonl", tmp_path / "out" / "b.jsonl"
    counts = export_jsonl(db, first)
    assert export_jsonl(db, second) == counts
    data = first.read_bytes()
    assert data == second.read_bytes()
    assert counts["runs"] == 1
    assert counts["questions"] == 1
    assert counts["sources"] == 1
    assert counts["source_uses"] == 1
    assert counts["forecasts"] == 2
    assert counts["llm_calls"] == 0
    assert counts["schema_migrations"] == 3
    assert not any(t.startswith("sqlite_") for t in counts)
    assert sum(counts.values()) == data.count(b"\n")
    assert FAKE_TOKEN.encode() not in data
    lines = data.decode("utf-8").splitlines()
    assert json.loads(lines[0])["table"] == "runs"
    assert all(list(json.loads(x)["row"]) == sorted(json.loads(x)["row"]) for x in lines)
    runs = [r for t, r in read_jsonl(first) if t == "runs"]
    assert json.loads(runs[0]["settings_json"])["note"] == "naïve ✓"
    members = [r["member"] for t, r in read_jsonl(first) if t == "forecasts"]
    assert members == [0, 1]


def test_blobs_round_trip_and_without_rowid_tables_export(tmp_path: Path) -> None:
    db = tmp_path / "plain.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE b (k INTEGER PRIMARY KEY AUTOINCREMENT, v BLOB, f REAL)")
    conn.execute("CREATE TABLE w (k TEXT PRIMARY KEY, n INTEGER) WITHOUT ROWID")
    conn.executemany("INSERT INTO b (v, f) VALUES (?, ?)", [(b"\x00\xff", 0.1), (None, None)])
    conn.executemany("INSERT INTO w VALUES (?, ?)", [("z", 1), ("a", 2)])
    conn.commit()
    conn.close()
    out = tmp_path / "plain.jsonl"
    assert export_jsonl(db, out) == {"b": 2, "w": 2}
    rows = list(read_jsonl(out))
    assert rows[0] == ("b", {"k": 1, "v": b"\x00\xff", "f": 0.1})
    assert rows[1] == ("b", {"k": 2, "v": None, "f": None})
    assert [r["k"] for t, r in rows if t == "w"] == ["a", "z"]


@pytest.mark.parametrize("db", COMMITTED, ids=lambda p: p.name)
def test_committed_ledgers_export_identically(tmp_path: Path, db: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    counts = export_jsonl(db, a)
    export_jsonl(db, b)
    assert a.read_bytes() == b.read_bytes()
    assert sum(1 for _ in read_jsonl(a)) == sum(counts.values())


def test_read_jsonl_names_the_bad_line(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"table": "runs", "row": {}}\n[1]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(read_jsonl(bad))


def test_main_exports_and_reports(tmp_path: Path, clock: FixedClock) -> None:
    db = tmp_path / "l.sqlite"
    _fill(db, clock)
    assert main([str(db), str(tmp_path / "d" / "x.jsonl")]) == 0
    assert (tmp_path / "d" / "x.jsonl").is_file()
    assert main([]) == 2
    with pytest.raises(FileNotFoundError, match="no ledger"):
        export_jsonl(tmp_path / "missing.sqlite", tmp_path / "y.jsonl")
