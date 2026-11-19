"""CLI smoke tests for ``score``, ``resolve`` and ``ledger import``."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from augur.cli import main
from augur.clock import FixedClock, SystemClock
from augur.export import export_jsonl
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.questions import question_from_post

from .conftest import T0
from .fakes.forecastbench import run_backtest, small_set
from .fakes.metaculus import binary_post


def _backtest_ledger(tmp_path: Path) -> tuple[Path, Path]:
    _, rs, questions = small_set(tmp_path)
    db = tmp_path / "bt.sqlite"
    asyncio.run(run_backtest(db, questions))
    return db, rs


def _live_ledger(tmp_path: Path) -> Path:
    db = tmp_path / "live.sqlite"
    with Ledger.open(db, FixedClock(T0)) as ledger:
        run_id = ledger.start_run(
            mode=RunMode.DRY_RUN,
            tournament="minibench",
            provider="replay",
            model="m",
            settings={},
            code_version="t",
        )
        post = binary_post(7, 70)
        ledger.record_question(run_id, question_from_post(post), T0, post)
    return db


def test_score_with_nothing_resolved_exits_1_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _ = _backtest_ledger(tmp_path)
    assert main(["score", str(db)]) == 1
    out = capsys.readouterr().out
    assert "nothing to score: 3 forecast questions have no resolution and 0 were annulled" in out
    assert "run augur resolve first" in out


def test_resolve_forecastbench_then_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, rs = _backtest_ledger(tmp_path)
    assert main(["resolve", "--ledger", str(db), "--forecastbench", str(rs)]) == 0
    resolved = capsys.readouterr().out
    assert "3 checked: 3 resolved" in resolved
    assert "3 reference forecasts" in resolved
    assert main(["score", str(db), "--resamples", "200"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("3 scored questions (binary 3)")
    assert "200 resamples" in out
    assert "vs market (n = 3)" in out
    assert "vs always 50% (n = 3)" in out
    # The scripted model said 30% on each: Brier = (0.7^2 + 0.3^2 + 0.3^2) / 3 = 0.2233.
    assert "Brier          0.2233" in out


def test_score_cluster_and_mode_options(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db, rs = _backtest_ledger(tmp_path)
    main(["resolve", "--ledger", str(db), "--forecastbench", str(rs)])
    capsys.readouterr()
    assert main(["score", str(db), "--cluster", "source", "--resamples", "200"]) == 0
    assert "over source clusters (3)" in capsys.readouterr().out
    assert main(["score", str(db), "--cluster", "week", "--resamples", "200"]) == 0
    assert "over week clusters (1)" in capsys.readouterr().out
    assert main(["score", str(db), "--mode", "live"]) == 1
    assert "nothing to score" in capsys.readouterr().out


def test_score_on_a_missing_ledger_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["score", str(tmp_path / "nope.sqlite")]) == 2
    assert "no ledger at" in capsys.readouterr().err


def test_resolve_manual_without_evidence_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = _live_ledger(tmp_path)
    assert main(["resolve", "--ledger", str(db), "--manual", "7", "yes"]) == 2
    assert "needs --evidence" in capsys.readouterr().err
    with Ledger.open(db, SystemClock()) as led:
        assert led.count("resolutions") == 0


def test_resolve_manual_with_evidence_records_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = _live_ledger(tmp_path)
    code = main(
        ["resolve", "--ledger", str(db), "--manual", "7", "no", "--evidence", "https://x.org/q"]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "post 7: resolved no"


def test_resolve_manual_for_a_post_not_in_the_ledger_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = _live_ledger(tmp_path)
    assert main(["resolve", "--ledger", str(db), "--manual", "8", "yes", "--evidence", "x"]) == 2
    assert "post 8 is not in this ledger" in capsys.readouterr().err


def test_resolve_on_a_missing_ledger_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["resolve", "--ledger", str(tmp_path / "nope.sqlite"), "--manual", "1", "yes"]) == 2
    assert "no ledger at" in capsys.readouterr().err


def test_resolve_from_metaculus_needs_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    db = _live_ledger(tmp_path)
    assert main(["resolve", "--ledger", str(db)]) == 2
    assert "METACULUS_TOKEN" in capsys.readouterr().err


def test_ledger_import_prints_the_row_count_and_is_idempotent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _ = _backtest_ledger(tmp_path)
    export = tmp_path / "ex" / "2026-01-01T000000Z.jsonl"
    counts = export_jsonl(db, export)
    expected = sum(n for t, n in counts.items() if t != "schema_migrations")
    out = tmp_path / "imported.sqlite"
    assert main(["ledger", "import", str(tmp_path / "ex"), "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip() == (
        f"{out}: {expected} rows from 1 exports (0 already present)"
    )
    assert main(["ledger", "import", str(export), "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip() == (
        f"{out}: 0 rows from 1 exports ({expected} already present)"
    )


def test_ledger_import_of_a_missing_path_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["ledger", "import", str(tmp_path / "nope"), "--out", str(tmp_path / "o.sqlite")])
    assert code == 2
    assert "no export at" in capsys.readouterr().err
