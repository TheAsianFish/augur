from __future__ import annotations

from pathlib import Path

import pytest

from augur.cli import main

from .test_leakage import _plant


def test_ledger_check_exits_1_on_a_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from augur.leakage import RunMode

    bad = tmp_path / "live.sqlite"
    _plant(bad, RunMode.LIVE, "late_source")
    assert main(["ledger", "check", str(bad)]) == 1
    assert "late_source" in capsys.readouterr().out


def test_ledger_check_on_a_missing_file_is_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ledger", "check", str(tmp_path / "nope.sqlite")]) == 2
    assert "no ledger at" in capsys.readouterr().err


def test_submit_without_a_token_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    code = main(["forecast", "--submit", "--ledger", str(tmp_path / "l.sqlite")])
    assert code == 2
    assert "METACULUS_TOKEN" in capsys.readouterr().err
