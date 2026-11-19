"""RESULTS.md's generated blocks: how they are found and replaced, and that the
committed file equals a fresh render (a number in a block cannot drift from its data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from augur import report
from augur.cli import main

REPO = Path(__file__).resolve().parent.parent


def _doc(name: str, body: str) -> str:
    return f"intro\n<!-- augur:begin {name} -->\n{body}<!-- augur:end {name} -->\noutro\n"


def test_render_replaces_only_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(report.BLOCKS, "demo", report.Block("demo", lambda _root: "fresh\n"))
    out = report.render(_doc("demo", "stale\n"), REPO)
    assert out == _doc("demo", "fresh\n")
    # Rendering is idempotent.
    assert report.render(out, REPO) == out


def test_only_renders_the_named_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(report.BLOCKS, "a", report.Block("a", lambda _root: "A\n"))
    monkeypatch.setitem(report.BLOCKS, "b", report.Block("b", lambda _root: "B\n"))
    text = _doc("a", "old\n") + _doc("b", "old\n")
    out = report.render(text, REPO, only=["b"])
    assert "old\n<!-- augur:end a -->" in out
    assert "B\n<!-- augur:end b -->" in out


def test_an_unknown_block_is_an_error() -> None:
    with pytest.raises(report.ReportError, match="cannot render"):
        report.render(_doc("no-such-block", "x\n"), REPO)


def test_check_names_the_stale_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(report.BLOCKS, "demo", report.Block("demo", lambda _root: "fresh\n"))
    path = tmp_path / "RESULTS.md"
    path.write_text(_doc("demo", "stale\n"), encoding="utf-8")
    assert report.check(path, tmp_path) == ["demo"]
    path.write_text(_doc("demo", "fresh\n"), encoding="utf-8")
    assert report.check(path, tmp_path) == []


def test_check_reports_a_missing_or_repeated_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(report.BLOCKS, "demo", report.Block("demo", lambda _root: "fresh\n"))
    path = tmp_path / "RESULTS.md"
    path.write_text("intro, the block deleted\n", encoding="utf-8")
    assert report.check(path, tmp_path, required=["demo"]) == ["demo (missing)"]
    # A stale first copy beside a fresh second one used to pass.
    path.write_text(_doc("demo", "stale\n") + _doc("demo", "fresh\n"), encoding="utf-8")
    assert report.check(path, tmp_path) == ["demo", "demo (twice)"]


def test_check_refuses_while_a_ledger_has_unsaved_rows(tmp_path: Path) -> None:
    ledgers = tmp_path / "data" / "ledgers"
    ledgers.mkdir(parents=True)
    (ledgers / "x.sqlite-wal").write_bytes(b"pending")
    path = tmp_path / "RESULTS.md"
    path.write_text("no blocks\n", encoding="utf-8")
    with pytest.raises(report.ReportError, match=r"x\.sqlite-wal"):
        report.check(path, tmp_path)
    (ledgers / "x.sqlite-wal").write_bytes(b"")
    assert report.check(path, tmp_path) == []


def test_cli_report_writes_then_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(report.BLOCKS, "demo", report.Block("demo", lambda _root: "fresh\n"))
    path = tmp_path / "RESULTS.md"
    path.write_text(_doc("demo", "stale\n"), encoding="utf-8")
    assert main(["report", "--path", str(path), "--check"]) == 1
    assert "stale in: demo" in capsys.readouterr().out
    assert main(["report", "--path", str(path)]) == 0
    assert main(["report", "--path", str(path), "--check"]) == 0


def test_the_committed_results_match_a_fresh_render() -> None:
    stale = report.check(REPO / "RESULTS.md", REPO, required=list(report.BLOCKS))
    assert stale == [], f"RESULTS.md blocks {stale} differ from a render: run `make results`"
