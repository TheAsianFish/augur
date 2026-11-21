from __future__ import annotations

import json
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


def test_submit_to_minibench_with_a_local_model_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("METACULUS_TOKEN", "not-a-real-token")
    monkeypatch.setenv("AUGUR_PROVIDER", "ollama")
    monkeypatch.setenv("AUGUR_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens; must not be called
    code = main(
        ["forecast", "--tournament", "minibench", "--submit", "--ledger", str(tmp_path / "l.db")]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "frontier-2026-11b" in err
    assert "not-a-real-token" not in err


def test_a_replay_run_never_fetches_the_model_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import augur.cli

    from .fakes.search import FixedSearcher

    async def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("a replay fetched OpenRouter's model list")

    monkeypatch.setattr(augur.cli, "fetch_catalog", no_network)
    monkeypatch.setattr(augur.cli, "WikipediaSearcher", lambda http, clock: FixedSearcher())
    recording = tmp_path / "empty.jsonl"
    recording.write_text("")
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    monkeypatch.setenv("AUGUR_PROVIDER", "replay")
    monkeypatch.setenv("AUGUR_REPLAY_PATH", str(recording))
    coverage = tmp_path / "out" / "coverage.json"
    code = main(["forecast", "--ledger", str(tmp_path / "l.sqlite"), "--coverage", str(coverage)])
    out = capsys.readouterr().out
    assert code == 1, "every draw misses the empty recording, so the question fails"
    assert "1 failed" in out
    report = json.loads(coverage.read_text())
    assert report["failed"] == 1
    assert report["questions"][0]["status"] == "failed"

    exported = tmp_path / "l.jsonl"
    assert main(["ledger", "export", str(tmp_path / "l.sqlite"), str(exported)]) == 0
    tables = {json.loads(line)["table"] for line in exported.read_text().splitlines()}
    assert {"runs", "llm_calls", "question_outcomes"} <= tables


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--retrieval", "open", "--research-from", "x.sqlite"], "reuse recorded research"),
        (["--retrieval", "now", "--first-member", "5"], "reuse recorded research"),
        (["--passes", "2"], "no research to deepen"),
        (["--passes", "2", "--research-from", "x.sqlite"], "no research to deepen"),
    ],
)
def test_backtest_flags_that_would_be_ignored_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: list[str], message: str
) -> None:
    code = main(
        [
            "backtest",
            "forecastbench",
            "--questions",
            str(tmp_path / "q.json"),
            "--resolutions",
            str(tmp_path / "r.json"),
            *flags,
        ]
    )
    assert code == 2
    assert message in capsys.readouterr().err
