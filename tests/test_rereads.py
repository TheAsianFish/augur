"""M14's synthetic re-reading simulation: offline and fast."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from augur import rereads

SEASONS = 2_000


def _tables() -> list[rereads.Table]:
    return rereads.simulate(round_sizes=(30,), seasons=SEASONS)


def test_shares_non_decreasing() -> None:
    (t,) = _tables()
    for series in (t.cumulative, t.separate):
        assert len(series) == rereads.ROUNDS
        assert all(a <= b for a, b in itertools.pairwise(series))


def test_first_look_is_shared() -> None:
    (t,) = _tables()
    assert t.cumulative[0] == t.separate[0]


def test_first_round_near_nominal() -> None:
    (t,) = _tables()
    assert abs(t.cumulative[0] - rereads.NOMINAL) < 0.02


def test_separate_near_analytic() -> None:
    (t,) = _tables()
    assert abs(t.separate[3] - rereads.analytic_separate(4)) < 0.03


def test_cumulative_below_separate_and_above_first() -> None:
    (t,) = _tables()
    assert t.cumulative[3] > t.cumulative[0]
    assert all(c <= s for c, s in zip(t.cumulative, t.separate, strict=True))
    assert t.cumulative[3] < t.separate[3]


def test_same_seed_identical() -> None:
    assert _tables() == _tables()


def test_snapshot_text_byte_stable_and_labelled() -> None:
    snap: dict[str, object] = {
        "label": "synthetic",
        "tables": [{"n": 30, "cumulative": [0.05], "separate": [0.05]}],
    }
    a = rereads.snapshot_text(snap)
    assert a == rereads.snapshot_text(json.loads(a))
    assert a.endswith("\n")
    assert json.loads(a)["label"] == "synthetic"


def test_cli_writes_then_checks_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from augur.cli import main

    small = rereads.simulate(seasons=500)
    monkeypatch.setattr(rereads, "simulate", lambda: small)
    out = tmp_path / "rereads.json"
    assert main(["rereads", "--out", str(out)]) == 0
    assert main(["rereads", "--out", str(out), "--check"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["label"] == "synthetic"
    out.write_text("{}\n", encoding="utf-8")
    assert main(["rereads", "--out", str(out), "--check"]) == 1
    assert "differs from a fresh simulation" in capsys.readouterr().out
