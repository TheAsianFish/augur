"""M14's synthetic re-reading simulation: offline and fast."""

from __future__ import annotations

import itertools
import json

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
