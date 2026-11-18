"""Offline tests of the measurement scripts under scripts/."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_measure_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qw = _load("question_windows")
cd = _load("cron_delay")


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# --- question windows -------------------------------------------------------


def _post(pid: int, qtype: str, opened: str, closes: str) -> dict[str, Any]:
    return {
        "id": pid,
        "status": "closed",
        "question": {
            "id": pid + 1,
            "type": qtype,
            "open_time": opened,
            "scheduled_close_time": closes,
        },
    }


def test_percentile_interpolates() -> None:
    assert qw.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert qw.percentile([5.0], 90) == 5.0
    assert qw.percentile([0.0, 10.0], 90) == 9.0


def test_questions_from_posts_flattens_groups_and_skips_undated() -> None:
    posts: list[dict[str, Any]] = [
        _post(1, "binary", "2026-01-01T01:00:00Z", "2026-01-01T04:00:00Z"),
        {
            "id": 10,
            "open_time": "2026-01-01T00:00:00Z",
            "scheduled_close_time": "2026-01-01T03:00:00Z",
            "group_of_questions": {"questions": [{"id": 11, "type": "numeric"}]},
        },
        {"id": 20, "question": {"id": 21, "type": "binary"}},
    ]
    rows = qw.questions_from_posts(posts, "r1")
    assert [r["question_id"] for r in rows] == [11, 2]
    assert rows[0]["open_time"] == "2026-01-01T00:00:00Z"


def test_window_stats_on_synthetic_round() -> None:
    posts = [
        _post(1, "binary", "2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z"),
        _post(3, "binary", "2026-01-01T01:00:00Z", "2026-01-01T04:00:00Z"),
        _post(5, "numeric", "2026-01-01T03:00:00Z", "2026-01-01T04:30:00Z"),
        _post(7, "binary", "2026-01-02T03:00:00Z", "2026-01-02T06:00:00Z"),
    ]
    rows = qw.questions_from_posts(posts, "r1")
    stats = qw.window_stats(rows, {"r1": _t("2026-01-01T00:00:00")}, days=(1, 2))
    assert stats["by_type"] == {"binary": 3, "numeric": 1}
    assert stats["window_hours"]["min"] == 1.5
    assert stats["window_hours"]["median"] == 3.0
    gaps = stats["gap_between_consecutive_opens_hours"]
    assert (gaps["min"], gaps["median"], gaps["max"]) == (1.0, 2.0, 24.0)
    assert stats["opens_by_utc_hour"]["3"] == 2
    share = stats["rounds"]["r1"]["share_opened_in_first_n_days"]
    assert share == {"1": 0.75, "2": 1.0}


def test_window_stats_anchors_at_an_early_open() -> None:
    posts = [
        _post(1, "binary", "2025-12-31T12:00:00Z", "2025-12-31T15:00:00Z"),
        _post(3, "binary", "2026-01-01T18:00:00Z", "2026-01-01T21:00:00Z"),
    ]
    rows = qw.questions_from_posts(posts, "r1")
    stats = qw.window_stats(rows, {"r1": _t("2026-01-01T00:00:00")}, days=(1,))
    assert stats["rounds"]["r1"]["anchor"] == "2025-12-31T12:00:00Z"
    assert stats["rounds"]["r1"]["share_opened_in_first_n_days"] == {"1": 0.5}


def test_client_error_text_has_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def boom(*_a: object, **_k: object) -> None:
        raise urllib.error.HTTPError("https://x", 500, "err", None, None)  # type: ignore[arg-type]  # HTTPError accepts None headers at runtime

    monkeypatch.setattr(qw.urllib.request, "urlopen", boom)
    client = qw.Client("secret-token-value", interval=0)
    with pytest.raises(qw.FetchError) as info:
        client.get("/posts/")
    assert "secret-token-value" not in str(info.value)


# --- cron parser and delays ---------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("*/15", {0, 15, 30, 45}),
        ("7,27,47", {7, 27, 47}),
        ("2-59/5", set(range(2, 60, 5))),
        ("5/20", {5, 25, 45}),
        ("1-3,10", {1, 2, 3, 10}),
        ("0", {0}),
    ],
)
def test_cron_minute_field(spec: str, expected: set[int]) -> None:
    assert set(cd._field(spec, 0, 59)) == expected


@pytest.mark.parametrize("expr", ["60 * * * *", "a * * * *", "0 0 * * 1", "0 0 */2 * *", "* *"])
def test_cron_rejects_what_it_does_not_model(expr: str) -> None:
    with pytest.raises(cd.UnsupportedCron):
        cd.Cron.parse(expr)


def test_schedule_union_and_intervals() -> None:
    sched = cd.Schedule([cd.Cron.parse("0 */6 * * *"), cd.Cron.parse("30 3 * * *")])
    assert sched.day == [0, 210, 360, 720, 1080]
    assert sched.min_interval() == 150
    assert cd.Schedule([cd.Cron.parse("7,27,47 * * * *")]).intervals() == [20] * 72


def test_tick_at_or_before_wraps_to_the_previous_day() -> None:
    sched = cd.Schedule([cd.Cron.parse("7 * * * *")])
    assert sched.tick_at_or_before(_t("2026-01-02T00:05:00")) == _t("2026-01-01T23:07:00")
    assert sched.tick_at_or_before(_t("2026-01-02T00:07:00")) == _t("2026-01-02T00:07:00")
    ticks = sched.ticks_between(_t("2026-01-01T22:00:00"), _t("2026-01-02T01:07:00"))
    assert [t.hour for t in ticks] == [22, 23, 0, 1]


def test_analyse_runs_delays_missed_and_other_triggers() -> None:
    sched = cd.Schedule([cd.Cron.parse("0 * * * *")])
    runs = [
        (_t("2026-01-01T00:03:00"), 1),
        (_t("2026-01-01T01:10:00"), 2),
        (_t("2026-01-01T03:02:00"), 5),
    ]
    stats = cd.analyse_runs(sched, runs)
    assert stats["delays_min"] == [3.0, 10.0, 2.0]
    assert (stats["expected_ticks"], stats["missed_ticks"], stats["ticks_with_a_run"]) == (4, 1, 3)
    assert stats["coverage"] == 0.75
    assert stats["missed_isolated"] == 1
    assert stats["other_event_runs_between"] == 2
    assert stats["gaps_min"] == [67.0, 112.0]


def test_analyse_runs_attributes_a_long_delay_to_the_later_tick() -> None:
    sched = cd.Schedule([cd.Cron.parse("*/20 * * * *")])
    stats = cd.analyse_runs(sched, [(_t("2026-01-01T00:05:00"), 1), (_t("2026-01-01T00:50:00"), 2)])
    # 00:50 is 50 min after 00:00 but is read as 10 min after 00:40.
    assert stats["delays_min"] == [5.0, 10.0]
    assert stats["missed_ticks"] == 1
    assert stats["longest_missed_streak"] == 1


def test_analyse_runs_separates_long_gaps() -> None:
    sched = cd.Schedule([cd.Cron.parse("0 * * * *")])
    runs = [(_t("2026-01-01T00:01:00"), 1), (_t("2026-01-01T08:01:00"), 2)]
    stats = cd.analyse_runs(sched, runs, long_gap=6)
    assert stats["missed_ticks"] == 7
    assert stats["missed_in_long_gaps"] == 7
    assert stats["missed_isolated"] == 0


def test_parse_workflow_crons() -> None:
    text = """
on:
  schedule:
    - cron: "7,27,47 * * * *" # every 20 min
    #- cron: '0 * * * *'
    - cron: '30  3 * * *'
    - cron: 15 4 * * *   # unquoted
"""
    assert cd.parse_workflow_crons(text) == ["7,27,47 * * * *", "30 3 * * *", "15 4 * * *"]
