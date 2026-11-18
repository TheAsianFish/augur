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
