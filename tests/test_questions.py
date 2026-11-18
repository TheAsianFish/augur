from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from augur.questions import question_from_post


def _post(**question: Any) -> dict[str, Any]:
    q: dict[str, Any] = {
        "id": 11,
        "title": "Will it rain?",
        "type": "binary",
        "status": "open",
        "open_time": "2026-11-17T18:00:00Z",
        "scheduled_close_time": "2026-11-17T21:00:00Z",
        "description": "d",
        "resolution_criteria": "r",
        "fine_print": "",
    }
    q.update(question)
    return {"id": 10, "title": "Will it rain?", "question": q}


def test_binary_post_parses() -> None:
    q = question_from_post(_post())
    assert (q.post_id, q.question_id, q.type) == (10, 11, "binary")
    assert q.url == "https://www.metaculus.com/questions/10/"
    assert q.open_time == datetime(2026, 11, 17, 18, tzinfo=UTC)
    assert q.is_open(datetime(2026, 11, 17, 19, tzinfo=UTC))
    assert not q.is_open(datetime(2026, 11, 17, 21, tzinfo=UTC))


def test_already_forecast_reads_my_forecasts() -> None:
    at = datetime(2026, 11, 17, 19, tzinfo=UTC)
    assert not question_from_post(_post(my_forecasts={"latest": None})).already_forecast(at)
    standing = {"latest": {"start_time": "2026-11-17T18:30:00Z", "forecast_values": [0.4, 0.6]}}
    assert question_from_post(_post(my_forecasts=standing)).already_forecast(at)
    withdrawn = {
        "latest": {
            "start_time": "2026-11-17T18:30:00Z",
            "end_time": "2026-11-17T18:40:00Z",
            "forecast_values": [0.4, 0.6],
        }
    }
    assert not question_from_post(_post(my_forecasts=withdrawn)).already_forecast(at)


def test_my_forecasts_times_are_epoch_seconds_in_the_live_api() -> None:
    # Shape observed on 2026-11-17 after a real submission.
    at = datetime(2026, 11, 17, 21, tzinfo=UTC)
    latest = {"start_time": 1794946058.250325, "end_time": None, "forecast_values": [0.95, 0.05]}
    q = question_from_post(_post(my_forecasts={"latest": latest}))
    assert q.my_latest_forecast is not None
    assert q.my_latest_forecast.start_time == datetime.fromtimestamp(1794946058.250325, tz=UTC)
    assert q.already_forecast(at)
    withdrawn = latest | {"end_time": 1794946100.0}
    assert not question_from_post(_post(my_forecasts={"latest": withdrawn})).already_forecast(at)


def test_unknown_type_is_other_and_group_posts_are_refused() -> None:
    assert question_from_post(_post(type="conditional_thing")).type == "other"
    with pytest.raises(ValueError, match="group or conditional"):
        question_from_post({"id": 1, "group_of_questions": {}})
