"""The workflow's standard-library poll, offline: the HTTP layer is a fake."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "poll.py"
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
TOKEN = "tok_poll_0123456789abcdef"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("augur_poll_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


poll_script = _load()


class FakeApi:
    """Serves a listing per tournament and a detail per post; records each request."""

    def __init__(
        self,
        listings: dict[str, list[dict[str, Any]]],
        details: dict[int, dict[str, Any]],
        fail: set[str] | None = None,
    ) -> None:
        self.listings, self.details, self.fail = listings, details, fail or set()
        self.urls: list[str] = []
        self.headers: list[Mapping[str, str]] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> Any:
        self.urls.append(url)
        self.headers.append(headers)
        if "/posts/?" in url:
            tournament = url.split("tournaments=")[1].split("&")[0]
            if tournament in self.fail:
                raise poll_script.PollError("GET /api/posts/ returned HTTP 503")
            return {"results": self.listings.get(tournament, [])}
        post_id = int(url.rstrip("/").rsplit("/", 1)[1])
        return self.details[post_id]


def _post(post_id: int, latest: dict[str, Any] | None, close: datetime) -> dict[str, Any]:
    question: dict[str, Any] = {
        "id": post_id + 1,
        "scheduled_close_time": close.isoformat().replace("+00:00", "Z"),
    }
    if latest is not None:
        question["my_forecasts"] = {"latest": latest}
    return {"id": post_id, "question": question}


def _api(fail: set[str] | None = None) -> FakeApi:
    soon, later = NOW + timedelta(hours=2), NOW + timedelta(days=1)
    posts = [
        _post(1, None, later),  # never forecast
        _post(2, {"forecast_values": [0.4, 0.6], "end_time": None}, soon),  # current
        _post(3, {"forecast_values": [0.4, 0.6], "end_time": NOW.timestamp() - 60}, soon),
        _post(4, {"forecast_values": None}, later),  # withdrawn
        _post(5, {"forecast_values": [0.5], "end_time": "2026-12-01T00:00:00Z"}, soon),
        {"id": 6, "group_of_questions": {}},  # a group post: not forecast by the bot
    ]
    return FakeApi({"minibench": posts}, {p["id"]: p for p in posts if "question" in p}, fail)


def _env(**extra: str) -> dict[str, str]:
    return {"METACULUS_TOKEN": TOKEN, **extra}


def test_counts_unforecast_questions_and_reports_soonest_close() -> None:
    api = _api()
    sleeps: list[float] = []
    has_work, tournaments, lines = poll_script.poll(
        _env(AUGUR_SCHEDULE="minibench"), api, sleeps.append, NOW
    )
    assert has_work is True
    assert tournaments == ["minibench"]
    assert lines == ["minibench: 5 open, 3 not yet forecast, closes soonest 2026-11-18T14:00Z"]
    assert len(api.urls) == 6  # one listing, five details
    assert len(sleeps) == 5
    assert "statuses=open" in api.urls[0]
    assert "forecast_type=binary%2Cmultiple_choice%2Cnumeric%2Cdiscrete" in api.urls[0]
    assert api.headers[0]["Authorization"] == f"Token {TOKEN}"
    assert api.headers[0]["User-Agent"].startswith("augur-forecast-poll")


def test_everything_forecast_means_no_work() -> None:
    done = {"forecast_values": [0.3, 0.7]}
    posts = [_post(1, done, NOW + timedelta(hours=1))]
    api = FakeApi({"minibench": posts, "33121": []}, {1: posts[0]})
    has_work, tournaments, lines = poll_script.poll(
        _env(AUGUR_SCHEDULE="all"), api, lambda _s: None, NOW
    )
    assert (has_work, tournaments) == (False, [])
    assert lines == ["minibench: 1 open, 0 not yet forecast", "33121: 0 open, 0 not yet forecast"]


@pytest.mark.parametrize("schedule", ["off", "", "weekly", "OFF"])
def test_off_or_unknown_schedule_makes_no_request(schedule: str) -> None:
    api = _api()
    has_work, _, lines = poll_script.poll(_env(AUGUR_SCHEDULE=schedule), api, lambda _s: None)
    assert has_work is False
    assert api.urls == []
    assert any("warning" in line for line in lines) == (schedule == "weekly")


def test_no_token_is_no_work_with_a_message() -> None:
    api = _api()
    has_work, _, lines = poll_script.poll({"AUGUR_SCHEDULE": "minibench"}, api, lambda _s: None)
    assert has_work is False
    assert api.urls == []
    assert "METACULUS_TOKEN" in lines[0]


def test_api_error_counts_as_work_and_never_shows_the_token() -> None:
    api = _api(fail={"minibench"})
    has_work, tournaments, lines = poll_script.poll(
        _env(AUGUR_SCHEDULE="minibench"), api, lambda _s: None, NOW
    )
    assert (has_work, tournaments) == (True, ["minibench"])
    assert lines[0].startswith("warning: minibench: GET /api/posts/ returned HTTP 503")
    assert TOKEN not in "\n".join(lines)


def test_override_polls_the_named_tournaments() -> None:
    api = FakeApi({"bot-testing-area": []}, {})
    _, _, lines = poll_script.poll(
        _env(AUGUR_SCHEDULE="all", AUGUR_POLL_TOURNAMENTS=" bot-testing-area ,"),
        api,
        lambda _s: None,
        NOW,
    )
    assert lines == ["bot-testing-area: 0 open, 0 not yet forecast"]
    assert len(api.urls) == 1


def test_all_polls_the_futureeval_project_named_by_the_variable() -> None:
    api = FakeApi({"minibench": [], "spring-futureeval-2027": []}, {})
    _, _, lines = poll_script.poll(
        _env(AUGUR_SCHEDULE="all", AUGUR_FUTUREEVAL=" Spring-FutureEval-2027 "),
        api,
        lambda _s: None,
        NOW,
    )
    assert lines == [
        "minibench: 0 open, 0 not yet forecast",
        "spring-futureeval-2027: 0 open, 0 not yet forecast",
    ]


def test_a_malformed_futureeval_variable_falls_back_to_fall_2026() -> None:
    api = FakeApi({"minibench": [], "33121": []}, {})
    _, _, lines = poll_script.poll(
        _env(AUGUR_SCHEDULE="all", AUGUR_FUTUREEVAL="33121&x=1"), api, lambda _s: None, NOW
    )
    assert lines[0].startswith("warning: AUGUR_FUTUREEVAL=")
    assert lines[1:] == [
        "minibench: 0 open, 0 not yet forecast",
        "33121: 0 open, 0 not yet forecast",
    ]


def test_minibench_schedule_ignores_the_futureeval_variable() -> None:
    api = FakeApi({"minibench": []}, {})
    _, _, lines = poll_script.poll(
        _env(AUGUR_SCHEDULE="minibench", AUGUR_FUTUREEVAL="99999"), api, lambda _s: None, NOW
    )
    assert lines == ["minibench: 0 open, 0 not yet forecast"]


def test_main_writes_github_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.txt"
    env = _env(AUGUR_SCHEDULE="minibench", GITHUB_OUTPUT=str(out))
    assert poll_script.main(env, _api(fail={"minibench"})) == 0
    assert out.read_text(encoding="utf-8") == "has_work=true\ntournaments=minibench\n"
    assert TOKEN not in capsys.readouterr().out


def test_http_errors_name_the_path_but_not_the_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    import urllib.request

    def refuse(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", None, None)  # type: ignore[arg-type]  # HTTPError accepts no headers

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(poll_script.PollError) as caught:
        poll_script.http_get_json(
            f"{poll_script.API}/posts/1/", {"Authorization": f"Token {TOKEN}"}
        )
    assert str(caught.value) == "GET /api/posts/1/ returned HTTP 401"
    assert caught.value.__cause__ is None
