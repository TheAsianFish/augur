"""The workflow's standard-library poll, offline: the HTTP layer is a fake."""

from __future__ import annotations

import importlib.util
import json
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


# -- shadow mode (M11) -------------------------------------------------------------


def _export(directory: Path, name: str, rows: list[tuple[str, dict[str, Any]]]) -> None:
    path = directory / "shadow" / "2026-11-18" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for table, row in rows:
            fh.write(json.dumps({"table": table, "row": row}, sort_keys=True) + "\n")


def _aggregate(qid: int) -> tuple[str, dict[str, Any]]:
    return "forecasts", {"forecast_id": f"f{qid}", "question_id": qid, "kind": "aggregate"}


def _member(qid: int) -> tuple[str, dict[str, Any]]:
    return "forecasts", {"forecast_id": f"m{qid}", "question_id": qid, "kind": "member"}


def _call(cid: str, qid: int, day: str = "2026-11-18") -> tuple[str, dict[str, Any]]:
    return "llm_calls", {"call_id": cid, "question_id": qid, "started_at": f"{day}T10:00:00Z"}


def _template(qid: int, day: str = "2026-11-18") -> tuple[str, dict[str, Any]]:
    return "reference_forecasts", {
        "reference_id": f"r{qid}",
        "question_id": qid,
        "forecaster": "template",
        "recorded_at": f"{day}T10:05:00Z",
    }


def _shadow_env(directory: Path, **extra: str) -> dict[str, str]:
    base = {"AUGUR_SCHEDULE": "minibench", "AUGUR_SHADOW": "true"}
    return _env(**(base | {"AUGUR_SHADOW_LEDGER": str(directory / "shadow")} | extra))


def test_shadow_work_is_what_the_shadow_ledger_does_not_hold(tmp_path: Path) -> None:
    # Questions are 2..6 (post id + 1); the ledger holds 2 and 4, and 3 only as a member row.
    _export(tmp_path, "tick1", [_aggregate(2), _aggregate(4), _member(3), _call("c1", 2)])
    api = _api()
    has_work, tournaments, lines = poll_script.poll(
        _shadow_env(tmp_path), api, lambda _s: None, NOW
    )
    assert has_work is True
    assert tournaments == ["minibench"]
    assert "minibench: 5 open, 3 not in the shadow ledger" in lines[-1]
    assert len(api.urls) == 1, "a shadow poll reads the listing alone"


def test_shadow_everything_held_is_no_work_however_my_forecasts_reads(tmp_path: Path) -> None:
    _export(tmp_path, "tick1", [r for q in (2, 3, 4) for r in (_aggregate(q), _template(q))])
    _export(tmp_path, "tick2", [r for q in (5, 6) for r in (_aggregate(q), _template(q))])
    has_work, _, lines = poll_script.poll(_shadow_env(tmp_path), _api(), lambda _s: None, NOW)
    assert has_work is False
    assert lines[-1] == "minibench: 5 open, 0 not in the shadow ledger"


def test_shadow_a_held_question_the_template_missed_is_work(tmp_path: Path) -> None:
    # Every question is held, but the template step (cut off by its timeout) never
    # recorded 5 and 6: the tick runs so `augur baseline --shadow-ledger` catches up.
    _export(tmp_path, "tick1", [r for q in (2, 3, 4) for r in (_aggregate(q), _template(q))])
    _export(tmp_path, "tick2", [_aggregate(5), _aggregate(6)])
    has_work, tournaments, lines = poll_script.poll(
        _shadow_env(tmp_path), _api(), lambda _s: None, NOW
    )
    assert has_work is True
    assert tournaments == ["minibench"]
    assert "minibench: 5 open, 0 not in the shadow ledger, 2 held without a template" in lines[-1]


def test_shadow_with_no_exports_yet_everything_is_work(tmp_path: Path) -> None:
    has_work, _, lines = poll_script.poll(_shadow_env(tmp_path), _api(), lambda _s: None, NOW)
    assert has_work is True
    assert lines[0].startswith("shadow ledger: 0 export(s), 0 question(s) held")


def test_an_unreadable_shadow_export_stops_the_tick(tmp_path: Path) -> None:
    bad = tmp_path / "shadow" / "x.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text("not json\n", encoding="utf-8")
    api = _api()
    has_work, _, lines = poll_script.poll(_shadow_env(tmp_path), api, lambda _s: None, NOW)
    assert has_work is False
    assert "shadow ledger unreadable" in lines[-1]
    assert api.urls == []


def _futureeval_api() -> FakeApi:
    api = _api()
    api.listings["33121"] = [_post(90, None, NOW + timedelta(days=3))]
    return api


def test_shadow_futureeval_waits_while_minibench_has_work(tmp_path: Path) -> None:
    api = _futureeval_api()
    _, tournaments, lines = poll_script.poll(
        _shadow_env(tmp_path, AUGUR_SCHEDULE="all"), api, lambda _s: None, NOW
    )
    assert tournaments == ["minibench"]
    assert "33121: not polled, minibench comes first this tick" in lines
    assert all("tournaments=33121" not in u for u in api.urls)


def test_shadow_futureeval_gets_what_the_day_has_left(tmp_path: Path) -> None:
    _export(tmp_path, "tick1", [r for q in (2, 3, 4, 5, 6) for r in (_aggregate(q), _template(q))])
    _, tournaments, _ = poll_script.poll(
        _shadow_env(tmp_path, AUGUR_SCHEDULE="all"), _futureeval_api(), lambda _s: None, NOW
    )
    assert tournaments == ["33121"]


def test_shadow_futureeval_is_not_polled_past_the_daily_allowance(tmp_path: Path) -> None:
    # 2 calls today on held question 2, 1 template today (10 calls), 1 call yesterday.
    rows = [_aggregate(q) for q in (2, 3, 4, 5, 6)]
    rows += [_template(q, day="2026-11-17") for q in (3, 4, 5, 6)]
    rows += [_call("a", 2), _call("b", 2), _call("old", 3, day="2026-11-17"), _template(2)]
    rows += [_call("a", 2)]  # the same call again, as a tick's second push repeats its first
    _export(tmp_path, "tick1", rows)
    env = _shadow_env(
        tmp_path,
        AUGUR_SCHEDULE="all",
        AUGUR_SHADOW_DAILY_CALLS="80",
        AUGUR_SHADOW_TEMPLATE_CALLS="10",
        AUGUR_MAX_QUESTIONS_PER_RUN="6",
    )
    _, tournaments, lines = poll_script.poll(env, _futureeval_api(), lambda _s: None, NOW)
    # Today: 2 Augur calls + 1 template x 10 = 12; a run: 6 x (1.5 + 10) = 69; 81 > 80.
    assert "today about 12 of 80 free calls used (11.5 a question)" in lines[0]
    assert tournaments == []
    assert any("33121: not polled, a run of 6 would pass" in line for line in lines)
    env["AUGUR_SHADOW_DAILY_CALLS"] = "81"
    _, tournaments, _ = poll_script.poll(env, _futureeval_api(), lambda _s: None, NOW)
    assert tournaments == ["33121"]


def test_shadow_override_polls_exactly_what_was_asked(tmp_path: Path) -> None:
    api = _futureeval_api()
    _, tournaments, _ = poll_script.poll(
        _shadow_env(tmp_path, AUGUR_POLL_TOURNAMENTS="33121"), api, lambda _s: None, NOW
    )
    assert tournaments == ["33121"]
