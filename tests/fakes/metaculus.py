"""An in-process fake of the four Metaculus endpoints the bot calls.

It validates writes the way the real API does (as far as observed), records
every request, and updates ``my_forecasts`` after a forecast so that a second
detail fetch shows the standing forecast.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from augur.clock import Clock, SystemClock

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "metaculus"
BINARY_DETAIL = "post_43327_binary_detail.json"
MINIBENCH_NUMERIC = "post_46616_minibench_numeric_list.json"
BOT_TESTING_NUMERIC = "post_46224_bot_testing_numeric_list.json"

FAKE_TOKEN = "fake-metaculus-token-0123456789"
_DETAIL = re.compile(r"^/posts/(\d+)/$")


def load_fixture(name: str) -> dict[str, Any]:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def binary_post(
    post_id: int, question_id: int, *, title: str | None = None, status: str = "open"
) -> dict[str, Any]:
    """A copy of the real binary fixture with new ids (for pagination tests)."""
    post = load_fixture(BINARY_DETAIL)
    post["id"] = post_id
    q = post["question"]
    q["id"] = question_id
    q["post_id"] = post_id
    q["status"] = status
    post["status"] = status
    if title is not None:
        post["title"] = title
        q["title"] = title
    return post


def group_post(post_id: int) -> dict[str, Any]:
    """A group post: no single ``question``, sub-questions under ``group_of_questions``."""
    post = load_fixture(BINARY_DETAIL)
    sub = post.pop("question")
    post["id"] = post_id
    post["group_of_questions"] = {"id": post_id + 1, "questions": [sub]}
    return post


def _error(status: int, detail: str | dict[str, Any]) -> httpx.Response:
    body = {"detail": detail} if isinstance(detail, str) else detail
    return httpx.Response(status, json=body)


class FakeMetaculus:
    def __init__(
        self,
        *,
        token: str | None = FAKE_TOKEN,
        rate_limited: int = 0,
        retry_after: str | None = "7",
        clock: Clock | None = None,
        base_path: str = "/api",
    ) -> None:
        self.token = token
        self.rate_limited = rate_limited
        self.retry_after = retry_after
        self.clock: Clock = clock or SystemClock()
        self.base_path = base_path.rstrip("/")
        self.posts: dict[int, dict[str, Any]] = {}
        self.tournaments: dict[int, set[str]] = {}
        self.forecasts: list[dict[str, Any]] = []
        self.comments: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []
        self.unavailable_comments = 0
        """Answer the next N comment posts with 503 without recording them (a
        write the server did not process)."""
        self.lost_forecast_answers = 0
        """Record the next N forecast posts, then answer 503 (a write the server
        processed but failed to acknowledge)."""

    # Setup.

    def add_post(self, post: dict[str, Any], tournament: str | None = None) -> None:
        """Store ``post``; it is listed under ``tournament`` and its project slugs and ids."""
        stored = copy.deepcopy(post)
        post_id = int(stored["id"])
        keys: set[str] = set()
        if tournament is not None:
            keys.add(tournament)
        projects = stored.get("projects") or {}
        for value in projects.values():
            for project in value if isinstance(value, list) else [value]:
                if isinstance(project, dict):
                    keys.update(str(project[k]) for k in ("slug", "id") if k in project)
        q = stored.get("question")
        if isinstance(q, dict):
            q.setdefault("my_forecasts", {"history": [], "latest": None, "score_data": {}})
        self.posts[post_id] = stored
        self.tournaments[post_id] = keys

    def add_fixture(self, name: str, tournament: str | None = None) -> dict[str, Any]:
        post = load_fixture(name)
        self.add_post(post, tournament)
        return post

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self, base_url: str = "https://www.metaculus.com") -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, base_url=base_url)

    # Handler.

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.rate_limited > 0:
            self.rate_limited -= 1
            headers = {"Retry-After": self.retry_after} if self.retry_after is not None else {}
            return httpx.Response(429, headers=headers, json={"detail": "Request was throttled."})
        if self.token is not None and request.headers.get("Authorization") != (
            f"Token {self.token}"
        ):
            return _error(403, "Authentication credentials were not provided.")
        path = request.url.path
        if not path.startswith(self.base_path + "/"):
            return _error(404, "Not found.")
        route = path[len(self.base_path) :]
        if request.method == "GET" and route == "/posts/":
            return self._list(request)
        match = _DETAIL.match(route)
        if request.method == "GET" and match:
            return self._detail(int(match.group(1)))
        if request.method == "POST" and route == "/questions/forecast/":
            response = self._forecast(request)
            if self.lost_forecast_answers > 0 and response.status_code < 300:
                self.lost_forecast_answers -= 1
                return httpx.Response(
                    503, json={"detail": "Service unavailable."}, headers={"Retry-After": "0"}
                )
            return response
        if request.method == "POST" and route == "/comments/create/":
            if self.unavailable_comments > 0:
                self.unavailable_comments -= 1
                return _error(503, "Service unavailable.")
            return self._comment(request)
        return _error(404, "Not found.")

    def _list(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        try:
            limit = int(params.get("limit", "20"))
            offset = int(params.get("offset", "0"))
        except ValueError:
            return _error(400, "limit and offset must be integers")
        tournaments = {t for t in params.get("tournaments", "").split(",") if t}
        statuses = {s for s in params.get("statuses", "").split(",") if s}
        types = {t for t in params.get("forecast_type", "").split(",") if t}
        matched: list[dict[str, Any]] = []
        for post_id, post in self.posts.items():
            if tournaments and not tournaments & self.tournaments[post_id]:
                continue
            if statuses and post.get("status") not in statuses:
                continue
            q = post.get("question")
            if types and isinstance(q, dict) and q.get("type") not in types:
                continue
            listed = copy.deepcopy(post)
            if isinstance(listed.get("question"), dict):
                listed["question"].pop("my_forecasts", None)
            matched.append(listed)
        page = matched[offset : offset + limit]
        nxt = None
        if offset + limit < len(matched):
            query = dict(params)
            query["offset"] = str(offset + limit)
            nxt = f"{request.url.copy_with(query=None)}?{urlencode(query)}"
        return httpx.Response(
            200, json={"count": len(matched), "next": nxt, "previous": None, "results": page}
        )

    def _detail(self, post_id: int) -> httpx.Response:
        post = self.posts.get(post_id)
        if post is None:
            return _error(404, "No Post matches the given query.")
        return httpx.Response(200, json=post)

    def _question(self, question_id: int) -> dict[str, Any] | None:
        for post in self.posts.values():
            q = post.get("question")
            if isinstance(q, dict) and q.get("id") == question_id:
                return q
        return None

    def _forecast(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
        except ValueError:
            return _error(400, "body is not JSON")
        if not isinstance(body, list) or not body:
            return _error(400, "Expected a list of forecasts.")
        accepted: list[tuple[dict[str, Any], float]] = []
        for item in body:
            if not isinstance(item, dict):
                return _error(400, "Each forecast must be an object.")
            qid = item.get("question")
            q = self._question(qid) if isinstance(qid, int) else None
            if q is None:
                return _error(404, {"question": [f"Question {qid!r} does not exist."]})
            if q.get("type") != "binary":
                return _error(400, {"question": ["Question is not binary."]})
            if q.get("status") != "open":
                return _error(400, {"question": ["Question is not open for forecasting."]})
            p = item.get("probability_yes")
            if isinstance(p, bool) or not isinstance(p, int | float) or not 0.001 <= p <= 0.999:
                return _error(400, {"probability_yes": ["Must be between 0.001 and 0.999."]})
            accepted.append((q, float(p)))
        now = self.clock.now().timestamp()  # the live API sends epoch seconds here
        for (q, p), item in zip(accepted, body, strict=True):
            latest = {"start_time": now, "end_time": None, "forecast_values": [1 - p, p]}
            mine = q.setdefault("my_forecasts", {"history": [], "latest": None, "score_data": {}})
            previous = mine.get("latest")
            if isinstance(previous, dict):
                previous["end_time"] = now
                mine["history"].append(previous)
            mine["latest"] = latest
            self.forecasts.append(item)
        return httpx.Response(201, json={})

    def _comment(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
        except ValueError:
            return _error(400, "body is not JSON")
        if not isinstance(body, dict):
            return _error(400, "Expected an object.")
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return _error(400, {"text": ["This field may not be blank."]})
        on_post = body.get("on_post")
        if not isinstance(on_post, int) or on_post not in self.posts:
            return _error(400, {"on_post": [f"Post {on_post!r} does not exist."]})
        if not isinstance(body.get("is_private"), bool):
            return _error(400, {"is_private": ["Must be a valid boolean."]})
        if not isinstance(body.get("included_forecast", False), bool):
            return _error(400, {"included_forecast": ["Must be a valid boolean."]})
        self.comments.append(body)
        comment_id = len(self.comments)
        return httpx.Response(
            201, json={"id": comment_id, "on_post": on_post, "is_private": body["is_private"]}
        )
