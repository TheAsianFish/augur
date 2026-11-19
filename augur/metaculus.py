"""Async client for the Metaculus API: list, read, forecast and comment.

Every request goes through a :class:`RateLimiter`; a 429 or 503 pauses the
limiter for the server's ``Retry-After`` and retries. Errors carry the method,
path, status and a short body excerpt, never a header or the token.
"""

from __future__ import annotations

import math
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr

import augur
from augur.config import Settings
from augur.errors import MetaculusError, MissingCredentialsError
from augur.log import get_logger, redact_text
from augur.questions import Question, question_from_post
from augur.ratelimit import RateLimiter

USER_AGENT = f"augur-forecast/{augur.__version__} (+https://github.com/TheAsianFish/augur)"
MIN_PROBABILITY = 0.001
MAX_PROBABILITY = 0.999
FORECAST_TYPES = "binary,multiple_choice,numeric,discrete"
DEFAULT_RETRY_AFTER = 5.0
MAX_RETRY_AFTER = 120.0
EXCERPT_CHARS = 300
_RETRYABLE = frozenset({429, 503})

_log = get_logger("augur.metaculus")


class SubmitResult(BaseModel):
    """What a write sent and what the server answered."""

    model_config = ConfigDict(frozen=True)

    payload: list[dict[str, Any]] | dict[str, Any]
    status_code: int
    response_excerpt: str


def _next_step(status: int | None) -> str:
    if status is None:
        return "check the network connection and the base URL, then rerun"
    if status in (401, 403):
        return "check METACULUS_TOKEN (a bot account token is required) and its permissions"
    if status == 404:
        return "check the post or question id and the tournament slug"
    if status in _RETRYABLE:
        return "the API is rate limiting or unavailable; wait and rerun"
    if 400 <= status < 500:
        return "the API rejected the request; check the payload against the API docs"
    return "the API failed; rerun later"


def _retry_after(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After")
    try:
        seconds = float(raw) if raw is not None else DEFAULT_RETRY_AFTER
    except ValueError:
        seconds = DEFAULT_RETRY_AFTER
    if not math.isfinite(seconds) or seconds < 0:
        seconds = DEFAULT_RETRY_AFTER
    return min(seconds, MAX_RETRY_AFTER)


def _check_probability(probability: float) -> float:
    value = float(probability)
    if not math.isfinite(value) or not MIN_PROBABILITY <= value <= MAX_PROBABILITY:
        raise ValueError(
            f"probability {probability!r} is outside [{MIN_PROBABILITY}, {MAX_PROBABILITY}]; "
            "clip it before submitting"
        )
    return value


def _payload_kind(body: dict[str, Any]) -> str:
    if body.get("probability_yes") is not None:
        return "binary"
    if body.get("probability_yes_per_category") is not None:
        return "multiple_choice"
    return "continuous"


class MetaculusClient:
    def __init__(
        self,
        base_url: str,
        token: SecretStr | None,
        http: httpx.AsyncClient,
        limiter: RateLimiter | None = None,
        max_retries: int = 3,
        *,
        page_size: int = 100,
        max_pages: int = 50,
    ) -> None:
        if max_retries < 0 or page_size < 1 or max_pages < 1:
            raise ValueError("max_retries must be >= 0 and page_size, max_pages >= 1")
        self._base = base_url.rstrip("/")
        self._site = self._base.removesuffix("/api")
        self._token = token
        self._http = http
        self._limiter = limiter or RateLimiter(rate_per_second=1.0, burst=2, max_concurrent=2)
        self._max_retries = max_retries
        self._page_size = page_size
        self._max_pages = max_pages

    @classmethod
    def from_settings(
        cls, settings: Settings, http: httpx.AsyncClient, limiter: RateLimiter | None = None
    ) -> MetaculusClient:
        return cls(settings.metaculus_base_url, settings.metaculus_token, http, limiter)

    @property
    def has_token(self) -> bool:
        return self._token is not None

    # Payloads, public so that a dry run can print exactly what would be sent.

    @staticmethod
    def binary_payload(question_id: int, probability: float) -> list[dict[str, Any]]:
        return [
            {
                "question": int(question_id),
                "source": "api",
                "probability_yes": _check_probability(probability),
                "probability_yes_per_category": None,
                "continuous_cdf": None,
            }
        ]

    @staticmethod
    def comment_payload(post_id: int, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("comment text is empty; render the reasoning before posting")
        return {
            "text": text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": int(post_id),
        }

    # Reads.

    async def list_open_questions(self, tournament: str) -> list[tuple[Question, dict[str, Any]]]:
        """Open single-question posts in ``tournament`` (slug or id), in hotness order."""
        found: list[tuple[Question, dict[str, Any]]] = []
        seen: set[int] = set()
        for page in range(self._max_pages):
            params: dict[str, str | int] = {
                "limit": self._page_size,
                "offset": page * self._page_size,
                "tournaments": tournament,
                "statuses": "open",
                "forecast_type": FORECAST_TYPES,
                "include_description": "true",
                "order_by": "-hotness",
            }
            data = await self._get_json("/posts/", params)
            results = data.get("results")
            if not isinstance(results, list):
                raise MetaculusError(
                    "GET /posts/ returned no results list; the API format may have changed",
                    None,
                )
            for post in results:
                if not isinstance(post, dict):
                    continue
                post_id = post.get("id")
                if isinstance(post_id, int):
                    if post_id in seen:
                        continue
                    seen.add(post_id)
                try:
                    question = question_from_post(post, site=self._site)
                except (ValueError, KeyError, TypeError) as exc:
                    _log.warning(
                        "metaculus.post_skipped",
                        post_id=post_id,
                        tournament=tournament,
                        reason=str(exc),
                    )
                    continue
                found.append((question, post))
            # ``next`` has been seen non-null on a short last page; trust the count.
            if len(results) < self._page_size:
                break
        else:
            _log.warning(
                "metaculus.max_pages_reached", tournament=tournament, max_pages=self._max_pages
            )
        return found

    @property
    def base_url(self) -> str:
        """The API root, without a trailing slash."""
        return self._base

    async def get_me(self) -> dict[str, Any]:
        """The account the token belongs to (``/users/me/``; ``/user/`` does not exist)."""
        return await self._get_json("/users/me/", None)

    async def get_tournament(self, slug: str) -> dict[str, Any]:
        """A tournament's project record: id, name, dates and score type."""
        return await self._get_json(f"/projects/tournaments/{slug}/", None)

    async def get_leaderboards(self, project_id: int) -> list[dict[str, Any]]:
        """Every leaderboard of a project (the primary one has ``is_primary_leaderboard``)."""
        path = f"/leaderboards/project/{int(project_id)}/"
        response = await self._request("GET", path, params=None)
        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, list) or not all(isinstance(b, dict) for b in data):
            raise MetaculusError(
                f"GET {path} returned no list of leaderboards; the API format may have changed",
                response.status_code,
            )
        return data

    async def list_posts(
        self, tournament: str, statuses: str | None = None
    ) -> list[dict[str, Any]]:
        """Every post of ``tournament`` in the given statuses (comma separated; all if None)."""
        found: list[dict[str, Any]] = []
        seen: set[int] = set()
        for page in range(self._max_pages):
            params: dict[str, str | int] = {
                "limit": self._page_size,
                "offset": page * self._page_size,
                "tournaments": tournament,
            }
            if statuses:
                params["statuses"] = statuses
            data = await self._get_json("/posts/", params)
            results = data.get("results")
            if not isinstance(results, list):
                raise MetaculusError(
                    "GET /posts/ returned no results list; the API format may have changed",
                    None,
                )
            for post in results:
                post_id = post.get("id") if isinstance(post, dict) else None
                if isinstance(post_id, int) and post_id not in seen:
                    seen.add(post_id)
                    found.append(post)
            if len(results) < self._page_size:
                break
        else:
            _log.warning(
                "metaculus.max_pages_reached", tournament=tournament, max_pages=self._max_pages
            )
        return found

    async def get_post(self, post_id: int) -> dict[str, Any]:
        """The post detail, which unlike the list includes ``question.my_forecasts``."""
        return await self._get_json(f"/posts/{int(post_id)}/", None)

    async def get_question(self, post_id: int) -> tuple[Question, dict[str, Any]]:
        post = await self.get_post(post_id)
        try:
            return question_from_post(post, site=self._site), post
        except (ValueError, KeyError, TypeError) as exc:
            raise MetaculusError(
                f"post {post_id} cannot be forecast as a single question ({exc}); skip it", None
            ) from None

    # Writes.

    async def submit_binary(self, question_id: int, probability: float) -> SubmitResult:
        return await self.submit_forecast(self.binary_payload(question_id, probability))

    async def submit_forecast(self, payload: list[dict[str, Any]]) -> SubmitResult:
        """Post a forecast body of any type (see :func:`augur.predictions.forecast_payload`)."""
        if len(payload) != 1 or not isinstance(payload[0].get("question"), int):
            raise ValueError("a forecast payload is one object naming its question id")
        question_id = payload[0]["question"]
        self._require_token()
        response = await self._request("POST", "/questions/forecast/", json=payload)
        _log.info(
            "metaculus.forecast_submitted",
            question_id=question_id,
            kind=_payload_kind(payload[0]),
            status_code=response.status_code,
        )
        return SubmitResult(
            payload=payload,
            status_code=response.status_code,
            response_excerpt=self._excerpt(response),
        )

    async def create_private_comment(self, post_id: int, text: str) -> SubmitResult:
        payload = self.comment_payload(post_id, text)
        self._require_token()
        response = await self._request("POST", "/comments/create/", json=payload, idempotent=False)
        _log.info(
            "metaculus.comment_created",
            post_id=post_id,
            chars=len(text),
            status_code=response.status_code,
        )
        return SubmitResult(
            payload=payload,
            status_code=response.status_code,
            response_excerpt=self._excerpt(response),
        )

    # Transport.

    def _require_token(self) -> None:
        if self._token is None:
            raise MissingCredentialsError("Metaculus submission", ("METACULUS_TOKEN",))

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self._token is not None:
            headers["Authorization"] = f"Token {self._token.get_secret_value()}"
        return headers

    def _excerpt(self, response: httpx.Response) -> str:
        text = response.text[:EXCERPT_CHARS]
        secrets = [self._token.get_secret_value()] if self._token is not None else []
        return redact_text(text, [s for s in secrets if s])

    async def _get_json(self, path: str, params: dict[str, str | int] | None) -> dict[str, Any]:
        response = await self._request("GET", path, params=params)
        try:
            data = response.json()
        except ValueError:
            raise MetaculusError(
                f"GET {path} returned {response.status_code} with a body that is not JSON: "
                f"{self._excerpt(response)!r}; rerun later",
                response.status_code,
            ) from None
        if not isinstance(data, dict):
            raise MetaculusError(
                f"GET {path} returned JSON that is not an object; the API format may have changed",
                response.status_code,
            )
        return data

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        json: Any = None,
        idempotent: bool = True,
    ) -> httpx.Response:
        """Send with retries. A non-idempotent write retries only on 429, which
        means the server refused it; a 503 may have been processed, and retrying
        would post the same comment twice."""
        url = f"{self._base}{path}"
        retry_on = _RETRYABLE if idempotent else frozenset({429})
        attempt = 0
        while True:
            async with self._limiter.acquire():
                try:
                    response = await self._http.request(
                        method, url, params=params, json=json, headers=self._headers()
                    )
                except httpx.TransportError as exc:
                    raise MetaculusError(
                        f"{method} {path} failed before a response ({type(exc).__name__}); "
                        f"{_next_step(None)}",
                        None,
                    ) from None
            status = response.status_code
            if status in retry_on and attempt < self._max_retries:
                attempt += 1
                delay = _retry_after(response)
                _log.warning(
                    "metaculus.retry",
                    method=method,
                    path=path,
                    status_code=status,
                    retry_after=delay,
                    attempt=attempt,
                )
                self._limiter.pause(delay)
                continue
            if response.is_success:
                return response
            retried = f" after {attempt} retries" if attempt else ""
            raise MetaculusError(
                f"{method} {path} returned {status}{retried}: {self._excerpt(response)!r}; "
                f"{_next_step(status)}",
                status,
            )
