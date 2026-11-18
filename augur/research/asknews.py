"""AskNews news search, bounded so that nothing published after ``as_of`` comes back.

Written against plain ``httpx``, matching what the official ``asknews`` SDK
(version 0.16.1, ``asknews_sdk``) sends; the SDK is not a dependency.

Authentication (``asknews_sdk/security.py``, ``OAuth2ClientCredentials``):
``POST https://auth.asknews.app/oauth2/token`` with
``Content-Type: application/x-www-form-urlencoded``, HTTP Basic auth of
``quote(client_id):quote(client_secret)``, and the body
``grant_type=client_credentials&scope=<space-separated scopes>``; the SDK
always adds ``offline`` and ``openid`` to the requested scopes (its defaults
are ``news stories chat analytics``). The JSON reply carries ``access_token``,
``token_type``, ``scope`` and ``expires_in`` (seconds); calls then send
``Authorization: Bearer <access_token>``, and a 401 makes the SDK fetch a new
token and retry once.

Search (``asknews_sdk/api/news.py``, ``search_news``):
``GET https://api.asknews.app/v1/news/search`` with query parameters
``query``, ``n_articles``, ``start_timestamp``, ``end_timestamp`` (Unix
seconds), ``time_filter`` (``crawl_date`` by default, or ``pub_date``),
``return_type`` (``string``, ``dicts`` or ``both``), ``method`` (``kw``,
``nl`` or ``both``), ``historical`` (search the archive rather than the last
48 hours), ``strategy`` (``default``, ``latest news``, ``news knowledge``) and
``categories`` (default ``All``). The SDK drops ``None`` values and repeats
list values. With ``return_type=dicts`` the reply is
``{"as_dicts": [article, ...], ...}``; each article has ``article_url``,
``eng_title``, ``summary``, ``pub_date`` (timezone-aware ISO 8601),
``source_id`` and more.

Leakage rules: every request carries ``end_timestamp = int(as_of)``; any
article whose ``pub_date`` is later than ``as_of`` is dropped anyway and logged
as a warning with its URL (a provider bug worth seeing). The default
``time_filter`` is ``crawl_date``, the SDK's default and the stricter bound:
an article must have been in the index by ``as_of``, not merely claim an
earlier publication date.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timedelta
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from augur.clock import Clock, ensure_utc, parse_iso
from augur.config import Settings
from augur.errors import MissingCredentialsError, SearchError
from augur.log import get_logger
from augur.ratelimit import RateLimiter
from augur.research.source import DateEvidence, Source
from augur.research.text import truncate_words
from augur.research.wikipedia import USER_AGENT, retry_after_seconds

DEFAULT_API_URL = "https://api.asknews.app"
DEFAULT_TOKEN_URL = "https://auth.asknews.app/oauth2/token"  # noqa: S105 (a URL, not a secret)
DEFAULT_SCOPES = ("news",)
HISTORICAL_AFTER = timedelta(hours=48)
TOKEN_EXPIRY_MARGIN = timedelta(seconds=60)
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

log = get_logger("augur.research.asknews")


class _Article(BaseModel):
    model_config = ConfigDict(extra="ignore")

    article_url: str
    eng_title: str = ""
    title: str = ""
    summary: str = ""
    pub_date: str
    source_id: str = ""


class _SearchReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    as_dicts: list[dict[str, object]] | None = None


class _TokenReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    access_token: SecretStr
    expires_in: int = 0


def _basic_auth(client_id: str, client_secret: str) -> str:
    raw = f"{quote(client_id)}:{quote(client_secret)}".encode()
    return "Basic " + base64.b64encode(raw).decode("ascii")


class AskNewsSearcher:
    """News articles published at or before ``as_of``, from the AskNews API."""

    name = "asknews"

    def __init__(
        self,
        http: httpx.AsyncClient,
        clock: Clock,
        client_id: SecretStr,
        client_secret: SecretStr,
        limiter: RateLimiter | None = None,
        api_url: str = DEFAULT_API_URL,
        token_url: str = DEFAULT_TOKEN_URL,
        scopes: tuple[str, ...] = DEFAULT_SCOPES,
        method: Literal["kw", "nl", "both"] = "kw",
        time_filter: Literal["crawl_date", "pub_date"] = "crawl_date",
        excerpt_chars: int = 2000,
        max_retries: int = 2,
    ) -> None:
        if excerpt_chars < 1:
            raise ValueError("excerpt_chars must be at least 1")
        self._http = http
        self._clock = clock
        self._client_id = client_id
        self._client_secret = client_secret
        # The free tier is tightly limited: one call every ten seconds by default.
        self._limiter = limiter or RateLimiter(rate_per_second=0.1, burst=1, max_concurrent=1)
        self._api_url = api_url.rstrip("/")
        self._token_url = token_url
        self._scope = " ".join(dict.fromkeys(("offline", "openid", *scopes)))
        self._method = method
        self._time_filter = time_filter
        self._excerpt_chars = excerpt_chars
        self._max_retries = max_retries
        self._token: SecretStr | None = None
        self._token_expires: datetime | None = None
        self._token_lock = asyncio.Lock()

    @classmethod
    def from_settings(
        cls, settings: Settings, http: httpx.AsyncClient, clock: Clock
    ) -> AskNewsSearcher:
        """A searcher with the credentials from ``settings``; raises if they are absent."""
        if settings.asknews_client_id is None or settings.asknews_client_secret is None:
            raise MissingCredentialsError(
                "AskNews search", ("ASKNEWS_CLIENT_ID", "ASKNEWS_CLIENT_SECRET")
            )
        return cls(http, clock, settings.asknews_client_id, settings.asknews_client_secret)

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        """Articles matching ``query`` published at or before ``as_of``."""
        as_of = ensure_utc(as_of)
        if limit < 1:
            return []
        end_timestamp = int(as_of.timestamp())
        historical = self._clock.now() - as_of > HISTORICAL_AFTER
        params = {
            "query": query,
            "n_articles": str(limit),
            "end_timestamp": str(end_timestamp),
            "time_filter": self._time_filter,
            "return_type": "dicts",
            "method": self._method,
            "historical": "true" if historical else "false",
            "strategy": "default",
            "categories": "All",
        }
        reply = await self._search_request(query, params)
        sources: list[Source] = []
        for raw in reply.as_dicts or []:
            found = self._to_source(raw, query, as_of, end_timestamp)
            if found is not None:
                sources.append(found)
        return sources[:limit]

    def _to_source(
        self, raw: dict[str, object], query: str, as_of: datetime, end_timestamp: int
    ) -> Source | None:
        try:
            article = _Article.model_validate(raw)
            published_at = parse_iso(article.pub_date)
        except (ValidationError, ValueError):
            log.warning("asknews_article_unusable", url=str(raw.get("article_url", "")))
            return None
        if published_at > as_of:
            log.warning(
                "asknews_article_after_as_of",
                url=article.article_url,
                pub_date=published_at.isoformat(),
                as_of=as_of.isoformat(),
                end_timestamp=end_timestamp,
            )
            return None
        excerpt = article.summary.strip()
        if not excerpt:
            return None
        return Source.build(
            searcher=self.name,
            query=query,
            url=article.article_url,
            title=article.eng_title or article.title,
            excerpt=truncate_words(excerpt, self._excerpt_chars),
            published_at=published_at,
            retrieved_at=self._clock.now(),
            as_of=as_of,
            date_evidence=DateEvidence.PROVIDER_PUBLISHED_BOUNDED,
            evidence_detail=(
                f"pub_date={article.pub_date} end_timestamp={end_timestamp} "
                f"time_filter={self._time_filter} source_id={article.source_id}"
            ),
        )

    async def _search_request(self, query: str, params: dict[str, str]) -> _SearchReply:
        url = f"{self._api_url}/v1/news/search"
        last_problem = "no attempt made"
        reauthorised = False
        attempt = 0
        while attempt <= self._max_retries:
            attempt += 1
            token = await self._access_token()
            async with self._limiter.acquire():
                try:
                    response = await self._http.get(
                        url,
                        params=params,
                        headers={
                            "Authorization": f"Bearer {token.get_secret_value()}",
                            "Accept": "application/json",
                            "User-Agent": USER_AGENT,
                        },
                    )
                except httpx.HTTPError as exc:
                    last_problem = type(exc).__name__
                    log.warning("asknews_transport_error", attempt=attempt, error=last_problem)
                    continue
            if response.status_code == 401 and not reauthorised:
                reauthorised = True
                attempt -= 1
                self._token = None
                log.info("asknews_token_rejected_refreshing")
                continue
            if response.status_code in _RETRY_STATUSES:
                wait = retry_after_seconds(response, default=10.0)
                last_problem = f"HTTP {response.status_code}"
                log.warning("asknews_retry", status=response.status_code, wait=wait)
                self._limiter.pause(wait)
                continue
            if response.status_code in (401, 403):
                raise SearchError(
                    f"AskNews refused the search for {query!r} (HTTP {response.status_code}); "
                    "check that the AskNews credentials are valid and include the news scope"
                )
            if response.status_code != 200:
                raise SearchError(
                    f"AskNews search for {query!r} failed with HTTP {response.status_code}; "
                    "retry later or run without the AskNews searcher"
                )
            try:
                return _SearchReply.model_validate(response.json())
            except (ValueError, ValidationError):
                raise SearchError(
                    f"AskNews search for {query!r} returned a reply that is not the expected "
                    "JSON; check the API URL or retry later"
                ) from None
        raise SearchError(
            f"AskNews search for {query!r} failed after {self._max_retries + 1} attempts "
            f"({last_problem}); retry later or run without the AskNews searcher"
        )

    async def _access_token(self) -> SecretStr:
        async with self._token_lock:
            now = self._clock.now()
            if (
                self._token is not None
                and self._token_expires is not None
                and now < self._token_expires
            ):
                return self._token
            token, expires_in = await self._fetch_token()
            self._token = token
            self._token_expires = now + timedelta(seconds=expires_in) - TOKEN_EXPIRY_MARGIN
            return token

    async def _fetch_token(self) -> tuple[SecretStr, int]:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": _basic_auth(
                self._client_id.get_secret_value(), self._client_secret.get_secret_value()
            ),
            "User-Agent": USER_AGENT,
        }
        body = {"grant_type": "client_credentials", "scope": self._scope}
        async with self._limiter.acquire():
            try:
                response = await self._http.post(self._token_url, headers=headers, data=body)
            except httpx.HTTPError as exc:
                raise SearchError(
                    f"could not reach the AskNews token endpoint ({type(exc).__name__}); "
                    "check the network or retry later"
                ) from None
        if response.status_code in (400, 401, 403):
            raise SearchError(
                f"AskNews rejected the client credentials (HTTP {response.status_code}); "
                "check ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET"
            )
        if response.status_code != 200:
            raise SearchError(
                f"the AskNews token endpoint failed with HTTP {response.status_code}; retry later"
            )
        try:
            reply = _TokenReply.model_validate(response.json())
        except (ValueError, ValidationError):
            raise SearchError(
                "the AskNews token endpoint returned an unexpected reply; check the token URL"
            ) from None
        log.info("asknews_token_fetched", expires_in=reply.expires_in)
        return reply.access_token, reply.expires_in
