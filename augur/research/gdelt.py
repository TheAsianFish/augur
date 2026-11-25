"""Dated news for backtests: discovery through GDELT, text only from the Wayback Machine.

GDELT's DOC 2.0 API indexes news articles with the instant its crawler first
saw each one (``seendate``). It is a discovery index, not a text source: an
article's page today may have been edited since the cutoff. So for one
planned query and a cutoff this searcher:

1. asks DOC 2.0 for an ``artlist`` bounded by ``enddatetime`` at the cutoff
   (``startdatetime`` :data:`LOOKBACK` before it), relevance order;
2. refuses every article whose ``seendate`` is after the cutoff, whatever the
   query bound promised (the bound is never trusted, as with Wayback's
   ``to=``), and every market or forecasting host;
3. takes the text from a Wayback capture of the article's URL stamped at or
   before the cutoff (:meth:`WaybackSearcher.capture`), never from the live
   page. No capture, no source.

The source's date is the capture timestamp (``archive_capture``), which proves
the text; the ``seendate`` is kept in its evidence. Nothing here assumes how
far back GDELT searches: an empty answer is recorded as empty.

GDELT asks for one request every 5 seconds and answers faster callers with
HTTP 429, so every request goes through a 0.2-per-second limiter, retries back
off, and every answer that arrives is cached on disk by its request URL, so a
resumed or repeated run never sends the same request twice.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

from augur.clock import Clock, ensure_utc
from augur.errors import SearchError
from augur.log import get_logger
from augur.ratelimit import RateLimiter
from augur.research.gkg import GkgIndex
from augur.research.source import DateEvidence, Source
from augur.research.wayback import WaybackSearcher, is_excluded_host
from augur.research.wikipedia import USER_AGENT

DEFAULT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
LOOKBACK = timedelta(days=90)
GKG_LOOKBACK = timedelta(days=30)
"""The local GKG index's window (registration item 7): 30 days, not the API's 90."""
MAX_RECORDS = 10
MAX_WORDS = 6
MAX_LOOKUPS = 4
"""Wayback lookups per query before the query is given up."""
REQUEST_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
BACKOFF_SECONDS = (10.0, 30.0, 60.0, 120.0, 240.0)
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
STOP_WORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "will", "into", "over",
        "under", "after", "before", "about", "news", "latest", "update", "updates",
        "what", "when", "which", "who", "how", "are", "was", "were", "has", "have",
        "its", "not", "but", "any", "all", "out", "than", "then", "between",
    }
)  # fmt: skip
_WORD = re.compile(r"[A-Za-z0-9]+")

log = get_logger("augur.research.gdelt")


def gdelt_query(text: str) -> str | None:
    """A planned query as GDELT keywords, or ``None`` if nothing usable is left.

    Punctuation removed, words under three letters and :data:`STOP_WORDS`
    dropped, at most :data:`MAX_WORDS` words, English sources only. GDELT
    rejects short or very common keywords, so they never reach it.
    """
    words: list[str] = []
    seen: set[str] = set()
    for word in _WORD.findall(text):
        low = word.lower()
        if len(word) < 3 or low in STOP_WORDS or low in seen:
            continue
        seen.add(low)
        words.append(word)
        if len(words) == MAX_WORDS:
            break
    if not words:
        return None
    return " ".join(words) + " sourcelang:english"


def gdelt_stamp(instant: datetime) -> str:
    """``instant`` as GDELT's ``YYYYMMDDhhmmss`` UTC stamp."""
    return ensure_utc(instant).strftime("%Y%m%d%H%M%S")


def parse_seendate(raw: str) -> datetime | None:
    """GDELT's ``20260601T123000Z`` as an aware UTC datetime, or ``None`` if malformed."""
    try:
        return datetime.strptime(raw.strip(), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (ValueError, AttributeError):
        return None


def request_url(api_url: str, keywords: str, as_of: datetime) -> str:
    """The exact ``artlist`` request for ``keywords`` bounded at ``as_of``; the cache key."""
    params = {
        "query": keywords,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(MAX_RECORDS),
        "sort": "hybridrel",
        "startdatetime": gdelt_stamp(as_of - LOOKBACK),
        "enddatetime": gdelt_stamp(as_of),
    }
    return f"{api_url}?{urlencode(params)}"


class ResponseCache:
    """Answers GDELT gave, one JSON file per request URL, so nothing is asked twice."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, url: str) -> Path:
        return self._dir / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".json")

    def get(self, url: str) -> str | None:
        path = self._path(url)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("url") != url:
            raise SearchError(f"GDELT cache file {path} belongs to another request; delete it")
        body = record.get("body")
        return body if isinstance(body, str) else None

    def put(self, url: str, body: str, fetched_at: datetime) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        record = {"url": url, "fetched_at": fetched_at.isoformat(), "body": body}
        path = self._path(url)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)


def parse_articles(body: str) -> list[dict[str, Any]]:
    """The ``articles`` of an ``artlist`` answer; an empty or non-JSON answer is none.

    GDELT answers some queries it cannot run (a keyword too short or too
    common, a malformed phrase) with HTTP 200 and a line of text, not JSON.
    """
    text = body.strip()
    if not text:
        return []
    try:
        data: Any = json.loads(text)
    except ValueError:
        log.info("gdelt_not_json", body=text[:200])
        return []
    articles = data.get("articles") if isinstance(data, dict) else None
    return [a for a in articles if isinstance(a, dict)] if isinstance(articles, list) else []


class GdeltSearcher:
    """News articles GDELT saw by ``as_of``, read from Wayback captures at or before it."""

    name = "gdelt"

    def __init__(
        self,
        http: httpx.AsyncClient,
        clock: Clock,
        wayback: WaybackSearcher,
        *,
        cache: ResponseCache | None = None,
        limiter: RateLimiter | None = None,
        api_url: str = DEFAULT_DOC_URL,
        backoff: tuple[float, ...] = BACKOFF_SECONDS,
        max_lookups: int = MAX_LOOKUPS,
        index: GkgIndex | None = None,
    ) -> None:
        self._http = http
        self._clock = clock
        self._wayback = wayback
        self._cache = cache
        self._limiter = limiter or RateLimiter(rate_per_second=0.2, burst=1, max_concurrent=1)
        self._api_url = api_url
        self._backoff = backoff
        self._max_lookups = max_lookups
        self._index = index
        self.api_refused = False
        """Set once the API has refused a request; the index answers from then on."""

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        """Up to ``limit`` news sources for ``query``, each captured at or before ``as_of``."""
        if limit < 1:
            return []
        as_of = ensure_utc(as_of)
        keywords = gdelt_query(query)
        if keywords is None:
            return []
        articles = parse_articles(await self._artlist(keywords, as_of))
        found: list[Source] = []
        lookups = 0
        for article in articles:
            if len(found) >= limit or lookups >= self._max_lookups:
                break
            url = str(article.get("url", "")).strip()
            seen = parse_seendate(str(article.get("seendate", "")))
            host = urlsplit(url).hostname or ""
            if not url.lower().startswith(("http://", "https://")) or not host:
                continue
            if seen is None:
                log.info("gdelt_undated_article", url=url)
                continue
            if seen > as_of:
                # GDELT's bound is not trusted: an article seen after the cutoff is refused.
                log.info("gdelt_article_after_as_of", url=url, seendate=seen.isoformat())
                continue
            if is_excluded_host(host):
                log.info("gdelt_market_host_excluded", url=url)
                continue
            lookups += 1
            try:
                captured = await self._wayback.capture(url, as_of, query)
            except SearchError as exc:
                log.warning("gdelt_capture_failed", url=url, error=str(exc))
                continue
            if captured is None or captured.published_at is None:
                continue
            if captured.published_at > as_of:  # capture() refuses these; checked again
                continue
            found.append(self._source(captured, query, url, seen, article, as_of))
        return found

    def _source(
        self,
        captured: Source,
        query: str,
        url: str,
        seen: datetime,
        article: dict[str, Any],
        as_of: datetime,
    ) -> Source:
        title = str(article.get("title") or "").strip() or captured.title
        domain = str(article.get("domain") or urlsplit(url).hostname or "")
        return Source.build(
            searcher=self.name,
            query=query,
            url=captured.url,
            title=title,
            excerpt=captured.excerpt,
            published_at=captured.published_at,
            retrieved_at=self._clock.now(),
            as_of=as_of,
            date_evidence=DateEvidence.ARCHIVE_CAPTURE,
            evidence_detail=(
                f"{captured.evidence_detail} seendate={seen.strftime('%Y%m%dT%H%M%SZ')} "
                f"domain={domain}"
            ),
        )

    async def _artlist(self, keywords: str, as_of: datetime) -> str:
        """The answer's body: the API's, or, once the API has refused, the local index's."""
        if self._index is not None and self.api_refused:
            return self._index_artlist(keywords, as_of)
        try:
            return await self._api_artlist(keywords, as_of)
        except SearchError as exc:
            if self._index is None:
                raise
            log.warning("gdelt_api_refused_using_gkg_index", error=str(exc))
            self.api_refused = True
            return self._index_artlist(keywords, as_of)

    def _index_artlist(self, keywords: str, as_of: datetime) -> str:
        """The local GKG index's answer, cached like an API answer (registration item 7)."""
        assert self._index is not None
        start = as_of - GKG_LOOKBACK
        key = f"gkg:{keywords}|{gdelt_stamp(start)}|{gdelt_stamp(as_of)}"
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        words = [w for w in keywords.split() if ":" not in w]
        body = self._index.artlist(words, start, as_of)
        if self._cache is not None:
            self._cache.put(key, body, self._clock.now())
        return body

    async def _api_artlist(self, keywords: str, as_of: datetime) -> str:
        """The API's answer, from the cache or GDELT (backing off on 429 and 5xx)."""
        url = request_url(self._api_url, keywords, as_of)
        if self._cache is not None:
            cached = self._cache.get(url)
            if cached is not None:
                return cached
        last_problem = "no attempt made"
        for attempt, wait in enumerate((*self._backoff, None)):
            async with self._limiter.acquire():
                try:
                    response = await self._http.get(
                        url,
                        headers={"User-Agent": USER_AGENT},
                        follow_redirects=False,
                        timeout=REQUEST_TIMEOUT,
                    )
                except httpx.HTTPError as exc:
                    response = None
                    last_problem = f"{type(exc).__name__}: {exc}"
            if response is not None and response.status_code == 200:
                if self._cache is not None:
                    self._cache.put(url, response.text, self._clock.now())
                return response.text
            if response is not None:
                last_problem = f"HTTP {response.status_code}"
                if response.status_code not in _RETRY_STATUSES:
                    break
            if wait is None:
                break
            log.warning("gdelt_retry", problem=last_problem, wait=wait, attempt=attempt)
            self._limiter.pause(wait)
        raise SearchError(
            f"GDELT DOC request for {keywords!r} failed ({last_problem}); GDELT asks for one "
            "request every 5 seconds: retry later from another address, or run without gdelt"
        )
