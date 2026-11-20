"""The Internet Archive's Wayback Machine, read as a page stood at an instant.

Wayback has no full-text search, so this searcher is driven by URLs. For a URL
and an ``as_of`` instant it:

1. asks the CDX index for the last few successful captures up to ``as_of``;
2. keeps only captures whose own timestamp is at or before ``as_of``, comparing
   datetimes. The CDX ``to=`` bound is matched at day granularity in places (a
   query ``to=20250101`` has returned a capture stamped ``20250101234441``), so
   the query is never trusted;
3. fetches the newest such capture raw (``<timestamp>id_``: no Wayback toolbar,
   no rewritten links) and keeps its prose as the excerpt.

The source's date is the capture timestamp
(:attr:`DateEvidence.ARCHIVE_CAPTURE`) and its URL is the permanent capture URL.

Known limitations, stated plainly:

* A capture proves the page's content existed at the capture time. A page may
  still embed scripts or widgets that load later content; the raw ``id_``
  capture avoids most of this, and the excerpt is the static HTML text only.
* A capture can be a redirect to another timestamp. It is followed only while
  the final timestamp is still at or before ``as_of``; otherwise it is dropped.
* The URLs come from the question's own text (description, resolution criteria
  and fine print), so a question that names no source page yields nothing.
* Prediction-market and forecasting-platform hosts are excluded
  (:data:`EXCLUDED_HOSTS`): their pages show a price or community forecast,
  which is measured as a separate arm and must not leak in through here.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

import httpx

from augur.clock import Clock, ensure_utc
from augur.errors import SearchError
from augur.log import get_logger
from augur.ratelimit import RateLimiter
from augur.research.source import DateEvidence, Source
from augur.research.text import html_to_text, truncate_words
from augur.research.wikipedia import USER_AGENT, retry_after_seconds

if TYPE_CHECKING:
    from augur.questions import Question

DEFAULT_CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_ROOT = "https://web.archive.org/web/"
EXCLUDED_HOSTS = frozenset(
    {
        "polymarket.com",
        "manifold.markets",
        "metaculus.com",
        "kalshi.com",
        "randforecastinginitiative.org",
        "infer-pub.com",
        "goodjudgment.io",
    }
)
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_REDIRECTS = 3
_URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]{}]+", re.IGNORECASE)
_TRAILING = ".,;:!?*_`'\"”’>"
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_STAMP_IN_URL = re.compile(r"/web/(\d{14})(?:[a-z]{2}_)?/")

log = get_logger("augur.research.wayback")


_MAIN = re.compile(r"<(main|article)\b[^>]*>(.*?)</\1>", re.S | re.I)


def main_content(html: str) -> str:
    """The page's ``<main>`` or ``<article>`` element if it has one, else the page.

    Full pages (a Wikipedia article, a news site) start with navigation that
    would otherwise fill the excerpt before the content does.
    """
    # A MediaWiki page: the article body starts here, after the menus.
    start = html.find('id="mw-content-text"')
    if start >= 0:
        return html[html.rfind("<", 0, start) :]
    match = _MAIN.search(html)
    return match.group(2) if match and match.group(2).strip() else html


def cdx_timestamp(instant: datetime) -> str:
    """``instant`` as the 14-digit ``YYYYMMDDhhmmss`` UTC stamp Wayback uses."""
    return ensure_utc(instant).strftime("%Y%m%d%H%M%S")


def parse_capture_timestamp(stamp: str) -> datetime | None:
    """A Wayback 14-digit stamp as an aware UTC datetime, or ``None`` if malformed."""
    if len(stamp) != 14 or not stamp.isdigit():
        return None
    try:
        return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def is_excluded_host(host: str) -> bool:
    """True for a market or forecasting platform host or any of its subdomains."""
    host = host.lower().rstrip(".")
    return any(host == h or host.endswith("." + h) for h in EXCLUDED_HOSTS)


def extract_urls(*texts: str) -> list[str]:
    """http(s) URLs in ``texts``, trailing punctuation stripped, deduplicated in order."""
    seen: set[str] = set()
    found: list[str] = []
    for text in texts:
        for match in _URL_RE.finditer(text):
            url = match.group(0).rstrip(_TRAILING)
            if urlsplit(url).hostname and url not in seen:
                seen.add(url)
                found.append(url)
    return found


class WaybackSearcher:
    """Archived pages as captured at or before ``as_of``; see the module docstring."""

    name = "wayback"

    def __init__(
        self,
        http: httpx.AsyncClient,
        clock: Clock,
        limiter: RateLimiter | None = None,
        cdx_url: str = DEFAULT_CDX_URL,
        excerpt_chars: int = 3000,
        max_retries: int = 2,
    ) -> None:
        if excerpt_chars < 1:
            raise ValueError("excerpt_chars must be at least 1")
        self._http = http
        self._clock = clock
        self._limiter = limiter or RateLimiter(rate_per_second=1.0, burst=1, max_concurrent=1)
        self._cdx_url = cdx_url
        self._excerpt_chars = excerpt_chars
        self._max_retries = max_retries

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        """The capture of ``query`` when it is itself a URL; otherwise nothing."""
        if limit < 1:
            return []
        text = query.strip()
        if not re.fullmatch(r"https?://\S+", text, re.IGNORECASE):
            return []
        found = await self.capture(text, as_of, query)
        return [found] if found is not None else []

    async def search_question(self, q: Question, as_of: datetime, limit: int) -> list[Source]:
        """Captures of the pages the question's own text links to, markets excluded."""
        if limit < 1:
            return []
        urls = extract_urls(q.description, q.resolution_criteria, q.fine_print)
        sources: list[Source] = []
        for url in urls:
            if len(sources) >= limit:
                break
            host = urlsplit(url).hostname or ""
            if is_excluded_host(host):
                log.info("wayback_market_host_excluded", url=url)
                continue
            try:
                found = await self.capture(url, as_of, q.title)
            except SearchError as exc:
                log.warning("wayback_capture_failed", url=url, error=str(exc))
                continue
            if found is not None:
                sources.append(found)
        return sources

    async def capture(self, url: str, as_of: datetime, query: str) -> Source | None:
        """The newest capture of ``url`` stamped at or before ``as_of``, or ``None``."""
        as_of = ensure_utc(as_of)
        candidates = await self._captures(url, as_of)
        if not candidates:
            log.info("wayback_no_capture", url=url, as_of=as_of.isoformat())
            return None
        for captured_at, stamp, original in candidates:
            html = await self._fetch(stamp, original, as_of)
            if html is None:
                continue
            text = html_to_text(main_content(html))
            if not text:
                log.info("wayback_empty_capture", url=url, capture=stamp)
                continue
            title_match = _TITLE_RE.search(html)
            title = " ".join(title_match.group(1).split()) if title_match else ""
            return Source.build(
                searcher=self.name,
                query=query,
                url=f"{WAYBACK_ROOT}{stamp}/{original}",
                title=title or original,
                excerpt=truncate_words(text, self._excerpt_chars),
                published_at=captured_at,
                retrieved_at=self._clock.now(),
                as_of=as_of,
                date_evidence=DateEvidence.ARCHIVE_CAPTURE,
                evidence_detail=f"capture={stamp} original={original}",
            )
        return None

    async def _captures(self, url: str, as_of: datetime) -> list[tuple[datetime, str, str]]:
        """Usable captures at or before ``as_of``, newest first (never trusts ``to=``)."""
        response = await self._send(
            self._cdx_url,
            url,
            params={
                "url": url,
                "to": cdx_timestamp(as_of),
                "output": "json",
                "filter": "statuscode:200",
                "limit": "-5",
                "fl": "timestamp,original,statuscode,mimetype",
            },
        )
        if response.status_code != 200:
            raise SearchError(
                f"Wayback CDX for {url!r} failed with HTTP {response.status_code}; "
                "retry later or run without the wayback searcher"
            )
        body = response.text.strip()
        if not body:
            return []
        try:
            rows: Any = response.json()
        except ValueError:
            raise SearchError(
                f"Wayback CDX for {url!r} returned a body that is not JSON; retry later"
            ) from None
        if not isinstance(rows, list):
            raise SearchError(f"Wayback CDX for {url!r} returned unexpected JSON; retry later")
        found: list[tuple[datetime, str, str]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 2:
                continue
            captured_at = parse_capture_timestamp(str(row[0]))
            if captured_at is None:
                continue  # the header row, or junk
            if captured_at > as_of:
                log.info("wayback_capture_after_as_of", url=url, capture=str(row[0]))
                continue
            mimetype = str(row[3]).lower() if len(row) > 3 else "text/html"
            if "html" not in mimetype and mimetype != "text/plain":
                continue
            found.append((captured_at, str(row[0]), str(row[1])))
        found.sort(key=lambda c: c[0], reverse=True)
        return found

    async def _fetch(self, stamp: str, original: str, as_of: datetime) -> str | None:
        """The raw capture's body, following redirects only while still at or before ``as_of``."""
        target = f"{WAYBACK_ROOT}{stamp}id_/{original}"
        for _ in range(_MAX_REDIRECTS + 1):
            response = await self._send(target, original, params=None)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location", "")
                match = _STAMP_IN_URL.search(location)
                landed = parse_capture_timestamp(match.group(1)) if match else None
                if landed is None or landed > as_of:
                    log.info(
                        "wayback_redirect_dropped",
                        url=original,
                        capture=stamp,
                        location=location,
                        as_of=as_of.isoformat(),
                    )
                    return None
                target = urljoin(target, location)
                continue
            if response.status_code == 404:
                log.info("wayback_capture_missing", url=original, capture=stamp)
                return None
            if response.status_code != 200:
                raise SearchError(
                    f"Wayback capture {stamp} of {original!r} failed with "
                    f"HTTP {response.status_code}; retry later or run without the wayback searcher"
                )
            return response.text
        log.info("wayback_too_many_redirects", url=original, capture=stamp)
        return None

    async def _send(
        self, target: str, subject: str, *, params: dict[str, str] | None
    ) -> httpx.Response:
        """GET with the limiter, retrying 429/5xx (honouring Retry-After) and transport errors."""
        last_problem = "no attempt made"
        for attempt in range(self._max_retries + 1):
            async with self._limiter.acquire():
                try:
                    response = await self._http.get(
                        target,
                        params=params,
                        headers={"User-Agent": USER_AGENT},
                        follow_redirects=False,
                    )
                except httpx.HTTPError as exc:
                    last_problem = f"{type(exc).__name__}: {exc}"
                    log.warning("wayback_transport_error", attempt=attempt, error=last_problem)
                    continue
            if response.status_code in _RETRY_STATUSES:
                wait = retry_after_seconds(response)
                last_problem = f"HTTP {response.status_code}"
                log.warning(
                    "wayback_retry", status=response.status_code, wait=wait, attempt=attempt
                )
                self._limiter.pause(wait)
                continue
            return response
        raise SearchError(
            f"Wayback request for {subject!r} failed after {self._max_retries + 1} attempts "
            f"({last_problem}); retry later or run without the wayback searcher"
        )
